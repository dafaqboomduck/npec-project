"""Azure ML training service.

Submits scripts/submit_pipeline.py --run-now as a subprocess, polls the
resulting Azure ML pipeline job for completion, and marks the local
training_runs row accordingly.

Required environment variables
-------------------------------
AZUREML_SUBSCRIPTION_ID    Azure subscription ID
AZUREML_RESOURCE_GROUP     Resource group name
AZUREML_WORKSPACE          Azure ML workspace name
AZUREML_POLL_INTERVAL      Seconds between job status polls (default: 30)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from api.src.repositories.training import TrainingRunRepository
from api.src.schemas.training import (
    TrainingRunDetail,
    TrainingRunStatus,
    TrainingRunSummary,
    TrainingStartRequest,
    TrainingStartResponse,
)
from api.src.services.training import AbstractTrainingService

logger = logging.getLogger(__name__)

_POLL_INTERVAL = int(os.getenv("AZUREML_POLL_INTERVAL", "30"))
_SUBSCRIPTION_ID = os.getenv("AZUREML_SUBSCRIPTION_ID", "")
_RESOURCE_GROUP = os.getenv("AZUREML_RESOURCE_GROUP", "")
_WORKSPACE = os.getenv("AZUREML_WORKSPACE", "")


def _get_ml_client():
    """Return an authenticated MLClient.

    Inside Azure ML compute, uses DefaultAzureCredential (managed identity).
    Locally, uses AzureCliCredential (az login).

    Returns:
        An authenticated MLClient bound to the configured workspace.
    """
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential, DefaultAzureCredential

    if all(
        k in os.environ
        for k in (
            "AZUREML_SUBSCRIPTION_ID",
            "AZUREML_RESOURCE_GROUP",
            "AZUREML_WORKSPACE",
        )
    ):
        credential = DefaultAzureCredential()
        subscription_id = os.environ["AZUREML_SUBSCRIPTION_ID"]
        resource_group = os.environ["AZUREML_RESOURCE_GROUP"]
        workspace = os.environ["AZUREML_WORKSPACE"]
    else:
        credential = AzureCliCredential(process_timeout=30)
        subscription_id = _SUBSCRIPTION_ID
        resource_group = _RESOURCE_GROUP
        workspace = _WORKSPACE

    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


class AzureMLTrainingService(AbstractTrainingService):
    """Submits training jobs via submit_pipeline.py and tracks them in the local DB."""

    def __init__(self, db: AsyncSession):
        """Initialise with a DB session for run row management.

        Args:
            db: An open async SQLAlchemy session.
        """
        self._db = db
        self._repo = TrainingRunRepository(db)

    async def start(self, hyperparameters: TrainingStartRequest | None) -> TrainingStartResponse:
        """Run submit_pipeline.py --run-now and return immediately.

        Creates the DB row first, submits the pipeline in a thread, persists
        the job name, then launches a background polling task.

        Args:
            hyperparameters: Ignored - the pipeline script controls its own hyperparameters.

        Returns:
            TrainingStartResponse with run_id and confirmation message.
        """
        row = await self._repo.create(hyperparameters=None)
        run_id = row.id
        await self._db.commit()

        try:
            job_name = await asyncio.to_thread(self._submit_job, run_id)
        except Exception as exc:
            await self._repo.mark_failed(run_id, reason=f"Job submission failed: {exc}")
            raise RuntimeError(f"Failed to submit pipeline: {exc}") from exc

        await self._repo.set_job_name(run_id, job_name)

        logger.info("AzureMLTrainingService: submitted pipeline job '%s' for run %s", job_name, run_id)

        asyncio.create_task(
            self._poll_until_done(run_id, job_name),
            name=f"azureml-poll-{run_id}",
        )

        return TrainingStartResponse(
            run_id=run_id,
            message=f"Pipeline job '{job_name}' submitted. Poll GET /training/runs/{run_id} for status.",
        )

    def _submit_job(self, run_id: uuid.UUID) -> str:
        """Run submit_pipeline.py --run-now and parse the job name from stdout.

        Args:
            run_id: UUID of the local training_runs row (for logging only).

        Returns:
            Azure ML job name, e.g. 'root-detection-weekly-retrain_abc123'.

        Raises:
            RuntimeError: If the script fails or the job name cannot be parsed.
        """
        result = subprocess.run(
            ["python", "/app/scripts/submit_pipeline.py", "--schedule", "--run-now"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith("Pipeline submitted"):
                return line.split(":", 1)[-1].strip()
        raise RuntimeError(f"Could not parse job name from submit_pipeline.py output:\n{result.stdout}")

    async def _poll_until_done(self, run_id: uuid.UUID, job_name: str) -> None:
        """Poll Azure ML until the pipeline job reaches a terminal state.

        Runs as a background asyncio task. Marks the training_runs row as
        'completed' or 'failed'. Model registration and deployment are handled
        entirely by submit_pipeline.py itself.

        Args:
            run_id: UUID of the local training_runs row.
            job_name: Azure ML pipeline job name returned by _submit_job().
        """
        try:
            status = await asyncio.to_thread(self._wait_for_job, job_name)

            if status == "Completed":
                await self._repo.mark_completed(run_id, model_version_id=None)
                logger.info("AzureMLTrainingService: run %s completed successfully", run_id)
            else:
                await self._repo.mark_failed(
                    run_id,
                    reason=f"Pipeline job ended with status: {status}",
                )
        except Exception as exc:
            logger.exception("AzureMLTrainingService: polling failed for run %s", run_id)
            await self._repo.mark_failed(run_id, reason=str(exc))

    def _wait_for_job(self, job_name: str) -> str:
        """Block until the Azure ML job reaches a terminal state.

        Args:
            job_name: Azure ML job name to poll.

        Returns:
            Final job status string e.g. 'Completed', 'Failed', 'Canceled'.
        """
        import time

        ml_client = _get_ml_client()
        terminal = {"Completed", "Failed", "Canceled", "NotResponding"}

        while True:
            job = ml_client.jobs.get(job_name)
            if job.status in terminal:
                return job.status
            logger.debug("AzureMLTrainingService: job '%s' status: %s", job_name, job.status)
            time.sleep(_POLL_INTERVAL)

    async def get_runs(self) -> list[TrainingRunSummary]:
        """Return all training runs ordered by started_at descending.

        Returns:
            List of TrainingRunSummary.
        """
        return await self._repo.list()

    async def get_run(self, run_id: uuid.UUID) -> TrainingRunDetail | None:
        """Return training run detail, or None if not found.

        Args:
            run_id: UUID of the training run.

        Returns:
            TrainingRunDetail or None.
        """
        return await self._repo.get(run_id)

    async def stop(self, run_id: uuid.UUID) -> bool:
        """Cancel the Azure ML job and mark the run as stopped.

        Args:
            run_id: UUID of the training run.

        Returns:
            True if stopped, False if not found.

        Raises:
            ValueError: If the run is not in running state.
        """
        row = await self._repo.get_raw(run_id)
        if row is None:
            return False

        if row.status != TrainingRunStatus.RUNNING.value:
            raise ValueError(
                f"Training run {run_id} is in state '{row.status}' - only running runs can be stopped",
            )

        if row.azureml_job_name:
            try:
                ml_client = _get_ml_client()
                await asyncio.to_thread(ml_client.jobs.cancel, row.azureml_job_name)
                logger.info("AzureMLTrainingService: cancelled job '%s'", row.azureml_job_name)
            except Exception as exc:
                logger.warning(
                    "AzureMLTrainingService: failed to cancel Azure ML job '%s': %s",
                    row.azureml_job_name,
                    exc,
                )

        await self._repo.mark_stopped(run_id)
        return True
