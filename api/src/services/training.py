"""Training service interface and local implementation.

AbstractTrainingService defines the interface the router depends on.

LocalTrainingService:
    - Inserts a training_runs row with status='running'
    - Launches the training script as a subprocess via asyncio
    - The subprocess writes per-epoch f1_validation metrics to
        monitoring_metrics during training
    - After the subprocess exits, writes the throughput benchmark
        metric and updates the run row to completed/failed
    - Exposes stop() which sends SIGTERM to the subprocess
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import TrainingRun
from api.src.repositories.training import TrainingRunRepository
from api.src.schemas.training import (
    TrainingRunDetail,
    TrainingRunStatus,
    TrainingRunSummary,
    TrainingStartRequest,
    TrainingStartResponse,
)

logger = logging.getLogger(__name__)

# Path to the training script inside the container.
_TRAINING_SCRIPT = Path(
    os.getenv(
        "NPEC_TRAINING_SCRIPT",
        "/app/scripts/train_local_and_report.py",
    )
)

# Runs older than this with no live pid are considered dead. Default 7h.
_TRAINING_MAX_RUNTIME_S = int(os.getenv("NPEC_TRAINING_MAX_RUNTIME_S", str(7 * 3600)))

# In-memory registry of running subprocesses keyed by run_id.
# Lives in the API process because asyncio subprocess handles
# cannot be serialised. On restart all entries are lost.
_running_processes: dict[uuid.UUID, asyncio.subprocess.Process] = {}


class AbstractTrainingService(ABC):
    """Interface for managing training runs."""

    @abstractmethod
    async def start(self, hyperparameters: TrainingStartRequest | None) -> TrainingStartResponse:
        """Insert a training_runs row with status='running' and launch the job.

        Args:
            hyperparameters: Optional overrides for learning_rate, batch_size,
                and epochs. Uses pipeline defaults when None.

        Returns:
            TrainingStartResponse containing the new run_id.
        """

    @abstractmethod
    async def get_runs(self) -> list[TrainingRunSummary]:
        """Return all training runs ordered by started_at descending.

        Returns:
          List of TrainingRunSummary.
        """

    @abstractmethod
    async def get_run(self, run_id: UUID) -> TrainingRunDetail | None:
        """Return the training run with run_id, or None if not found.

        Args:
            run_id: UUID of the training run to fetch.

        Returns:
            TrainingRunDetail if found, None otherwise.
        """

    @abstractmethod
    async def stop(self, run_id: UUID) -> bool:
        """Terminate the running subprocess and set status to 'stopped'.

        Args:
            run_id: UUID of the training run to stop.

        Returns:
            True if the run was stopped, False if it was not found.

        Raises:
            ValueError: If the run exists but is not in 'running' state.
        """


class LocalTrainingService(AbstractTrainingService):
    """Training service for local and on-premise deployments.

    Runs the training script as an asyncio subprocess. The subprocess
    communicates back by writing rows directly to monitoring_metrics
    using its own DB connection - it does not share the FastAPI session.

    After the subprocess exits:
        - The run is updated to 'completed' or 'failed'.
        - A throughput benchmark is run and written to monitoring_metrics.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise with a DB session for run row management.

        Args:
            db: An open async SQLAlchemy session.
        """
        self._db = db

    async def start(self, hyperparameters: TrainingStartRequest | None) -> TrainingStartResponse:
        """Insert a training_runs row and launch the training subprocess.

        Returns immediately - training runs in the background.

        Args:
            hyperparameters: Optional hyperparameter overrides.

        Returns:
            TrainingStartResponse with the new run_id.

        Raises:
            RuntimeError: If a training run is already in progress.
        """
        # Assemble latest confirmed corrections
        data_dir, n_pairs = await self._materialize_feedback_dataset()
        logger.info("Local training will use %d corrected pair(s).", n_pairs)
        self._data_dir = data_dir

        # Prevent concurrent training runs
        existing = await self._get_running_id()
        if existing is not None:
            raise RuntimeError(
                f"A training run ({existing}) is already in progress. "
                "Stop it via DELETE /training/runs/{id} before starting a new one."
            )

        repo = TrainingRunRepository(self._db)
        params = hyperparameters.model_dump(exclude_none=True) if hyperparameters else {}
        row = await repo.create(hyperparameters=params or None)
        await self._db.commit()

        run_id = row.id

        # Launch the subprocess in the background
        asyncio.create_task(
            self._run_training_subprocess(run_id=run_id, hyperparameters=params),
            name=f"training-{run_id}",
        )

        logger.info("Training run %s started (hyperparameters=%s)", run_id, params)
        return TrainingStartResponse(
            run_id=run_id,
            message=f"Training run {run_id} started. Call GET /training/runs/{run_id} for progress.",
        )

    async def get_runs(self) -> list[TrainingRunSummary]:
        """Return all training runs ordered by started_at descending.

        Returns:
            List of TrainingRunSummary.
        """
        return await TrainingRunRepository(self._db).list()

    async def get_run(self, run_id: UUID) -> TrainingRunDetail | None:
        """Return training run detail, or None if not found.

        Args:
            run_id: UUID of the training run.

        Returns:
            TrainingRunDetail or None.
        """
        return await TrainingRunRepository(self._db).get(run_id)

    async def stop(self, run_id: UUID) -> bool:
        """Terminate the running subprocess and set status to 'stopped'.

        Args:
            run_id: UUID of the training run to stop.

        Returns:
            True if the run was found and stopped, False if not found.

        Raises:
            ValueError: If the run exists but is not in 'running' state.
        """
        repo = TrainingRunRepository(self._db)
        row = await repo.get_raw(run_id)

        if row is None:
            return False

        if row.status != TrainingRunStatus.RUNNING.value:
            raise ValueError(f"Training run {run_id} is in state '{row.status}' - " "only running runs can be stopped.")

        # Terminate the subprocess if we have a handle
        proc = _running_processes.pop(run_id, None)
        if proc is not None:
            try:
                proc.terminate()
                # Give it 5 seconds to exit cleanly, then kill
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                logger.info("Terminated subprocess for training run %s", run_id)
            except ProcessLookupError:
                # Process already exited on its own
                logger.debug("Subprocess for run %s already exited", run_id)
        elif row.pid is not None:
            import os
            import signal

            try:
                os.kill(row.pid, signal.SIGTERM)
                logger.info("Sent SIGTERM to pid %s for training run %s (no in-memory handle)", row.pid, run_id)
            except ProcessLookupError:
                logger.debug(
                    "Pid %s for run %s already exited (no in-memory handle)",
                    row.pid,
                    run_id,
                )
            except PermissionError:
                logger.error(
                    "No permission to signal pid %s for run %s",
                    row.pid,
                    run_id,
                )
        else:
            logger.warning(
                "No subprocess handle found for %s - "
                "updating DB status only (process may have been lost on restart).",
                run_id,
            )

        row.status = TrainingRunStatus.STOPPED.value
        row.completed_at = datetime.now(timezone.utc)
        row.pid = None
        await self._db.commit()

        logger.info("Training run %s stopped", run_id)
        return True

    # Internal helpers

    async def _get_running_id(self) -> uuid.UUID | None:
        """Return the run_id of any currently running training run, or None.

        Returns:
            UUID of the running run, or None.
        """
        result = await self._db.execute(
            select(TrainingRun.id).where(TrainingRun.status == TrainingRunStatus.RUNNING.value).limit(1)
        )
        return result.scalar_one_or_none()

    async def _run_training_subprocess(
        self,
        run_id: uuid.UUID,
        hyperparameters: dict[str, Any],
    ) -> None:
        """Launch the training script and monitor it to completion.

        1. Builds the subprocess command with hyperparameters as env vars.
        2. Registers the process in _running_processes so stop() can find it.
        3. Awaits process exit.
        4. Updates the run row to 'completed' or 'failed'.
        5. On success, runs the throughput bechmark.

        Args:
            run_id: UUID of the training_runs row to update.
            hyperparameters: Hyperparameter overrides passed as env vars.
        """
        from api.src.db.session import get_sessionmaker

        output_dir, version = _next_version_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Training run %s will write artifacts to '%s'", run_id, output_dir)

        deployed_artifact_dir = await _get_deployed_artifact_dir()

        env = _build_subprocess_env(
            run_id=run_id,
            hyperparameters=hyperparameters,
            output_dir=output_dir,
            artifact_dir=deployed_artifact_dir,
            data_dir=self._data_dir,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(_TRAINING_SCRIPT),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _running_processes[run_id] = proc

            async with get_sessionmaker()() as pid_db:
                from api.src.db.orm import TrainingRun

                pid_row = await pid_db.get(TrainingRun, run_id)
                if pid_row is not None:
                    pid_row.pid = proc.pid
                    await pid_db.commit()

            logger.info(
                "Training subprocess started (run_id=%s, pid=%s)",
                run_id,
                proc.pid,
            )

            stdout, stderr = await proc.communicate()

            if stdout:
                logger.info("Training stdout [%s]:\n%s", run_id, stdout.decode())
            if stderr:
                logger.warning("Training stderr [%s]:\n%s", run_id, stderr.decode())

        except Exception as exc:
            logger.error("Failed to launch training subprocess for run %s: %s", run_id, exc)
            await _update_run_status(
                run_id=run_id,
                status=TrainingRunStatus.FAILED,
                session_factory=get_sessionmaker(),
            )
            return
        finally:
            _running_processes.pop(run_id, None)

        session_factory = get_sessionmaker()
        model_version_id: uuid.UUID | None = None

        if proc.returncode == 0:
            # Parse final_f1 from stdout if the script printed it
            final_f1 = _parse_final_f1(stdout.decode())
            if final_f1 is None:
                final_f1 = await _recover_best_f1_from_metrics(run_id, session_factory)

            final_f1_value = final_f1 or 0.0

            model_version_id = await _register_model_if_better(
                run_id=run_id,
                output_dir=output_dir,
                final_f1=final_f1_value,
                session_factory=session_factory,
            )

            await _update_run_status(
                run_id=run_id,
                status=TrainingRunStatus.COMPLETED,
                final_f1=final_f1,
                model_version_id=model_version_id,
                session_factory=session_factory,
            )

            # Backfill model_version_id on epoch metrics if registration succeeded
            effective_model_version_id = model_version_id
            if effective_model_version_id is None:
                # Link metrics to the currently deployed model
                effective_model_version_id = await _get_deployed_model_id_async(session_factory)

            if effective_model_version_id is not None:
                await _backfill_metric_model_version(
                    run_id=run_id,
                    model_version_id=effective_model_version_id,
                    session_factory=session_factory,
                )

            # Throughput benchmark
            await _write_throughput_metric(
                run_id=run_id,
                session_factory=session_factory,
            )
            logger.info(
                "Training run %s completed (final_f1=%s, model_version_id=%s)",
                run_id,
                final_f1,
                model_version_id,
            )
        else:
            logger.error("Training run %s failed (returncode=%s)", run_id, proc.returncode)
            # Recover best F1 from epoch metrics written before the crash
            recovered_f1 = await _recover_best_f1_from_metrics(run_id, session_factory)
            if recovered_f1 is not None:
                logger.info("Recovered best F1=%.4f for failed run %s", recovered_f1, run_id)
            await _update_run_status(
                run_id=run_id,
                status=TrainingRunStatus.FAILED,
                final_f1=recovered_f1,
                model_version_id=model_version_id,
                session_factory=session_factory,
            )

    async def _materialize_feedback_dataset(self) -> tuple[Path]:
        """Write all human-confirmed (image, corrected-mask) pairs into NPEC_DATA_DIR.

        The local trainer pairs by identical stem and treats any non-zero mask
        pixel as foreground, so write images/<stem>.png + masks/<stem>.png.
        The feedback set is rebuilt each run so the trainer always fine-tunes
        on the current confirmed set.
        """
        import asyncio
        import uuid as _uuid

        from sqlalchemy import select

        from api.src.db.orm import Annotation, Prediction
        from api.src.schemas.library import AnnotationStatus
        from api.src.services.storage import get_storage_backend

        data_dir = Path(os.getenv("NPEC_DATA_DIR", "/app/artifacts/training"))
        img_dir, msk_dir = data_dir / "images", data_dir / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for d in (img_dir, msk_dir):
            for p in d.glob("ann_*"):
                p.unlink(missing_ok=True)

        rows = (
            await self._db.execute(
                select(Annotation.id, Annotation.image_uri, Annotation.mask_uri)
                .join(Prediction, Prediction.id == Annotation.prediction_id)
                .where(
                    Annotation.status == AnnotationStatus.ANNOTATED.value,
                    Annotation.image_uri.is_not(None),
                    Annotation.mask_uri.is_not(None),
                    Prediction.confirmed_at.is_not(None),
                )
            )
        ).all()

        storage = get_storage_backend()
        written = 0
        for ann_id, image_uri, mask_uri in rows:
            stem = f"ann_{_uuid.UUID(str(ann_id)).hex}"
            try:
                img = await asyncio.to_thread(storage.download, image_uri)
                mask = await asyncio.to_thread(storage.download, mask_uri)
                await asyncio.to_thread(_write_png, img, img_dir / f"{stem}.png", "RGB")
                await asyncio.to_thread(_write_png, mask, msk_dir / f"{stem}.png", "L")
                written += 1
            except Exception as exc:
                logger.warning("Skipping annotation %s for local training: %s", ann_id, exc)

        logger.info("Materialised %d confirmed pair(s) into %s", written, data_dir)
        return data_dir, written


# Module-level helpers
def _build_subprocess_env(
    run_id: uuid.UUID,
    hyperparameters: dict[str, Any],
    output_dir: Path,
    artifact_dir: Path,
    data_dir: Path | None = None,
) -> dict[str, str]:
    """Build the environment for the training subprcess.

    Passes DATABASE_URL and hyperparameters as environment variables so the
    training script can write metrics back to the DB and apply overrides
    without needing a config file.

    Args:
        run_id: UUID of the training run.
        hyperparameters: Hyperparameter overrides dict.
        output_dir: Path to the output directory for models.
        artifact_dir: Path to the artifact directory.
        data_dir: Path to the data directory.

    Returns:
        Dict of environment variables for the subprocess.
    """
    env = {**os.environ}
    env["TRAINING_RUN_ID"] = str(run_id)
    env["NPEC_OUTPUT_DIR"] = str(output_dir)
    env["NPEC_ARTIFACT_DIR"] = str(artifact_dir)
    if data_dir is not None:
        env["NPEC_DATA_DIR"] = str(data_dir)
    if "learning_rate" in hyperparameters:
        env["NPEC_LEARNING_RATE"] = str(hyperparameters["learning_rate"])
    if "batch_size" in hyperparameters:
        env["NPEC_BATCH_SIZE"] = str(hyperparameters["batch_size"])
    if "epochs" in hyperparameters:
        env["NPEC_EPOCHS"] = str(hyperparameters["epochs"])
    return env


def _next_version_dir() -> tuple[Path, str]:
    """Return (output_dir, version_string) for the next training run.

    Scans NPEC_ARTIFACT_DIR for existing v{n} directories and returns
    the next one. v1 is reserved for the baseline model so training
    starts at v2.

    Returns:
        tuple of (Path to new version dir, version string e.g. 'v3').
    """
    artifact_dir = Path(os.getenv("NPEC_ARTIFACT_DIR", "/app/artifacts/models"))
    existing = (
        [
            int(p.name[1:])
            for p in artifact_dir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]
        if artifact_dir.exists()
        else []
    )

    next_n = max(existing, default=1) + 1
    version = f"v{next_n}"
    return artifact_dir / version, version


async def _get_deployed_model_id_async(session_factory) -> uuid.UUID | None:
    try:
        async with session_factory() as db:
            from api.src.db.orm import ModelVersion

            result = await db.execute(select(ModelVersion.id).where(ModelVersion.status == "deployed").limit(1))
            return result.scalar_one_or_none()
    except Exception as exc:
        logger.warning("Could not fetch deployed model ID: %s", exc)
        return None


async def _get_deployed_artifact_dir() -> Path:
    from api.src.db.orm import ModelVersion
    from api.src.db.session import get_sessionmaker
    from api.src.schemas.library import ModelStatus

    fallback = Path(os.getenv("NPEC_ARTIFACT_DIR", "/app/artifacts/models")) / "v1"
    try:
        async with get_sessionmaker()() as db:
            result = await db.execute(
                select(ModelVersion.artifact_uri).where(ModelVersion.status == ModelStatus.DEPLOYED.value).limit(1)
            )
            uri = result.scalar_one_or_none()
            if uri:
                path = Path(uri)
                if path.is_file():
                    return path.parent
                if path.is_dir():
                    if (path / "root_det_model_512px.keras").exists():
                        return path
                    logger.warning(
                        "artifact_uri '%s' is a directory but contains no model files - using fallback",
                        path,
                    )
    except Exception as exc:
        logger.warning("Could not resolve deployed artifact dir, using fallback: %s", exc)
    return fallback


async def _update_run_status(
    run_id: uuid.UUID,
    status: TrainingRunStatus,
    final_f1: float | None = None,
    model_version_id: uuid.UUID | None = None,
    session_factory=None,
) -> None:
    """Update a training_runs row status using a fresh DB session.

    Uses its own session so this can be called from an asyncio Task that
    otulives the original request session.

    Args:
        run_id: UUID of the run to update.
        status: New status to set.
        final_f1: Optional final F1 score to record on completion.
        model_version_id: Optional UUID of the registered model version.
        session_factory: The async_sessionmaker to create a session from.
    """
    try:
        async with session_factory() as db:
            row = await db.get(TrainingRun, run_id)
            if row is None:
                logger.warning(
                    "_update_run_status: run %s not found in DB",
                    run_id,
                )
                return
            row.pid = None
            row.status = status.value
            row.completed_at = datetime.now(timezone.utc)
            if final_f1 is not None:
                row.final_f1 = final_f1
            if model_version_id is not None:
                row.model_version_id = model_version_id
            await db.commit()
            logger.debug("Updated training run %s to status=%s", run_id, status.value)
    except Exception as exc:
        logger.error("Failed to update training run %s status to %s: %s", run_id, status.value, exc)


def _parse_final_f1(stdout: str) -> float | None:
    """Parse the final F1 score from training script stdout.

    The training script is expected to print a line in the fromat:
        FINAL_F1=0.923

    Args:
        stdout: Decoded stdout from the training subprocess.

    Returns:
        Parsed float or None if the line is missing or malformed.
    """
    for line in stdout.splitlines():
        if line.startswith("FINAL_F1="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                logger.warning("Could not parse FINAL_F1 from line: %r", line)
    return None


async def _recover_best_f1_from_metrics(
    run_id: uuid.UUID,
    session_factory,
) -> float | None:
    """Read the best f1_validation metric for this run from monitoring_metrics."""
    try:
        from sqlalchemy import func

        async with session_factory() as db:
            from api.src.db.orm import MonitoringMetric

            result = await db.execute(
                select(func.max(MonitoringMetric.metric_value))
                .where(MonitoringMetric.metric_name == "f1_validation")
                .where(MonitoringMetric.labels["run_id"].astext == str(run_id))
            )
            value = result.scalar_one_or_none()
            return float(value) if value is not None else None
    except Exception as exc:
        logger.warning("Could not recover F1 from metrics for run %s: %s", run_id, exc)
        return None


async def _write_throughput_metric(
    run_id: uuid.UUID,
    session_factory=None,
) -> None:
    """Run a quick throughput benchmark and write the result to monitoring_metrics.

    Passes a small batch of validation images through the inference service,
    measures wall-clock time, and computes images per second.

    Args:
        run_id: UUID of the training run that just completed (for logging).
        session_factory: The async_sessionmaker to use.
    """
    import time
    from pathlib import Path

    try:
        # Find validation images
        val_dir = Path(os.getenv("NPEC_VALIDATION_DIR", "/app/artifacts/validation"))
        if not val_dir.exists():
            logger.warning(
                "Throughput benchmark skipped - validation dir not found at %s",
                val_dir,
            )
            return

        image_paths = list(val_dir.glob("*.png"))[:10]
        if not image_paths:
            logger.warning("Throughput benchmark skipped - no .png images in %s", val_dir)
            return

        from api.src.dependencies import get_inference_service

        service = get_inference_service()

        start = time.perf_counter()
        for path in image_paths:
            image_bytes = path.read_bytes()
            await service.run(image_bytes=image_bytes, filename=path.name)
        elapsed = time.perf_counter() - start

        throughput = round(len(image_paths) / elapsed, 2) if elapsed > 0 else 0.0

        async with session_factory() as db:
            from api.src.db.orm import ModelVersion, MonitoringMetric
            from api.src.schemas.library import ModelStatus

            result = await db.execute(
                select(ModelVersion.id).where(ModelVersion.status == ModelStatus.DEPLOYED.value).limit(1)
            )
            model_version_id = result.scalar_one_or_none()

            db.add(
                MonitoringMetric(
                    metric_name="throughput_img_per_s",
                    metric_value=throughput,
                    model_version_id=model_version_id,
                    labels={"run_id": str(run_id)},
                    recorded_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()

        logger.info(
            "Throughput benchmark for run %s: %.2f img/s (%d images in %.2fs)",
            run_id,
            throughput,
            len(image_paths),
            elapsed,
        )

    except Exception as exc:
        logger.error("Throughput benchmark failed for run %s: %s", run_id, exc)


async def _register_model_if_better(
    run_id: uuid.UUID,
    output_dir: Path,
    final_f1: float,
    session_factory,
) -> uuid.UUID | None:
    """Conditionally register the trained model if it beats the deployed one.

    Reads MODEL_PROMOTION_MIN_IMPROVEMENT from env (default 0.01).
    Uses a versioned name based on timestamp so versions never collide.

    Args:
        run_id: UUID of the completed training run (for the version string).
        output_dir: Directory where the training script saved its output.
        final_f1: F1 score achieved during training.
        session_factory: The async_sessionmaker to use.

    Returns:
        UUID of the registered model version, or None if not registered.
    """
    min_improvement = float(os.getenv("MODEL_PROMOTION_MIN_IMPROVEMENT", "0.01"))
    artifact_uri = str(output_dir)

    if not Path(artifact_uri).exists():
        logger.warning(
            "Training output not found at '%s' - skipping registration.",
            artifact_uri,
        )
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    version = f"unet-npec-local-{timestamp}"
    name = f"Local fine-tune {timestamp}"

    try:
        async with session_factory() as db:
            from api.src.repositories.models import ModelRepository

            model_id, was_registered = await ModelRepository(db).register_and_promote_if_better(
                version=version,
                name=name,
                artifact_uri=artifact_uri,
                final_f1=final_f1,
                min_improvement=min_improvement,
            )
            return model_id if was_registered else None
    except Exception as exc:
        logger.error("Model registration failed for run %s: %s", run_id, exc)
        return None


async def _backfill_metric_model_version(
    run_id: uuid.UUID,
    model_version_id: uuid.UUID,
    session_factory,
) -> None:
    """Link all f1_validation rows for this run to the registered model.

    Called after successful model registration so the dashboard can
    filter metrics by model version.

    Args:
        run_id: UUID of the completed training run.
        model_version_id: UUID of the newly registered model version.
        session_factory: The async_sessionmaker to use.
    """
    try:
        async with session_factory() as db:
            from sqlalchemy import update

            from api.src.db.orm import MonitoringMetric

            await db.execute(
                update(MonitoringMetric)
                .where(
                    MonitoringMetric.metric_name == "f1_validation",
                    MonitoringMetric.labels["run_id"].astext == str(run_id),
                )
                .values(model_version_id=model_version_id)
            )
            await db.commit()
            logger.info(
                "Backfilled model_version_id=%s on f1_validation metrics for run %s",
                model_version_id,
                run_id,
            )
    except Exception as exc:
        logger.error(
            "Failed to backfill metric model_version_id for run %s: %s",
            run_id,
            exc,
        )


def _pid_alive(pid: int | None) -> bool:
    """Return True if `pid` is a live process in this PID namespace."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Exists but owned by another user


async def reconcile_orphaned_training_runs() -> None:
    """Mark stale 'running' rows as failed at startup.

    Subprocess handles do not survive an API restart, so any 'running' row
    whose pid is dead (or that exceeded the max runtime) is orphaned. Marking
    these failed unblocks `start()` and the retraining threshold check.
    """
    from datetime import timedelta

    from api.src.db.session import get_sessionmaker

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_TRAINING_MAX_RUNTIME_S)
    try:
        async with get_sessionmaker()() as db:
            rows = (
                (await db.execute(select(TrainingRun).where(TrainingRun.status == TrainingRunStatus.RUNNING.value)))
                .scalars()
                .all()
            )
            reconciled = 0
            for row in rows:
                if not _pid_alive(row.pid) or (row.started_at and row.started_at < cutoff):
                    row.status = TrainingRunStatus.FAILED.value
                    row.completed_at = datetime.now(timezone.utc)
                    row.pid = None
                    reconciled += 1
            if reconciled:
                await db.commit()
                logger.warning("Reconciled %d orphaned training run(s) to 'failed'.", reconciled)
    except Exception as exc:
        logger.warning("Training-run reconciliation failed: %s", exc)


def _write_png(raw: bytes, dest: Path, mode: str) -> None:
    """Decode blob bytes and write a single-frame PNG in the given mode ('RGB'|'L')."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
        im.seek(0)
        im.convert(mode).save(dest, format="PNG")
