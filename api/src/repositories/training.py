"""Training run repository.

Read/write access to the training_runs table. Used by
LocalTrainingService and the training router.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import TrainingRun
from api.src.schemas.training import TrainingRunDetail, TrainingRunStatus, TrainingRunSummary

logger = logging.getLogger(__name__)


def _row_to_summary(row: TrainingRun) -> TrainingRunSummary:
    """Convert a TrainingRun ORM row to a TrainingRunSummary.

    Args:
        row: ORM row from the database.

    Returns:
        TrainingRunSummary
    """
    return TrainingRunSummary(
        run_id=row.id,
        status=TrainingRunStatus(row.status),
        started_at=row.started_at,
        completed_at=row.completed_at,
        final_f1=row.final_f1,
        hyperparameters=row.hyperparameters,
    )


def _row_to_detail(row: TrainingRun) -> TrainingRunDetail:
    """Convert a TrainingRun ORM row to a TrainingRunDetail.

    Args:
        row: ORM row from the database.

    Returns:
        TrainingRunDetail
    """
    return TrainingRunDetail(
        run_id=row.id,
        status=TrainingRunStatus(row.status),
        started_at=row.started_at,
        completed_at=row.completed_at,
        final_f1=row.final_f1,
        hyperparameters=row.hyperparameters,
        model_version_id=row.model_version_id,
    )


class TrainingRunRepository:
    """Read/write access to the training_runs table."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            db: An open async session injected by get_session.
        """
        self._db = db

    async def create(
        self,
        hyperparameters: dict | None = None,
    ) -> TrainingRun:
        """Insert a new training run row with status='running'.

        Does not commit - the caller is responsible for comitting.

        Args:
            hyperparameters: Optional dict of hyperparameter overrides.

        Returns:
            The newly created TrainingRun ORM row.
        """
        row = TrainingRun(
            id=uuid.uuid4(),
            status=TrainingRunStatus.RUNNING.value,
            started_at=datetime.now(timezone.utc),
            hyperparameters=hyperparameters,
        )
        self._db.add(row)
        await self._db.flush()
        logger.info("Created training run %s", row.id)
        return row

    async def set_job_name(self, run_id: uuid.UUID, job_name: str) -> None:
        """Persist the Azure ML job name after submission.

        Args:
            run_id: UUID of the training run.
            job_name: Azure ML job name to store.
        """
        row = await self._db.get(TrainingRun, run_id)
        if row is not None:
            row.azureml_job_name = job_name
            await self._db.commit()

    async def mark_completed(
        self,
        run_id: uuid.UUID,
        final_f1: float | None = None,
        model_version_id: uuid.UUID | None = None,
    ) -> None:
        """Mark a run as completed.

        Args:
            run_id: UUID of the training run.
            final_f1: Optional final F1 score from the job.
            model_version_id: UUID of the registered model_versions row.
        """
        row = await self._db.get(TrainingRun, run_id)
        if row is None:
            logger.warning("mark_completed: run %s not found", run_id)
            return
        row.status = TrainingRunStatus.COMPLETED.value
        row.completed_at = datetime.now(timezone.utc)
        if final_f1 is not None:
            row.final_f1 = final_f1
        if model_version_id is not None:
            row.model_version_id = model_version_id
        await self._db.commit()
        logger.info("Training run %s marked completed", run_id)

    async def mark_failed(
        self,
        run_id: uuid.UUID,
        reason: str | None = None,
    ) -> None:
        """Mark a run as failed.

        Args:
            run_id: UUID of the training run.
            reason: Optional fialure reason for logging.
        """
        row = await self._db.get(TrainingRun, run_id)
        if row is None:
            logger.warning("mark_failed: run %s not found", run_id)
            return
        row.status = TrainingRunStatus.FAILED.value
        row.completed_at = datetime.now(timezone.utc)
        await self._db.commit()
        logger.warning("Training run %s marked failed: %s", run_id, reason)

    async def mark_stopped(self, run_id: uuid.UUID) -> None:
        """Mark a run as stopped.

        Args:
            run_id: UUID of the training run.
        """
        row = await self._db.get(TrainingRun, run_id)
        if row is None:
            logger.warning("mark_stopped: run %s not found", run_id)
            return
        row.status = TrainingRunStatus.STOPPED.value
        row.completed_at = datetime.now(timezone.utc)
        await self._db.commit()
        logger.info("Training run %s marked stopped", run_id)

    async def get_raw(self, run_id: uuid.UUID) -> TrainingRun | None:
        """Return the raw ORM row for run_id, or None.

        Used by LocalTrainingService to update status fields directly.

        Args:
            run_id: UUID of the training run.

        Returns:
            TrainingRun ORM row or None.
        """
        return await self._db.get(TrainingRun, run_id)

    async def list(self) -> list[TrainingRunSummary]:
        """Return all training runs ordered by started_at descending.

        Returns:
            List of TrainingRunSummary
        """
        result = await self._db.execute(select(TrainingRun).order_by(TrainingRun.started_at.desc()))
        return [_row_to_summary(r) for r in result.scalars().all()]

    async def get(self, run_id: uuid.UUID) -> TrainingRunDetail | None:
        """Return the detail for run_id, or None.

        Args:
            run_id: UUID of the training run.

        Returns:
            TrainingRunDetail if found, None otherwise.
        """
        row = await self._db.get(TrainingRun, run_id)
        if row is None:
            return None
        return _row_to_detail(row)
