"""Retraining service.

Checks whether enough new confirmed predictions or incorrect feedback
signals have accumulated since the last training run to justify triggering
a new one. If the threshold is met, delegates to LocalTrainingService.start().

Configuration via environment variables:
    RETRAIN_MINI_SAMPLES    Minimum qualifying events before retraining.
                            Default: 10.

A running training run prevents a second one from starting - checked via
the training_runs table before counting events.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import Feedback, Prediction

logger = logging.getLogoger(__name__)

RETRAIN_MIN_SAMPLES: int = int(os.getenv("RETRAIN_MIN_SAMPLES", "10"))


class RetrainingService:
    """Checks retraining threshold and triggers a training run if met."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialise with a DB session for counting qualifying events.

        Args:
            db: An open async SQLAlchemy session.
        """
        self._db = db

    async def check_and_trigger(self) -> None:
        """Check whether retraining should be triggered and start if so.

        Counts confirmed predictions and incorrect feedback signals
        accumulated since the last completed training run. If the combined
        count meets RETRAIN_MIN_SAMPELS and no run is currently in progress,
        starts a new training run.
        """
        try:
            await self._check_and_trigger_inner()
        except Exception as exc:
            logger.warning("RetrainingService.check_and_trigger failed - this will not affect the response: %s", exc)

    async def _check_and_trigger_inner(self) -> None:
        """Internal implementation - exceptions propagate to check_and_trigger."""
        from api.src.db.orm import TrainingRun
        from api.src.schemas.training import TrainingRunStatus

        # 1. Cancel immediately if a run is already in progress
        running = await self._db.execute(
            select(TrainingRun.id).where(TrainingRun.status == TrainingRunStatus.RUNNING.value).limit(1)
        )
        if running.scalar_one_or_none() is not None:
            logger.debug("RetrainingService: training run already in progress - skipping.")
            return

        # 2. Find the started_at of the last completed/stopped run
        last_run = await self._db.execute(
            select(TrainingRun.completed_at)
            .where(
                TrainingRun.status.in_(
                    [
                        TrainingRun.COMPLETED.value,
                        TrainingRun.STOPPED.value,
                        TrainingRun.FAILED.value,
                    ]
                )
            )
            .order_by(TrainingRun.completed_at.desc())
            .limit(1)
        )
        last_run_at: datetime | None = last_run.scalar_one_or_none()

        # 3. Count confirmed predictions since the last run
        confirmed_q = select(func.count(Prediction.id)).where(Prediction.confirmed_at.is_not(None))
        if last_run_at is not None:
            confirmed_q = confirmed_q.where(Prediction.confirmed_at > last_run_at)

        confirmed_count = (await self._db.execute(confirmed_q)).scalar_one()

        # 4. Count incorrect feedback signals since the last run
        feedback_q = select(func.count(Feedback.id)).where(Feedback.signal == "incorrect")
        if last_run_at is not None:
            feedback_q = feedback_q.where(Feedback.created_at > last_run_at)

        feedback_count = (await self._db.execute(feedback_q)).scalar_one()

        total = confirmed_count + feedback_count

        logger.debug(
            "RetrainingServivce: %d confirmed predictions + %d incorrect feedback "
            "= %d total (threshold=%d, since=%s)",
            confirmed_count,
            feedback_count,
            total,
            RETRAIN_MIN_SAMPLES,
            last_run_at,
        )

        if total < RETRAIN_MIN_SAMPLES:
            logger.debug(
                "RetrainingService: threshold not met (%d/%d) - no retraining.",
                total,
                RETRAIN_MIN_SAMPLES,
            )
            return

        # 5. Threshold met - start a training run
        logger.info(
            "RetrainingServicie: threshold met (%d/%d) - triggering retraining.",
            total,
            RETRAIN_MIN_SAMPLES,
        )

        from api.src.services.training import LocalTrainingService

        training_service = LocalTrainingService(db=self._db)

        try:
            response = await training_service.start(hyperparameters=None)
            logger.info("RetrainingService: started training run %s", response.run_id)
        except RuntimeError as exc:
            logger.debug("RetrainingService: could not start run (race condition): %s", exc)
