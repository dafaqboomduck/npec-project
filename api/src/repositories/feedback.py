"""Feedback repository.

Writes to the feedback table. Multiple rows per
(user_id, prediction_id) are allowed by design - the most
recent row represents the user's current opinion.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import Feedback, Prediction

logger = logging.getLogger(__name__)


class FeedbackRepository:
    """Write access to the `feedback` table."""

    def __init__(self, db: AsyncSession) -> None:
        """Write access to the `feedback` table.

        Args:
            db: Injected database session.
        """
        self._db = db

    async def prediction_exists(self, prediction_id: uuid.UUID) -> bool:
        """Return `True` if a prediction with `prediction_id` exists.

        Args:
            prediction_id: UUID to check.

        Returns:
            `True` if the prediction row exists, `False` otherwise.
        """
        result = await self._db.execute(select(Prediction.id).where(Prediction.id == prediction_id))
        return result.scalar_one_or_none() is not None

    async def create(
        self,
        user_id: uuid.UUID,
        prediction_id: uuid.UUID,
        signal: str,
        comment: str | None,
    ) -> None:
        """Insert a new feedback row.

        Args:
            user_id: UUID of the user submitting feedback.
            prediction_id: UUID of the prediction being rated.
            signal: `correct or `incorrect`.
            comment: Optional free-text note, stored as NULL when absent.
        """
        row = Feedback(
            user_id=user_id,
            prediction_id=prediction_id,
            signal=signal,
            comment=comment,
        )
        self._db.add(row)
        await self._db.commit()
        logger.debug(
            "Recorded feedback signal=%s for prediction %s by user %s",
            signal,
            prediction_id,
            user_id,
        )

    async def get_latest_for_prediction(
        self,
        prediction_id: uuid.UUID,
    ) -> Feedback | None:
        """Return the most recent feedback row for a prediction.

        Args:
            prediction_id: UUID of the prediction.

        Returns:
            The most recent `Feedback` ORM row, or None if no feedback exists.
        """
        result = await self._db.execute(
            select(Feedback)
            .where(Feedback.prediction_id == prediction_id)
            .order_by(Feedback.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
