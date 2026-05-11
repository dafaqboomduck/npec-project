"""PostgreSQL-backed prediction repository.

Implements AbstractPredictionRepository using the existing ORM and
session setup from db/. The confirm method also creates a linked
annotation record in the same transaction.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.src.db.orm import ModelVersion, Prediction
from api.src.repositories.annotations import AnnotationRepository
from api.src.repositories.base import AbstractPredictionRepository
from api.src.schemas.predictions import (
    Plant,
    PredictionMetadata,
    PredictionResponse,
    PredictionStatus,
)

logger = logging.getLogger(__name__)

# Sentinel used when no deployed model version row exists yet.
_UNKNOWN_MODEL_VERSION_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class DatabasePredictionRepository(AbstractPredictionRepository):
    """Persists predictions to the `predictions` table."""

    def __init__(self, db: AsyncSession, user_id: uuid.UUID) -> None:
        """Persists predictions to the `predictions` table.

        Args:
            db: An open SQLAlchemy `AsyncSession` injected by `get_session`.
            user_id: The UUID of the authenticated user making the request.
        """
        self._db = db
        self._user_id = user_id

    # Internal helpers

    async def _get_model_version_id(self) -> uuid.UUID:
        """Return the UUID of the currently deployed model version.

        Falls back to a sentinel UUID if no deployed model exists yet so
        that predictions can still be stored before model management
        endpoints are live.

        Returns:
            UUID of the deployed `ModelVersion` row, or the sentinel UUID.
        """
        result = await self._db.execute(select(ModelVersion.id).where(ModelVersion.status == "deployed"))
        model_id = result.scalar_one_or_none()

        if model_id is None:
            logger.warning(
                "No deployed model version found - using sentinel UUID. "
                "Register and deploy a model version to fix this."
            )
            return _UNKNOWN_MODEL_VERSION_ID

        return model_id

    def _orm_to_schema(self, row: Prediction) -> PredictionResponse:
        """Convert a `Prediction` ORM row to a `PredictionResponse`.

        Args:
            row: The ORM row fetched from the database.

        Returns:
            A fully populated `PredictionResponse`.
        """
        plants = [Plant(**p) for p in (row.plants or [])]

        return PredictionResponse(
            prediction_id=row.id,
            status=PredictionStatus(row.status),
            input_filename=row.input_filename,
            metadata=PredictionMetadata(
                model_version=row.model_version.version if row.model_version else "unknown",
                inference_timestamp=row.inference_timestamp,
                processing_time_ms=row.processing_time_ms or 0,
                input_resolution=[row.input_width_px or 0, row.input_height_px or 0],
                input_channels=row.input_channels or 1,
                normalisation_applied=row.normalisation_applied,
                scale_factor_mm_per_px=row.scale_factor_mm_per_px or 0.0,
            ),
            segmentation_mask_url=row.segmentation_mask_uri,
            plants=plants,
            overall_confidence=row.overall_confidence or 0.0,
            warnings=list(row.warnings or []),
        )

    async def _schema_to_orm(self, prediction: PredictionResponse, filename: str | None = None) -> Prediction:
        """Convert a `PredictionResponse` to a `Prediction` ORM row.

        Args:
            prediction: The schema object returned by the inference service.
            filename: Original uploaded filename, stored for display purposes.

        Returns:
            An unsaved `Prediction` ORM instance ready to be added to the
            session.
        """
        meta = prediction.metadata

        model_id = await self._get_model_version_id()
        return Prediction(
            id=prediction.prediction_id,
            user_id=self._user_id,
            model_version_id=model_id,
            status=prediction.status.value,
            input_filename=filename or prediction.input_filename,
            input_width_px=meta.input_resolution[0] if meta.input_resolution else None,
            input_height_px=meta.input_resolution[1] if meta.input_resolution else None,
            input_channels=meta.input_channels,
            normalisation_applied=meta.normalisation_applied,
            scale_factor_mm_per_px=meta.scale_factor_mm_per_px,
            processing_time_ms=meta.processing_time_ms,
            segmentation_mask_uri=prediction.segmentation_mask_url,
            overall_confidence=prediction.overall_confidence,
            plants=[p.model_dump() for p in prediction.plants],
            warnings=prediction.warnings,
            inference_timestamp=meta.inference_timestamp,
        )

    # AbstractPredictionRepository interface

    async def save(
        self,
        prediction: PredictionResponse,
        filename: str | None = None,
    ) -> PredictionResponse:
        """Persist `prediction` to the `predictions` table.

        Args:
            prediction: The fully populated prediction to store.
            filename: Original uploaded filename for display.

        Returns:
            The stored prediction (identical to input for this implementation).
        """
        row = await self._schema_to_orm(prediction, filename)
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)
        logger.debug("Persisted prediction %s to DB", prediction.prediction_id)
        return prediction

    async def get(self, prediction_id: uuid.UUID) -> PredictionResponse | None:
        """Fetch the prediction with `prediction_id` from the DB.

        Args:
            prediction_id: UUID to look up.

        Returns:
            The matching prediction, or `None` if not found.
        """
        result = await self._db.execute(
            select(Prediction).options(selectinload(Prediction.model_version)).where(Prediction.id == prediction_id)
        )
        row = result.scalar_one_or_none()

        if row is None:
            return None

        return self._orm_to_schema(row)

    async def delete(self, prediction_id: uuid.UUID) -> bool:
        """Delete the prediction with `prediction_id` from the DB.

        Args:
            prediction_id: UUID to remove.

        Returns:
            `True` if a row was deleted, `False` if it did not exist.
        """
        row = await self._db.get(Prediction, prediction_id)
        if row is None:
            return False
        await self._db.delete(row)
        await self._db.commit()
        logger.debug("Deleted prediction %s from DB", prediction_id)
        return True

    async def list_by_user(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> list[PredictionResponse]:
        """Return a paginated list of predictions for the authenticated user.

        Uses `(user_id, creasted_at DESC)` index for efficient pagination.
        SQL OFFSET/LIMIT - no in-memory slicing.

        Args:
            page: 1-indexed page number.
            page_size: Number of record per page (1-100).

        Returns:
            `list[PredictionResponse]`
        """
        offset = (page - 1) * page_size

        # Paginated rows
        result = await self._db.execute(
            select(Prediction)
            .options(selectinload(Prediction.model_version))
            .where(Prediction.user_id == self._user_id)
            .order_by(Prediction.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )

        return [self._orm_to_schema(r) for r in result.scalars().all()]

    async def confirm(
        self,
        prediction_id: uuid.UUID,
        species: str | None,
        age_days: int | None,
        treatment: str | None,
    ) -> tuple[PredictionResponse, bool]:
        """Confirm a prediction and create a linked annotation record.

        Args:
            prediction_id: UUID of the prediction to confirm.
            species: Optional plant species for the annotation record.
            age_days: Optional plant age in days.
            treatment: Optional experimental treatment.

        Returns:
            `(prediction, already_confirmed)` - if `already_confirmed` is
            `True` the router returns 409 without another DB round-trip.

        Raises:
            ValueError: If the prediction does not exist.
        """
        result = await self._db.execute(
            select(Prediction).options(selectinload(Prediction.model_version)).where(Prediction.id == prediction_id)
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise ValueError(f"Prediction {prediction_id} not found.")

        if row.confirmed_at is not None:
            return self._orm_to_schema(row), True

        now = datetime.now(timezone.utc)
        row.confirmed_at = now
        await self._db.flush()  # write confirmed_at before annotation insert

        annotation_repo = AnnotationRepository(self._db)
        await annotation_repo.create_from_prediction(
            prediction_id=prediction_id,
            mask_uri=row.segmentation_mask_uri,
            species=species,
            age_days=age_days,
            treatment=treatment,
        )

        logger.info(f"Confirmed prediction {prediction_id} and created annotation")

        await self._db.commit()
        await self._db.refresh(row)
        return self._orm_to_schema(row), False
