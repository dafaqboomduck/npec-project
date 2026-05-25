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

from api.src.db.orm import Annotation, ModelVersion, Prediction
from api.src.repositories.annotations import AnnotationRepository
from api.src.repositories.base import AbstractPredictionRepository
from api.src.repositories.datasets import DatasetRepository
from api.src.repositories.feedback import FeedbackRepository
from api.src.schemas.auth import UserRole
from api.src.schemas.predictions import (
    Plant,
    PredictionMetadata,
    PredictionResponse,
    PredictionStatus,
)
from api.src.services.storage import AbstractStorageBackend

logger = logging.getLogger(__name__)

# Sentinel used when no deployed model version row exists yet.
_UNKNOWN_MODEL_VERSION_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class DatabasePredictionRepository(AbstractPredictionRepository):
    """Persists predictions to the `predictions` table."""

    def __init__(self, db: AsyncSession, user_id: uuid.UUID, storage: AbstractStorageBackend) -> None:
        """Persists predictions to the `predictions` table.

        Args:
            db: An open SQLAlchemy `AsyncSession` injected by `get_session`.
            user_id: The UUID of the authenticated user making the request.
            storage: Injected storage backend.
        """
        self._db = db
        self._user_id = user_id
        self._storage = storage

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

    def _orm_to_schema(
        self,
        row: Prediction,
        feedback_signal: str | None = None,
    ) -> PredictionResponse:
        """Convert a `Prediction` ORM row to a `PredictionResponse`.

        Args:
            row: The ORM row fetched from the database.
            feedback_signal: `FeedbackSignal` value.

        Returns:
            A fully populated `PredictionResponse`.
        """
        plants = [Plant(**p) for p in (row.plants or [])]

        return PredictionResponse(
            prediction_id=row.id,
            status=PredictionStatus(row.status),
            input_filename=row.input_filename,
            owner_email=row.user.email if row.user else None,
            metadata=PredictionMetadata(
                model_version=row.model_version.version if row.model_version else "unknown",
                inference_timestamp=row.inference_timestamp,
                processing_time_ms=row.processing_time_ms or 0,
                input_resolution=[row.input_width_px or 0, row.input_height_px or 0],
                input_channels=row.input_channels or 1,
                normalisation_applied=row.normalisation_applied,
                scale_factor_mm_per_px=row.scale_factor_mm_per_px or 0.0,
            ),
            segmentation_mask_url=self._storage.sign(row.segmentation_mask_uri) if row.segmentation_mask_uri else None,
            input_url=self._storage.sign(row.input_uri) if row.input_uri else None,
            tip_photo_url=self._storage.sign(row.tip_photo_uri) if row.tip_photo_uri else None,
            plants=plants,
            feedback_signal=feedback_signal,
            warnings=list(row.warnings or []),
        )

    async def _schema_to_orm(
        self,
        prediction: PredictionResponse,
        filename: str | None = None,
        input_hash: str | None = None,
        input_uri: str | None = None,
        tip_photo_uri: str | None = None,
    ) -> Prediction:
        """Convert a `PredictionResponse` to a `Prediction` ORM row.

        Args:
            prediction: The schema object returned by the inference service.
            filename: Original uploaded filename, stored for display purposes.
            input_hash: Hash string of the file.
            input_uri: Local file URI of the saved input image.
            tip_photo_uri: Local file URI of the saved tip visualization.

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
            input_hash=input_hash,
            input_uri=input_uri,
            input_width_px=meta.input_resolution[0] if meta.input_resolution else None,
            input_height_px=meta.input_resolution[1] if meta.input_resolution else None,
            input_channels=meta.input_channels,
            normalisation_applied=meta.normalisation_applied,
            scale_factor_mm_per_px=meta.scale_factor_mm_per_px,
            processing_time_ms=meta.processing_time_ms,
            segmentation_mask_uri=prediction.segmentation_mask_url,
            tip_photo_uri=tip_photo_uri,
            plants=[p.model_dump() for p in prediction.plants],
            warnings=prediction.warnings,
            inference_timestamp=meta.inference_timestamp,
        )

    # AbstractPredictionRepository interface

    async def save(
        self,
        prediction: PredictionResponse,
        filename: str | None = None,
        input_hash: str | None = None,
        input_uri: str | None = None,
        tip_photo_uri: str | None = None,
    ) -> PredictionResponse:
        """Persist `prediction` to the `predictions` table.

        Args:
            prediction: The fully populated prediction to store.
            filename: Original uploaded filename for display.
            input_hash: Hash string of the file.
            input_uri: Local file URI of the saved input image.
            tip_photo_uri: Local file URI of the saved tip visualization.

        Returns:
            The stored prediction (identical to input for this implementation).
        """
        row = await self._schema_to_orm(
            prediction,
            filename,
            input_hash,
            input_uri,
            tip_photo_uri,
        )
        self._db.add(row)
        await self._db.commit()

        result = await self._db.execute(
            select(Prediction)
            .options(
                selectinload(Prediction.model_version),
                selectinload(Prediction.user),
            )
            .where(Prediction.id == row.id)
        )
        saved_row = result.scalar_one()
        logger.debug("Persisted prediction %s to DB", prediction.prediction_id)
        return self._orm_to_schema(saved_row)

    async def get(
        self,
        prediction_id: uuid.UUID,
        role: UserRole = UserRole.USER,
    ) -> PredictionResponse | None:
        """Fetch the prediction with `prediction_id` from the DB.

        Args:
            prediction_id: UUID to look up.
            role: `UserRole` of the logged in user.

        Returns:
            The matching prediction, or `None` if not found.
        """
        q = (
            select(Prediction)
            .options(
                selectinload(Prediction.model_version),
                selectinload(Prediction.user),
            )
            .where(Prediction.id == prediction_id)
            .where(Prediction.user_id == self._user_id)
        )
        if role != UserRole.ADMIN:
            q = q.where(Prediction.user_id == self._user_id)

        result = await self._db.execute(q)
        row = result.scalar_one_or_none()

        if row is None:
            return None

        feedback_signal: str | None = None
        fb = await FeedbackRepository(self._db).get_latest_for_prediction(prediction_id)
        if fb is not None:
            feedback_signal = fb.signal

        return self._orm_to_schema(row, feedback_signal=feedback_signal)

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
        confirmed: bool | None = None,
        role: UserRole = UserRole.USER,
    ) -> list[PredictionResponse]:
        """Return a paginated list of predictions for the authenticated user.

        Uses `(user_id, creasted_at DESC)` index for efficient pagination.
        SQL OFFSET/LIMIT - no in-memory slicing.

        Args:
            page: 1-indexed page number.
            page_size: Number of record per page (1-100).
            confirmed: Boolean determining the confirmation status.
            role: UserRole = UserRole.USER,

        Returns:
            `list[PredictionResponse]`
        """
        offset = (page - 1) * page_size

        q = (
            select(Prediction)
            .options(
                selectinload(Prediction.model_version),
                selectinload(Prediction.user),
            )
            .where(Prediction.user_id == self._user_id)
        )
        if role != UserRole.ADMIN:
            q = q.where(Prediction.user_id == self._user_id)

        if confirmed is True:
            q = q.where(Prediction.confirmed_at.isnot(None))
        elif confirmed is False:
            q = q.where(Prediction.confirmed_at.is_(None))

        # Paginated rows
        q = q.order_by(Prediction.created_at.desc()).offset(offset).limit(page_size)

        result = await self._db.execute(q)
        return [self._orm_to_schema(r) for r in result.scalars().all()]

    async def confirm(
        self,
        prediction_id: uuid.UUID,
        species: str | None,
        age_days: int | None,
        treatment: str | None,
        dataset_id: uuid.UUID | None = None,
    ) -> tuple[PredictionResponse, bool, uuid.UUID | None]:
        """Confirm a prediction and create a linked annotation record.

        Args:
            prediction_id: UUID of the prediction to confirm.
            species: Optional plant species for the annotation record.
            age_days: Optional plant age in days.
            treatment: Optional experimental treatment.
            dataset_id: Optional UUID of the dataset to link the annotation to.
                When provided, datasets.count is incremented in the transaction.

        Returns:
            `(prediction, annotation_id, already_confirmed)` - if `already_confirmed` is
            `True` the router returns 409 without another DB round-trip.

        Raises:
            ValueError: If the prediction does not exist.
            ValueError: If dataset_id is provided but the dataset does not exist.
        """
        result = await self._db.execute(
            select(Prediction)
            .options(
                selectinload(Prediction.model_version),
                selectinload(Prediction.user),
            )
            .where(Prediction.id == prediction_id)
            .where(Prediction.user_id == self._user_id)
        )
        row = result.scalar_one_or_none()

        if row is None:
            raise ValueError(f"Prediction {prediction_id} not found.")

        if row.confirmed_at is not None:
            existing = await self._db.execute(select(Annotation).where(Annotation.prediction_id == prediction_id))
            existing_annotation = existing.scalar_one_or_none()
            existing_id = existing_annotation.id if existing_annotation else None
            return self._orm_to_schema(row), True, existing_id

        now = datetime.now(timezone.utc)
        row.confirmed_at = now
        await self._db.flush()  # write confirmed_at before annotation insert

        if dataset_id is not None:
            dataset_repo = DatasetRepository(self._db)
            dataset = await dataset_repo.get(dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_id} not found.")

        annotation_repo = AnnotationRepository(self._db, self._storage)
        annotation_id = await annotation_repo.create_from_prediction(
            prediction_id=prediction_id,
            mask_uri=row.segmentation_mask_uri,
            tip_photo_uri=row.tip_photo_uri,
            input_hash=row.input_hash,
            image_uri=row.input_uri,
            species=species,
            age_days=age_days,
            treatment=treatment,
            dataset_id=dataset_id,
            created_by=self._user_id,
        )

        if dataset_id is not None:
            await dataset_repo.increment_count(dataset_id)

        logger.info(f"Confirmed prediction {prediction_id} and created annotation")

        await self._db.commit()
        await self._db.refresh(row)
        return self._orm_to_schema(row), False, annotation_id
