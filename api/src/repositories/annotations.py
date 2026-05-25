"""Annotation repository.

Reads from annotations table via the ORM. Pagination is handled
entirely in SQL via OFFSET/LIMIT - no post-fetch slicing.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.src.db.orm import Annotation
from api.src.db.orm import Prediction as PredictionORM
from api.src.schemas.auth import UserRole
from api.src.schemas.library import AnnotationResponse, AnnotationStatus
from api.src.services.storage import AbstractStorageBackend

logger = logging.getLogger(__name__)


def _row_to_schema(
    row: Annotation,
    storage: AbstractStorageBackend,
    owner_email: str | None = None,
) -> AnnotationResponse:
    """Convert a `Annotation` ORM row to a `AnnotationResponse`.

    Args:
        row: The ORM row from the database.
        storage: The injected backend storage.
        owner_email: Annotation owner's email.

    Returns:
        AnnotationResponse
    """
    return AnnotationResponse(
        annotation_id=row.id,
        prediction_id=row.prediction_id,
        dataset_id=row.dataset_id,
        status=AnnotationStatus(row.status),
        image_uri=storage.sign(row.image_uri) if row.image_uri else None,
        mask_url=storage.sign(row.mask_uri) if row.mask_uri else None,
        tip_photo_url=storage.sign(row.tip_photo_uri) if row.tip_photo_uri else None,
        species=row.species,
        age_days=row.age_days,
        treatment=row.treatment,
        growth_stage=row.growth_stage,
        owner_email=owner_email,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class AnnotationRepository:
    """Paginated, filtered read access to the `annotations` table."""

    def __init__(self, db: AsyncSession, storage: AbstractStorageBackend) -> None:
        """Paginated, filtered read access to the `annotations` table.

        Args:
            db: An open async SQLAlchemy session injected by ``get_session``.
            storage: Injected storage backend.
        """
        self._db = db
        self._storage = storage

    async def list(
        self,
        species: str | None = None,
        growth_stage: str | None = None,
        status: str | None = None,
        dataset_id: uuid.UUID | None = None,
        page: int = 1,
        page_size: int = 20,
        role: UserRole = UserRole.USER,
        requesting_user_id: uuid.UUID | None = None,
        order: str = "desc",
    ) -> list[AnnotationResponse]:
        """Return a paginated, filtered list of annotations.

        Args:
            species: If provided, filter to this species.
            growth_stage: If provided, filter to `seedling` or `mature`.
            status: If provided, filter to the annotation status.
            dataset_id: If provided, filter to annotation linked to this dataset.
            page: 1-indexed page number.
            page_size: Number of records per page (1-100).
            role: `UserRole` of the logged in user.
            requesting_user_id: ID of the logged in user.
            order: Sort direction for `created_at` - `"asc"` for oldest
                first, `"desc"` for newest first.

        Returns:
            A list of `AnnotationResponse`.
        """
        q = select(Annotation)

        if role != UserRole.ADMIN and requesting_user_id is not None:
            q = q.where(Annotation.created_by == requesting_user_id)

        if species:
            q = q.where(Annotation.species == species)
        if growth_stage:
            q = q.where(Annotation.growth_stage == growth_stage)
        if status:
            q = q.where(Annotation.status == status)
        if dataset_id is not None:
            q = q.where(Annotation.dataset_id == dataset_id)

        offset = (page - 1) * page_size
        sort = Annotation.created_at.asc() if order == "asc" else Annotation.created_at.desc()
        q = q.order_by(sort, Annotation.id.asc()).offset(offset).limit(page_size)
        result = await self._db.execute(q)
        return [_row_to_schema(r, self._storage) for r in result.scalars().all()]

    async def list_pending_for_dataset(
        self,
        dataset_id: uuid.UUID,
    ) -> list[Annotation]:
        """Return all pending annotations for a dataset that have an image_uri.

        Used by the run-inference background task to find images to process.
        Returns raw ORM rows so the caller can update them in-place.

        Args:
            dataset_id: UUID of the dataset to query.

        Returns:
            List of Annotation ORM rows with status='pending' and image_uri set.
        """
        result = await self._db.execute(
            select(Annotation)
            .where(Annotation.dataset_id == dataset_id)
            .where(Annotation.status == AnnotationStatus.PENDING)
            .where(Annotation.image_uri.isnot(None))
            .order_by(Annotation.created_at.asc())
        )
        return list(result.scalars().all())

    async def get(
        self,
        annotation_id: uuid.UUID,
        role: UserRole = UserRole.USER,
        requesting_user_id: uuid.UUID | None = None,
    ) -> AnnotationResponse | None:
        """Return the annotation with `annotation_id`, or `None`.

        Args:
            annotation_id: UUID to look up.
            role: `UserRole` of the logged in user.
            requesting_user_id: ID of the logged in user.

        Returns:
            `AnnotationResponse` if found, otherwise `None`.
        """
        row = await self._db.get(Annotation, annotation_id)
        if row is None:
            return None

        # Ownership check
        if role != UserRole.ADMIN and requesting_user_id is not None:
            if row.created_by is not None:
                if row.created_by != requesting_user_id:
                    return None
            elif row.prediction_id is not None:
                pred_result = await self._db.execute(select(PredictionORM).where(PredictionORM.id == row.prediction_id))
                pred_row = pred_result.scalar_one_or_none()
                if pred_row is not None and pred_row.user_id != requesting_user_id:
                    return None

        # Get owner email
        owner_email: str | None = None
        if row.prediction_id is not None:
            pred_result = await self._db.execute(
                select(PredictionORM)
                .options(selectinload(PredictionORM.user))
                .where(PredictionORM.id == row.prediction_id)
            )
            pred_row = pred_result.scalar_one_or_none()
            if pred_row is not None and pred_row.user is not None:
                owner_email = pred_row.user.email

        # Fall back to direct creator relationship
        if owner_email is None and row.created_by is not None:
            from api.src.db.orm import User as UserORM

            user_result = await self._db.execute(select(UserORM).where(UserORM.id == row.created_by))
            user_row = user_result.scalar_one_or_none()
            if user_row is not None:
                owner_email = user_row.email

        return _row_to_schema(
            row,
            self._storage,
            owner_email=owner_email,
        )

    async def delete(self, annotation_id: uuid.UUID) -> bool:
        """Delete the annotation with `annotation_id`.

        Args:
            annotation_id: UUID to remove.

        Returns:
            `True` if delted, `False` if not found.
        """
        row = await self._db.get(Annotation, annotation_id)
        if row is None:
            return False
        await self._db.delete(row)
        await self._db.commit()
        logger.debug("Deleted annotation %s", annotation_id)
        return True

    async def confirm(
        self,
        annotation_id: uuid.UUID,
        mask_url: str,
    ) -> AnnotationResponse | None:
        """Set the annotation status to `annotated` and update its mask URL.

        Args:
            annotation_id: UUID of the annotation to confirm.
            mask_url: URL of the saved mask.

        Returns:
            The updated `AnnotationResponse`, or `None` if the annotation
            does not exist.
        """
        row = await self._db.get(Annotation, annotation_id)
        if row is None:
            return None

        row.mask_uri = mask_url
        row.status = AnnotationStatus.ANNOTATED
        row.updated_at = datetime.now(timezone.utc)
        await self._db.commit()
        await self._db.refresh(row)
        logger.info("Confirmed annotation %s", annotation_id)
        return _row_to_schema(row, self._storage)

    async def create_from_prediction(
        self,
        prediction_id: uuid.UUID,
        mask_uri: str | None,
        tip_photo_uri: str | None,
        input_hash: str | None,
        image_uri: str | None,
        species: str | None,
        age_days: int | None,
        treatment: str | None,
        dataset_id: uuid.UUID | None = None,
        created_by: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Create a new annotation record from a confirmed prediction.

        Called by `DatabasePredictionRepository.confirm()` in the same
        transaction as setting `predictions.confirmed_at`.

        Args:
            prediction_id: UUID of the prediction to link.
            mask_uri: URI of the inference mask to use as initial ground truth.
            tip_photo_uri: URI of the inference tip photo.
            input_hash: Hash string of the uploaded image.
            image_uri: URI of the uploaded image.
            species: Optional plant species.
            age_days: Optional plant age in days.
            treatment: Optional experimental treatment.
            dataset_id: Optional UUID of the dataset to link this annotation to.
                When provided, the caller must also call
                DatasetRepository.increment_count() in the same transaction.
            created_by: UUID of the logged in user.

        Returns:
            UUID of the newly created annotation.
        """
        if input_hash and dataset_id:
            existing = await self._db.execute(
                select(Annotation).where(
                    Annotation.dataset_id == dataset_id,
                    Annotation.image_hash == input_hash,
                )
            )
            if existing.scalar_one_or_none() is not None:
                raise ValueError("An identical image already exists in this dataset.")

        now = datetime.now(timezone.utc)
        annotation = Annotation(
            id=uuid.uuid4(),
            prediction_id=prediction_id,
            dataset_id=dataset_id,
            status=AnnotationStatus.ANNOTATED,
            image_uri=image_uri,
            mask_uri=mask_uri,
            tip_photo_uri=tip_photo_uri,
            species=species,
            age_days=age_days,
            treatment=treatment,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        self._db.add(annotation)
        # Caller is responsible for flush/commit
        logger.info("Created annotation for prediction %s", prediction_id)
        return annotation.id

    async def create_pending(
        self,
        dataset_id: uuid.UUID,
        image_uri: str,
        image_hash: str,
        created_by: uuid.UUID | None = None,
    ) -> AnnotationResponse:
        """Create a pending annotation for an uploaded dataset image.

        Called by POST /datasets/{id}/upload. There is no prediction link
        this annotation represents a raw uploaded image awaiting inference
        or manual annotation.

        The caller is responsible for comitting the transaction and
        calling DatasetRepository.increment_count() in the same transaction.

        Args:
            dataset_id: UUID of the dataset this image belongs to.
            image_uri: Storage URI of the uploaded image file.
            image_hash: Hash string of the uploaded image.
            created_by: UUID of the logged in user.

        Returns:
            AnnotationResponse with status='pending'.
        """
        now = datetime.now(timezone.utc)
        annotation = Annotation(
            id=uuid.uuid4(),
            prediction_id=None,
            dataset_id=dataset_id,
            status=AnnotationStatus.PENDING,
            image_hash=image_hash,
            image_uri=image_uri,
            mask_uri=None,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        self._db.add(annotation)
        logger.info(
            "Created pending annotation %s for dataset %s",
            annotation.id,
            dataset_id,
        )
        return _row_to_schema(annotation, self._storage)

    async def update_metadata(
        self,
        annotation_id: uuid.UUID,
        species: str | None,
        age_days: int | None,
        treatment: str | None,
        growth_stage: str | None,
    ) -> AnnotationResponse | None:
        """Update editable metadata fields on an annotation.

        Passes None to leave existing values unchanged (explicit patch semantics).

        Args:
            annotation_id: UUID of the annotation to update.
            species: Updated species field.
            age_days: Updated age_days field.
            treatment: Updated treatment field.
            growth_stage: Updated growth_stage field.

        Returns:
            AnnotationResponse
        """
        row = await self._db.get(Annotation, annotation_id)
        if row is None:
            return None
        if species is not None:
            row.species = species
        if age_days is not None:
            row.age_days = age_days
        if treatment is not None:
            row.treatment = treatment
        if growth_stage is not None:
            row.growth_stage = growth_stage
        row.updated_at = datetime.now(timezone.utc)
        await self._db.commit()
        await self._db.refresh(row)
        return _row_to_schema(row, self._storage)

    async def set_status(
        self,
        prediction_id: uuid.UUID,
        status: AnnotationStatus,
    ) -> AnnotationResponse | None:
        """Set annotation status to review_required.

        Args:
            prediction_id: UUID of the prediction that the annotation is linked to.
            status: AnnotationStatus to set.

        Returns:
            Updated AnnotationResponse, or None if not found.
        """
        result = await self._db.execute(select(Annotation).where(Annotation.prediction_id == prediction_id))

        row = result.scalar_one_or_none()
        if row is None:
            return None
        row.status = status
        row.updated_at = datetime.now(timezone.utc)
        await self._db.commit()
        await self._db.refresh(row)
        logger.info("Annotation %s flagged as review required.", row.id)
        return _row_to_schema(row, self._storage)
