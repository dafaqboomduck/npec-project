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

from api.src.db.orm import Annotation
from api.src.schemas.library import AnnotationResponse, AnnotationStatus

logger = logging.getLogger(__name__)


def _row_to_schema(row: Annotation) -> AnnotationResponse:
    """Convert a `Annotation` ORM row to a `AnnotationResponse`.

    Args:
        row: The ORM row from the database.

    Returns:
        AnnotationResponse
    """
    return AnnotationResponse(
        annotation_id=row.id,
        prediction_id=row.prediction_id,
        dataset_id=row.dataset_id,
        status=AnnotationStatus(row.status),
        image_uri=row.image_uri,
        mask_url=row.mask_uri,
        species=row.species,
        age_days=row.age_days,
        treatment=row.treatment,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class AnnotationRepository:
    """Paginated, filtered read access to the `annotations` table."""

    def __init__(self, db: AsyncSession) -> None:
        """Paginated, filtered read access to the `annotations` table.

        Args:
            db: An open async SQLAlchemy session injected by ``get_session``.
        """
        self._db = db

    async def list(
        self,
        species: str | None = None,
        growth_stage: str | None = None,
        dataset_id: uuid.UUID | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[AnnotationResponse]:
        """Return a paginated, filtered list of annotations.

        Args:
            species: If provided, filter to this species.
            growth_stage: If provided, filter to `seedling` or `mature`.
            dataset_id: If provided, filter to annotation linked to this dataset.
            page: 1-indexed page number.
            page_size: Number of records per page (1-100).

        Returns:
            A list of `AnnotationResponse`.
        """
        q = select(Annotation)
        if species:
            q = q.where(Annotation.species == species)
        if growth_stage:
            q = q.where(Annotation.growth_stage == growth_stage)
        if dataset_id is not None:
            q = q.where(Annotation.dataset_id == dataset_id)

        offset = (page - 1) * page_size
        q = q.order_by(Annotation.created_at.desc()).offset(offset).limit(page_size)

        result = await self._db.execute(q)
        return [_row_to_schema(r) for r in result.scalars().all()]

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

    async def get(self, annotation_id: uuid.UUID) -> AnnotationResponse | None:
        """Return the annotation with `annotation_id`, or `None`.

        Args:
            annotation_id: UUID to look up.

        Returns:
            `AnnotationResponse` if found, otherwise `None`.
        """
        row = await self._db.get(Annotation, annotation_id)
        if row is None:
            return None
        return _row_to_schema(row)

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
        return _row_to_schema(row)

    async def create_from_prediction(
        self,
        prediction_id: uuid.UUID,
        mask_uri: str | None,
        species: str | None,
        age_days: int | None,
        treatment: str | None,
        dataset_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Create a new annotation record from a confirmed prediction.

        Called by `DatabasePredictionRepository.confirm()` in the same
        transaction as setting `predictions.confirmed_at`.

        Args:
            prediction_id: UUID of the prediction to link.
            mask_uri: URI of the inference mask to use as initial ground truth.
            species: Optional plant species.
            age_days: Optional plant age in days.
            treatment: Optional experimental treatment.
            dataset_id: Optional UUID of the dataset to link this annotation to.
                When provided, the caller must also call
                DatasetRepository.increment_count() in the same transaction.

        Returns:
            UUID of the newly created annotation.
        """
        now = datetime.now(timezone.utc)
        annotation = Annotation(
            id=uuid.uuid4(),
            prediction_id=prediction_id,
            dataset_id=dataset_id,
            status=AnnotationStatus.ANNOTATED,
            mask_uri=mask_uri,
            species=species,
            age_days=age_days,
            treatment=treatment,
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

        Returns:
            AnnotationResponse with status='pending'.
        """
        now = datetime.now(timezone.utc)
        annotation = Annotation(
            id=uuid.uuid4(),
            prediction_id=None,
            dataset_id=dataset_id,
            status=AnnotationStatus.PENDING,
            image_uri=image_uri,
            mask_uri=None,
            created_at=now,
            updated_at=now,
        )
        self._db.add(annotation)
        logger.info(
            "Created pending annotation %s for dataset %s",
            annotation.id,
            dataset_id,
        )
        return _row_to_schema(annotation)
