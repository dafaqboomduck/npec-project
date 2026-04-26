"""Annotation repository.

Reads from annotations table via the ORM. Pagination is handled
entirely in SQL via OFFSET/LIMIT - no post-fetch slicing.
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.orm import Annotation
from src.api.schemas.library import AnnotationResponse, AnnotationStatus

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
        scan_id=row.scan_id,
        status=AnnotationStatus(row.status),
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
        page: int = 1,
        page_size: int = 20,
    ) -> list[AnnotationResponse]:
        """Return a paginated, filtered list of annotations.

        Args:
            species: If provided, filter to this species.
            growth_stage: If provided, filter to `seedling` or `mature`.
            page: 1-indexed page number.
            page_size: Number of records per page (1-100).

        Returns:
            A list of `AnnotationResponse`.
        """
        q = select(Annotation)
        if species:
            q.where(Annotation.species == species)
        if growth_stage:
            q.where(Annotation.growth_stage == growth_stage)

        offset = (page - 1) * page_size
        q = q.order_by(Annotation.created_at.desc()).offset(offset).limit(page_size)

        result = await self._db.execute(q)
        return [_row_to_schema(r) for r in result.scalars().all()]

    async def get(self, annotation_id: UUID) -> AnnotationResponse | None:
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
