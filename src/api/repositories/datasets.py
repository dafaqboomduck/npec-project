"""Dataset repository.

Reads from the datasets table via the ORM.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.orm import Dataset
from src.api.schemas.library import DatasetDetail, DatasetSummary

logger = logging.getLogger(__name__)


def _row_to_summary(row: Dataset) -> DatasetSummary:
    """Convert a `Dataset` ORM row to a `DatasetSummary`.

    Args:
        row: The ORM row from the database.

    Returns:
        DatasetSummary
    """
    return DatasetSummary(
        dataset_id=row.id,
        name=row.name,
        created_at=row.created_at,
        modified_at=row.modified_at,
        count=row.count,
    )


def _row_to_detail(row: Dataset) -> DatasetDetail:
    """Convert a `Dataset` ORM row to a `DatasetDetail`.

    Args:
        row: The ORM row from the database.

    Returns:
        DatasetDetail
    """
    return DatasetDetail(
        dataset_id=row.id,
        name=row.name,
        created_at=row.created_at,
        modified_at=row.modified_at,
        count=row.count,
        description=row.description,
        species=row.species,
    )


class DatasetRepository:
    """Read-only access to the `datasets` table."""

    def __init__(self, db: AsyncSession) -> None:
        """Read-only access to the `datasets` table.

        Args:
            db: An open async SQLAlchemy session injected by `get_session`.
        """
        self._db = db

    async def list(
        self,
        species: str | None,
        growth_stage: str | None = None,
    ) -> list[DatasetSummary]:
        """Return all datasets, optionally filtered.

        Args:
            species: If provided, only return datasets matching this species.
            growth_stage: If provided, only return datasets matching this
                growth_stage (`seedling` or `mature`).

        Returns:
            A list of `DatasetSummary`.
        """
        q = select(Dataset)
        if species:
            q = q.where(Dataset.species == species)
        if growth_stage:
            q = q.where(Dataset.growth_stage == growth_stage)
        result = await self._db.execute(q)
        rows = result.scalars().all()
        return [_row_to_summary(r) for r in rows]

    async def get(self, dataset_id: uuid.UUID) -> DatasetDetail | None:
        """Return the dataset with `dataset_id`, or `None`.

        Args:
            dataset_id: UUID to look up.

        Returns:
            `DatasetDetail` if found, otherwise `None`.
        """
        row = await self._db.get(Dataset, dataset_id)
        if row is None:
            return None
        return _row_to_detail(row)
