"""Dataset repository.

Reads from the datasets table via the ORM.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.src.db.orm import Dataset
from api.src.schemas.auth import UserRole
from api.src.schemas.library import DatasetDetail, DatasetSummary

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
        owner_email=row.creator.email if row.creator else None,  # ADD
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
        owner_email=row.creator.email if row.creator else None,  # ADD
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
        role: UserRole = UserRole.USER,
        requesting_user_id: uuid.UUID | None = None,
    ) -> list[DatasetSummary]:
        """Return all datasets, optionally filtered.

        Args:
            species: If provided, only return datasets matching this species.
            growth_stage: If provided, only return datasets matching this
                growth_stage (`seedling` or `mature`).
            role: `UserRole` of the logged in user.
            requesting_user_id: ID of the logged in user.

        Returns:
            A list of `DatasetSummary`.
        """
        q = select(Dataset).options(selectinload(Dataset.creator))
        if role != UserRole.ADMIN and requesting_user_id is not None:
            q = q.where(Dataset.created_by == requesting_user_id)
        if species:
            q = q.where(Dataset.species == species)
        if growth_stage:
            q = q.where(Dataset.growth_stage == growth_stage)
        result = await self._db.execute(q)
        rows = result.scalars().all()
        return [_row_to_summary(r) for r in rows]

    async def increment_count(self, dataset_id: uuid.UUID) -> None:
        """Increment the denormalised count column by 1.

        Does not commit - the caller is responsible for committing the
        encolsing transaction. Raises ValueError if the dataset does not exist.

        Args:
            dataset_id: UUID of the dataset whose count to increment.

        Raises:
            ValueError: If no dataset with dataset_id exists.
        """
        row = await self._db.get(Dataset, dataset_id)
        if row is None:
            raise ValueError(f"Dataset {dataset_id} not found.")
        row.count += 1
        logger.debug("Incremented count for dataset %s to %d", dataset_id, row.count)

    async def get_true_count(self, dataset_id: uuid.UUID) -> int:
        """Return the real count of annotations linked to this dataset."""
        from api.src.db.orm import Annotation

        result = await self._db.execute(select(func.count(Annotation.id)).where(Annotation.dataset_id == dataset_id))
        return result.scalar_one_or_none() or 0

    async def get(
        self,
        dataset_id: uuid.UUID,
        role: UserRole = UserRole.USER,
        requesting_user_id: uuid.UUID | None = None,
    ) -> DatasetDetail | None:
        """Return the dataset with `dataset_id`, or `None`.

        Args:
            dataset_id: UUID to look up.
            role: `UserRole` of the logged in user.
            requesting_user_id: ID of the logged in user.

        Returns:
            `DatasetDetail` if found, otherwise `None`.
        """
        q = select(Dataset).options(selectinload(Dataset.creator)).where(Dataset.id == dataset_id)
        if role != UserRole.ADMIN and requesting_user_id is not None:
            q = q.where(Dataset.created_by == requesting_user_id)

        result = await self._db.execute(q)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        detail = _row_to_detail(row)
        detail = detail.model_copy(
            update={"count": await self.get_true_count(dataset_id)},
        )
        return detail

    async def create(
        self,
        name: str,
        description: str | None = None,
        species: str | None = None,
        growth_stage: str | None = None,
        created_by: uuid.UUID | None = None,
    ) -> DatasetDetail:
        """Create a new dataset.

        Args:
            name: Dataset name.
            description: Optional dataset description.
            species: Optional dataset species.
            growth_stage: Optional dataset growth stage.
            created_by: UUID of the logged in user.

        Returns:
            The created DatasetDetail.
        """
        row = Dataset(
            name=name,
            description=description,
            species=species,
            growth_stage=growth_stage,
            created_by=created_by,
        )
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)

        # Refresh with creator loaded
        result = await self._db.execute(
            select(Dataset).options(selectinload(Dataset.creator)).where(Dataset.id == row.id)
        )
        row = result.scalar_one()
        return _row_to_detail(row)

    async def delete(self, dataset_id: uuid.UUID) -> bool:
        """Delete a dataset by its ID.

        Args:
            dataset_id: UUID of the dataset to delete.

        Returns:
            True if the dataset was deleted, False if it did not exist.
        """
        row = await self._db.get(Dataset, dataset_id)
        if row is None:
            return False
        await self._db.delete(row)
        await self._db.commit()
        return True
