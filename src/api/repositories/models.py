"""Model repository.

The GET /models/{model_id} returns additional internal fields
(artifact_uri, deployed_at) for admin tokens. The repository
always fetches all fields. Field filtering is applied in the
router based on the JWT role claim.
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.orm import ModelVersion
from src.api.schemas.library import ModelDetail, ModelStatus, ModelSummary

logger = logging.getLogger(__name__)


def _row_to_summary(row: ModelVersion) -> ModelSummary:
    """Convert a `ModelVersion` ORM row to a `ModelSummary`.

    Args:
        row: The ORM row from the database.

    Returns:
        ModelSummary
    """
    return ModelSummary(
        model_id=row.id,
        name=row.name,
        status=ModelStatus(row.status),
        created_at=row.created_at,
    )


def _row_to_detail(row: ModelVersion, include_admin_fields: bool = False) -> ModelDetail:
    """Convert a `ModelVersion` ORM row to a `ModelDetail`.

    Args:
        row: The ORM row from the database.
        include_admin_fields: If `True`, populate `artifact_uri` and
            `deployed_at`. Pass `False` for user-role tokens.

    Returns:
        ModelDetail
    """
    return ModelDetail(
        model_id=row.id,
        name=row.name,
        status=ModelStatus(row.status),
        created_at=row.created_at,
        description=row.description,
        f1_score=row.f1_score,
        precision=row.precision,
        recall=row.recall,
        inference_latency_ms=row.inference_latency_ms,
        # Admin-only fields
        artifact_uri=row.artifact_uri if include_admin_fields else None,
        deployed_at=row.deployed_at if include_admin_fields else None,
    )


class ModelRepository:
    """Read access to the `model_versions` table."""

    def __init__(self, db: AsyncSession) -> None:
        """Read access to the `model_versions` table.

        Args:
            db: An open async SQLAlchemy session injected by `get_session`.
        """
        self._db = db

    async def list(self) -> list[ModelSummary]:
        """Return all model version summaries ordered by creation date.

        Returns:
            A list of `ModelSummary`
        """
        result = await self._db.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc()))
        rows = result.scalars().all()
        return [_row_to_summary(r) for r in rows]

    async def get(
        self,
        model_id: uuid.UUID,
        include_admin_fields: bool = False,
    ) -> ModelDetail | None:
        """Return the model version with `model_id`, or `None`.

        Args:
            model_id: UUID to look up.
            include_admin_fields: Whether to populate admin-only fields.

        Returns:
            `ModelDetail` if found, otherwise `None`.
        """
        row = await self._db.get(ModelVersion, model_id)
        if row is None:
            return None
        return _row_to_detail(row, include_admin_fields=include_admin_fields)
