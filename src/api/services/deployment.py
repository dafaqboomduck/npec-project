"""Deployment service interface and stub implementation.

The router never contains deployment logic - it delegates entirely to
AbstractDeploymentServic. StubDeploymentService updates the DB status
to deployed. In a later sprint a real implementation will trigger an
Azure ML endpoint deployment and only mark the model as deployed once
the cloud operation succeeds.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.library import ModelStatus

logger = logging.getLogger(__name__)


class AbstractDeploymentService(ABC):
    """Interface for deploying a registered model."""

    @abstractmethod
    async def deploy(self, model_id: UUID) -> None:
        """Trigger deployment for the model with `model_id`.

        Args:
            model_id: UUID of the `model_versions` row to deploy.

        Raises:
            ValueError: If the model does not exist or cannot be deployed.
        """


class StubDeploymentService(AbstractDeploymentService):
    """Stub deployment - updates DB status only, no cloud operation.

    Marks the model as `deployed` in the database. First demotes any
    currently deployed model back to `registered` to satisfy the partial
    unique index invariant (only one deployed model at a time).

    Replace with `AzureMLDeploymentService` in a later sprint without
    touching the router.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Stub deployment - updates DB status only, no cloud operation.

        Args:
            db: An open async SQLAlchemy session.
        """
        self._db = db

    async def deploy(self, model_id: UUID) -> None:
        """Set the model's sstatus to `deployed` in the database.

        Args:
            model_id: UUID of the model to deploy.

        Raises:
            ValueError: If the model does not exist.
        """
        from sqlalchemy import update

        from db.orm import ModelVersion

        # Demote any currently deployed model first

        await self._db.execute(
            update(ModelVersion)
            .where(ModelVersion.status == ModelStatus.DEPLOYED)
            .values(status=ModelStatus.REGISTERED)
        )

        row = await self._db.get(ModelVersion, model_id)
        if row is None:
            raise ValueError(f"Model {model_id} not found.")

        row.status = "deployed"
        row.deployed_at = datetime.now(timezone.utc)
        await self._db.commit()

        logger.info("Deployed model %s (stub - DB status only)", model_id)
