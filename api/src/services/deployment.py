"""Deployment service interface and stub implementation.

The router never contains deployment logic - it delegates entirely to
AbstractDeploymentServic.

LocalDeploymentService (active for local + on-premise):
    - Reads artifact_uri from the model_versions row
    - Verifies the artifact directory exists on disk
    - Copies it to the versioned path ./artifacts/models/{version}/
    - Archives the previously deployed model in the same DB transaction
    - Signals the UNetInferenceService singleton to reload on next request
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.schemas.library import ModelStatus
from api.src.services.base import AbstractInferenceService

logger = logging.getLogger(__name__)
_ARTIFACT_DIR = Path(os.getenv("NPEC_ARTIFACT_DIR", "/app/artifacts/models"))


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


class LocalDeploymentService(AbstractDeploymentService):
    """Local/on-premise deployment service.

    Copies the model artifact to the versioned active path, archives the
    previous deployed model in the same DB transaction, and signals the
    inference service singleton to reload weights on the next request.

    The entire operation is at the DB level - the file copy happens
    before the transaction commits so a copy failure leaves the DB unchanged.
    """

    def __init__(
        self,
        db: AsyncSession,
        inference_service: AbstractInferenceService,
    ) -> None:
        """Initialise with a DB session and the inference service singleton.

        Args:
            db: An open async SQLAlchemy session.
            inference_service: The active AbstractInferenceService singleton.
        """
        self._db = db
        self._inference_service = inference_service

    async def deploy(self, model_id: UUID) -> None:
        """Deploy the model with `model_id`.

        Steps:
            1. Load the model_versions row and verify artifact_uri is set.
            2. Verify the source path is a directory on disk.
            3. Copy the entire directory to ./artifacts/models/{version}/.
            4. Archive the currently deployed model.
            5. Mark this model as deployed with deployed_at = now().
            6. Commit the transaction.
            7. Signal the inference service singleton to reload.

        Args:
            model_id: UUID of the model to deploy.

        Raises:
            ValueError: If the model does not exist or artifact_uri is NULL.
            FileNotFoundError: If the directory at artifact_uri does not exist.
        """
        from api.src.db.orm import ModelVersion

        # 1. Load row
        row = await self._db.get(ModelVersion, model_id)
        if row is None:
            raise ValueError(f"Model {model_id} not found.")

        if not row.artifact_uri:
            raise ValueError(
                f"Model {model_id} has no artifact_uri set. "
                "Register the model with an artifact_uri before deploying."
            )

        # 2. Verify source directory exists
        source_path = Path(row.artifact_uri)
        if not source_path.is_dir():
            raise FileNotFoundError(f"Artifact directory not found at '{source_path}'.")

        # 3. Copy entire directory to versioned active path
        dest_dir = _ARTIFACT_DIR / row.version
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(source_path, dest_dir)
        logger.info("Copied artifact directory from '%s' to '%s'", source_path, dest_dir)

        # 4. Archive the currently deployed model
        await self._db.execute(
            update(ModelVersion)
            .where(ModelVersion.status == ModelStatus.DEPLOYED.value)
            .where(ModelVersion.id != model_id)
            .values(status=ModelStatus.ARCHIVED.value)
        )

        # 5. Mark this model as deployed
        row.status = ModelStatus.DEPLOYED.value
        row.artifact_uri = str(dest_dir)
        row.deployed_at = datetime.now(timezone.utc)

        # 6. Commit
        await self._db.commit()
        logger.info(
            "Deployed model %s (version='%s') - artifact at '%s'",
            model_id,
            row.version,
            dest_dir,
        )

        # 7. Signal the inference service to reload on next request
        from api.src.services.unet_inference import UNetInferenceService

        if isinstance(self._inference_service, UNetInferenceService):
            self._inference_service.set_artifact_path(dest_dir)
            logger.info(
                "Inference service artifact path updated to '%s' - " "pipeline will reload on next request.",
                dest_dir,
            )
        else:
            logger.debug(
                "Inference service is not UNetInferenceService (%s) - " "artifact path update skipped.",
                type(self._inference_service).__name__,
            )
