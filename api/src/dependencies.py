"""Dependency injection wiring.

This module is the single place where concrete implementations are bound
to their abstract interfaces. Routers import only the get_* functions
below. They never import a concrete class directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.session import get_session
from api.src.services.base import AbstractInferenceService
from api.src.services.deployment import AbstractDeploymentService
from api.src.services.mock_inference import MockInferenceService
from api.src.services.storage import AbstractStorageBackend, get_storage_backend
from api.src.services.training import AbstractTrainingService, LocalTrainingService

logger = logging.getLogger(__name__)

# Feature flags

_USE_MOCK: bool = os.getenv("USE_MOCK_INFERENCE", "true").lower() != "false"

# Fallback artifact directory used when no model is deployed in the DB yet.
_FALLBACK_ARTIFACT_DIR = Path(os.getenv("NPEC_ARTIFACT_DIR", "/app/artifacts/models")) / "v1"

# Module-level singletons
_storage_backend: AbstractStorageBackend = get_storage_backend()
_inference_service: AbstractInferenceService | None = None


# Getters


def get_inference_service() -> AbstractInferenceService:
    """Yield the active inference service.

    On the first call, the singleton is created. Controlled by
    the ``USE_MOCK_INFERENCE`` environment variable.

    Returns:
        The currently configured `AbstractInferenceService`.
    """
    global _inference_service

    if _inference_service is None:
        if _USE_MOCK:
            _inference_service = MockInferenceService()
            logger.info("Inference service: MockInferenceService")
        else:
            from api.src.services.unet_inference import UNetInferenceService

            _inference_service = UNetInferenceService(
                artifact_path=_FALLBACK_ARTIFACT_DIR,
            )
            logger.info(
                "Inference service: UNetInferenceService (artifact_path=%s)",
                _FALLBACK_ARTIFACT_DIR,
            )

    return _inference_service


def get_storage() -> AbstractStorageBackend:
    """Return the module-level storage backend singleton.

    Returns:
        The configured `AbstractStorageBackend`.
    """
    return _storage_backend


def get_training_service(
    db: AsyncSession = Depends(get_session),
) -> AbstractTrainingService:
    """Return the active training service.

    Args:
        db: Injected database session.

    Returns:
      The configured AbstractTrainingService implementation.
    """
    return LocalTrainingService(db)


def get_deployment_service(db) -> AbstractDeploymentService:
    """Return the active deployment service.

    Injects the DB session required by LocalDeploymentService.

    Args:
        db: An open AsyncSession from get_session().

    Returns:
        The configured AbstractDeploymentService.
    """
    from api.src.services.deployment import LocalDeploymentService

    return LocalDeploymentService(db=db, inference_service=get_inference_service())
