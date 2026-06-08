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
from api.src.services.storage import AbstractStorageBackend, get_storage_backend
from api.src.services.training import AbstractTrainingService

logger = logging.getLogger(__name__)

# Deployment environment
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "local")

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
        if DEPLOYMENT_ENV == "cloud":
            from api.src.services.azure_inference import AzureMLInferenceService

            _inference_service = AzureMLInferenceService()
            logger.info("Inference service: AzureMLInferenceService")
        elif _USE_MOCK:
            from api.src.services.mock_inference import MockInferenceService

            _inference_service = MockInferenceService()
            logger.info("Inference service: MockInferenceService")
        else:
            from api.src.services.unet_inference import UNetInferenceService

            _inference_service = UNetInferenceService(artifact_path=_FALLBACK_ARTIFACT_DIR)
            logger.info("Inference service: UNetInferenceService (artifact_path=%s)", _FALLBACK_ARTIFACT_DIR)

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

    Controlled by DEPLOYMENT_ENV.

    Args:
        db: Injected database session.

    Returns:
      The configured AbstractTrainingService implementation.
    """
    if DEPLOYMENT_ENV == "cloud":
        from api.src.services.azure_training import AzureMLTrainingService

        return AzureMLTrainingService(db)
    from api.src.services.training import LocalTrainingService

    return LocalTrainingService(db)


def get_deployment_service(
    db: AsyncSession = Depends(get_session),
) -> AbstractDeploymentService:
    """Return the active deployment service.

    Cloud model deployment is handled by the Azure ML pipeline, so the local
    service is used for local + on-premise deployments.

    Args:
        db: Injected database session.

    Returns:
        The configured AbstractDeploymentService.
    """
    from api.src.services.deployment import LocalDeploymentService

    return LocalDeploymentService(db=db, inference_service=get_inference_service())


async def warmup_inference_service() -> None:
    """Pre-load the inference models at startup.

    Loads the .keras weights before the first request arrives, moving the
    ~30s cold-start cost to server startup. With process isolation on, the
    models are loaded inside the warm worker process (not this long-lived one),
    so they can still be released after the idle timeout.
    """
    if DEPLOYMENT_ENV == "cloud":
        logger.info("Inference warmup skipped — AzureMLInferenceService has no local pipeline.")
        return

    if _USE_MOCK:
        logger.info("Inference warmup skipped (mock mode)")
        return

    try:
        await get_inference_service().warmup()
    except Exception as exc:
        logger.warning(
            "Inference pipeline warmup failed: %s — " "first prediction will trigger a retry",
            exc,
        )


async def shutdown_inference_service() -> None:
    """Release inference resources at server shutdown.

    Cancels the idle reaper and tears down the inference worker (if any) so
    worker processes — and their GPU contexts — are not orphaned when the API
    stops.
    """
    if _inference_service is not None:
        await _inference_service.aclose()
        logger.info("Inference worker shut down.")
