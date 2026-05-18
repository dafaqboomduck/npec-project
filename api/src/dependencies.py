"""Dependency injection wiring.

This module is the single place where concrete implementations are bound
to their abstract interfaces. Routers import only the get_* functions
below. They never import a concrete class directly.
"""

from __future__ import annotations

import logging
import os

from api.src.services.base import AbstractInferenceService
from api.src.services.mock_inference import MockInferenceService
from api.src.services.storage import AbstractStorageBackend, get_storage_backend
from api.src.services.training import AbstractTrainingService, StubTrainingService

logger = logging.getLogger(__name__)

# Feature flags

_USE_MOCK: bool = os.getenv("USE_MOCK_INFERENCE", "true").lower() != "false"

# Module-level singletons
_storage_backend: AbstractStorageBackend = get_storage_backend()
_inference_service: AbstractInferenceService | None = None


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

            _inference_service = UNetInferenceService()
            logger.info("Inference service: UNetInferenceService")

    return _inference_service


def get_storage() -> AbstractStorageBackend:
    """Return the module-level storage backend singleton.

    Returns:
        The configured `AbstractStorageBackend`.
    """
    return _storage_backend


def get_training_service() -> AbstractTrainingService:
    """Return the active training service.

    Returns:
      The configured AbstractTrainingService.
    """
    return StubTrainingService()
