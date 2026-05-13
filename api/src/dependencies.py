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

logger = logging.getLogger(__name__)

# Feature flags

_USE_MOCK: bool = os.getenv("USE_MOCK_INFERENCE", "true").lower() != "false"

# Module-level singletons
_storage_backend: AbstractStorageBackend = get_storage_backend()


def get_inference_service() -> AbstractInferenceService:
    """Yield the active inference service.

    Controlled by the ``USE_MOCK_INFERENCE`` environment variable.

    Returns:
        The currently configured `AbstractInferenceService`.
    """
    if _USE_MOCK:
        return MockInferenceService()

    # Deferred import keeps TensorFlow out of the import chain when mocking.
    from api.src.services.unet_inference import UNetInferenceService

    return UNetInferenceService()


def get_storage() -> AbstractStorageBackend:
    """Return the module-level storage backend singleton.

    Returns:
        The configured `AbstractStorageBackend`.
    """
    return _storage_backend
