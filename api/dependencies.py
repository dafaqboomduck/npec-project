"""Dependency injection wiring.

This module is the single place where concrete implementations are bound
to their abstract interfaces. Routers import only the get_* functions
below. They never import a concrete class directly.
"""

from __future__ import annotations

import logging
import os

from .services.base import AbstractInferenceService
from .services.mock_inference import MockInferenceService

logger = logging.getLogger(__name__)

# Feature flags

_USE_MOCK: bool = os.getenv("USE_MOCK_INFERENCE", "true").lower() != "false"
_USE_DB: bool = os.getenv("USE_DB_STORAGE", "false").lower() == "true"

if _USE_MOCK:
    logger.info("Inference: MockInferenceService active (USE_MOCK_INFERENCE=true)")
else:
    logger.info("Inference: UNetInferenceService active (USE_MOCK_INFERENCE=false)")

if _USE_DB:
    logger.info("Storage: DatabasePredictionRepository active (USE_DB_STORAGE=true)")
else:
    logger.info("Storage: InMemoryPredictionRepository active (USE_DB_STORAGE=false)")


# Module-level singletons


def get_inference_service() -> AbstractInferenceService:
    """Yield the active inference service.

    Controlled by the ``USE_MOCK_INFERENCE`` environment variable.

    Returns:
        The currently configured `AbstractInferenceService`.
    """
    if _USE_MOCK:
        return MockInferenceService()

    # Deferred import keeps TensorFlow out of the import chain when mocking.
    from .services.unet_inference import UNetInferenceService

    return UNetInferenceService()
