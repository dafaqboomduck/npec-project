"""Dependency injection wiring.

This module is the single place where concrete implementations are bound
to their abstract interfaces. Routers import only the get_* functions
below. They never import a concrete class directly.
"""

from src.api.repositories.base import AbstractPredictionRepository
from src.api.repositories.memory import InMemoryPredictionRepository
from src.api.services.base import AbstractInferenceService
from src.api.services.mock_inference import MockInferenceService

# Module-level singletons

_prediction_repo: AbstractPredictionRepository = InMemoryPredictionRepository()
_inference_service: AbstractInferenceService = MockInferenceService()


def get_prediction_repo() -> AbstractPredictionRepository:
    """Yield the active prediction repository.

    Returns:
        The currently configured :class:`~api.repositoriess.base.AbstractPredictionRepository`.
    """
    return _prediction_repo


def get_inference_service() -> AbstractInferenceService:
    """Yield the active inference service.

    Returns:
        The currently configured :class:`~api.repositories.base.AbstractInferenceService`.
    """
    return _inference_service
