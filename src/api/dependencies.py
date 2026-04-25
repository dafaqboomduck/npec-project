"""Dependency injection wiring.

This module is the single place where concrete implementations are bound
to their abstract interfaces. Routers import only the get_* functions
below. They never import a concrete class directly.
"""

import logging
import os
import uuid
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from src.api.core.security import get_current_user
from src.api.repositories.base import AbstractPredictionRepository
from src.api.repositories.memory import InMemoryPredictionRepository
from src.api.schemas.auth import TokenData
from src.api.services.base import AbstractInferenceService
from src.api.services.mock_inference import MockInferenceService

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
    from src.api.services.unet_inference import UNetInferenceService

    return UNetInferenceService()


def get_prediction_repo(
    user: Annotated[TokenData, Depends(get_current_user)],
    db: AsyncSession = Depends(get_session),
) -> AbstractPredictionRepository:
    """Yield the active prediction repository.

    Controlled by the ``USE_DB_STORAGE`` environment variable.

    Args:
        user: The authenticated user.
        db: The current request's database session.

    Returns:
        The currently configured `AbstractPredictionRepository`.
        Scoped to the current user and request session.
    """
    if _USE_DB:
        from src.api.repositories.database import DatabasePredictionRepository

        return DatabasePredictionRepository(db=db, user_id=uuid.UUID(user.sub))

    return _in_memory_repo


# Module-level in-memory singleton - shared across requests, lost on restart.
_in_memory_repo: AbstractPredictionRepository = InMemoryPredictionRepository()
