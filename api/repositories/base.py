"""Abstract repository interfaces.

Every concrete storage backend (in-memory, PostgreSQL, Azure DB, ...)
must subclass these ABCs. Routers and services depend only on these
interfaces, never on a concrete implementation. Swapping the storage layer
is therefore a one-line change in dependencies.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from ..schemas.predictions import PredictionResponse


class AbstractPredictionRepository(ABC):
    """CRUD interface for :class:~api.schemas.predictions.PredictionResponse`."""

    @abstractmethod
    async def save(self, prediction: PredictionResponse) -> PredictionResponse:
        """Persist a prediction and return it.

        Args:
            prediction: The fully populated prediction to store.

        Returns:
            The stored prediction.
        """

    @abstractmethod
    async def get(self, prediction_id: UUID) -> PredictionResponse:
        """Retrieve a prediction by its ID.

        Args:
            prediction_id: UUID of the prediction to fetch.

        Returns:
            The prediction if found, otherwise `None`.
        """

    @abstractmethod
    async def delete(self, prediction_id: UUID) -> bool:
        """Remove a prediction from the store.

        Args:
            prediction_id: UUID of the prediction to remove.

        Returns:
            `True` if a record was deleted, `False` if it did not exist.
        """
