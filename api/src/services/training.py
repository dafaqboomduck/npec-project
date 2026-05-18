"""Training service interface and stub implementation.

AbstractTrainingService defines the interface the router depends on.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from uuid import UUID

from api.src.schemas.training import TrainingRunDetail, TrainingRunSummary, TrainingStartRequest, TrainingStartResponse

logger = logging.getLogger(__name__)


class AbstractTrainingService(ABC):
    """Interface for managing training runs."""

    @abstractmethod
    async def start(self, hyperparameters: TrainingStartRequest | None) -> TrainingStartResponse:
        """Insert a training_runs row with status='running' and launch the job.

        Args:
            hyperparameters: Optional overrides for learning_rate, batch_size,
                and epochs. Uses pipeline defaults when None.

        Returns:
            TrainingStartResponse containing the new run_id.
        """

    @abstractmethod
    async def get_runs(self) -> list[TrainingRunSummary]:
        """Return all training runs ordered by started_at descending.

        Returns:
          List of TrainingRunSummary.
        """

    @abstractmethod
    async def get_run(self, run_id: UUID) -> TrainingRunDetail | None:
        """Return the training run with run_id, or None if not found.

        Args:
            run_id: UUID of the training run to fetch.

        Returns:
            TrainingRunDetail if found, None otherwise.
        """

    @abstractmethod
    async def stop(self, run_id: UUID) -> bool:
        """Terminate the running subprocess and set status to 'stopped'.

        Args:
            run_id: UUID of the training run to stop.

        Returns:
            True if the run was stopped, False if it was not found.

        Raises:
            ValueError: If the run exists but is not in 'running' state.
        """


class StubTrainingService(AbstractTrainingService):
    """Placeholder implementation - all methods raise NotImplementedError."""

    async def start(self, hyperparameters: TrainingStartRequest | None) -> TrainingStartResponse:
        """Not implemented yet."""
        raise NotImplementedError

    async def get_runs(self) -> list[TrainingRunSummary]:
        """Not implemented yet."""
        raise NotImplementedError

    async def get_run(self, run_id: UUID) -> TrainingRunDetail | None:
        """Not implemented yet."""
        raise NotImplementedError

    async def stop(self, run_id: UUID) -> bool:
        """Not implemented yet."""
        raise NotImplementedError
