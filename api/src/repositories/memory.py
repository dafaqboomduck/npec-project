"""In-memory implementation of AbstractPredictionRepository.

Stores predictions in a plain Python dict keyed by UUID. This is intentionally
simple: it is process-scoped, non-persistent, and not thread-safe beyond the
GIL. It exists purely so that all other layers can be developed and tested
before a real database is available.
"""

from __future__ import annotations

import logging
from uuid import UUID

from api.src.repositories.base import AbstractPredictionRepository
from api.src.schemas.predictions import (
    Plant,
    PredictionMetadata,
    PredictionResponse,
    PredictionStatus,
)

logger = logging.getLogger(__name__)


class InMemoryPredictionRepository(AbstractPredictionRepository):
    """Thread-safe-enough in-memory store backed by a plain `dict`.

    Note:
        Suitable for development and testing only. All data is lost on
        process restart.
    """

    def __init__(self) -> None:
        """Thread-safe-enough in-memory store backed by a plain `dict`."""
        self._store: dict[UUID, PredictionResponse] = {
            UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"): PredictionResponse(
                prediction_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
                status=PredictionStatus.OK,
                metadata=PredictionMetadata(
                    model_version="string",
                    inference_timestamp="2026-04-22T13:04:00.034Z",
                    processing_time_ms=0,
                    input_resolution=[0, 0],
                    input_channels=0,
                    normalisation_applied=True,
                    scale_factor_mm_per_px=0,
                ),
                segmentation_mask_url=None,
                plants=[Plant(plant_id=0, tip_x=0, tip_y=0, confidence=1, root_length_px=0, root_length_mm=0)],
                overall_confidence=1,
                warnings=[],
            )
        }

    async def save(self, prediction: PredictionResponse) -> PredictionResponse:
        """Store `prediction` under its UUID.

        Args:
            prediction: The prediction to persist.

        Returns:
            The stored prediction (identical to input for this implementation).
        """
        self._store[prediction.prediction_id] = prediction
        logger.debug("Saved prediction %s", prediction.prediction_id)
        return prediction

    async def get(self, prediction_id: UUID) -> PredictionResponse | None:
        """Return the prediction with `prediction_id` or `None`.

        Args:
            prediction_id: UUID to look up.

        Returns:
            The matching prediction, or `None` if not found.
        """
        return self._store.get(prediction_id)

    async def delete(self, prediction_id: UUID) -> bool:
        """Remove the prediction with `prediction_id` if it exists.

        Args:
            prediction_id: UUID to remove.

        Returns:
            `True` if the record existed and was deleted, `False` otherwise.
        """
        existed = prediction_id in self._store
        self._store.pop(prediction_id, None)
        if existed:
            logger.debug("Deleted prediction %s", prediction_id)
        return existed
