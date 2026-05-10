"""Abstract inference service interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas.predictions import PredictionResponse


class AbstractInferenceService(ABC):
    """Interface for running inference on an uploaded image."""

    @abstractmethod
    async def run(
        self,
        image_bytes: bytes,
        filename: str,
        confidence_threshold: float,
    ) -> PredictionResponse:
        """Execute inference on raw image bytes.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            filename: Original filename, used to determine format.
            confidence_threshold: Minimum per-plant confidence to include in results.

        Returns:
            A fully populated `PredictionResponse`.

        Raises:
            ValueError: If the image cannot be decoded or is otherwise invalid.
        """
