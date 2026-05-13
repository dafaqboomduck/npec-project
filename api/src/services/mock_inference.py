"""Mock inference service for development and frontend integration."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from api.src.schemas.predictions import (
    Plant,
    PredictionMetadata,
    PredictionResponse,
    PredictionStatus,
)
from api.src.services.base import AbstractInferenceService

logger = logging.getLogger(__name__)

# Version string returned while the mock is active
_MOCK_MODEL_VERSION = "mock-v0.0.0"


class MockInferenceService(AbstractInferenceService):
    """Fake inference that satisfies the API contract without a real model.

    The returned plants and confidence values are hard-coded to match the
    example response in `api_spec.yaml`, making it easy to verify that the
    frontend renders results correctly.
    """

    async def run(
        self,
        image_bytes: bytes,
        filename: str,
        confidence_threshold: float,
    ) -> tuple[PredictionResponse, None]:
        """Return a mock prediction with two plants.

        Args:
            image_bytes: Ignored by this implementation.
            filename: Ignored by this implementation.
            confidence_threshold: Plants below this threshold are excluded.

        Returns:
            A mock `PredictionResponse`.
        """
        start = time.perf_counter()

        mock_plants = [
            Plant(
                plant_id=1,
                tip_x=412,
                tip_y=1803,
                confidence=0.94,
                root_length_px=1643.2,
                root_length_mm=88.7,
            ),
            Plant(
                plant_id=2,
                tip_x=967,
                tip_y=1765,
                confidence=0.87,
                root_length_px=1501.4,
                root_length_mm=81.1,
            ),
        ]

        # Respect the caller's confidence threshold so the endpoint behavior
        # is realistic even with mock data.
        filtered_plants = [p for p in mock_plants if p.confidence >= confidence_threshold]

        overall_confidence = (
            sum(p.confidence for p in filtered_plants) / len(filtered_plants) if filtered_plants else 0.0
        )

        processing_ms = int((time.perf_counter() - start) * 1000)

        prediction = PredictionResponse(
            prediction_id=uuid.uuid4(),
            status=PredictionStatus.OK,
            metadata=PredictionMetadata(
                model_version=_MOCK_MODEL_VERSION,
                inference_timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_ms,
                input_resolution=[2775, 2048],
                input_channels=1,
                normalisation_applied=True,
                scale_factor_mm_per_px=0.05405,
            ),
            segmentation_mask_url=None,  # No mask generated in mock mode
            plants=filtered_plants,
            overall_confidence=round(overall_confidence, 4),
            warnings=["MOCK_INFERENCE"] if not filtered_plants else [],
        )

        logger.info(
            "MockInferenceService: returning %d plant(s) for '%s'",
            len(filtered_plants),
            filename,
        )

        return prediction, None
