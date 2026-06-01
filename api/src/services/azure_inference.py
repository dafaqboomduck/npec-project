"""Azure ML inference service.

Calls the deployed managed online endpoint instead of running
inference locally.

Required environment variables
-------------------------------
AZUREML_ENDPOINT_URI    Full scoring URI of the deployed endpoint
AZUREML_ENDPOINT_KEY    Primary or secondary key for Bearer auth
AZUREML_REQUEST_TIMEOUT Request timeout in seconds (default: 180)
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.schemas.predictions import Plant, PredictionMetadata, PredictionResponse, PredictionStatus
from api.src.services.base import AbstractInferenceService

logger = logging.getLogger(__name__)

_ENDPOINT_URI = os.getenv("AZUREML_ENDPOINT_URI", "")
_ENDPOINT_KEY = os.getenv("AZUREML_ENDPOINT_KEY", "")
_REQUEST_TIMEOUT = int(os.getenv("AZUREML_REQUEST_TIMEOUT", "180"))


class AzureMLInferenceService(AbstractInferenceService):
    """Calls the Azure ML managed online endpoint for inference."""

    async def run(
        self,
        image_bytes: bytes,
        filename: str,
        *,
        db: Optional[AsyncSession] = None,
    ) -> tuple[PredictionResponse, object | None]:
        """Send the image to the Azure ML endpoint and parse the response.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            filename: Original filename - sent to the endpoint for validation.
            db: Unused; accepted for interface compatibility.

        Returns:
            Tuple of (PredictionResponse, None). The endpoint inlines the
            mask as base64 so there is ImageResult object to return.

        Raises:
            ValueError: If the endpoint returns a non-ok status or HTTP error.
        """
        start = time.perf_counter()

        image_b64 = base64.b64encode(image_bytes).decode()

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.post(
                _ENDPOINT_URI,
                headers={
                    "Authorization": f"Bearer {_ENDPOINT_KEY}",
                    "Content-Type": "application/json",
                },
                json={"image_b64": image_b64, "filename": filename},
            )

        if resp.status_code != 200:
            raise ValueError(f"Azure ML endpoint returned HTTP {resp.status_code}: {resp.text[:200]}")

        result = resp.json()

        if isinstance(result, str):
            import json

            result = json.loads(result)

        if result.get("status") != "ok":
            raise ValueError(f"Inference error [{result.get('error_code')}]: {result.get('message')}")

        processing_ms = int((time.perf_counter() - start) * 1000)

        plants = [
            Plant(
                plant_id=p["plant_id"],
                tip_x=p["tip_x"],
                tip_y=p["tip_y"],
                root_length_px=p.get("root_length_px", 0.0),
                root_length_mm=p.get("root_length_mm", 0.0),
            )
            for p in result.get("plants", [])
        ]

        meta = result.get("metadata", {})
        prediction = PredictionResponse(
            prediction_id=uuid.uuid4(),
            status=PredictionStatus.OK,
            metadata=PredictionMetadata(
                model_version=meta.get("model_version", "unknown"),
                inference_timestamp=datetime.now(timezone.utc),
                processing_time_ms=meta.get("processing_time_ms", processing_ms),
                input_resolution=meta.get("input_resolution", [0, 0]),
                input_channels=meta.get("input_channels", 3),
                normalisation_applied=meta.get("normalisation_applied", True),
                scale_factor_mm_per_px=meta.get("scale_factor_mm_per_px", 0.0),
            ),
            segmentation_mask_url=None,  # set later from the mask bytes
            plants=plants,
            warnings=result.get("warnings", []),
        )

        # Return mask bytes as a simple wrapper so artifact_storage can save it
        mask_b64 = result.get("segmentation_mask_b64")
        image_result = (
            _AzureImageResult(
                mask_bytes=base64.b64decode(mask_b64),
                plants=result.get("plants", []),
            )
            if mask_b64
            else None
        )

        logger.info(
            "AzureMLInferenceService: %d plant(s) detected for '%s' in %dms",
            len(plants),
            filename,
            processing_ms,
        )

        return prediction, image_result


class _AzureImageResult:
    """Thin wrapper so save_segmentation_mask can handle cloud responses.

    The local pipeline returns an ImageResult with per-plant masks.
    The Azure endpoint inlines a single mask as base64 PNG.
    This wrapper makes it compatible with save_segmentation mask.
    """

    def __init__(self, mask_bytes: bytes, plants: list[dict]) -> None:
        """Initialise the wrapper.

        Args:
            mask_bytes: bytes of the segmentation mask.
            plants: list of the detected plants.
        """
        import io

        from PIL import Image

        combined_mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))

        # One synthetic plant result per detected plant carrying its tip coord.
        self.plant_results = {
            p["plant_id"]: _AzurePlantResult(
                mask=combined_mask,
                tip_row=p["tip_y"],
                tip_col=p["tip_x"],
            )
            for p in plants
        }


class _AzurePlantResult:
    """Synthetic plant result carrying the pre-ecoded mask array."""

    def __init__(self, mask: "np.ndarray", tip_row: int, tip_col: int) -> None:
        """Initialise the plant result object.

        Args:
            mask: `np.ndarray` segmentation mask.
            tip_row: the y axis location of the tip.
            tip_col: the x axis location of the tip.

        """
        self.mask = mask
        self.detected = True
        self.end_point = (tip_row, tip_col)
