"""U-Net inference service.

Wraps the npec_cv pipeline behind AbstractInferenceService so the
router never import npec_cv directly.

The segmentation mask is written to MASK_OUTPUT_DIR and the filename is
returned ad segmentation_mask_url.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from api.src.schemas.predictions import (
    Plant,
    PredictionMetadata,
    PredictionResponse,
    PredictionStatus,
)
from api.src.services.base import AbstractInferenceService

logger = logging.getLogger(__name__)

# Config

# Directory where artifact models are stored
DEFAULT_ARTIFACT_DIR = Path("npec_cv/artifacts/models")
ARTIFACT_DIR = Path(os.getenv("NPEC_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))

# Directory where output masks are written
DEFAULT_MASK_OUTPUT_DIR = Path("/tmp/npec_masks")
MASK_OUTPUT_DIR = Path(os.getenv("NPEC_MASK_OUTPUT_DIR", str(DEFAULT_MASK_OUTPUT_DIR)))

SCALE_FACTOR_MM_PER_PX: float = float(os.getenv("NPEC_SCALE_FACTOR", "0.05405"))
MODEL_VERSION: str = os.getenv("NPEC_MODEL_VERSION", "unet-npec-v1.0.0")


class UNetInferenceService(AbstractInferenceService):
    """Inference service backed by the `npec_cv` U-Net pipeline."""

    def __init__(self, confidence_default: float = 0.01) -> None:
        """Inference service backed by the `npec_cv` U-Net pipeline.

        The pipeline is loaded once on first use and reused for subsequent
        requests. Heavy TesnorFlow imports are deferred to `_get_pipeline()`
        so that the module can be imported without loading TensorFlow.

        Args:
        confidence_default: Baseline root-detection threshold passed to the
            pipeline. The pipeline lowers this adaptively until every plant
            zone is detected.
        """
        self._confidence_default = confidence_default
        self._pipeline = None  # loaded lazily

    def _get_pipeline(self):
        """Return the CVPipeline, building it on first call.

        Returns:
            A configured `CVPipeline` instance.

        Raises:
            RuntimeError: If the artifact directory does not contain the expected model files.
        """
        if self._pipeline is None:
            # Deferred import - keeps TF out of the import chain until needed.
            from npec_cv import CVPipeline
            from npec_cv.configs.inference import build_inference_config
            from npec_cv.models.inference import ModelConfig

            if not ARTIFACT_DIR.exists():
                raise RuntimeError(
                    f"Artifact directory not found: {ARTIFACT_DIR}. "
                    "Set the NPEC_ARTIFACT_DIR env variable to the "
                    "directory containing the model .keras files."
                )

            root_base_cfg, anchor_cfg = build_inference_config(ARTIFACT_DIR)
            root_cfg = ModelConfig(
                checkpoint=root_base_cfg.checkpoint,
                threshold=self._confidence_default,
                patch_size=root_base_cfg.patch_size,
                batch_size=root_base_cfg.batch_size,
                custom_objects=root_base_cfg.custom_objects,
            )
            self._pipeline = CVPipeline(inference_config=(root_cfg, anchor_cfg))
            logger.info(
                "Loaded U-Net pipeline from %s (threshold=%.3f)",
                ARTIFACT_DIR,
                self._confidence_default,
            )

        return self._pipeline

    async def run(
        self,
        image_bytes: bytes,
        filename: str,
        confidence_threshold: float,
    ) -> tuple[PredictionResponse, object | None]:
        """Run U-Net inference on `image_bytes`.

        Writes the raw image to a temporary file, runs the pipeline, maps the
        `ImageResult` to a `PredictionResponse`, saves the mask, and
        returns the response.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            filename: Original filename - used for the temp file.
            confidence_threshold: Minimum per-plant confidence to include in
                results.

        Returns:
            `(PredictionResponse, image_result)` on success.

        Raises:
            HTTPException: 422 if the pipeline raises a known inference error.
            HTTPException: 500 for unexpected pipeline failures.
        """
        import tempfile

        from fastapi import HTTPException, status

        from api.src.schemas.common import ErrorCode, ErrorResponse

        start = time.perf_counter()
        prediction_id = uuid.uuid4()

        # Write bytes to a temp file, the pipeline expects a Path
        suffix = Path(filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        try:
            import asyncio

            def _run_pipeline():
                pipeline = self._get_pipeline()
                return pipeline.process_single_image(tmp_path)

            image_result = await asyncio.to_thread(_run_pipeline)
        except RuntimeError as exc:
            logger.error("Pipeline RuntimeError for '%s': $s", filename, exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=ErrorResponse(
                    error_code=ErrorCode.CORRUPT_FILE,
                    message=f"Inference failed: {exc}",
                ).model_dump(),
            )
        except Exception:
            logger.exception("Unexpected pipeline error for '%s'", filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message="An unexpected error occurred during inference.",
                ).model_dump(),
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        processing_ms = int((time.perf_counter() - start) * 1000)

        # Map ImageResult -> PredictionResponse
        plants: list[Plant] = []
        for plant_id, pr in image_result.plant_results.items():
            if not pr.detected:
                continue
            confidence = float(pr.threshold_used) if pr.threshold_used is not None else 0.0
            if confidence < confidence_threshold:
                continue

            tip_x = int(pr.end_point[1]) if pr.end_point is not None else 0
            tip_y = int(pr.end_point[0]) if pr.end_point is not None else 0
            length_px = float(pr.length) if pr.length is not None else 0.0

            plants.append(
                Plant(
                    plant_id=int(plant_id),
                    tip_x=tip_x,
                    tip_y=tip_y,
                    confidence=round(confidence, 4),
                    root_length_px=round(length_px, 2),
                    root_length_mm=round(length_px * SCALE_FACTOR_MM_PER_PX, 2),
                )
            )

        overall_confidence = round(sum(p.confidence for p in plants) / len(plants), 4) if plants else 0.0

        warnings: list[str] = []
        if image_result.num_detected < image_result.total_plants:
            warnings.append("PARTIAL_DETECTION")

        # Save segmentation mask
        mask_url: str | None = _save_mask(image_result, prediction_id)

        # Image metadata
        import io as _io

        from PIL import Image as _PIL

        with _PIL.open(_io.BytesIO(image_bytes)) as img:
            width, height = img.size
            channels = len(img.getbands())

        prediction = PredictionResponse(
            prediction_id=prediction_id,
            status=PredictionStatus.OK,
            input_filename=filename,
            metadata=PredictionMetadata(
                model_version=MODEL_VERSION,
                inference_timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_ms,
                input_resolution=[width, height],
                input_channels=channels,
                normalisation_applied=True,
                scale_factor_mm_per_px=SCALE_FACTOR_MM_PER_PX,
            ),
            segmentation_mask_url=mask_url,
            plants=plants,
            overall_confidence=overall_confidence,
            warnings=warnings,
        )

        logger.info(
            "Inference complete for '%s': %d plant(s) detected in %dms",
            filename,
            len(plants),
            processing_ms,
        )

        return prediction, image_result


# Mask saving helper


def _save_mask(image_result, prediction_id: uuid.UUID) -> str | None:
    """Write the segmentation mask to disk and return a file URL.

    Args:
        image_result: The `ImageResult` from the pipeline.
        prediction_id: UUID of the prediction.

    Returns:
        A file:// URL pointing to the saved mask, or `None` if no
        mask was available.

    Note:
        In a later sprint this will upload to Azure Blob Storage and
        return a presigned HTTPS URL instead.
    """
    try:
        import numpy as np
        from PIL import Image as _PIL

        # Collect masks from all plant results
        masks = [pr.mask for pr in image_result.plant_results.values() if pr.mask is not None]
        if not masks:
            return None

        # Combine all plant masks into a single uint8 image
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)

        MASK_OUTPUT_DIR.mkdir(parents=True, exists_ok=True)
        mask_path = MASK_OUTPUT_DIR / f"{prediction_id}_mask.png"
        _PIL.fromarray(combined).save(mask_path)

        logger.debug("Saved mask to %s", mask_path)
        return mask_path.as_uri()

    except Exception:
        logger.warning("Failed to save mask for prediction %s", prediction_id, exc_info=True)
        return None
