"""U-Net inference service.

Wraps the npec_cv pipeline behind AbstractInferenceService so the
router never imports npec_cv directly.

The active model artifact path is passed in at construction time by
LocalDeploymentService (via dependencies.py), so swapping the deployment
model only requires calling reload() on the singleton.

Audit logging
-------------
When ``run()`` is called with a DB session, one row is appended to the
``inference_audit_log`` table per request (added in migration 0010).
The audit row is added to the same session as the prediction so both
commit or roll back together. If the session is not passed (e.g. by
tests using a mock service), audit logging is skipped with a warning.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import ModelVersion
from api.src.schemas.predictions import (
    Plant,
    PredictionMetadata,
    PredictionResponse,
    PredictionStatus,
)
from api.src.services.base import AbstractInferenceService

logger = logging.getLogger(__name__)

# Config

_DEPLOYMENT_ENV: str = os.getenv("DEPLOYMENT_ENV", "local")

# Directory where output masks are written
DEFAULT_MASK_OUTPUT_DIR = Path("/tmp/npec_masks")
MASK_OUTPUT_DIR = Path(os.getenv("NPEC_MASK_OUTPUT_DIR", str(DEFAULT_MASK_OUTPUT_DIR)))

SCALE_FACTOR_MM_PER_PX: float = float(os.getenv("NPEC_SCALE_FACTOR", "0.05405"))

# Fallback model name/version when no deployed ModelVersion row exists.
# Used only for the audit row; the actual model artifact is whatever
# LocalDeploymentService loaded at startup.
_FALLBACK_MODEL_NAME = "root-detection"


class UNetInferenceService(AbstractInferenceService):
    """Inference service backed by the `npec_cv` U-Net pipeline."""

    def __init__(self, artifact_path: Path | None = None) -> None:
        """Inference service backed by the `npec_cv` U-Net pipeline.

        The pipeline is loaded once on first use and reused for subsequent
        requests. Heavy TensorFlow imports are deferred to `_get_pipeline()`
        so that the module can be imported without loading TensorFlow.

        Args:
            artifact_path: Path to the directory containing the .keras files.
                When None, the path is resolved lazily from the deployed
                model_versions row on first call to _get_pipeline().
        """
        self._artifact_path = artifact_path
        self._pipeline = None  # loaded lazily

    def set_artifact_path(self, path: Path) -> None:
        """Update the artifact path and mark the pipeline for reload.

        Called by LocalDeploymentService after copying a new artifact.

        Args:
            path: Path to the directory containing the new .keras files.
        """
        self._artifact_path = path
        self._pipeline = None
        logger.info("UNetInferenceService: artifact path updated to %s", path)

    def _get_pipeline(self):
        """Return the CVPipeline, building it on first call.

        Returns:
            A configured `CVPipeline` instance.

        Raises:
            RuntimeError: If the artifact directory does not contain the expected model files.
        """
        if self._pipeline is None:
            if self._artifact_path is None:
                raise RuntimeError(
                    "UNetInferenceService has no artifact_path set. "
                    "Ensure LocalDeploymentService.deploy() has been called "
                    "or NPEC_ARTIFACT_DIR is set in the environment."
                )

            if not self._artifact_path.exists():
                raise RuntimeError(
                    f"Artifact directory not found: {self._artifact_path}. "
                    "Place the .keras model files there or deploy a model "
                    "via POST /models/{id}/deploy."
                )

            # Deferred import - keeps TF out of the import chain until needed.
            from npec_cv import CVPipeline
            from npec_cv.configs.inference import build_inference_config
            from npec_cv.models.inference import ModelConfig

            artifact_dir = self._artifact_path if self._artifact_path.is_dir() else self._artifact_path.parent

            root_base_cfg, anchor_cfg = build_inference_config(artifact_dir)
            root_cfg = ModelConfig(
                checkpoint=root_base_cfg.checkpoint,
                threshold=0.01,  # default
                patch_size=root_base_cfg.patch_size,
                batch_size=root_base_cfg.batch_size,
                custom_objects=root_base_cfg.custom_objects,
            )
            self._pipeline = CVPipeline(inference_config=(root_cfg, anchor_cfg))
            logger.info("Loaded U-Net pipeline from %s", self._artifact_path)

        return self._pipeline

    async def run(
        self,
        image_bytes: bytes,
        filename: str,
        *,
        db: Optional[AsyncSession] = None,
    ) -> tuple[PredictionResponse, object | None]:
        """Run U-Net inference on `image_bytes`.

        Writes the raw image to a temporary file, runs the pipeline, maps the
        `ImageResult` to a `PredictionResponse`, saves the mask, and
        returns the response.

        When ``db`` is provided, one ``inference_audit_log`` row is added
        to the session (not committed — the caller's transaction handles
        commit). Failed predictions still produce an audit row with
        ``error`` populated.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            filename: Original filename - used for the temp file.
            db: Optional async DB session for audit logging. Passing None
                disables audit logging (useful for tests).

        Returns:
            `(PredictionResponse, image_result)` on success.

        Raises:
            HTTPException: 422 if the pipeline raises a known inference error.
            HTTPException: 500 for unexpected pipeline failures.
        """
        import tempfile

        from fastapi import HTTPException

        start = time.perf_counter()
        prediction_id = uuid.uuid4()

        # Resolve the deployed ModelVersion (for audit linkage + the
        # PredictionResponse's model_version field).
        mv_row = await self._get_deployed_model_version(db)
        if mv_row is not None:
            mv_id = mv_row.id
            mv_label = mv_row.version
            mv_name = mv_row.name
        else:
            mv_id = None
            mv_label = os.getenv("NPEC_MODEL_VERSION", "unet-npec-v1.0.0")
            mv_name = _FALLBACK_MODEL_NAME

        # Build the audit logger (no-op if db is None).
        audit = None
        if db is not None and _DEPLOYMENT_ENV == "cloud":
            from lineage import InferenceLogger

            audit = InferenceLogger(
                db,
                model_version_id=mv_id,
                model_name=mv_name,
                model_version_label=mv_label,
            )
        input_ref = InferenceLogger.hash_input(image_bytes)

        # Write bytes to a temp file, the pipeline expects a Path
        suffix = Path(filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        # Run inference under the audit logger when one is available.
        # The context manager produces exactly one audit row whether the
        # block succeeds or raises.
        try:
            if audit is not None:
                async with audit.measure(input_ref=input_ref) as audit_rec:
                    audit_rec.set_prediction_id(prediction_id)
                    image_result = await self._run_pipeline_or_raise(tmp_path, filename)
                    # Quick output summary; full plant list goes on the
                    # prediction row, not the audit row.
                    audit_rec.set_output(
                        {
                            "num_detected": int(image_result.num_detected),
                            "total_plants": int(image_result.total_plants),
                        }
                    )
            else:
                image_result = await self._run_pipeline_or_raise(tmp_path, filename)
        except HTTPException:
            # _run_pipeline_or_raise already mapped the error to HTTP.
            # The audit_rec context manager has already persisted a row
            # with error populated (if audit was active).
            raise
        finally:
            tmp_path.unlink(missing_ok=True)

        processing_ms = int((time.perf_counter() - start) * 1000)

        # Map ImageResult -> PredictionResponse
        plants: list[Plant] = []
        for plant_id, pr in image_result.plant_results.items():
            if not pr.detected:
                continue

            tip_x = int(pr.end_point[1]) if pr.end_point is not None else 0
            tip_y = int(pr.end_point[0]) if pr.end_point is not None else 0
            length_px = float(pr.length) if pr.length is not None else 0.0

            plants.append(
                Plant(
                    plant_id=int(plant_id),
                    tip_x=tip_x,
                    tip_y=tip_y,
                    root_length_px=round(length_px, 2),
                    root_length_mm=round(length_px * SCALE_FACTOR_MM_PER_PX, 2),
                )
            )

        warnings: list[str] = []
        if image_result.num_detected < image_result.total_plants:
            warnings.append("PARTIAL_DETECTION")

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
                model_version=mv_label,
                inference_timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_ms,
                input_resolution=[width, height],
                input_channels=channels,
                normalisation_applied=True,
                scale_factor_mm_per_px=SCALE_FACTOR_MM_PER_PX,
            ),
            segmentation_mask_url=None,
            plants=plants,
            warnings=warnings,
        )

        logger.info(
            "Inference complete for '%s': %d plant(s) detected in %dms (model=%s:%s)",
            filename,
            len(plants),
            processing_ms,
            mv_name,
            mv_label,
        )

        return prediction, image_result

    async def reload(self) -> None:
        """Mark the pipeline for reload on the next request.

        Sets _pipeline to None so _get_pipeline() re-loads weights from
        _artifact_path on the next POST /predict call. Called by
        LocalDeploymentService after a new model is deployed.
        """
        self._pipeline = None
        logger.info("UNetInferenceService: pipeline marked for reload on next request.")

    # ── internal helpers ─────────────────────────────────────────────

    async def _get_deployed_model_version(self, db: Optional[AsyncSession]) -> Optional[ModelVersion]:
        """Return the currently deployed ModelVersion row, or None.

        Returns None when:
          - ``db`` is None (no session available, e.g. tests), or
          - No row has status='deployed' in this database.
        """
        if db is None:
            return None
        try:
            result = await db.execute(select(ModelVersion).where(ModelVersion.status == "deployed").limit(1))
            return result.scalar_one_or_none()
        except Exception:
            logger.exception("Failed to look up deployed ModelVersion")
            return None

    async def _run_pipeline_or_raise(self, tmp_path: Path, filename: str):
        """Execute the pipeline in a worker thread and translate errors.

        Centralises the 422 / 500 translation that was previously inline
        in ``run()`` so it can be invoked from both the audit-logged and
        un-audited code paths without duplicating the try/except.
        """
        import asyncio

        from fastapi import HTTPException, status

        from api.src.schemas.common import ErrorCode, ErrorResponse

        try:

            def _run():
                pipeline = self._get_pipeline()
                return pipeline.process_single_image(tmp_path)

            return await asyncio.to_thread(_run)
        except RuntimeError as exc:
            logger.error("Pipeline RuntimeError for '%s': %s", filename, exc)
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
