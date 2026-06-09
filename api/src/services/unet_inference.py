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

import asyncio
import concurrent.futures
import hashlib
import logging
import multiprocessing as mp
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

# GPU memory release
# ------------------
# TensorFlow never returns GPU memory to the driver for the life of a process
# (set_memory_growth only defers allocation; it is not a cap and never frees).
# So the only reliable way to release VRAM is to run the pipeline in a separate
# process and let process exit reclaim it.
#
# When _ISOLATE_INFERENCE is true (default), inference runs in a single warm
# worker process that keeps the models loaded across requests. A background
# reaper tears the worker down after _IDLE_TIMEOUT_SECONDS with no inference,
# releasing its GPU memory; the next request spawns a fresh worker and reloads.
_ISOLATE_INFERENCE: bool = os.getenv("NPEC_ISOLATE_INFERENCE", "true").lower() in ("1", "true", "yes")
_IDLE_TIMEOUT_SECONDS: float = max(0.0, float(os.getenv("NPEC_INFERENCE_IDLE_TIMEOUT_MINUTES", "5")) * 60.0)

# Default root-model detection threshold used when building the pipeline.
_ROOT_THRESHOLD: float = 0.01  # default

# Per-worker-process cache: the loaded pipeline persists for the life of the
# worker so models stay warm across requests. It dies with the process when the
# worker is recycled on idle, which is what frees the GPU memory.
_WORKER_PIPELINE = None
_WORKER_KEY: tuple | None = None


def _worker_get_pipeline(artifact_path_str: str, root_threshold: float):
    """Return the worker's cached pipeline, building it once per process.

    Runs inside the spawned worker process. The pipeline (and the GPU memory
    its TensorFlow/CUDA context holds) lives until the worker process exits.

    Args:
        artifact_path_str: Path to the deployed artifact dir (or a file inside it).
        root_threshold: Root-model detection threshold.

    Returns:
        The cached ``CVPipeline`` for this worker.
    """
    global _WORKER_PIPELINE, _WORKER_KEY

    key = (artifact_path_str, root_threshold)
    if _WORKER_PIPELINE is None or _WORKER_KEY != key:
        from npec_cv import CVPipeline
        from npec_cv.configs.inference import build_inference_config
        from npec_cv.models.inference import ModelConfig

        artifact_path = Path(artifact_path_str)
        artifact_dir = artifact_path if artifact_path.is_dir() else artifact_path.parent

        root_base_cfg, anchor_cfg = build_inference_config(artifact_dir)
        root_cfg = ModelConfig(
            checkpoint=root_base_cfg.checkpoint,
            threshold=root_threshold,
            patch_size=root_base_cfg.patch_size,
            batch_size=root_base_cfg.batch_size,
            custom_objects=root_base_cfg.custom_objects,
        )
        _WORKER_PIPELINE = CVPipeline(inference_config=(root_cfg, anchor_cfg))
        _WORKER_KEY = key

    return _WORKER_PIPELINE


def _warmup_worker(artifact_path_str: str, root_threshold: float) -> bool:
    """Pre-load the pipeline in the worker process (no inference).

    Must stay at module level so it is importable/picklable under 'spawn'.
    """
    _worker_get_pipeline(artifact_path_str, root_threshold)
    return True


def _pipeline_worker(artifact_path_str: str, image_path_str: str, root_threshold: float):
    """Run inference on one image in the (warm) worker process.

    Reuses the worker's cached pipeline so the models are loaded only on the
    first call after the worker spawns. Returns the pipeline's ``ImageResult``
    (a plain dataclass of numpy arrays), which is pickled back to the parent.

    Must stay at module level so it is importable/picklable under the 'spawn'
    start method.

    Args:
        artifact_path_str: Path to the deployed artifact dir (or a file inside it).
        image_path_str: Path to the image to run inference on.
        root_threshold: Root-model detection threshold.

    Returns:
        The ``ImageResult`` produced by the pipeline.
    """
    pipeline = _worker_get_pipeline(artifact_path_str, root_threshold)
    return pipeline.process_single_image(Path(image_path_str))


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
        self._pipeline = None  # loaded lazily (non-isolated path only)
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        # Monotonic timestamp of the last inference; drives idle-based release.
        self._last_activity = 0.0
        self._reaper_task: asyncio.Task | None = None
        # Serialise inference: one GPU, one warm worker — concurrent runs would
        # race the idle-reaper and risk GPU OOM.
        self._inference_lock = asyncio.Lock()

    def set_artifact_path(self, path: Path) -> None:
        """Update the artifact path and mark the pipeline for reload.

        Called by LocalDeploymentService after copying a new artifact.

        Args:
            path: Path to the directory containing the new .keras files.
        """
        self._artifact_path = path
        self._pipeline = None
        # Recycle any warm worker so the next request loads the new artifact.
        self._shutdown_executor()
        logger.info("UNetInferenceService: artifact path updated to %s", path)

    def _resolve_artifact_path(self) -> Path:
        """Return the artifact path, raising if it is unset or missing.

        Returns:
            The configured artifact path.

        Raises:
            RuntimeError: If no artifact path is set or it does not exist.
        """
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
        return self._artifact_path

    def _get_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        """Return the inference process pool, creating it on first use.

        A single-slot pool using the 'spawn' start method holding one warm
        worker that keeps the models loaded across requests. The worker is torn
        down only by the idle reaper (or on reload/shutdown), which destroys its
        TensorFlow/CUDA context and frees its GPU memory.

        Returns:
            The process pool executor.
        """
        if self._executor is None:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context("spawn"),
            )
            logger.info(
                "Inference worker started (spawn); GPU memory released after %.0f min idle.",
                _IDLE_TIMEOUT_SECONDS / 60.0,
            )
        return self._executor

    def _shutdown_executor(self) -> None:
        """Tear down the inference worker if one exists (frees its GPU memory).

        Uses wait=True so the OS process is fully gone (and its CUDA context
        released) before this method returns. The idle reaper only fires when no
        futures are in-flight, so there is nothing to wait on except the process
        exit itself — a few hundred milliseconds at most, which is negligible
        compared to the cold-start model reload the next request will do anyway.
        Crucially, wait=False would release the lock before the old process exits,
        letting a new worker spawn while the old one still holds the GPU context.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _ensure_reaper(self) -> None:
        """Start the idle reaper task if it is not already running."""
        if _IDLE_TIMEOUT_SECONDS <= 0:
            return  # 0 disables idle release — keep the worker warm indefinitely.
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._idle_reaper())

    async def _idle_reaper(self) -> None:
        """Tear down the warm worker once inference has been idle long enough.

        Runs for the life of the service. Each tick it checks, under the
        inference lock (so it can never kill a worker mid-request), whether the
        worker has been idle for ``_IDLE_TIMEOUT_SECONDS`` and, if so, releases
        its GPU memory. The next request transparently spawns a fresh worker.
        """
        check_interval = min(_IDLE_TIMEOUT_SECONDS, 60.0)
        while True:
            await asyncio.sleep(check_interval)
            async with self._inference_lock:
                if self._executor is None:
                    continue
                idle = time.monotonic() - self._last_activity
                if idle >= _IDLE_TIMEOUT_SECONDS:
                    logger.info(
                        "Releasing GPU: inference idle for %.0fs (>= %.0fs).",
                        idle,
                        _IDLE_TIMEOUT_SECONDS,
                    )
                    self._shutdown_executor()

    async def _run_isolated(self, artifact_path: Path, tmp_path: Path):
        """Run one inference in the warm worker process.

        Holds ``_inference_lock`` so a request can never race the idle reaper.
        Refreshes the activity timestamp so the worker is kept warm while
        traffic continues and only released after a quiet period.
        """
        async with self._inference_lock:
            self._ensure_reaper()
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    self._get_executor(),
                    _pipeline_worker,
                    str(artifact_path),
                    str(tmp_path),
                    _ROOT_THRESHOLD,
                )
            finally:
                self._last_activity = time.monotonic()

    async def warmup(self) -> None:
        """Pre-load the models at startup, off the request path.

        With isolation on, the models are preloaded inside the warm worker
        process (so they can still be released after the idle timeout). With
        isolation off, the in-process pipeline is built instead. A missing
        artifact is logged and skipped rather than raised, so a server with no
        model deployed yet still starts.
        """
        try:
            artifact_path = self._resolve_artifact_path()
        except RuntimeError as exc:
            logger.warning("Inference warmup skipped — %s", exc)
            return

        if not _ISOLATE_INFERENCE:
            await asyncio.to_thread(self._get_pipeline)
            logger.info("Inference pipeline warmed up (in-process).")
            return

        async with self._inference_lock:
            self._ensure_reaper()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._get_executor(),
                _warmup_worker,
                str(artifact_path),
                _ROOT_THRESHOLD,
            )
            self._last_activity = time.monotonic()
        logger.info("Inference worker warmed up (models preloaded).")

    async def aclose(self) -> None:
        """Cancel the reaper and tear down the worker. Called at app shutdown."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        self._shutdown_executor()

    def _get_pipeline(self):
        """Return the CVPipeline, building it on first call.

        Returns:
            A configured `CVPipeline` instance.

        Raises:
            RuntimeError: If the artifact directory does not contain the expected model files.
        """
        if self._pipeline is None:
            artifact_path = self._resolve_artifact_path()

            # Deferred import - keeps TF out of the import chain until needed.
            from npec_cv import CVPipeline
            from npec_cv.configs.inference import build_inference_config
            from npec_cv.models.inference import ModelConfig

            artifact_dir = artifact_path if artifact_path.is_dir() else artifact_path.parent

            root_base_cfg, anchor_cfg = build_inference_config(artifact_dir)
            root_cfg = ModelConfig(
                checkpoint=root_base_cfg.checkpoint,
                threshold=_ROOT_THRESHOLD,
                patch_size=root_base_cfg.patch_size,
                batch_size=root_base_cfg.batch_size,
                custom_objects=root_base_cfg.custom_objects,
            )
            self._pipeline = CVPipeline(inference_config=(root_cfg, anchor_cfg))
            logger.info("Loaded U-Net pipeline from %s", artifact_path)

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
        input_ref = f"sha256: {hashlib.sha256(image_bytes).hexdigest()}"

        audit = None
        if db is not None and _DEPLOYMENT_ENV == "cloud":
            from lineage import InferenceLogger

            audit = InferenceLogger(
                db,
                model_version_id=mv_id,
                model_name=mv_name,
                model_version_label=mv_label,
            )

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
            input_url=None,
            tip_photo_url=None,
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
        # Recycle any warm worker so the next request loads the new artifact.
        self._shutdown_executor()
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
        """Execute the pipeline and translate errors.

        When ``_ISOLATE_INFERENCE`` is on (default), the pipeline runs in a warm
        worker process whose GPU memory is released once inference has been idle
        for ``_IDLE_TIMEOUT_SECONDS``. Otherwise it runs in a worker thread
        against the in-process pipeline.

        Centralises the 422 / 500 translation that was previously inline
        in ``run()`` so it can be invoked from both the audit-logged and
        un-audited code paths without duplicating the try/except.
        """
        from fastapi import HTTPException, status

        from api.src.schemas.common import ErrorCode, ErrorResponse

        try:
            if _ISOLATE_INFERENCE:
                # Validate eagerly in the parent so a missing artifact maps to a
                # clean error instead of a worker crash, then hand off to a
                # recyclable child process that owns the GPU context.
                artifact_path = self._resolve_artifact_path()
                return await self._run_isolated(artifact_path, tmp_path)

            def _run():
                pipeline = self._get_pipeline()
                return pipeline.process_single_image(tmp_path)

            return await asyncio.to_thread(_run)
        except concurrent.futures.process.BrokenProcessPool:
            # The worker died mid-task (e.g. GPU OOM / segfault). The pool is
            # now unusable — drop it so the next request spawns a fresh one.
            logger.exception("Inference worker process died (possible GPU OOM) for '%s'", filename)
            self._shutdown_executor()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message="An unexpected error occurred during inference.",
                ).model_dump(),
            )
        except TypeError:
            # A TypeError propagated out of the worker (e.g. npec_cv's
            # restore_mask receiving None from a failed GPU allocation).  Treat
            # the worker as corrupt — drop it so the next request gets a fresh
            # process with a clean CUDA context.
            logger.exception(
                "Inference worker returned unexpected None (possible GPU init failure) for '%s'",
                filename,
            )
            self._shutdown_executor()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message="An unexpected error occurred during inference.",
                ).model_dump(),
            )
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
