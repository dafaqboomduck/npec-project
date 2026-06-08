"""Durable inference queue backed by `inference_jobs`.

- enqueue_batch(): inserts one 'queued' row per pending annotation.
- consume_forever(): per-worker loop that claims one queued row at a time with
  FOR UPDATE SKIP LOCKED (no two workers ever grab the same row), runs inference
  through the InferenceGate, and persists the result. Survives restarts.
- queue_counts(): UX snapshot of running/queued/done/failed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.src.db.orm import Annotation, InferenceJob
from api.src.schemas.library import AnnotationStatus

logger = logging.getLogger(__name__)

_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
_POLL_INTERVAL_S = float(os.getenv("INFERENCE_QUEUE_POLL_S", "2.0"))
_MAX_ATTEMPTS = int(os.getenv("INFERENCE_QUEUE_MAX_ATTEMPTS", "3"))


async def enqueue_batch(db, dataset_id: uuid.UUID, requested_by: uuid.UUID) -> int:
    """Queue all pending annotations for a dataset. Returns the number queued.

    Idempotent: annotations already represented by a queued/running job are skipped.
    Caller wraps this in advisory_xact_lock(dataset) to avoid double-enqueue races.
    """
    pending = (
        await db.execute(
            select(Annotation.id, Annotation.image_uri).where(
                Annotation.dataset_id == dataset_id,
                Annotation.status == AnnotationStatus.PENDING.value,
            )
        )
    ).all()

    active_ann = set(
        (
            await db.execute(
                select(InferenceJob.annotation_id).where(
                    InferenceJob.dataset_id == dataset_id,
                    InferenceJob.status.in_(["queued", "running"]),
                )
            )
        )
        .scalars()
        .all()
    )

    queued = 0
    for ann_id, image_uri in pending:
        if ann_id in active_ann:
            continue
        db.add(
            InferenceJob(
                kind="batch",
                dataset_id=dataset_id,
                annotation_id=ann_id,
                image_uri=image_uri,
                status="queued",
                requested_by=requested_by,
            )
        )
        queued += 1
    await db.commit()
    if queued > 0:
        from api.src.services.broadcaster import inference_broadcaster

        await inference_broadcaster.broadcast("queue_updated")
    return queued


async def queue_counts(db) -> dict:
    """Return running/queued/done/failed counts for the UX dashboard."""
    rows = (await db.execute(select(InferenceJob.status, func.count()).group_by(InferenceJob.status))).all()
    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for status_value, n in rows:
        counts[status_value] = n
    return counts


async def reconcile_orphaned_jobs(session_factory: async_sessionmaker) -> None:
    """Reset rows left 'running' by a crashed worker back to 'queued' (or fail them)."""
    async with session_factory() as db:
        await db.execute(
            update(InferenceJob)
            .where(InferenceJob.status == "running", InferenceJob.attempts < _MAX_ATTEMPTS)
            .values(status="queued", worker_id=None, started_at=None)
        )
        await db.execute(
            update(InferenceJob)
            .where(InferenceJob.status == "running", InferenceJob.attempts >= _MAX_ATTEMPTS)
            .values(
                status="failed", finished_at=datetime.now(timezone.utc), error="exceeded max attempts after restart"
            )
        )
        await db.commit()


async def _claim_one(db) -> InferenceJob | None:
    """Atomically claim the oldest queued job.

    FOR UPDATE SKIP LOCKED guarantees no two workers claim the same row.
    """
    row = (
        await db.execute(
            select(InferenceJob)
            .where(InferenceJob.status == "queued")
            .order_by(InferenceJob.queued_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    row.status = "running"
    row.worker_id = _WORKER_ID
    row.attempts += 1
    row.started_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(row)
    return row


async def consume_forever(session_factory: async_sessionmaker, stop: asyncio.Event) -> None:
    """Per-worker consumer loop. Multiple workers may run this safely."""
    from api.src.dependencies import get_inference_service, get_storage
    from api.src.repositories.database import DatabasePredictionRepository
    from api.src.services.artifact_storage import save_segmentation_mask, save_tip_photo
    from api.src.services.inference_gate import get_inference_gate

    logger.info("Inference queue consumer started (worker=%s)", _WORKER_ID)
    storage = get_storage()
    inference_service = get_inference_service()
    gate = get_inference_gate()

    while not stop.is_set():
        async with session_factory() as db:
            job = await _claim_one(db)
        if job is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=_POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                pass
            continue

        async with session_factory() as db:
            try:
                image_bytes = await asyncio.to_thread(storage.download, job.image_uri)
                filename = (job.image_uri or "image").split("/")[-1]

                async with gate.slot():  # serialised model evaluation
                    pred_resp, image_result = await inference_service.run(
                        image_bytes=image_bytes,
                        filename=filename,
                        db=db,
                    )

                mask_uri = tip_uri = None
                if image_result is not None:
                    mask_uri = await asyncio.to_thread(
                        save_segmentation_mask,
                        prediction_id=pred_resp.prediction_id,
                        image_result=image_result,
                        backend=storage,
                    )
                    tip_uri = await asyncio.to_thread(
                        save_tip_photo,
                        prediction_id=pred_resp.prediction_id,
                        image_result=image_result,
                        original_image_bytes=image_bytes,
                        backend=storage,
                    )

                pred_resp = pred_resp.model_copy(update={"segmentation_mask_url": mask_uri})
                repo = DatabasePredictionRepository(db=db, user_id=job.requested_by, storage=storage)
                await repo.save(pred_resp, filename=filename, input_uri=job.image_uri, tip_photo_uri=tip_uri)

                ann = await db.get(Annotation, job.annotation_id)
                if ann is not None:
                    ann.prediction_id = pred_resp.prediction_id
                    ann.mask_uri = mask_uri
                    ann.tip_photo_uri = tip_uri
                    ann.status = AnnotationStatus.ANNOTATED
                    ann.updated_at = datetime.now(timezone.utc)

                j = await db.get(InferenceJob, job.id)
                j.status, j.finished_at = "done", datetime.now(timezone.utc)
                await db.commit()

                from api.src.services.broadcaster import inference_broadcaster

                await inference_broadcaster.broadcast("queue_updated")
            except Exception as exc:
                await db.rollback()
                logger.error("Inference job %s failed: %s", job.id, exc)
                async with session_factory() as err_db:
                    j = await err_db.get(InferenceJob, job.id)
                    if j is not None:
                        if j.attempts >= _MAX_ATTEMPTS:
                            j.status, j.finished_at = "failed", datetime.now(timezone.utc)
                            j.error = f"{type(exc).__name__}: {exc}"[:1000]
                            ann = await err_db.get(Annotation, j.annotation_id)
                            if ann is not None:
                                ann.status = AnnotationStatus.REVIEW_REQUIRED
                        else:
                            j.status, j.worker_id, j.started_at = "queued", None, None  # retry
                    await err_db.commit()

                from api.src.services.broadcaster import inference_broadcaster

                await inference_broadcaster.broadcast("queue_updated")
    logger.info("Inference queue consumer stopping (worker=%s)", _WORKER_ID)
