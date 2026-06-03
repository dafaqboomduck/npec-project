"""Per-prediction audit logger backed by ``inference_audit_log``.

Wraps the actual prediction call. One row per call. Survives deletion
of the source prediction (FK is SET NULL). Designed to be used inside
the inference service alongside the existing ``predictions`` table
write, so one user request produces one prediction row and one audit
row in the same transaction.

Usage in the inference service::

    audit = InferenceLogger(
        db,
        model_version_id=mv.id,
        model_name="root-detection",
        model_version_label=mv.version,
    )
    async with audit.measure(request_id=req_id, input_ref=input_ref) as rec:
        prediction = await model.predict(...)
        rec.set_output({"plants": len(prediction.plants)})
        rec.set_prediction_id(prediction.id)

A failed prediction still produces an audit row, with ``error`` set
and ``output_summary`` None. The audit row's hash format ``"sha256:..."``
is compatible with the ``predictions.input_hash`` column added in
migration 0006 — the two values can be reconciled in queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from lineage import resource_sampler

log = logging.getLogger(__name__)


# Deployment label written into every audit row. Set NPEC_DEPLOYMENT at
# container start to one of: 'local', 'portainer', 'azureml'. Defaults
# to 'unknown' for safety.
ENV_DEPLOYMENT = "NPEC_DEPLOYMENT"
VALID_DEPLOYMENTS = ("local", "portainer", "azureml", "unknown")


class InferenceLogger:
    """Audit logger that writes one row per inference call."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        model_version_id: Optional[uuid.UUID],
        model_name: str,
        model_version_label: str,
        deployment: Optional[str] = None,
        endpoint: str = "api",
    ) -> None:
        """Construct an audit logger bound to a request-scoped DB session."""
        self._db = db
        self._model_version_id = model_version_id
        self._model_name = model_name
        self._model_version_label = model_version_label
        self._endpoint = endpoint
        self._deployment = deployment or os.environ.get(ENV_DEPLOYMENT, "unknown")
        if self._deployment not in VALID_DEPLOYMENTS:
            log.warning(
                "Unknown deployment label %r; coercing to 'unknown'.",
                self._deployment,
            )
            self._deployment = "unknown"
        if model_version_id is None:
            log.warning(
                "InferenceLogger constructed without model_version_id; " "audit rows will not join to model_versions."
            )

    @asynccontextmanager
    async def measure(
        self,
        *,
        request_id: Optional[str] = None,
        input_ref: str,
    ) -> AsyncIterator["_Record"]:
        """Async context manager that wraps a prediction call.

        On entry, opens a record. On exit (normal or exceptional), adds
        one row to the session. The write uses ``db.add()`` without
        committing; the caller's transaction handles commit so the audit
        row lives or dies with the prediction row.
        """
        rec = _Record(
            request_id=request_id or _new_request_id(),
            input_ref=input_ref,
        )
        start = time.perf_counter()
        rec.cpu_start = resource_sampler.process_cpu_seconds()
        try:
            yield rec
        except Exception as exc:
            rec.set_error(exc)
            await self._persist(rec, start)
            raise
        else:
            await self._persist(rec, start)

    async def log_prediction(
        self,
        *,
        request_id: Optional[str],
        input_ref: str,
        output: Any,
        prediction_id: Optional[uuid.UUID] = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Emit one audit row without using the context manager."""
        rec = _Record(
            request_id=request_id or _new_request_id(),
            input_ref=input_ref,
        )
        rec.set_output(output)
        if prediction_id is not None:
            rec.set_prediction_id(prediction_id)
        if error is not None:
            rec.error = error
        rec.latency_ms = latency_ms
        await self._persist(rec, start=None)

    @staticmethod
    def hash_input(payload: bytes | str) -> str:
        """Return a sha256 hex reference for use as ``input_ref``.

        The output format matches ``predictions.input_hash`` (sha256 hex)
        and can be compared by stripping the ``sha256:`` prefix.
        """
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        return "sha256:" + hashlib.sha256(data).hexdigest()

    async def _persist(self, rec: "_Record", start: Optional[float]) -> None:
        from api.src.db.orm import InferenceAuditLog  # late import

        if rec.latency_ms is None and start is not None:
            rec.latency_ms = int((time.perf_counter() - start) * 1000)

        # Resource snapshot — fully guarded, must never break the write.
        res = resource_sampler.sample_system()
        cpu_pct_process = resource_sampler.process_cpu_pct(
            rec.cpu_start,
            resource_sampler.process_cpu_seconds(),
            rec.latency_ms,
        )

        row = InferenceAuditLog(
            prediction_id=rec.prediction_id,
            model_version_id=self._model_version_id,
            model_name=self._model_name,
            model_version_label=self._model_version_label,
            request_id=rec.request_id,
            input_ref=rec.input_ref,
            output_summary=rec.output_summary,
            latency_ms=rec.latency_ms,
            error=rec.error,
            endpoint=self._endpoint,
            deployment=self._deployment,
            cpu_pct_system=res["cpu_pct_system"],
            cpu_pct_process=cpu_pct_process,
            cpu_count=res["cpu_count"],
            ram_used_mb=res["ram_used_mb"],
            ram_available_mb=res["ram_available_mb"],
            gpu_pct=res["gpu_pct"],
            vram_used_mb=res["vram_used_mb"],
            disk_free_mb=res["disk_free_mb"],
        )
        self._db.add(row)
        try:
            await self._db.flush()
        except Exception:
            log.exception("Failed to flush audit row for request %s", rec.request_id)


class _Record:
    """Mutable per-call state filled in inside the measure() block."""

    __slots__ = (
        "request_id",
        "input_ref",
        "output_summary",
        "latency_ms",
        "error",
        "prediction_id",
        "cpu_start",
    )

    def __init__(self, *, request_id: str, input_ref: str) -> None:
        self.request_id = request_id
        self.input_ref = input_ref
        self.output_summary: Optional[dict] = None
        self.latency_ms: Optional[int] = None
        self.error: Optional[str] = None
        self.prediction_id: Optional[uuid.UUID] = None
        self.cpu_start: Optional[float] = None

    def set_output(self, output: Any) -> None:
        self.output_summary = _summarize(output)

    def set_error(self, exc: BaseException) -> None:
        self.error = f"{type(exc).__name__}: {exc}"

    def set_prediction_id(self, prediction_id: uuid.UUID) -> None:
        self.prediction_id = prediction_id


def _summarize(output: Any, max_len: int = 2000) -> Optional[dict]:
    """Compress arbitrary prediction output into a JSONB-storable summary."""
    if output is None:
        return None
    if isinstance(output, dict):
        try:
            json.dumps(output, default=str)
            return output
        except (TypeError, ValueError):
            return {"repr": str(output)[:max_len]}
    if isinstance(output, (str, int, float, bool)):
        return {"value": output}
    try:
        serialized = json.dumps(output, default=str)
    except (TypeError, ValueError):
        serialized = str(output)
    if len(serialized) > max_len:
        serialized = serialized[:max_len] + "...(truncated)"
    return {"repr": serialized}


def _new_request_id() -> str:
    return str(uuid.uuid4())
