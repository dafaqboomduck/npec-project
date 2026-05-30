"""Operational and business metrics repository.

Aggregates over ``inference_audit_log`` - the per-request audit ledger
added for data lineage. This is the authoritative source for the
operational and plant-detection metrics in ticket 2561 because it is the
only table that:

* records *failed* predictions (the ``error`` column; failures still
  write a row), so the error rate is computable here and nowhere else,
  and
* stores per-call model output (``output_summary``), so plant-detection
  counts are derivable per request.

``monitoring_metrics`` is intentionally NOT used here: it stores
model-quality metrics (f1, throughput, drift) and ``MonitoringService``
only writes it on a successful prediction, so it cannot answer
error-rate or business questions.

Tables read:
- ``inference_audit_log`` - one row per inference request

Postgres-specific: uses ``percentile_cont`` (an ordered-set aggregate),
``FILTER`` clauses, ``jsonb_typeof`` and ``jsonb_array_length``. This
query does NOT run on SQLite - test it against the dev Postgres
container, not an in-memory SQLite fixture.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import Integer, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import InferenceAuditLog as AL
from api.src.schemas.dashboard import ModelMetricRow, OperationalMetricsResponse

logger = logging.getLogger(__name__)

# The key inside ``output_summary`` that holds the per-prediction plant
# count. CONFIRM against what unet_inference.py's ``set_output({...})``
# writes; if it is e.g. "roots"/"shoots"/"plant_count", change it here.
PLANT_COUNT_KEY = "total_plants"


def _plant_count_expr():
    """Type-robust per-row plant count from ``output_summary``.

    Handles the value being either a number (used directly) or a JSON
    array of detections (its length is used). Anything else - missing
    key, null, string - contributes 0. This guarantees the aggregate
    never raises at query time regardless of the exact output shape, so
    the latency/error metrics can't be taken down by a surprise plant
    payload.
    """
    elem = AL.output_summary[PLANT_COUNT_KEY]
    return case(
        (func.jsonb_typeof(elem) == "number", cast(elem.astext, Integer)),
        (func.jsonb_typeof(elem) == "array", func.jsonb_array_length(elem)),
        else_=0,
    )


class OperationalMetricsRepository:
    """Read-only operational + business aggregates over the audit log."""

    def __init__(self, db: AsyncSession) -> None:
        """Read-only aggregates over ``inference_audit_log``.

        Args:
            db: Injected database session.
        """
        self._db = db

    async def get_operational_metrics(
        self,
        since: datetime | None = None,
    ) -> OperationalMetricsResponse:
        """Return operational and plant-detection metrics.

        Operational: total / failed request counts, error rate, and
        average / p50 / p95 latency in milliseconds. Business: total
        plants detected, average plants per request, and a per-model
        breakdown of all of the above.

        Returns zero-defaults (matching the Dashboard convention) when
        the audit log is empty for the requested window.

        Args:
            since: Optional lower bound on ``logged_at``. When given,
                only requests at or after this timestamp are counted.

        Returns:
            ``OperationalMetricsResponse``
        """
        plants = _plant_count_expr()

        totals_stmt = select(
            func.count().label("total"),
            func.count().filter(AL.error.isnot(None)).label("failed"),
            func.coalesce(func.avg(AL.latency_ms), 0.0).label("avg_ms"),
            func.percentile_cont(0.5).within_group(AL.latency_ms.asc()).label("p50"),
            func.percentile_cont(0.95).within_group(AL.latency_ms.asc()).label("p95"),
            func.coalesce(func.sum(plants).filter(AL.error.is_(None)), 0).label("plants"),
        )
        if since is not None:
            totals_stmt = totals_stmt.where(AL.logged_at >= since)

        row = (await self._db.execute(totals_stmt)).one()
        total = row.total or 0

        if total == 0:
            logger.info("get_operational_metrics: audit log empty for window.")
            return OperationalMetricsResponse()

        per_model_stmt = (
            select(
                AL.model_name.label("model_name"),
                func.count().label("requests"),
                func.count().filter(AL.error.isnot(None)).label("failed"),
                func.coalesce(func.avg(AL.latency_ms), 0.0).label("avg_ms"),
                func.coalesce(func.sum(plants).filter(AL.error.is_(None)), 0).label("plants"),
            )
            .group_by(AL.model_name)
            .order_by(func.count().desc())
        )
        if since is not None:
            per_model_stmt = per_model_stmt.where(AL.logged_at >= since)

        per_model_rows = (await self._db.execute(per_model_stmt)).all()

        failed = row.failed or 0
        plants_total = int(row.plants or 0)

        return OperationalMetricsResponse(
            total_requests=total,
            failed_requests=failed,
            error_rate_pct=round(100.0 * failed / total, 2),
            avg_latency_ms=round(float(row.avg_ms or 0.0), 1),
            p50_latency_ms=round(float(row.p50 or 0.0), 1),
            p95_latency_ms=round(float(row.p95 or 0.0), 1),
            total_plants_detected=plants_total,
            avg_plants_per_request=round(plants_total / total, 2),
            per_model=[
                ModelMetricRow(
                    model_name=m.model_name or "unknown",
                    requests=m.requests,
                    error_rate_pct=(round(100.0 * (m.failed or 0) / m.requests, 2) if m.requests else 0.0),
                    avg_latency_ms=round(float(m.avg_ms or 0.0), 1),
                    total_plants_detected=int(m.plants or 0),
                )
                for m in per_model_rows
            ],
        )
