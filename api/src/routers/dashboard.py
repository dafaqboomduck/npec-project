"""Dashboard router.

Four endpoints feeding the monitoring dashboard:

- GET /dashboard/metrics              - top-level KPIs (f1, throughput)
- GET /dashboard/f1-history           - validation f1 time-series for the deployed model
- GET /dashboard/data-distribution    - sample counts per species
- GET /dashboard/operational-metrics  - request latency, error rate, plant counts
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.core.security import get_current_user
from api.src.db.session import get_session
from api.src.repositories.dashboard import DashboardRepository
from api.src.repositories.operational import OperationalMetricsRepository
from api.src.schemas.auth import TokenData
from api.src.schemas.common import ErrorResponse
from api.src.schemas.dashboard import (
    DashboardMetricsResponse,
    DataDistributionResponse,
    F1HistoryResponse,
    OperationalMetricsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get(
    "/metrics",
    response_model=DashboardMetricsResponse,
    summary="Top-level KPIs",
    responses={401: {"model": ErrorResponse}},
)
async def get_dashboard_metrics(
    run_id: UUID | None = None,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> DashboardMetricsResponse:
    """Return top-level KPIs for the deployed model.

    Reads the latest `f1_validation` and `throughput_img_per_s` values
    from `monitoring_metrics`. Returns zero defaults when no deployed
    model or metrics exist yet.

    Args:
        run_id: Optional UUID of the run to filter on.
        user: Authenticated user from Bearer Token.
        db: Injected database session.

    Returns:
        `DashboardMetricsResponse`
    """
    return await DashboardRepository(db).get_metrics(run_id=run_id)


@router.get(
    "/f1-history",
    response_model=F1HistoryResponse,
    summary="Validation f1 time-series",
    description=(
        "Time-series of validation f1 score across training epochs for the "
        "deployed model. Data points are ordered by epoch ascending so the "
        "frontend can plot them directly."
    ),
    responses={401: {"model": ErrorResponse}},
)
async def get_f1_history(
    run_id: UUID | None = None,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> F1HistoryResponse:
    """Return the full f1 training history for the run with run_id.

    If run_id is not provided, returns the f1 history of the
    currently deployed model.

    Each data point is an `{epoch, f1_score}` pair read from
    `monitoring_metrics` rows with `metric_name = 'f1_validation'`.
    Returns an empty `data_points` list when no history exists.

    Args:
        run_id: UUID of the run
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `F1HistoryResponse`
    """
    return await DashboardRepository(db).get_f1_history(run_id=run_id)


@router.get(
    "/data-distribution",
    response_model=DataDistributionResponse,
    summary="Sample counts per species",
    responses={401: {"model": ErrorResponse}},
)
async def get_data_distribution(
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> DataDistributionResponse:
    """Return confirmed annotation counts grouped by species.

    Aggregation is performed at the DB level via GROUP BY.  Species with
    zero confirmed samples are excluded.

    Args:
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `DataDistributionResponse`
    """
    return await DashboardRepository(db).get_data_distribution()


@router.get(
    "/operational-metrics",
    response_model=OperationalMetricsResponse,
    summary="Operational and plant-detection metrics",
    description=(
        "Per-request operational metrics (total/failed requests, error rate, "
        "average / p50 / p95 latency) and business/plant metrics (total plants "
        "detected, average per request), with a per-model breakdown. All values "
        "are aggregated from `inference_audit_log`. Pass `since` (ISO-8601) to "
        "restrict to a time window; omit it for all-time totals."
    ),
    responses={401: {"model": ErrorResponse}},
)
async def get_operational_metrics(
    since: datetime | None = None,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> OperationalMetricsResponse:
    """Return request-level operational metrics and plant-detection totals.

    Operational: total/failed requests, error rate, avg/p50/p95 latency.
    Business: total plants detected and average per request, plus a
    per-model breakdown. All derived from `inference_audit_log`, which is
    the only table that records failed predictions and per-call output.

    Args:
        since: Optional ISO-8601 lower bound on the request timestamp.
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `OperationalMetricsResponse`
    """
    return await OperationalMetricsRepository(db).get_operational_metrics(since=since)
