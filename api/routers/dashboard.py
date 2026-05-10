"""Dashboard router.

Three endpoints feeding the monitoring dashboard:

- GET /dashboard/metrics           - top-level KPIs (f1, throughput)
- GET /dashboard/f1-history        - validation f1 time-series for the deployed model
- GET /dashboard/data-distribution - sample counts per species
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session

from ..core.security import get_current_user
from ..repositories.dashboard import DashboardRepository
from ..schemas.auth import TokenData
from ..schemas.common import ErrorResponse
from ..schemas.dashboard import (
    DashboardMetricsResponse,
    DataDistributionResponse,
    F1HistoryResponse,
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
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> DashboardMetricsResponse:
    """Return top-level KPIs for the deployed model.

    Reads the latest `f1_validation` and `throughput_img_per_s` values
    from `monitoring_metrics`. Returns zero defaults when no deployed
    model or metrics exist yet.

    Args:
        user: Authenticated user from Bearer Token.
        db: Injected database session.

    Returns:
        `DashboardMetricsResponse`
    """
    return await DashboardRepository(db).get_metrics()


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
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> F1HistoryResponse:
    """Return the full f1 training history for the deployed model.

    Each data point is an `{epoch, f1_score}` pair read from
    `monitoring_metrics` rows with `metric_name = 'f1_validation'`.
    Returns an empty `data_points` list when no history exists.

    Args:
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `F1HistoryResponse`
    """
    return await DashboardRepository(db).get_f1_history()


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
