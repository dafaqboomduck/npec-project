"""Overview router.

Four endpoints feeding the application's Overview/Home page:

- GET /overview/status        - lab summary statistics (authenticated)
- GET /overview/activity      - paginated activity feed (authenticated)
- GET /overview/model-health  - deployed model health (authenticated)
- GET /overview/environment   - system stats (admin only)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from src.api.core.security import get_current_user, require_admin
from src.api.repositories.overview import OverviewRepository
from src.api.schemas.auth import TokenData
from src.api.schemas.common import ErrorResponse
from src.api.schemas.overview import (
    ActivityFeedResponse,
    EnvironmentResponse,
    ModelHealthResponse,
    OverviewStatusResponse,
)
from src.api.services.environment import get_environment_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/overview", tags=["Overview"])


@router.get(
    "/status",
    response_model=OverviewStatusResponse,
    summary="Lab summary statistics",
    responses={401: {"model": ErrorResponse}},
)
async def get_status(
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> OverviewStatusResponse:
    """Return high-level lab summary statistics.

    Queries `predictions` and `model_versions` to compute counts.

    Args:
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `OverviewStatusResponse`
    """
    return await OverviewRepository(db).get_status()


@router.get(
    "/activity",
    response_model=ActivityFeedResponse,
    summary="Recent activity feed",
    responses={401: {"model": ErrorResponse}},
)
async def get_activity(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Events per page"),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> ActivityFeedResponse:
    """Return a paginated feed of recent lab events.

    Aggregates uploads, deployments, and annotations
    into a unified list sorted by timestamp descending.

    Args:
        page: 1-indexed page number.
        page_size: Events per page (max 100).
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `ActivityFeedResponse`
    """
    return await OverviewRepository(db).get_activity(page=page, page_size=page_size)


@router.get(
    "/model-health",
    response_model=ModelHealthResponse,
    summary="Active model healt summary",
    responses={401: {"model": ErrorResponse}},
)
async def get_model_health(
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> ModelHealthResponse:
    """Return health metrics for the currently deployed model.

    Reads f1_score from model_versions and the latest latency and drift
    scores from monitoring_metrics. Returns defaults (0.0 / None) when
    no metrics exist yet rather than erroring.

    Args:
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `ModelHealthResponse`
    """
    return await OverviewRepository(db).get_model_health()


@router.get(
    "/environment",
    response_model=EnvironmentResponse,
    summary="System environment stats (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_environment(
    user: Annotated[TokenData, Depends(require_admin)] = None,
) -> EnvironmentResponse:
    """Return GPU temperature, memory usage, and system status.

    Admin only. GPU temperature returns `null` if no NVIDIA GPU
    is present or `pynvml` is unavailable.

    `system_status` is derived from memory percent:
    - Below 80%  -> `optimal`
    - 80-95%     -> `degraded`
    - Above 95%  -> `critical`

    Args:
        user: Admin-role authenticated user.

    Returns:
        `EnvironmentResponse`
    """
    return get_environment_stats()
