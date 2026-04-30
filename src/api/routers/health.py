"""Health check router.

Exposes GET /health.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.orm import ModelVersion
from db.session import get_session
from src.api.schemas.health import HealthResponse, ServiceStatus
from src.api.schemas.library import ModelStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

API_VERSION = "0.1.0"


async def _check_db(db: AsyncSession) -> bool:
    """Run a lightweight DB ping.

    Args:
        db: An open async SQLAlchemy session.

    Returns:
        `True` if the DB responded correctly, `False` on any error.
    """
    try:
        await db.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Health check - DB ping failed %s", exc)
        return False


async def _check_model(db: AsyncSession) -> bool:
    """Check whether a deployed model version exists.

    Args:
        db: An open async SQLAlchemy session.

    Returns:
        `True` if at least one `model_versions` row has
        `status = 'deployed'`, `False` otherwise.
    """
    try:
        result = await db.execute(
            select(ModelVersion.id).where(ModelVersion.status == ModelStatus.DEPLOYED.value).limit(1)
        )
        found = result.scalar_one_or_none() is not None
        if not found:
            logger.warning("Health check - no deployed model version found.")
        return found
    except Exception as exc:
        logger.warning("Health check - model version query failed: %s", exc)
        return False


# Endpoints


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness check",
    description=(
        "Returns API status, version, and UTC timestamp. "
        "Runs a DB ping and deployed model check to determine status. "
        "No authentication required."
    ),
)
async def get_health(
    db: AsyncSession = Depends(get_session),
) -> HealthResponse:
    """Return the current liveness status of the service.

    Runs two isolated dependency checks and aggregates the results:
    - Both pass -> `ok`
    - One fails -> `degraded`
    - Both fail -> `unavailable`

    Args:
        db: Injected database session.

    Returns:
        `HealthResponse`
    """
    db_ok = await _check_db(db)
    model_ok = await _check_model(db)

    passes = sum([db_ok, model_ok])
    if passes == 2:
        service_status = ServiceStatus.OK
    elif passes == 1:
        service_status = ServiceStatus.DEGRADED
    else:
        service_status = ServiceStatus.UNAVAILABLE

    return HealthResponse(
        status=service_status,
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc),
    )
