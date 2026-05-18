"""Health check router.

Exposes GET /health.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import ModelVersion
from api.src.db.session import get_session
from api.src.schemas.health import HealthResponse, ServiceStatus
from api.src.schemas.library import ModelStatus

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
    """Check whether a deployed model version exists and its artifact is on disk.

    Returns `degraded` (False) if:
        - No deployed row exists.
        - artifact_uri is NULL.
        - The file at artifact_uri does not exist on disk.

    Args:
        db: An open async SQLAlchemy session.

    Returns:
        `True` if both the `model_versions` row and the artifact
        file are present.
    """
    try:
        result = await db.execute(
            select(ModelVersion.artifact_uri).where(ModelVersion.status == ModelStatus.DEPLOYED.value).limit(1)
        )
        artifact_uri: str | None = result.scalar_one_or_none()

        if artifact_uri is None:
            logger.warning("Health check - no deployed model found or artifact_uri is NULL.")
            return False

        artifact_path = Path(artifact_uri)
        if not artifact_path.exists():
            logger.warning(
                "Health check - artifact file missing at '%s'. "
                "Place the .keras file there or deploy a model via "
                "POST /models/{id}/deploy.",
                artifact_path,
            )
            return False

        return True
    except Exception as exc:
        logger.warning("Health check - model artifact check failed: %s", exc)
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
