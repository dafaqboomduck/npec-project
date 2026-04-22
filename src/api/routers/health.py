"""Health check router.

Exposes GET /health.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from api.main import API_VERSION
from api.schemas.health import HealthResponse, ServiceStatus

router = APIRouter(tags=["Health"])

# Endpoints


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness check",
)
async def get_health() -> HealthResponse:
    """Return the current liveness status of the service."""
    return HealthResponse(status=ServiceStatus.OK, version=API_VERSION, timestamp=datetime.now(timezone.utc))
