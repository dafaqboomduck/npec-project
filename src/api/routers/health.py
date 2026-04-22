"""Health check router.

Exposes GET /health.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas.health import HealthResponse, ServiceStatus

router = APIRouter(tags=["Health"])

# Endpoints


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness check",
)
async def get_health() -> HealthResponse:
    """Return the current liveness status of the service."""
    return HealthResponse(status=ServiceStatus.OK, version="0.1.0", timestamp=datetime.now(timezone.utc))
