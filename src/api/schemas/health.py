"""Service liveness -related Pydantic schemas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

# Enums


class ServiceStatus(str, Enum):
    """Service status."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


# Sub-models


class HealthResponse(BaseModel):
    """Service health information."""

    status: ServiceStatus
    version: str
    timestamp: datetime
