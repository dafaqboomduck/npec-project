"""Overview Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# Enums


class EventType(str, Enum):
    """Event type enum."""

    RETRAINING = "retraining"
    UPLOAD = "upload"
    ANNOTATION = "annotation"
    DEPLOYMENT = "deployment"


class DriftStatus(str, Enum):
    """Drift status enum."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SystemStatus(str, Enum):
    """System status enum."""

    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# Response models


class OverviewStatusResponse(BaseModel):
    """Returned by GET /overview/status."""

    total_experiments: int
    active_model_count: int
    data_collected_tb: float
    quota_percent: int = Field(..., ge=0, le=100)


class ActivityItem(BaseModel):
    """Single event in the activity feed."""

    event_type: EventType
    title: str
    description: str
    timestamp: datetime
    resource_url: str | None = None


class ActivityFeedResponse(BaseModel):
    """Paginated activity feed returned by GET /overview/activity."""

    items: list[ActivityItem]
    total: int
    page: int
    page_size: int


class ModelHealthResponse(BaseModel):
    """Returned by GET /overview/model-health."""

    f1_score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: int
    drift_status: DriftStatus


class EnvironmentResponse(BaseModel):
    """Returned by GET /overview/environment (admin only)."""

    gpu_temp_celsius: float | None = None
    memory_percent: int = Field(..., ge=0, le=100)
    system_status: SystemStatus
