"""Dashboard Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DashboardMetricsResponse(BaseModel):
    """Returned by GET /dashboard/metrics."""

    f1_score: float = Field(..., ge=0.0, le=1.0)
    throughput_img_per_s: float


class F1HistoryPoint(BaseModel):
    """Single epoch/f1 data point in the training history."""

    epoch: int
    f1_score: float = Field(..., ge=0.0, le=1.0)


class F1HistoryResponse(BaseModel):
    """Returned by GET /dashboard/f1-history."""

    model_version: str
    data_points: list[F1HistoryPoint]


class SpeciesCount(BaseModel):
    """Single species and its sample count."""

    species: str
    count: int


class DataDistributionResponse(BaseModel):
    """Returned by GET /dashboard/data-distribution."""

    species_counts: list[SpeciesCount]
