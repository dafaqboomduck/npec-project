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


class ModelMetricRow(BaseModel):
    """Per-model operational + plant-detection rollup.

    One row per distinct ``model_name`` seen in the audit log. Lets the
    dashboard show, e.g., that root-detection served 120 requests at a
    0.8% error rate while shoot-detection served 95.
    """

    model_name: str
    requests: int = Field(default=0, ge=0)
    error_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    total_plants_detected: int = Field(default=0, ge=0)


class OperationalMetricsResponse(BaseModel):
    """Returned by GET /dashboard/operational-metrics.

    Operational metrics (request counts, error rate, latency
    percentiles) and business/plant metrics (plants detected), all
    derived from the ``inference_audit_log`` ledger. All fields default
    to zero so the endpoint can return an empty-but-valid body when no
    predictions have been logged yet.
    """

    total_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    error_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    p50_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_latency_ms: float = Field(default=0.0, ge=0.0)
    total_plants_detected: int = Field(default=0, ge=0)
    avg_plants_per_request: float = Field(default=0.0, ge=0.0)
    per_model: list[ModelMetricRow] = Field(default_factory=list)
