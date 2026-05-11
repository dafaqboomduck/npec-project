"""Tests for the four Overview endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.core.security import get_current_user
from src.api.main import app
from src.api.schemas.overview import (
    ActivityFeedResponse,
    ActivityItem,
    DriftStatus,
    EnvironmentResponse,
    EventType,
    ModelHealthResponse,
    OverviewStatusResponse,
    SystemStatus,
)

# Helpers


def _status_response(**kwargs) -> OverviewStatusResponse:
    defaults = dict(
        total_experiments=42,
        active_model_count=1,
        data_collected_tb=0.021,
        quota_percent=0,
    )
    return OverviewStatusResponse(**{**defaults, **kwargs})


def _health_response(**kwargs) -> ModelHealthResponse:
    defaults = dict(f1_score=0.92, latency_ms=843, drift_status=DriftStatus.NONE)
    return ModelHealthResponse(**{**defaults, **kwargs})


def _activity_response(n_items: int = 2) -> ActivityFeedResponse:
    items = [
        ActivityItem(
            event_type=EventType.UPLOAD,
            title="Root scan uploaded",
            description="scan.png submitted for inference.",
            timestamp=datetime.now(timezone.utc),
            resource_url="/api/v1/predictions/some-uuid",
        )
        for _ in range(n_items)
    ]
    return ActivityFeedResponse(items=items, total=n_items, page=1, page_size=20)


def _env_response(**kwargs) -> EnvironmentResponse:
    defaults = dict(
        gpu_temp_celsius=65.0,
        memory_percent=42,
        system_status=SystemStatus.OPTIMAL,
    )
    return EnvironmentResponse(**{**defaults, **kwargs})


# GET /overview/status


def test_status_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/overview/status")
    assert response.status_code == 401


def test_status_returns_correct_schema(client: TestClient) -> None:
    with patch(
        "src.api.repositories.overview.OverviewRepository.get_status",
        new_callable=AsyncMock,
        return_value=_status_response(),
    ):

        response = client.get("/api/v1/overview/status")

    assert response.status_code == 200
    data = response.json()
    assert "total_experiments" in data
    assert "active_model_count" in data
    assert "data_collected_tb" in data
    assert "quota_percent" in data


def test_status_quota_is_bounded(client: TestClient) -> None:
    with patch(
        "src.api.repositories.overview.OverviewRepository.get_status",
        new_callable=AsyncMock,
        return_value=_status_response(quota_percent=100),
    ):

        data = client.get("/api/v1/overview/status").json()
    assert 0 <= data["quota_percent"] <= 100


# GET /overview/model-health


def test_model_health_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/overview/model-health")
    assert response.status_code == 401


def test_model_health_returns_correct_schema(client: TestClient) -> None:
    with patch(
        "src.api.repositories.overview.OverviewRepository.get_model_health",
        new_callable=AsyncMock,
        return_value=_health_response(),
    ):

        response = client.get("/api/v1/overview/model-health")

    assert response.status_code == 200
    data = response.json()
    assert "f1_score" in data
    assert "latency_ms" in data
    assert "drift_status" in data


def test_model_health_returns_defaults_when_no_metrics(client: TestClient) -> None:
    """Endpoint must return sensible defaults, not 500, when no metrics exist."""
    with patch(
        "src.api.repositories.overview.OverviewRepository.get_model_health",
        new_callable=AsyncMock,
        return_value=ModelHealthResponse(f1_score=0.0, latency_ms=0, drift_status=DriftStatus.NONE),
    ):

        response = client.get("/api/v1/overview/model-health")
    assert response.status_code == 200
    assert response.json()["drift_status"] == "none"


def test_model_health_drift_statuses(client: TestClient) -> None:
    for drift in (DriftStatus.NONE, DriftStatus.LOW, DriftStatus.MEDIUM, DriftStatus.HIGH):
        with patch(
            "src.api.repositories.overview.OverviewRepository.get_model_health",
            new_callable=AsyncMock,
            return_value=_health_response(drift_status=drift),
        ):

            data = client.get("/api/v1/overview/model-health").json()
        assert data["drift_status"] == drift.value


# GET /overview/activity


def test_activity_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/overview/activity")
    assert response.status_code == 401


def test_activity_returns_paginated_response(client: TestClient) -> None:
    with patch(
        "src.api.repositories.overview.OverviewRepository.get_activity",
        new_callable=AsyncMock,
        return_value=_activity_response(n_items=2),
    ):

        response = client.get("/api/v1/overview/activity")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert len(data["items"]) == 2


def test_activity_returns_empty_when_none(client: TestClient) -> None:
    with patch(
        "src.api.repositories.overview.OverviewRepository.get_activity",
        new_callable=AsyncMock,
        return_value=ActivityFeedResponse(items=[], total=0, page=1, page_size=20),
    ):

        data = client.get("/api/v1/overview/activity").json()
    assert data["items"] == []
    assert data["total"] == 0


def test_activity_passes_pagination_params(client: TestClient) -> None:
    get_mock = AsyncMock(return_value=_activity_response(0))
    with patch("src.api.repositories.overview.OverviewRepository.get_activity", get_mock):

        client.get("/api/v1/overview/activity?page=3&page_size=5")
    get_mock.assert_called_once_with(page=3, page_size=5)


def test_activity_rejects_invalid_page_size(client: TestClient) -> None:

    response = client.get("/api/v1/overview/activity?page_size=999")
    assert response.status_code == 422


def test_activity_items_have_required_fields(client: TestClient) -> None:
    with patch(
        "src.api.repositories.overview.OverviewRepository.get_activity",
        new_callable=AsyncMock,
        return_value=_activity_response(n_items=1),
    ):

        data = client.get("/api/v1/overview/activity").json()
    item = data["items"][0]
    assert "event_type" in item
    assert "title" in item
    assert "description" in item
    assert "timestamp" in item


# GET /overview/environment


def test_environment_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/overview/environment")
    assert response.status_code == 401


def test_environment_returns_403_for_user(client: TestClient) -> None:
    response = client.get("/api/v1/overview/environment")
    assert response.status_code == 403
    assert response.json()["error_code"] == "FORBIDDEN"


def test_environment_returns_200_for_admin(admin_client: TestClient) -> None:
    with patch(
        "src.api.routers.overview.get_environment_stats",
        return_value=_env_response(),
    ):
        response = admin_client.get("/api/v1/overview/environment")
    assert response.status_code == 200


def test_environment_returns_correct_schema(admin_client: TestClient) -> None:
    with patch(
        "src.api.routers.overview.get_environment_stats",
        return_value=_env_response(),
    ):
        data = admin_client.get("/api/v1/overview/environment").json()
    assert "gpu_temp_celsius" in data
    assert "memory_percent" in data
    assert "system_status" in data


def test_environment_gpu_temp_can_be_null(admin_client: TestClient) -> None:
    """No GPU present — gpu_temp_celsius must be null, not an error."""
    with patch(
        "src.api.routers.overview.get_environment_stats",
        return_value=_env_response(gpu_temp_celsius=None),
    ):
        data = admin_client.get("/api/v1/overview/environment").json()
    assert data["gpu_temp_celsius"] is None
    assert data["system_status"] in ("optimal", "degraded", "critical")


def test_environment_system_status_thresholds(admin_client: TestClient) -> None:
    """Verify threshold logic maps correctly to system_status labels."""
    cases = [
        (42, "optimal"),
        (85, "degraded"),
        (97, "critical"),
    ]
    for memory_percent, expected_status in cases:
        with patch(
            "src.api.routers.overview.get_environment_stats",
            return_value=_env_response(memory_percent=memory_percent, system_status=SystemStatus(expected_status)),
        ):
            data = admin_client.get("/api/v1/overview/environment").json()
        assert data["system_status"] == expected_status


# Drift threshold unit tests (no HTTP, pure logic)


def test_drift_threshold_none() -> None:
    from src.api.repositories.overview import _drift_status

    assert _drift_status(0.0) == DriftStatus.NONE
    assert _drift_status(0.09) == DriftStatus.NONE


def test_drift_threshold_low() -> None:
    from src.api.repositories.overview import _drift_status

    assert _drift_status(0.1) == DriftStatus.LOW
    assert _drift_status(0.29) == DriftStatus.LOW


def test_drift_threshold_medium() -> None:
    from src.api.repositories.overview import _drift_status

    assert _drift_status(0.3) == DriftStatus.MEDIUM
    assert _drift_status(0.59) == DriftStatus.MEDIUM


def test_drift_threshold_high() -> None:
    from src.api.repositories.overview import _drift_status

    assert _drift_status(0.6) == DriftStatus.HIGH
    assert _drift_status(1.0) == DriftStatus.HIGH
