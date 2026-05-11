"""Tests for the three Dashboard endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.src.core.security import get_current_user
from api.src.main import app
from api.src.schemas.dashboard import (
    DashboardMetricsResponse,
    DataDistributionResponse,
    F1HistoryPoint,
    F1HistoryResponse,
    SpeciesCount,
)

# Helpers


def _metrics(**kwargs) -> DashboardMetricsResponse:
    return DashboardMetricsResponse(**{"f1_score": 0.92, "throughput_img_per_s": 3, **kwargs})


def _history(n_points: int = 3) -> F1HistoryResponse:
    return F1HistoryResponse(
        model_version="unet-npec-v1.0.0",
        data_points=[F1HistoryPoint(epoch=i + 1, f1_score=round(0.7 + i * 0.05, 2)) for i in range(n_points)],
    )


def _distribution(species: list[str] | None = None) -> DataDistributionResponse:
    if species is None:
        species = ["Arabidopsis thaliana", "Nicotiana benthamiana"]
    return DataDistributionResponse(species_counts=[SpeciesCount(species=s, count=10) for s in species])


# GET /dashboard/metrics


def test_metrics_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/dashboard/metrics")
    assert response.status_code == 401


def test_metrics_returns_correct_schema(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_metrics",
        new_callable=AsyncMock,
        return_value=_metrics(),
    ):

        response = client.get("/api/v1/dashboard/metrics")

    assert response.status_code == 200
    data = response.json()
    assert "f1_score" in data
    assert "throughput_img_per_s" in data


def test_metrics_returns_zero_defaults_when_no_model(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_metrics",
        new_callable=AsyncMock,
        return_value=DashboardMetricsResponse(f1_score=0.0, throughput_img_per_s=0),
    ):

        data = client.get("/api/v1/dashboard/metrics").json()

    assert data["f1_score"] == 0.0
    assert data["throughput_img_per_s"] == 0


def test_metrics_f1_score_is_bounded(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_metrics",
        new_callable=AsyncMock,
        return_value=_metrics(f1_score=0.92),
    ):

        data = client.get("/api/v1/dashboard/metrics").json()

    assert 0.0 <= data["f1_score"] <= 1.0


# GET /dashboard/f1-history


def test_f1_history_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/dashboard/f1-history")
    assert response.status_code == 401


def test_f1_history_returns_correct_schema(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_f1_history",
        new_callable=AsyncMock,
        return_value=_history(),
    ):

        response = client.get("/api/v1/dashboard/f1-history")

    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "data_points" in data
    assert isinstance(data["data_points"], list)


def test_f1_history_data_points_have_required_fields(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_f1_history",
        new_callable=AsyncMock,
        return_value=_history(n_points=2),
    ):

        data = client.get("/api/v1/dashboard/f1-history").json()

    for point in data["data_points"]:
        assert "epoch" in point
        assert "f1_score" in point


def test_f1_history_ordered_by_epoch_ascending(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_f1_history",
        new_callable=AsyncMock,
        return_value=_history(n_points=3),
    ):

        points = client.get("/api/v1/dashboard/f1-history").json()["data_points"]

    epochs = [p["epoch"] for p in points]
    assert epochs == sorted(epochs)


def test_f1_history_empty_when_no_model(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_f1_history",
        new_callable=AsyncMock,
        return_value=F1HistoryResponse(model_version="unknown", data_points=[]),
    ):

        data = client.get("/api/v1/dashboard/f1-history").json()

    assert data["data_points"] == []
    assert data["model_version"] == "unknown"


# GET /dashboard/data-distribution


def test_data_distribution_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/dashboard/data-distribution")
    assert response.status_code == 401


def test_data_distribution_returns_correct_schema(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_data_distribution",
        new_callable=AsyncMock,
        return_value=_distribution(),
    ):

        response = client.get("/api/v1/dashboard/data-distribution")

    assert response.status_code == 200
    data = response.json()
    assert "species_counts" in data
    assert isinstance(data["species_counts"], list)


def test_data_distribution_items_have_required_fields(client: TestClient) -> None:
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_data_distribution",
        new_callable=AsyncMock,
        return_value=_distribution(),
    ):

        data = client.get("/api/v1/dashboard/data-distribution").json()

    for item in data["species_counts"]:
        assert "species" in item
        assert "count" in item


def test_data_distribution_returns_empty_before_migration(
    client: TestClient,
) -> None:
    """Until annotations table is migrated, endpoint returns empty list."""
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_data_distribution",
        new_callable=AsyncMock,
        return_value=DataDistributionResponse(species_counts=[]),
    ):

        data = client.get("/api/v1/dashboard/data-distribution").json()

    assert data["species_counts"] == []


def test_data_distribution_excludes_zero_count_species(client: TestClient) -> None:
    """Only species with at least one sample should appear."""
    with patch(
        "api.src.repositories.dashboard.DashboardRepository.get_data_distribution",
        new_callable=AsyncMock,
        return_value=_distribution(species=["Arabidopsis thaliana"]),
    ):

        data = client.get("/api/v1/dashboard/data-distribution").json()

    assert all(item["count"] > 0 for item in data["species_counts"])
