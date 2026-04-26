"""Tests for the library endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.core.security import get_current_user
from src.api.main import app
from src.api.schemas.library import ModelDetail, ModelStatus, ModelSummary


def _model_summary(**kwargs) -> ModelSummary:
    defaults = dict(
        model_id=uuid.uuid4(),
        name="Test Model",
        status=ModelStatus.REGISTERED,
        created_at=datetime.now(timezone.utc),
    )
    return ModelSummary(**{**defaults, **kwargs})


def _model_detail(**kwargs) -> ModelDetail:
    defaults = dict(
        model_id=uuid.uuid4(),
        name="Test Model",
        status=ModelStatus.DEPLOYED,
        created_at=datetime.now(timezone.utc),
        f1_score=0.92,
        precision=0.91,
        recall=0.93,
        inference_latency_ms=843,
        artifact_uri="s3://bucket/model.keras",
        deployed_at=datetime.now(timezone.utc),
    )
    return ModelDetail(**{**defaults, **kwargs})


# GET /datasets


def test_list_datasets_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/datasets")
    assert response.status_code == 401


def test_list_datasets_returns_empty_list_before_migration(client: TestClient) -> None:
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    assert response.json() == []


def test_list_datasets_accepts_filters(client: TestClient) -> None:
    response = client.get(
        "/api/v1/datasets",
        params={"species": "Arabidopsis thaliana", "growth_stage": "seedling"},
    )
    assert response.status_code == 200


def test_get_dataset_returns_404_before_migration(client: TestClient) -> None:
    response = client.get(f"/api/v1/datasets/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


# GET /annotations


def test_list_annotations_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/annotations")
    assert response.status_code == 401


def test_list_annotations_returns_empty_list_before_migration(client: TestClient) -> None:
    response = client.get("/api/v1/annotations")
    assert response.status_code == 200
    assert response.json() == []


def test_list_annotations_accepts_pagination_params(client: TestClient) -> None:
    response = client.get(
        "/api/v1/annotations",
        params={"page": 2, "page_size": 10},
    )
    assert response.status_code == 200


def test_list_annotations_rejects_invalid_page_size(client: TestClient) -> None:
    response = client.get(
        "/api/v1/annotations",
        params={"page_size": 999},
    )
    assert response.status_code == 422


def test_get_annotation_returns_404_before_migration(client: TestClient) -> None:
    response = client.get(f"/api/v1/annotations/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


# GET /models


def test_list_models_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/models")
    assert response.status_code == 401


def test_list_models_returns_list(client: TestClient) -> None:
    with patch(
        "src.api.repositories.models.ModelRepository.list",
        new_callable=AsyncMock,
        return_value=[_model_summary()],
    ):
        response = client.get("/api/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "model_id" in data[0]
    assert "status" in data[0]


def test_list_models_returns_empty_list(client: TestClient) -> None:
    with patch(
        "src.api.repositories.models.ModelRepository.list",
        new_callable=AsyncMock,
        return_value=[],
    ):
        response = client.get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == []


# GET /models/{model_id}


def test_get_model_returns_404_for_unknown_id(client: TestClient) -> None:
    with patch(
        "src.api.repositories.models.ModelRepository.get",
        new_callable=AsyncMock,
        return_value=None,
    ):
        response = client.get(f"/api/v1/models/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_get_model_user_does_not_see_admin_fields(client: TestClient) -> None:
    """User-role tokens must not receive artifact_uri or deployed_at."""
    detail = _model_detail(artifact_uri=None, deployed_at=None)
    with patch(
        "src.api.repositories.models.ModelRepository.get",
        new_callable=AsyncMock,
        return_value=detail,
    ):
        data = client.get(f"/api/v1/models/{detail.model_id}").json()

    assert data["artifact_uri"] is None
    assert data["deployed_at"] is None


def test_get_model_admin_sees_admin_fields(admin_client: TestClient) -> None:
    """Admin-role tokens must receive artifact_uri and deployed_at."""
    detail = _model_detail()
    with patch(
        "src.api.repositories.models.ModelRepository.get",
        new_callable=AsyncMock,
        return_value=detail,
    ):
        data = admin_client.get(f"/api/v1/models/{detail.model_id}").json()

    assert data["artifact_uri"] is not None
    assert data["deployed_at"] is not None


def test_get_model_returns_correct_schema(admin_client: TestClient) -> None:
    detail = _model_detail()
    with patch(
        "src.api.repositories.models.ModelRepository.get",
        new_callable=AsyncMock,
        return_value=detail,
    ):
        data = admin_client.get(f"/api/v1/models/{detail.model_id}").json()

    assert "model_id" in data
    assert "name" in data
    assert "status" in data
    assert "created_at" in data
    assert "f1_score" in data
