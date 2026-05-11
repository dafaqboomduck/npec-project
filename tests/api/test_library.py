"""Tests for the Library and Inference endpoints.

Covers:
- GET    /datasets
- GET    /datasets/{id}
- GET    /annotations
- GET    /annotations/{id}
- DELETE /annotations/{id}
- POST   /annotations/{id}/confirm
- GET    /models
- GET    /models/{id}
- DELETE /models/{id}
- POST   /models/{id}/deploy
- POST   /predictions/{id}/confirm
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from src.api.core.security import get_current_user
from src.api.main import app
from src.api.schemas.library import (
    AnnotationResponse,
    AnnotationStatus,
    ModelDetail,
    ModelStatus,
    ModelSummary,
)
from src.api.schemas.predictions import (
    PredictionMetadata,
    PredictionResponse,
    PredictionStatus,
)

# Helpers


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


def _make_tiff(width: int = 512, height: int = 512) -> bytes:
    """Create a minimal valid single-frame TIFF."""
    buf = io.BytesIO()
    Image.new("L", (width, height), 128).save(buf, format="TIFF")
    return buf.getvalue()


def _make_annotation(annotation_id: uuid.UUID | None = None) -> AnnotationResponse:
    return AnnotationResponse(
        annotation_id=annotation_id or uuid.uuid4(),
        prediction_id=uuid.uuid4(),
        status=AnnotationStatus.ANNOTATED,
        mask_url="file:///tmp/mask.tiff",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_prediction(prediction_id: uuid.UUID | None = None) -> PredictionResponse:
    return PredictionResponse(
        prediction_id=prediction_id or uuid.uuid4(),
        status=PredictionStatus.OK,
        metadata=PredictionMetadata(
            model_version="mock-v0.0.0",
            inference_timestamp=datetime.now(timezone.utc),
            processing_time_ms=100,
            input_resolution=[512, 512],
            input_channels=1,
            normalisation_applied=True,
            scale_factor_mm_per_px=0.05405,
        ),
        plants=[],
        overall_confidence=0.0,
        warnings=[],
    )


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


# DELETE /annotations/{id}


def test_delete_annotation_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.delete(f"/api/v1/annotations/{uuid.uuid4()}")
    assert response.status_code == 401


def test_delete_annotation_returns_404_before_migration(client: TestClient) -> None:
    """Until the annotations table is migrated, delete always returns 404."""
    response = client.delete(f"/api/v1/annotations/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_delete_annotation_returns_200_when_found(client: TestClient) -> None:
    annotation_id = uuid.uuid4()
    with patch(
        "src.api.repositories.annotations.AnnotationRepository.delete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        response = client.delete(f"/api/v1/annotations/{annotation_id}")
    assert response.status_code == 200
    assert "message" in response.json()


# POST /annotations/{id}/confirm


def test_confirm_annotation_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    tiff = _make_tiff()
    response = client.post(
        f"/api/v1/annotations/{uuid.uuid4()}/confirm",
        files={"mask": ("mask.tiff", tiff, "image/tiff")},
    )
    assert response.status_code == 401


def test_confirm_annotation_rejects_non_tiff(client: TestClient) -> None:
    """Files that are not .tiff must return 422 before any DB call."""
    response = client.post(
        f"/api/v1/annotations/{uuid.uuid4()}/confirm",
        files={"mask": ("mask.png", b"fake-png", "image/png")},
    )
    assert response.status_code == 422


def test_confirm_annotation_returns_404_when_not_found(client: TestClient) -> None:
    tiff = _make_tiff()
    with (
        patch("src.api.routers.library.validate_tiff_mask"),
        patch(
            "src.api.routers.library.save_annotation_mask",
            return_value="file:///tmp/mask.tiff",
        ),
        patch(
            "src.api.repositories.annotations.AnnotationRepository.confirm",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        response = client.post(
            f"/api/v1/annotations/{uuid.uuid4()}/confirm",
            files={"mask": ("mask.tiff", tiff, "image/tiff")},
        )
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_confirm_annotation_returns_updated_annotation(client: TestClient) -> None:
    annotation = _make_annotation()
    tiff = _make_tiff()
    with (
        patch("src.api.routers.library.validate_tiff_mask"),
        patch(
            "src.api.routers.library.save_annotation_mask",
            return_value="file:///tmp/mask.tiff",
        ),
        patch(
            "src.api.repositories.annotations.AnnotationRepository.confirm",
            new_callable=AsyncMock,
            return_value=annotation,
        ),
    ):
        response = client.post(
            f"/api/v1/annotations/{annotation.annotation_id}/confirm",
            files={"mask": ("mask.tiff", tiff, "image/tiff")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "annotated"
    assert data["mask_url"] == "file:///tmp/mask.tiff"


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


# GET /models/{id}


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


# ---------------------------------------------------------------------------
# DELETE /models/{id}
# ---------------------------------------------------------------------------


def test_delete_model_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.delete(f"/api/v1/models/{uuid.uuid4()}")
    assert response.status_code == 401


def test_delete_model_returns_403_for_user(client: TestClient) -> None:
    response = client.delete(f"/api/v1/models/{uuid.uuid4()}")
    assert response.status_code == 403
    assert response.json()["error_code"] == "FORBIDDEN"


def test_delete_model_returns_404_for_unknown_id(admin_client: TestClient) -> None:
    with patch(
        "src.api.repositories.models.ModelRepository.delete",
        new_callable=AsyncMock,
        return_value=False,
    ):
        response = admin_client.delete(f"/api/v1/models/{uuid.uuid4()}")
    assert response.status_code == 404


def test_delete_model_returns_200_for_admin(admin_client: TestClient) -> None:
    with patch(
        "src.api.repositories.models.ModelRepository.delete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        response = admin_client.delete(f"/api/v1/models/{uuid.uuid4()}")
    assert response.status_code == 200
    assert "message" in response.json()


# POST /models/{id}/deploy


def test_deploy_model_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post(f"/api/v1/models/{uuid.uuid4()}/deploy")
    assert response.status_code == 401


def test_deploy_model_returns_403_for_user(client: TestClient) -> None:
    response = client.post(f"/api/v1/models/{uuid.uuid4()}/deploy")
    assert response.status_code == 403
    assert response.json()["error_code"] == "FORBIDDEN"


def test_deploy_model_returns_404_for_unknown_id(admin_client: TestClient) -> None:
    with patch(
        "src.api.repositories.models.ModelRepository.exists",
        new_callable=AsyncMock,
        return_value=False,
    ):
        response = admin_client.post(f"/api/v1/models/{uuid.uuid4()}/deploy")
    assert response.status_code == 404


def test_deploy_model_returns_200_for_admin(admin_client: TestClient) -> None:
    with (
        patch(
            "src.api.repositories.models.ModelRepository.exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.api.services.deployment.StubDeploymentService.deploy",
            new_callable=AsyncMock,
        ),
    ):
        response = admin_client.post(f"/api/v1/models/{uuid.uuid4()}/deploy")
    assert response.status_code == 200
    assert "message" in response.json()


# POST /predictions/{id}/confirm


def test_confirm_prediction_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post(
        f"/api/v1/predictions/{uuid.uuid4()}/confirm",
    )
    assert response.status_code == 401


def test_confirm_prediction_returns_404_for_unknown(client: TestClient) -> None:
    with patch(
        "src.api.repositories.database.DatabasePredictionRepository.confirm",
        new_callable=AsyncMock,
        side_effect=ValueError("not found"),
    ):
        response = client.post(
            f"/api/v1/predictions/{uuid.uuid4()}/confirm",
        )
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_confirm_prediction_returns_409_if_already_confirmed(client: TestClient) -> None:
    prediction = _make_prediction()
    with patch(
        "src.api.repositories.database.DatabasePredictionRepository.confirm",
        new_callable=AsyncMock,
        return_value=(prediction, True),  # already_confirmed=True
    ):
        response = client.post(
            f"/api/v1/predictions/{prediction.prediction_id}/confirm",
        )
    assert response.status_code == 409
    assert response.json()["error_code"] == "CONFLICT"


def test_confirm_prediction_returns_200_on_success(client: TestClient) -> None:
    prediction = _make_prediction()
    with patch(
        "src.api.repositories.database.DatabasePredictionRepository.confirm",
        new_callable=AsyncMock,
        return_value=(prediction, False),  # already_confirmed=False
    ):
        response = client.post(
            f"/api/v1/predictions/{prediction.prediction_id}/confirm",
            json={
                "species": "Arabidopsis thaliana",
                "age_days": 7,
                "treatment": "control",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "message" in data
    assert data["message"] == "Prediction confirmed and annotation created."


def test_confirm_prediction_works_without_body(client: TestClient) -> None:
    """Body is optional — confirm without any metadata should still succeed."""
    prediction = _make_prediction()
    with patch(
        "src.api.repositories.database.DatabasePredictionRepository.confirm",
        new_callable=AsyncMock,
        return_value=(prediction, False),
    ):
        response = client.post(
            f"/api/v1/predictions/{prediction.prediction_id}/confirm",
        )
    assert response.status_code == 200
