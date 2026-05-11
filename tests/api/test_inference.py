"""Unit tests for the inference endpoints.

Covers:
- POST /api/v1/predict - happy path and all validation error branches
- GET  /api/v1/predictions/{id} - found and not-found
- DELETE /api/v1/predictions/{id} - found and not-found
"""

import io
import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from src.api.core.security import get_current_user
from src.api.main import app
from tests.api.test_library import _make_prediction

# Helpers


def _upload(
    client: TestClient,
    image_bytes: bytes,
    filename: str = "scan.png",
    content_type: str = "image/png",
    threshold: float = 0.50,
):
    """POST an image to /api/v1/predict and return the response."""
    return client.post(
        "/api/v1/predict",
        files={"file": (filename, image_bytes, content_type)},
        data={"confidence_threshold": str(threshold)},
    )


def _make_png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), 128).save(buf, format="PNG")
    return buf.getvalue()


def _make_tiff(width: int, height: int, frames: int = 1) -> bytes:
    buf = io.BytesIO()
    frames_list = [Image.new("L", (width, height), 128) for _ in range(frames)]
    frames_list[0].save(
        buf,
        format="TIFF",
        save_all=True,
        append_images=frames_list[1:],
    )
    return buf.getvalue()


# POST /predict - happy path


def test_predict_returns_200_with_valid_png(client: TestClient, valid_png_512: bytes) -> None:
    response = _upload(client, valid_png_512)
    assert response.status_code == 200


def test_predict_response_has_required_fields(client: TestClient, valid_png_512: bytes) -> None:
    data = _upload(client, valid_png_512).json()
    assert "prediction_id" in data
    assert "status" in data
    assert "metadata" in data
    assert "plants" in data
    assert "overall_confidence" in data
    assert "warnings" in data


def test_predict_status_is_ok(client: TestClient, valid_png_512: bytes) -> None:
    data = _upload(client, valid_png_512).json()
    assert data["status"] == "ok"


def test_predict_returns_unique_ids(client: TestClient, valid_png_512: bytes) -> None:
    id1 = _upload(client, valid_png_512).json()["prediction_id"]
    id2 = _upload(client, valid_png_512).json()["prediction_id"]
    assert id1 != id2


def test_predict_confidence_threshold_filters_plants(client: TestClient, valid_png_512: bytes) -> None:
    """A threshold of 0.95 should exclude the second mock plant (confidence=0.87)."""
    data = _upload(client, valid_png_512, threshold=0.95).json()
    assert all(p["confidence"] >= 0.95 for p in data["plants"])


def test_predict_accepts_tiff(client: TestClient) -> None:
    tiff = _make_tiff(512, 512)
    response = _upload(client, tiff, filename="scan.tiff", content_type="image/tiff")
    assert response.status_code == 200


# POST /predict - file size validation (413)


def test_predict_rejects_oversized_file(client: TestClient) -> None:
    """Files larger than 50 MB must return HTTP 413."""
    oversized = b"x" * (50 * 1024 * 1024 + 1)
    response = _upload(client, oversized)
    assert response.status_code == 413
    assert response.json()["error_code"] == "FILE_TOO_LARGE"


# POST /predict - format validation (415)


def test_predict_rejects_unsupported_format(client: TestClient) -> None:
    response = _upload(
        client,
        b"GIF89a...",
        filename="scan.gif",
        content_type="image/gif",
    )
    assert response.status_code == 415
    assert response.json()["error_code"] == "UNSUPPORTED_FORMAT"


# POST /predict - resolution validation (422)


def test_predict_rejects_image_too_small(client: TestClient) -> None:
    tiny = _make_png(128, 128)
    response = _upload(client, tiny)
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_RESOLUTION"


def test_predict_rejects_image_too_large(client: TestClient) -> None:
    huge = _make_png(5002, 5002)
    response = _upload(client, huge)
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_RESOLUTION"


def test_predict_rejects_multi_frame_tiff(client: TestClient) -> None:
    multi = _make_tiff(512, 512, frames=2)
    response = _upload(client, multi, filename="multi.tiff", content_type="image/tiff")
    assert response.status_code == 422
    assert response.json()["error_code"] == "MULTI_PAGE_TIFF"


def test_predict_rejects_corrupt_file(client: TestClient) -> None:
    """Corrupt bytes with a valid extension should fail at image decode (422)."""
    response = _upload(client, b"not-an-image")
    assert response.status_code == 422
    assert response.json()["error_code"] == "CORRUPT_FILE"


def test_predict_still_returns_200_when_monitoring_fails(client: TestClient, valid_png_512: bytes) -> None:
    """A monitoring write failure must not affect the predict response."""
    with patch(
        "src.api.services.monitoring.MonitoringService.record_prediction_metrics",
        new_callable=AsyncMock,
        side_effect=Exception("DB connection lost"),
    ):
        response = client.post(
            "/api/v1/predict",
            files={"file": ("scan.png", valid_png_512, "image/png")},
            data={"confidence_threshold": "0.5"},
        )
    assert response.status_code == 200


def test_predict_calls_monitoring_on_success(client: TestClient, valid_png_512: bytes) -> None:
    record_mock = AsyncMock()
    with patch(
        "src.api.services.monitoring.MonitoringService.record_prediction_metrics",
        record_mock,
    ):

        client.post(
            "/api/v1/predict",
            files={"file": ("scan.png", valid_png_512, "image/png")},
            data={"confidence_threshold": "0.5"},
        )
    record_mock.assert_called_once()


# GET /predictions/{id}


def test_get_prediction_returns_stored_result(client, valid_png_512):
    prediction = _upload(client, valid_png_512).json()

    with patch(
        "src.api.repositories.database.DatabasePredictionRepository.get",
        new_callable=AsyncMock,
        return_value=prediction,
    ):
        response = client.get(f"/api/v1/predictions/{prediction['prediction_id']}")

    assert response.status_code == 200


def test_get_prediction_returns_404_for_unknown_id(client: TestClient) -> None:
    response = client.get(f"/api/v1/predictions/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


# DELETE /predictions/{id}


def test_delete_prediction_returns_200(client: TestClient, valid_png_512: bytes) -> None:
    prediction_id = _upload(client, valid_png_512).json()["prediction_id"]

    with patch(
        "src.api.repositories.database.DatabasePredictionRepository.delete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        response = client.delete(f"/api/v1/predictions/{prediction_id}")

    assert response.status_code == 200
    assert "message" in response.json()


def test_delete_prediction_removes_from_store(client: TestClient, valid_png_512: bytes) -> None:
    prediction_id = _upload(client, valid_png_512).json()["prediction_id"]
    client.delete(f"/api/v1/predictions/{prediction_id}")
    response = client.get(f"/api/v1/predictions/{prediction_id}")
    assert response.status_code == 404


def test_delete_prediction_returns_404_for_unknown_id(client: TestClient) -> None:
    response = client.delete(f"/api/v1/predictions/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_list_predictions_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/predictions")
    assert response.status_code == 401


# GET /predictions


def test_list_predictions_returns_paginated_response(client: TestClient) -> None:
    prediction = _make_prediction()

    with patch(
        "src.api.repositories.database.DatabasePredictionRepository.list_by_user",
        new_callable=AsyncMock,
        return_value=[prediction],
    ):
        response = client.get("/api/v1/predictions")

    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0


def test_list_predictions_returns_empty_when_none(client: TestClient) -> None:
    response = client.get("/api/v1/predictions")

    assert response.status_code == 200
    assert response.json() == []


def test_list_predictions_rejects_invalid_page_size(client: TestClient) -> None:

    response = client.get("/api/v1/predictions?page_size=999")
    assert response.status_code == 422
