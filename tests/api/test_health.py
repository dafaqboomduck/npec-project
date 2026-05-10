"""Tests for GET /api/v1/health."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.core.security import get_current_user
from api.main import app


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_requires_no_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/v1/health")
    assert response.status_code != 401
    assert response.status_code != 403


def test_health_returns_ok_when_all_pass(client: TestClient) -> None:
    with (
        patch("api.routers.health._check_db", new_callable=AsyncMock, return_value=True),
        patch("api.routers.health._check_model", new_callable=AsyncMock, return_value=True),
    ):
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_degraded_when_one_fails(client: TestClient) -> None:
    with (
        patch("api.routers.health._check_db", new_callable=AsyncMock, return_value=True),
        patch("api.routers.health._check_model", new_callable=AsyncMock, return_value=False),
    ):
        response = client.get("/api/v1/health")
    assert response.json()["status"] == "degraded"


def test_health_returns_unavailable_when_all_fail(client: TestClient) -> None:
    with (
        patch("api.routers.health._check_db", new_callable=AsyncMock, return_value=False),
        patch("api.routers.health._check_model", new_callable=AsyncMock, return_value=False),
    ):
        response = client.get("/api/v1/health")
    assert response.json()["status"] == "unavailable"
