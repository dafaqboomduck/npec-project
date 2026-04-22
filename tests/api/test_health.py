"""Tests for GET /api/v1/health."""

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_schema(client: TestClient) -> None:
    data = client.get("/api/v1/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_health_requires_no_auth(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code != 401
    assert response.status_code != 403
