"""Tests for JWT authentication endpoints.

Since the auth layer hits a real database, tests use a mocked
UserRepository injected via FastAPI's dependency override system.
This keeps tests fast and self-contained — no Postgres required.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

from api.src.core.security import get_current_user
from api.src.main import app
from api.src.repositories.users import UserRepository
from tests.api.conftest import _ADMIN_ID, _USER_ID, _make_user

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_repo(
    client: TestClient,
    *,
    user: MagicMock | None = None,
    rotate_result=None,
) -> AsyncMock:
    """Patch UserRepository methods for a single test."""
    repo = AsyncMock(spec=UserRepository)
    repo.get_by_email.return_value = user
    repo.get_by_id.return_value = user
    repo.issue_refresh_token.return_value = "mock-refresh-token"
    repo.rotate_refresh_token.return_value = rotate_result
    repo.revoke_refresh_token.return_value = None

    # Override at the class level so the router's `UserRepository(db)` call
    # returns our mock.
    import api.src.routers.auth as auth_module

    auth_module.UserRepository = lambda db: repo  # type: ignore[assignment]
    return repo


# POST /auth/token


def test_login_returns_200(client):
    _mock_repo(client, user=_make_user("user"))
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "user@npec.nl", "password": "userpass", "grant_type": "password"},
    )
    assert response.status_code == 200


def test_login_response_schema(client):
    _mock_repo(client, user=_make_user("user"))
    data = client.post(
        "/api/v1/auth/token",
        data={"username": "user@npec.nl", "password": "userpass", "grant_type": "password"},
    ).json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600
    assert data["role"] == "user"


def test_login_admin_role(client):
    _mock_repo(client, user=_make_user("admin", _ADMIN_ID))
    data = client.post(
        "/api/v1/auth/token",
        data={"username": "admin@npec.nl", "password": "adminpass", "grant_type": "password"},
    ).json()
    assert data["role"] == "admin"


def test_login_wrong_password_returns_401(client):
    _mock_repo(client, user=_make_user("user"))
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "user@npec.nl", "password": "wrong", "grant_type": "password"},
    )
    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"


def test_login_unknown_user_returns_401(client):
    _mock_repo(client, user=None)
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "nobody@npec.nl", "password": "x", "grant_type": "password"},
    )
    assert response.status_code == 401


def test_login_missing_fields_returns_422(client):
    response = client.post("/api/v1/auth/token", data={})
    assert response.status_code == 422


# POST /auth/refresh


def test_refresh_returns_new_tokens(client):
    _mock_repo(client, user=_make_user(), rotate_result=("new-token", _USER_ID))
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": "old-token"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["refresh_token"] == "new-token"


def test_refresh_invalid_token_returns_401(client):
    _mock_repo(client, rotate_result=None)
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": "bad-token"})
    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"


# POST /auth/logout


def _get_token(client, role: str = "user") -> str:
    _mock_repo(client, user=_make_user(role))
    return client.post(
        "/api/v1/auth/token",
        data={"username": f"{role}@npec.nl", "password": f"{role}pass", "grant_type": "password"},
    ).json()["access_token"]


def test_logout_returns_200(client):
    token = _get_token(client)
    response = client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
        json={"refresh_token": "mock-refresh-token"},
    )
    assert response.status_code == 200
    assert "message" in response.json()


def test_logout_requires_auth(client):
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post("/api/v1/auth/logout")
    assert response.status_code == 401


# Protected endpoint access


def test_protected_endpoint_returns_401_without_token(client):
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get(f"/api/v1/predictions/{uuid4()}")
    assert response.status_code == 401


def test_admin_endpoint_returns_403_for_user_token(client):
    token = _get_token(client, role="user")
    response = client.delete(
        f"/api/v1/models/{uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    assert response.json()["error_code"] == "FORBIDDEN"
