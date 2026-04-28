"""Shared pytest fixtures.

Provides a TestClient with dependency overrides so tests are fully
isolated from any real storage or inference backend. Each test gets
a fresh in-memory repository, preventing state leakage.
"""

import io
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from db.session import get_session
from src.api.core.security import get_current_user, hash_password
from src.api.dependencies import get_inference_service, get_prediction_repo
from src.api.main import app
from src.api.repositories.memory import InMemoryPredictionRepository
from src.api.services.mock_inference import MockInferenceService

_USER_ID = uuid4()
_ADMIN_ID = uuid4()


def _make_user(role: str = "user", user_id=_USER_ID) -> MagicMock:
    """Return a mock User ORM object."""
    u = MagicMock()
    u.id = user_id
    u.sub = str(user_id)
    u.email = f"{role}@npec.nl"
    u.password_hash = hash_password(f"{role}pass")
    u.role = role
    u.is_active = True
    return u


def _make_mock_db() -> AsyncMock:
    """Return an AsyncSession mock whose execute/get return sensible defaults."""
    mock_db = AsyncMock()

    execute_result = MagicMock()
    execute_result.scalars.return_value.all.return_value = []
    execute_result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=execute_result)

    mock_db.get = AsyncMock(return_value=None)

    return mock_db


@pytest.fixture()
def client() -> TestClient:
    """Return a `TestClient` with fresh, isolated dependencies.

    Each call creates a new in-memory repository and mock inference service,
    so tests cannot affect one another through shared state.

    Returns:
        A configured `TestClient`.
    """
    repo = InMemoryPredictionRepository()

    app.dependency_overrides[get_session] = lambda: _make_mock_db()
    app.dependency_overrides[get_prediction_repo] = lambda: repo
    app.dependency_overrides[get_inference_service] = lambda: MockInferenceService()
    app.dependency_overrides[get_current_user] = lambda: _make_user()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def admin_client() -> TestClient:
    """Client authenticated as admin — for endpoints requiring admin role."""
    repo = InMemoryPredictionRepository()

    app.dependency_overrides[get_session] = lambda: _make_mock_db()
    app.dependency_overrides[get_prediction_repo] = lambda: repo
    app.dependency_overrides[get_inference_service] = lambda: MockInferenceService()
    app.dependency_overrides[get_current_user] = lambda: _make_user("admin", _ADMIN_ID)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _make_png(width: int = 512, height: int = 512) -> bytes:
    """Create a minimal valid PNG image as bytes.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Raw PNG bytes.
    """
    buf = io.BytesIO()
    Image.new("L", (width, height), color=128).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def valid_png_512() -> bytes:
    """Minimal valid 512×512 greyscale PNG."""
    return _make_png(512, 512)


@pytest.fixture()
def valid_png_1024() -> bytes:
    """Minimal valid 1024×1024 greyscale PNG."""
    return _make_png(1024, 1024)
