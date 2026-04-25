"""Shared pytest fixtures.

Provides a TestClient with dependency overrides so tests are fully
isolated from any real storage or inference backend. Each test gets
a fresh in-memory repository, preventing state leakage.
"""

import io
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from db.session import get_session
from src.api.dependencies import get_inference_service, get_prediction_repo
from src.api.main import app
from src.api.repositories.memory import InMemoryPredictionRepository
from src.api.services.mock_inference import MockInferenceService


@pytest.fixture()
def client() -> TestClient:
    """Return a `TestClient` with fresh, isolated dependencies.

    Each call creates a new in-memory repository and mock inference service,
    so tests cannot affect one another through shared state.

    Returns:
        A configured `TestClient`.
    """
    mock_db = AsyncMock()
    repo = InMemoryPredictionRepository()
    service = MockInferenceService()

    app.dependency_overrides[get_session] = lambda: mock_db
    app.dependency_overrides[get_prediction_repo] = lambda: repo
    app.dependency_overrides[get_inference_service] = lambda: service

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
