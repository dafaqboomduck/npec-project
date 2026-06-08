import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import io
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.src.core.security import get_current_user, hash_password
from api.src.db.session import get_session
from api.src.dependencies import get_inference_service
from api.src.main import app
from api.src.services.mock_inference import MockInferenceService

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
    # `Session.add` is sync; default AsyncMock would make it return an
    # un-awaited coroutine and trigger a RuntimeWarning.
    mock_db.add = MagicMock()

    return mock_db


@pytest.fixture()
def _isolate_external_io():
    """Stub every external I/O call the API may attempt during a request.

    The API talks to MinIO (artifact storage), an async DB, and an MLflow-backed
    inference pipeline. None of those are available in unit-test runs, so we
    replace the boundary functions with no-ops that return the values the
    routers expect. Individual tests can still `patch(...)` to override any of
    these for assertion purposes.

    Not autouse — repository/service unit tests that call these targets
    directly opt out by not requesting this fixture. The `client` /
    `admin_client` fixtures depend on it so all router tests get it.
    """
    patches = [
        # Lifespan startup: reconcile tries to open a real asyncpg connection
        # and can hang on Windows even with a bogus DATABASE_URL. No-op it at
        # both the import site (`api.src.main`) and the source module so any
        # TestClient starts up instantly regardless of how it's referenced.
        patch(
            "api.src.services.training.reconcile_orphaned_training_runs",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "api.src.main.reconcile_orphaned_training_runs",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "api.src.services.inference_queue.reconcile_orphaned_jobs",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "api.src.main.reconcile_orphaned_jobs",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "api.src.main.consume_forever",
            new_callable=AsyncMock,
            return_value=None,
        ),
        # Same defensive pair for the inference warmup — also part of lifespan.
        patch(
            "api.src.main.warmup_inference_service",
            new_callable=AsyncMock,
            return_value=None,
        ),
        # Artifact storage (MinIO) — called by inference router and library router
        patch("api.src.routers.inference.save_input_image", return_value=None),
        patch("api.src.routers.inference.save_segmentation_mask", return_value=None),
        patch("api.src.routers.inference.save_tip_photo", return_value=None),
        # DB writes for predictions — return the prediction unchanged
        patch(
            "api.src.repositories.database.DatabasePredictionRepository.save",
            new_callable=AsyncMock,
            side_effect=lambda prediction, **_: prediction,
        ),
        # Annotation status updates triggered by feedback/confirm flows
        patch(
            "api.src.repositories.annotations.AnnotationRepository.set_status",
            new_callable=AsyncMock,
            return_value=None,
        ),
        # Fire-and-forget retraining check — the routers call this via
        # asyncio.create_task. Patching the coroutine function alone is enough:
        # the resulting task is essentially `await AsyncMock()()` which
        # resolves immediately. (Patching `asyncio.create_task` itself is
        # tempting but would clobber the symbol globally and break TestClient.)
        patch(
            "api.src.routers.feedback._trigger_retraining_check",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "api.src.routers.inference._trigger_retraining_check",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ]
    started = [p.start() for p in patches]
    try:
        yield started
    finally:
        for p in patches:
            p.stop()


@pytest.fixture()
def client(_isolate_external_io) -> TestClient:
    """Return a `TestClient` with fresh, isolated dependencies.

    Each call creates a new in-memory repository and mock inference service,
    so tests cannot affect one another through shared state.

    Returns:
        A configured `TestClient`.
    """

    app.dependency_overrides[get_session] = lambda: _make_mock_db()
    app.dependency_overrides[get_inference_service] = lambda: MockInferenceService()
    app.dependency_overrides[get_current_user] = lambda: _make_user()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def admin_client(_isolate_external_io) -> TestClient:
    """Client authenticated as admin — for endpoints requiring admin role."""

    app.dependency_overrides[get_session] = lambda: _make_mock_db()
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
