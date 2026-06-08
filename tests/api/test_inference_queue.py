import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.services.inference_queue import (
    _claim_one,
    consume_forever,
    enqueue_batch,
    queue_counts,
    reconcile_orphaned_jobs,
)


@pytest.mark.asyncio
async def test_enqueue_batch(admin_client):
    from api.src.db.session import get_session
    from api.src.main import app

    db_gen = app.dependency_overrides[get_session]()
    db = await db_gen.__anext__() if hasattr(db_gen, "__anext__") else db_gen

    dataset_id = uuid.uuid4()
    req_by = uuid.uuid4()

    db.execute = AsyncMock()
    # Mock pending return
    from unittest.mock import MagicMock

    pending_mock = MagicMock()
    pending_mock.all.return_value = [(uuid.uuid4(), "s3://img1.png")]

    # Mock active return
    active_mock = MagicMock()
    active_mock.scalars.return_value.all.return_value = []

    db.execute.side_effect = [pending_mock, active_mock]

    with patch("api.src.services.broadcaster.inference_broadcaster.broadcast", new_callable=AsyncMock) as bcast:
        res = await enqueue_batch(db, dataset_id, req_by)
        assert res == 1
        bcast.assert_awaited_once_with("queue_updated")


@pytest.mark.asyncio
async def test_reconcile_orphaned_jobs():
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()

    async def session_factory():
        yield mock_db

    class FactoryProxy:
        def __call__(self):
            class Context:
                async def __aenter__(self):
                    return mock_db

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return Context()

    await reconcile_orphaned_jobs(FactoryProxy())
    assert mock_db.execute.call_count == 2
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_counts():
    from unittest.mock import MagicMock

    mock_db = AsyncMock()
    exec_mock = MagicMock()
    exec_mock.all.return_value = [("running", 2), ("queued", 5)]
    mock_db.execute.return_value = exec_mock

    counts = await queue_counts(mock_db)
    assert counts["running"] == 2
    assert counts["queued"] == 5
    assert counts["failed"] == 0
    assert counts["done"] == 0


@pytest.mark.asyncio
async def test_claim_one():
    from unittest.mock import MagicMock

    mock_db = AsyncMock()
    exec_mock = MagicMock()

    fake_job = MagicMock()
    fake_job.status = "queued"
    fake_job.attempts = 0
    exec_mock.scalar_one_or_none.return_value = fake_job
    mock_db.execute.return_value = exec_mock

    res = await _claim_one(mock_db)
    assert res is fake_job
    assert fake_job.status == "running"
    assert fake_job.attempts == 1
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_forever_stop_immediately():
    stop = asyncio.Event()
    stop.set()

    class FactoryProxy:
        def __call__(self):
            class Context:
                async def __aenter__(self):
                    return AsyncMock()

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return Context()

    with patch("api.src.services.inference_queue._POLL_INTERVAL_S", 0.001):
        await consume_forever(FactoryProxy(), stop)


@pytest.mark.asyncio
async def test_consume_forever_process_job():
    stop = asyncio.Event()

    fake_job = MagicMock()
    fake_job.id = uuid.uuid4()
    fake_job.image_uri = "s3://images/img1.png"
    fake_job.annotation_id = uuid.uuid4()
    fake_job.requested_by = uuid.uuid4()

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=MagicMock())
    mock_db.commit = AsyncMock()

    class FactoryProxy:
        def __call__(self):
            class Context:
                async def __aenter__(self):
                    return mock_db

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return Context()

    claim_count = 0

    async def mock_claim(db):
        nonlocal claim_count
        claim_count += 1
        if claim_count == 1:
            return fake_job
        else:
            stop.set()
            return None

    mock_storage = MagicMock()
    mock_storage.download.return_value = b"fake_bytes"

    mock_inference_service = AsyncMock()
    pred_resp = MagicMock()
    pred_resp.prediction_id = uuid.uuid4()
    mock_inference_service.run.return_value = (pred_resp, MagicMock())

    class FakeGate:
        import contextlib

        @contextlib.asynccontextmanager
        async def slot(self):
            yield

    with (
        patch("api.src.services.inference_queue._claim_one", side_effect=mock_claim),
        patch("api.src.dependencies.get_storage", return_value=mock_storage),
        patch("api.src.dependencies.get_inference_service", return_value=mock_inference_service),
        patch("api.src.services.inference_gate.get_inference_gate", return_value=FakeGate()),
        patch("api.src.services.artifact_storage.save_segmentation_mask", return_value="s3://mask"),
        patch("api.src.services.artifact_storage.save_tip_photo", return_value="s3://tip"),
        patch("api.src.repositories.database.DatabasePredictionRepository.save", new_callable=AsyncMock) as mock_save,
        patch("api.src.services.broadcaster.inference_broadcaster.broadcast", new_callable=AsyncMock) as mock_bcast,
        patch("api.src.services.inference_queue._POLL_INTERVAL_S", 0.001),
    ):

        await consume_forever(FactoryProxy(), stop)

        assert claim_count >= 1
        mock_save.assert_called_once()
        mock_db.commit.assert_awaited()
        mock_bcast.assert_awaited_with("queue_updated")


@pytest.mark.asyncio
async def test_consume_forever_job_exception():
    stop = asyncio.Event()

    fake_job = MagicMock()
    fake_job.id = uuid.uuid4()
    fake_job.attempts = 1
    fake_job.image_uri = "s3://images/img1.png"

    mock_db = AsyncMock()
    mock_db.get.return_value = fake_job

    class FactoryProxy:
        def __call__(self):
            class Context:
                async def __aenter__(self):
                    return mock_db

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return Context()

    claim_count = 0

    async def mock_claim(db):
        nonlocal claim_count
        claim_count += 1
        if claim_count == 1:
            return fake_job
        else:
            stop.set()
            return None

    mock_storage = MagicMock()
    mock_storage.download.side_effect = Exception("Download failed")

    with (
        patch("api.src.services.inference_queue._claim_one", side_effect=mock_claim),
        patch("api.src.dependencies.get_storage", return_value=mock_storage),
        patch("api.src.services.inference_queue._POLL_INTERVAL_S", 0.001),
        patch("api.src.services.broadcaster.inference_broadcaster.broadcast", new_callable=AsyncMock) as mock_bcast,
    ):

        await consume_forever(FactoryProxy(), stop)

        mock_db.rollback.assert_awaited_once()
        mock_db.commit.assert_awaited()
        mock_bcast.assert_awaited_with("queue_updated")
