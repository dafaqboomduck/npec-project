"""Unit tests for all repository classes.

Each test mocks the AsyncSession so we exercise the repository's own
logic — filter chains, row → schema conversions, ownership checks,
error paths — without touching Postgres.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.src.repositories.annotations import AnnotationRepository, _row_to_schema
from api.src.repositories.dashboard import DashboardRepository
from api.src.repositories.database import DatabasePredictionRepository, NoDeployedModelError
from api.src.repositories.datasets import DatasetRepository
from api.src.repositories.feedback import FeedbackRepository
from api.src.repositories.memory import InMemoryPredictionRepository
from api.src.repositories.models import ModelRepository
from api.src.repositories.operational import OperationalMetricsRepository
from api.src.repositories.overview import OverviewRepository, _drift_status
from api.src.repositories.training import TrainingRunRepository
from api.src.repositories.users import UserRepository, _hash_token
from api.src.schemas.auth import UserRole
from api.src.schemas.library import AnnotationStatus, ModelStatus
from api.src.schemas.overview import DriftStatus
from api.src.schemas.training import TrainingRunStatus

# Helpers


def _execute_result(
    *,
    scalar_one_or_none=None,
    scalars_all: list | None = None,
    all_rows: list | None = None,
    first_row=None,
    one_row=None,
):
    """Build a mock result object configurable for each Result API call."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_one_or_none
    result.scalars.return_value.all.return_value = scalars_all or []
    result.all.return_value = all_rows or []
    result.first.return_value = first_row
    result.one.return_value = one_row
    result.scalar_one.return_value = scalar_one_or_none
    return result


def _make_db(execute_results: list | None = None, get_results: list | None = None) -> AsyncMock:
    """Return an AsyncSession mock that returns execute/get results in order."""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    db.delete = AsyncMock()

    if execute_results is not None:
        db.execute = AsyncMock(side_effect=list(execute_results))
    else:
        db.execute = AsyncMock(return_value=_execute_result())

    if get_results is not None:
        db.get = AsyncMock(side_effect=list(get_results))
    else:
        db.get = AsyncMock(return_value=None)

    return db


def _orm_dataset(**overrides):
    """Build a fake Dataset ORM row."""
    creator = SimpleNamespace(email="owner@example.com")
    base = dict(
        id=uuid.uuid4(),
        name="My Dataset",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
        count=0,
        description="desc",
        species="arabidopsis",
        creator=creator,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _orm_model_version(**overrides):
    base = dict(
        id=uuid.uuid4(),
        name="unet",
        version="v1.0.0",
        status=ModelStatus.REGISTERED.value,
        created_at=datetime.now(timezone.utc),
        description="d",
        f1_score=0.9,
        precision=0.9,
        recall=0.9,
        inference_latency_ms=100,
        artifact_uri="s3://x",
        deployed_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _orm_annotation(**overrides):
    base = dict(
        id=uuid.uuid4(),
        prediction_id=None,
        dataset_id=None,
        status=AnnotationStatus.PENDING.value,
        image_uri=None,
        mask_uri=None,
        tip_photo_uri=None,
        species=None,
        age_days=None,
        treatment=None,
        growth_stage=None,
        created_by=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        image_hash=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _orm_training_run(**overrides):
    base = dict(
        id=uuid.uuid4(),
        status=TrainingRunStatus.RUNNING.value,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        final_f1=None,
        hyperparameters={"lr": 0.001},
        model_version_id=None,
        azureml_job_name=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_storage():
    s = MagicMock()
    s.sign = lambda uri: f"signed::{uri}"
    return s


# FeedbackRepository


class TestFeedbackRepository:
    async def test_prediction_exists_true(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=uuid.uuid4())])
        repo = FeedbackRepository(db)
        assert await repo.prediction_exists(uuid.uuid4()) is True

    async def test_prediction_exists_false(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        repo = FeedbackRepository(db)
        assert await repo.prediction_exists(uuid.uuid4()) is False

    async def test_create_adds_row_and_commits(self) -> None:
        db = _make_db()
        repo = FeedbackRepository(db)
        await repo.create(uuid.uuid4(), uuid.uuid4(), "correct", "good!")
        db.add.assert_called_once()
        db.commit.assert_awaited_once()

    async def test_get_latest_for_prediction_returns_row(self) -> None:
        fb = SimpleNamespace(prediction_id=uuid.uuid4())
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=fb)])
        repo = FeedbackRepository(db)
        assert await repo.get_latest_for_prediction(fb.prediction_id) is fb

    async def test_get_latest_returns_none_when_empty(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        repo = FeedbackRepository(db)
        assert await repo.get_latest_for_prediction(uuid.uuid4()) is None


# UserRepository
# `asyncio_mode = "auto"` in pyproject.toml already discovers `async def` tests —
# no module-level pytestmark needed (and it'd mis-mark the sync test below).


class TestUserRepository:
    async def test_get_by_email_returns_user(self) -> None:
        user = SimpleNamespace(id=uuid.uuid4(), email="x@y.com")
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=user)])
        repo = UserRepository(db)
        assert await repo.get_by_email("x@y.com") is user

    async def test_get_by_email_returns_none(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        repo = UserRepository(db)
        assert await repo.get_by_email("missing@y.com") is None

    async def test_get_by_id_delegates_to_session_get(self) -> None:
        user = SimpleNamespace(id=uuid.uuid4())
        db = _make_db(get_results=[user])
        repo = UserRepository(db)
        assert await repo.get_by_id(user.id) is user

    async def test_issue_refresh_token_persists_hash_and_returns_raw(self) -> None:
        db = _make_db()
        repo = UserRepository(db)
        raw = await repo.issue_refresh_token(uuid.uuid4())
        assert isinstance(raw, str) and len(raw) > 20
        db.add.assert_called_once()
        db.commit.assert_awaited_once()
        # Persisted token is hashed, not raw
        stored = db.add.call_args.args[0]
        assert stored.token_hash == _hash_token(raw)

    async def test_rotate_refresh_token_returns_none_for_unknown(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        repo = UserRepository(db)
        assert await repo.rotate_refresh_token("bogus") is None

    async def test_rotate_refresh_token_rotates_and_revokes(self) -> None:
        old = SimpleNamespace(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            token_hash="old-hash",
            revoked_at=None,
            replaced_by=None,
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )

        async def flush_side_effect():
            # Mirror real flush: ensure new record has an id assigned
            stored = db.add.call_args.args[0]
            stored.id = uuid.uuid4()

        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=old)])
        db.flush = AsyncMock(side_effect=flush_side_effect)
        repo = UserRepository(db)
        result = await repo.rotate_refresh_token("anything")
        assert result is not None
        new_raw, user_id = result
        assert user_id == old.user_id
        # The new raw token is fresh and distinct from input
        assert new_raw != "anything"
        # Old row got revoked and linked
        assert old.revoked_at is not None
        assert old.replaced_by is not None

    async def test_revoke_refresh_token_executes_update(self) -> None:
        db = _make_db()
        repo = UserRepository(db)
        await repo.revoke_refresh_token("anything")
        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()


# ModelRepository


class TestModelRepository:
    async def test_create_raises_when_version_exists(self) -> None:
        existing = _orm_model_version(version="v1")
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=existing)])
        repo = ModelRepository(db)
        with pytest.raises(ValueError, match="already exists"):
            await repo.create(version="v1", name="n", artifact_uri="s3://x")

    async def test_create_inserts_and_returns_detail(self) -> None:
        async def refresh(row):
            row.id = uuid.uuid4()
            row.created_at = datetime.now(timezone.utc)

        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        db.refresh = AsyncMock(side_effect=refresh)
        repo = ModelRepository(db)
        detail = await repo.create(
            version="v2",
            name="unet",
            artifact_uri="s3://m",
            description="d",
            f1_score=0.95,
        )
        assert detail.name == "unet"
        assert detail.artifact_uri == "s3://m"  # admin fields included on create
        db.commit.assert_awaited()

    async def test_list_returns_summaries(self) -> None:
        rows = [_orm_model_version(name="m1"), _orm_model_version(name="m2")]
        db = _make_db(execute_results=[_execute_result(scalars_all=rows)])
        repo = ModelRepository(db)
        out = await repo.list()
        assert [m.name for m in out] == ["m1", "m2"]

    async def test_get_returns_none_when_missing(self) -> None:
        db = _make_db(get_results=[None])
        assert await ModelRepository(db).get(uuid.uuid4()) is None

    async def test_get_includes_admin_fields_when_requested(self) -> None:
        row = _orm_model_version(artifact_uri="s3://x", deployed_at=datetime.now(timezone.utc))
        db = _make_db(get_results=[row])
        detail = await ModelRepository(db).get(row.id, include_admin_fields=True)
        assert detail.artifact_uri == "s3://x"
        assert detail.deployed_at is not None

    async def test_get_hides_admin_fields_by_default(self) -> None:
        row = _orm_model_version(artifact_uri="s3://x", deployed_at=datetime.now(timezone.utc))
        db = _make_db(get_results=[row])
        detail = await ModelRepository(db).get(row.id)
        assert detail.artifact_uri is None
        assert detail.deployed_at is None

    async def test_get_deployed_artifact_uri(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none="s3://deployed")])
        assert await ModelRepository(db).get_deployed_artifact_uri() == "s3://deployed"

    async def test_get_deployed_f1(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=0.87)])
        assert await ModelRepository(db).get_deployed_f1() == 0.87

    async def test_register_and_promote_no_deployed_registers(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        db.refresh = AsyncMock()
        model_id, registered = await ModelRepository(db).register_and_promote_if_better(
            version="v9",
            name="unet",
            artifact_uri="s3://m",
            final_f1=0.9,
        )
        assert registered is True
        assert model_id is not None
        # Two adds: ModelVersion + MonitoringMetric
        assert db.add.call_count == 2

    async def test_register_and_promote_not_better_skips(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=0.95)])
        model_id, registered = await ModelRepository(db).register_and_promote_if_better(
            version="v10",
            name="unet",
            artifact_uri="s3://m",
            final_f1=0.90,
        )
        assert registered is False
        assert model_id is None
        db.add.assert_not_called()

    async def test_register_and_promote_significantly_better(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=0.80)])
        db.refresh = AsyncMock()
        model_id, registered = await ModelRepository(db).register_and_promote_if_better(
            version="v11",
            name="unet",
            artifact_uri="s3://m",
            final_f1=0.95,
            min_improvement=0.01,
        )
        assert registered is True
        assert model_id is not None

    async def test_exists_true(self) -> None:
        db = _make_db(get_results=[_orm_model_version()])
        assert await ModelRepository(db).exists(uuid.uuid4()) is True

    async def test_exists_false(self) -> None:
        db = _make_db(get_results=[None])
        assert await ModelRepository(db).exists(uuid.uuid4()) is False

    async def test_delete_missing_returns_false(self) -> None:
        db = _make_db(get_results=[None])
        assert await ModelRepository(db).delete(uuid.uuid4()) is False

    async def test_delete_existing_returns_true(self) -> None:
        row = _orm_model_version()
        db = _make_db(get_results=[row])
        assert await ModelRepository(db).delete(row.id) is True
        db.delete.assert_awaited_once_with(row)
        db.commit.assert_awaited_once()


# DatasetRepository


class TestDatasetRepository:
    async def test_list_no_filters(self) -> None:
        rows = [_orm_dataset(name="A"), _orm_dataset(name="B")]
        db = _make_db(execute_results=[_execute_result(scalars_all=rows)])
        out = await DatasetRepository(db).list(species=None)
        assert [d.name for d in out] == ["A", "B"]
        assert out[0].owner_email == "owner@example.com"

    async def test_list_filters_for_non_admin(self) -> None:
        rows = [_orm_dataset()]
        db = _make_db(execute_results=[_execute_result(scalars_all=rows)])
        out = await DatasetRepository(db).list(
            species="arabidopsis",
            growth_stage="seedling",
            role=UserRole.USER,
            requesting_user_id=uuid.uuid4(),
        )
        assert len(out) == 1

    async def test_increment_count_raises_when_missing(self) -> None:
        db = _make_db(get_results=[None])
        with pytest.raises(ValueError, match="not found"):
            await DatasetRepository(db).increment_count(uuid.uuid4())

    async def test_increment_count_increments_in_place(self) -> None:
        row = _orm_dataset(count=5)
        db = _make_db(get_results=[row])
        await DatasetRepository(db).increment_count(row.id)
        assert row.count == 6

    async def test_get_true_count(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=42)])
        assert await DatasetRepository(db).get_true_count(uuid.uuid4()) == 42

    async def test_get_true_count_handles_none(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        assert await DatasetRepository(db).get_true_count(uuid.uuid4()) == 0

    async def test_get_returns_none_when_missing(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        assert await DatasetRepository(db).get(uuid.uuid4()) is None

    async def test_get_returns_detail(self) -> None:
        row = _orm_dataset()
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=row),
                _execute_result(scalar_one_or_none=7),
            ]
        )
        detail = await DatasetRepository(db).get(row.id)
        assert detail is not None
        assert detail.count == 7
        assert detail.owner_email == "owner@example.com"

    async def test_get_admin_skips_ownership_filter(self) -> None:
        row = _orm_dataset()
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=row),
                _execute_result(scalar_one_or_none=0),
            ]
        )
        out = await DatasetRepository(db).get(
            row.id,
            role=UserRole.ADMIN,
            requesting_user_id=uuid.uuid4(),
        )
        assert out is not None

    async def test_create_returns_detail(self) -> None:
        async def refresh(row):
            row.id = uuid.uuid4()
            row.created_at = datetime.now(timezone.utc)
            row.modified_at = datetime.now(timezone.utc)
            row.count = 0

        created_row = _orm_dataset()
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=created_row)])
        db.refresh = AsyncMock(side_effect=refresh)
        detail = await DatasetRepository(db).create(
            name="X", description="d", species="s", growth_stage="seedling", created_by=uuid.uuid4()
        )
        assert detail.name == created_row.name

    async def test_delete_missing_returns_false(self) -> None:
        db = _make_db(get_results=[None])
        assert await DatasetRepository(db).delete(uuid.uuid4()) is False

    async def test_delete_existing(self) -> None:
        row = _orm_dataset()
        db = _make_db(get_results=[row])
        assert await DatasetRepository(db).delete(row.id) is True
        db.delete.assert_awaited_once_with(row)


# AnnotationRepository


class TestAnnotationRepository:
    async def test_row_to_schema_signs_uris(self) -> None:
        storage = _fake_storage()
        row = _orm_annotation(image_uri="i", mask_uri="m", tip_photo_uri="t")
        s = _row_to_schema(row, storage)
        assert s.image_uri == "signed::i"
        assert s.mask_url == "signed::m"
        assert s.tip_photo_url == "signed::t"

    async def test_list_default(self) -> None:
        rows = [_orm_annotation()]
        db = _make_db(execute_results=[_execute_result(scalars_all=rows)])
        out = await AnnotationRepository(db, _fake_storage()).list()
        assert len(out) == 1

    async def test_list_with_all_filters(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalars_all=[])])
        out = await AnnotationRepository(db, _fake_storage()).list(
            species="arabidopsis",
            growth_stage="seedling",
            status="annotated",
            dataset_id=uuid.uuid4(),
            page=2,
            page_size=10,
            role=UserRole.USER,
            requesting_user_id=uuid.uuid4(),
            order="asc",
        )
        assert out == []

    async def test_list_pending_for_dataset(self) -> None:
        rows = [_orm_annotation()]
        db = _make_db(execute_results=[_execute_result(scalars_all=rows)])
        out = await AnnotationRepository(db, _fake_storage()).list_pending_for_dataset(uuid.uuid4())
        assert out == rows

    async def test_get_returns_none_when_missing(self) -> None:
        db = _make_db(get_results=[None])
        assert await AnnotationRepository(db, _fake_storage()).get(uuid.uuid4()) is None

    async def test_get_returns_annotation_for_admin(self) -> None:
        row = _orm_annotation(prediction_id=None, created_by=uuid.uuid4())
        db = _make_db(get_results=[row], execute_results=[_execute_result(scalar_one_or_none=None)])
        out = await AnnotationRepository(db, _fake_storage()).get(
            row.id, role=UserRole.ADMIN, requesting_user_id=uuid.uuid4()
        )
        assert out is not None

    async def test_get_denies_when_not_owner(self) -> None:
        owner_id = uuid.uuid4()
        other_id = uuid.uuid4()
        row = _orm_annotation(created_by=owner_id)
        db = _make_db(get_results=[row])
        out = await AnnotationRepository(db, _fake_storage()).get(
            row.id, role=UserRole.USER, requesting_user_id=other_id
        )
        assert out is None

    async def test_get_allows_owner(self) -> None:
        owner_id = uuid.uuid4()
        row = _orm_annotation(created_by=owner_id)
        # The `get` flow runs: ownership check, then owner-email lookup via prediction. With prediction_id=None,
        # only the creator lookup runs.
        user_row = SimpleNamespace(email="owner@x.com")
        db = _make_db(get_results=[row], execute_results=[_execute_result(scalar_one_or_none=user_row)])
        out = await AnnotationRepository(db, _fake_storage()).get(
            row.id, role=UserRole.USER, requesting_user_id=owner_id
        )
        assert out is not None
        assert out.owner_email == "owner@x.com"

    async def test_delete_missing(self) -> None:
        db = _make_db(get_results=[None])
        assert await AnnotationRepository(db, _fake_storage()).delete(uuid.uuid4()) is False

    async def test_delete_existing(self) -> None:
        row = _orm_annotation()
        db = _make_db(get_results=[row])
        assert await AnnotationRepository(db, _fake_storage()).delete(row.id) is True

    async def test_confirm_missing_returns_none(self) -> None:
        db = _make_db(get_results=[None])
        assert await AnnotationRepository(db, _fake_storage()).confirm(uuid.uuid4(), "url") is None

    async def test_confirm_updates_row(self) -> None:
        row = _orm_annotation()
        db = _make_db(get_results=[row])
        out = await AnnotationRepository(db, _fake_storage()).confirm(row.id, "s3://mask")
        assert out is not None
        assert row.mask_uri == "s3://mask"
        assert row.status == AnnotationStatus.ANNOTATED

    async def test_create_from_prediction_basic(self) -> None:
        db = _make_db()
        new_id = await AnnotationRepository(db, _fake_storage()).create_from_prediction(
            prediction_id=uuid.uuid4(),
            mask_uri=None,
            tip_photo_uri=None,
            input_hash=None,
            image_uri=None,
            species=None,
            age_days=None,
            treatment=None,
        )
        assert isinstance(new_id, uuid.UUID)
        db.add.assert_called_once()

    async def test_create_from_prediction_dedup_within_dataset(self) -> None:
        existing = _orm_annotation()
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=existing)])
        with pytest.raises(ValueError, match="identical image"):
            await AnnotationRepository(db, _fake_storage()).create_from_prediction(
                prediction_id=uuid.uuid4(),
                mask_uri=None,
                tip_photo_uri=None,
                input_hash="abc",
                image_uri=None,
                species=None,
                age_days=None,
                treatment=None,
                dataset_id=uuid.uuid4(),
            )

    async def test_create_pending(self) -> None:
        db = _make_db()
        out = await AnnotationRepository(db, _fake_storage()).create_pending(
            dataset_id=uuid.uuid4(),
            image_uri="s3://img",
            image_hash="h",
            created_by=uuid.uuid4(),
        )
        assert out.status == AnnotationStatus.PENDING
        db.add.assert_called_once()

    async def test_update_metadata_missing(self) -> None:
        db = _make_db(get_results=[None])
        out = await AnnotationRepository(db, _fake_storage()).update_metadata(
            uuid.uuid4(), species=None, age_days=None, treatment=None, growth_stage=None
        )
        assert out is None

    async def test_update_metadata_partial(self) -> None:
        row = _orm_annotation()
        db = _make_db(get_results=[row])
        out = await AnnotationRepository(db, _fake_storage()).update_metadata(
            row.id, species="x", age_days=5, treatment="t", growth_stage="seedling"
        )
        assert out is not None
        assert row.species == "x"
        assert row.age_days == 5
        assert row.treatment == "t"
        assert row.growth_stage == "seedling"

    async def test_set_status_missing_returns_none(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        out = await AnnotationRepository(db, _fake_storage()).set_status(uuid.uuid4(), AnnotationStatus.REVIEW_REQUIRED)
        assert out is None

    async def test_set_status_updates_existing(self) -> None:
        row = _orm_annotation()
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=row)])
        out = await AnnotationRepository(db, _fake_storage()).set_status(uuid.uuid4(), AnnotationStatus.ANNOTATED)
        assert out is not None
        assert row.status == AnnotationStatus.ANNOTATED

    async def test_set_status_creates_when_missing_and_pred_exists(self) -> None:
        pred = SimpleNamespace(
            id=uuid.uuid4(),
            input_uri="s3://img",
            segmentation_mask_uri=None,
            tip_photo_uri=None,
            input_hash="h",
            user_id=uuid.uuid4(),
        )
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=None),
                _execute_result(scalar_one_or_none=pred),
            ]
        )
        out = await AnnotationRepository(db, _fake_storage()).set_status(
            pred.id, AnnotationStatus.REVIEW_REQUIRED, create_if_missing=True
        )
        assert out is not None
        db.add.assert_called_once()


# DashboardRepository


class TestDashboardRepository:
    async def test_get_metrics_returns_zeros_when_no_deployed(self) -> None:
        db = _make_db(execute_results=[_execute_result(first_row=None)])
        out = await DashboardRepository(db).get_metrics()
        assert out.f1_score == 0.0
        assert out.throughput_img_per_s == 0.0

    async def test_get_metrics_with_metrics(self) -> None:
        deployed = (uuid.uuid4(), "v1", 0.5)
        db = _make_db(
            execute_results=[
                _execute_result(first_row=deployed),
                _execute_result(scalar_one_or_none=0.92),
                _execute_result(scalar_one_or_none=12.5),
            ]
        )
        out = await DashboardRepository(db).get_metrics()
        assert out.f1_score == pytest.approx(0.92)
        assert out.throughput_img_per_s == 12.5

    async def test_get_metrics_falls_back_to_model_f1(self) -> None:
        deployed = (uuid.uuid4(), "v1", 0.88)
        db = _make_db(
            execute_results=[
                _execute_result(first_row=deployed),
                _execute_result(scalar_one_or_none=None),
                _execute_result(scalar_one_or_none=None),
            ]
        )
        out = await DashboardRepository(db).get_metrics()
        assert out.f1_score == pytest.approx(0.88)

    async def test_get_metrics_with_run_id(self) -> None:
        deployed = (uuid.uuid4(), "v1", 0.7)
        db = _make_db(
            execute_results=[
                _execute_result(first_row=deployed),
                _execute_result(scalar_one_or_none=0.93),
                _execute_result(scalar_one_or_none=8.0),
            ]
        )
        out = await DashboardRepository(db).get_metrics(run_id=uuid.uuid4())
        assert out.f1_score == pytest.approx(0.93)

    async def test_get_f1_history_no_deployed(self) -> None:
        db = _make_db(execute_results=[_execute_result(first_row=None)])
        out = await DashboardRepository(db).get_f1_history()
        assert out.data_points == []
        assert out.model_version == "unknown"

    async def test_get_f1_history_filters_rows_without_epoch(self) -> None:
        deployed = (uuid.uuid4(), "v1", 0.5)
        rows = [(0.91, {"epoch": 2}), (0.80, {}), (0.85, {"epoch": 1})]
        db = _make_db(
            execute_results=[
                _execute_result(first_row=deployed),
                _execute_result(all_rows=rows),
            ]
        )
        out = await DashboardRepository(db).get_f1_history(run_id=uuid.uuid4())
        # Sorted by epoch ascending; the epoch-less row is dropped.
        assert [p.epoch for p in out.data_points] == [1, 2]

    async def test_get_data_distribution(self) -> None:
        rows = [("arabidopsis", 5), ("maize", 3)]
        db = _make_db(execute_results=[_execute_result(all_rows=rows)])
        out = await DashboardRepository(db).get_data_distribution()
        assert [s.species for s in out.species_counts] == ["Arabidopsis", "Maize"]
        assert [s.count for s in out.species_counts] == [5, 3]


# OverviewRepository


class TestOverviewRepository:
    def test_drift_status_thresholds(self) -> None:
        assert _drift_status(0.0) == DriftStatus.NONE
        assert _drift_status(0.15) == DriftStatus.LOW
        assert _drift_status(0.35) == DriftStatus.MEDIUM
        assert _drift_status(0.95) == DriftStatus.HIGH

    async def test_get_status_handles_empty(self) -> None:
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=0),
                _execute_result(scalar_one_or_none=0),
            ]
        )
        out = await OverviewRepository(db).get_status()
        assert out.total_experiments == 0
        assert out.active_model_count == 0
        assert out.quota_percent == 0

    async def test_get_status_with_data(self) -> None:
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=2_000_000),  # many experiments
                _execute_result(scalar_one_or_none=2),
            ]
        )
        out = await OverviewRepository(db).get_status()
        assert out.total_experiments == 2_000_000
        assert out.active_model_count == 2
        assert 0 <= out.quota_percent <= 100

    async def test_get_model_health_no_deployed(self) -> None:
        db = _make_db(execute_results=[_execute_result(first_row=None)])
        out = await OverviewRepository(db).get_model_health()
        assert out.f1_score == 0.0
        assert out.drift_status == DriftStatus.NONE

    async def test_get_model_health_full(self) -> None:
        deployed = (uuid.uuid4(), 0.92)
        db = _make_db(
            execute_results=[
                _execute_result(first_row=deployed),
                _execute_result(scalar_one_or_none=42),
                _execute_result(scalar_one_or_none=0.20),
            ]
        )
        out = await OverviewRepository(db).get_model_health()
        assert out.f1_score == pytest.approx(0.92)
        assert out.latency_ms == 42
        assert out.drift_status == DriftStatus.LOW

    async def test_get_activity_paginates(self) -> None:
        now = datetime.now(timezone.utc)
        pred_rows = [SimpleNamespace(id=uuid.uuid4(), input_filename="f", created_at=now)]
        deploy_rows = [SimpleNamespace(id=uuid.uuid4(), name="m", deployed_at=now - timedelta(hours=1))]
        annot_rows = [SimpleNamespace(id=uuid.uuid4(), updated_at=now - timedelta(hours=2))]
        db = _make_db(
            execute_results=[
                _execute_result(all_rows=pred_rows),
                _execute_result(all_rows=deploy_rows),
                _execute_result(all_rows=annot_rows),
            ]
        )
        out = await OverviewRepository(db).get_activity(page=1, page_size=10)
        assert out.total == 3
        assert out.page == 1
        # Newest first
        assert out.items[0].timestamp >= out.items[-1].timestamp


# OperationalMetricsRepository


class TestOperationalMetricsRepository:
    async def test_returns_zero_defaults_when_audit_empty(self) -> None:
        row = SimpleNamespace(total=0, failed=0, avg_ms=0.0, p50=0.0, p95=0.0, plants=0)
        db = _make_db(execute_results=[_execute_result(one_row=row)])
        out = await OperationalMetricsRepository(db).get_operational_metrics()
        assert out.total_requests == 0
        assert out.error_rate_pct == 0.0

    async def test_computes_metrics_from_audit(self) -> None:
        row = SimpleNamespace(total=100, failed=5, avg_ms=120.5, p50=100.0, p95=200.0, plants=300)
        per_model = [
            SimpleNamespace(model_name="unet-v1", requests=60, failed=2, avg_ms=100.0, plants=200),
            SimpleNamespace(model_name=None, requests=40, failed=3, avg_ms=150.0, plants=100),
        ]
        db = _make_db(
            execute_results=[
                _execute_result(one_row=row),
                _execute_result(all_rows=per_model),
            ]
        )
        out = await OperationalMetricsRepository(db).get_operational_metrics(
            since=datetime.now(timezone.utc) - timedelta(days=1)
        )
        assert out.total_requests == 100
        assert out.failed_requests == 5
        assert out.error_rate_pct == 5.0
        assert out.total_plants_detected == 300
        assert out.avg_plants_per_request == 3.0
        assert len(out.per_model) == 2
        # Unknown name fallback
        assert any(m.model_name == "unknown" for m in out.per_model)


# TrainingRunRepository


class TestTrainingRunRepository:
    async def test_create_inserts_and_flushes(self) -> None:
        db = _make_db()
        row = await TrainingRunRepository(db).create({"lr": 0.01})
        db.add.assert_called_once()
        db.flush.assert_awaited_once()
        assert row.status == TrainingRunStatus.RUNNING.value

    async def test_set_job_name_updates_when_present(self) -> None:
        row = _orm_training_run()
        db = _make_db(get_results=[row])
        await TrainingRunRepository(db).set_job_name(row.id, "job-1")
        assert row.azureml_job_name == "job-1"

    async def test_set_job_name_noop_when_missing(self) -> None:
        db = _make_db(get_results=[None])
        await TrainingRunRepository(db).set_job_name(uuid.uuid4(), "job-1")
        db.commit.assert_not_awaited()

    async def test_mark_completed(self) -> None:
        row = _orm_training_run()
        db = _make_db(get_results=[row])
        await TrainingRunRepository(db).mark_completed(row.id, final_f1=0.91, model_version_id=uuid.uuid4())
        assert row.status == TrainingRunStatus.COMPLETED.value
        assert row.final_f1 == 0.91

    async def test_mark_completed_missing_logs_only(self) -> None:
        db = _make_db(get_results=[None])
        await TrainingRunRepository(db).mark_completed(uuid.uuid4())
        db.commit.assert_not_awaited()

    async def test_mark_failed(self) -> None:
        row = _orm_training_run()
        db = _make_db(get_results=[row])
        await TrainingRunRepository(db).mark_failed(row.id, "boom")
        assert row.status == TrainingRunStatus.FAILED.value

    async def test_mark_failed_missing(self) -> None:
        db = _make_db(get_results=[None])
        await TrainingRunRepository(db).mark_failed(uuid.uuid4())
        db.commit.assert_not_awaited()

    async def test_mark_stopped(self) -> None:
        row = _orm_training_run()
        db = _make_db(get_results=[row])
        await TrainingRunRepository(db).mark_stopped(row.id)
        assert row.status == TrainingRunStatus.STOPPED.value

    async def test_mark_stopped_missing(self) -> None:
        db = _make_db(get_results=[None])
        await TrainingRunRepository(db).mark_stopped(uuid.uuid4())
        db.commit.assert_not_awaited()

    async def test_get_raw_delegates(self) -> None:
        row = _orm_training_run()
        db = _make_db(get_results=[row])
        assert await TrainingRunRepository(db).get_raw(row.id) is row

    async def test_list(self) -> None:
        rows = [_orm_training_run() for _ in range(3)]
        db = _make_db(execute_results=[_execute_result(scalars_all=rows)])
        out = await TrainingRunRepository(db).list()
        assert len(out) == 3

    async def test_get_returns_none_when_missing(self) -> None:
        db = _make_db(get_results=[None])
        assert await TrainingRunRepository(db).get(uuid.uuid4()) is None

    async def test_get_returns_detail(self) -> None:
        row = _orm_training_run()
        db = _make_db(get_results=[row])
        out = await TrainingRunRepository(db).get(row.id)
        assert out is not None
        assert out.run_id == row.id


# InMemoryPredictionRepository


class TestInMemoryPredictionRepository:
    async def test_seed_prediction_retrievable(self) -> None:
        repo = InMemoryPredictionRepository()
        seeded = uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")
        out = await repo.get(seeded)
        assert out is not None

    async def test_save_and_get(self) -> None:
        from api.src.schemas.predictions import (
            Plant,
            PredictionMetadata,
            PredictionResponse,
            PredictionStatus,
        )

        repo = InMemoryPredictionRepository()
        pid = uuid.uuid4()
        pred = PredictionResponse(
            prediction_id=pid,
            status=PredictionStatus.OK,
            metadata=PredictionMetadata(
                model_version="v",
                inference_timestamp="2026-04-22T13:04:00Z",
                processing_time_ms=1,
                input_resolution=[1, 1],
                input_channels=1,
                normalisation_applied=True,
                scale_factor_mm_per_px=0.05,
            ),
            segmentation_mask_url=None,
            plants=[Plant(plant_id=0, tip_x=0, tip_y=0, root_length_px=0, root_length_mm=0)],
            warnings=[],
        )
        out = await repo.save(pred)
        assert out is pred
        assert await repo.get(pid) is pred

    async def test_delete_existing_returns_true(self) -> None:
        repo = InMemoryPredictionRepository()
        seeded = uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")
        assert await repo.delete(seeded) is True
        assert await repo.get(seeded) is None

    async def test_delete_missing_returns_false(self) -> None:
        repo = InMemoryPredictionRepository()
        assert await repo.delete(uuid.uuid4()) is False


# DatabasePredictionRepository


def _orm_prediction(**overrides):
    base = dict(
        id=uuid.uuid4(),
        status="ok",
        input_filename="scan.png",
        user=None,
        model_version=None,
        inference_timestamp=datetime.now(timezone.utc),
        processing_time_ms=10,
        input_width_px=512,
        input_height_px=512,
        input_channels=1,
        normalisation_applied=True,
        scale_factor_mm_per_px=0.05,
        segmentation_mask_uri=None,
        input_uri=None,
        tip_photo_uri=None,
        plants=[],
        warnings=[],
        confirmed_at=None,
        input_hash="hash",
        user_id=uuid.uuid4(),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestDatabasePredictionRepositoryHelpers:
    async def test_no_deployed_model_error_raised(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        repo = DatabasePredictionRepository(db=db, user_id=uuid.uuid4(), storage=_fake_storage())
        with pytest.raises(NoDeployedModelError):
            await repo._get_model_version_id()

    async def test_get_returns_none_when_missing(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        repo = DatabasePredictionRepository(db=db, user_id=uuid.uuid4(), storage=_fake_storage())
        assert await repo.get(uuid.uuid4()) is None

    async def test_delete_returns_false_when_missing(self) -> None:
        db = _make_db(get_results=[None])
        repo = DatabasePredictionRepository(db=db, user_id=uuid.uuid4(), storage=_fake_storage())
        assert await repo.delete(uuid.uuid4()) is False

    async def test_delete_removes_existing_row(self) -> None:
        row = _orm_prediction()
        db = _make_db(get_results=[row])
        repo = DatabasePredictionRepository(db=db, user_id=uuid.uuid4(), storage=_fake_storage())
        assert await repo.delete(row.id) is True
        db.delete.assert_awaited_once_with(row)

    async def test_get_returns_prediction_with_feedback(self) -> None:
        row = _orm_prediction()
        fb = SimpleNamespace(signal="correct")
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=row),
                _execute_result(scalar_one_or_none=fb),
            ]
        )
        repo = DatabasePredictionRepository(db=db, user_id=row.user_id, storage=_fake_storage())
        out = await repo.get(row.id)
        assert out is not None
        assert out.feedback_signal == "correct"

    async def test_list_by_user_returns_paginated(self) -> None:
        rows = [_orm_prediction(), _orm_prediction()]
        db = _make_db(execute_results=[_execute_result(scalars_all=rows)])
        repo = DatabasePredictionRepository(db=db, user_id=uuid.uuid4(), storage=_fake_storage())
        out = await repo.list_by_user(page=1, page_size=10, confirmed=True)
        assert len(out) == 2

    async def test_list_by_user_filters_unconfirmed(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalars_all=[])])
        repo = DatabasePredictionRepository(db=db, user_id=uuid.uuid4(), storage=_fake_storage())
        out = await repo.list_by_user(confirmed=False, role=UserRole.ADMIN)
        assert out == []

    async def test_confirm_raises_when_missing(self) -> None:
        db = _make_db(execute_results=[_execute_result(scalar_one_or_none=None)])
        repo = DatabasePredictionRepository(db=db, user_id=uuid.uuid4(), storage=_fake_storage())
        with pytest.raises(ValueError, match="not found"):
            await repo.confirm(uuid.uuid4(), species=None, age_days=None, treatment=None)

    async def test_confirm_already_confirmed_returns_tuple(self) -> None:
        row = _orm_prediction(confirmed_at=datetime.now(timezone.utc))
        existing_ann = SimpleNamespace(id=uuid.uuid4())
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=row),
                _execute_result(scalar_one_or_none=existing_ann),
            ]
        )
        repo = DatabasePredictionRepository(db=db, user_id=row.user_id, storage=_fake_storage())
        _, already, ann_id = await repo.confirm(row.id, species=None, age_days=None, treatment=None)
        assert already is True
        assert ann_id == existing_ann.id

    async def test_confirm_happy_path(self) -> None:
        row = _orm_prediction()
        # 1: select prediction; create_from_prediction adds to session (no execute)
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=row),
            ]
        )
        repo = DatabasePredictionRepository(db=db, user_id=row.user_id, storage=_fake_storage())
        _, already, ann_id = await repo.confirm(row.id, species="arabidopsis", age_days=7, treatment="control")
        assert already is False
        assert ann_id is not None
        assert row.confirmed_at is not None

    async def test_confirm_raises_when_dataset_missing(self) -> None:
        row = _orm_prediction()
        db = _make_db(
            execute_results=[
                _execute_result(scalar_one_or_none=row),
                _execute_result(scalar_one_or_none=None),  # DatasetRepository.get
            ]
        )
        repo = DatabasePredictionRepository(db=db, user_id=row.user_id, storage=_fake_storage())
        with pytest.raises(ValueError, match="Dataset"):
            await repo.confirm(
                row.id,
                species=None,
                age_days=None,
                treatment=None,
                dataset_id=uuid.uuid4(),
            )
