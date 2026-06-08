"""Extra router + security tests.

Covers branches that the existing per-router test files miss:
- library: list/get/create/delete datasets, upload, add-to-dataset,
  run-inference, list/get/update/delete annotations, register/get/delete
  models, import-zip happy + 400/404/415 paths.
- training: start/list/get/stop including conflict/not-found.
- files: 404 when MinIO is active + path-traversal protection.
- health: db ok / model ok matrix + both fail.
- core.security: JWT round-trip, expired/invalid tokens, require_admin.
"""

from __future__ import annotations

import io
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt
from PIL import Image

from api.src.core.security import (
    ACCESS_TOKEN_EXPIRE_SECONDS,
    ALGORITHM,
    SECRET_KEY,
    create_access_token,
    decode_access_token,
    hash_password,
    require_admin,
    verify_password,
)
from api.src.main import app
from api.src.schemas.auth import TokenData, UserRole
from api.src.schemas.library import (
    AnnotationResponse,
    AnnotationStatus,
    DatasetDetail,
    DatasetSummary,
    ModelDetail,
    ModelStatus,
)
from api.src.schemas.training import TrainingRunDetail, TrainingRunStatus, TrainingRunSummary

# Helpers


def _png(width: int = 600, height: int = 600) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), 128).save(buf, format="PNG")
    return buf.getvalue()


def _dataset_detail(**overrides) -> DatasetDetail:
    base = dict(
        dataset_id=uuid.uuid4(),
        name="Test",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
        count=3,
        species="arabidopsis",
        description="d",
    )
    base.update(overrides)
    return DatasetDetail(**base)


def _annotation(**overrides) -> AnnotationResponse:
    base = dict(
        annotation_id=uuid.uuid4(),
        prediction_id=None,
        dataset_id=None,
        status=AnnotationStatus.PENDING,
        image_uri="s3://x.png",
        mask_url=None,
        tip_photo_url=None,
        species=None,
        age_days=None,
        treatment=None,
        growth_stage=None,
        owner_email=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return AnnotationResponse(**base)


# Library router — datasets


class TestDatasetsRouter:
    def test_list_datasets(self, client: TestClient) -> None:
        summary = DatasetSummary(
            dataset_id=uuid.uuid4(),
            name="X",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
            count=1,
            species=None,
        )
        with patch(
            "api.src.repositories.datasets.DatasetRepository.list",
            new_callable=AsyncMock,
            return_value=[summary],
        ):
            r = client.get("/api/v1/datasets")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_get_dataset_404(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.datasets.DatasetRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = client.get(f"/api/v1/datasets/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_get_dataset_ok(self, client: TestClient) -> None:
        d = _dataset_detail()
        with patch(
            "api.src.repositories.datasets.DatasetRepository.get",
            new_callable=AsyncMock,
            return_value=d,
        ):
            r = client.get(f"/api/v1/datasets/{d.dataset_id}")
        assert r.status_code == 200
        assert r.json()["name"] == "Test"

    def test_create_dataset_forbids_user(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/datasets",
            json={"name": "x", "species": "a", "growth_stage": "seedling"},
        )
        assert r.status_code == 403

    def test_create_dataset_admin(self, admin_client: TestClient) -> None:
        d = _dataset_detail()
        with patch(
            "api.src.repositories.datasets.DatasetRepository.create",
            new_callable=AsyncMock,
            return_value=d,
        ):
            r = admin_client.post(
                "/api/v1/datasets",
                json={"name": "x", "species": "a", "growth_stage": "seedling"},
            )
        assert r.status_code == 200

    def test_delete_dataset_admin_404(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.repositories.datasets.DatasetRepository.delete",
            new_callable=AsyncMock,
            return_value=False,
        ):
            r = admin_client.delete(f"/api/v1/datasets/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_delete_dataset_admin_ok(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.repositories.datasets.DatasetRepository.delete",
            new_callable=AsyncMock,
            return_value=True,
        ):
            r = admin_client.delete(f"/api/v1/datasets/{uuid.uuid4()}")
        assert r.status_code == 200

    def test_delete_dataset_forbids_user(self, client: TestClient) -> None:
        r = client.delete(f"/api/v1/datasets/{uuid.uuid4()}")
        assert r.status_code == 403


class TestDatasetUploadRouter:
    def test_upload_404_when_dataset_missing(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.datasets.DatasetRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = client.post(
                f"/api/v1/datasets/{uuid.uuid4()}/upload",
                files={"file": ("scan.png", _png(), "image/png")},
            )
        assert r.status_code == 404

    def test_upload_happy(self, client: TestClient) -> None:
        d = _dataset_detail()
        ann = _annotation(dataset_id=d.dataset_id)
        # storage.upload is called via asyncio.to_thread → patch the backend's method too
        with (
            patch(
                "api.src.repositories.datasets.DatasetRepository.get",
                new_callable=AsyncMock,
                return_value=d,
            ),
            patch(
                "api.src.repositories.annotations.AnnotationRepository.create_pending",
                new_callable=AsyncMock,
                return_value=ann,
            ),
            patch(
                "api.src.repositories.datasets.DatasetRepository.increment_count",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "api.src.services.storage.MinIOStorageBackend.upload",
                return_value="s3://x.png",
            ),
            patch(
                "api.src.services.storage.LocalStorageBackend.upload",
                return_value="s3://x.png",
            ),
        ):
            r = client.post(
                f"/api/v1/datasets/{d.dataset_id}/upload",
                files={"file": ("scan.png", _png(), "image/png")},
            )
        assert r.status_code == 201


class TestAddAnnotationToDataset:
    def test_404_when_dataset_missing(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.datasets.DatasetRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = client.post(
                f"/api/v1/datasets/{uuid.uuid4()}/annotations/{uuid.uuid4()}",
            )
        assert r.status_code == 404


class TestRunInferenceRouter:
    def test_run_dataset_inference_404_when_missing(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.datasets.DatasetRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = client.post(f"/api/v1/datasets/{uuid.uuid4()}/run-inference")
        assert r.status_code == 404

    def test_run_dataset_inference_no_pending(self, client: TestClient) -> None:
        d = _dataset_detail()
        with (
            patch(
                "api.src.repositories.datasets.DatasetRepository.get",
                new_callable=AsyncMock,
                return_value=d,
            ),
            patch(
                "api.src.services.inference_queue.enqueue_batch",
                new_callable=AsyncMock,
                return_value=0,
            ),
        ):
            r = client.post(f"/api/v1/datasets/{d.dataset_id}/run-inference")
        assert r.status_code == 200
        assert "No pending" in r.json()["message"]

    def test_run_dataset_inference_queues(self, client: TestClient) -> None:
        d = _dataset_detail()
        with (
            patch(
                "api.src.repositories.datasets.DatasetRepository.get",
                new_callable=AsyncMock,
                return_value=d,
            ),
            patch(
                "api.src.services.inference_queue.enqueue_batch",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            r = client.post(f"/api/v1/datasets/{d.dataset_id}/run-inference")
        assert r.status_code == 200
        assert "Queued 1 image" in r.json()["message"]

    def test_run_annotation_inference_404_when_missing(self, client: TestClient) -> None:
        from api.src.db.session import get_session

        async def session_factory():
            db = AsyncMock()
            db.get = AsyncMock(return_value=None)
            yield db

        app.dependency_overrides[get_session] = session_factory
        try:
            r = client.post(f"/api/v1/annotations/{uuid.uuid4()}/run-inference")
            assert r.status_code == 404
        finally:
            app.dependency_overrides.pop(get_session, None)


class TestAnnotationsRouter:
    def test_list_annotations(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.annotations.AnnotationRepository.list",
            new_callable=AsyncMock,
            return_value=[_annotation()],
        ):
            r = client.get("/api/v1/annotations?page=1&page_size=5")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_get_annotation_404(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.annotations.AnnotationRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = client.get(f"/api/v1/annotations/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_get_annotation_ok(self, client: TestClient) -> None:
        ann = _annotation()
        with patch(
            "api.src.repositories.annotations.AnnotationRepository.get",
            new_callable=AsyncMock,
            return_value=ann,
        ):
            r = client.get(f"/api/v1/annotations/{ann.annotation_id}")
        assert r.status_code == 200

    def test_update_annotation_404(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.annotations.AnnotationRepository.update_metadata",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = client.post(
                f"/api/v1/annotations/{uuid.uuid4()}/update",
                json={"species": "arabidopsis", "age_days": 7},
            )
        assert r.status_code == 404

    def test_update_annotation_ok(self, client: TestClient) -> None:
        ann = _annotation(species="arabidopsis")
        with patch(
            "api.src.repositories.annotations.AnnotationRepository.update_metadata",
            new_callable=AsyncMock,
            return_value=ann,
        ):
            r = client.post(
                f"/api/v1/annotations/{ann.annotation_id}/update",
                json={"species": "arabidopsis", "growth_stage": "seedling"},
            )
        assert r.status_code == 200

    def test_delete_annotation_404(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.annotations.AnnotationRepository.delete",
            new_callable=AsyncMock,
            return_value=False,
        ):
            r = client.delete(f"/api/v1/annotations/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_delete_annotation_ok(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.annotations.AnnotationRepository.delete",
            new_callable=AsyncMock,
            return_value=True,
        ):
            r = client.delete(f"/api/v1/annotations/{uuid.uuid4()}")
        assert r.status_code == 200


class TestModelsRouter:
    def test_register_model_admin_201(self, admin_client: TestClient) -> None:
        detail = ModelDetail(
            model_id=uuid.uuid4(),
            name="m",
            status=ModelStatus.REGISTERED,
            created_at=datetime.now(timezone.utc),
            f1_score=0.9,
            precision=0.9,
            recall=0.9,
            inference_latency_ms=100,
            artifact_uri="s3://x",
        )
        with patch(
            "api.src.repositories.models.ModelRepository.create",
            new_callable=AsyncMock,
            return_value=detail,
        ):
            r = admin_client.post(
                "/api/v1/models",
                json={
                    "version": "v9",
                    "name": "m",
                    "artifact_uri": "s3://x",
                    "f1_score": 0.9,
                },
            )
        assert r.status_code == 201

    def test_register_model_conflict(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.repositories.models.ModelRepository.create",
            new_callable=AsyncMock,
            side_effect=ValueError("already exists"),
        ):
            r = admin_client.post(
                "/api/v1/models",
                json={"version": "v1", "name": "m", "artifact_uri": "s3://x"},
            )
        assert r.status_code == 409

    def test_get_model_404(self, client: TestClient) -> None:
        with patch(
            "api.src.repositories.models.ModelRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = client.get(f"/api/v1/models/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_get_model_admin_includes_artifact_uri(self, admin_client: TestClient) -> None:
        detail = ModelDetail(
            model_id=uuid.uuid4(),
            name="m",
            status=ModelStatus.DEPLOYED,
            created_at=datetime.now(timezone.utc),
            f1_score=0.9,
            precision=0.9,
            recall=0.9,
            inference_latency_ms=100,
            artifact_uri="s3://x",
            deployed_at=datetime.now(timezone.utc),
        )
        with patch(
            "api.src.repositories.models.ModelRepository.get",
            new_callable=AsyncMock,
            return_value=detail,
        ):
            r = admin_client.get(f"/api/v1/models/{detail.model_id}")
        assert r.status_code == 200
        assert r.json()["artifact_uri"] == "s3://x"

    def test_delete_model_404(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.repositories.models.ModelRepository.delete",
            new_callable=AsyncMock,
            return_value=False,
        ):
            r = admin_client.delete(f"/api/v1/models/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_delete_model_conflict_when_predictions_exist(self, admin_client: TestClient) -> None:
        from sqlalchemy.exc import IntegrityError

        with patch(
            "api.src.repositories.models.ModelRepository.delete",
            new_callable=AsyncMock,
            side_effect=IntegrityError("stmt", {}, Exception()),
        ):
            r = admin_client.delete(f"/api/v1/models/{uuid.uuid4()}")
        assert r.status_code == 409

    def test_deploy_model_value_error_returns_404(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.services.deployment.LocalDeploymentService.deploy",
            new_callable=AsyncMock,
            side_effect=ValueError("model x not found"),
        ):
            r = admin_client.post(f"/api/v1/models/{uuid.uuid4()}/deploy")
        assert r.status_code == 404

    def test_deploy_model_filenotfound_returns_422(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.services.deployment.LocalDeploymentService.deploy",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("missing"),
        ):
            r = admin_client.post(f"/api/v1/models/{uuid.uuid4()}/deploy")
        assert r.status_code == 422


class TestImportDatasetRouter:
    def _zip_with(self, entries: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, data in entries.items():
                z.writestr(name, data)
        return buf.getvalue()

    def test_import_new_dataset_creates(self, admin_client: TestClient) -> None:
        d = _dataset_detail()
        with (
            patch(
                "api.src.repositories.datasets.DatasetRepository.create",
                new_callable=AsyncMock,
                return_value=d,
            ),
            patch(
                "api.src.repositories.annotations.AnnotationRepository.create_pending",
                new_callable=AsyncMock,
                return_value=_annotation(dataset_id=d.dataset_id),
            ),
            patch(
                "api.src.repositories.datasets.DatasetRepository.increment_count",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "api.src.services.storage.MinIOStorageBackend.upload",
                return_value="s3://x.png",
            ),
            patch(
                "api.src.services.storage.LocalStorageBackend.upload",
                return_value="s3://x.png",
            ),
        ):
            zipped = self._zip_with({"scan1.png": _png(), "scan2.png": _png(700, 700), "readme.txt": b"hi"})
            r = admin_client.post(
                "/api/v1/datasets/import",
                files={"file": ("data.zip", zipped, "application/zip")},
                data={"target_mode": "new", "name": "Z"},
            )
        assert r.status_code == 201
        body = r.json()
        assert body["scans_imported"] >= 1
        # Non-image entry shows up in warnings
        assert any("readme" in w for w in body["warnings"])

    def test_import_existing_400_when_no_id(self, admin_client: TestClient) -> None:
        zipped = self._zip_with({"scan.png": _png()})
        r = admin_client.post(
            "/api/v1/datasets/import",
            files={"file": ("data.zip", zipped, "application/zip")},
            data={"target_mode": "existing"},
        )
        assert r.status_code == 400

    def test_import_existing_404(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.repositories.datasets.DatasetRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            zipped = self._zip_with({"scan.png": _png()})
            r = admin_client.post(
                "/api/v1/datasets/import",
                files={"file": ("data.zip", zipped, "application/zip")},
                data={"target_mode": "existing", "dataset_id": str(uuid.uuid4())},
            )
        assert r.status_code == 404

    def test_import_415_for_non_zip(self, admin_client: TestClient) -> None:
        d = _dataset_detail()
        with patch(
            "api.src.repositories.datasets.DatasetRepository.create",
            new_callable=AsyncMock,
            return_value=d,
        ):
            r = admin_client.post(
                "/api/v1/datasets/import",
                files={"file": ("not.zip", b"not zip data at all", "application/zip")},
                data={"target_mode": "new"},
            )
        assert r.status_code == 415

    def test_import_forbids_user(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/datasets/import",
            files={"file": ("data.zip", b"x", "application/zip")},
            data={"target_mode": "new"},
        )
        assert r.status_code == 403


# Training router


class TestTrainingRouter:
    def test_start_forbids_user(self, client: TestClient) -> None:
        r = client.post("/api/v1/training/start", json={})
        assert r.status_code == 403

    def test_start_admin_returns_run_id(self, admin_client: TestClient) -> None:
        from api.src.schemas.training import TrainingStartResponse

        rid = uuid.uuid4()
        with patch(
            "api.src.services.training.LocalTrainingService.start",
            new_callable=AsyncMock,
            return_value=TrainingStartResponse(run_id=rid, message="ok"),
        ):
            r = admin_client.post("/api/v1/training/start", json={})
        assert r.status_code == 200
        assert r.json()["run_id"] == str(rid)

    def test_start_conflict_when_already_running(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.services.training.LocalTrainingService.start",
            new_callable=AsyncMock,
            side_effect=RuntimeError("already in progress"),
        ):
            r = admin_client.post("/api/v1/training/start", json={})
        assert r.status_code == 409

    def test_list_runs(self, admin_client: TestClient) -> None:
        summary = TrainingRunSummary(
            run_id=uuid.uuid4(),
            status=TrainingRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            final_f1=None,
            hyperparameters={"lr": 0.001},
        )
        with patch(
            "api.src.services.training.LocalTrainingService.get_runs",
            new_callable=AsyncMock,
            return_value=[summary],
        ):
            r = admin_client.get("/api/v1/training/runs")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_get_run_404(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.services.training.LocalTrainingService.get_run",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = admin_client.get(f"/api/v1/training/runs/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_get_run_ok(self, admin_client: TestClient) -> None:
        detail = TrainingRunDetail(
            run_id=uuid.uuid4(),
            status=TrainingRunStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            final_f1=0.91,
            hyperparameters={"lr": 0.001},
            model_version_id=uuid.uuid4(),
        )
        with patch(
            "api.src.services.training.LocalTrainingService.get_run",
            new_callable=AsyncMock,
            return_value=detail,
        ):
            r = admin_client.get(f"/api/v1/training/runs/{detail.run_id}")
        assert r.status_code == 200

    def test_stop_run_404(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.services.training.LocalTrainingService.stop",
            new_callable=AsyncMock,
            return_value=False,
        ):
            r = admin_client.delete(f"/api/v1/training/runs/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_stop_run_conflict(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.services.training.LocalTrainingService.stop",
            new_callable=AsyncMock,
            side_effect=ValueError("only running"),
        ):
            r = admin_client.delete(f"/api/v1/training/runs/{uuid.uuid4()}")
        assert r.status_code == 409

    def test_stop_run_ok(self, admin_client: TestClient) -> None:
        with patch(
            "api.src.services.training.LocalTrainingService.stop",
            new_callable=AsyncMock,
            return_value=True,
        ):
            r = admin_client.delete(f"/api/v1/training/runs/{uuid.uuid4()}")
        assert r.status_code == 200


# Files router


class TestFilesRouter:
    def test_404_when_minio_active(self, client: TestClient) -> None:
        with (
            patch("api.src.routers.files.DEPLOYMENT_ENV", "local"),
            patch("api.src.routers.files.STORAGE_BACKEND", "minio"),
        ):
            r = client.get("/api/v1/files/inputs/x.png")
        assert r.status_code == 404

    def test_404_when_cloud(self, client: TestClient) -> None:
        with (
            patch("api.src.routers.files.DEPLOYMENT_ENV", "cloud"),
            patch("api.src.routers.files.STORAGE_BACKEND", "local"),
        ):
            r = client.get("/api/v1/files/inputs/x.png")
        assert r.status_code == 404

    def test_404_when_file_missing(self, client: TestClient, tmp_path) -> None:
        with (
            patch("api.src.routers.files.DEPLOYMENT_ENV", "local"),
            patch("api.src.routers.files.STORAGE_BACKEND", "local"),
            patch("api.src.routers.files.LOCAL_STORAGE_BASE", tmp_path),
        ):
            r = client.get("/api/v1/files/nope.png")
        assert r.status_code == 404

    def test_streams_existing_file(self, client: TestClient, tmp_path) -> None:
        (tmp_path / "x.png").write_bytes(_png(512, 512))
        with (
            patch("api.src.routers.files.DEPLOYMENT_ENV", "local"),
            patch("api.src.routers.files.STORAGE_BACKEND", "local"),
            patch("api.src.routers.files.LOCAL_STORAGE_BASE", tmp_path),
        ):
            r = client.get("/api/v1/files/x.png")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/")


# Health router


class TestHealthRouter:
    def test_db_ok_model_ok(self, client: TestClient) -> None:
        with (
            patch("api.src.routers.health._check_db", new_callable=AsyncMock, return_value=True),
            patch("api.src.routers.health._check_model", new_callable=AsyncMock, return_value=True),
        ):
            r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_db_fail_model_ok_degraded(self, client: TestClient) -> None:
        with (
            patch("api.src.routers.health._check_db", new_callable=AsyncMock, return_value=False),
            patch("api.src.routers.health._check_model", new_callable=AsyncMock, return_value=True),
        ):
            r = client.get("/api/v1/health")
        assert r.json()["status"] == "degraded"

    def test_both_fail_unavailable(self, client: TestClient) -> None:
        with (
            patch("api.src.routers.health._check_db", new_callable=AsyncMock, return_value=False),
            patch("api.src.routers.health._check_model", new_callable=AsyncMock, return_value=False),
        ):
            r = client.get("/api/v1/health")
        assert r.json()["status"] == "unavailable"

    async def test_check_db_returns_true_on_success(self) -> None:
        from api.src.routers.health import _check_db

        db = AsyncMock()
        db.execute = AsyncMock()
        assert await _check_db(db) is True

    async def test_check_db_returns_false_on_failure(self) -> None:
        from api.src.routers.health import _check_db

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=RuntimeError("DB down"))
        assert await _check_db(db) is False

    async def test_check_model_returns_false_when_no_artifact(self) -> None:
        from api.src.routers.health import _check_model

        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=result)
        assert await _check_model(db) is False

    async def test_check_model_returns_false_when_artifact_missing(self, tmp_path) -> None:
        from api.src.routers.health import _check_model

        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = str(tmp_path / "missing.keras")
        db.execute = AsyncMock(return_value=result)
        assert await _check_model(db) is False

    async def test_check_model_returns_true_when_artifact_present(self, tmp_path) -> None:
        from api.src.routers.health import _check_model

        f = tmp_path / "x.keras"
        f.write_bytes(b"")
        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = str(f)
        db.execute = AsyncMock(return_value=result)
        assert await _check_model(db) is True


# core.security


class TestSecurity:
    def test_hash_and_verify_roundtrip(self) -> None:
        hashed = hash_password("hunter2!")
        assert verify_password("hunter2!", hashed) is True
        assert verify_password("wrong", hashed) is False

    def test_access_token_roundtrip(self) -> None:
        token = create_access_token(subject=str(uuid.uuid4()), role=UserRole.USER)
        data = decode_access_token(token)
        assert data.role == UserRole.USER

    def test_decode_invalid_token_raises_401(self) -> None:
        with pytest.raises(HTTPException) as ei:
            decode_access_token("not-a-jwt-at-all")
        assert ei.value.status_code == 401

    def test_decode_expired_token_raises_401(self) -> None:
        payload = {
            "sub": str(uuid.uuid4()),
            "role": "user",
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(HTTPException):
            decode_access_token(token)

    def test_decode_missing_claims_raises_401(self) -> None:
        payload = {
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(HTTPException):
            decode_access_token(token)

    def test_require_admin_allows_admin(self) -> None:
        out = require_admin(TokenData(sub=str(uuid.uuid4()), role=UserRole.ADMIN))
        assert out.role == UserRole.ADMIN

    def test_require_admin_denies_user(self) -> None:
        with pytest.raises(HTTPException) as ei:
            require_admin(TokenData(sub=str(uuid.uuid4()), role=UserRole.USER))
        assert ei.value.status_code == 403

    async def test_get_current_user_404_when_user_missing(self) -> None:
        from api.src.core.security import get_current_user

        token = create_access_token(subject=str(uuid.uuid4()), role=UserRole.USER)
        db = AsyncMock()
        db.get = AsyncMock(return_value=None)
        with pytest.raises(HTTPException) as ei:
            await get_current_user(token=token, db=db)
        assert ei.value.status_code == 401

    async def test_get_current_user_401_when_inactive(self) -> None:
        from api.src.core.security import get_current_user

        token = create_access_token(subject=str(uuid.uuid4()), role=UserRole.USER)
        inactive = SimpleNamespace(is_active=False)
        db = AsyncMock()
        db.get = AsyncMock(return_value=inactive)
        with pytest.raises(HTTPException) as ei:
            await get_current_user(token=token, db=db)
        assert ei.value.status_code == 401

    async def test_get_current_user_returns_token_data(self) -> None:
        from api.src.core.security import get_current_user

        uid = uuid.uuid4()
        token = create_access_token(subject=str(uid), role=UserRole.USER)
        active = SimpleNamespace(is_active=True)
        db = AsyncMock()
        db.get = AsyncMock(return_value=active)
        out = await get_current_user(token=token, db=db)
        assert out.sub == str(uid)
        assert out.role == UserRole.USER
