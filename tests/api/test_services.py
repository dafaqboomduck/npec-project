"""Unit tests for service-layer modules.

Targets:
    - services/image_validation.py — pure functions
    - services/mask_storage.py     — pure functions + backend interaction
    - services/artifact_storage.py — backend interaction (PIL/numpy heavy)
    - services/storage.py          — Local backend + factory + MinIO/Azure mocks
    - services/deployment.py       — happy path + failure paths
    - services/environment.py      — module mocking around psutil/pynvml
    - services/training.py         — pure helpers + LocalTrainingService.stop
    - services/mock_inference.py   — already 100% but verify shape
    - services/monitoring.py       — pure DB write
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from api.src.schemas.common import ErrorCode
from api.src.schemas.library import ModelStatus
from api.src.schemas.overview import SystemStatus
from api.src.schemas.training import TrainingRunStatus
from api.src.services import deployment as deployment_mod
from api.src.services import environment as env_mod
from api.src.services import image_validation as iv
from api.src.services import mask_storage as ms
from api.src.services import monitoring as monitoring_mod
from api.src.services import storage as storage_mod
from api.src.services import training as training_mod
from api.src.services.artifact_storage import (
    _mime_from_ext,
    save_input_image,
    save_segmentation_mask,
    save_tip_photo,
)
from api.src.services.deployment import LocalDeploymentService
from api.src.services.mock_inference import MockInferenceService
from api.src.services.storage import (
    LocalStorageBackend,
    MinIOStorageBackend,
    get_storage_backend,
)
from api.src.services.training import (
    LocalTrainingService,
    _build_subprocess_env,
    _parse_final_f1,
    _pid_alive,
)


def _make_png(width: int = 600, height: int = 600) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), 128).save(buf, format="PNG")
    return buf.getvalue()


def _make_tiff(width: int = 600, height: int = 600, frames: int = 1) -> bytes:
    buf = io.BytesIO()
    imgs = [Image.new("L", (width, height), 128) for _ in range(frames)]
    imgs[0].save(buf, format="TIFF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


# image_validation


class TestImageValidation:
    def test_validate_file_size_ok(self) -> None:
        iv.validate_file_size(b"x" * 1000)

    def test_validate_file_size_too_large(self) -> None:
        big = b"x" * (iv.MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(HTTPException) as ei:
            iv.validate_file_size(big)
        assert ei.value.status_code == 413

    def test_validate_content_type_ok(self) -> None:
        iv.validate_content_type("image/png", "scan.png")

    @pytest.mark.parametrize(
        "ct,fn",
        [
            ("image/png", "scan.xyz"),
            ("application/json", "scan.png"),
            ("", "scan.bin"),
        ],
    )
    def test_validate_content_type_rejects(self, ct, fn) -> None:
        with pytest.raises(HTTPException) as ei:
            iv.validate_content_type(ct, fn)
        assert ei.value.status_code == 415

    def test_validate_image_ok(self) -> None:
        img = iv.validate_image(_make_png(600, 600), "scan.png")
        assert img.size == (600, 600)

    def test_validate_image_too_small(self) -> None:
        with pytest.raises(HTTPException) as ei:
            iv.validate_image(_make_png(100, 100), "scan.png")
        assert ei.value.status_code == 422
        assert ei.value.detail["error_code"] == ErrorCode.INVALID_RESOLUTION.value

    def test_validate_image_too_large(self) -> None:
        with pytest.raises(HTTPException) as ei:
            iv.validate_image(_make_png(6000, 6000), "scan.png")
        assert ei.value.status_code == 422

    def test_validate_image_corrupt(self) -> None:
        with pytest.raises(HTTPException) as ei:
            iv.validate_image(b"not an image", "scan.png")
        assert ei.value.status_code == 422
        assert ei.value.detail["error_code"] == ErrorCode.CORRUPT_FILE.value

    def test_validate_image_multipage_tiff_rejected(self) -> None:
        with pytest.raises(HTTPException) as ei:
            iv.validate_image(_make_tiff(600, 600, frames=3), "scan.tiff")
        assert ei.value.status_code == 422
        assert ei.value.detail["error_code"] == ErrorCode.MULTI_PAGE_TIFF.value

    def test_validate_image_single_frame_tiff_ok(self) -> None:
        iv.validate_image(_make_tiff(600, 600, frames=1), "scan.tiff")


# mask_storage


class TestMaskStorage:
    def test_validate_mask_rejects_bad_extension(self) -> None:
        with pytest.raises(ValueError, match="PNG or TIFF"):
            ms.validate_mask(_make_png(), "mask.jpg")

    def test_validate_mask_rejects_corrupt(self) -> None:
        with pytest.raises(ValueError):
            ms.validate_mask(b"not-an-image", "mask.png")

    def test_validate_mask_rejects_multiframe(self) -> None:
        with pytest.raises(ValueError, match="single-frame"):
            ms.validate_mask(_make_tiff(frames=3), "mask.tiff")

    def test_validate_mask_accepts_png(self) -> None:
        ms.validate_mask(_make_png(), "mask.png")

    def test_normalise_to_png_grayscale_png_passthrough(self) -> None:
        original = _make_png()
        out = ms.normalise_to_png(original)
        # Grayscale PNG bytes are returned unchanged
        assert out == original

    def test_normalise_to_png_converts_rgb(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), (1, 2, 3)).save(buf, format="PNG")
        out = ms.normalise_to_png(buf.getvalue())
        # Re-encoded; verify it's a single-channel PNG now
        img = Image.open(io.BytesIO(out))
        assert img.mode == "L"

    def test_normalise_to_png_converts_tiff(self) -> None:
        out = ms.normalise_to_png(_make_tiff())
        img = Image.open(io.BytesIO(out))
        assert img.format == "PNG"

    def test_save_annotation_mask_calls_backend_upload(self) -> None:
        backend = MagicMock()
        backend.upload.return_value = "key"
        key = ms.save_annotation_mask(uuid.uuid4(), _make_png(), backend=backend)
        assert key == "key"
        backend.upload.assert_called_once()


# artifact_storage


class TestArtifactStorage:
    def test_mime_from_ext_known(self) -> None:
        assert _mime_from_ext(".png") == "image/png"
        assert _mime_from_ext(".JPG") == "image/jpeg"
        assert _mime_from_ext(".tif") == "image/tiff"
        assert _mime_from_ext(".unknown") == "application/octet-stream"

    def test_save_input_image_happy(self) -> None:
        backend = MagicMock()
        backend.upload.return_value = "url"
        assert save_input_image(uuid.uuid4(), b"data", "scan.png", backend) == "url"

    def test_save_input_image_handles_no_extension(self) -> None:
        backend = MagicMock()
        backend.upload.return_value = "url"
        assert save_input_image(uuid.uuid4(), b"data", "noext", backend) == "url"

    def test_save_input_image_swallows_exception(self) -> None:
        backend = MagicMock()
        backend.upload.side_effect = RuntimeError("boom")
        assert save_input_image(uuid.uuid4(), b"data", "scan.png", backend) is None

    def test_save_segmentation_mask_no_masks_returns_none(self) -> None:
        backend = MagicMock()
        image_result = SimpleNamespace(plant_results={"a": SimpleNamespace(mask=None)})
        assert save_segmentation_mask(uuid.uuid4(), image_result, backend) is None

    def test_save_segmentation_mask_combines_masks(self) -> None:
        backend = MagicMock()
        backend.upload.return_value = "u"
        m1 = np.zeros((4, 4), dtype=np.uint8)
        m1[0, 0] = 1
        m2 = np.zeros((4, 4), dtype=np.uint8)
        m2[1, 1] = 1
        image_result = SimpleNamespace(
            plant_results={
                "1": SimpleNamespace(mask=m1),
                "2": SimpleNamespace(mask=m2),
            }
        )
        assert save_segmentation_mask(uuid.uuid4(), image_result, backend) == "u"
        backend.upload.assert_called_once()

    def test_save_segmentation_mask_swallows_exceptions(self) -> None:
        backend = MagicMock()
        backend.upload.side_effect = RuntimeError("boom")
        m1 = np.zeros((4, 4), dtype=np.uint8)
        image_result = SimpleNamespace(plant_results={"1": SimpleNamespace(mask=m1)})
        assert save_segmentation_mask(uuid.uuid4(), image_result, backend) is None

    def test_save_tip_photo_no_detected_returns_none(self) -> None:
        backend = MagicMock()
        image_result = SimpleNamespace(plant_results={"a": SimpleNamespace(detected=False, end_point=None)})
        assert save_tip_photo(uuid.uuid4(), image_result, _make_png(), backend) is None

    def test_save_tip_photo_draws_and_uploads(self) -> None:
        backend = MagicMock()
        backend.upload.return_value = "url"
        image_result = SimpleNamespace(
            plant_results={
                "p1": SimpleNamespace(detected=True, end_point=(50, 50)),
                "p2": SimpleNamespace(detected=True, end_point=(100, 100)),
            }
        )
        out = save_tip_photo(uuid.uuid4(), image_result, _make_png(), backend)
        assert out == "url"
        backend.upload.assert_called_once()

    def test_save_tip_photo_swallows_exception(self) -> None:
        backend = MagicMock()
        backend.upload.side_effect = RuntimeError("boom")
        image_result = SimpleNamespace(plant_results={"p": SimpleNamespace(detected=True, end_point=(50, 50))})
        assert save_tip_photo(uuid.uuid4(), image_result, _make_png(), backend) is None


# storage — LocalStorageBackend + factory


class TestLocalStorageBackend:
    def test_round_trip(self, tmp_path: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_path)
        key = backend.upload("subdir/file.png", b"data", "image/png")
        assert key == "subdir/file.png"
        assert (tmp_path / "subdir" / "file.png").read_bytes() == b"data"
        assert backend.download(str(tmp_path / "subdir" / "file.png")) == b"data"
        assert backend.download(f"file://{tmp_path / 'subdir' / 'file.png'}") == b"data"

    def test_sign_returns_proxy_route(self, tmp_path: Path) -> None:
        backend = LocalStorageBackend(base_path=tmp_path)
        assert backend.sign("a/b.png") == "/api/v1/files/a/b.png"


class TestGetStorageBackend:
    def test_default_minio(self, monkeypatch) -> None:
        monkeypatch.setattr(storage_mod, "_DEPLOYMENT_ENV", "local")
        monkeypatch.setattr(storage_mod, "STORAGE_BACKEND", "minio")
        assert isinstance(get_storage_backend(), MinIOStorageBackend)

    def test_local(self, monkeypatch) -> None:
        monkeypatch.setattr(storage_mod, "_DEPLOYMENT_ENV", "local")
        monkeypatch.setattr(storage_mod, "STORAGE_BACKEND", "local")
        assert isinstance(get_storage_backend(), LocalStorageBackend)

    def test_minio_rejects_localhost_public_endpoint_in_non_local(self, monkeypatch) -> None:
        monkeypatch.setattr(storage_mod, "_DEPLOYMENT_ENV", "staging")
        monkeypatch.setattr(storage_mod, "STORAGE_BACKEND", "minio")
        monkeypatch.setattr(storage_mod, "MINIO_PUBLIC_ENDPOINT", "localhost:9000")
        with pytest.raises(RuntimeError, match="MINIO_PUBLIC_ENDPOINT"):
            get_storage_backend()


class TestMinIOStorageBackend:
    def test_upload_init_creates_client_and_uploads(self) -> None:
        backend = MinIOStorageBackend(bucket="bkt")
        with patch.object(MinIOStorageBackend, "_get_client") as mock_get:
            client = MagicMock()
            client.bucket_exists.return_value = True
            mock_get.return_value = client
            key = backend.upload("x.png", b"d", "image/png")
            assert key == "x.png"
            client.put_object.assert_called_once()

    def test_get_client_creates_bucket_when_missing(self) -> None:
        backend = MinIOStorageBackend()
        fake_minio = MagicMock()
        client = MagicMock()
        client.bucket_exists.return_value = False
        fake_minio.return_value = client
        with patch.dict("sys.modules", {"minio": MagicMock(Minio=fake_minio)}):
            backend._client = None
            out = backend._get_client()
            assert out is client
            client.make_bucket.assert_called_once()

    def test_sign_uses_public_client(self) -> None:
        backend = MinIOStorageBackend()
        public = MagicMock()
        public.presigned_get_object.return_value = "https://example.com/x"
        with patch.object(MinIOStorageBackend, "_get_public_client", return_value=public):
            assert backend.sign("x.png") == "https://example.com/x"

    def test_download_uses_get_client(self) -> None:
        backend = MinIOStorageBackend(bucket="bkt")
        response = MagicMock()
        response.read.return_value = b"data"
        client = MagicMock()
        client.get_object.return_value = response
        with patch.object(MinIOStorageBackend, "_get_client", return_value=client):
            out = backend.download("https://host/bkt/x.png")
            assert out == b"data"
            response.close.assert_called_once()
            response.release_conn.assert_called_once()

    def test_download_strips_bucket_prefix_from_path(self) -> None:
        backend = MinIOStorageBackend(bucket="bkt")
        response = MagicMock()
        response.read.return_value = b"data"
        client = MagicMock()
        client.get_object.return_value = response
        with patch.object(MinIOStorageBackend, "_get_client", return_value=client):
            backend.download("https://host/bkt/folder/x.png")
            args, _ = client.get_object.call_args
            # bucket name stripped from key
            assert args == ("bkt", "folder/x.png")


# deployment


class TestLocalDeploymentService:
    async def test_deploy_raises_when_missing(self) -> None:
        db = AsyncMock()
        db.get = AsyncMock(return_value=None)
        svc = LocalDeploymentService(db=db, inference_service=MockInferenceService())
        with pytest.raises(ValueError, match="not found"):
            await svc.deploy(uuid.uuid4())

    async def test_deploy_raises_when_no_artifact_uri(self) -> None:
        row = SimpleNamespace(artifact_uri=None, status="registered", version="v1")
        db = AsyncMock()
        db.get = AsyncMock(return_value=row)
        svc = LocalDeploymentService(db=db, inference_service=MockInferenceService())
        with pytest.raises(ValueError, match="no artifact_uri"):
            await svc.deploy(uuid.uuid4())

    async def test_deploy_raises_when_artifact_dir_missing(self, tmp_path: Path) -> None:
        row = SimpleNamespace(
            artifact_uri=str(tmp_path / "nope"),
            status="registered",
            version="v1",
            deployed_at=None,
        )
        db = AsyncMock()
        db.get = AsyncMock(return_value=row)
        svc = LocalDeploymentService(db=db, inference_service=MockInferenceService())
        with pytest.raises(FileNotFoundError):
            await svc.deploy(uuid.uuid4())

    async def test_deploy_copies_and_marks_deployed(self, tmp_path: Path, monkeypatch) -> None:
        # Source has a fake model dir
        src = tmp_path / "source"
        src.mkdir()
        (src / "model.keras").write_bytes(b"weights")

        # Override the destination root
        dest_root = tmp_path / "deployed"
        monkeypatch.setattr(deployment_mod, "_ARTIFACT_DIR", dest_root)

        row = SimpleNamespace(
            artifact_uri=str(src),
            status="registered",
            version="v9",
            deployed_at=None,
        )
        db = AsyncMock()
        db.get = AsyncMock(return_value=row)
        db.execute = AsyncMock(return_value=MagicMock())
        db.commit = AsyncMock()
        svc = LocalDeploymentService(db=db, inference_service=MockInferenceService())

        await svc.deploy(uuid.uuid4())
        assert row.status == ModelStatus.DEPLOYED.value
        assert (dest_root / "v9" / "model.keras").exists()
        db.commit.assert_awaited_once()

    async def test_deploy_overwrites_existing_dest(self, tmp_path: Path, monkeypatch) -> None:
        src = tmp_path / "source"
        src.mkdir()
        (src / "model.keras").write_bytes(b"weights")
        dest_root = tmp_path / "deployed"
        # Pre-create the destination with stale content
        (dest_root / "v9").mkdir(parents=True)
        (dest_root / "v9" / "stale.txt").write_text("stale")
        monkeypatch.setattr(deployment_mod, "_ARTIFACT_DIR", dest_root)

        row = SimpleNamespace(
            artifact_uri=str(src),
            status="registered",
            version="v9",
            deployed_at=None,
        )
        db = AsyncMock()
        db.get = AsyncMock(return_value=row)
        db.execute = AsyncMock(return_value=MagicMock())
        db.commit = AsyncMock()
        svc = LocalDeploymentService(db=db, inference_service=MockInferenceService())

        await svc.deploy(uuid.uuid4())
        # Stale file was wiped
        assert not (dest_root / "v9" / "stale.txt").exists()


# environment


class TestEnvironment:
    def test_system_status_thresholds(self) -> None:
        assert env_mod._system_status(0) == SystemStatus.OPTIMAL
        assert env_mod._system_status(99) == SystemStatus.CRITICAL

    def test_get_memory_percent_returns_int(self, monkeypatch) -> None:
        fake = MagicMock()
        fake.virtual_memory.return_value = MagicMock(percent=37)
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
        assert env_mod._get_memory_percent() == 37

    def test_get_memory_percent_handles_failure(self, monkeypatch) -> None:
        fake = MagicMock()
        fake.virtual_memory.side_effect = RuntimeError("nope")
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
        assert env_mod._get_memory_percent() == 0

    def test_get_cpu_percent_returns_float_or_none(self, monkeypatch) -> None:
        fake = MagicMock()
        fake.cpu_percent.return_value = 12.3
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
        assert env_mod._get_cpu_percent() == 12.3

    def test_gpu_helpers_handle_no_gpu(self, monkeypatch) -> None:
        """When pynvml init fails, all GPU helpers should swallow and return None."""
        fake = MagicMock()
        fake.nvmlInit.side_effect = RuntimeError("no NVIDIA driver")
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
        assert env_mod._get_gpu_temp() is None
        assert env_mod._get_gpu_utilisation() is None
        assert env_mod._get_vram_percent() is None

    def test_cuda_available_is_bool(self, monkeypatch) -> None:
        # Stub pynvml so the test doesn't touch real NVIDIA drivers (which can
        # block briefly on Windows and produces noisy deprecation warnings).
        fake = MagicMock()
        fake.nvmlInit.return_value = None
        fake.nvmlDeviceGetCount.return_value = 0
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
        assert env_mod._cuda_available() is False

    def test_get_disk_percent_returns_int_or_none(self, monkeypatch) -> None:
        fake = MagicMock()
        fake.disk_usage.return_value = MagicMock(percent=55.7)
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
        assert env_mod._get_disk_percent() == 55

    def test_gpu_helpers_no_pynvml_installed(self, monkeypatch) -> None:
        """When pynvml itself isn't importable, every helper returns None."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert env_mod._get_gpu_temp() is None
        assert env_mod._get_gpu_utilisation() is None
        assert env_mod._get_vram_percent() is None
        assert env_mod._cuda_available() is False

    def test_get_environment_stats_returns_response(self, monkeypatch) -> None:
        fake_psutil = MagicMock()
        fake_psutil.virtual_memory.return_value = MagicMock(percent=42)
        fake_psutil.cpu_percent.return_value = 10.0
        fake_psutil.disk_usage.return_value = MagicMock(percent=55)
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlInit.side_effect = RuntimeError("no driver")
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)
        out = env_mod.get_environment_stats()
        assert out.memory_percent == 42
        assert out.system_status in SystemStatus


# monitoring


class TestMonitoringService:
    async def test_record_prediction_metrics_adds_and_commits(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: uuid.uuid4()))
        svc = monitoring_mod.MonitoringService(db)
        await svc.record_prediction_metrics(uuid.uuid4(), 123.4)
        db.add.assert_called_once()
        db.commit.assert_awaited_once()


# dependencies / lifespan startup helpers


class TestWarmupInferenceService:
    async def test_skipped_in_mock_mode(self, monkeypatch) -> None:
        from api.src import dependencies as dep

        monkeypatch.setattr(dep, "_USE_MOCK", True)
        monkeypatch.setattr(dep, "DEPLOYMENT_ENV", "local")
        # Should return without touching the inference service.
        await dep.warmup_inference_service()

    async def test_skipped_in_cloud(self, monkeypatch) -> None:
        from api.src import dependencies as dep

        monkeypatch.setattr(dep, "DEPLOYMENT_ENV", "cloud")
        await dep.warmup_inference_service()

    async def test_warmup_loads_pipeline_when_not_mock(self, monkeypatch) -> None:
        from api.src import dependencies as dep

        monkeypatch.setattr(dep, "_USE_MOCK", False)
        monkeypatch.setattr(dep, "DEPLOYMENT_ENV", "local")
        svc = MagicMock()
        svc._get_pipeline = MagicMock(return_value=None)
        monkeypatch.setattr(dep, "get_inference_service", lambda: svc)
        await dep.warmup_inference_service()
        svc._get_pipeline.assert_called_once()

    async def test_warmup_swallows_load_failure(self, monkeypatch) -> None:
        from api.src import dependencies as dep

        monkeypatch.setattr(dep, "_USE_MOCK", False)
        monkeypatch.setattr(dep, "DEPLOYMENT_ENV", "local")
        svc = MagicMock()
        svc._get_pipeline = MagicMock(side_effect=RuntimeError("missing weights"))
        monkeypatch.setattr(dep, "get_inference_service", lambda: svc)
        # Must not raise — failure is logged and the API still starts.
        await dep.warmup_inference_service()


class TestReconcileFallbacks:
    async def test_reconcile_swallows_db_failure(self) -> None:
        """`reconcile_orphaned_training_runs` must never raise — the API has to
        start even when the DB is unreachable. Force the sessionmaker to raise
        synchronously; without this on Windows asyncpg can hang trying to dial
        the fake DATABASE_URL.
        """

        def boom(*_a, **_kw):
            raise RuntimeError("db unreachable")

        with patch("api.src.db.session.get_sessionmaker", side_effect=boom):
            await training_mod.reconcile_orphaned_training_runs()


# AzureBlobStorageBackend


class TestAzureBlobStorageBackend:
    def _make(self):
        from api.src.services.storage import AzureBlobStorageBackend

        return AzureBlobStorageBackend(
            connection_string="DefaultEndpointsProtocol=https;AccountName=acc;AccountKey=key;EndpointSuffix=core.windows.net",
            container="bkt",
        )

    def test_upload_calls_blob_client(self) -> None:
        backend = self._make()
        blob_client = MagicMock()
        client = MagicMock()
        client.get_blob_client.return_value = blob_client
        with (
            patch.object(backend.__class__, "_get_client", return_value=client),
            # _content_settings does an inline `from azure.storage.blob import …`;
            # bypass it so the test runs without azure-storage-blob installed.
            patch.object(backend.__class__, "_content_settings", return_value=MagicMock()),
        ):
            out = backend.upload("x.png", b"data", "image/png")
        assert out == "x.png"
        blob_client.upload_blob.assert_called_once()

    def test_download_from_url(self) -> None:
        backend = self._make()
        blob = MagicMock()
        downloaded = MagicMock()
        downloaded.readall.return_value = b"data"
        blob.download_blob.return_value = downloaded
        client = MagicMock()
        client.get_blob_client.return_value = blob
        with patch.object(backend.__class__, "_get_client", return_value=client):
            out = backend.download("https://acct.blob.core/bkt/folder/x.png")
        assert out == b"data"

    def test_download_from_key(self) -> None:
        backend = self._make()
        blob = MagicMock()
        downloaded = MagicMock()
        downloaded.readall.return_value = b"data"
        blob.download_blob.return_value = downloaded
        client = MagicMock()
        client.get_blob_client.return_value = blob
        with patch.object(backend.__class__, "_get_client", return_value=client):
            out = backend.download("folder/x.png")
        assert out == b"data"


# environment — extra branches via psutil mocking


class TestEnvironmentExtra:
    def test_get_cpu_percent_failure_returns_none(self, monkeypatch) -> None:
        from api.src.services import environment as env_mod

        fake = MagicMock()
        fake.cpu_percent.side_effect = RuntimeError("nope")
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
        assert env_mod._get_cpu_percent() is None

    def test_get_disk_percent_failure_returns_none(self, monkeypatch) -> None:
        from api.src.services import environment as env_mod

        fake = MagicMock()
        fake.disk_usage.side_effect = RuntimeError("nope")
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
        assert env_mod._get_disk_percent() is None

    def test_get_gpu_temp_via_mocked_pynvml(self, monkeypatch) -> None:
        from api.src.services import environment as env_mod

        fake = MagicMock()
        fake.nvmlInit.return_value = None
        fake.nvmlDeviceGetHandleByIndex.return_value = "h"
        fake.nvmlDeviceGetTemperature.return_value = 65
        fake.NVML_TEMPERATURE_GPU = 0
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
        assert env_mod._get_gpu_temp() == 65.0

    def test_get_gpu_utilisation_via_mocked_pynvml(self, monkeypatch) -> None:
        from api.src.services import environment as env_mod

        fake = MagicMock()
        fake.nvmlInit.return_value = None
        fake.nvmlDeviceGetHandleByIndex.return_value = "h"
        rates = MagicMock()
        rates.gpu = 42
        fake.nvmlDeviceGetUtilizationRates.return_value = rates
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
        assert env_mod._get_gpu_utilisation() == 42.0

    def test_get_vram_percent_via_mocked_pynvml(self, monkeypatch) -> None:
        from api.src.services import environment as env_mod

        fake = MagicMock()
        fake.nvmlInit.return_value = None
        fake.nvmlDeviceGetHandleByIndex.return_value = "h"
        mem = MagicMock(used=2_000_000_000, total=8_000_000_000)
        fake.nvmlDeviceGetMemoryInfo.return_value = mem
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
        assert env_mod._get_vram_percent() == 25.0

    def test_cuda_available_via_mocked_pynvml(self, monkeypatch) -> None:
        from api.src.services import environment as env_mod

        fake = MagicMock()
        fake.nvmlInit.return_value = None
        fake.nvmlDeviceGetCount.return_value = 1
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
        assert env_mod._cuda_available() is True


# mock_inference


class TestMockInferenceService:
    async def test_run_returns_two_plants(self) -> None:
        svc = MockInferenceService()
        prediction, image_result = await svc.run(image_bytes=b"x", filename="a.png")
        assert len(prediction.plants) == 2
        assert image_result is None


# training helpers


class TestTrainingHelpers:
    def test_parse_final_f1_extracts(self) -> None:
        assert _parse_final_f1("blah\nFINAL_F1=0.821\nmore\n") == 0.821

    def test_parse_final_f1_returns_none_when_missing(self) -> None:
        assert _parse_final_f1("no marker here") is None

    def test_parse_final_f1_handles_malformed(self) -> None:
        assert _parse_final_f1("FINAL_F1=notanumber") is None

    def test_build_subprocess_env_includes_required_keys(self, tmp_path: Path) -> None:
        run_id = uuid.uuid4()
        env = _build_subprocess_env(
            run_id=run_id,
            hyperparameters={"learning_rate": 0.001, "batch_size": 8, "epochs": 5},
            output_dir=tmp_path / "out",
            artifact_dir=tmp_path / "art",
        )
        assert env["TRAINING_RUN_ID"] == str(run_id)
        assert env["NPEC_LEARNING_RATE"] == "0.001"
        assert env["NPEC_BATCH_SIZE"] == "8"
        assert env["NPEC_EPOCHS"] == "5"
        assert env["NPEC_OUTPUT_DIR"] == str(tmp_path / "out")

    def test_build_subprocess_env_omits_unset_hparams(self, tmp_path: Path) -> None:
        env = _build_subprocess_env(
            run_id=uuid.uuid4(),
            hyperparameters={},
            output_dir=tmp_path,
            artifact_dir=tmp_path,
        )
        assert "NPEC_LEARNING_RATE" not in env
        assert "NPEC_BATCH_SIZE" not in env
        assert "NPEC_EPOCHS" not in env

    def test_pid_alive_returns_false_for_none(self) -> None:
        assert _pid_alive(None) is False
        assert _pid_alive(0) is False

    def test_pid_alive_returns_true_for_existing_pid(self, monkeypatch) -> None:
        # Mock `os.kill` so the test never makes a real OS syscall. The real
        # `os.kill(os.getpid(), 0)` is unreliable on Windows — depending on
        # AV / process scanners it can block for 30+ seconds.
        monkeypatch.setattr("os.kill", MagicMock(return_value=None))
        assert _pid_alive(12345) is True

    def test_pid_alive_handles_lookup_error(self, monkeypatch) -> None:
        monkeypatch.setattr("os.kill", MagicMock(side_effect=ProcessLookupError()))
        assert _pid_alive(12345) is False

    def test_pid_alive_handles_permission_error(self, monkeypatch) -> None:
        monkeypatch.setattr("os.kill", MagicMock(side_effect=PermissionError()))
        assert _pid_alive(12345) is True

    def test_next_version_dir_returns_next_when_versions_exist(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "v1").mkdir()
        (tmp_path / "v3").mkdir()
        (tmp_path / "not-a-version").mkdir()
        monkeypatch.setenv("NPEC_ARTIFACT_DIR", str(tmp_path))
        out, version = training_mod._next_version_dir()
        assert version == "v4"
        assert out == tmp_path / "v4"

    def test_next_version_dir_defaults_when_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NPEC_ARTIFACT_DIR", str(tmp_path))
        out, version = training_mod._next_version_dir()
        assert version == "v2"


# LocalTrainingService — narrow tests around stop() and start() guards


class TestLocalTrainingService:
    async def test_start_raises_when_already_running(self) -> None:
        db = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: uuid.uuid4()))
        svc = LocalTrainingService(db)
        with pytest.raises(RuntimeError, match="already in progress"):
            await svc.start(None)

    async def test_stop_returns_false_when_run_missing(self) -> None:
        db = AsyncMock()
        db.get = AsyncMock(return_value=None)
        # _get_raw uses execute? actually it uses .get on TrainingRun via repo
        with patch(
            "api.src.repositories.training.TrainingRunRepository.get_raw",
            new_callable=AsyncMock,
            return_value=None,
        ):
            svc = LocalTrainingService(db)
            assert await svc.stop(uuid.uuid4()) is False

    async def test_stop_raises_when_not_running(self) -> None:
        row = SimpleNamespace(status=TrainingRunStatus.COMPLETED.value, pid=None)
        with patch(
            "api.src.repositories.training.TrainingRunRepository.get_raw",
            new_callable=AsyncMock,
            return_value=row,
        ):
            svc = LocalTrainingService(AsyncMock())
            with pytest.raises(ValueError, match="only running"):
                await svc.stop(uuid.uuid4())

    async def test_stop_terminates_and_marks_stopped(self) -> None:
        run_id = uuid.uuid4()
        row = SimpleNamespace(status=TrainingRunStatus.RUNNING.value, pid=None, completed_at=None)
        proc = MagicMock()
        proc.terminate = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        training_mod._running_processes[run_id] = proc
        try:
            db = AsyncMock()
            db.commit = AsyncMock()
            with patch(
                "api.src.repositories.training.TrainingRunRepository.get_raw",
                new_callable=AsyncMock,
                return_value=row,
            ):
                svc = LocalTrainingService(db)
                assert await svc.stop(run_id) is True
            assert row.status == TrainingRunStatus.STOPPED.value
            proc.terminate.assert_called_once()
        finally:
            training_mod._running_processes.pop(run_id, None)

    async def test_get_runs_delegates_to_repo(self) -> None:
        with patch(
            "api.src.repositories.training.TrainingRunRepository.list",
            new_callable=AsyncMock,
            return_value=[],
        ):
            assert await LocalTrainingService(AsyncMock()).get_runs() == []

    async def test_get_run_delegates_to_repo(self) -> None:
        with patch(
            "api.src.repositories.training.TrainingRunRepository.get",
            new_callable=AsyncMock,
            return_value=None,
        ):
            assert await LocalTrainingService(AsyncMock()).get_run(uuid.uuid4()) is None


# reconcile_orphaned_training_runs


class TestReconcile:
    async def test_marks_dead_pid_runs_failed(self, monkeypatch) -> None:
        from datetime import datetime, timedelta, timezone

        running_row = SimpleNamespace(
            status=TrainingRunStatus.RUNNING.value,
            pid=12345,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
        )
        ancient_row = SimpleNamespace(
            status=TrainingRunStatus.RUNNING.value,
            pid=None,
            started_at=datetime.now(timezone.utc) - timedelta(days=30),
            completed_at=None,
        )
        live_row = SimpleNamespace(
            status=TrainingRunStatus.RUNNING.value,
            pid=os.getpid(),
            started_at=datetime.now(timezone.utc),
            completed_at=None,
        )

        db = AsyncMock()
        db.commit = AsyncMock()
        scalars = MagicMock()
        scalars.all.return_value = [running_row, ancient_row, live_row]
        result = MagicMock()
        result.scalars.return_value = scalars
        db.execute = AsyncMock(return_value=result)

        class _CM:
            async def __aenter__(self_inner):
                return db

            async def __aexit__(self_inner, *a):
                return False

        sessionmaker_factory = MagicMock(return_value=_CM())

        # _pid_alive(12345) should return False; _pid_alive(os.getpid()) → True.
        def fake_pid_alive(pid):
            return pid == os.getpid()

        with (
            patch("api.src.db.session.get_sessionmaker", return_value=sessionmaker_factory),
            patch.object(training_mod, "_pid_alive", side_effect=fake_pid_alive),
        ):
            await training_mod.reconcile_orphaned_training_runs()

        assert running_row.status == TrainingRunStatus.FAILED.value
        assert ancient_row.status == TrainingRunStatus.FAILED.value
        assert live_row.status == TrainingRunStatus.RUNNING.value  # untouched
