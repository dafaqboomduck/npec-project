"""Additional CLI branch coverage tests.

Covers download-models, predict error paths (extension, size, missing path,
unreadable image, resolution), and info with custom artifact dir.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from npec_cv.cli.main import app
from PIL import Image
from typer.testing import CliRunner

runner = CliRunner()


def _png(width: int = 600, height: int = 600) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), 128).save(buf, format="PNG")
    return buf.getvalue()


def _write_png(path: Path, width: int = 600, height: int = 600) -> Path:
    path.write_bytes(_png(width, height))
    return path


def _fake_result(image_name: str = "scan"):
    pr = SimpleNamespace(
        detected=True,
        length=123,
        end_point=(10, 20),
        threshold_used=0.01,
        pixel_count=400,
        mask=None,
    )
    return SimpleNamespace(
        image_name=image_name,
        num_detected=1,
        total_plants=1,
        plant_results={1: pr},
    )


def _fake_pipeline(result=None):
    pipe = MagicMock()
    pipe.process_single_image.return_value = result or _fake_result()
    return pipe


class TestDownloadModelsCommand:
    def test_calls_get_models_dir(self, tmp_path: Path) -> None:
        with patch("npec_cv.helpers.model_loader.get_models_dir", return_value=tmp_path):
            out = runner.invoke(app, ["download-models"])
        assert out.exit_code == 0
        assert str(tmp_path) in out.stdout

    def test_force_flag_propagates(self, tmp_path: Path) -> None:
        with patch("npec_cv.helpers.model_loader.get_models_dir", return_value=tmp_path) as m:
            runner.invoke(app, ["download-models", "--force"])
        m.assert_called_with(force_download=True)


class TestPredictErrorPaths:
    def test_path_does_not_exist(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        out = runner.invoke(app, ["predict", str(tmp_path / "nope.png"), "-a", str(models)])
        assert out.exit_code != 0

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        bad = tmp_path / "x.txt"
        bad.write_bytes(b"hi")
        out = runner.invoke(app, ["predict", str(bad), "-a", str(models)])
        assert out.exit_code == 1
        assert "unsupported file format" in out.stderr.lower()

    def test_too_large(self, tmp_path: Path, monkeypatch) -> None:
        models = tmp_path / "models"
        models.mkdir()
        # Patch MAX_FILE_SIZE_BYTES to 100 bytes so a real PNG trips the limit
        img = _write_png(tmp_path / "scan.png")
        monkeypatch.setattr("npec_cv.cli.main.MAX_FILE_SIZE_BYTES", 100)
        out = runner.invoke(app, ["predict", str(img), "-a", str(models)])
        assert out.exit_code == 1
        assert "exceed" in out.stderr.lower() or "limit" in out.stderr.lower()

    def test_resolution_too_small(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        small = _write_png(tmp_path / "tiny.png", 100, 100)
        out = runner.invoke(app, ["predict", str(small), "-a", str(models)])
        assert out.exit_code == 1
        assert "must be" in out.stderr.lower() or "minimum" in out.stderr.lower()

    def test_resolution_too_large(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        huge = _write_png(tmp_path / "huge.png", 6000, 6000)
        out = runner.invoke(app, ["predict", str(huge), "-a", str(models)])
        assert out.exit_code == 1

    def test_corrupt_image(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not really a png")
        out = runner.invoke(app, ["predict", str(bad), "-a", str(models)])
        assert out.exit_code == 1

    def test_predict_single_image_happy(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        img = _write_png(tmp_path / "scan.png")
        with patch("npec_cv.cli.main._build_pipeline", return_value=_fake_pipeline()):
            out = runner.invoke(app, ["predict", str(img), "-a", str(models)])
        assert out.exit_code == 0
        assert '"num_detected"' in out.stdout

    def test_predict_directory_no_images(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        empty = tmp_path / "empty"
        empty.mkdir()
        out = runner.invoke(app, ["predict", str(empty), "-a", str(models)])
        assert out.exit_code == 1
        assert "no supported images" in out.stderr.lower()

    def test_predict_directory_processes_each(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        d = tmp_path / "ims"
        d.mkdir()
        _write_png(d / "a.png")
        _write_png(d / "b.png")
        with patch("npec_cv.cli.main._build_pipeline", return_value=_fake_pipeline()):
            out = runner.invoke(app, ["predict", str(d), "-a", str(models)])
        assert out.exit_code == 0
        # batch → JSON array, so two entries
        assert out.stdout.count('"num_detected"') == 2

    def test_predict_writes_output_file(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        img = _write_png(tmp_path / "scan.png")
        out_file = tmp_path / "out.json"
        with patch("npec_cv.cli.main._build_pipeline", return_value=_fake_pipeline()):
            out = runner.invoke(app, ["predict", str(img), "-a", str(models), "--output", str(out_file)])
        assert out.exit_code == 0
        assert out_file.exists()
        assert '"num_detected"' in out_file.read_text(encoding="utf-8")

    def test_predict_handles_pipeline_build_failure(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        img = _write_png(tmp_path / "scan.png")
        with patch(
            "npec_cv.cli.main._build_pipeline",
            side_effect=FileNotFoundError("missing artifact"),
        ):
            out = runner.invoke(app, ["predict", str(img), "-a", str(models)])
        assert out.exit_code == 1


class TestInfoCommand:
    def test_default_info_runs(self) -> None:
        out = runner.invoke(app, ["info"])
        assert out.exit_code == 0
        assert '"package"' in out.stdout
        assert '"models"' in out.stdout

    def test_info_with_artifact_dir(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        # Create the required model files
        (models / "root_det_model_512px.keras").write_bytes(b"")
        (models / "anchor_det_model_512px.keras").write_bytes(b"")
        out = runner.invoke(app, ["info", "-a", str(models)])
        assert out.exit_code == 0


class TestConfigsInference:
    def test_build_inference_config_resolves_paths(self, tmp_path: Path) -> None:
        from npec_cv.configs.inference import build_inference_config

        (tmp_path / "root_det_model_512px.keras").write_bytes(b"")
        (tmp_path / "anchor_det_model_512px.keras").write_bytes(b"")
        root_cfg, anchor_cfg = build_inference_config(tmp_path)
        assert "root_det" in str(root_cfg.checkpoint)
        assert "anchor_det" in str(anchor_cfg.checkpoint)

    def test_build_inference_config_accepts_legacy_names(self, tmp_path: Path) -> None:
        from npec_cv.configs.inference import build_inference_config

        (tmp_path / "root_model.keras").write_bytes(b"")
        (tmp_path / "anchor_model.keras").write_bytes(b"")
        root_cfg, anchor_cfg = build_inference_config(tmp_path)
        assert "root_model" in str(root_cfg.checkpoint)
        assert "anchor_model" in str(anchor_cfg.checkpoint)

    def test_build_inference_config_raises_for_missing_root(self, tmp_path: Path) -> None:
        from npec_cv.configs.inference import build_inference_config

        (tmp_path / "anchor_det_model_512px.keras").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="Missing root"):
            build_inference_config(tmp_path)

    def test_build_inference_config_raises_for_missing_anchor(self, tmp_path: Path) -> None:
        from npec_cv.configs.inference import build_inference_config

        (tmp_path / "root_det_model_512px.keras").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="Missing anchor"):
            build_inference_config(tmp_path)

    def test_build_inference_config_raises_for_missing_dir(self, tmp_path: Path) -> None:
        from npec_cv.configs.inference import build_inference_config

        with pytest.raises(ValueError, match="existing directory"):
            build_inference_config(tmp_path / "nope")


class TestPipelineHelpers:
    def test_cv_pipeline_raises_for_missing_dir(self, tmp_path: Path) -> None:
        from npec_cv.pipeline import cv_pipeline

        fake_master = MagicMock()
        fake_master.num_plants = 5
        with patch("npec_cv.pipeline.PlantMaster", return_value=fake_master):
            with pytest.raises(ValueError, match="does not exist"):
                cv_pipeline(tmp_path / "nope")

    def test_cv_pipeline_raises_when_no_images(self, tmp_path: Path) -> None:
        from npec_cv.pipeline import cv_pipeline

        fake_master = MagicMock()
        fake_master.num_plants = 5
        with patch("npec_cv.pipeline.PlantMaster", return_value=fake_master):
            with pytest.raises(ValueError, match="No valid images"):
                cv_pipeline(tmp_path)

    def test_pipeline_init_failure_raises(self) -> None:
        from npec_cv.exceptions import InitializationError
        from npec_cv.pipeline import CVPipeline

        with patch("npec_cv.pipeline.PlantMaster", side_effect=RuntimeError("boom")):
            with pytest.raises(InitializationError):
                CVPipeline()

    def test_process_single_image_raises_for_missing(self, tmp_path: Path) -> None:
        from npec_cv.pipeline import CVPipeline

        fake_master = MagicMock()
        fake_master.num_plants = 5
        with patch("npec_cv.pipeline.PlantMaster", return_value=fake_master):
            pipe = CVPipeline()
        # cv2.imread returns None for missing file → ValueError
        with pytest.raises(ValueError, match="Could not load"):
            pipe.process_single_image(tmp_path / "nope.png")

    def test_process_single_image_returns_result(self, tmp_path: Path) -> None:
        from npec_cv.pipeline import CVPipeline

        fake_master = MagicMock()
        fake_master.num_plants = 5
        fake_master.process_image.return_value = _fake_result()
        img_path = _write_png(tmp_path / "scan.png")
        with patch("npec_cv.pipeline.PlantMaster", return_value=fake_master):
            pipe = CVPipeline()
        out = pipe.process_single_image(img_path)
        assert out.num_detected == 1

    def test_process_directory_csv_path(self, tmp_path: Path) -> None:
        from npec_cv.pipeline import CVPipeline

        fake_master = MagicMock()
        fake_master.num_plants = 5
        fake_master.process_image.return_value = _fake_result()
        ims = tmp_path / "ims"
        ims.mkdir()
        _write_png(ims / "a.png")
        _write_png(ims / "b.png")
        with patch("npec_cv.pipeline.PlantMaster", return_value=fake_master):
            pipe = CVPipeline()
            out = pipe.process_directory(ims)
        assert out.suffix == ".csv"
        assert out.exists()
