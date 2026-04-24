"""Pytest tests for the npec-cv CLI.

Uses Typer's CliRunner so no subprocess is spawned and no TensorFlow
models are loaded (the _build_pipeline helper is monkeypatched in conftest).

Test coverage
-------------
  predict
    - valid single image             → exit 0, JSON object with expected keys
    - valid directory (batch)        → exit 0, JSON array with one entry per image
    - with --confidence option       → pipeline receives custom threshold
    - --output writes file           → file exists, same JSON as stdout
    - invalid path                   → exit 1, error message
    - unsupported file format        → exit 1, error message
    - empty directory                → exit 1, error message
  info
    - prints JSON with package/model keys → exit 0
  version
    - prints a non-empty version string   → exit 0
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# predict — happy paths
# ---------------------------------------------------------------------------


class TestPredictSingleImage:
    """Tests for the predict command with a single image input."""

    def test_exit_code_is_zero(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that predicting a valid image returns exit code 0."""
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir)])
        assert result.exit_code == 0, result.output

    def test_output_is_valid_json(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that the command output is a valid JSON object."""
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir)])
        payload = json.loads(result.output)
        assert isinstance(payload, dict)

    def test_output_contains_required_keys(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that the output JSON contains all required top-level keys."""
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir)])
        payload = json.loads(result.output)
        assert "image" in payload
        assert "num_detected" in payload
        assert "total_plants" in payload
        assert "plants" in payload

    def test_plant_entries_contain_required_keys(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that each plant entry in the output contains all required keys."""
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir)])
        payload = json.loads(result.output)
        for plant in payload["plants"].values():
            assert "detected" in plant
            assert "length_px" in plant
            assert "tip_location" in plant
            assert "confidence_score" in plant
            assert "mask_available" in plant

    def test_detected_plant_has_tip_location(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that detected plants have a valid two-element tip_location."""
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir)])
        payload = json.loads(result.output)
        for plant in payload["plants"].values():
            if plant["detected"]:
                assert plant["tip_location"] is not None
                assert len(plant["tip_location"]) == 2

    def test_pipeline_called_once(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that process_single_image is called exactly once for a single image."""
        runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir)])
        mock_pipeline.process_single_image.assert_called_once()


class TestPredictDirectory:
    """Tests for the predict command with a directory of images as input."""

    def test_exit_code_is_zero(self, mock_pipeline, image_directory, artifact_dir):
        """Verify that predicting a valid image directory returns exit code 0."""
        result = runner.invoke(app, ["predict", str(image_directory), "-a", str(artifact_dir)])
        assert result.exit_code == 0, result.output

    def test_output_is_json_array(self, mock_pipeline, image_directory, artifact_dir):
        """Verify that the output for a directory input is a JSON array."""
        result = runner.invoke(app, ["predict", str(image_directory), "-a", str(artifact_dir)])
        payload = json.loads(result.output)
        assert isinstance(payload, list)

    def test_one_entry_per_image(self, mock_pipeline, image_directory, artifact_dir):
        """Verify that the output array contains exactly one entry per image in the directory."""
        num_images = len(list(image_directory.glob("*.png")))
        result = runner.invoke(app, ["predict", str(image_directory), "-a", str(artifact_dir)])
        payload = json.loads(result.output)
        assert len(payload) == num_images

    def test_pipeline_called_for_each_image(self, mock_pipeline, image_directory, artifact_dir):
        """Verify that process_single_image is called once for each image in the directory."""
        num_images = len(list(image_directory.glob("*.png")))
        runner.invoke(app, ["predict", str(image_directory), "-a", str(artifact_dir)])
        assert mock_pipeline.process_single_image.call_count == num_images


class TestPredictConfidenceOption:
    """Tests for the --confidence / -c option of the predict command."""

    def test_confidence_option_accepted(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that the --confidence flag is accepted without errors."""
        result = runner.invoke(app, ["predict", str(valid_image), "--confidence", "0.05", "-a", str(artifact_dir)])
        assert result.exit_code == 0, result.output

    def test_short_flag_accepted(self, mock_pipeline, valid_image, artifact_dir):
        """Verify that the short -c flag is accepted without errors."""
        result = runner.invoke(app, ["predict", str(valid_image), "-c", "0.1", "-a", str(artifact_dir)])
        assert result.exit_code == 0, result.output

    def test_custom_confidence_passed_to_build_pipeline(self, monkeypatch, valid_image, artifact_dir):
        """Verify _build_pipeline receives the --confidence value."""
        received: list[float] = []
        received_artifact_dirs: list[str] = []

        def fake_build(confidence: float, model_dir):
            received.append(confidence)
            received_artifact_dirs.append(str(model_dir))
            pipeline = MagicMock()
            from tests.cli.conftest import _make_image_result

            pipeline.process_single_image.return_value = _make_image_result()
            return pipeline

        monkeypatch.setattr("src.cli.main._build_pipeline", fake_build)
        runner.invoke(app, ["predict", str(valid_image), "--confidence", "0.07", "-a", str(artifact_dir)])
        assert received == [pytest.approx(0.07)]
        assert received_artifact_dirs == [str(artifact_dir)]


class TestPredictOutputFile:
    """Tests for the --output option that writes results to a file."""

    def test_writes_json_file(self, mock_pipeline, valid_image, artifact_dir, tmp_path):
        """Verify that the --output flag creates the specified file."""
        out_file = tmp_path / "results.json"
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir), "--output", str(out_file)])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_file_content_is_valid_json(self, mock_pipeline, valid_image, artifact_dir, tmp_path):
        """Verify that the file written by --output contains valid JSON with expected keys."""
        out_file = tmp_path / "results.json"
        runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir), "--output", str(out_file)])
        payload = json.loads(out_file.read_text())
        assert "image" in payload

    def test_confirmation_message_shown(self, mock_pipeline, valid_image, artifact_dir, tmp_path):
        """Verify that the output path is mentioned in the confirmation message."""
        out_file = tmp_path / "results.json"
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir), "--output", str(out_file)])
        assert str(out_file) in result.output


# ---------------------------------------------------------------------------
# predict — error paths
# ---------------------------------------------------------------------------


class TestPredictErrors:
    """Tests for error handling in the predict command."""

    def test_missing_artifact_dir_exits_nonzero(self, valid_image):
        """Verify that --artifact-dir is required for predict."""
        result = runner.invoke(app, ["predict", str(valid_image)])
        assert result.exit_code != 0

    def test_missing_artifact_dir_message(self, valid_image):
        """Verify that missing --artifact-dir produces a required-option message."""
        result = runner.invoke(app, ["predict", str(valid_image)])
        assert "--artifact-dir" in result.output

    def test_artifact_dir_missing_models_exits_1(self, valid_image, artifact_dir, monkeypatch):
        """Verify that predict surfaces model-resolution failures as CLI errors."""
        monkeypatch.setattr(
            "src.cli.main._build_pipeline",
            lambda confidence, model_dir: (_ for _ in ()).throw(FileNotFoundError("Missing root model")),
        )
        result = runner.invoke(app, ["predict", str(valid_image), "-a", str(artifact_dir)])
        assert result.exit_code == 1
        assert "Missing root model" in result.output

    def test_nonexistent_path_exits_1(self, artifact_dir):
        """Verify that a non-existent input path causes exit code 1."""
        result = runner.invoke(app, ["predict", "/nonexistent/path/image.png", "-a", str(artifact_dir)])
        assert result.exit_code == 1

    def test_nonexistent_path_error_message(self, artifact_dir):
        """Verify that a non-existent path produces a 'does not exist' error message."""
        result = runner.invoke(app, ["predict", "/nonexistent/path/image.png", "-a", str(artifact_dir)])
        assert "does not exist" in result.output

    def test_unsupported_format_exits_1(self, tmp_path, artifact_dir):
        """Verify that an unsupported file format causes exit code 1."""
        txt_file = tmp_path / "not_an_image.txt"
        txt_file.write_text("hello")
        result = runner.invoke(app, ["predict", str(txt_file), "-a", str(artifact_dir)])
        assert result.exit_code == 1

    def test_unsupported_format_error_message(self, tmp_path, artifact_dir):
        """Verify that an unsupported file extension produces an 'unsupported' error message."""
        txt_file = tmp_path / "data.csv"
        txt_file.write_text("a,b,c")
        result = runner.invoke(app, ["predict", str(txt_file), "-a", str(artifact_dir)])
        assert "unsupported" in result.output.lower()

    def test_empty_directory_exits_1(self, tmp_path, artifact_dir):
        """Verify that an empty directory causes exit code 1."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(app, ["predict", str(empty_dir), "-a", str(artifact_dir)])
        assert result.exit_code == 1

    def test_empty_directory_error_message(self, tmp_path, artifact_dir):
        """Verify that an empty directory produces a 'no supported images' error message."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(app, ["predict", str(empty_dir), "-a", str(artifact_dir)])
        assert "no supported images" in result.output

    def test_file_too_large_exits_1(self, tmp_path, monkeypatch, artifact_dir):
        """Simulate a file that exceeds MAX_FILE_SIZE_BYTES."""
        from unittest.mock import MagicMock

        img_path = tmp_path / "big.png"
        img_path.write_bytes(b"x")

        fake_stat_result = MagicMock()
        fake_stat_result.st_size = 51 * 1024 * 1024
        monkeypatch.setattr("src.cli.main.Path.stat", lambda self: fake_stat_result)
        result = runner.invoke(app, ["predict", str(img_path), "-a", str(artifact_dir)])
        assert result.exit_code == 1

    def test_image_too_small_exits_1(self, tmp_path, artifact_dir):
        """Verify that an image below the 512px minimum causes exit code 1."""
        from PIL import Image

        img_path = tmp_path / "tiny.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(str(img_path))
        result = runner.invoke(app, ["predict", str(img_path), "-a", str(artifact_dir)])
        assert result.exit_code == 1

    def test_image_too_small_error_message(self, tmp_path, artifact_dir):
        """Verify that an undersized image produces an error message referencing the size limit."""
        from PIL import Image

        img_path = tmp_path / "tiny.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(str(img_path))
        result = runner.invoke(app, ["predict", str(img_path), "-a", str(artifact_dir)])
        assert "512" in result.output or "minimum" in result.output.lower()


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


class TestInfo:
    """Tests for the info command."""

    def test_exit_code_is_zero(self):
        """Verify that the info command returns exit code 0."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0, result.output

    def test_output_is_valid_json(self):
        """Verify that the info command output is a valid JSON object."""
        result = runner.invoke(app, ["info"])
        payload = json.loads(result.output)
        assert isinstance(payload, dict)

    def test_contains_package_key(self):
        """Verify that the info output contains the correct package name."""
        result = runner.invoke(app, ["info"])
        payload = json.loads(result.output)
        assert payload["package"] == "npec_cv"

    def test_contains_version_key(self):
        """Verify that the info output contains a version key."""
        result = runner.invoke(app, ["info"])
        payload = json.loads(result.output)
        assert "version" in payload

    def test_contains_models_key(self):
        """Verify that the info output contains model entries for root and anchor detection."""
        result = runner.invoke(app, ["info"])
        payload = json.loads(result.output)
        assert "models" in payload
        assert "root_detection" in payload["models"]
        assert "anchor_detection" in payload["models"]

    def test_model_entries_have_required_keys(self):
        """Verify that each model entry contains all required configuration keys."""
        result = runner.invoke(app, ["info"])
        payload = json.loads(result.output)
        for model_info in payload["models"].values():
            assert "checkpoint" in model_info
            assert "patch_size" in model_info
            assert "batch_size" in model_info
            assert "default_threshold" in model_info

    def test_contains_supported_formats(self):
        """Verify that the info output lists accepted file extensions."""
        result = runner.invoke(app, ["info"])
        payload = json.loads(result.output)
        assert "accepted_extensions" in payload
        assert isinstance(payload["accepted_extensions"], list)


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------


class TestVersion:
    """Tests for the version command."""

    def test_exit_code_is_zero(self):
        """Verify that the version command returns exit code 0."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0, result.output

    def test_prints_non_empty_string(self):
        """Verify that the version command prints a non-empty string."""
        result = runner.invoke(app, ["version"])
        assert result.output.strip() != ""

    def test_version_is_semver_like(self):
        """Version should look like X.Y.Z (or 'unknown' if package not installed)."""
        result = runner.invoke(app, ["version"])
        ver = result.output.strip()
        assert ver == "unknown" or ver[0].isdigit(), f"Unexpected version format: {ver!r}"


# ---------------------------------------------------------------------------
# --help flag
# ---------------------------------------------------------------------------


class TestHelp:
    """Tests for the --help flag across all commands."""

    def test_root_help_exits_zero(self):
        """Verify that the root --help flag returns exit code 0."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_root_help_lists_commands(self):
        """Verify that the root --help output lists all available commands."""
        result = runner.invoke(app, ["--help"])
        assert "predict" in result.output
        assert "info" in result.output
        assert "version" in result.output

    def test_predict_help_exits_zero(self):
        """Verify that predict --help returns exit code 0."""
        result = runner.invoke(app, ["predict", "--help"])
        assert result.exit_code == 0

    def test_predict_help_shows_options(self):
        """Verify that predict --help lists confidence, artifact-dir, and output options."""
        result = runner.invoke(app, ["predict", "--help"])
        assert "--confidence" in result.output
        assert "--artifact-dir" in result.output
        assert "--output" in result.output

    def test_info_help_exits_zero(self):
        """Verify that info --help returns exit code 0."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0

    def test_version_help_exits_zero(self):
        """Verify that version --help returns exit code 0."""
        result = runner.invoke(app, ["version", "--help"])
        assert result.exit_code == 0
