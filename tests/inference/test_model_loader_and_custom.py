"""Tests for npec_cv.helpers.model_loader and inference.custom_functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from npec_cv.helpers.model_loader import (
    EXPECTED_MODEL_FILENAMES,
    ModelDownloadError,
    get_models_dir,
)
from npec_cv.inference.custom_functions import dice_bce_loss, f1, iou

# model_loader


class TestGetModelsDir:
    def test_returns_cache_when_complete(self, tmp_path: Path, monkeypatch) -> None:
        version = "9.9.9"
        cache_root = tmp_path / "cache"
        monkeypatch.setenv("NPEC_MODELS_CACHE_DIR", str(cache_root))
        # Populate cache with both expected files
        target = cache_root / f"v{version}"
        target.mkdir(parents=True)
        for name in EXPECTED_MODEL_FILENAMES:
            (target / name).write_bytes(b"weights")
        # Should not need GITHUB_TOKEN — short-circuits on cache
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        out = get_models_dir(version=version)
        assert out == target

    def test_missing_token_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NPEC_MODELS_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with pytest.raises(ModelDownloadError, match="GITHUB_TOKEN"):
            get_models_dir(version="1.0.0")

    def test_release_404_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NPEC_MODELS_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
        resp = MagicMock()
        resp.status_code = 404
        with patch("requests.get", return_value=resp):
            with pytest.raises(ModelDownloadError, match="not found"):
                get_models_dir(version="1.0.0")

    def test_missing_asset_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NPEC_MODELS_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")

        # Only the anchor asset is present
        release_resp = MagicMock()
        release_resp.status_code = 200
        release_resp.json.return_value = {"assets": [{"name": "anchor_det_model_512px.keras", "url": "https://x"}]}
        with patch("requests.get", return_value=release_resp):
            with pytest.raises(ModelDownloadError, match="missing"):
                get_models_dir(version="1.0.0")

    def test_full_download_writes_files(self, tmp_path: Path, monkeypatch) -> None:
        cache_root = tmp_path / "cache"
        monkeypatch.setenv("NPEC_MODELS_CACHE_DIR", str(cache_root))
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")

        release_resp = MagicMock()
        release_resp.status_code = 200
        release_resp.json.return_value = {
            "assets": [{"name": name, "url": f"https://x/{name}"} for name in EXPECTED_MODEL_FILENAMES]
        }

        download_resp = MagicMock()
        download_resp.status_code = 200
        download_resp.iter_content.return_value = [b"chunk1", b"chunk2"]
        download_resp.raise_for_status.return_value = None

        def fake_get(url, headers=None, stream=False):
            if "/releases/tags/" in url:
                return release_resp
            return download_resp

        with patch("requests.get", side_effect=fake_get):
            out = get_models_dir(version="1.2.3")
        assert out == cache_root / "v1.2.3"
        for name in EXPECTED_MODEL_FILENAMES:
            assert (out / name).exists()
            assert (out / name).read_bytes() == b"chunk1chunk2"

    def test_fallback_version_when_package_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NPEC_MODELS_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")

        with patch(
            "importlib.metadata.version",
            side_effect=__import__("importlib.metadata", fromlist=["PackageNotFoundError"]).PackageNotFoundError(),
        ):
            release_resp = MagicMock()
            release_resp.status_code = 404
            with patch("requests.get", return_value=release_resp):
                with pytest.raises(ModelDownloadError):
                    get_models_dir()  # no explicit version


# custom_functions


class TestCustomFunctions:
    def test_iou_perfect_overlap(self) -> None:
        y_true = np.ones((2, 2), dtype=np.float32)
        y_pred = np.ones((2, 2), dtype=np.float32)
        out = float(iou(y_true, y_pred))
        assert out == pytest.approx(1.0, abs=1e-3)

    def test_iou_zero_overlap(self) -> None:
        y_true = np.ones((2, 2), dtype=np.float32)
        y_pred = np.zeros((2, 2), dtype=np.float32)
        out = float(iou(y_true, y_pred))
        assert out == pytest.approx(0.0, abs=1e-3)

    def test_f1_perfect(self) -> None:
        y_true = np.array([[1, 0], [1, 0]], dtype=np.float32)
        y_pred = np.array([[1, 0], [1, 0]], dtype=np.float32)
        out = float(f1(y_true, y_pred))
        assert out == pytest.approx(1.0, abs=1e-3)

    def test_f1_zero_when_no_overlap(self) -> None:
        y_true = np.array([[1, 0], [1, 0]], dtype=np.float32)
        y_pred = np.array([[0, 1], [0, 1]], dtype=np.float32)
        out = float(f1(y_true, y_pred))
        assert out == pytest.approx(0.0, abs=1e-3)

    def test_dice_bce_loss_runs(self) -> None:
        y_true = np.array([[[[1.0]], [[0.0]]]], dtype=np.float32)
        y_pred = np.array([[[[0.9]], [[0.1]]]], dtype=np.float32)
        loss = dice_bce_loss(y_true, y_pred)
        # Returned tensor — convert to scalar via numpy
        loss_arr = np.asarray(loss)
        # Loss should be a finite, non-negative value
        assert np.all(np.isfinite(loss_arr))
        assert float(np.mean(loss_arr)) >= 0
