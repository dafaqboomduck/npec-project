"""Shared pytest fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — build mock pipeline objects
# ---------------------------------------------------------------------------


def _make_plant_result(
    plant_id: int,
    detected: bool = True,
    length: int = 350,
    end_point: tuple[int, int] = (420, 180),
    threshold_used: float = 0.45,
    pixel_count: int = 1024,
) -> MagicMock:
    """Return a mock PlantResult."""
    pr = MagicMock()
    pr.plant_id = plant_id
    pr.detected = detected
    pr.length = length if detected else None
    pr.end_point = end_point if detected else None
    pr.threshold_used = threshold_used if detected else None
    pr.pixel_count = pixel_count if detected else None
    pr.mask = np.zeros((512, 512), dtype=np.uint8) if detected else None
    return pr


def _make_image_result(image_name: str = "test_image", num_plants: int = 2) -> MagicMock:
    """Return a mock ImageResult with `num_plants` plant entries."""
    result = MagicMock()
    result.image_name = image_name
    result.plant_results = {i: _make_plant_result(i, detected=True) for i in range(num_plants)}
    result.num_detected = num_plants
    result.total_plants = num_plants
    return result


def _make_pipeline(image_result: Any = None) -> MagicMock:
    """Return a mock CVPipeline whose process_single_image returns image_result."""
    pipeline = MagicMock()
    pipeline.process_single_image.return_value = image_result or _make_image_result()
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_pipeline(monkeypatch) -> MagicMock:
    """Patch _build_pipeline so no TensorFlow models are loaded during tests."""
    pipeline = _make_pipeline()
    monkeypatch.setattr("npec_cv.cli.main._build_pipeline", lambda confidence, artifact_dir: pipeline)
    return pipeline


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    """Create a model directory accepted by the CLI --artifact-dir option."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "root_det_model_512px.keras").write_text("", encoding="utf-8")
    (models / "anchor_det_model_512px.keras").write_text("", encoding="utf-8")
    return models


@pytest.fixture()
def valid_image(tmp_path: Path) -> Path:
    """Write a 512x512 PNG to a temp directory and return its path.

    512px satisfies the MIN_RESOLUTION constraint.
    Uses Pillow instead of cv2 to avoid pulling in the real cv2 library.
    """
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "plant_01.png"
    img_array = np.zeros((512, 512, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(str(img_path))
    return img_path


@pytest.fixture()
def image_directory(tmp_path: Path) -> Path:
    """Create a temp directory containing two 512x512 PNG images."""
    from PIL import Image  # noqa: PLC0415

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("plant_01.png", "plant_02.png"):
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(str(img_dir / name))
    return img_dir
