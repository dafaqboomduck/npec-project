"""Shared fixtures for npec_cv preprocessing tests.

Hybrid fixture strategy:
  - Real images from tests/fixtures/images/ for happy-path / integrative tests
    (extract_petri_dish on realistic dishes, patch_image, restore_mask)
  - Synthetic images at runtime for edge cases that real images can't cover
    (edge-clamping, no-contours fallback, padding-math with known dimensions)

Real image conventions:
  - Files live at tests/fixtures/images/test_im1.png ... test_im5.png
  - If a file is missing, the relevant test is skipped with a clear message
    (rather than failing) so you can iterate while populating the dir
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-import: restore real cv2 if another conftest has mocked it.
# ---------------------------------------------------------------------------
# tests/cli/conftest.py mocks cv2 at sys.modules level (module-import time)
# to dodge an opencv-contrib-python gapi.wip.draw bug. That mock leaks across
# the whole pytest session — when the full suite is collected (e.g. by the
# VS Code test extension), CLI conftest loads first and sys.modules["cv2"]
# becomes a MagicMock for every test that follows. Preprocessing tests need
# the real cv2 to actually exercise the image pipeline, so we restore it
# before any cv2-dependent import in this file runs.
from pathlib import Path
from typing import Callable, Tuple
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Real test image discovery
# ---------------------------------------------------------------------------

# tests/preprocessing/conftest.py → tests/ → tests/fixtures/images/
FIXTURES_IMAGES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
TEST_IMAGE_INDICES = (1, 2, 3, 4, 5)


def _load_test_image(index: int) -> np.ndarray:
    """Load test_im{index}.png from the fixtures dir, or skip the test."""
    path = FIXTURES_IMAGES_DIR / f"test_im{index}.png"
    if not path.exists():
        pytest.skip(f"Real test image not found: {path}. Add it to run this test.")
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"cv2 could not decode test image: {path}")
    return img


# ---------------------------------------------------------------------------
# Real-image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_test_image() -> np.ndarray:
    """Default real test image (test_im1.png) — BGR.

    Use for tests that need a single representative real image.
    """
    return _load_test_image(1)


@pytest.fixture(params=TEST_IMAGE_INDICES, ids=lambda i: f"test_im{i}")
def real_test_image_any(request) -> np.ndarray:
    """Parametrized fixture that cycles through all 5 real test images.

    Use for invariants that should hold for ANY valid input image
    (e.g. "output is always square", "crop_coords are always within bounds").
    Tests using this fixture run 5 times — once per image.
    """
    return _load_test_image(request.param)


@pytest.fixture
def real_grayscale_image(real_test_image) -> np.ndarray:
    """Real test_im1 converted to 2D grayscale.

    Use for tests of the bgr2gray early-return path / 2D input handling.
    """
    return cv2.cvtColor(real_test_image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Synthetic image factories — kept for edge cases real images can't cover
# ---------------------------------------------------------------------------


@pytest.fixture
def make_dish_image() -> Callable[..., np.ndarray]:
    """Return a factory that builds a black canvas with a bright filled rectangle.

    The rectangle simulates a petri dish — Otsu thresholding picks it up as
    a single contour. Use for tests that need exact, known dish dimensions
    or specific edge-touching configurations that real images don't have.

    Usage:
        def test_something(make_dish_image):
            img = make_dish_image(
                shape=(2000, 2000),
                dish_bbox=(500, 500, 1500, 1500),  # x_min, y_min, x_max, y_max
            )
    """

    def _factory(
        shape: Tuple[int, int] = (2000, 2000),
        dish_bbox: Tuple[int, int, int, int] = (500, 500, 1500, 1500),
        channels: int = 3,
        fill_value: int = 200,
        dtype: np.dtype = np.uint8,
    ) -> np.ndarray:
        h, w = shape
        if channels == 1:
            img = np.zeros((h, w), dtype=dtype)
        else:
            img = np.zeros((h, w, channels), dtype=dtype)

        x_min, y_min, x_max, y_max = dish_bbox
        if channels == 1:
            img[y_min:y_max, x_min:x_max] = fill_value
        else:
            img[y_min:y_max, x_min:x_max, :] = fill_value
        return img

    return _factory


@pytest.fixture
def edge_dish_top_image(make_dish_image) -> np.ndarray:
    """Synthetic: dish flush against the top edge."""
    return make_dish_image(shape=(1000, 1000), dish_bbox=(200, 0, 800, 600))


@pytest.fixture
def edge_dish_right_image(make_dish_image) -> np.ndarray:
    """Synthetic: dish flush against the right edge."""
    return make_dish_image(shape=(1000, 1000), dish_bbox=(400, 200, 1000, 800))


@pytest.fixture
def corner_dish_image(make_dish_image) -> np.ndarray:
    """Synthetic: dish in the top-right corner — both clamps trigger."""
    return make_dish_image(shape=(1000, 1000), dish_bbox=(500, 0, 1000, 500))


@pytest.fixture
def empty_image() -> np.ndarray:
    """Synthetic: all-zeros canvas — no contours will be found."""
    return np.zeros((500, 500, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Mocking helpers — used by test_patch_image.py and test_restore_mask.py only
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    """Create an empty file that satisfies the `checkpoint_path.exists()` check.

    InferenceManager only checks existence — not file validity — before
    delegating to `load_model`. With `load_model` mocked, an empty file
    is enough.
    """
    checkpoint = tmp_path / "fake_model.keras"
    checkpoint.touch()
    return checkpoint


@pytest.fixture
def mock_keras_model() -> MagicMock:
    """A MagicMock standing in for a loaded Keras model.

    Tests that need predict() to return a specific shape should configure
    `mock_keras_model.predict.return_value` themselves.
    """
    model = MagicMock()
    model.predict.return_value = None
    return model


@pytest.fixture
def patch_load_model(monkeypatch, mock_keras_model: MagicMock):
    """Patch `load_model` in the inference_manager module to return our mock.

    Returns the mock so tests can assert on it if needed.
    """
    monkeypatch.setattr(
        "src.npec_cv.inference.inference_manager.load_model",
        lambda *args, **kwargs: mock_keras_model,
    )
    return mock_keras_model
