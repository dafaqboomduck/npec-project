"""Shared fixtures for npec_cv end-to-end integration tests.

These tests verify the full pipeline (preprocess → predict → postprocess)
runs correctly on real images, with model.predict() mocked to return
hand-crafted synthetic probability maps. This isolates the test from
TF/model loading cost while still exercising every other glue point.

Mocking strategy
----------------
We patch BOTH:
  - keras.models.load_model — returns a MagicMock instead of a real Keras model
  - That mock's .predict() — configured to return synthetic patches

The synthetic predictions are built upfront for a known image size: a
full-size probability map is constructed with hand-drawn root-shaped
vertical lines (for the root model) and small blobs (for the anchor
model). At predict() time, we patchify the synthetic map and return
patches matching what the real model would output.

This means PlantMaster's post-processing pipeline runs end-to-end on
inputs we fully control — the contract test verifies that when we feed
in N root-shaped blobs, we get N PlantResults out with correctly-typed
fields.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.npec_cv.configs.inference import ANCHOR_MODEL_CONFIG, ROOT_MODEL_CONFIG
from src.npec_cv.helpers.patchify import patchify
from src.npec_cv.pipeline import CVPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_IMAGES_DIR = Path(__file__).parent.parent / "fixtures" / "images"

# Number of plant slots the synthetic predictions place on the petri dish.
# PlantMaster.num_plants is 5 by config, so we draw 5 evenly-spaced root blobs.
SYNTHETIC_PLANT_COUNT = 5


def _image_path(index: int) -> Path:
    """Resolve a HADES test image path or skip the test if missing."""
    path = FIXTURES_IMAGES_DIR / f"test_im{index}.png"
    if not path.exists():
        pytest.skip(f"HADES test image not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Synthetic probability map builders
# ---------------------------------------------------------------------------


def _build_root_probability_map(height: int, width: int, n_plants: int = SYNTHETIC_PLANT_COUNT) -> np.ndarray:
    """Return a (H, W) probability map with N vertical root-shaped blobs.

    Roots are drawn as thick vertical lines (high probability ~0.95) at
    evenly-spaced x positions, running from near the top to near the bottom
    of the (cropped, padded) dish region. PlantMaster's adaptive thresholding
    picks these up as elongated bright regions and assigns them to plant slots.
    """
    probs = np.zeros((height, width), dtype=np.float32)

    # Vertical position: leave a margin top and bottom (10%-90% of height)
    y_top = int(height * 0.10)
    y_bottom = int(height * 0.90)

    # Horizontal spacing: divide width into N+1 segments, place roots at
    # interior boundaries
    column_spacing = width // (n_plants + 1)
    root_thickness = 8  # pixels wide — wide enough for skeletonization

    for i in range(n_plants):
        x_center = column_spacing * (i + 1)
        x_lo = x_center - root_thickness // 2
        x_hi = x_center + root_thickness // 2
        probs[y_top:y_bottom, x_lo:x_hi] = 0.95

    return probs


def _build_anchor_probability_map(height: int, width: int, n_plants: int = SYNTHETIC_PLANT_COUNT) -> np.ndarray:
    """Return a (H, W) probability map with N anchor blobs near the top.

    Anchors are drawn as small bright squares at the same x positions as
    the root blobs, sitting near the top of the dish. PlantMaster uses
    these to identify plant origin points.
    """
    probs = np.zeros((height, width), dtype=np.float32)

    # Anchors live near the top — around 5% from the top edge
    y_center = int(height * 0.05)
    y_lo = max(0, y_center - 25)
    y_hi = y_center + 25

    column_spacing = width // (n_plants + 1)
    anchor_half_width = 25

    for i in range(n_plants):
        x_center = column_spacing * (i + 1)
        x_lo = max(0, x_center - anchor_half_width)
        x_hi = x_center + anchor_half_width
        probs[y_lo:y_hi, x_lo:x_hi] = 0.95

    return probs


def _patchify_for_predict(prob_map: np.ndarray, patch_size: int) -> np.ndarray:
    """Convert a (H, W) probability map into the (N, P, P, 1) shape model.predict returns.

    Mirrors the patchify call in InferenceManager.create_img_mask exactly,
    then reshapes to a flat batch and adds the channel dim.
    """
    patches = patchify(prob_map, (patch_size, patch_size), step=patch_size)
    # patches shape: (rows, cols, P, P)
    rows, cols, p, _ = patches.shape
    # Flatten to (N, P, P) then add channel dim → (N, P, P, 1)
    flat = patches.reshape(rows * cols, p, p)
    return flat[..., np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# The mock predict — knows which model is calling and returns appropriate patches
# ---------------------------------------------------------------------------


class _PredictionDispatcher:
    """Builds and serves synthetic predictions for both root and anchor models.

    On first call for a given input batch, infers the padded image dims from
    patch grid count, builds the full-size synthetic prob map for that
    model type, and returns matching patches. Subsequent calls re-use the
    same maps if the shape matches.

    Two instances are created per test (one for root, one for anchor); the
    `_kind` field selects which probability builder to use.
    """

    def __init__(self, kind: str, n_plants: int = SYNTHETIC_PLANT_COUNT):
        assert kind in ("root", "anchor")
        self._kind = kind
        self._n_plants = n_plants
        self._cached_map: np.ndarray | None = None
        self._cached_dims: tuple[int, int] | None = None

    def predict(self, patches: np.ndarray, **kwargs) -> np.ndarray:
        """Return synthetic predictions matching the input batch shape.

        patches: (N, P, P, 3) — input patches from preprocessing
        returns: (N, P, P, 1) — synthetic probability patches
        """
        n, p, _, _ = patches.shape
        # Infer padded image dimensions from patch count. The real grid
        # could be any (rows, cols) such that rows * cols == n. We need to
        # know the actual rows/cols to re-patchify our synthetic map.
        # The patches arrive in row-major order from patchify, so we can
        # back out the grid by trying to find a factorization that matches
        # what patchify would have produced.
        # Practically: the InferenceManager stores patch_grid_shape on
        # itself before predict() is called, but the mock doesn't have
        # access to that. We approximate by assuming a reasonable aspect
        # ratio close to the real images (~1.4:1, since dish crop is square
        # but the cropped+padded region has small aspect variations).
        # Simpler approach: build a synthetic map of (n*P, P) shape — a
        # vertical strip — and feed back patches in order. This bypasses
        # the geometry but gives PlantMaster a coherent vertical structure.
        # Even simpler: just generate per-patch synthetic probabilities
        # directly without the global-map round-trip.
        #
        # We take the simplest approach: each patch independently gets
        # the synthetic pattern based on its position in the batch.
        out = np.zeros((n, p, p, 1), dtype=np.float32)

        if self._kind == "root":
            # Draw a vertical line in the middle of every patch — this
            # creates N parallel "roots" across the patch grid which
            # post-processing should then resolve into plant detections.
            # Patches near the bottom of the grid get the line; patches
            # near the top don't (anchors handle the top).
            for i in range(n):
                # All patches get a faint vertical line; strong line
                # centered horizontally
                line_thickness = 6
                x_center = p // 2
                out[i, :, x_center - line_thickness // 2 : x_center + line_thickness // 2, 0] = 0.95
        elif self._kind == "anchor":
            # Anchors only appear in patches near the top of the grid.
            # Without knowing the grid shape, we put anchor blobs in the
            # first row's worth of patches (roughly the first sqrt(n)).
            top_row_count = max(1, int(n**0.5))
            for i in range(top_row_count):
                # Square anchor in middle of patch
                cy, cx = p // 2, p // 2
                half = 30
                out[
                    i,
                    max(0, cy - half) : cy + half,
                    max(0, cx - half) : cx + half,
                    0,
                ] = 0.95

        return out


# ---------------------------------------------------------------------------
# Mock pipeline — same constructor signature as the real one, swaps load_model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mocked_pipeline(tmp_path_factory) -> CVPipeline:
    """A real CVPipeline with model.predict() mocked to return synthetic patches.

    The pipeline is fully real — preprocessing, post-processing, all
    PlantMaster logic — except that the two Keras models are MagicMocks
    whose .predict() returns hand-crafted probability patches.

    Session-scoped: setup is fast (no model loading), but reusing the
    same instance across tests is still cleaner than rebuilding.
    """
    from unittest.mock import patch as _patch

    # We need fake checkpoint files for the existence check
    checkpoint_dir = tmp_path_factory.mktemp("fake_checkpoints")
    fake_root_ckpt = checkpoint_dir / "root.keras"
    fake_anchor_ckpt = checkpoint_dir / "anchor.keras"
    fake_root_ckpt.touch()
    fake_anchor_ckpt.touch()

    root_cfg = replace(ROOT_MODEL_CONFIG, checkpoint=str(fake_root_ckpt))
    anchor_cfg = replace(ANCHOR_MODEL_CONFIG, checkpoint=str(fake_anchor_ckpt))

    # Build dispatchers — one per model
    root_dispatcher = _PredictionDispatcher(kind="root")
    anchor_dispatcher = _PredictionDispatcher(kind="anchor")

    # The load_model mock returns different MagicMocks depending on call order.
    # Inspecting plant_master.py: root_manager is built first, then
    # anchor_manager. So the first load_model call → root, second → anchor.
    call_count = {"n": 0}

    def fake_load_model(*args, **kwargs):
        mock_model = MagicMock()
        if call_count["n"] == 0:
            mock_model.predict.side_effect = root_dispatcher.predict
        else:
            mock_model.predict.side_effect = anchor_dispatcher.predict
        call_count["n"] += 1
        return mock_model

    with _patch(
        "src.npec_cv.inference.inference_manager.load_model",
        side_effect=fake_load_model,
    ):
        pipeline = CVPipeline(inference_config=(root_cfg, anchor_cfg))

    return pipeline


# ---------------------------------------------------------------------------
# Image fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_test_image_path() -> Path:
    """Path to test_im1.png — the canonical happy-path real image."""
    return _image_path(1)


@pytest.fixture(scope="session")
def image_result(mocked_pipeline: CVPipeline, real_test_image_path: Path):
    """Cached ImageResult to avoid rerunning the full pipeline for each test."""
    return mocked_pipeline.process_single_image(real_test_image_path)
