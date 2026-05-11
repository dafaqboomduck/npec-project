"""Shared fixtures for npec_cv prediction inference tests.

These tests run REAL inference on REAL HADES images using the production
CVPipeline. Model loading is slow (~30s) and inference per image is also
slow (~25s), so all expensive fixtures are session-scoped — every unique
operation happens exactly once per pytest invocation.

Image expectations (from project domain knowledge):
  test_im1 → 5 plants detected  (full happy path, Arabidopsis)
  test_im2 → 4 plants detected  (Arabidopsis)
  test_im3 → 0 plants detected  (potato root — see note on EXPECTED_DETECTIONS)
  test_im4 → 0 plants detected  (intentionally empty image)
  test_im5 → 5 plants detected  (Arabidopsis)

Model checkpoint resolution:
  The default ROOT_MODEL_CONFIG / ANCHOR_MODEL_CONFIG store BARE filenames
  (e.g. "root_det_model_512px.keras") and rely on the caller's cwd to
  resolve them. That assumption breaks for tests run from anywhere other
  than the artifacts directory. We work around this by building configs
  with absolute checkpoint paths in the `pipeline` fixture below — a
  test-only workaround that doesn't touch package code.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from npec_cv.configs.inference import ANCHOR_MODEL_CONFIG, ROOT_MODEL_CONFIG
from npec_cv.models import ImageResult
from npec_cv.pipeline import CVPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# tests/inference/conftest.py → tests/ → repo root → src/npec_cv/artifacts/models
REPO_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_MODELS_DIR = REPO_ROOT / "src" / "npec_cv" / "artifacts" / "models"

ROOT_CHECKPOINT = ARTIFACTS_MODELS_DIR / "root_det_model_512px.keras"
ANCHOR_CHECKPOINT = ARTIFACTS_MODELS_DIR / "anchor_det_model_512px.keras"

FIXTURES_IMAGES_DIR = Path(__file__).parent.parent / "fixtures" / "images"

# Ground-truth detection counts for HADES test images.
#
# These are PIPELINE outputs, not visual ground truth. Specifically:
#   test_im3 contains 1 visible plant, but it's a POTATO root. The U-Net
#   detects it, but the downstream PlantMaster post-processing was designed
#   for Arabidopsis thaliana morphology and rejects the potato root during
#   plant assignment. Until post-processing is generalized for other species,
#   the pipeline correctly returns 0 detections on this image.
#
# See TestArabidopsisOnlyPostProcessing in test_prediction.py — that test
# pins the species limitation explicitly so a future fix surfaces here too.
EXPECTED_DETECTIONS: dict[int, int] = {
    1: 5,
    2: 4,
    3: 0,  # potato root — not detected by current implementation
    4: 0,
    5: 5,
}


def _image_path(index: int) -> Path:
    """Resolve a HADES test image path or skip the test if missing."""
    path = FIXTURES_IMAGES_DIR / f"test_im{index}.png"
    if not path.exists():
        pytest.skip(f"HADES test image not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Pipeline — loaded once per pytest session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline() -> CVPipeline:
    """Real CVPipeline with model checkpoints resolved to absolute paths.

    Session-scoped: model loading is the slowest single operation in this
    suite (~30s), so it happens exactly once. All inference fixtures
    depend on this one instance.

    Skips the entire inference suite if either model checkpoint is missing
    (the package doesn't ship the models — they live under
    src/npec_cv/artifacts/models/ in the source tree only).
    """
    if not ROOT_CHECKPOINT.exists():
        pytest.skip(f"Root model checkpoint not found: {ROOT_CHECKPOINT}")
    if not ANCHOR_CHECKPOINT.exists():
        pytest.skip(f"Anchor model checkpoint not found: {ANCHOR_CHECKPOINT}")

    # Build configs with absolute paths to override the bare-filename defaults
    root_cfg = replace(ROOT_MODEL_CONFIG, checkpoint=str(ROOT_CHECKPOINT))
    anchor_cfg = replace(ANCHOR_MODEL_CONFIG, checkpoint=str(ANCHOR_CHECKPOINT))

    return CVPipeline(inference_config=(root_cfg, anchor_cfg))


# ---------------------------------------------------------------------------
# Predictions — all 5 images inferred exactly once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def predictions(pipeline: CVPipeline) -> dict[int, ImageResult]:
    """Run inference on all 5 HADES test images, exactly once per session.

    Returns a dict keyed by image index. Tests pull whichever images they
    need from this dict — no redundant inference, no fixture proliferation.

    Total inference cost: 5 × per-image inference time, paid once at
    session start when the first dependent test runs.
    """
    return {index: pipeline.process_single_image(_image_path(index)) for index in sorted(EXPECTED_DETECTIONS.keys())}


# ---------------------------------------------------------------------------
# Parametrized fixture across all 5 images — for cross-image invariant tests
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="session",
    params=sorted(EXPECTED_DETECTIONS.keys()),
    ids=lambda i: f"test_im{i}",
)
def prediction_with_expected_count(request, predictions: dict[int, ImageResult]):
    """(ImageResult, expected_detection_count) for each HADES test image.

    Pulls from the cached `predictions` dict — no extra inference per param.
    """
    index = request.param
    return predictions[index], EXPECTED_DETECTIONS[index]
