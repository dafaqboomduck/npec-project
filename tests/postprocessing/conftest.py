"""Shared fixtures for npec_cv post-processing tests.

Covers tests for:
  - npec_cv.core.plant_segmentation.PlantSegmentation
  - npec_cv.core.root_lengths.LengthCalculator

Both modules consume binary masks (anchor / root predictions, labeled
component arrays) and produce structured detection results. Synthetic
masks give us full control over the geometry, which is exactly what
unit-level tests want — no model, no images, no I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from npec_cv.models import Anchor, LengthCalculatorConfig, SegmentationConfig

# ---------------------------------------------------------------------------
# Plant segmentation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seg_config() -> SegmentationConfig:
    """Minimal-but-realistic segmentation config with three plant zones.

    Zones span x in [50, 150) / [150, 250) / [250, 350]. Tests build masks
    inside an image that's at least 400 wide and 300 tall.
    """
    return SegmentationConfig(
        plant_zones={
            0: (50, 150),
            1: (150, 250),
            2: (250, 350),
        },
        min_anchor_area=20,
        min_root_area=10,
        y_top_ratio=0.4,
        y_bot_ratio=0.5,
        anchor_bbox_size=60,
        bbox_tolerance=10,
        vertical_offset=5,
        root_merge_distance=15,
    )


@pytest.fixture
def empty_anchor_mask() -> np.ndarray:
    """All-zero 300x400 anchor mask — no components."""
    return np.zeros((300, 400), dtype=np.uint8)


@pytest.fixture
def three_zone_anchor_mask() -> np.ndarray:
    """Anchor mask with one anchor in each of the three zones, near top.

    Anchors are 20x20 squares centered at (cx=100, cy=50), (200, 50),
    (300, 50). All comfortably within y_top_ratio=0.4 (max_y=120) and
    above min_anchor_area=20.
    """
    mask = np.zeros((300, 400), dtype=np.uint8)
    for cx in (100, 200, 300):
        mask[40:60, cx - 10 : cx + 10] = 255
    return mask


@pytest.fixture
def too_low_anchor_mask() -> np.ndarray:
    """Anchor mask where the only anchor sits below the y_top cutoff.

    With y_top_ratio=0.4 on a 300-row image, max_y = 120. Component is
    centered around y=200, so it must be filtered out.
    """
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[190:210, 90:110] = 255
    return mask


@pytest.fixture
def too_small_anchor_mask() -> np.ndarray:
    """Anchor mask with a component below min_anchor_area=20."""
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[40:43, 90:93] = 255  # 9 px, well under 20
    return mask


@pytest.fixture
def root_mask_one_zone() -> np.ndarray:
    """Vertical root component starting under zone 0's anchor."""
    mask = np.zeros((300, 400), dtype=np.uint8)
    # vertical 5x150 stripe from y=70 down to y=220 at x≈100
    mask[70:220, 98:103] = 255
    return mask


@pytest.fixture
def anchors_at_zone0(seg_config) -> list[Anchor]:
    """A single anchor object placed in zone 0 with bbox already populated."""
    return [
        Anchor(
            label_id=1,
            centroid=(100, 50),
            area=400,
            mask=np.zeros((300, 400), dtype=np.uint8),
            zone_id=0,
            bbox=(70, 25, 130, 85),  # tight box centered at (100, 50)
        )
    ]


# ---------------------------------------------------------------------------
# Length calculator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def length_config() -> LengthCalculatorConfig:
    """Standard length calculator config used by the suite."""
    return LengthCalculatorConfig(
        num_plants=3,
        max_gap=5,
        max_path=10,
        use_farthest_endpoint=True,
    )


@pytest.fixture
def empty_mask() -> np.ndarray:
    """All-zero binary mask."""
    return np.zeros((50, 50), dtype=bool)


@pytest.fixture
def vertical_line_mask() -> np.ndarray:
    """Single vertical 1-pixel-wide line of length 30."""
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 25] = True
    return mask


@pytest.fixture
def diagonal_line_mask() -> np.ndarray:
    """Diagonal 1-pixel-wide line — checks Euclidean weighting (sqrt 2)."""
    mask = np.zeros((50, 50), dtype=bool)
    for i in range(20):
        mask[10 + i, 10 + i] = True
    return mask


@pytest.fixture
def two_plant_labels() -> np.ndarray:
    """Labeled array with two vertical roots: label 1 and label 2.

    Label 1 → plant_id 0, label 2 → plant_id 1. Both span y in [10, 40).
    """
    labels = np.zeros((50, 60), dtype=np.int32)
    labels[10:40, 15] = 1
    labels[10:40, 35] = 2
    return labels
