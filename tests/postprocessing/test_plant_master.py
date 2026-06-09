"""Unit tests for npec_cv.core.plant_master.PlantMaster.

All tests use synthetic numpy data — no real model checkpoints required.
InferenceManager, PlantSegmentation, and LengthCalculator are replaced by
MagicMocks at construction time so the suite is fast and free of I/O.

Coverage targets (lines uncovered in latest report):
    _init_components failures   : 85-87, 92-94, 99-101, 106-108, 111-116
    _preprocess_image crop mode : 126-131
    _run_inference crop coords  : 156-161
    _find_endpoints_at_threshold: 189, 216-222, 231-241
    detect_endpoints_adaptive   : 270
    _segment_roots              : 298-348
    process_image plant results : 430->439, 436-437, 440-444
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from npec_cv.core.plant_master import PlantMaster
from npec_cv.exceptions import ConfigurationError
from npec_cv.models import (
    AdaptiveThresholdConfig,
    Anchor,
    ModelConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_config():
    return ModelConfig(checkpoint="stub.keras", threshold=0.5, patch_size=512, batch_size=4, custom_objects={})


@pytest.fixture
def threshold_config():
    return AdaptiveThresholdConfig(
        adaptive_threshold_start=0.9,
        adaptive_threshold_end=0.3,
        adaptive_threshold_step=0.1,
        shoot_overlap_penalty=0.3,
        shoot_overlap_min_conf=0.5,
    )


def _build(seg_config, model_config, length_config, threshold_config, crop_to_dish=False):
    """Construct a PlantMaster with all heavy components replaced by MagicMocks."""
    with (
        patch("npec_cv.core.plant_master.InferenceManager"),
        patch("npec_cv.core.plant_master.PlantSegmentation"),
        patch("npec_cv.core.plant_master.LengthCalculator"),
    ):
        pm = PlantMaster(
            inference_config=(model_config, model_config),
            segmentation_config=seg_config,
            calculator_config=length_config,
            thresholds_config=threshold_config,
            crop_to_dish=crop_to_dish,
        )
    pm.root_manager.crop_coords = None
    pm.anchor_manager.crop_coords = None
    return pm


@pytest.fixture
def plant_master(seg_config, model_config, length_config, threshold_config):
    return _build(seg_config, model_config, length_config, threshold_config)


@pytest.fixture
def plant_master_crop(seg_config, model_config, length_config, threshold_config):
    return _build(seg_config, model_config, length_config, threshold_config, crop_to_dish=True)


# ---------------------------------------------------------------------------
# _init_components — failure paths (lines 85-87, 92-94, 99-101, 106-108, 111-116)
# ---------------------------------------------------------------------------


class TestInitComponentsFailures:
    def _kwargs(self, seg_config, model_config, length_config, threshold_config):
        return dict(
            inference_config=(model_config, model_config),
            segmentation_config=seg_config,
            calculator_config=length_config,
            thresholds_config=threshold_config,
        )

    def test_root_model_failure_raises(self, seg_config, model_config, length_config, threshold_config):
        with (
            patch("npec_cv.core.plant_master.InferenceManager", side_effect=RuntimeError("bad root")),
            patch("npec_cv.core.plant_master.PlantSegmentation"),
            patch("npec_cv.core.plant_master.LengthCalculator"),
            pytest.raises(ConfigurationError),
        ):
            PlantMaster(**self._kwargs(seg_config, model_config, length_config, threshold_config))

    def test_anchor_model_failure_raises(self, seg_config, model_config, length_config, threshold_config):
        with (
            patch(
                "npec_cv.core.plant_master.InferenceManager",
                side_effect=[MagicMock(), RuntimeError("bad anchor")],
            ),
            patch("npec_cv.core.plant_master.PlantSegmentation"),
            patch("npec_cv.core.plant_master.LengthCalculator"),
            pytest.raises(ConfigurationError),
        ):
            PlantMaster(**self._kwargs(seg_config, model_config, length_config, threshold_config))

    def test_segmentor_failure_raises(self, seg_config, model_config, length_config, threshold_config):
        with (
            patch("npec_cv.core.plant_master.InferenceManager"),
            patch("npec_cv.core.plant_master.PlantSegmentation", side_effect=RuntimeError("bad segmentor")),
            patch("npec_cv.core.plant_master.LengthCalculator"),
            pytest.raises(ConfigurationError),
        ):
            PlantMaster(**self._kwargs(seg_config, model_config, length_config, threshold_config))

    def test_calculator_failure_raises(self, seg_config, model_config, length_config, threshold_config):
        with (
            patch("npec_cv.core.plant_master.InferenceManager"),
            patch("npec_cv.core.plant_master.PlantSegmentation"),
            patch("npec_cv.core.plant_master.LengthCalculator", side_effect=RuntimeError("bad calculator")),
            pytest.raises(ConfigurationError),
        ):
            PlantMaster(**self._kwargs(seg_config, model_config, length_config, threshold_config))


# ---------------------------------------------------------------------------
# _preprocess_image — crop_to_dish branch (lines 126-131)
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def test_crop_to_dish_calls_extract_petri_dish(self, plant_master_crop):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cropped = np.zeros((200, 200, 3), dtype=np.uint8)

        with patch("npec_cv.core.plant_master.extract_petri_dish", return_value=cropped) as mock_extract:
            result, offset = plant_master_crop._preprocess_image(img)

        mock_extract.assert_called_once_with(img)
        assert result is cropped
        assert offset is None

    def test_no_crop_returns_original_image(self, plant_master):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        result, offset = plant_master._preprocess_image(img)
        assert result is img
        assert offset is None


# ---------------------------------------------------------------------------
# _run_inference — crop_coords path (lines 156-161)
# ---------------------------------------------------------------------------


class TestRunInference:
    def test_crop_coords_slice_restored_masks(self, plant_master_crop):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        full_probs = np.ones((300, 400), dtype=np.float32) * 0.6

        plant_master_crop.root_manager.restore_mask.return_value = full_probs.copy()
        plant_master_crop.anchor_manager.restore_mask.return_value = full_probs.copy()

        # crop_coords = (x1, y1, x2, y2) → slices [y1:y2, x1:x2]
        plant_master_crop.root_manager.crop_coords = (50, 50, 350, 250)
        plant_master_crop.anchor_manager.crop_coords = (50, 50, 350, 250)

        root_probs, anchor_probs = plant_master_crop._run_inference(img)

        assert root_probs.shape == (200, 300)  # y2-y1=200, x2-x1=300
        assert anchor_probs.shape == (200, 300)

    def test_none_crop_coords_returns_full_probs(self, plant_master_crop):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        full_probs = np.zeros((300, 400), dtype=np.float32)

        plant_master_crop.root_manager.restore_mask.return_value = full_probs.copy()
        plant_master_crop.anchor_manager.restore_mask.return_value = full_probs.copy()
        # crop_coords is already None from the fixture

        root_probs, anchor_probs = plant_master_crop._run_inference(img)

        assert root_probs.shape == (300, 400)
        assert anchor_probs.shape == (300, 400)


# ---------------------------------------------------------------------------
# _find_endpoints_at_threshold (lines 189, 216-222, 231-241)
# ---------------------------------------------------------------------------


class TestFindEndpointsAtThreshold:
    # seg_config zones: {0: (50,150), 1: (150,250), 2: (250,350)}, bbox_tolerance=10

    def _anchor(self):
        """Anchor in zone 0 with centroid (100,50) and bbox (80,30,120,70)."""
        return Anchor(
            label_id=1,
            centroid=(100, 50),
            area=400,
            mask=np.zeros((300, 400), dtype=np.uint8),
            zone_id=0,
            bbox=(80, 30, 120, 70),
        )

    def test_anchor_without_zone_id_is_skipped(self, plant_master):
        """Anchor with zone_id=None hits the early continue (line 189)."""
        anchor = Anchor(label_id=1, centroid=(100, 50), area=400, mask=np.zeros((300, 400), dtype=np.uint8))
        probs = np.zeros((300, 400), dtype=np.float32)

        results = plant_master._find_endpoints_at_threshold(probs, [anchor], threshold=0.5)

        assert all(not r["detected"] for r in results.values())

    def test_root_stripe_within_bbox_is_detected(self, plant_master):
        """Component overlapping the anchor bbox records start + end points (lines 216-222, 231-241)."""
        probs = np.zeros((300, 400), dtype=np.float32)
        # Vertical stripe from y=50 to y=199 centered in zone 0
        probs[50:200, 90:110] = 0.8

        results = plant_master._find_endpoints_at_threshold(probs, [self._anchor()], threshold=0.5)

        assert results[0]["detected"] is True
        start_y, start_x = results[0]["start"]
        end_y, _ = results[0]["end"]
        assert start_y == 50
        assert 90 <= start_x <= 110
        assert end_y > 100  # bottom of stripe

    def test_component_outside_bbox_not_detected(self, plant_master):
        """Component whose top row misses the anchor bbox leaves zone undetected."""
        probs = np.zeros((300, 400), dtype=np.float32)
        # Stripe well below the anchor bbox (anchor y_max+tol = 90)
        probs[150:250, 90:110] = 0.8

        results = plant_master._find_endpoints_at_threshold(probs, [self._anchor()], threshold=0.5)

        assert results[0]["detected"] is False


# ---------------------------------------------------------------------------
# detect_endpoints_adaptive — early return and fallback (line 270)
# ---------------------------------------------------------------------------


class TestDetectEndpointsAdaptive:
    def test_early_return_when_all_plants_found(self, plant_master):
        """Returns at the first threshold where all zones are detected (line 270)."""
        all_detected = {zid: {"detected": True, "start": (10, 100), "end": (200, 100)} for zid in (0, 1, 2)}
        plant_master._find_endpoints_at_threshold = MagicMock(return_value=all_detected)

        thresh, results, all_found = plant_master.detect_endpoints_adaptive(
            np.zeros((300, 400), dtype=np.float32), anchors=[]
        )

        assert all_found is True
        assert thresh == pytest.approx(plant_master.thresholds_config.adaptive_threshold_start)

    def test_fallback_when_not_all_found(self, plant_master):
        """Returns adaptive_threshold_end with all_found=False when detection is partial."""
        partial = {0: {"detected": True}, 1: {"detected": False}, 2: {"detected": False}}
        plant_master._find_endpoints_at_threshold = MagicMock(return_value=partial)

        thresh, results, all_found = plant_master.detect_endpoints_adaptive(
            np.zeros((300, 400), dtype=np.float32), anchors=[]
        )

        assert all_found is False
        assert thresh == pytest.approx(plant_master.thresholds_config.adaptive_threshold_end)


# ---------------------------------------------------------------------------
# _segment_roots (lines 298-348)
# ---------------------------------------------------------------------------


class TestSegmentRoots:
    def test_detected_zone_produces_labelled_mask(self, plant_master):
        """Root stripe in zone 0 with valid start region → filled filtered_mask (lines 298-349)."""
        root_probs = np.zeros((300, 400), dtype=np.float32)
        root_probs[50:200, 90:110] = 0.8  # stripe well within zone 0 (x=[50,150))

        endpoint_info = {
            0: {"detected": True, "start": (50, 100), "end": (199, 100)},
            1: {"detected": False},
            2: {"detected": False},
        }

        filtered_mask, final_labels = plant_master._segment_roots(root_probs, endpoint_info, optimal_threshold=0.5)

        assert np.any(filtered_mask > 0)
        assert np.any(final_labels == 1)  # zone 0 → label = zid+1 = 1
        assert endpoint_info[0].get("threshold_used") == 0.5

    def test_fallback_to_component_mask_when_start_region_empty(self, plant_master):
        """Stripe below the start point → no zone_components → uses component_mask fallback."""
        root_probs = np.zeros((300, 400), dtype=np.float32)
        # Stripe starts at y=110; start_y=20 so start_region covers y=[0:70] — no overlap
        root_probs[110:200, 90:110] = 0.8

        component_mask = root_probs > 0.5
        endpoint_info = {
            0: {
                "detected": True,
                "start": (20, 100),
                "end": (199, 100),
                "component_mask": component_mask,
            },
            1: {"detected": False},
            2: {"detected": False},
        }

        filtered_mask, _ = plant_master._segment_roots(root_probs, endpoint_info, optimal_threshold=0.5)

        assert np.any(filtered_mask > 0)

    def test_zone_skipped_when_no_components_and_no_fallback_mask(self, plant_master):
        """No zone_components and no component_mask key → zone silently skipped (line 324 continue)."""
        root_probs = np.zeros((300, 400), dtype=np.float32)
        root_probs[110:200, 90:110] = 0.8  # stripe below start, no component_mask provided

        endpoint_info = {
            0: {"detected": True, "start": (20, 100), "end": (199, 100)},  # no component_mask key
            1: {"detected": False},
            2: {"detected": False},
        }

        filtered_mask, final_labels = plant_master._segment_roots(root_probs, endpoint_info, optimal_threshold=0.5)

        assert not np.any(filtered_mask > 0)
        assert not np.any(final_labels > 0)


# ---------------------------------------------------------------------------
# process_image — plant result population (lines 430->439, 436-437, 440-444)
# ---------------------------------------------------------------------------


class TestProcessImagePlantResults:
    def _setup_mocks(self, plant_master, endpoint_info, labels, length_results):
        """Wire up all sub-method mocks so process_image reaches the plant-result loop."""
        root_probs = np.zeros((300, 400), dtype=np.float32)
        anchor_probs = np.zeros((300, 400), dtype=np.float32)

        plant_master._run_inference = MagicMock(return_value=(root_probs, anchor_probs))
        plant_master.segmentor.extract_anchors.return_value = []
        plant_master.segmentor.create_anchor_bboxes.return_value = []
        plant_master.detect_endpoints_adaptive = MagicMock(return_value=(0.5, endpoint_info, False))
        plant_master._segment_roots = MagicMock(return_value=(np.zeros((300, 400), dtype=np.uint8), labels))
        plant_master.calculator.analyze_plant_lengths.return_value = length_results

    def test_detected_plant_populates_length_path_and_endpoint(self, plant_master):
        """Plant 0 detected: path_result branch (436-437) and endpoint block (440-444) both run.
        Plants 1 and 2 absent from length_results, exercising the 430->439 branch skip.
        """
        labels = np.zeros((300, 400), dtype=np.int32)
        labels[100:200, 90:110] = 1  # plant 0 (label = pid+1 = 1)

        endpoint_info = {
            0: {"detected": True, "start": (50, 100), "end": (199, 100), "threshold_used": 0.5},
            1: {"detected": False},
            2: {"detected": False},
        }

        lr = MagicMock()
        lr.detected = True
        lr.length = 150.0
        lr.pixel_count = 150
        lr.path_result = MagicMock()
        lr.path_result.skeleton = np.zeros((300, 400), dtype=bool)
        lr.path_result.path = [(50, 100), (199, 100)]

        self._setup_mocks(plant_master, endpoint_info, labels, length_results={0: lr})

        result = plant_master.process_image(np.zeros((300, 400, 3), dtype=np.uint8))

        pr0 = result.plant_results[0]
        assert pr0.detected is True
        assert pr0.length == 150.0
        assert pr0.skeleton is not None  # from path_result (line 436)
        assert pr0.path is not None  # from path_result (line 437)
        assert pr0.start_point == (50, 100)  # from endpoint_info (line 441)
        assert pr0.end_point == (199, 100)  # from endpoint_info (line 442)
        assert pr0.threshold_used == 0.5  # from endpoint_info (line 443)
        assert pr0.mask is not None  # from endpoint_info (line 444)

        # Plants 1 and 2 skipped the length block (430->439 branch)
        for pid in (1, 2):
            assert not result.plant_results[pid].detected

    def test_no_path_result_skips_skeleton_and_path(self, plant_master):
        """lr.path_result=None skips skeleton/path assignment (435->439 branch)."""
        labels = np.zeros((300, 400), dtype=np.int32)
        endpoint_info = {
            0: {"detected": True, "start": (50, 100), "end": (199, 100), "threshold_used": 0.5},
            1: {"detected": False},
            2: {"detected": False},
        }

        lr = MagicMock()
        lr.detected = True
        lr.length = 100.0
        lr.pixel_count = 100
        lr.path_result = None  # falsy → skeleton/path lines are skipped

        self._setup_mocks(plant_master, endpoint_info, labels, length_results={0: lr})

        result = plant_master.process_image(np.zeros((300, 400, 3), dtype=np.uint8))

        pr0 = result.plant_results[0]
        assert pr0.detected is True
        assert pr0.length == 100.0
        assert pr0.skeleton is None
        assert pr0.path is None
