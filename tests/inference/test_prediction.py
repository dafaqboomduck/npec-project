"""Tests for npec_cv.pipeline.CVPipeline.process_single_image.

Module under test: src/npec_cv/pipeline.py
Method:            CVPipeline.process_single_image(img_path) -> ImageResult

Real-inference integration tests: a single CVPipeline instance is loaded
once per session (the `pipeline` fixture in conftest), all 5 HADES test
images are inferred exactly once into a dict (the `predictions` fixture),
and tests pull whichever results they need. Total per-session cost:
~30s model load + 5 × ~25s inference.

Test coverage maps 1:1 to the prediction-module acceptance criteria:
  - Output shape         → TestPredictionShape
  - Confidence in [0,1]  → TestPredictionConfidence
  - Tips within bounds   → TestPredictionTipLocations
  - No-detection path    → TestEmptyImagePrediction (test_im4 is empty)
  - Cross-image hold     → TestDetectionCountAcrossImages (×5 images)
  - Species limitation   → TestArabidopsisOnlyPostProcessing (test_im3)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from npec_cv.models import ImageResult
from npec_cv.models.plant_master import PlantResult

# ---------------------------------------------------------------------------
# Helpers — image dimensions for bounds checks
# ---------------------------------------------------------------------------

_IMAGES_DIR = Path(__file__).parent.parent / "fixtures" / "images"


def _image_height_width(index: int) -> tuple[int, int]:
    """Return (height, width) of test_im{index}.png — for tip bounds checks."""
    img = cv2.imread(str(_IMAGES_DIR / f"test_im{index}.png"))
    return img.shape[:2]


# ---------------------------------------------------------------------------


class TestPredictionShape:
    """Output structure / shape contract on the canonical happy-path image."""

    def test_returns_image_result(self, predictions):
        """process_single_image returns an ImageResult instance."""
        assert isinstance(predictions[1], ImageResult)

    def test_image_name_matches_input_stem(self, predictions):
        """ImageResult.image_name equals the input file's stem."""
        assert predictions[1].image_name == "test_im1"

    def test_plant_results_is_dict_of_plant_results(self, predictions):
        """plant_results is a dict[int, PlantResult]."""
        result = predictions[1]
        assert isinstance(result.plant_results, dict)
        for plant_id, pr in result.plant_results.items():
            assert isinstance(plant_id, int)
            assert isinstance(pr, PlantResult)

    def test_total_plants_is_five(self, predictions):
        """PlantMaster is configured for 5 plant slots — total_plants reflects that."""
        assert predictions[1].total_plants == 5

    def test_all_five_plants_detected_on_happy_path(self, predictions):
        """test_im1 is the canonical happy path — all 5 plants detected."""
        assert predictions[1].num_detected == 5

    def test_detected_plants_have_required_fields_populated(self, predictions):
        """Every detected plant exposes length, end_point, threshold_used as non-None."""
        detected = [pr for pr in predictions[1].plant_results.values() if pr.detected]
        assert len(detected) > 0  # smoke check; domain knowledge says 5

        for pr in detected:
            assert pr.length is not None, f"plant_id={pr.plant_id}: length is None"
            assert pr.end_point is not None, f"plant_id={pr.plant_id}: end_point is None"
            assert pr.threshold_used is not None, f"plant_id={pr.plant_id}: threshold_used is None"


class TestPredictionConfidence:
    """Confidence scores (PlantResult.threshold_used) lie in [0, 1]."""

    def test_all_detected_confidences_in_unit_range(self, predictions):
        """Every detected plant has threshold_used in [0.0, 1.0]."""
        for pr in predictions[1].plant_results.values():
            if pr.detected:
                assert 0.0 <= pr.threshold_used <= 1.0, (
                    f"plant_id={pr.plant_id}: threshold_used={pr.threshold_used} " f"outside [0, 1]"
                )

    def test_confidence_is_float(self, predictions):
        """threshold_used is a Python float (or numpy floating type)."""
        for pr in predictions[1].plant_results.values():
            if pr.detected:
                assert isinstance(pr.threshold_used, (float, np.floating))


class TestPredictionTipLocations:
    """Detected tip locations (end_point) are within input image bounds.

    PlantResult.end_point is (row, col) per the inference code's mapping in
    src/api/services/unet_inference.py:
        tip_y = end_point[0]   # row
        tip_x = end_point[1]   # col
    """

    def test_end_point_is_two_element_tuple_of_ints(self, predictions):
        """end_point is (row, col) — both integer types."""
        for pr in predictions[1].plant_results.values():
            if pr.detected:
                assert len(pr.end_point) == 2
                assert all(
                    isinstance(v, (int, np.integer)) for v in pr.end_point
                ), f"plant_id={pr.plant_id}: end_point={pr.end_point} non-int types"

    def test_end_point_row_within_image_height(self, predictions):
        """end_point[0] (row) is in [0, image_height)."""
        h, _ = _image_height_width(1)
        for pr in predictions[1].plant_results.values():
            if pr.detected:
                row = pr.end_point[0]
                assert 0 <= row < h, f"plant_id={pr.plant_id}: end_point row={row} outside [0, {h})"

    def test_end_point_col_within_image_width(self, predictions):
        """end_point[1] (col) is in [0, image_width)."""
        _, w = _image_height_width(1)
        for pr in predictions[1].plant_results.values():
            if pr.detected:
                col = pr.end_point[1]
                assert 0 <= col < w, f"plant_id={pr.plant_id}: end_point col={col} outside [0, {w})"

    def test_start_point_within_bounds_when_present(self, predictions):
        """start_point (if populated) also lies within image bounds.

        Bonus coverage — same bounds invariant applies to the root start
        as to the tip. start_point is Optional even on detected plants,
        so we only assert when populated.
        """
        h, w = _image_height_width(1)
        for pr in predictions[1].plant_results.values():
            if pr.detected and pr.start_point is not None:
                row, col = pr.start_point
                assert 0 <= row < h, f"plant_id={pr.plant_id}: start_point row={row} outside [0, {h})"
                assert 0 <= col < w, f"plant_id={pr.plant_id}: start_point col={col} outside [0, {w})"


class TestEmptyImagePrediction:
    """test_im4 intentionally contains no plants — verify the zero-detect path."""

    def test_zero_detections(self, predictions):
        """ImageResult.num_detected is 0 for the empty image."""
        assert predictions[4].num_detected == 0

    def test_all_plants_marked_undetected(self, predictions):
        """Every PlantResult.detected is False on an empty image."""
        for pr in predictions[4].plant_results.values():
            assert pr.detected is False, f"plant_id={pr.plant_id}: marked detected on empty image"

    def test_undetected_plants_have_none_fields(self, predictions):
        """Plants with detected=False have None for length / end_point / threshold."""
        for pr in predictions[4].plant_results.values():
            assert pr.length is None
            assert pr.end_point is None
            assert pr.threshold_used is None


class TestDetectionCountAcrossImages:
    """Domain-knowledge ground truth holds across all 5 HADES test images."""

    def test_detection_count_matches_expected(self, prediction_with_expected_count):
        """[×5] num_detected matches the pinned ground truth for each image.

        Pinned counts (from EXPECTED_DETECTIONS in conftest):
          test_im1 → 5, test_im2 → 4, test_im3 → 0, test_im4 → 0, test_im5 → 5
        """
        result, expected_count = prediction_with_expected_count
        assert result.num_detected == expected_count, (
            f"Image '{result.image_name}': expected {expected_count} detections, " f"got {result.num_detected}"
        )


class TestArabidopsisOnlyPostProcessing:
    """Document the species limitation: PlantMaster post-processing is tuned
    for Arabidopsis thaliana morphology and does not assign potato roots to
    plant slots, even when the U-Net successfully segments them.

    This test pins the *current* behavior so that if/when post-processing
    is generalized to handle other species, the failure here surfaces the
    change explicitly instead of silently changing the detection count
    in the cross-image test.
    """

    def test_potato_root_is_not_assigned_to_a_plant_slot(self, predictions):
        """test_im3 contains a visible potato root that the model segments
        but PlantMaster does not promote to a detected plant.

        If this test ever STARTS detecting plants on test_im3:
          - Verify it's intentional (post-processing was generalized)
          - Update EXPECTED_DETECTIONS[3] in conftest.py
          - Remove or repurpose this test
        """
        result = predictions[3]

        assert result.num_detected == 0, (
            f"Expected 0 detections on potato-root image (Arabidopsis-only "
            f"post-processing), got {result.num_detected}. If this is "
            f"intentional, update EXPECTED_DETECTIONS[3] in conftest.py and "
            f"remove this test."
        )
