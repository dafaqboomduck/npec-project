"""End-to-end integration tests for npec_cv.pipeline.CVPipeline.

Module under test: src/npec_cv/pipeline.py
Method:            CVPipeline.process_single_image(img_path) -> ImageResult

Verifies the full pipeline (preprocess → predict → postprocess) runs
correctly on a real image with model.predict() mocked to return synthetic
probability patches. The synthetic patches deliberately don't form
detectable root patterns — this suite tests the WIRING, not the model.

What this suite uniquely tests
------------------------------
  - Pipeline runs end-to-end on a real image without raising
  - Output is a well-formed ImageResult with all 5 plant slots populated
  - filtered_mask, anchor_mask, root_probs are correctly shaped ndarrays
  - The ImageResult exposes every field the API service reads when
    converting to PredictionResponse (the contract this suite primarily
    guards — see TestApiContractAdapter)

What this suite does NOT test
-----------------------------
  - Whether the model finds plants (covered by tests/inference, slow)
  - Whether confidence values are in [0,1] for detected plants (same)
  - Whether tip locations land in image bounds (same)
"""

from __future__ import annotations

import numpy as np

from src.npec_cv.models import ImageResult
from src.npec_cv.models.plant_master import PlantResult

# ---------------------------------------------------------------------------
# API contract adapter — mimics the mapping in src/api/services/unet_inference.py
# ---------------------------------------------------------------------------
#
# The npec_cv package produces ImageResult / PlantResult; the API service
# converts those into PredictionResponse / Plant objects. The mapping logic
# below mirrors the relevant slice of that conversion. If npec_cv ever
# changes the field names or types this adapter reads, this test fails —
# which is the signal that the API service will break.
#
# Reference: src/api/services/unet_inference.py — the PlantResult → Plant
# mapping inside UNetInferenceService.run.

SCALE_FACTOR_MM_PER_PX = 150 / 2775  # ~0.054 mm/px, per docs/IO specifications.md


def _adapt_to_api_plant_dict(plant_result: PlantResult) -> dict:
    """Convert one PlantResult to the dict shape the API's Plant schema expects.

    Mirrors the mapping in UNetInferenceService.run. Skips undetected plants
    the same way the API service does — caller checks `detected` first.
    """
    confidence = float(plant_result.threshold_used) if plant_result.threshold_used is not None else 0.0
    tip_x = int(plant_result.end_point[1]) if plant_result.end_point is not None else 0
    tip_y = int(plant_result.end_point[0]) if plant_result.end_point is not None else 0
    length_px = float(plant_result.length) if plant_result.length is not None else 0.0

    return {
        "plant_id": int(plant_result.plant_id),
        "tip_x": tip_x,
        "tip_y": tip_y,
        "confidence": round(confidence, 4),
        "root_length_px": round(length_px, 2),
        "root_length_mm": round(length_px * SCALE_FACTOR_MM_PER_PX, 2),
    }


# ---------------------------------------------------------------------------


class TestPipelineRunsEndToEnd:
    """The pipeline accepts a real image and produces a well-formed result."""

    def test_returns_image_result(self, image_result):
        """process_single_image returns an ImageResult instance."""
        result = image_result

        assert isinstance(result, ImageResult)

    def test_image_name_matches_input_stem(self, image_result, real_test_image_path):
        """ImageResult.image_name equals the input file's stem."""
        result = image_result

        assert result.image_name == real_test_image_path.stem

    def test_all_five_plant_slots_populated(self, image_result):
        """plant_results has exactly 5 entries (one per configured plant slot)."""
        result = image_result

        assert result.total_plants == 5
        assert len(result.plant_results) == 5

    def test_plant_results_keyed_by_int_with_plant_result_values(self, image_result):
        """plant_results is dict[int, PlantResult]."""
        result = image_result

        for plant_id, pr in result.plant_results.items():
            assert isinstance(plant_id, int)
            assert isinstance(pr, PlantResult)
            assert pr.plant_id == plant_id


class TestImageResultMaskFields:
    """The mask-related fields the API reads are populated and correctly shaped."""

    def test_filtered_mask_is_ndarray(self, image_result):
        """filtered_mask (the overall segmentation mask) is an ndarray."""
        result = image_result

        assert result.filtered_mask is not None
        assert isinstance(result.filtered_mask, np.ndarray)

    def test_filtered_mask_matches_input_image_shape(self, image_result, real_test_image_path):
        """filtered_mask has the same H, W as the input image."""
        import cv2

        img = cv2.imread(str(real_test_image_path))
        result = image_result

        assert result.filtered_mask.shape[:2] == img.shape[:2]

    def test_filtered_mask_is_2d(self, image_result):
        """filtered_mask is a 2D array (no channel dim)."""
        result = image_result

        assert result.filtered_mask.ndim == 2

    def test_anchor_mask_populated(self, image_result):
        """anchor_mask is populated as an ndarray for visualization."""
        result = image_result

        assert result.anchor_mask is not None
        assert isinstance(result.anchor_mask, np.ndarray)


class TestPlantResultUndetectedContract:
    """With synthetic predictions producing 0 detections, all PlantResults
    are in the "undetected" state. Verify that state populates None correctly
    on every field the API service reads."""

    def test_all_plants_marked_undetected(self, image_result):
        """Every PlantResult.detected is False (synthetic predictions don't
        form valid root patterns)."""
        result = image_result

        for pr in result.plant_results.values():
            assert pr.detected is False

    def test_undetected_plant_fields_are_none(self, image_result):
        """Per-plant fields the API reads (length, end_point, threshold_used)
        are None for undetected plants. This is the contract the API's
        UNetInferenceService.run depends on when it skips undetected plants."""
        result = image_result

        for pr in result.plant_results.values():
            assert pr.length is None, f"plant_id={pr.plant_id}: length not None"
            assert pr.end_point is None, f"plant_id={pr.plant_id}: end_point not None"
            assert pr.threshold_used is None, f"plant_id={pr.plant_id}: threshold_used not None"


class TestApiContractAdapter:
    """The PlantResult → API-Plant-dict mapping produces correctly-typed fields.

    This is the primary purpose of the integration suite: prove that the
    field names and types in ImageResult / PlantResult are exactly what
    the API service expects to read when converting to PredictionResponse.
    """

    def test_adapter_produces_required_keys(self, image_result):
        """The adapter dict has every key the API's Plant schema requires."""
        result = image_result
        required_keys = {
            "plant_id",
            "tip_x",
            "tip_y",
            "confidence",
            "root_length_px",
            "root_length_mm",
        }

        for pr in result.plant_results.values():
            adapted = _adapt_to_api_plant_dict(pr)
            assert set(adapted.keys()) == required_keys, (
                f"plant_id={pr.plant_id}: adapter keys mismatch. "
                f"Got {set(adapted.keys())}, expected {required_keys}"
            )

    def test_adapter_field_types_match_api_schema(self, image_result):
        """Each adapted field has the type the API's Plant schema declares."""
        result = image_result

        for pr in result.plant_results.values():
            adapted = _adapt_to_api_plant_dict(pr)

            assert isinstance(adapted["plant_id"], int)
            assert isinstance(adapted["tip_x"], int)
            assert isinstance(adapted["tip_y"], int)
            assert isinstance(adapted["confidence"], float)
            assert isinstance(adapted["root_length_px"], float)
            assert isinstance(adapted["root_length_mm"], float)

    def test_adapter_undetected_plant_yields_zero_confidence(self, image_result):
        """An undetected plant adapts to confidence=0.0 (per API service mapping)."""
        result = image_result

        for pr in result.plant_results.values():
            if not pr.detected:
                adapted = _adapt_to_api_plant_dict(pr)
                assert adapted["confidence"] == 0.0
                assert adapted["root_length_px"] == 0.0
                assert adapted["root_length_mm"] == 0.0

    def test_adapter_confidence_is_in_unit_range(self, image_result):
        """Adapted confidence value is always in [0.0, 1.0] regardless of detection state.

        Required by the API's Plant schema (Field(..., ge=0.0, le=1.0)).
        """
        result = image_result

        for pr in result.plant_results.values():
            adapted = _adapt_to_api_plant_dict(pr)
            assert 0.0 <= adapted["confidence"] <= 1.0, (
                f"plant_id={pr.plant_id}: confidence {adapted['confidence']} " f"outside [0, 1]"
            )
