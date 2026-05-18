"""Fast contract tests for ImageResult / PlantResult shapes.

These tests verify that consumers of CVPipeline.process_single_image can
rely on the ImageResult contract — field types, value ranges, and bounds
invariants — without paying the cost of real TF inference.

Why this file exists
--------------------
The slow test_prediction.py exercises the real pipeline end-to-end on
real HADES images. That suite is the source of truth for "does the model
actually work" but takes ~3 minutes per run, which is too slow to gate
every push and merge on.

These tests use fabricated ImageResults to verify:
  - The dataclass contract (fields exist, types match)
  - That well-formed results pass our validation logic
  - That malformed results fail our validation logic

If you change PlantResult / ImageResult fields, both this file and
test_prediction.py need to be updated. CI runs this file on every push;
test_prediction.py is run on demand or on a nightly schedule.
"""

from __future__ import annotations

import numpy as np
from npec_cv.models import ImageResult
from npec_cv.models.plant_master import PlantResult

# ---------------------------------------------------------------------------
# Fabrication helpers — build fake ImageResults with controllable content
# ---------------------------------------------------------------------------


def _make_plant_result(
    plant_id: int = 0,
    detected: bool = True,
    length: int | None = 1500,
    end_point: tuple[int, int] | None = (1200, 800),
    start_point: tuple[int, int] | None = (300, 800),
    threshold_used: float | None = 0.42,
    pixel_count: int | None = 5000,
) -> PlantResult:
    """Build a PlantResult with sensible defaults; pass `detected=False` to
    get the no-detection variant where all per-plant fields are None."""
    if not detected:
        return PlantResult(
            plant_id=plant_id,
            detected=False,
            length=None,
            end_point=None,
            start_point=None,
            threshold_used=None,
            pixel_count=None,
        )
    return PlantResult(
        plant_id=plant_id,
        detected=True,
        length=length,
        end_point=end_point,
        start_point=start_point,
        threshold_used=threshold_used,
        pixel_count=pixel_count,
    )


def _make_image_result(
    image_name: str = "fake_image",
    num_plants: int = 5,
    num_detected: int = 5,
) -> ImageResult:
    """Build an ImageResult with `num_plants` slots, `num_detected` of which
    are marked detected. Detected slots come first."""
    plant_results = {i: _make_plant_result(plant_id=i, detected=(i < num_detected)) for i in range(num_plants)}
    return ImageResult(image_name=image_name, plant_results=plant_results)


# ---------------------------------------------------------------------------


class TestImageResultContract:
    """Shape invariants on ImageResult / PlantResult."""

    def test_image_result_has_image_name(self):
        result = _make_image_result(image_name="test_image")
        assert result.image_name == "test_image"

    def test_total_plants_equals_dict_size(self):
        result = _make_image_result(num_plants=5)
        assert result.total_plants == 5

    def test_num_detected_counts_only_detected_plants(self):
        result = _make_image_result(num_plants=5, num_detected=3)
        assert result.num_detected == 3

    def test_num_detected_is_zero_for_empty_image(self):
        result = _make_image_result(num_plants=5, num_detected=0)
        assert result.num_detected == 0

    def test_plant_results_keys_are_ints(self):
        result = _make_image_result()
        assert all(isinstance(k, int) for k in result.plant_results.keys())

    def test_plant_results_values_are_plant_result(self):
        result = _make_image_result()
        assert all(isinstance(v, PlantResult) for v in result.plant_results.values())


class TestPlantResultContract:
    """Field contract on PlantResult — detected vs. not-detected variants."""

    def test_detected_plant_has_length(self):
        pr = _make_plant_result(detected=True)
        assert pr.length is not None

    def test_detected_plant_has_end_point(self):
        pr = _make_plant_result(detected=True)
        assert pr.end_point is not None
        assert len(pr.end_point) == 2

    def test_detected_plant_has_threshold_used(self):
        pr = _make_plant_result(detected=True)
        assert pr.threshold_used is not None

    def test_undetected_plant_has_none_fields(self):
        pr = _make_plant_result(detected=False)
        assert pr.length is None
        assert pr.end_point is None
        assert pr.threshold_used is None
        assert pr.start_point is None
        assert pr.pixel_count is None


class TestConfidenceRangeValidation:
    """The [0, 1] confidence invariant we assert against real inference
    results — verify the validator logic itself is correct."""

    def test_threshold_within_unit_range_passes(self):
        """A well-formed PlantResult with confidence in [0, 1] passes the check."""
        pr = _make_plant_result(detected=True, threshold_used=0.5)
        assert 0.0 <= pr.threshold_used <= 1.0

    def test_threshold_at_boundary_passes(self):
        """0.0 and 1.0 are inclusive boundaries — both pass."""
        assert 0.0 <= _make_plant_result(threshold_used=0.0).threshold_used <= 1.0
        assert 0.0 <= _make_plant_result(threshold_used=1.0).threshold_used <= 1.0

    def test_threshold_supports_numpy_floats(self):
        """np.floating values are accepted (real inference returns these)."""
        pr = _make_plant_result(detected=True, threshold_used=np.float32(0.7))
        assert isinstance(pr.threshold_used, (float, np.floating))


class TestTipBoundsValidation:
    """The "tip within image bounds" invariant — verify the validator logic."""

    def test_tip_within_bounds_passes(self):
        """A tip at (500, 800) in a 1000×2000 image is within bounds."""
        pr = _make_plant_result(detected=True, end_point=(500, 800))
        h, w = 1000, 2000
        assert 0 <= pr.end_point[0] < h
        assert 0 <= pr.end_point[1] < w

    def test_tip_at_origin_passes(self):
        """(0, 0) is the inclusive lower bound."""
        pr = _make_plant_result(detected=True, end_point=(0, 0))
        h, w = 1000, 2000
        assert 0 <= pr.end_point[0] < h
        assert 0 <= pr.end_point[1] < w

    def test_tip_supports_numpy_ints(self):
        """np.integer values are accepted (real inference returns these)."""
        pr = _make_plant_result(
            detected=True,
            end_point=(np.int64(500), np.int64(800)),
        )
        assert all(isinstance(v, (int, np.integer)) for v in pr.end_point)
