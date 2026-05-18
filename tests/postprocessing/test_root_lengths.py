"""Tests for npec_cv.core.root_lengths.LengthCalculator.

Strategy: build small synthetic skeletons / labeled arrays whose geometry
is known by construction, then exercise the public API plus the helpers
that account for the bulk of this file's red lines.

The four return codes from `calculate_geodesic_length` each get their own
test, and the public `analyze_plant_lengths` is exercised end-to-end on a
two-plant labeled array to cover `_process_single_plant`,
`_get_crop_endpoints`, `_expand_skeleton`, and `_log_result` together.
"""

from __future__ import annotations

import numpy as np
import pytest
from npec_cv.configs.lengths import (
    RETURN_EMPTY_MASK,
    RETURN_INSUFFICIENT_PIXELS,
    RETURN_SUCCESS,
)
from npec_cv.core.root_lengths import LengthCalculator
from npec_cv.exceptions import ConfigurationError


class TestLengthCalculatorInit:
    """Construction-time validation."""

    def test_none_config_raises(self):
        """A None config is rejected with ConfigurationError."""
        with pytest.raises(ConfigurationError):
            LengthCalculator(None)

    def test_valid_config_initializes(self, length_config):
        """A valid config produces a usable instance."""
        calc = LengthCalculator(length_config)
        assert calc.config is length_config


class TestNeighborHelpers:
    """`_get_neighbors`, `_find_endpoints`, `_are_connected`."""

    def test_neighbors_returns_only_pixels_in_set(self, length_config):
        """Neighbors are filtered to those in the provided pixel set."""
        calc = LengthCalculator(length_config)
        pixel_set = {(5, 5), (5, 6), (6, 5)}
        neighbors = calc._get_neighbors((5, 5), pixel_set)
        assert set(neighbors) == {(5, 6), (6, 5)}

    def test_endpoints_of_simple_line(self, length_config, vertical_line_mask):
        """A 1-pixel-wide line of length 30 has exactly two endpoints."""
        calc = LengthCalculator(length_config)
        endpoints = calc._find_endpoints(vertical_line_mask)
        assert len(endpoints) == 2
        # One endpoint is the topmost pixel, one is the bottommost
        ys = sorted(p[0] for p in endpoints)
        assert ys[0] == 10 and ys[1] == 39

    def test_are_connected_same_point_is_true(self, length_config):
        """A point is trivially connected to itself."""
        calc = LengthCalculator(length_config)
        assert calc._are_connected({(0, 0)}, (0, 0), (0, 0), max_depth=5) is True

    def test_are_connected_disjoint_returns_false(self, length_config):
        """Two points in disjoint components are not connected."""
        calc = LengthCalculator(length_config)
        pixel_set = {(0, 0), (10, 10)}
        assert calc._are_connected(pixel_set, (0, 0), (10, 10), max_depth=5) is False

    def test_are_connected_finds_path(self, length_config):
        """Connected pixels return True within max_depth."""
        calc = LengthCalculator(length_config)
        pixel_set = {(0, 0), (1, 0), (2, 0)}
        assert calc._are_connected(pixel_set, (0, 0), (2, 0), max_depth=5) is True


class TestDrawLine:
    """Bresenham line drawing — used by gap bridging."""

    def test_draws_horizontal_line(self, length_config):
        """A horizontal line lights up exactly the cells between two points."""
        calc = LengthCalculator(length_config)
        img = np.zeros((10, 10), dtype=bool)
        result = calc._draw_line(img, (5, 2), (5, 7))
        assert result[5, 2:8].all()

    def test_draws_diagonal_line(self, length_config):
        """A 45-degree diagonal lights up the diagonal cells."""
        calc = LengthCalculator(length_config)
        img = np.zeros((10, 10), dtype=bool)
        result = calc._draw_line(img, (2, 2), (7, 7))
        for i in range(6):
            assert result[2 + i, 2 + i]


class TestFindNearestSkeletonPoint:
    """Nearest-pixel search with radius cap."""

    def test_finds_within_radius(self, length_config):
        """A pixel within max_search distance is returned."""
        calc = LengthCalculator(length_config)
        pixels = np.array([[10, 10], [50, 50]])
        result = calc._find_nearest_skeleton_point(pixels, target=(11, 11), max_search=5.0)
        assert result == (10, 10)

    def test_returns_none_when_outside_radius(self, length_config):
        """No pixel within max_search → None."""
        calc = LengthCalculator(length_config)
        pixels = np.array([[10, 10]])
        assert calc._find_nearest_skeleton_point(pixels, target=(50, 50), max_search=5.0) is None

    def test_empty_pixels_returns_none(self, length_config):
        """Empty pixel array short-circuits to None."""
        calc = LengthCalculator(length_config)
        assert calc._find_nearest_skeleton_point(np.empty((0, 2)), (0, 0), 5.0) is None


class TestBridgeSkeletonGaps:
    """Gap-bridging behavior on small synthetic skeletons."""

    def test_no_endpoints_returns_unchanged(self, length_config):
        """A skeleton with <2 endpoints is returned as-is."""
        calc = LengthCalculator(length_config)
        skel = np.zeros((10, 10), dtype=bool)
        skel[5, 5] = True  # single isolated pixel — 0 endpoints
        result = calc.bridge_skeleton_gaps(skel, max_gap=5, max_path=10)
        assert np.array_equal(result, skel)

    def test_bridges_small_gap(self, length_config):
        """Two short collinear segments separated by a 2-pixel gap get bridged."""
        calc = LengthCalculator(length_config)
        skel = np.zeros((20, 20), dtype=bool)
        skel[5, 5:8] = True  # left segment — endpoints at (5,5) and (5,7)
        skel[5, 10:13] = True  # right segment — endpoints at (5,10) and (5,12)
        result = calc.bridge_skeleton_gaps(skel, max_gap=5, max_path=2)
        # Bridge fills the 2-pixel gap between (5,7) and (5,10)
        assert result[5, 8] and result[5, 9]


class TestCalculateGeodesicLength:
    """All four return codes for the public length method."""

    def test_empty_mask_returns_empty_code(self, length_config, empty_mask):
        """RETURN_EMPTY_MASK on a fully-zero mask."""
        calc = LengthCalculator(length_config)
        result, code = calc.calculate_geodesic_length(empty_mask, max_gap=5, max_path=10)
        assert code == RETURN_EMPTY_MASK
        assert result.length == 0.0
        assert result.path == []

    def test_insufficient_pixels_returns_dedicated_code(self, length_config):
        """A mask whose skeleton is <2 pixels triggers RETURN_INSUFFICIENT_PIXELS."""
        calc = LengthCalculator(length_config)
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True  # single pixel skeletonizes to itself
        _, code = calc.calculate_geodesic_length(mask, max_gap=5, max_path=10)
        assert code == RETURN_INSUFFICIENT_PIXELS

    def test_vertical_line_succeeds(self, length_config, vertical_line_mask):
        """A vertical line of 30 pixels yields RETURN_SUCCESS with length≈29."""
        calc = LengthCalculator(length_config)
        result, code = calc.calculate_geodesic_length(vertical_line_mask, max_gap=5, max_path=10)
        assert code == RETURN_SUCCESS
        # 30 cardinal pixels => 29 unit-weight edges
        assert result.length == pytest.approx(29.0, abs=0.5)
        assert result.start is not None and result.end is not None
        assert len(result.path) > 0

    def test_diagonal_line_uses_sqrt2_weighting(self, length_config, diagonal_line_mask):
        """A 20-pixel diagonal yields ≈ 19 * sqrt(2) due to 8-connectivity weights."""
        calc = LengthCalculator(length_config)
        result, code = calc.calculate_geodesic_length(diagonal_line_mask, max_gap=5, max_path=10)
        assert code == RETURN_SUCCESS
        assert result.length == pytest.approx(19 * np.sqrt(2), abs=0.5)

    def test_provided_endpoints_are_used(self, length_config, vertical_line_mask):
        """Explicit start/end points are honored when within search radius."""
        calc = LengthCalculator(length_config)
        result, code = calc.calculate_geodesic_length(
            vertical_line_mask,
            start_point=(10, 25),
            end_point=(39, 25),
            max_gap=5,
            max_path=10,
        )
        assert code == RETURN_SUCCESS
        assert result.start == (10, 25)
        assert result.end == (39, 25)

    def test_use_farthest_endpoint_false_uses_bottom(self, length_config, vertical_line_mask):
        """With use_farthest_endpoint=False and no end_point, end is bottom-most pixel."""
        calc = LengthCalculator(length_config)
        result, code = calc.calculate_geodesic_length(
            vertical_line_mask,
            use_farthest_endpoint=False,
            max_gap=5,
            max_path=10,
        )
        assert code == RETURN_SUCCESS
        assert result.end[0] == 39  # bottom of the line


class TestAnalyzePlantLengths:
    """End-to-end analysis on a labeled array — covers _process_single_plant
    and most of the supporting helpers."""

    def test_two_plant_labels_produce_two_results(self, length_config, two_plant_labels):
        """Both labeled plants get a detected LengthResult; the third is empty."""
        calc = LengthCalculator(length_config)
        results = calc.analyze_plant_lengths(
            two_plant_labels,
            num_plants=3,
            max_gap=5,
            max_path=10,
        )
        assert set(results.keys()) == {0, 1, 2}
        assert results[0].detected is True
        assert results[1].detected is True
        assert results[2].detected is False  # untouched, defaults to EMPTY_RESULT
        # Lengths are roughly the line length (30 pixels → ~29)
        assert results[0].length > 20
        assert results[1].length > 20
        # path_result is populated and remapped to full-image coordinates
        assert results[0].path_result is not None
        assert results[0].path_result.skeleton.shape == two_plant_labels.shape

    def test_invalid_plant_id_label_skipped(self, length_config):
        """Labels mapping to plant_id outside [0, num_plants) are ignored."""
        calc = LengthCalculator(length_config)
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:40, 25] = 99  # plant_id = 98, way out of range
        results = calc.analyze_plant_lengths(labels, num_plants=3, max_gap=5, max_path=10)
        assert all(not r.detected for r in results.values())

    def test_endpoint_info_passed_through(self, length_config, two_plant_labels):
        """Provided endpoint_info is honored for matching plant_ids."""
        calc = LengthCalculator(length_config)
        endpoint_info = {
            0: {"detected": True, "start": (10, 15), "end": (39, 15)},
            1: {"detected": False},  # falls back to defaults
        }
        results = calc.analyze_plant_lengths(
            two_plant_labels,
            num_plants=3,
            max_gap=5,
            max_path=10,
            endpoint_info=endpoint_info,
        )
        assert results[0].detected and results[1].detected
