"""Tests for npec_cv.helpers.extract_petri_dish.extract_petri_dish.

Module under test: src/npec_cv/helpers/extract_petri_dish.py
Function:          extract_petri_dish(img, padding=30, return_coords=False)

Detects the largest object via Otsu thresholding + contour finding, then
returns either a padded square crop or its (x_start, y_start, x_end, y_end)
coordinates. Boundary clamping keeps the crop within the image; if the
square would exceed image bounds, both the start and end are adjusted to
preserve squareness.

Image source policy
-------------------
  REAL images for happy-path tests (return types, square invariant, input
  formats) — exercises the function on data it'll actually see in production.

  SYNTHETIC images for tests that need precise, controlled inputs:
    - Padding-math tests (need known dish dimensions to verify the formula)
    - Edge-clamping tests (real images don't have edge-flush dishes)
    - No-contours fallback (need an all-zeros canvas)
    - Center-alignment test (need known dish center)

Test coverage
-------------
  TestReturnTypes                  [REAL]
  TestSquareInvariant              [REAL + SYNTHETIC for edges]
  TestPaddingMath                  [SYNTHETIC]
  TestEdgeClamping                 [SYNTHETIC]
  TestInputFormats                 [REAL + SYNTHETIC for center]
  TestNoContoursFallback           [SYNTHETIC]
"""

from __future__ import annotations

import numpy as np

from npec_cv.helpers.extract_petri_dish import extract_petri_dish


class TestReturnTypes:
    """Two return modes — ndarray vs coordinate tuple — must be consistent. [REAL]"""

    def test_default_returns_ndarray(self, real_test_image):
        """Default call returns a numpy array."""
        result = extract_petri_dish(real_test_image)

        assert isinstance(result, np.ndarray)
        assert result.ndim in (2, 3)

    def test_return_coords_returns_four_int_tuple(self, real_test_image):
        """return_coords=True returns (x_start, y_start, x_end, y_end) of ints."""
        result = extract_petri_dish(real_test_image, return_coords=True)

        assert isinstance(result, tuple)
        assert len(result) == 4
        # Accept both Python int and numpy integer types (cv2 sometimes returns np.int*)
        assert all(isinstance(v, (int, np.integer)) for v in result)

        h, w = real_test_image.shape[:2]
        x_start, y_start, x_end, y_end = result
        assert 0 <= x_start <= w
        assert 0 <= y_start <= h
        assert 0 <= x_end <= w
        assert 0 <= y_end <= h

    def test_two_modes_describe_same_region(self, real_test_image):
        """The cropped image's shape matches the coords' span."""
        cropped = extract_petri_dish(real_test_image)
        x_start, y_start, x_end, y_end = extract_petri_dish(real_test_image, return_coords=True)

        assert cropped.shape[:2] == (y_end - y_start, x_end - x_start)


class TestSquareInvariant:
    """The output crop is always square — this is the contract."""

    def test_output_is_square_for_any_real_image(self, real_test_image_any):
        """[REAL ×5] Square invariant holds for all 5 test images.

        Parametrized: this test runs once per fixture image. Failing on
        any single one points to a real-image edge case worth investigating.
        """
        cropped = extract_petri_dish(real_test_image_any)

        assert cropped.shape[0] == cropped.shape[1]

    def test_output_is_square_for_edge_dish(self, edge_dish_top_image):
        """[SYNTHETIC] Edge-touching dish still produces a square crop after clamping."""
        cropped = extract_petri_dish(edge_dish_top_image)

        assert cropped.shape[0] == cropped.shape[1]

    def test_output_is_square_for_corner_dish(self, corner_dish_image):
        """[SYNTHETIC] Corner-touching dish (two clamps) still produces a square crop."""
        cropped = extract_petri_dish(corner_dish_image)

        assert cropped.shape[0] == cropped.shape[1]


class TestPaddingMath:
    """Padding math for the interior (no-clamp) case. [SYNTHETIC — needs known dims]"""

    def test_interior_dish_side_equals_max_dim_plus_double_padding(self, make_dish_image):
        """When dish is well inside the image, side = max(w, h) + 2 * padding."""
        # 600x800 dish inside a 2000x2000 canvas — well clear of edges
        img = make_dish_image(
            shape=(2000, 2000),
            dish_bbox=(500, 500, 1100, 1300),  # w=600, h=800
        )

        x_start, y_start, x_end, y_end = extract_petri_dish(img, padding=30, return_coords=True)

        # max(600, 800) + 2*30 = 860
        assert (x_end - x_start) == 860
        assert (y_end - y_start) == 860

    def test_zero_padding_produces_tight_square_crop(self, make_dish_image):
        """padding=0 produces a square sized exactly max(w, h)."""
        img = make_dish_image(
            shape=(2000, 2000),
            dish_bbox=(500, 500, 1100, 1300),  # w=600, h=800
        )

        x_start, y_start, x_end, y_end = extract_petri_dish(img, padding=0, return_coords=True)

        assert (x_end - x_start) == 800
        assert (y_end - y_start) == 800

    def test_large_padding_does_not_exceed_image_bounds(self, make_dish_image):
        """Large padding clamps safely — no negative starts, no out-of-bound ends."""
        img = make_dish_image(
            shape=(1000, 1000),
            dish_bbox=(450, 450, 550, 550),  # tiny 100x100 dish
        )

        x_start, y_start, x_end, y_end = extract_petri_dish(img, padding=2000, return_coords=True)

        assert x_start >= 0
        assert y_start >= 0
        assert x_end <= 1000
        assert y_end <= 1000
        # still square after clamping
        assert (x_end - x_start) == (y_end - y_start)


class TestEdgeClamping:
    """Branches inside calculate_square that handle image-boundary clamps. [SYNTHETIC]"""

    def test_dish_at_top_edge_clamps_y_start_to_zero(self, edge_dish_top_image):
        """When dish is at top edge, y_start is exactly 0."""
        x_start, y_start, x_end, y_end = extract_petri_dish(edge_dish_top_image, return_coords=True)

        assert y_start == 0

    def test_dish_at_right_edge_clamps_x_end_to_width(self, edge_dish_right_image):
        """When dish is at right edge, x_end is exactly img width."""
        x_start, y_start, x_end, y_end = extract_petri_dish(edge_dish_right_image, return_coords=True)

        assert x_end == edge_dish_right_image.shape[1]

    def test_corner_dish_double_clamps_and_adjusts_start(self, corner_dish_image):
        """In a corner, both end-clamps fire and start coords adjust to keep square."""
        x_start, y_start, x_end, y_end = extract_petri_dish(corner_dish_image, return_coords=True)

        h, w = corner_dish_image.shape[:2]
        # both end-clamps fired
        assert y_start == 0
        assert x_end == w
        # output stayed square via the start-adjustment branch
        assert (x_end - x_start) == (y_end - y_start)


class TestInputFormats:
    """Both 2D and 3D inputs are accepted via the bgr2gray helper."""

    def test_grayscale_2d_input_works(self, real_grayscale_image):
        """[REAL] A 2D grayscale image goes through bgr2gray's early-return path."""
        x_start, y_start, x_end, y_end = extract_petri_dish(real_grayscale_image, return_coords=True)

        # valid square region detected
        assert (x_end - x_start) > 0
        assert (y_end - y_start) > 0
        assert (x_end - x_start) == (y_end - y_start)

    def test_bgr_3d_input_works(self, real_test_image):
        """[REAL] A standard 3D BGR image is accepted."""
        x_start, y_start, x_end, y_end = extract_petri_dish(real_test_image, return_coords=True)

        assert (x_end - x_start) > 0
        assert (y_end - y_start) > 0
        assert (x_end - x_start) == (y_end - y_start)

    def test_dish_center_alignment(self, make_dish_image):
        """[SYNTHETIC] The crop's center is approximately the dish's bounding-box center."""
        # Dish centered at (1000, 1000) in a 2000x2000 canvas
        img = make_dish_image(
            shape=(2000, 2000),
            dish_bbox=(500, 500, 1500, 1500),
        )
        expected_center_x = 1000
        expected_center_y = 1000
        padding = 30

        x_start, y_start, x_end, y_end = extract_petri_dish(img, padding=padding, return_coords=True)

        crop_center_x = (x_start + x_end) // 2
        crop_center_y = (y_start + y_end) // 2

        # tolerance accounts for clamping side-effects + integer-division rounding
        assert abs(crop_center_x - expected_center_x) <= padding
        assert abs(crop_center_y - expected_center_y) <= padding


class TestNoContoursFallback:
    """Behavior when Otsu finds no contours — important for safety. [SYNTHETIC]"""

    def test_all_zeros_image_returns_none_in_default_mode(self, empty_image):
        """LOCKED: empty image returns None when return_coords=False."""
        result = extract_petri_dish(empty_image)

        assert result is None

    def test_all_zeros_image_returns_full_bounds_in_coords_mode(self, empty_image):
        """LOCKED: empty image returns (0, 0, width, height) when return_coords=True."""
        h, w = empty_image.shape[:2]

        result = extract_petri_dish(empty_image, return_coords=True)

        assert result == (0, 0, w, h)
