"""Tests for npec_cv.helpers.extract_petri_dish.extract_petri_dish.

Module under test: src/npec_cv/helpers/extract_petri_dish.py
Function:          extract_petri_dish(img, padding=30, return_coords=False,
                                       *, dark_thresh=25, min_dish_ratio=0.25)

Detects the largest BRIGHT connected component (the petri dish on its black
border) and returns either a padded *bounding-box* crop or its
``(x_start, y_start, x_end, y_end)`` coordinates. **The crop is NOT forced
to be square** — the U-Net works on arbitrary aspect ratios because patching
pads to a multiple of patch_size regardless. Earlier versions of this helper
did force a square crop; that's been intentionally removed because it
sometimes cut off real dish content when the dish was wider than tall.

Image source policy
-------------------
  REAL images for happy-path tests (return types, input formats) — exercises
  the function on data it'll actually see in production.

  SYNTHETIC images for tests that need precise, controlled inputs
  (padding math, edge clamping, no-detection fallback).

  Many synthetic fixtures use a small 600×800-ish bright rectangle on a
  large canvas, which falls below the default ``min_dish_ratio=0.25`` sanity
  check (the fallback kicks in). Tests that want the rectangle to be picked
  up as the "dish" pass an explicit ``min_dish_ratio=0.01``.

Test coverage
-------------
  TestReturnTypes                  [REAL]
  TestPaddingMath                  [SYNTHETIC]
  TestEdgeClamping                 [SYNTHETIC]
  TestInputFormats                 [REAL + SYNTHETIC for center]
  TestFallback                     [SYNTHETIC]
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
        assert all(isinstance(v, (int, np.integer)) for v in result)

        h, w = real_test_image.shape[:2]
        x_start, y_start, x_end, y_end = result
        assert 0 <= x_start <= w
        assert 0 <= y_start <= h
        assert 0 <= x_end <= w
        assert 0 <= y_end <= h
        # End must be > start (non-empty crop) in both axes
        assert x_end > x_start
        assert y_end > y_start

    def test_two_modes_describe_same_region(self, real_test_image):
        """The cropped image's shape matches the coords' span."""
        cropped = extract_petri_dish(real_test_image)
        x_start, y_start, x_end, y_end = extract_petri_dish(real_test_image, return_coords=True)

        assert cropped.shape[:2] == (y_end - y_start, x_end - x_start)


class TestPaddingMath:
    """Padding math for the interior (no-clamp) case. [SYNTHETIC — needs known dims]

    The new contract: the returned box is the dish bbox grown by ``padding``
    on each side independently, clamped to image bounds. Width and height are
    independent — the crop is rectangular, not square.
    """

    def test_interior_dish_dimensions_equal_dish_size_plus_double_padding(self, make_dish_image):
        """When dish is well inside the image, width = bw + 2p, height = bh + 2p."""
        # 600x800 dish inside a 2000x2000 canvas — well clear of edges.
        # 12% of image area, below the default 25% sanity check — pass
        # min_dish_ratio=0.01 so the synthetic blob is accepted as the dish.
        img = make_dish_image(
            shape=(2000, 2000),
            dish_bbox=(500, 500, 1100, 1300),  # w=600, h=800
        )

        x_start, y_start, x_end, y_end = extract_petri_dish(img, padding=30, return_coords=True, min_dish_ratio=0.01)

        # 600 + 2*30 = 660 wide ; 800 + 2*30 = 860 tall
        assert (x_end - x_start) == 660
        assert (y_end - y_start) == 860

    def test_zero_padding_produces_tight_bbox_crop(self, make_dish_image):
        """padding=0 produces a crop sized exactly to the dish bbox."""
        img = make_dish_image(
            shape=(2000, 2000),
            dish_bbox=(500, 500, 1100, 1300),  # w=600, h=800
        )

        x_start, y_start, x_end, y_end = extract_petri_dish(img, padding=0, return_coords=True, min_dish_ratio=0.01)

        assert (x_end - x_start) == 600
        assert (y_end - y_start) == 800

    def test_large_padding_clamps_to_image_bounds(self, make_dish_image):
        """Excessive padding clamps safely — no negative starts, no out-of-bound ends."""
        img = make_dish_image(
            shape=(1000, 1000),
            dish_bbox=(450, 450, 550, 550),  # tiny 100x100 dish
        )

        x_start, y_start, x_end, y_end = extract_petri_dish(img, padding=2000, return_coords=True, min_dish_ratio=0.01)

        assert x_start >= 0
        assert y_start >= 0
        assert x_end <= 1000
        assert y_end <= 1000
        # With padding far exceeding distance-to-edge, the crop spans the full image.
        assert (x_start, y_start, x_end, y_end) == (0, 0, 1000, 1000)


class TestEdgeClamping:
    """Boundary clamping when the padded bbox would exceed image edges. [SYNTHETIC]"""

    def test_dish_at_top_edge_clamps_y_start_to_zero(self, edge_dish_top_image):
        """When dish touches the top edge, y_start is clamped to 0."""
        x_start, y_start, x_end, y_end = extract_petri_dish(
            edge_dish_top_image, return_coords=True, min_dish_ratio=0.01
        )

        assert y_start == 0

    def test_dish_at_right_edge_clamps_x_end_to_width(self, edge_dish_right_image):
        """When dish touches the right edge, x_end is clamped to img width."""
        x_start, y_start, x_end, y_end = extract_petri_dish(
            edge_dish_right_image, return_coords=True, min_dish_ratio=0.01
        )

        assert x_end == edge_dish_right_image.shape[1]

    def test_corner_dish_clamps_both_dimensions(self, corner_dish_image):
        """A corner dish triggers two end-clamps; start coords stay non-negative."""
        x_start, y_start, x_end, y_end = extract_petri_dish(corner_dish_image, return_coords=True, min_dish_ratio=0.01)

        h, w = corner_dish_image.shape[:2]
        # both end-clamps fired
        assert y_start == 0
        assert x_end == w
        # coordinates stay inside the image
        assert x_start >= 0
        assert y_end <= h


class TestInputFormats:
    """Both 2D and 3D inputs are accepted; color is collapsed to luminance internally."""

    def test_grayscale_2d_input_works(self, real_grayscale_image):
        """[REAL] A 2D grayscale image is accepted (no-op color collapse)."""
        x_start, y_start, x_end, y_end = extract_petri_dish(real_grayscale_image, return_coords=True)

        # valid non-empty region detected
        assert (x_end - x_start) > 0
        assert (y_end - y_start) > 0

    def test_bgr_3d_input_works(self, real_test_image):
        """[REAL] A standard 3D BGR image is accepted."""
        x_start, y_start, x_end, y_end = extract_petri_dish(real_test_image, return_coords=True)

        assert (x_end - x_start) > 0
        assert (y_end - y_start) > 0

    def test_dish_center_alignment(self, make_dish_image):
        """[SYNTHETIC] The crop's center is approximately the dish's bbox center."""
        # 1000x1000 dish (25% of canvas — exactly the default min_dish_ratio).
        # Pass an explicit lower ratio so this isn't borderline.
        img = make_dish_image(
            shape=(2000, 2000),
            dish_bbox=(500, 500, 1500, 1500),
        )
        expected_center_x = 1000
        expected_center_y = 1000
        padding = 30

        x_start, y_start, x_end, y_end = extract_petri_dish(
            img, padding=padding, return_coords=True, min_dish_ratio=0.01
        )

        crop_center_x = (x_start + x_end) // 2
        crop_center_y = (y_start + y_end) // 2

        # tolerance accounts for clamping side-effects + integer-division rounding
        assert abs(crop_center_x - expected_center_x) <= padding
        assert abs(crop_center_y - expected_center_y) <= padding


class TestFallback:
    """Behavior when no plausible dish is found — safe degradation. [SYNTHETIC]"""

    def test_all_zeros_image_returns_full_image_copy_in_default_mode(self, empty_image):
        """LOCKED: empty image falls back to a copy of the full image (not None)."""
        result = extract_petri_dish(empty_image)

        # Returns the full image as a fallback. (Previous versions returned
        # None here; the new contract returns a real array so downstream code
        # doesn't need to None-check.)
        assert isinstance(result, np.ndarray)
        assert result.shape == empty_image.shape
        # It's a copy — same content, but writable without mutating the input
        assert np.array_equal(result, empty_image)

    def test_all_zeros_image_returns_full_bounds_in_coords_mode(self, empty_image):
        """LOCKED: empty image returns (0, 0, width, height) when return_coords=True."""
        h, w = empty_image.shape[:2]

        result = extract_petri_dish(empty_image, return_coords=True)

        assert result == (0, 0, w, h)

    def test_tiny_blob_below_min_ratio_falls_back(self, make_dish_image):
        """A blob smaller than min_dish_ratio is rejected → full-image fallback."""
        # Tiny 50x50 bright square on a 2000x2000 canvas: ~0.0006 of image area.
        img = make_dish_image(
            shape=(2000, 2000),
            dish_bbox=(975, 975, 1025, 1025),
        )

        result = extract_petri_dish(img, return_coords=True, min_dish_ratio=0.25)

        # Bbox spans the full image (not the tiny blob) → fallback fired.
        assert result == (0, 0, 2000, 2000)
