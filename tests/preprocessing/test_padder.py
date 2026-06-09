"""Tests for npec_cv.helpers.padder.padder.

Module under test: src/npec_cv/helpers/padder.py
Function:          padder(image, patch_size) -> np.ndarray

Pads an image so its dimensions are multiples of patch_size. Padding is
distributed evenly on all sides (top/bottom, left/right), matching the
preprocessing the deployed model was trained with. The formula always adds
at least some padding: when dimensions are already multiples of patch_size,
a full extra patch is added on each axis. These tests pin the current
centered-padding contract.

Test coverage
-------------
  TestPadderDivisibility
    - output H, W are multiples of patch_size for various inputs
    - already-divisible inputs receive a full extra patch (not returned as-is)
  TestPadderCornerAlignment
    - original content is centered (offset by top/left padding)
    - padding is present on all four sides
  TestPadderFillValue
    - padded regions are exactly zero (BGR and grayscale)
  TestPadderTypePreservation
    - dtype preserved
    - 2D input stays 2D, 3D input stays 3D
"""

from __future__ import annotations

import numpy as np
import pytest
from npec_cv.helpers.padder import padder


class TestPadderDivisibility:
    """Output dimensions must be multiples of patch_size."""

    @pytest.mark.parametrize(
        "shape,patch_size",
        [
            ((100, 150), 64),
            ((513, 513), 256),
            ((1000, 1500), 128),
            ((37, 37), 16),
        ],
    )
    def test_output_dims_are_multiples_of_patch_size(self, shape, patch_size):
        """For arbitrary input dims, output dims are exact multiples of patch_size."""
        h, w = shape
        img = np.zeros((h, w, 3), dtype=np.uint8)

        out = padder(img, patch_size)

        assert out.shape[0] % patch_size == 0
        assert out.shape[1] % patch_size == 0

    def test_already_divisible_adds_full_extra_patch(self):
        """LOCKED BEHAVIOR: input already divisible by patch_size receives a full extra patch.

        Formula: ((h // patch_size) + 1) * patch_size - h, which equals patch_size
        when h is already a multiple of patch_size. No early return for aligned inputs.
        """
        img = np.zeros((512, 512, 3), dtype=np.uint8)

        out = padder(img, patch_size=512)

        assert out.shape == (1024, 1024, 3)

    def test_padding_amount_for_non_divisible_input(self):
        """Output is the next multiple of patch_size above the input."""
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        patch_size = 64

        out = padder(img, patch_size)

        # 100 -> 128 (next multiple of 64); 150 -> 192 (next multiple of 64)
        assert out.shape[:2] == (128, 192)


class TestPadderCornerAlignment:
    """Padding is distributed on all sides; original content is centered."""

    def test_original_content_offset_by_top_left_padding(self):
        """The original image content is offset by top and left padding amounts."""
        h, w = 100, 100
        patch_size = 64
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)

        out = padder(img, patch_size)

        # height_padding = ((100//64)+1)*64 - 100 = 128 - 100 = 28; top = 14
        # width_padding  = ((100//64)+1)*64 - 100 = 128 - 100 = 28; left = 14
        top = 14
        left = 14
        assert np.array_equal(out[top : top + h, left : left + w], img)

    def test_padding_present_on_all_four_sides(self):
        """Padding rows/columns appear at the top, bottom, left, and right."""
        img = np.full((99, 99, 3), 255, dtype=np.uint8)
        patch_size = 32

        out = padder(img, patch_size)

        # find rows and columns that contain non-zero (original) content
        row_nonzero = np.where(out.any(axis=(1, 2)))[0]
        col_nonzero = np.where(out.any(axis=(0, 2)))[0]

        top_pad = int(row_nonzero[0])
        left_pad = int(col_nonzero[0])
        bottom_pad = out.shape[0] - 1 - int(row_nonzero[-1])
        right_pad = out.shape[1] - 1 - int(col_nonzero[-1])

        assert top_pad > 0
        assert left_pad > 0
        assert bottom_pad > 0
        assert right_pad > 0


class TestPadderFillValue:
    """Padded regions must be exactly zero."""

    def test_padded_region_is_zero_for_bgr(self):
        """All-bright BGR input → all padded borders are exactly [0, 0, 0]."""
        h, w = 100, 100
        patch_size = 64
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        out = padder(img, patch_size)

        # height_padding=28 top=14 bottom=14; width_padding=28 left=14 right=14
        top = 14
        left = 14
        mask = np.ones(out.shape[:2], dtype=bool)
        mask[top : top + h, left : left + w] = False

        assert (out[mask] == 0).all()

    def test_padded_region_is_zero_for_grayscale(self):
        """All-bright grayscale input → all padded borders are exactly 0."""
        h, w = 100, 100
        patch_size = 64
        img = np.full((h, w), 255, dtype=np.uint8)

        out = padder(img, patch_size)

        top = 14
        left = 14
        mask = np.ones(out.shape, dtype=bool)
        mask[top : top + h, left : left + w] = False

        assert (out[mask] == 0).all()


class TestPadderTypePreservation:
    """Input dtype and dimensionality must round-trip."""

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.int32])
    def test_dtype_preserved(self, dtype):
        """Output dtype equals input dtype."""
        img = np.zeros((100, 100, 3), dtype=dtype)

        out = padder(img, patch_size=64)

        assert out.dtype == img.dtype

    def test_2d_input_stays_2d(self):
        """2D grayscale mask input produces 2D output."""
        img = np.zeros((100, 100), dtype=np.uint8)

        out = padder(img, patch_size=64)

        assert out.ndim == 2

    def test_3d_input_stays_3d(self):
        """3D BGR input produces 3D output with 3 channels."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        out = padder(img, patch_size=64)

        assert out.ndim == 3
        assert out.shape[2] == 3
