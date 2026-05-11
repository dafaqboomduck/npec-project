"""Tests for npec_cv.helpers.padder.padder.

Module under test: src/npec_cv/helpers/padder.py
Function:          padder(image, patch_size) -> np.ndarray

Pads an image so its dimensions are multiples of patch_size, with
padding distributed evenly using cv2.copyMakeBorder + BORDER_CONSTANT.

Test coverage
-------------
  TestPadderDivisibility
    - output H, W are multiples of patch_size for various inputs
    - LOCKED quirk: already-divisible inputs still get a full extra patch
  TestPadderSymmetry
    - top/bottom and left/right padding differ by at most 1px
    - original content is positioned at (top, left) in the output
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

    def test_already_divisible_still_pads_full_patch(self):
        """LOCKED BEHAVIOR: input divisible by patch_size still gets +patch_size.

        Formula: ((h // P) + 1) * P - h. When h % P == 0 this equals P, not 0.
        Pinning this to surface any silent change. May be reconsidered later.
        """
        img = np.zeros((512, 512, 3), dtype=np.uint8)

        out = padder(img, patch_size=512)

        assert out.shape == (1024, 1024, 3)


class TestPadderSymmetry:
    """Padding is split evenly across opposite sides."""

    def test_padding_symmetric_within_one_pixel(self):
        """top/bottom and left/right padding sizes differ by at most 1 pixel.

        Use a uniformly-bright image so the zero-padded border is identifiable
        by simply finding rows/cols that are all zero.
        """
        img = np.full((99, 99, 3), 255, dtype=np.uint8)
        patch_size = 32

        out = padder(img, patch_size)

        # find non-zero rows/cols (where the original content lives)
        row_nonzero = np.where(out.any(axis=(1, 2)))[0]
        col_nonzero = np.where(out.any(axis=(0, 2)))[0]

        top = int(row_nonzero[0])
        bottom = out.shape[0] - 1 - int(row_nonzero[-1])
        left = int(col_nonzero[0])
        right = out.shape[1] - 1 - int(col_nonzero[-1])

        assert abs(top - bottom) <= 1
        assert abs(left - right) <= 1

    def test_original_content_placed_at_top_left_offset(self):
        """The original image content sits at (top_pad, left_pad) in the output."""
        h, w = 100, 100
        patch_size = 64
        # unique pattern: ascending values so equality is meaningful
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)

        out = padder(img, patch_size)

        # mirror padder's formula
        height_padding = ((h // patch_size) + 1) * patch_size - h
        width_padding = ((w // patch_size) + 1) * patch_size - w
        top = height_padding // 2
        left = width_padding // 2

        assert np.array_equal(out[top : top + h, left : left + w], img)


class TestPadderFillValue:
    """Padded regions must be exactly zero."""

    def test_padded_region_is_zero_for_bgr(self):
        """All-bright BGR input → padded border is exactly [0, 0, 0]."""
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        patch_size = 64

        out = padder(img, patch_size)

        # locate where the original image content sits
        height_padding = ((100 // patch_size) + 1) * patch_size - 100
        width_padding = ((100 // patch_size) + 1) * patch_size - 100
        top = height_padding // 2
        left = width_padding // 2

        # build a mask of the "padded region" — everything except [top:top+100, left:left+100]
        mask = np.ones(out.shape[:2], dtype=bool)
        mask[top : top + 100, left : left + 100] = False

        assert (out[mask] == 0).all()

    def test_padded_region_is_zero_for_grayscale(self):
        """All-bright grayscale input → padded border is exactly 0."""
        img = np.full((100, 100), 255, dtype=np.uint8)
        patch_size = 64

        out = padder(img, patch_size)

        height_padding = ((100 // patch_size) + 1) * patch_size - 100
        width_padding = ((100 // patch_size) + 1) * patch_size - 100
        top = height_padding // 2
        left = width_padding // 2

        mask = np.ones(out.shape, dtype=bool)
        mask[top : top + 100, left : left + 100] = False

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
