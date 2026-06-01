"""Tests for npec_cv.helpers.padder.padder.

Module under test: src/npec_cv/helpers/padder.py
Function:          padder(image, patch_size) -> np.ndarray

Pads an image so its dimensions are multiples of patch_size. Padding is
added on the BOTTOM and RIGHT only (top-left alignment), matching how the
training pipeline pads inputs. Earlier versions of this helper distributed
padding evenly on all sides; that introduced left/top zero-columns the model
never saw during training and caused a right-side prediction bias. These
tests pin the current corner-padding contract.

Test coverage
-------------
  TestPadderDivisibility
    - output H, W are multiples of patch_size for various inputs
    - already-divisible inputs are returned UNPADDED (no extra phantom patch)
  TestPadderCornerAlignment
    - original content sits at (0, 0)
    - padding is added on the bottom and right only
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

    def test_already_divisible_returns_unpadded(self):
        """LOCKED BEHAVIOR: input already divisible by patch_size is returned as-is.

        Formula: (patch_size - h % patch_size) % patch_size, which is 0 when
        h is already a multiple of patch_size. No phantom extra patch is added.
        """
        img = np.zeros((512, 512, 3), dtype=np.uint8)

        out = padder(img, patch_size=512)

        assert out.shape == (512, 512, 3)

    def test_minimum_padding_added(self):
        """Output is the smallest multiple of patch_size that fits the input."""
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        patch_size = 64

        out = padder(img, patch_size)

        # 100 -> 128 (next multiple of 64); 150 -> 192 (next multiple of 64)
        assert out.shape[:2] == (128, 192)


class TestPadderCornerAlignment:
    """Padding is added on the bottom and right; original sits at (0, 0)."""

    def test_original_content_placed_at_origin(self):
        """The original image content sits at (0, 0) in the output."""
        h, w = 100, 100
        patch_size = 64
        # unique pattern: ascending values so equality is meaningful
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)

        out = padder(img, patch_size)

        assert np.array_equal(out[:h, :w], img)

    def test_padding_added_to_bottom_and_right_only(self):
        """top/left padding = 0; bottom and right hold all the zero padding."""
        img = np.full((99, 99, 3), 255, dtype=np.uint8)
        patch_size = 32

        out = padder(img, patch_size)

        # find non-zero rows/cols (where the original content lives)
        row_nonzero = np.where(out.any(axis=(1, 2)))[0]
        col_nonzero = np.where(out.any(axis=(0, 2)))[0]

        top = int(row_nonzero[0])
        left = int(col_nonzero[0])

        # original content starts at the very top-left
        assert top == 0
        assert left == 0

        # bottom and right contain padding rows/cols
        bottom = out.shape[0] - 1 - int(row_nonzero[-1])
        right = out.shape[1] - 1 - int(col_nonzero[-1])
        assert bottom > 0
        assert right > 0


class TestPadderFillValue:
    """Padded regions must be exactly zero."""

    def test_padded_region_is_zero_for_bgr(self):
        """All-bright BGR input → padded border (bottom + right) is exactly [0, 0, 0]."""
        h, w = 100, 100
        patch_size = 64
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        out = padder(img, patch_size)

        # Padded region is everything outside [:h, :w]
        mask = np.ones(out.shape[:2], dtype=bool)
        mask[:h, :w] = False

        assert (out[mask] == 0).all()

    def test_padded_region_is_zero_for_grayscale(self):
        """All-bright grayscale input → padded border is exactly 0."""
        h, w = 100, 100
        patch_size = 64
        img = np.full((h, w), 255, dtype=np.uint8)

        out = padder(img, patch_size)

        mask = np.ones(out.shape, dtype=bool)
        mask[:h, :w] = False

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
