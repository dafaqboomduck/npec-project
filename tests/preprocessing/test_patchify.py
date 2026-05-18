"""Tests for npec_cv.helpers.patchify (vendored from dovahcrow/patchify.py).

Module under test: src/npec_cv/helpers/patchify.py
Functions:         patchify, unpatchify, NonUniformStepSizeError

This is vendored library code — we don't aim for full coverage of the
upstream package. Only the behaviors we depend on at the application
boundary are pinned here:

  - Round-trip identity (patchify → unpatchify) for 2D and 3D images
  - Correct grid shape for non-overlapping patches
  - NonUniformStepSizeError raised when reconstruction is impossible

If patchify ever breaks more deeply than these checks, downstream tests
in test_patch_image.py and test_restore_mask.py will catch it loudly.

Test coverage
-------------
  TestPatchifyRoundTrip
    - 2D: patchify(img, (P, P), step=P) → unpatchify → original
    - 3D: same with channels
  TestPatchifyShape
    - patch count and grid shape for non-overlapping case
  TestUnpatchifyErrors
    - NonUniformStepSizeError on impossible reconstruction
"""

from __future__ import annotations

import numpy as np
import pytest
from npec_cv.helpers.patchify import NonUniformStepSizeError, patchify, unpatchify


class TestPatchifyRoundTrip:
    """patchify followed by unpatchify recovers the original image."""

    def test_round_trip_2d_identity(self):
        """A 2D image splits and reassembles with bit-exact equality."""
        # seeded RNG → reproducible, unique-enough pattern that array_equal is meaningful
        rng = np.random.default_rng(seed=42)
        img = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)

        patches = patchify(img, (64, 64), step=64)
        reassembled = unpatchify(patches, img.shape)

        assert np.array_equal(reassembled, img)

    def test_round_trip_3d_identity(self):
        """A 3D BGR image splits and reassembles with bit-exact equality."""
        rng = np.random.default_rng(seed=42)
        img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

        patches = patchify(img, (64, 64, 3), step=64)
        reassembled = unpatchify(patches, img.shape)

        assert np.array_equal(reassembled, img)


class TestPatchifyShape:
    """Patch count and grid layout for non-overlapping patching."""

    def test_grid_shape_for_non_overlapping_patches(self):
        """For HxW image with patch_size=P and step=P, grid is (H/P, W/P, P, P)."""
        img = np.zeros((512, 768), dtype=np.uint8)

        patches = patchify(img, (128, 128), step=128)

        assert patches.shape == (4, 6, 128, 128)

    def test_grid_shape_for_3d_non_overlapping(self):
        """3D grid includes a channel-axis of size 1 between counts and patch dims."""
        img = np.zeros((512, 768, 3), dtype=np.uint8)

        patches = patchify(img, (128, 128, 3), step=128)

        assert patches.shape == (4, 6, 1, 128, 128, 3)


class TestUnpatchifyErrors:
    """unpatchify must reject reconstructions it can't perform uniformly."""

    def test_non_uniform_step_raises(self):
        """Mismatched dims trigger NonUniformStepSizeError.

        Reconstruction step is computed as (img_dim - patch_dim) / (n_patches - 1).
        With n_patches=3, patch=100, target=350: step = (350 - 100) / 2 = 125 ✓
        With target=351: step = 251 / 2 = 125.5 → not an integer → raises.
        """
        # build a fake patches array claiming to reconstruct a 351-wide image
        patches = np.zeros((1, 3, 100, 100), dtype=np.uint8)

        with pytest.raises(NonUniformStepSizeError):
            unpatchify(patches, (100, 351))
