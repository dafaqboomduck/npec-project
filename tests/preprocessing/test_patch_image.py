"""Tests for npec_cv.inference.inference_manager.InferenceManager.patch_image.

Module under test: src/npec_cv/inference/inference_manager.py
Method:            InferenceManager.patch_image(img, patch_size=None,
                                                scaling_factor=1.0, normalize=False)

This is the integration point where the preprocessing pipeline composes:
extract_petri_dish → padder → patchify → optional channel fix and normalize.
It also stores state (img_shape, crop_coords, patch_grid_shape, image_to_patch)
needed later by restore_mask.

Mocking strategy
----------------
We mock keras.models.load_model so InferenceManager can be instantiated
without real weights. The `patch_load_model` and `fake_checkpoint`
fixtures (in conftest.py) handle the setup. The mock model is never
actually called by patch_image — only by create_img_mask, which is
covered in Suite 2.

Image source policy
-------------------
  REAL images for everything that exercises the preprocessing pipeline
  on production-shaped inputs (shape contracts, state population,
  scaling, channel handling, normalization).

  Constructor tests use no images at all.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.npec_cv.exceptions import ConfigurationError
from src.npec_cv.helpers.extract_petri_dish import extract_petri_dish
from src.npec_cv.inference.inference_manager import InferenceManager
from src.npec_cv.models.inference import ModelConfig

# ---------------------------------------------------------------------------
# Helpers — small ModelConfig builders for these tests
# ---------------------------------------------------------------------------


def _make_config(checkpoint_path, patch_size=128, batch_size=4):
    """Build a minimal ModelConfig for tests."""
    return ModelConfig(
        checkpoint=str(checkpoint_path),
        threshold=0.01,
        patch_size=patch_size,
        batch_size=batch_size,
        custom_objects={},
    )


# ---------------------------------------------------------------------------


class TestPatchImageOutputShape:
    """Shape contract of the patch array returned by patch_image. [REAL]"""

    def test_output_has_four_dims_with_correct_patch_size(self, patch_load_model, fake_checkpoint, real_test_image):
        """Output shape is (N, patch_size, patch_size, 3)."""
        manager = InferenceManager(_make_config(fake_checkpoint, patch_size=128))

        patches = manager.patch_image(real_test_image)

        assert patches.shape[1:] == (128, 128, 3)

    def test_patch_count_matches_grid_shape(self, patch_load_model, fake_checkpoint, real_test_image):
        """patches.shape[0] == patch_grid_shape[0] * patch_grid_shape[1]."""
        manager = InferenceManager(_make_config(fake_checkpoint, patch_size=128))

        patches = manager.patch_image(real_test_image)

        assert patches.shape[0] == manager.patch_grid_shape[0] * manager.patch_grid_shape[1]

    def test_grid_shape_matches_padded_dish_dimensions(self, patch_load_model, fake_checkpoint, real_test_image):
        """image_to_patch dimensions = grid_shape × patch_size."""
        manager = InferenceManager(_make_config(fake_checkpoint, patch_size=128))

        manager.patch_image(real_test_image)

        rows, cols = manager.patch_grid_shape
        assert manager.image_to_patch.shape[0] == rows * 128
        assert manager.image_to_patch.shape[1] == cols * 128


class TestPatchImageState:
    """patch_image must populate state needed by restore_mask. [REAL]"""

    def test_img_shape_stored(self, patch_load_model, fake_checkpoint, real_test_image):
        """img_shape equals the input image's shape exactly."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        manager.patch_image(real_test_image)

        assert manager.img_shape == real_test_image.shape

    def test_crop_coords_is_four_int_tuple(self, patch_load_model, fake_checkpoint, real_test_image):
        """crop_coords is (x1, y1, x2, y2) of ints, with x1<x2 and y1<y2."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        manager.patch_image(real_test_image)

        assert len(manager.crop_coords) == 4
        assert all(isinstance(v, (int, np.integer)) for v in manager.crop_coords)
        x1, y1, x2, y2 = manager.crop_coords
        assert x1 < x2
        assert y1 < y2

    def test_crop_coords_within_image_bounds_for_any_real_image(
        self, patch_load_model, fake_checkpoint, real_test_image_any
    ):
        """[×5] All crop coordinates lie within image bounds for every real image."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        manager.patch_image(real_test_image_any)

        h, w = real_test_image_any.shape[:2]
        x1, y1, x2, y2 = manager.crop_coords
        assert 0 <= x1 <= w
        assert 0 <= x2 <= w
        assert 0 <= y1 <= h
        assert 0 <= y2 <= h

    def test_image_to_patch_is_padded_dish_region(self, patch_load_model, fake_checkpoint, real_test_image):
        """image_to_patch dimensions are multiples of patch_size."""
        manager = InferenceManager(_make_config(fake_checkpoint, patch_size=128))

        manager.patch_image(real_test_image)

        assert manager.image_to_patch.shape[0] % 128 == 0
        assert manager.image_to_patch.shape[1] % 128 == 0

    def test_patch_grid_shape_populated(self, patch_load_model, fake_checkpoint, real_test_image):
        """patch_grid_shape is a 2-tuple of positive ints."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        manager.patch_image(real_test_image)

        rows, cols = manager.patch_grid_shape
        assert rows > 0
        assert cols > 0


class TestPatchImageNormalization:
    """The normalize flag controls dtype and range of the output. [REAL]"""

    def test_normalize_true_produces_floats_in_unit_range(self, patch_load_model, fake_checkpoint, real_test_image):
        """With normalize=True, output is float and values lie in [0.0, 1.0]."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        patches = manager.patch_image(real_test_image, normalize=True)

        assert np.issubdtype(patches.dtype, np.floating)
        assert patches.min() >= 0.0
        assert patches.max() <= 1.0

    def test_normalize_false_preserves_uint8_range(self, patch_load_model, fake_checkpoint, real_test_image):
        """With normalize=False, output keeps uint8 dtype and 0–255 range."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        patches = manager.patch_image(real_test_image, normalize=False)

        assert patches.dtype == np.uint8
        # real image has bright pixels well above 1
        assert patches.max() > 1


class TestPatchImageScaling:
    """The scaling_factor argument and crop_coords rescale logic. [REAL]"""

    def test_default_scaling_is_no_op(self, patch_load_model, fake_checkpoint, real_test_image):
        """scaling_factor=1.0 (default) leaves crop_coords in original space."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        manager.patch_image(real_test_image, scaling_factor=1.0)

        # at scale=1.0, manager's crop_coords should match a direct call to
        # extract_petri_dish on the raw image
        reference = extract_petri_dish(real_test_image, return_coords=True)
        assert manager.crop_coords == reference

    def test_scaling_rescales_crop_coords_back_to_original_space(
        self, patch_load_model, fake_checkpoint, real_test_image
    ):
        """With scaling_factor=0.5, crop_coords are reported in ORIGINAL coords.

        This is the trickiest invariant in the file: the function detects
        the dish in the down-scaled image, then multiplies the coords by
        1/scaling_factor before storing. The 0.5× → 2× round-trip plus
        independent dish detection at half-resolution introduces drift on
        the order of ~2/scaling_factor pixels per coord, plus the padding
        interaction at scale boundaries.
        """
        manager = InferenceManager(_make_config(fake_checkpoint))

        manager.patch_image(real_test_image, scaling_factor=0.5)

        h, w = real_test_image.shape[:2]
        x1, y1, x2, y2 = manager.crop_coords

        # 1. Coords are in original image space, not the scaled half-resolution
        #    space. If they were in scaled space, x2/y2 would max out around w/2.
        assert x2 > w / 2, f"x2={x2} suggests coords are still in scaled space"
        assert y2 > h / 2, f"y2={y2} suggests coords are still in scaled space"

        # 2. Detected region overlaps heavily with the reference (>= 95% IoU)
        ref_x1, ref_y1, ref_x2, ref_y2 = extract_petri_dish(real_test_image, return_coords=True)

        # intersection
        ix1, iy1 = max(x1, ref_x1), max(y1, ref_y1)
        ix2, iy2 = min(x2, ref_x2), min(y2, ref_y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        # union
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
        union = area_a + area_b - inter

        iou = inter / union
        assert iou >= 0.95, (
            f"IoU={iou:.3f} between scaled and reference crops too low. "
            f"crop_coords={manager.crop_coords}, reference=({ref_x1}, {ref_y1}, "
            f"{ref_x2}, {ref_y2})"
        )


class TestPatchImageChannelHandling:
    """2D grayscale must be promoted to 3-channel before patching. [REAL]"""

    def test_grayscale_input_promoted_to_three_channels(self, patch_load_model, fake_checkpoint, real_grayscale_image):
        """A 2D grayscale image yields 3-channel patches via cv2.cvtColor."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        patches = manager.patch_image(real_grayscale_image)

        assert patches.shape[-1] == 3

    def test_bgr_input_keeps_three_channels(self, patch_load_model, fake_checkpoint, real_test_image):
        """A 3D BGR image keeps 3 channels in the output."""
        manager = InferenceManager(_make_config(fake_checkpoint))

        patches = manager.patch_image(real_test_image)

        assert patches.shape[-1] == 3


class TestPatchSizeResolution:
    """patch_size precedence: argument > config > error. [REAL]"""

    def test_argument_overrides_config(self, patch_load_model, fake_checkpoint, real_test_image):
        """Explicit patch_size arg takes precedence over config value."""
        manager = InferenceManager(_make_config(fake_checkpoint, patch_size=512))

        patches = manager.patch_image(real_test_image, patch_size=128)

        assert patches.shape[1:3] == (128, 128)

    def test_falls_back_to_config_when_arg_none(self, patch_load_model, fake_checkpoint, real_test_image):
        """Without an explicit arg, config.patch_size is used."""
        manager = InferenceManager(_make_config(fake_checkpoint, patch_size=128))

        patches = manager.patch_image(real_test_image)

        assert patches.shape[1:3] == (128, 128)

    def test_raises_when_neither_provided(self, patch_load_model, fake_checkpoint, real_test_image):
        """If both config.patch_size and arg are None, ConfigurationError is raised."""
        # ModelConfig is a dataclass with patch_size: int — annotations aren't
        # enforced at runtime, so passing None works.
        config = _make_config(fake_checkpoint, patch_size=None)
        manager = InferenceManager(config)

        with pytest.raises(ConfigurationError):
            manager.patch_image(real_test_image, patch_size=None)


class TestInferenceManagerConstructor:
    """Constructor-level tests; cheap and complete the picture. [no images]"""

    def test_none_config_raises_configuration_error(self):
        """InferenceManager(None) raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            InferenceManager(None)

    def test_missing_checkpoint_raises_file_not_found(self, tmp_path):
        """A non-existent checkpoint path raises FileNotFoundError.

        Note: this test does NOT use patch_load_model — the existence check
        fires before load_model is even called.
        """
        nonexistent = tmp_path / "does_not_exist.keras"
        config = _make_config(nonexistent)

        with pytest.raises(FileNotFoundError):
            InferenceManager(config)

    def test_load_model_failure_wraps_in_import_error(self, fake_checkpoint):
        """When load_model raises internally, InferenceManager wraps it as ImportError."""
        config = _make_config(fake_checkpoint)

        # patch load_model directly (bypassing the patch_load_model fixture)
        # so we can make it raise instead of returning a mock
        def raising_load_model(*args, **kwargs):
            raise RuntimeError("simulated keras failure")

        with patch(
            "src.npec_cv.inference.inference_manager.load_model",
            side_effect=raising_load_model,
        ):
            with pytest.raises(ImportError):
                InferenceManager(config)
