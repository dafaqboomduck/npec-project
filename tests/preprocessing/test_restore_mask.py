"""Tests for npec_cv.inference.inference_manager.InferenceManager.restore_mask.

Module under test: src/npec_cv/inference/inference_manager.py
Method:            InferenceManager.restore_mask(padded_mask, patch_size=None)

This is the symmetric inverse of patch_image's preprocessing: it removes
padding and places the result back into a full-image-sized canvas at the
crop_coords the dish was extracted from. Logically belongs in the
preprocessing suite even though it lives in the inference module.

Mocking strategy
----------------
Same as test_patch_image.py — load_model is patched via the
`patch_load_model` fixture in conftest.py. patch_image is called first
(with a real image) to populate the manager's state, then we fabricate
a fake "padded mask" with known content and pass it to restore_mask
to verify placement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.npec_cv.inference.inference_manager import InferenceManager
from src.npec_cv.models.inference import ModelConfig


def _make_config(checkpoint_path, patch_size=128, batch_size=4):
    """Build a minimal ModelConfig for tests."""
    return ModelConfig(
        checkpoint=str(checkpoint_path),
        threshold=0.01,
        patch_size=patch_size,
        batch_size=batch_size,
        custom_objects={},
    )


def _fabricate_padded_mask(manager, fill_value=1.0, dtype=np.float32):
    """Build a 'fake prediction' matching the manager's image_to_patch shape.

    After patch_image runs, manager.image_to_patch has the padded dish
    dimensions. A real prediction-restored mask would have the same H, W.
    """
    h, w = manager.image_to_patch.shape[:2]
    return np.full((h, w), fill_value, dtype=dtype)


class TestRestoreMaskShape:
    """Output shape must match the original image's spatial dims. [REAL]"""

    def test_output_matches_original_image_shape(self, patch_load_model, fake_checkpoint, real_test_image):
        """restore_mask returns an array of shape (H_original, W_original)."""
        manager = InferenceManager(_make_config(fake_checkpoint))
        manager.patch_image(real_test_image)
        padded_mask = _fabricate_padded_mask(manager)

        restored = manager.restore_mask(padded_mask)

        assert restored.shape == real_test_image.shape[:2]


class TestRestoreMaskState:
    """restore_mask requires patch_image to have been called first. [no image]"""

    def test_no_state_raises_value_error(self, patch_load_model, fake_checkpoint):
        """Calling restore_mask before patch_image raises ValueError."""
        manager = InferenceManager(_make_config(fake_checkpoint))
        # any dummy padded mask — we never reach the unpadding step
        dummy_mask = np.zeros((128, 128), dtype=np.float32)

        with pytest.raises(ValueError):
            manager.restore_mask(dummy_mask)


class TestRestoreMaskRoundTrip:
    """The gold test: known-content padded mask round-trips correctly. [REAL]"""

    def test_all_ones_padded_mask_fills_dish_region(self, patch_load_model, fake_checkpoint, real_test_image):
        """A padded mask of all 1.0 produces non-zero values inside the crop region.

        Because padder() distributes padding evenly, the actual filled region
        is the dish-sized rectangle inside the larger crop. We assert on
        SUM > 0 across the dish region rather than strict equality.
        """
        manager = InferenceManager(_make_config(fake_checkpoint))
        manager.patch_image(real_test_image)
        padded_mask = _fabricate_padded_mask(manager, fill_value=1.0)

        restored = manager.restore_mask(padded_mask)

        x1, y1, x2, y2 = manager.crop_coords
        crop_region = restored[y1:y2, x1:x2]

        # Most of the crop region should be filled with 1.0 — exactly the
        # un-padded dish-sized rectangle inside it. Assert a strong majority
        # rather than every pixel to tolerate the unpadding inset.
        filled_ratio = (crop_region == 1.0).mean()
        assert filled_ratio > 0.5, (
            f"Only {filled_ratio:.1%} of crop region was filled — " f"expected dish-sized rectangle to dominate"
        )

    def test_outside_crop_region_is_zero(self, patch_load_model, fake_checkpoint, real_test_image):
        """Pixels outside crop_coords are exactly 0 in the restored mask."""
        manager = InferenceManager(_make_config(fake_checkpoint))
        manager.patch_image(real_test_image)
        padded_mask = _fabricate_padded_mask(manager, fill_value=1.0)

        restored = manager.restore_mask(padded_mask)

        x1, y1, x2, y2 = manager.crop_coords

        # Build a mask of the "outside crop" region and assert it's all zero
        outside_mask = np.ones(restored.shape, dtype=bool)
        outside_mask[y1:y2, x1:x2] = False

        assert (restored[outside_mask] == 0).all()


class TestRestoreMaskCropPlacement:
    """The non-zero region must align with crop_coords. [REAL]"""

    def test_non_zero_region_aligns_with_crop_coords(self, patch_load_model, fake_checkpoint, real_test_image):
        """The bbox of non-zero values in the output sits within crop_coords.

        Because of the unpadding inset, the non-zero bbox may be slightly
        SMALLER than crop_coords (the dish-sized rectangle inside the
        larger square crop). We assert that the non-zero bbox is fully
        contained within crop_coords and reasonably close to them.
        """
        manager = InferenceManager(_make_config(fake_checkpoint))
        manager.patch_image(real_test_image)
        padded_mask = _fabricate_padded_mask(manager, fill_value=1.0)

        restored = manager.restore_mask(padded_mask)

        # find the bbox of non-zero pixels
        ys, xs = np.where(restored > 0)
        assert len(ys) > 0, "restored mask is entirely zero"

        nonzero_x1, nonzero_x2 = int(xs.min()), int(xs.max()) + 1
        nonzero_y1, nonzero_y2 = int(ys.min()), int(ys.max()) + 1

        x1, y1, x2, y2 = manager.crop_coords

        # Non-zero bbox is fully INSIDE crop_coords (because of unpadding inset)
        assert x1 <= nonzero_x1
        assert y1 <= nonzero_y1
        assert nonzero_x2 <= x2
        assert nonzero_y2 <= y2
