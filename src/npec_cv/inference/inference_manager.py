"""Module for managing model inference and mask restoration."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from keras.models import load_model

from npec_cv.helpers.patchify import patchify, unpatchify

logger = logging.getLogger(__name__)

from npec_cv.exceptions import ConfigurationError
from npec_cv.helpers.extract_petri_dish import extract_petri_dish
from npec_cv.helpers.padder import padder
from npec_cv.models.inference import ModelConfig


class InferenceManager:
    """Manages image patching, model inference, and mask restoration for segmentation.

    This class handles the complete pipeline for running inference on images that
    require petri dish extraction and patch-based processing. It maintains state
    between patching and unpatching operations to correctly restore predictions
    to the original image coordinate space.

    Attributes:
        config (ModelConfig): Configuration object containing model parameters.
        model: Loaded Keras model instance.
        img_shape (tuple): Shape of the last processed image (H, W, C).
        crop_coords (tuple): Coordinates of extracted petri dish (x1, y1, x2, y2).
        patch_grid_shape (tuple): Grid dimensions of patches (N_rows, N_cols).
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize the InferenceManager and load the model.

        Args:
            model_config: Configuration object containing:
                - checkpoint: Path to saved Keras model
                - patch_size: Size of square patches for inference
                - batch_size: Batch size for model prediction
                - threshold: Confidence threshold for mask binarization
                - custom_objects: Dict of custom Keras objects (losses, metrics)

        Raises:
            ConfigurationError: If model_config is None.
            FileNotFoundError: If checkpoint path doesn't exist.
            ImportError: If model fails to load.
        """
        if model_config is None:
            raise ConfigurationError("model_config is required")

        self.config = model_config
        self.model = None

        # State variables for coordinate tracking between patch/unpatch operations
        self.img_shape: Optional[tuple] = None
        self.crop_coords: Optional[tuple] = None
        self.patch_grid_shape: Optional[tuple] = None
        self.image_to_patch: Optional[np.ndarray] = None

        self._init_model()

        logger.debug(
            f"InferenceManager initialized with patch_size={self.config.patch_size}, "
            f"batch_size={self.config.batch_size}"
        )

    def _init_model(self) -> None:
        """Load the Keras model from checkpoint.

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist.
            ImportError: If model loading fails.
        """
        checkpoint_path = Path(self.config.checkpoint)
        custom_objects = self.config.custom_objects

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        logger.debug(f"Loading model from {checkpoint_path}...")

        try:
            self.model = load_model(checkpoint_path, custom_objects=custom_objects)
            logger.debug(f"Model loaded successfully from {checkpoint_path}")
        except Exception as e:
            logger.exception(f"Failed to load model from {checkpoint_path}")
            raise ImportError(
                f"Could not load model from {checkpoint_path}. " f"Ensure it's a valid .keras or .h5 file."
            ) from e

    def _get_patch_size(self, patch_size: Optional[int]) -> int:
        """Resolve patch size from argument or config.

        Args:
            patch_size: Explicitly provided patch size, or None to use config.

        Returns:
            Resolved patch size.

        Raises:
            ConfigurationError: If no patch size available.
        """
        if patch_size is not None:
            return patch_size
        if self.config.patch_size is not None:
            return self.config.patch_size
        raise ConfigurationError("patch_size must be provided either as argument or in config")

    def _get_batch_size(self, batch_size: Optional[int]) -> int:
        """Resolve batch size from argument or config.

        Args:
            batch_size: Explicitly provided batch size, or None to use config.

        Returns:
            Resolved batch size.

        Raises:
            ConfigurationError: If no batch size available.
        """
        if batch_size is not None:
            return batch_size
        if self.config.batch_size is not None:
            return self.config.batch_size
        raise ConfigurationError("batch_size must be provided either as argument or in config")

    def patch_image(
        self,
        img: np.ndarray,
        patch_size: Optional[int] = None,
        scaling_factor: float = 1.0,
        normalize: bool = False,
    ) -> np.ndarray:
        """Preprocess image into patches ready for model inference.

        This method performs the following steps:
        1. Optionally scales the image
        2. Extracts the petri dish region
        3. Pads the region to be divisible by patch_size
        4. Splits into non-overlapping patches

        The method stores internal state (crop_coords, patch_grid_shape) needed
        for later reconstruction via restore_mask().

        Args:
            img: Input image as numpy array (H, W, C) or (H, W).
            patch_size: Square patch dimension. Uses config value if None.
            scaling_factor: Image scaling before processing. Default 1.0 (no scaling).
            normalize: If True, normalize pixel values to [0, 1]. Default False.

        Returns:
            Array of patches with shape (N_patches, patch_size, patch_size, 3).

        Raises:
            ConfigurationError: If patch_size cannot be determined.

        Note:
            Must be called before create_img_mask() and restore_mask().
        """
        patch_size = self._get_patch_size(patch_size)
        self.img_shape = img.shape

        logger.debug(
            f"Patching image: shape={img.shape}, patch_size={patch_size}, "
            f"scaling={scaling_factor}, normalize={normalize}"
        )

        # Apply scaling if scaling factor provided
        if scaling_factor != 1.0:
            scaled_img = cv2.resize(
                img,
                (0, 0),
                fx=scaling_factor,
                fy=scaling_factor,
                interpolation=cv2.INTER_LINEAR,
            )
            logger.debug(f"Scaled image from {img.shape} to {scaled_img.shape}")
        else:
            scaled_img = img

        # Extract petri dish region
        self.crop_coords = extract_petri_dish(scaled_img, return_coords=True)
        x1_scaled, y1_scaled, x2_scaled, y2_scaled = self.crop_coords

        logger.debug(f"Petri dish detected at (scaled): {self.crop_coords}")

        # Rescale coordinates back to original image space if scaling was applied
        if scaling_factor != 1.0:
            scale_ratio = 1.0 / scaling_factor
            x1 = int(x1_scaled * scale_ratio)
            y1 = int(y1_scaled * scale_ratio)
            x2 = int(x2_scaled * scale_ratio)
            y2 = int(y2_scaled * scale_ratio)
            self.crop_coords = (x1, y1, x2, y2)
            logger.debug(f"Coordinates rescaled to original space: {self.crop_coords}")
        else:
            x1, y1, x2, y2 = self.crop_coords

        # Crop and pad
        dish = img[y1:y2, x1:x2]
        padded_img = padder(dish, patch_size)

        # Ensure 3-channel image
        if padded_img.ndim < 3:
            padded_img = cv2.cvtColor(padded_img, cv2.COLOR_GRAY2BGR)
            logger.debug("Converted grayscale to BGR")

        self.image_to_patch = padded_img

        # Create patches
        patches = patchify(padded_img, (patch_size, patch_size, 3), step=patch_size)
        self.patch_grid_shape = patches.shape[:2]

        # Reshape for model input: (N_patches, H, W, C)
        num_patches = patches.shape[0] * patches.shape[1]
        patches = patches.reshape(num_patches, patch_size, patch_size, 3)

        if normalize:
            patches = patches / 255.0

        logger.debug(f"Created {num_patches} patches " f"(grid: {self.patch_grid_shape[0]}x{self.patch_grid_shape[1]})")

        return patches

    def _unpatch_pred_mask(self, preds: np.ndarray) -> np.ndarray:
        """Reconstruct full mask from patch predictions.

        Args:
            preds: Model predictions with shape (N_patches, H, W) or (N_patches, H, W, 1).

        Returns:
            Reconstructed mask with shape (H_padded, W_padded).

        Raises:
            ValueError: If patch_image() hasn't been called or patch count mismatches.
        """
        if self.patch_grid_shape is None or self.img_shape is None:
            raise ValueError(
                "Cannot unpatch: patch_image() must be called first to initialize " "grid shape and image dimensions."
            )

        # Remove channel dimension if present
        preds = preds.squeeze(axis=-1)

        N_rows, N_cols = self.patch_grid_shape
        expected_patches = N_rows * N_cols
        P = preds.shape[-1]  # Patch size

        if preds.shape[0] != expected_patches:
            raise ValueError(
                f"Patch count mismatch: expected {expected_patches}, got {preds.shape[0]}. "
                f"Grid shape: {self.patch_grid_shape}"
            )

        # Reshape and reconstruct
        preds_reshaped = preds.reshape(N_rows, N_cols, P, P)
        H_padded, W_padded = N_rows * P, N_cols * P

        return unpatchify(preds_reshaped, (H_padded, W_padded))

    def create_img_mask(self, patches: np.ndarray, batch_size: Optional[int] = None) -> Optional[np.ndarray]:
        """Run model inference on patches and reconstruct the padded mask.

        Args:
            patches: Preprocessed patches from patch_image().
            batch_size: Inference batch size. Uses config value if None.

        Returns:
            Reconstructed mask (H_padded, W_padded) as float probabilities,
            or None if inference fails.

        Raises:
            ConfigurationError: If batch_size cannot be determined.
            RuntimeError: If model is not initialized.
        """
        batch_size = self._get_batch_size(batch_size)

        if self.model is None:
            raise RuntimeError("Model not initialized. This should not happen if __init__ succeeded.")

        logger.debug(f"Running inference on {len(patches)} patches, batch_size={batch_size}")

        try:
            preds = self.model.predict(patches, batch_size=batch_size, verbose=0)
            padded_mask = self._unpatch_pred_mask(preds)

            logger.debug(f"Inference complete. Padded mask shape: {padded_mask.shape}")
            return padded_mask

        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            return None

    def restore_mask(self, padded_mask: np.ndarray, patch_size: Optional[int] = None) -> np.ndarray:
        """Restore prediction mask to original image dimensions.

        Reverses the padding and cropping applied during patch_image() to place
        the prediction back into the original image's coordinate space.

        Args:
            padded_mask: Mask from create_img_mask() with shape (H_padded, W_padded).
            patch_size: Patch size used during patching. Uses config value if None.

        Returns:
            Mask with same dimensions as original input image (H_original, W_original).
            Regions outside the petri dish are set to 0.

        Raises:
            ConfigurationError: If patch_size cannot be determined.
            ValueError: If patch_image() hasn't been called first.
        """
        patch_size = self._get_patch_size(patch_size)

        if self.crop_coords is None or self.img_shape is None:
            raise ValueError(
                "Cannot restore mask: patch_image() must be called first to "
                "initialize crop coordinates and image shape."
            )

        x_start, y_start, x_end, y_end = self.crop_coords
        h_crop = y_end - y_start
        w_crop = x_end - x_start

        # Calculate padding that was added (mirrors padder() logic)
        height_padding = ((h_crop // patch_size) + 1) * patch_size - h_crop
        width_padding = ((w_crop // patch_size) + 1) * patch_size - w_crop
        top_pad = height_padding // 2
        left_pad = width_padding // 2

        # Remove padding
        unpadded_mask = padded_mask[top_pad : top_pad + h_crop, left_pad : left_pad + w_crop]

        # Place into full-size canvas
        final_mask = np.zeros(self.img_shape[:2], dtype=padded_mask.dtype)
        final_mask[y_start:y_end, x_start:x_end] = unpadded_mask

        logger.debug(
            f"Mask restored: padded {padded_mask.shape} -> "
            f"unpadded {unpadded_mask.shape} -> final {final_mask.shape}"
        )

        return final_mask

    def run_inference(
        self,
        img: np.ndarray,
        patch_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        scaling_factor: float = 1.0,
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """Convenience method to run the full inference pipeline.

        Combines patch_image(), create_img_mask(), and restore_mask() into
        a single call for simple use cases.

        Args:
            img: Input image (H, W, C) or (H, W).
            patch_size: Square patch dimension.
            batch_size: Inference batch size.
            scaling_factor: Image scaling factor.
            normalize: Whether to normalize patches.

        Returns:
            Final mask in original image coordinates, or None if inference fails.
        """
        logger.debug(f"Running full inference pipeline on image shape {img.shape}")

        patches = self.patch_image(
            img,
            patch_size=patch_size,
            scaling_factor=scaling_factor,
            normalize=normalize,
        )

        padded_mask = self.create_img_mask(patches, batch_size=batch_size)
        if padded_mask is None:
            return None

        return self.restore_mask(padded_mask, patch_size=patch_size)
