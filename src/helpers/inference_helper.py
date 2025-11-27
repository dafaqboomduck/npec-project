import os
import numpy as np
from typing import Optional

from .padder import padder
from .extract_petri_dish import extract_petri_dish

class InferenceHelper:
    """
    Manages image patching and unpatching for model inference.
    """    
    def __init__(self):
        # Initialize state variables to ensure they exist
        self.img_shape = None
        self.image_to_patch = None
        self.patch_grid_shape = None # Stores (N_rows, N_cols)

    def patch_image(self, img: np.array, patch_size: int = 256, scaling_factor: float = 1.0, normalize: bool = False) -> np.array:
        import cv2
        from patchify import patchify
        """Scales, pads, extracts, and patches the image for inference."""

        self.img_shape = img.shape[::-1]
        
        if scaling_factor != 1.0:
            # Note: Imports like cv2 should ideally be at the top of the file
            scaled_im = cv2.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor)
        else:
            scaled_im = img

        # Extract the petri dish area
        dish = extract_petri_dish(scaled_im)

        # Pad the image
        padded_im = padder(dish, patch_size)

        if padded_im.shape != 3:
            padded_im = cv2.cvtColor(padded_im, cv2.COLOR_GRAY2BGR)

        self.image_to_patch = padded_im

        # Patch the image (Assuming 3-channel image)
        patches = patchify(padded_im, (patch_size, patch_size, 3), step=patch_size)
        
        # Store the shape of the patch grid for unpatching
        self.patch_grid_shape = patches.shape[:2]

        if normalize:
            # Flatten the patches for model input, then normalize
            num_patches = patches.shape[0] * patches.shape[1] * patches.shape[2]
            # Reshape to (N_total_patches, patch_size, patch_size, 3)
            patches = patches.reshape(num_patches, patch_size, patch_size, 3)
            # Normalization logic
            patches = patches / 255.0 
    
        return patches

    def unpatch_pred_mask(self, preds: np.array, img: Optional[np.array] = None) -> np.array:
        from patchify import unpatchify
        """Unpatches the prediction mask into the original padded image size."""
        
        # Use stored padded image if no image is passed
        image_to_unpatch_to = self.image_to_patch if img is None else img
        
        # Check for required state
        if self.patch_grid_shape is None or image_to_unpatch_to is None:
             raise ValueError("Must run patch_image() first or provide img and ensure patches were created.")

        preds = preds.squeeze(axis=-1)

        # Reshape predictions from (N_total_patches, P, P, C_out) 
        # to (N_rows, N_cols, N_depth, P, P, C_out)
        if preds.ndim < 4:
            # target_shape is (N_rows, N_cols) + (P, P)
            target_shape = self.patch_grid_shape + preds.shape[-2:]
            preds_reshaped = preds.reshape(target_shape)
        else:
            preds_reshaped = preds
            
        # Unpatch and return the mask
        return unpatchify(preds_reshaped, image_to_unpatch_to.shape[:2])






