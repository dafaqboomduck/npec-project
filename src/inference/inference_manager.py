import os
import cv2
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Union
from patchify import unpatchify, patchify
from keras.models import load_model # Import necessary Keras function

import logging
logger = logging.getLogger(__name__)

# Assuming these are defined elsewhere and accessible
from .custom_functions import f1, iou, dice_bce_loss
from src.helpers.padder import padder
from src.helpers.extract_petri_dish import extract_petri_dish

class InferenceManager:
    """
    Manages image patching, model inference, and mask unpatching/restoration 
    for image segmentation, specifically handling petridish extraction and padding.
    """    
    def __init__(self, checkpoint_path: Union[Path, str], custom_objects: dict):
        """
        Initializes the InferenceManager and loads the Keras model once.

        Args:
            checkpoint_path (Path | str): Path to the saved Keras model checkpoint (.keras or .h5).
            custom_objects (dict): Dictionary of custom functions required for model loading (e.g., custom losses, metrics).
        """
        # Initialize state variables
        self.img_shape = None
        self.crop_coords = None
        self.patch_grid_shape = None # Stores (N_rows, N_cols)
        self.model = None
        
        
        # Load model upon initialization for efficiency
        self._init_model(Path(checkpoint_path), custom_objects)
    
    def _init_model(self, checkpoint_path: Path, custom_objects: dict):
        """
        Loads the Keras model from the specified checkpoint path.
        
        Args:
            checkpoint_path (Path): The path to the saved model checkpoint.
            custom_objects (dict): Custom functions/objects needed for loading the model.
        
        Raises:
            FileExistsError: If the checkpoint path is invalid.
            ImportError: If the model cannot be loaded (e.g., file corruption, wrong format).
        """

        if not checkpoint_path.exists():
            logger.error(f"Could not find the path to the saved model checkpoint: {checkpoint_path}. Please make sure the path is correct.")
            raise FileExistsError
        else:
            try:
                self.model = load_model(checkpoint_path, custom_objects=custom_objects)
                logger.debug(f"Model loaded successfully from {checkpoint_path}.")
            except Exception as e:
                logger.exception(f"Model could not be loaded from model checkpoint: {checkpoint_path}. Please make sure this is a .keras or a .h5 file containing a saved Keras model.")
                raise ImportError

    def patch_image(self, img: np.ndarray, patch_size: int = 256, scaling_factor: float = 1.0, normalize: bool = False) -> np.ndarray:
        """
        Scales, pads, extracts, and patches the image for inference.

        Args:
            img (np.ndarray): The raw input image (H, W, C or H, W).
            patch_size (int): The square size of the patches (P x P). Defaults to 256.
            scaling_factor (float): Factor by which to scale the image before processing. Defaults to 1.0.
            normalize (bool): If True, patches are normalized to [0, 1]. Defaults to False.

        Returns:
            np.ndarray: The resulting array of patches ready for model input 
                        (N_total_patches, P, P, 3).
        """
        self.img_shape = img.shape
        
        if scaling_factor != 1.0:
            # Scale the image
            scaled_im = cv2.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
        else:
            scaled_im = img

        # Extract the petri dish area using the potentially scaled image
        self.crop_coords = extract_petri_dish(scaled_im, return_coords=True)
        x1_scaled, y1_scaled, x2_scaled, y2_scaled = self.crop_coords

        if scaling_factor != 1.0:
            # CRITICAL FIX: Rescale coordinates back to the original image's space
            # before cropping the original image.
            scale_ratio = 1.0 / scaling_factor
            x1 = int(x1_scaled * scale_ratio)
            y1 = int(y1_scaled * scale_ratio)
            x2 = int(x2_scaled * scale_ratio)
            y2 = int(y2_scaled * scale_ratio)
            self.crop_coords = (x1, y1, x2, y2)
        else:
            x1, y1, x2, y2 = self.crop_coords


        # Crop the image using the original coordinates
        dish = img[y1:y2, x1:x2]

        # Pad the image
        padded_im = padder(dish, patch_size)

        # Convert grayscale to 3-channel if necessary (assuming padder maintains channels)
        if padded_im.ndim < 3:
            padded_im = cv2.cvtColor(padded_im, cv2.COLOR_GRAY2BGR)

        self.image_to_patch = padded_im

        # Patch the image (Assuming 3-channel image)
        patches = patchify(padded_im, (patch_size, patch_size, 3), step=patch_size)
        
        # Store the shape of the patch grid for unpatching
        self.patch_grid_shape = patches.shape[:2]

        # Flatten the patches for model input
        num_patches = patches.shape[0] * patches.shape[1]
        # Reshape to (N_total_patches, patch_size, patch_size, 3)
        patches = patches.reshape(num_patches, patch_size, patch_size, 3)
        
        if normalize:
            # Normalization
            patches = patches / 255.0 
    
        return patches

    def _unpatch_pred_mask(self, preds: np.ndarray) -> np.ndarray:
        """
        Unpatches the prediction mask into the original padded image size.

        Args:
            preds (np.ndarray): The model predictions, typically (N_total_patches, P, P, C_out).

        Returns:
            np.ndarray: The stitched mask, sized to the padded crop (H_padded, W_padded).
            
        Raises:
            ValueError: If patching state variables are not initialized.
        """
        
        # Check for required state
        if self.patch_grid_shape is None or self.img_shape is None:
             raise ValueError("Must run patch_image() first or provide img and ensure patches were created.")

        # Squeeze the channel dimension if it's 1
        preds = preds.squeeze(axis=-1)

        N_rows, N_cols = self.patch_grid_shape
        P = preds.shape[-1] # Patch width/height (assuming square)

        # Reshape predictions from (N_total_patches, P, P) or (N_total_patches, P, P, C_out)
        # to (N_rows, N_cols, P, P)
        target_shape = (N_rows, N_cols, P, P)
        
        # The number of patches must match N_rows * N_cols
        if preds.shape[0] != N_rows * N_cols:
             raise ValueError(f"Mismatched patch count. Expected {N_rows * N_cols}, got {preds.shape[0]}.")

        preds_reshaped = preds.reshape(target_shape)

        # The target shape for unpatchify is the shape of the full padded image
        H_padded = N_rows * P
        W_padded = N_cols * P
            
        # Unpatch and return the mask
        return unpatchify(preds_reshaped, (H_padded, W_padded))
    
    def create_root_mask(self, patches: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs model prediction on the patches and unpatches the results.

        Args:
            patches (np.ndarray): The patches ready for model input.

        Returns:
            Optional[np.ndarray]: The stitched, padded mask (H_padded, W_padded), 
                                  or None if prediction fails.
        """
        if self.model is None:
            logger.error("Model is not initialized.")
            return None
            
        try:
            # verbose=0 suppresses progress bar during batch processing
            preds = self.model.predict(patches, verbose=0) 
            padded_mask = self._unpatch_pred_mask(preds)

            return padded_mask
        
        except Exception as e:
            logger.exception(f"UNEXPECTED EXCEPTION DURING PREDICTION/UNPATCHING: {e}")
            return None
    
    def restore_mask(self, padded_mask: np.ndarray, patch_size: int = 256) -> np.ndarray:
        """
        Reverses the padding and cropping steps to restore the prediction 
        to the original image's shape and coordinate system.

        Args:
            padded_mask (np.ndarray): The stitched prediction (padded size).
            patch_size (int): The patch size used in padder. Defaults to 256.

        Returns:
            np.ndarray: The mask restored to the original image dimensions (H_original, W_original).
        
        Raises:
            ValueError: If crop_coords or img_shape are not initialized.
        """
        if self.crop_coords is None or self.img_shape is None:
             raise ValueError("restore_mask requires self.crop_coords and self.img_shape to be set by patch_image().")
             
        # 1. Unpack coordinates from extract_petri_dish (in original image coordinates)
        x_start, y_start, x_end, y_end = self.crop_coords
        
        # Calculate the dimensions of the crop BEFORE padding
        h_crop = y_end - y_start
        w_crop = x_end - x_start

        # 2. Re-calculate the padding that was added 
        # (This logic MUST mirror the 'padder' utility function)
        height_padding = ((h_crop // patch_size) + 1) * patch_size - h_crop
        width_padding  = ((w_crop // patch_size) + 1) * patch_size - w_crop

        # Calculate exactly how much was added to top and left
        top_pad = height_padding // 2
        left_pad = width_padding // 2

        # 3. Remove the padding (Un-pad)
        # Extract the region starting at top/left pad, with the size of the original crop.
        unpadded_mask = padded_mask[top_pad : top_pad + h_crop, left_pad : left_pad + w_crop]

        # 4. Restore to Original Canvas (Un-crop)
        # Create a blank array with the original raw dimensions
        final_mask = np.zeros(self.img_shape[:2], dtype=padded_mask.dtype)

        # Place the unpadded mask into the specific coordinates where the dish was found
        final_mask[y_start : y_end, x_start : x_end] = unpadded_mask
        
        return final_mask