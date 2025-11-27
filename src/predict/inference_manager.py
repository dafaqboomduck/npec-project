import os
import re
import shutil
import glob
import json
import cv2
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
from keras.models import load_model
from typing import List, Dict, Optional
from patchify import unpatchify

from .custom_functions import f1

MODEL_PATH = "../artifacts/models/unet_model_256px_v1.keras"
INPUT_DIR = "./test_images/test"


class InferenceManager:
    """
    Manages image patching and unpatching for model inference.
    Uses saved metadata to correctly reconstruct images without distortion.
    """

    def __init__(self, model_path: str, patch_size: int = 256):
        self.patch_size = patch_size
        self.model_path = model_path
        self.metadata: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)

        self.logger.debug(f"Loading model from {self.model_path}...")
        if not os.path.exists(model_path):
            msg = f"Model file not found at {model_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.model = load_model(self.model_path, custom_objects={'f1': f1})
        self.logger.debug("Model loaded successfully.")

    def load_metadata(self, metadata_path: Path) -> None:
        """Load the patch metadata JSON file created during preprocessing."""
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.logger.info(f"Loaded metadata for {len(self.metadata)} images")

    def _get_sorted_patches(self, patches: List[str]) -> List[str]:
        def natural_keys(text):
            match = re.search(r'patch_(\d+)', text)
            return int(match.group(1)) if match else 0
        return sorted(patches, key=natural_keys)

    def unpatch_pred_mask(
        self, 
        preds: np.ndarray, 
        grid_shape: tuple, 
        padded_shape: tuple,
        cropped_shape: tuple
    ) -> np.ndarray:
        """
        Unpatches the prediction mask and crops to original size.
        
        Args:
            preds: Predicted patch masks (N_total, P, P, C).
            grid_shape: (rows, cols) - how patches are arranged.
            padded_shape: (H, W) of the padded image.
            cropped_shape: (H, W) of the original cropped image (before padding).
        
        Returns:
            The reconstructed mask cropped to the original image size.
        """
        # Squeeze channel dim if present
        if preds.shape[-1] == 1:
            preds = preds.squeeze(axis=-1)

        rows, cols = grid_shape
        expected_patches = rows * cols

        if len(preds) != expected_patches:
            raise ValueError(
                f"Grid shape {grid_shape} expects {expected_patches} patches, "
                f"but got {len(preds)} predictions."
            )

        # Reshape to grid: (rows, cols, patch_h, patch_w)
        target_shape = (rows, cols) + preds.shape[-2:]
        preds_reshaped = preds.reshape(target_shape)

        # Unpatch to full padded size
        full_mask = unpatchify(preds_reshaped, padded_shape)

        # Crop to original (pre-padding) size to remove padding artifacts
        crop_h, crop_w = cropped_shape
        cropped_mask = full_mask[:crop_h, :crop_w]

        return cropped_mask

    def process_directory(
        self, 
        input_dir: str, 
        metadata_path: Optional[str] = None,
        cleanup: bool = False
    ):
        """
        Iterates through patches, groups them, predicts, and saves merged masks.
        
        Args:
            input_dir: Path to folder containing the patches (e.g., 'test_images/test/').
            metadata_path: Path to patch_metadata.json. If None, looks in parent of input_dir.
            cleanup: If True, deletes the input directory after processing.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist.")

        # Load metadata
        if metadata_path is None:
            # Assume metadata is in the root patch directory (parent of parent)
            meta_file = input_path.parent.parent / "patch_metadata.json"
        else:
            meta_file = Path(metadata_path)
        
        self.load_metadata(meta_file)

        # Setup output directory
        output_dir = input_path.parent / "predicted_masks"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Processing images. Output: {output_dir}")

        # Group files by base image name
        image_groups: Dict[str, List[str]] = {}
        all_files = glob.glob(str(input_path / "*_patch_*.png"))
        self.logger.debug(f"Found {len(all_files)} patch files.")

        for file_path in all_files:
            filename = os.path.basename(file_path)
            match = re.match(r'(.+?)_patch_\d+\.png$', filename)
            if match:
                base_name = match.group(1)
                if base_name not in image_groups:
                    image_groups[base_name] = []
                image_groups[base_name].append(file_path)

        # Process each group
        for base_name, patch_files in image_groups.items():
            if base_name not in self.metadata:
                self.logger.warning(f"No metadata for '{base_name}'. Skipping.")
                continue

            img_meta = self.metadata[base_name]
            grid_shape = tuple(img_meta["grid_shape"])
            padded_shape = tuple(img_meta["padded_shape"])
            cropped_shape = tuple(img_meta["cropped_shape"])

            self.logger.debug(
                f"Processing {base_name}: grid={grid_shape}, "
                f"padded={padded_shape}, cropped={cropped_shape}"
            )

            sorted_files = self._get_sorted_patches(patch_files)

            # Validate patch count
            if len(sorted_files) != img_meta["num_patches"]:
                self.logger.warning(
                    f"Expected {img_meta['num_patches']} patches for {base_name}, "
                    f"found {len(sorted_files)}. Skipping."
                )
                continue

            # Load and preprocess batch
            batch_images = []
            for p_file in sorted_files:
                img = cv2.imread(p_file)
                if img is None:
                    self.logger.warning(f"Could not read: {p_file}")
                    continue
                img = cv2.resize(img, (self.patch_size, self.patch_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_images.append(img)

            batch_np = np.array(batch_images)

            # Predict
            predictions = self.model.predict(batch_np, verbose=0)

            # Reconstruct mask using saved metadata
            full_mask = self.unpatch_pred_mask(
                predictions,
                grid_shape=grid_shape,
                padded_shape=padded_shape,
                cropped_shape=cropped_shape
            )

            # Threshold to binary
            binary_mask = (full_mask > 0.5).astype(np.uint8) * 255

            # Save
            save_path = output_dir / f"{base_name}_mask.tif"
            cv2.imwrite(str(save_path), binary_mask)
            self.logger.debug(f"Saved mask: {save_path}")

        # Cleanup
        if cleanup:
            self.logger.info(f"Cleaning up: {input_dir}")
            try:
                shutil.rmtree(input_dir)
            except OSError as e:
                self.logger.error(f"Error deleting directory: {e}")

        self.logger.info("Inference process completed.")