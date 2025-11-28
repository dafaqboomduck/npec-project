import os
import shutil
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict
from patchify import patchify
from tqdm.auto import tqdm

from src.helpers.padder import padder
from src.helpers.extract_petri_dish import extract_petri_dish


class ImagePreprocessor:
    """
    Manages the preprocessing pipeline: loading, cropping, padding, patchifying,
    and cleaning up temporary patch directories.
    
    Now saves metadata about image dimensions for accurate reconstruction.
    """

    def __init__(self):
        self.img_files: List[Path] = []
        self.patch_dir: Optional[Path] = None
        self.logger = logging.getLogger(__name__)

    def load_images(self, img_dir: Path) -> None:
        if not img_dir.exists():
            self.logger.error(f"Image directory not found: {img_dir}")
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        files = sorted([f for f in img_dir.iterdir() if f.is_file()])
        self.img_files = [f for f in files if f.suffix.lower() == ".png"]
        self.logger.info(f"Loaded {len(self.img_files)} images from {img_dir}")

    def create_and_save_patches(
        self,
        img_dir: str | Path,
        patch_dir: Optional[str | Path] = None,
        patch_size: int = 256,
        scaling_factor: float = 1.0,
        subset_name: str = "test"
    ) -> Path:
        """
        Generates patches from images and saves them to disk.
        Also saves a metadata JSON file with dimensions for each image.
        """
        img_dir = Path(img_dir)

        if patch_dir is None:
            dataset_root_name = img_dir.parent.parent.name
            self.patch_dir = img_dir.parent.parent.parent / f"{dataset_root_name}_patches_{patch_size}px"
        else:
            self.patch_dir = Path(patch_dir)

        self.logger.info(f"Destination Patch Dir: {self.patch_dir}")
        self.patch_dir.mkdir(parents=True, exist_ok=True)

        self.load_images(img_dir)

        if not self.img_files:
            self.logger.warning(f"No .png images found in {img_dir}")
            return self.patch_dir

        save_img_dir = self.patch_dir / f"{subset_name}_images" / subset_name
        save_img_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store metadata for all images
        metadata: Dict[str, Dict] = {}

        for img_path in tqdm(self.img_files, desc="Generating Patches", unit="img"):
            im = cv2.imread(str(img_path))
            if im is None:
                self.logger.warning(f"Failed to load image: {img_path}. Skipping.")
                continue

            # 1. Scale
            if scaling_factor != 1.0:
                im = cv2.resize(im, (0, 0), fx=scaling_factor, fy=scaling_factor)

            # 2. Extract Petri Dish
            cropped_im = extract_petri_dish(im, return_coords=False)
            if cropped_im is None or cropped_im.size == 0:
                self.logger.warning(f"Extraction failed/empty for {img_path.name}")
                continue

            # Store cropped dimensions BEFORE padding
            cropped_h, cropped_w = cropped_im.shape[:2]

            # 3. Pad
            padded_im = padder(cropped_im, patch_size)
            padded_h, padded_w = padded_im.shape[:2]

            # 4. Patchify
            patches_im = patchify(padded_im, (patch_size, patch_size, 3), step=patch_size)
            
            # Calculate grid shape BEFORE reshaping
            grid_rows, grid_cols = patches_im.shape[0], patches_im.shape[1]
            
            # Flatten to (N, patch_h, patch_w, Channels)
            patches_im = patches_im.reshape(-1, patch_size, patch_size, 3)

            image_base_name = img_path.stem

            # Save metadata for this image
            metadata[image_base_name] = {
                "cropped_shape": [cropped_h, cropped_w],
                "padded_shape": [padded_h, padded_w],
                "grid_shape": [grid_rows, grid_cols],
                "num_patches": len(patches_im),
                "patch_size": patch_size
            }

            # 5. Save patches (row-major order from patchify)
            for j, patch in enumerate(patches_im):
                # Calculate row and column index for clarity in debugging
                row_idx = j // grid_cols
                col_idx = j % grid_cols
                save_name = f"{image_base_name}_r{row_idx:03d}_c{col_idx:03d}.png"
                cv2.imwrite(str(save_img_dir / save_name), patch)

            self.logger.debug(f"Saved {len(patches_im)} patches for {img_path.name}")

        # Save metadata JSON file
        metadata_path = self.patch_dir / "patch_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata to {metadata_path}")

        self.logger.info("Patch Generation Complete")
        return self.patch_dir

    def delete_patch_directory(self, target_dir: Optional[str | Path] = None) -> None:
        dir_to_delete = Path(target_dir) if target_dir else self.patch_dir
        if dir_to_delete and dir_to_delete.exists() and dir_to_delete.is_dir():
            self.logger.info(f"Cleaning up: Removing directory {dir_to_delete}")
            shutil.rmtree(dir_to_delete)
            if dir_to_delete == self.patch_dir:
                self.patch_dir = None
        else:
            self.logger.info(f"Cleanup skipped: Directory {dir_to_delete} not found or not set.")