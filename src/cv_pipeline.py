import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pathlib import Path
from typing import Union, List

import logging
logger = logging.getLogger(__name__)

from src.inference import InferenceManager
from src.core.processing import PlantSegmentor
from core.root_length import analyze_plant_lengths

from src.config.inference_config import MODEL_PATH
from src.config.global_config import IMG_DIR
from src.inference.custom_functions import f1, iou, dice_bce_loss
from src.helpers import extract_sort_keys

class CVPipeline:

    def __init__(self):
        self.img_files: List[str] = []
        self.all_results: List[dict] = []
        custom_objects={"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}    
        self._init_inference_manager(str(MODEL_PATH), custom_objects)
        self._init_plant_segmentor()

    def _init_inference_manager(self, checkpoint, custom_objects):
        try:   
            # Initialize the InferenceManager
            self.manager = InferenceManager(checkpoint, custom_objects)
        except Exception as e:
            logger.exception(f"Unexpected Exception encountered while initializing the InferenceManager Class: {e}.")

    def _init_plant_segmentor(self):
        try:
            self.segmentor = PlantSegmentor()
        except Exception as e:
            logger.exception(f"Unexpected Exception encountered while initializing the PlantSegmentor Class: {e}.")

    def _load_images(self, input_dir: Union[Path, str]):
        """
        Loads the list of image files (ending with .png) from the specified directory.

        Args:
            input_dir (Path | str): The path to the directory containing the image files.
        """
        files = sorted(os.listdir(input_dir))
        self.img_files = [f for f in files if f.lower().endswith(".png")]
        logger.debug(f"Found {len(self.img_files)} image files.")

    def _run_inference(self, img: np.ndarray, img_filename: str) -> np.ndarray | None:
        try:
            # 1. Patch the image
            patches = self.manager.patch_image(img, normalize=True)
            # 2. Predict the mask (model is already loaded)
            padded_mask = self.manager.create_root_mask(patches)
            
            if padded_mask is None:
                logger.error(f"Prediction failed for {img_filename}. Skipping.")
                return None
            
            # 3. Restore mask to original image size
            corrected_mask = self.manager.restore_mask(padded_mask)
            # 4. Convert to binary and save
            binary_mask = (corrected_mask > 0.5).astype(np.uint8) * 255
            return binary_mask
        except Exception as e:
            logger.exception(f" UNHANDLED EXCEPTION during inference: {e}")

    def _segment_components(self, mask: np.ndarray):
        # Connected components
        _, labels = cv2.connectedComponents(mask.astype(np.uint8))

        # Filter components
        labels = self.segmentor.filter_plant_components(labels)

        # Merge and assign zone-based labels
        merged_labels = self.segmentor.merge_components_by_distance(labels, distance_thresh=200)

        return merged_labels

    def process_directory(self, input_dir: Union[Path, str]):
        """
        Iterates through images in the input directory, performs patching, 
        prediction, and saves the merged masks.
        
        Args:
            input_dir (Path | str): Path to folder containing the raw images.
        
        Raises:
            ValueError: If the input directory does not exist or if images haven't been loaded.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory {input_dir} does not exist.")
            raise ValueError

        # Ensure images are loaded
        if not self.img_files:
            self._load_images(input_path)
            if not self.img_files:
                logger.warning(f"No PNG images found in {input_dir}. Exiting.")
                return
        
        for img_filename in tqdm(self.img_files, desc="Processing images", unit="img"):
            img_file_path = input_path / img_filename
            
            # Read image
            img = cv2.imread(str(img_file_path))
            if img is None:
                logger.warning(f"Could not read: {img_file_path}. Skipping.")
                continue
            
            # Get the base image name 
            img_name = os.path.splitext(img_filename)[0]

            logger.debug(f"Processing {img_name}.")

            mask = self._run_inference(img, img_filename)

            if mask is None:
                continue

            if img.shape[:2] != mask.shape[:2]:
                logger.error(f"Mask shape differs from image shape. Skipping.")
                continue

            labels = self._segment_components(mask)

            results = analyze_plant_lengths(labels)
            for plant_id, data in results.items():
                status = f"DETECTED (length={data['length']}px)" if data['detected'] else "MISSING (length=0)"
                logger.debug(f"  Plant {plant_id}: {status}")
                
                # Add to results list for CSV
                self.all_results.append({
                    'image_name': img_name,
                    'plant_id': plant_id,
                    'length': data['length']
                })
            
            # Save results to CSV
            # Format: image_name_plant_X, length (plant numbering starts from 1)
            csv_rows = []
            for result in self.all_results:
                plant_id_str = f"{result['image_name']}_plant_{result['plant_id'] + 1}"  # +1 for 1-based numbering
                csv_rows.append({
                    'Plant ID': plant_id_str,
                    'Length (px)': result['length']
                })

            df = pd.DataFrame(csv_rows)

            df['sort_key'] = df['Plant ID'].apply(extract_sort_keys)
            df = df.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)

            output_dir = input_dir.parent

            csv_path = output_dir / "plant_lengths.csv"
            df.to_csv(csv_path, index=False)

def cv_pipeline(input_dir: Path | str):
    pipeline = CVPipeline()
    pipeline.process_directory(input_dir)


if __name__ == "__main__":
    path = IMG_DIR
    cv_pipeline(path)




    
