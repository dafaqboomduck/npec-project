import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import IMG_DIR
from src.configs.inference import ROOT_MODEL_PATH, ANCHOR_MODEL_PATH
from src.configs.segmentation import PLANT_ZONES, SEGMENTATION_CONFIG
from src.core.root_length import analyze_plant_lengths
from src.core.segmenation import PlantSegmentor
from src.helpers import extract_sort_keys
from src.inference import InferenceManager
from src.inference.custom_functions import dice_bce_loss, f1, iou
from src.models import (
    InferenceError,
    InitializationError,
    PlantResult,
    SegmentationConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CVPipeline:
    def __init__(self):
        """Initialize the pipeline with model and segmentation components.

        Raises:
            InitializationError: If the inference managers or segmentor
                fails to initialize.
        """
        self._img_files: List[str] = []
        self.results: List[PlantResult] = []

        custom_objects = {"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}
        self.root_manager = self._init_inference_manager(
            str(ROOT_MODEL_PATH), custom_objects
        )
        self.anchor_manager = self._init_inference_manager(
            str(ANCHOR_MODEL_PATH), custom_objects
        )
        self.segmentor = self._init_plant_segmentor(PLANT_ZONES, SEGMENTATION_CONFIG)

    def _init_inference_manager(self, checkpoint: str, custom_objects: dict) -> None:
        """Initialize the inference manager with model checkpoint.

        Args:
            checkpoint: Path to the model checkpoint file.
            custom_objects: Dictionary of custom Keras objects required
                for model loading (loss functions, metrics, etc.).

        Returns:
            InferenceManager: The initialized inference manager instance.

        Raises:
            InitializationError: If the InferenceManager fails to initialize.
        """
        try:
            manager = InferenceManager(checkpoint, custom_objects)
            logger.info(f"InferenceManager initialized with checkpoint: {checkpoint}")
            return manager
        except Exception as e:
            logger.exception("Failed to initialize InferenceManager")
            raise InitializationError(
                f"InferenceManager initialization failed: {e}"
            ) from e

    def _init_plant_segmentor(
        self,
        zones: Optional[Dict[int, Tuple[int, int]]] = None,
        config: Optional[SegmentationConfig] = None,
    ) -> None:
        """Initialize the plant segmentation component.

        Args:
            zones: Dictionary containing the plant zones
            comfig: SegmentationConfig object containing the config parameters

        Returns:
            PlantSegmentor: The initialized plant segmentor instance.

        Raises:
            InitializationError: If the PlantSegmentor fails to initialize.
        """
        try:
            segmentor = PlantSegmentor(zones, config)
            logger.info("PlantSegmentor initialized successfully")
            return segmentor
        except Exception as e:
            logger.exception("Failed to initialize PlantSegmentor")
            raise InitializationError(
                f"PlantSegmentor initialization failed: {e}"
            ) from e

    def _load_images(self, input_dir: Union[Path, str]):
        """
        Loads the list of image files (ending with .png) from the specified directory.

        Args:
            input_dir (Path | str): The path to the directory containing the image files.
        """
        files = sorted(os.listdir(input_dir))
        self._img_files = [f for f in files if f.lower().endswith(".png")]
        logger.debug(f"Found {len(self._img_files)} image files.")

    def _run_inference(
        self,
        manager: InferenceManager,
        img: np.ndarray,
        img_filename: str,
        threshold: int = 0.5,
    ) -> np.ndarray:
        """Run model inference on a single image to generate a binary mask.

        The inference process involves:
            1. Splitting the image into patches
            2. Running each patch through the segmentation model
            3. Reconstructing the full-size mask from patch predictions
            4. Applying binary thresholding (>0.5)

        Args:
            img: Input image as a NumPy array (BGR format from cv2).
            img_filename: Filename for logging purposes.

        Returns:
            Binary mask as uint8 array with values 0 or 255.

        Raises:
            InferenceError: If patching, prediction, or reconstruction fails.
        """
        try:
            patches = manager.patch_image(img, normalize=True)
        except Exception as e:
            raise InferenceError(f"Failed to patch image {img_filename}: {e}") from e

        try:
            padded_mask = manager.create_img_mask(patches)
        except Exception as e:
            raise InferenceError(
                f"Model prediction failed for {img_filename}: {e}"
            ) from e

        if padded_mask is None:
            raise InferenceError(f"Model returned None for {img_filename}")

        try:
            corrected_mask = manager.restore_mask(padded_mask)
        except Exception as e:
            raise InferenceError(
                f"Mask reconstruction failed for {img_filename}: {e}"
            ) from e

        binary_mask = (corrected_mask > threshold).astype(np.uint8) * 255
        return binary_mask

    def _analyze_lengths(self, labels: np.ndarray, img_name: str) -> None:
        """Analyze labeled regions and record plant measurements.

        Args:
            labels: Label array from component segmentation.
            img_name: Base name of the source image for result tracking.
        """
        results = analyze_plant_lengths(labels)

        for plant_id, data in results.items():
            detected = data["detected"]
            length = data["length"]

            if detected:
                logger.debug(f"  Plant {plant_id}: detected, length={length}px")
            else:
                logger.debug(f"  Plant {plant_id}: not detected")

            self.results.append(
                PlantResult(image_name=img_name, plant_id=plant_id, length=length)
            )

    def _save_results_csv(self, output_dir: Path) -> Path:
        """Export results to a sorted CSV file.

        Args:
            output_dir: Directory where the CSV file will be saved.

        Returns:
            Path to the created CSV file.

        Note:
            Results are sorted using extract_sort_keys for natural ordering
            of image names and plant IDs.
        """
        csv_rows = [
            {"Plant ID": r.plant_label, "Length (px)": r.length} for r in self.results
        ]

        df = pd.DataFrame(csv_rows)
        df["sort_key"] = df["Plant ID"].apply(extract_sort_keys)
        df = (
            df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
        )

        csv_path = output_dir / "plant_lengths.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        return csv_path

    def _validate_mask_dimensions(self, img, mask, img_filename):
        # Validate mask dimensions
        if img.shape[:2] != mask.shape[:2]:
            logger.error(
                f"Shape mismatch for {img_filename}: "
                f"image={img.shape[:2]}, mask={mask.shape[:2]}"
            )
            return 0
        return 1

    def process_directory(self, input_dir: Union[Path, str]) -> Path:
        """Process all images in a directory through the full pipeline.

        Iterates through PNG images in the specified directory, running
        inference, segmentation, and analysis on each. Results are
        accumulated and exported to a CSV file upon completion.

        Args:
            input_dir: Path to folder containing raw plant images.

        Returns:
            Path to the generated CSV results file.

        Raises:
            ValueError: If the input directory does not exist.
            ValueError: If no PNG images are found in the directory.
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if not self._img_files:
            self._load_images(input_path)

        if not self._img_files:
            raise ValueError(f"No PNG images found in {input_dir}")

        logger.info(f"Starting pipeline processing for {len(self._img_files)} images")

        for img_filename in tqdm(self._img_files, desc="Processing images", unit="img"):
            img_path = input_path / img_filename
            img_name = os.path.splitext(img_filename)[0]

            # Step 0: Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            logger.debug(f"Processing: {img_name}")

            # Step 1: Run anchor model
            try:
                anchor_mask = self._run_inference(
                    self.anchor_manager, img, img_filename, threshold=0.01
                )
            except InferenceError as e:
                logger.error(str(e))
                continue

            validate = self._validate_mask_dimensions(img, anchor_mask, img_filename)
            if not bool(validate):
                continue

            # Step 2: Extract and filter anchors
            all_anchors = self.segmentor.get_anchors(anchor_mask)
            anchors = self.segmentor.get_largest_anchor_by_zone(all_anchors)
            anchors = self.segmentor.create_anchor_bboxes(img.shape, anchors)
            logger.debug((f" Anchors: {len(anchors)}"))

            # Step 3: Run root model
            try:
                root_mask = self._run_inference(
                    self.anchor_manager, img, img_filename, threshold=0.001
                )
            except InferenceError as e:
                logger.error(str(e))
                continue

            validate = self._validate_mask_dimensions(img, root_mask, img_filename)
            if not bool(validate):
                continue

            # Step 4: Process roots by anchor regions
            _, final_labels = self.segmentor.process_roots_by_anchor(root_mask, anchors)
            # Step 5: Calculate lengths
            self._analyze_lengths(final_labels, img_name)

        # Save results
        output_dir = input_path.parent
        return self._save_results_csv(output_dir)


def cv_pipeline(input_dir: Union[Path, str]) -> Path:
    """Convenience function to run the full CV pipeline.

    Creates a CVPipeline instance and processes all images in the
    specified directory.

    Args:
        input_dir: Path to folder containing raw plant images.

    Returns:
        Path to the generated CSV results file.

    Raises:
        InitializationError: If pipeline components fail to initialize.
        ValueError: If the directory is invalid or contains no images.
    """
    pipeline = CVPipeline()
    return pipeline.process_directory(input_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cv_pipeline(IMG_DIR)
