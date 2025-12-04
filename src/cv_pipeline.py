import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pathlib import Path
from typing import Union, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

from src.inference import InferenceManager
from src.core.segmenation import PlantSegmentor
from src.core.root_length import analyze_plant_lengths

from src.configs.inference import MODEL_PATH, SECONDARY_MODEL_PATH, ENSEMBLE_STRATEGY
from src.configs.segmentation import SEGMENTATION_CONFIG, PLANT_ZONES
from src.config import IMG_DIR
from src.inference.custom_functions import f1, iou, dice_bce_loss
from src.helpers import extract_sort_keys

from src.models import InferenceError, InitializationError, PlantResult
from src.models.inference_models import MaskCombineStrategy, ModelConfig




class CVPipeline:
    DEFAULT_PATCH_SIZE = 512
    SECONDARY_PATCH_SIZE = 256
    BINARY_THRESHOLD = 0.5

    def __init__(
        self,
        use_ensemble: bool = True,
        combine_strategy: MaskCombineStrategy = ENSEMBLE_STRATEGY
    ):
        """Initialize the pipeline with model and segmentation components.
        
        Args:
            use_ensemble: If True, use both primary and secondary models
                for inference. If False, use only the primary model.
            combine_strategy: Strategy for combining masks when using
                ensemble inference.
        
        Raises:
            InitializationError: If the inference manager or segmentor 
                fails to initialize.
        """
        self._img_files: List[str] = []
        self.results: List[PlantResult] = []
        self.use_ensemble = use_ensemble
        self.combine_strategy = combine_strategy
        
        custom_objects = {"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}
        
        # Initialize primary model
        primary_config = ModelConfig(
            checkpoint=MODEL_PATH,
            patch_size=self.DEFAULT_PATCH_SIZE,
            name="primary"
        )
        self.primary_manager = self._create_inference_manager(primary_config, custom_objects)
        
        # Initialize secondary model for ensemble
        self.secondary_manager: Optional[InferenceManager] = None
        if use_ensemble:
            secondary_config = ModelConfig(
                checkpoint=SECONDARY_MODEL_PATH,
                patch_size=self.SECONDARY_PATCH_SIZE,
                name="secondary"
            )
            self.secondary_manager = self._create_inference_manager(secondary_config, custom_objects)
        
        self._init_plant_segmentor()

    def _create_inference_manager(
        self, 
        config: ModelConfig, 
        custom_objects: dict
    ) -> InferenceManager:
        """Create an inference manager for a given model configuration.
        
        Args:
            config: Model configuration specifying checkpoint and settings.
            custom_objects: Dictionary of custom Keras objects required 
                for model loading.
        
        Returns:
            Initialized InferenceManager instance.
        
        Raises:
            InitializationError: If the InferenceManager fails to initialize.
        """
        try:
            manager = InferenceManager(str(config.checkpoint), custom_objects)
            logger.info(
                f"InferenceManager '{config.name}' initialized "
                f"(checkpoint: {config.checkpoint}, patch_size: {config.patch_size})"
            )
            return manager
        except Exception as e:
            logger.exception(f"Failed to initialize InferenceManager '{config.name}'")
            raise InitializationError(
                f"InferenceManager '{config.name}' initialization failed: {e}"
            ) from e

    def _init_plant_segmentor(self) -> None:
        """Initialize the plant segmentation component.
        
        Raises:
            InitializationError: If the PlantSegmentor fails to initialize.
        """
        try:
            self.segmentor = PlantSegmentor(PLANT_ZONES, SEGMENTATION_CONFIG)
            logger.info("PlantSegmentor initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize PlantSegmentor")
            raise InitializationError(f"PlantSegmentor initialization failed: {e}") from e

    def _load_images(self, input_dir: Union[Path, str]):
        """Load the list of PNG image files from the specified directory.

        Args:
            input_dir: Path to the directory containing image files.
        """
        files = sorted(os.listdir(input_dir))
        self._img_files = [f for f in files if f.lower().endswith(".png")]
        logger.debug(f"Found {len(self._img_files)} image files.")

    def _run_single_inference(
        self,
        manager: InferenceManager,
        img: np.ndarray,
        patch_size: int,
        model_name: str,
        img_filename: str
    ) -> np.ndarray:
        """Run inference using a single model.
        
        Args:
            manager: The InferenceManager to use.
            img: Input image as a NumPy array (BGR format).
            patch_size: Size of patches to extract.
            model_name: Name of the model for logging.
            img_filename: Filename for logging purposes.
        
        Returns:
            Binary mask as uint8 array with values 0 or 255.
        
        Raises:
            InferenceError: If any step of inference fails.
        """
        # Patch extraction
        try:
            patches = manager.patch_image(img, patch_size=patch_size, normalize=True)
            logger.debug(
                f"  [{model_name}] Created {patches.shape[0]} patches "
                f"of shape {patches.shape[1:]}"
            )
        except Exception as e:
            raise InferenceError(
                f"[{model_name}] Failed to patch image {img_filename}: {e}"
            ) from e

        # Model prediction
        try:
            padded_mask = manager.create_root_mask(patches)
        except Exception as e:
            raise InferenceError(
                f"[{model_name}] Prediction failed for {img_filename}: {e}"
            ) from e

        if padded_mask is None:
            raise InferenceError(f"[{model_name}] Model returned None for {img_filename}")

        # Mask reconstruction
        try:
            corrected_mask = manager.restore_mask(padded_mask, patch_size=patch_size)
        except Exception as e:
            raise InferenceError(
                f"[{model_name}] Mask reconstruction failed for {img_filename}: {e}"
            ) from e

        binary_mask = (corrected_mask > self.BINARY_THRESHOLD).astype(np.uint8) * 255
        
        logger.debug(
            f"  [{model_name}] Mask shape: {binary_mask.shape}, "
            f"positive pixels: {np.sum(binary_mask > 0)}"
        )
        
        return binary_mask

    def _combine_masks(
        self,
        mask_primary: np.ndarray,
        mask_secondary: np.ndarray
    ) -> np.ndarray:
        """Combine two binary masks using the configured strategy.
        
        Args:
            mask_primary: Binary mask from primary model.
            mask_secondary: Binary mask from secondary model.
        
        Returns:
            Combined binary mask.
        """
        if self.combine_strategy == MaskCombineStrategy.UNION:
            combined = cv2.bitwise_or(mask_primary, mask_secondary)
        else:  # INTERSECTION
            combined = cv2.bitwise_and(mask_primary, mask_secondary)
        
        logger.debug(
            f"  [combined] Strategy: {self.combine_strategy.value}, "
            f"positive pixels: {np.sum(combined > 0)}"
        )
        
        return combined

    def _run_inference(self, img: np.ndarray, img_filename: str) -> np.ndarray:
        """Run model inference on a single image to generate a binary mask.
        
        Uses ensemble inference if enabled, combining predictions from
        both primary and secondary models.
        
        Args:
            img: Input image as a NumPy array (BGR format from cv2).
            img_filename: Filename for logging purposes.
        
        Returns:
            Binary mask as uint8 array with values 0 or 255.
        
        Raises:
            InferenceError: If inference fails for any model.
        """
        # Primary model inference
        mask_primary = self._run_single_inference(
            manager=self.primary_manager,
            img=img,
            patch_size=self.DEFAULT_PATCH_SIZE,
            model_name="primary",
            img_filename=img_filename
        )
        
        # Single model mode
        if not self.use_ensemble or self.secondary_manager is None:
            return mask_primary
        
        # Secondary model inference
        mask_secondary = self._run_single_inference(
            manager=self.secondary_manager,
            img=img,
            patch_size=self.SECONDARY_PATCH_SIZE,
            model_name="secondary",
            img_filename=img_filename
        )
        
        # Combine masks
        return self._combine_masks(mask_primary, mask_secondary)

    def _segment_components(self, mask: np.ndarray) -> np.ndarray:
        """Segment and filter connected components from binary mask.
        
        Args:
            mask: Binary mask (uint8) with plant regions as white pixels.
        
        Returns:
            Label array where each unique positive integer represents 
            a distinct plant region.
        """
        _, labels = cv2.connectedComponents(mask.astype(np.uint8))
        labels = self.segmentor.filter_plant_components(labels)
        merged_labels = self.segmentor.merge_components_by_distance(
            labels, distance_thresh=200
        )
        return merged_labels
    
    def _analyze_lengths(self, labels: np.ndarray, img_name: str) -> None:
        """Analyze labeled regions and record plant measurements.
        
        Args:
            labels: Label array from component segmentation.
            img_name: Base name of the source image for result tracking.
        """
        results = analyze_plant_lengths(labels)
        
        for plant_id, data in results.items():
            detected = data['detected']
            length = data['length']
            
            if detected:
                logger.debug(f"  Plant {plant_id}: detected, length={length}px")
            else:
                logger.debug(f"  Plant {plant_id}: not detected")

            self.results.append(PlantResult(
                image_name=img_name,
                plant_id=plant_id,
                length=length
            ))
    
    def _save_results_csv(self, output_dir: Path) -> Path:
        """Export results to a sorted CSV file.
        
        Args:
            output_dir: Directory where the CSV file will be saved.
        
        Returns:
            Path to the created CSV file.
        """
        csv_rows = [
            {'Plant ID': r.plant_label, 'Length (px)': r.length}
            for r in self.results
        ]
        
        df = pd.DataFrame(csv_rows)
        df['sort_key'] = df['Plant ID'].apply(extract_sort_keys)
        df = df.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)
        
        csv_path = output_dir / "plant_lengths.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        return csv_path

    def process_directory(self, input_dir: Union[Path, str]) -> Path:
        """Process all images in a directory through the full pipeline.
        
        Args:
            input_dir: Path to folder containing raw plant images.
        
        Returns:
            Path to the generated CSV results file.
        
        Raises:
            ValueError: If the input directory does not exist or contains no images.
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if not self._img_files:
            self._load_images(input_path)
        
        if not self._img_files:
            raise ValueError(f"No PNG images found in {input_dir}")

        mode = "ensemble" if self.use_ensemble else "single-model"
        logger.info(
            f"Starting pipeline ({mode}) for {len(self._img_files)} images"
        )

        for img_filename in tqdm(self._img_files, desc="Processing images", unit="img"):
            img_path = input_path / img_filename
            img_name = os.path.splitext(img_filename)[0]
            
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            logger.debug(f"Processing: {img_name}")

            try:
                mask = self._run_inference(img, img_filename)
            except InferenceError as e:
                logger.error(str(e))
                continue

            if img.shape[:2] != mask.shape[:2]:
                logger.error(
                    f"Shape mismatch for {img_filename}: "
                    f"image={img.shape[:2]}, mask={mask.shape[:2]}"
                )
                continue

            labels = self._segment_components(mask)
            self._analyze_lengths(labels, img_name)

        output_dir = input_path.parent
        return self._save_results_csv(output_dir)


def cv_pipeline(
    input_dir: Union[Path, str],
    use_ensemble: bool = True,
    combine_strategy: MaskCombineStrategy = MaskCombineStrategy.UNION
) -> Path:
    """Convenience function to run the full CV pipeline.
    
    Args:
        input_dir: Path to folder containing raw plant images.
        use_ensemble: Whether to use dual-model ensemble inference.
        combine_strategy: How to combine masks when using ensemble.
    
    Returns:
        Path to the generated CSV results file.
    """
    pipeline = CVPipeline(
        use_ensemble=use_ensemble,
        combine_strategy=combine_strategy
    )
    return pipeline.process_directory(input_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    cv_pipeline(IMG_DIR)