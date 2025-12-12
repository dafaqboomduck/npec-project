"""
Robotics Pipeline for Plant Root Analysis and Robotic Manipulation

This module provides a complete pipeline for:
- Detecting plant roots and anchors using deep learning models
- Segmenting plants into zones and extracting root tip coordinates
- Interfacing with an OT-2 robotic environment for automated manipulation
- Converting between pixel and real-world millimeter coordinates

The pipeline integrates computer vision, deep learning inference, and robotics
control to enable automated plant phenotyping and manipulation tasks.
"""

import cv2
import numpy as np
from pathlib import Path
import gymnasium as gym
import logging
from typing import Tuple, List, Dict

# Import pipeline components
from src.inference import InferenceManager
from src.core.segmenation import PlantSegmentor
from src.core.root_length import calculate_geodesic_length
from src.configs.inference import ROOT_MODEL_PATH, ANCHOR_MODEL_PATH
from src.configs.segmentation import PLANT_ZONES, SEGMENTATION_CONFIG
from src.inference.custom_functions import f1, iou, dice_bce_loss
from src.helpers.extract_petri_dish import extract_petri_dish
import src.robotics.ot2_env  # needed to trigger registration
from src.robotics.pid import PIDController
from src.models import (
    InferenceError,
    InitializationError,
    SegmentationConfig,
)
from dataclasses import dataclass

# Robot control constants
MAX_STEPS = 5000  # Maximum steps per episode
DROP_THRESHOLD = 0.0001  # Distance threshold (mm) to trigger drop action
DROP_COOLDOWN = 50  # Steps to wait after dropping before moving to next target

# Custom loss functions and metrics for model loading
custom_objects = {"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}

# Simulation time step (240 Hz)
dt = 1.0 / 240.0

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for the robotics environment.
    
    Attributes:
        num_agents: Number of robotic agents in the environment
        render_mode: Rendering mode ('rgb_array', 'human', or None)
        target: Target coordinates for the robot (if predetermined)
        max_steps: Maximum steps allowed per episode
    """
    num_agents: int = 1
    render_mode: str | None = None
    target: np.ndarray | None = None
    max_steps: int = 1000


@dataclass
class ModelConfig:
    """Configuration for deep learning model inference.
    
    Attributes:
        checkpoint: Path to the model checkpoint file
        threshold: Confidence threshold for binary predictions
        patch_size: Size of image patches for inference (pixels)
        custom_objects: Dictionary of custom loss/metric functions
    """
    checkpoint: str | Path
    threshold: float
    patch_size: int
    custom_objects: dict


# Model configurations
ROOT_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ROOT_MODEL_PATH),
    threshold=0.5,  # 50% confidence threshold
    patch_size=512,
    custom_objects=custom_objects,
)

ANCHOR_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ANCHOR_MODEL_PATH),
    threshold=0.01,  # Lower threshold for anchor detection
    patch_size=512,
    custom_objects=custom_objects,
)

# Pipeline configurations
INFERENCE_CONFIG = (ROOT_MODEL_CONFIG, ANCHOR_MODEL_CONFIG)
SEGMENTOR_CONFIG = (PLANT_ZONES, SEGMENTATION_CONFIG)
ENV_CONFIG = EnvConfig(render_mode="rgb_array")


class RoboticsPipeline:
    """Main pipeline for plant root analysis and robotic manipulation.
    
    This class integrates:
    - Deep learning models for root and anchor detection
    - Plant segmentation for zone-based analysis
    - Gymnasium robotics environment for OT-2 robot control
    - Coordinate conversion between pixel and millimeter spaces
    
    Attributes:
        inference_config: Configuration tuple for root and anchor models
        segmentor_config: Configuration for plant segmentation
        env_config: Configuration for robotics environment
        root_manager: Inference manager for root detection model
        anchor_manager: Inference manager for anchor detection model
        segmentor: Plant segmentation processor
        env: Gymnasium robotics environment
    """
    
    def __init__(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentor_config: Tuple[dict, SegmentationConfig],
        env_config: EnvConfig,
    ):
        """Initialize the robotics pipeline.
        
        Args:
            inference_config: Tuple of (root_config, anchor_config)
            segmentor_config: Tuple of (plant_zones, segmentation_config)
            env_config: Environment configuration
        """
        self.inference_config = inference_config
        self.segmentor_config = segmentor_config
        self.env_config = env_config
        self._init_components(
            self.inference_config, self.segmentor_config, self.env_config
        )

    def _init_components(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentor_config: Tuple[dict, SegmentationConfig],
        env_config: EnvConfig,
    ):
        """Initialize all pipeline components with error handling.
        
        Loads deep learning models, segmentor, and robotics environment.
        Each component is loaded independently with specific error handling.
        
        Args:
            inference_config: Configuration for inference models
            segmentor_config: Configuration for plant segmentation
            env_config: Configuration for robotics environment
            
        Raises:
            InitializationError: If any component fails to initialize
        """
        logger.debug("Initializing pipeline components...")
        try:
            # Load root detection model
            try:
                self.root_manager = InferenceManager(
                    inference_config[0].checkpoint, inference_config[0].custom_objects
                )
                logger.debug("✓ Root model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Root model. Exception: {e}")
                raise InitializationError

            # Load anchor detection model
            try:
                self.anchor_manager = InferenceManager(
                    inference_config[1].checkpoint, inference_config[1].custom_objects
                )
                logger.debug("✓ Anchor model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Anchor model. Exception: {e}")
                raise InitializationError

            # Initialize plant segmentor
            try:
                self.segmentor = PlantSegmentor(
                    segmentor_config[0], segmentor_config[1]
                )
                logger.debug("✓ Segmentor instance loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Segmentor. Exception: {e}")
                raise InitializationError

            # Initialize robotics environment
            try:
                self.env = gym.make("OT2ENV-v0", env_config)
                logger.debug("✓ Environment loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Environment. Exception: {e}")
                raise InitializationError
            
            logger.debug("Pipeline Components initialized successfully!")
        except Exception as e:
            logger.exception(
                f"UNEXPECTED ERROR: Encountered an unexpected error while initializing the pipeline components. Exception: {e}"
            )
            raise InitializationError

    def _get_dish_size_px(self, img: np.ndarray) -> int:
        """Extract petri dish from image and return its size in pixels.
        
        Args:
            img: Input image containing petri dish
            
        Returns:
            Size of petri dish in pixels (height dimension)
        """
        dish = extract_petri_dish(img)
        return dish.shape[0]

    def _convert_mm2px(
        self,
        img: np.ndarray,
        points: List[Tuple[float, float]] | Tuple[float, float],
        size_mm: float,
    ) -> List[Tuple[float, float]]:
        """Convert coordinates from pixels to millimeters.
        
        Uses the petri dish size as a reference to calculate the conversion factor
        between pixel space and real-world millimeter measurements.
        
        Args:
            img: Image containing the petri dish for scale reference
            points: Single point or list of (x, y) coordinates in pixels
            size_mm: Known size of petri dish in millimeters
            
        Returns:
            List of (x, y) coordinates converted to millimeters
        """
        # Calculate mm per pixel conversion factor
        factor = size_mm / self._get_dish_size_px(img)
        
        # Ensure points is a list
        if not isinstance(points, list):
            points = [points]
        
        # Convert each point from pixels to millimeters
        points_mm = []
        for point in points:
            x_px, y_px = point
            x_mm, y_mm = x_px * factor, y_px * factor
            points_mm.append((x_mm, y_mm))
        
        return points_mm

    def _run_inference(
        self,
        manager: InferenceManager,
        img: np.ndarray,
        img_filename: str,
        patch_size: int,
        threshold: float,
    ) -> np.ndarray:
        """Run model inference on an image using patch-based processing.
        
        Splits the image into patches, runs inference on each patch,
        reassembles the predictions, and applies thresholding.
        
        Args:
            manager: InferenceManager instance with loaded model
            img: Input image to process
            img_filename: Filename for logging purposes
            patch_size: Size of patches to split image into
            threshold: Confidence threshold for binary mask
            
        Returns:
            Binary mask (0 or 255) of predictions
            
        Raises:
            InferenceError: If patching or prediction fails
        """
        # Split image into patches with normalization
        try:
            patches = manager.patch_image(img, patch_size=patch_size, normalize=True)
        except Exception as e:
            logger.exception(f"Failed to patch image {img_filename}: {e}")
            raise InferenceError

        # Run model prediction on patches
        try:
            padded_mask = manager.create_img_mask(patches)
        except Exception as e:
            logger.exception(f"Model prediction failed for {img_filename}: {e}")
            raise InferenceError

        # Validate prediction output
        if padded_mask is None:
            logger.exception(f"Model returned None for {img_filename}")
            raise InferenceError

        # Apply threshold to create binary mask
        binary_mask = (padded_mask > threshold).astype(np.uint8) * 255
        return binary_mask

    def _validate_mask_dimensions(self, img, mask, img_filename):
        """Validate that mask dimensions match the input image.
        
        Args:
            img: Original input image
            mask: Generated prediction mask
            img_filename: Filename for logging
            
        Returns:
            1 if dimensions match, 0 otherwise
        """
        if img.shape[:2] != mask.shape[:2]:
            logger.error(
                f"Shape mismatch for {img_filename}: "
                f"image={img.shape[:2]}, mask={mask.shape[:2]}"
            )
            return 0
        return 1

    def _get_root_tip_coordinates(
        self,
        img: np.ndarray,
        img_filename: str,
        root_config: ModelConfig,
        anchor_config: ModelConfig,
        max_gap: int,
    ) -> Dict[List[tuple, tuple]]:
        """Extract root tip coordinates for all detected plants.
        
        This is the main processing pipeline that:
        1. Detects plant anchors (seed locations)
        2. Segments roots using the root detection model
        3. Associates roots with their anchors
        4. Calculates geodesic paths along each root
        5. Extracts start (anchor) and end (tip) coordinates
        
        Args:
            img: Input image containing plants in petri dishes
            img_filename: Filename for logging and debugging
            root_config: Configuration for root detection model
            anchor_config: Configuration for anchor detection model
            max_gap: Maximum gap (pixels) to bridge in geodesic calculation
            
        Returns:
            Dictionary mapping plant_id to [(start_y, start_x), (end_y, end_x)]
            Returns None if processing fails
        """
        # Step 1: Run anchor detection model
        try:
            anchor_mask = self._run_inference(
                self.anchor_manager,
                img=img,
                img_filename=img_filename,
                patch_size=anchor_config.patch_size,
                threshold=anchor_config.threshold,
            )
        except InferenceError as e:
            logger.error(
                f"InferenceError during Anchor Model Inference. Details: {str(e)}"
            )
            return None

        # Validate anchor mask dimensions
        validate = self._validate_mask_dimensions(img, anchor_mask, img_filename)
        if not bool(validate):
            return None

        # Step 2: Extract and filter anchors by zone
        all_anchors = self.segmentor.get_anchors(anchor_mask)
        anchors = self.segmentor.get_largest_anchor_by_zone(all_anchors)
        anchors = self.segmentor.create_anchor_bboxes(img.shape, anchors)
        logger.debug((f" Anchors: {len(anchors)}"))

        # Step 3: Run root detection model
        try:
            root_mask = self._run_inference(
                self.root_manager,
                img=img,
                img_filename=img_filename,
                patch_size=root_config.patch_size,
                threshold=root_config.threshold,
            )
        except InferenceError as e:
            logger.error(
                f"InferenceError during Root Model Inference. Details: {str(e)}"
            )
            return None

        # Validate root mask dimensions
        validate = self._validate_mask_dimensions(img, root_mask, img_filename)
        if not bool(validate):
            return None

        # Step 4: Associate roots with anchors by spatial regions
        _, labels = self.segmentor.process_roots_by_anchor(root_mask, anchors)

        # Step 5: Extract root tip coordinates for each plant
        unique = [l for l in np.unique(labels) if l > 0]
        if not unique:
            logger.error("No plants detected")
            return None

        all_results = {}

        for i, label_id in enumerate(unique):
            # Extract mask for this specific plant
            mask = (labels == label_id).astype(np.uint8)
            y_idx, x_idx = np.where(mask)

            if len(y_idx) == 0:
                continue

            # Crop region with padding for better edge handling
            pad = 10
            y_min, y_max = max(0, y_idx.min() - pad), min(
                mask.shape[0], y_idx.max() + pad
            )
            x_min, x_max = max(0, x_idx.min() - pad), min(
                mask.shape[1], x_idx.max() + pad
            )
            crop = mask[y_min:y_max, x_min:x_max]

            # Calculate geodesic path along the root
            result = calculate_geodesic_length(crop > 0, max_gap=max_gap)
            plant_id = label_id - 1

            path = result["path"]
            if path:
                # Extract start point (anchor/seed location) and end point (root tip)
                sy, sx = int(path[0][0]), int(round(path[0][1]))
                ey, ex = int(path[-1][0]), int(round(path[-1][1]))
                all_results[plant_id] = [(sy, sx), (ey, ex)]

        return all_results

    def get_coordinates(
        self,
        img: np.ndarray,
        img_filename: str,
        root_config: ModelConfig,
        anchor_config: ModelConfig,
        max_gap: int,
    ) -> Dict[List[tuple, tuple]]:
        """Public method to get root tip coordinates from an image.
        
        Convenience wrapper around _get_root_tip_coordinates.
        
        Args:
            img: Input image containing plants
            img_filename: Filename for logging
            root_config: Root model configuration
            anchor_config: Anchor model configuration
            max_gap: Maximum gap for geodesic calculation
            
        Returns:
            Dictionary mapping plant_id to [(start_coords), (end_coords)]
        """
        return self._get_root_tip_coordinates(
            img, img_filename, root_config, anchor_config, max_gap
        )