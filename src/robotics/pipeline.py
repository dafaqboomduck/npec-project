import cv2
import numpy as np
from pathlib import Path
import gymnasium as gym
import logging
from typing import Tuple, List

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

MAX_STEPS = 5000
DROP_THRESHOLD = 0.0001  # Distance threshold to trigger drop
DROP_COOLDOWN = 50  # Steps to wait after dropping before moving to next target

custom_objects = {"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}
dt = 1.0 / 240.0

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    num_agents: int = 1
    render_mode: str | None = None
    target: np.ndarray | None = None
    max_steps: int = 1000


@dataclass
class ModelConfig:
    checkpoint: str | Path
    threshold: float
    patch_size: int
    custom_objects: dict


ROOT_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ROOT_MODEL_PATH),
    threshold=0.5,
    patch_size=512,
    custom_objects=custom_objects,
)
ANCHOR_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ANCHOR_MODEL_PATH),
    threshold=0.01,
    patch_size=512,
    custom_objects=custom_objects,
)
INFERENCE_CONFIG = (ROOT_MODEL_CONFIG, ANCHOR_MODEL_CONFIG)
SEGMENTOR_CONFIG = (PLANT_ZONES, SEGMENTATION_CONFIG)
ENV_CONFIG = EnvConfig(render_mode="rgb_array")


class RoboticsPipeline:
    def __init__(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentor_config: Tuple[dict, SegmentationConfig],
        env_config: EnvConfig,
    ):
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
        logger.debug("Initializing pipeline components...")
        try:
            try:
                self.root_manager = InferenceManager(
                    inference_config[0].checkpoint, inference_config[0].custom_objects
                )
                logger.debug("✓ Root model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Root model. Exception: {e}")
                raise InitializationError

            try:
                self.anchor_manager = InferenceManager(
                    inference_config[1].checkpoint, inference_config[1].custom_objects
                )
                logger.debug("✓ Anchor model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Anchor model. Exception: {e}")
                raise InitializationError

            try:
                self.segmentor = PlantSegmentor(
                    segmentor_config[0], segmentor_config[1]
                )
                logger.debug("✓ Segmentor instance loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Segmentor. Exception: {e}")
                raise InitializationError

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

        dish = extract_petri_dish(img)
        return dish.shape[0]

    def _convert_mm2px(
        self,
        img: np.ndarray,
        points: List[Tuple[float, float]] | Tuple[float, float],
        size_mm: float,
    ) -> List[Tuple[float, float]]:
        factor = size_mm / self._get_dish_size_px(img)
        if not isinstance(points, list):
            points = [points]
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
        try:
            patches = manager.patch_image(img, patch_size=patch_size, normalize=True)
        except Exception as e:
            logger.exception(f"Failed to patch image {img_filename}: {e}")
            raise InferenceError

        try:
            padded_mask = manager.create_img_mask(patches)
        except Exception as e:
            logger.exception(f"Model prediction failed for {img_filename}: {e}")
            raise InferenceError

        if padded_mask is None:
            logger.exception(f"Model returned None for {img_filename}")
            raise InferenceError

        binary_mask = (padded_mask > threshold).astype(np.uint8) * 255
        return binary_mask

    def _validate_mask_dimensions(self, img, mask, img_filename):
        # Validate mask dimensions
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
    ) -> np.ndarray:
        # Step 1: Run anchor model
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

        validate = self._validate_mask_dimensions(img, anchor_mask, img_filename)
        if not bool(validate):
            return None

        # Step 2: Extract and filter anchors
        all_anchors = self.segmentor.get_anchors(anchor_mask)
        anchors = self.segmentor.get_largest_anchor_by_zone(all_anchors)
        anchors = self.segmentor.create_anchor_bboxes(img.shape, anchors)
        logger.debug((f" Anchors: {len(anchors)}"))

        # Step 3: Run root model
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

        validate = self._validate_mask_dimensions(img, root_mask, img_filename)
        if not bool(validate):
            return None

        # Step 4: Process roots by anchor regions
        _, final_labels = self.segmentor.process_roots_by_anchor(root_mask, anchors)

        return np.array(final_labels)

    def get_coordinates(
        self,
        img: np.ndarray,
        img_filename: str,
        root_config: ModelConfig,
        anchor_config: ModelConfig,
    ) -> np.ndarray:
        return self._get_root_tip_coordinates(
            img, img_filename, root_config, anchor_config
        )
    

