"""
Robotics Pipeline for Plant Root Analysis and Robotic Manipulation

This module provides a complete pipeline for:
- Detecting plant roots and anchors using deep learning models
- Segmenting plants into zones and extracting root tip coordinates
- Interfacing with an OT-2 robotic environment for automated manipulation
- Converting between pixel and real-world millimeter coordinates
"""

import cv2
import numpy as np
import gymnasium as gym
import logging
from typing import Tuple, List, Dict, Optional

from src.inference import InferenceManager
from src.core.plant_segmenation import PlantSegmentation
from src.core.root_lengths import LengthCalculator
from src.configs.inference import INFERENCE_CONFIG
from src.configs.segmentation import SEGMENTATION_CONFIG
from src.configs.robotics import ENV_CONFIG, CONTROLLER_CONFIG
from src.configs.lenghts import LENGTH_CALCULATOR_CONFIG
from src.helpers.extract_petri_dish import extract_petri_dish
import src.robotics.ot2_env
from src.robotics.pid import PIDController
from src.models import (
    SegmentationConfig,
    ModelConfig,
    EnvConfig,
    ControllerConfig,
)
from src.exceptions import InferenceError, InitializationError
from src.models.length import LengthCalculatorConfig

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PLANT_ZONES = {
    0: (0, 566),
    1: (566, 1132),
    2: (1132, 1698),
    3: (1698, 2264),
    4: (2264, 2834),
}

SEGMENTATION_CONFIG.plant_zones = PLANT_ZONES


class RoboticsPipeline:
    """Main pipeline for plant root analysis and robotic manipulation."""

    def __init__(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: SegmentationConfig,
        env_config: EnvConfig,
        controller_config: ControllerConfig,
        length_config: Optional[LengthCalculatorConfig] = None,
    ):
        self.inference_config = inference_config
        self.segmentation_config = segmentation_config
        self.env_config = env_config
        self.controller_config = controller_config
        self.length_config = length_config or LENGTH_CALCULATOR_CONFIG
        self._init_components()

    def _init_components(self):
        """Initialize all pipeline components with error handling."""
        logger.debug("Initializing pipeline components...")

        try:
            # Load root detection model - pass ModelConfig object directly
            try:
                self.root_manager = InferenceManager(self.inference_config[0])
                logger.debug("✓ Root model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Root model: {e}")
                raise InitializationError(f"Root model initialization failed: {e}")

            # Load anchor detection model - pass ModelConfig object directly
            try:
                self.anchor_manager = InferenceManager(self.inference_config[1])
                logger.debug("✓ Anchor model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Anchor model: {e}")
                raise InitializationError(f"Anchor model initialization failed: {e}")

            # Initialize plant segmentor - now PlantSegmentation class
            try:
                self.segmentor = PlantSegmentation(self.segmentation_config)
                logger.debug("✓ Segmentor instance loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Segmentor: {e}")
                raise InitializationError(f"Segmentor initialization failed: {e}")

            # Initialize length calculator
            try:
                self.length_calculator = LengthCalculator(self.length_config)
                logger.debug("✓ LengthCalculator instance loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load LengthCalculator: {e}")
                raise InitializationError(
                    f"LengthCalculator initialization failed: {e}"
                )

            # Initialize robotics environment
            try:
                self.env = gym.make(
                    "OT2ENV-v0",
                    num_agents=self.env_config.num_agents,
                    render_mode=self.env_config.render_mode,
                    target=self.env_config.target,
                    max_steps=self.env_config.max_steps,
                )
                logger.debug("✓ Environment loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Environment: {e}")
                raise InitializationError(f"Environment initialization failed: {e}")

            logger.debug("Pipeline components initialized successfully!")

        except InitializationError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during initialization: {e}")
            raise InitializationError(f"Unexpected initialization error: {e}")

    def _get_dish_size_px(self, img: np.ndarray) -> int:
        """Extract petri dish from image and return its size in pixels."""
        dish = extract_petri_dish(img)
        return dish.shape[0]

    def _convert_mm2px(
        self,
        img: np.ndarray,
        points: List[Tuple[float, float]] | np.ndarray,
        size_mm: float,
    ) -> np.ndarray:
        """Convert coordinates from pixels to millimeters."""
        dish_px = self._get_dish_size_px(img)
        if dish_px == 0:
            logger.error("Could not detect Petri Dish size.")
            raise ValueError("Petri dish size detection failed")

        factor = size_mm / dish_px

        if not isinstance(points, list):
            points = [points]

        points_arr = np.array(points)
        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(1, 2)

        return points_arr * factor

    def _run_inference(
        self,
        manager: InferenceManager,
        img: np.ndarray,
        img_filename: str,
        patch_size: int,
        threshold: float,
    ) -> np.ndarray:
        """Run model inference on an image using patch-based processing."""
        try:
            patches = manager.patch_image(img, patch_size=patch_size, normalize=True)
        except Exception as e:
            logger.exception(f"Failed to patch image {img_filename}: {e}")
            raise InferenceError(f"Patching failed: {e}")

        try:
            padded_mask = manager.create_img_mask(patches)
        except Exception as e:
            logger.exception(f"Model prediction failed for {img_filename}: {e}")
            raise InferenceError(f"Prediction failed: {e}")

        if padded_mask is None:
            logger.exception(f"Model returned None for {img_filename}")
            raise InferenceError("Model returned None")

        # Use the manager's restore_mask method instead of custom _unpadd_mask
        restored_mask = manager.restore_mask(padded_mask, patch_size=patch_size)

        # Extract just the dish region for consistency with old behavior
        if manager.crop_coords is not None:
            x1, y1, x2, y2 = manager.crop_coords
            dish_mask = restored_mask[y1:y2, x1:x2]
        else:
            dish_mask = restored_mask

        binary_mask = (dish_mask > threshold).astype(np.uint8) * 255
        return binary_mask

    def _validate_mask_dimensions(
        self, img: np.ndarray, mask: np.ndarray, img_filename: str
    ) -> bool:
        """Validate that mask dimensions match the input image."""
        if img.shape[:2] != mask.shape[:2]:
            logger.error(
                f"Shape mismatch for {img_filename}: "
                f"image={img.shape[:2]}, mask={mask.shape[:2]}"
            )
            return False
        return True

    def _get_root_tip_coordinates(
        self,
        img: np.ndarray,
        img_filename: str,
        root_config: Optional[ModelConfig] = None,
        anchor_config: Optional[ModelConfig] = None,
        segmentation_config: Optional[SegmentationConfig] = None,
    ) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        """Extract root tip coordinates for all detected plants."""
        if root_config is None:
            root_config = self.inference_config[0]
        if anchor_config is None:
            anchor_config = self.inference_config[1]
        if segmentation_config is None:
            segmentation_config = self.segmentation_config

        # Step 1: Run anchor detection
        try:
            anchor_mask = self._run_inference(
                self.anchor_manager,
                img=img,
                img_filename=img_filename,
                patch_size=anchor_config.patch_size,
                threshold=anchor_config.threshold,
            )
        except InferenceError as e:
            logger.error(f"InferenceError during Anchor Model: {e}")
            return None

        dish = extract_petri_dish(img)

        if not self._validate_mask_dimensions(dish, anchor_mask, img_filename):
            return None

        # Step 2: Extract anchors using updated PlantSegmentation API
        anchors = self.segmentor.extract_anchors(anchor_mask)

        # Create bounding boxes for anchors
        anchors = self.segmentor.create_anchor_bboxes(
            anchors=anchors,
            img_shape=dish.shape,
        )

        logger.debug(f"Detected {len(anchors)} anchors")

        if not anchors:
            logger.warning("No anchors detected")
            return None

        # Step 3: Run root detection
        try:
            root_mask = self._run_inference(
                self.root_manager,
                img=img,
                img_filename=img_filename,
                patch_size=root_config.patch_size,
                threshold=root_config.threshold,
            )
        except InferenceError as e:
            logger.error(f"InferenceError during Root Model: {e}")
            return None

        if not self._validate_mask_dimensions(dish, root_mask, img_filename):
            return None

        # Step 4: Segment roots using updated API
        segmentation_result = self.segmentor.segment_roots(
            root_mask=root_mask,
            anchors=anchors,
        )

        labels = segmentation_result.labels

        # Step 5: Extract root tip coordinates using LengthCalculator
        unique_labels = [l for l in np.unique(labels) if l > 0]
        if not unique_labels:
            logger.warning("No plants detected")
            return {}

        all_results = {}

        # Prepare endpoint info from anchors
        endpoint_info = {}
        for anchor in anchors:
            if anchor.zone_id is not None and anchor.bbox is not None:
                x_min, y_min, x_max, y_max = anchor.bbox
                # Use anchor centroid as start point
                cy, cx = anchor.centroid[1], anchor.centroid[0]
                endpoint_info[anchor.zone_id] = {
                    "detected": True,
                    "start": (cy, cx),
                    "end": None,  # Let the algorithm find the farthest point
                }

        for label_id in unique_labels:
            plant_id = int(label_id) - 1

            # Extract mask for this plant
            mask = (labels == label_id).astype(np.uint8)
            y_idx, x_idx = np.where(mask)

            if len(y_idx) == 0:
                continue

            # Crop region with padding
            pad = 10
            y_min = max(0, y_idx.min() - pad)
            y_max = min(mask.shape[0], y_idx.max() + pad)
            x_min = max(0, x_idx.min() - pad)
            x_max = min(mask.shape[1], x_idx.max() + pad)
            crop = mask[y_min:y_max, x_min:x_max]

            # Get start point in crop coordinates if available
            start_point = None
            if plant_id in endpoint_info and endpoint_info[plant_id].get("start"):
                orig_start = endpoint_info[plant_id]["start"]
                start_point = (orig_start[0] - y_min, orig_start[1] - x_min)

            # Calculate geodesic path using LengthCalculator
            path_result, return_code = self.length_calculator.calculate_geodesic_length(
                binary_mask=crop > 0,
                start_point=start_point,
                max_gap=(
                    self.length_config.max_gap
                    if hasattr(self.length_config, "max_gap")
                    else 50
                ),
                max_path=(
                    self.length_config.max_path
                    if hasattr(self.length_config, "max_path")
                    else 100
                ),
            )

            if path_result.path:
                # Extract start and end points
                sy, sx = int(path_result.path[0][0]), int(path_result.path[0][1])
                ey, ex = int(path_result.path[-1][0]), int(path_result.path[-1][1])

                # Convert back to full image coordinates
                sy += y_min
                ey += y_min
                sx += x_min
                ex += x_min

                all_results[plant_id] = [(sy, sx), (ey, ex)]
                logger.debug(
                    f"Plant {plant_id}: path length={path_result.length:.2f}px, "
                    f"start=({sy}, {sx}), end=({ey}, {ex})"
                )

        return all_results

    def get_coordinates(
        self,
        img: np.ndarray,
        img_filename: str,
        inference_config: Optional[Tuple[ModelConfig, ModelConfig]] = None,
        segmentation_config: Optional[SegmentationConfig] = None,
    ) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        """Public method to get root tip coordinates from an image."""
        root_config, anchor_config = None, None
        if inference_config is not None:
            root_config, anchor_config = inference_config
        return self._get_root_tip_coordinates(
            img, img_filename, root_config, anchor_config, segmentation_config
        )

    def run_robotic_manipulation(
        self,
        inference_config: Optional[Tuple[ModelConfig, ModelConfig]] = None,
        segmentation_config: Optional[SegmentationConfig] = None,
        controller_config: Optional[ControllerConfig] = None,
        starting_point: Tuple[float, float] = (0.0, 0.0),
        pid_params: Optional[Dict] = None,
    ) -> int:
        """Run complete robotic manipulation pipeline."""
        if inference_config is None:
            inference_config = self.inference_config
        if controller_config is None:
            controller_config = self.controller_config
        if pid_params is None:
            pid_params = {
                "x": {"kp": 9.0, "ki": 0.048, "kd": 0.0},
                "y": {"kp": 9.0, "ki": 0.048, "kd": 0.0},
                "z": {"kp": 5.0, "ki": 0.024, "kd": 0.0},
            }

        pid_x = PIDController(**pid_params["x"])
        pid_y = PIDController(**pid_params["y"])
        pid_z = PIDController(**pid_params["z"])

        # Step 1: Get image from robot camera
        logger.debug("Capturing image from robot camera...")
        unwrapped_env = self.env.unwrapped
        image_path = unwrapped_env.sim.get_plate_image()
        img = cv2.imread(image_path)

        if img is None:
            logger.error(f"Failed to load image from {image_path}")
            return 0

        # Step 2: Detect root tips
        logger.info("Detecting root tips...")
        root_tips = self.get_coordinates(
            img=img,
            img_filename=image_path,
            inference_config=inference_config,
            segmentation_config=segmentation_config,
        )

        if root_tips is None or len(root_tips) == 0:
            logger.error("No root tips detected")
            return 0

        # Step 3: Convert to mm coordinates
        logger.debug(f"Converting coordinates for {len(root_tips)} plants...")
        targets = []
        for plant_id, coords in root_tips.items():
            end_y, end_x = coords[1]

            tip_mm = self._convert_mm2px(
                img=img,
                points=[(end_x, end_y)],
                size_mm=controller_config.dish_size,
            )

            x_mm, y_mm = tip_mm[0]
            x_m = starting_point[0] + (y_mm / 1000.0)
            y_m = starting_point[1] + (x_mm / 1000.0)
            z_m = 0.1215

            targets.append([x_m, y_m, z_m])
            logger.debug(
                f"Plant {plant_id}: Root tip at ({x_m:.4f}, {y_m:.4f}, {z_m:.4f})"
            )

        # Step 4: Execute robotic manipulation
        logger.info(f"Starting robotic manipulation for {len(targets)} targets...")

        obs, _ = self.env.reset(seed=42)

        if len(targets) > 0:
            unwrapped_env.target = np.array(targets[0], dtype=np.float32)

        target_idx = 0
        dropped_at_target = False
        cooldown_counter = 0

        logger.debug(f"Moving to target {target_idx}")

        for step in range(controller_config.max_steps):
            if target_idx >= len(targets):
                logger.info(f"All targets completed at step {step}!")
                break

            target = targets[target_idx]
            x, y, z = obs[6:9]
            target_x, target_y, target_z = target

            error_x = target_x - x
            error_y = target_y - y
            error_z = target_z - z
            distance = np.sqrt(error_x**2 + error_y**2 + error_z**2)

            dt = controller_config.dt

            control_x = pid_x.compute(error_x, dt, output_limits=(-1.0, 1.0))
            control_y = pid_y.compute(error_y, dt, output_limits=(-1.0, 1.0))
            control_z = pid_z.compute(error_z, dt, output_limits=(-1.0, 1.0))

            if not dropped_at_target:
                if distance <= controller_config.drop_threshold:
                    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    dropped_at_target = True
                    cooldown_counter = 0
                    logger.debug(
                        f"Dropping at target {target_idx} (step {step}), distance: {distance:.6f}"
                    )
                else:
                    action = np.array(
                        [control_x, control_y, control_z, 0.0], dtype=np.float32
                    )
            else:
                cooldown_counter += 1
                action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

                if cooldown_counter >= controller_config.drop_cooldown:
                    target_idx += 1
                    dropped_at_target = False
                    cooldown_counter = 0

                    pid_x.reset()
                    pid_y.reset()
                    pid_z.reset()

                    if target_idx < len(targets):
                        logger.debug(f"Moving to target {target_idx}")
                        unwrapped_env.target = np.array(
                            targets[target_idx], dtype=np.float32
                        )

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                logger.debug(f"Episode ended at step {step}")
                break

        logger.info(f"Finished! Dropped at {target_idx} targets")
        return target_idx


def main():
    pipeline = RoboticsPipeline(
        inference_config=INFERENCE_CONFIG,
        segmentation_config=SEGMENTATION_CONFIG,
        env_config=ENV_CONFIG,
        controller_config=CONTROLLER_CONFIG,
    )

    num_drops = pipeline.run_robotic_manipulation(
        INFERENCE_CONFIG,
        SEGMENTATION_CONFIG,
        CONTROLLER_CONFIG,
        starting_point=(0.10775, 0.062),
    )

    print(f"Successfully dropped at {num_drops} root tips")


if __name__ == "__main__":
    main()
