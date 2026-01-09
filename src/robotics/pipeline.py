"""
Robotics Pipeline for Plant Root Analysis and Robotic Manipulation

This module provides a robotics interface built on top of PlantMaster for:
- Detecting plant roots and extracting root tip coordinates
- Interfacing with an OT-2 robotic environment for automated manipulation
- Converting between pixel and real-world millimeter coordinates
"""

import cv2
import numpy as np
import gymnasium as gym
import logging
from typing import Tuple, List, Dict, Optional

from src.core.plant_master import PlantMaster
from src.configs.inference import INFERENCE_CONFIG
from src.configs.segmentation import SEGMENTATION_CONFIG
from src.configs.robotics import ENV_CONFIG, CONTROLLER_CONFIG
from src.configs.lenghts import LENGTH_CALCULATOR_CONFIG
from src.configs.thresholds import ADAPTIVE_THRESHOLD_CONFIG  # You may need to create/import this
from src.helpers.extract_petri_dish import extract_petri_dish
import src.robotics.ot2_env  # Import used for serialization
from src.robotics.pid import PIDController
from src.models import (
    SegmentationConfig,
    ModelConfig,
    EnvConfig,
    ControllerConfig,
    AdaptiveThresholdConfig,
    ImageResult,
)
from src.models.length import LengthCalculatorConfig
from src.exceptions import InitializationError

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
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
    """
    Robotics pipeline for plant root analysis and robotic manipulation.
    
    Uses PlantMaster in crop_to_dish mode for all CV operations,
    then handles coordinate conversion and robotic control.
    """

    def __init__(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: SegmentationConfig,
        env_config: EnvConfig,
        controller_config: ControllerConfig,
        length_config: Optional[LengthCalculatorConfig] = None,
        thresholds_config: Optional[AdaptiveThresholdConfig] = None,
    ):
        self.env_config = env_config
        self.controller_config = controller_config
        self.length_config = length_config or LENGTH_CALCULATOR_CONFIG
        
        # Initialize PlantMaster with crop_to_dish=True for dish-relative coordinates
        self._init_plant_master(
            inference_config,
            segmentation_config,
            self.length_config,
            thresholds_config,
        )
        
        # Initialize robotics environment
        self._init_robotics_env()

    def _init_plant_master(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: SegmentationConfig,
        length_config: LengthCalculatorConfig,
        thresholds_config: Optional[AdaptiveThresholdConfig] = None,
    ) -> None:
        """Initialize PlantMaster for CV operations."""
        try:
            # Use default thresholds if not provided
            if thresholds_config is None:
                thresholds_config = ADAPTIVE_THRESHOLD_CONFIG
            
            self.plant_master = PlantMaster(
                inference_config=inference_config,
                segmentation_config=segmentation_config,
                calculator_config=length_config,
                thresholds_config=thresholds_config,
                crop_to_dish=True,  # Key: operate in dish-relative mode
            )
            logger.debug("✓ PlantMaster initialized (crop_to_dish=True)")
        except Exception as e:
            logger.exception(f"⚠ Failed to initialize PlantMaster: {e}")
            raise InitializationError(f"PlantMaster initialization failed: {e}")

    def _init_robotics_env(self) -> None:
        """Initialize the robotics environment."""
        try:
            self.env = gym.make(
                "OT2ENV-v0",
                num_agents=self.env_config.num_agents,
                render_mode=self.env_config.render_mode,
                target=self.env_config.target,
                max_steps=self.env_config.max_steps,
            )
            logger.debug("✓ Robotics environment loaded")
        except Exception as e:
            logger.exception(f"⚠ Failed to load Environment: {e}")
            raise InitializationError(f"Environment initialization failed: {e}")

    def _get_dish_size_px(self, img: np.ndarray) -> int:
        """Extract petri dish from image and return its size in pixels."""
        dish = extract_petri_dish(img)
        return dish.shape[0]

    def _convert_px_to_mm(
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

    def _extract_root_tips(
        self, result: ImageResult
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Extract root tip coordinates from PlantMaster result.
        
        Returns:
            Dict mapping plant_id to [(start_y, start_x), (end_y, end_x)]
        """
        tips = {}
        for plant_id, plant_result in result.plant_results.items():
            if (
                plant_result.detected
                and plant_result.start_point is not None
                and plant_result.end_point is not None
            ):
                tips[plant_id] = [
                    plant_result.start_point,
                    plant_result.end_point,
                ]
                logger.debug(
                    f"Plant {plant_id}: start={plant_result.start_point}, "
                    f"end={plant_result.end_point}, length={plant_result.length}px"
                )
        return tips

    def get_coordinates(
        self,
        img: np.ndarray,
        img_filename: str = "unknown",
    ) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        """
        Get root tip coordinates from an image.
        
        Args:
            img: BGR image array (full image, will be cropped internally)
            img_filename: Name for logging
            
        Returns:
            Dict mapping plant_id to [(start_y, start_x), (end_y, end_x)]
            Coordinates are relative to the cropped dish region.
        """
        try:
            result = self.plant_master.process_image(img, img_filename)
            tips = self._extract_root_tips(result)
            
            if not tips:
                logger.warning("No root tips detected")
                return None
                
            return tips
        except Exception as e:
            logger.exception(f"Error processing image {img_filename}: {e}")
            return None

    def get_full_analysis(
        self,
        img: np.ndarray,
        img_filename: str = "unknown",
    ) -> Optional[ImageResult]:
        """
        Get full analysis result from PlantMaster.
        
        Use this when you need more than just coordinates (e.g., masks, skeletons).
        """
        try:
            return self.plant_master.process_image(img, img_filename)
        except Exception as e:
            logger.exception(f"Error processing image {img_filename}: {e}")
            return None

    def run_robotic_manipulation(
        self,
        controller_config: Optional[ControllerConfig] = None,
        starting_point: Tuple[float, float] = (0.0, 0.0),
        pid_params: Optional[Dict] = None,
    ) -> int:
        """
        Run complete robotic manipulation pipeline.
        
        Args:
            controller_config: Robot controller configuration
            starting_point: (x, y) offset in meters for coordinate conversion
            pid_params: PID controller parameters for x, y, z axes
            
        Returns:
            Number of targets successfully reached
        """
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
        logger.info("Capturing image from robot camera...")
        unwrapped_env = self.env.unwrapped
        image_path = unwrapped_env.sim.get_plate_image()
        img = cv2.imread(image_path)

        if img is None:
            logger.error(f"Failed to load image from {image_path}")
            return 0

        # Step 2: Detect root tips using PlantMaster
        logger.info("Detecting root tips...")
        root_tips = self.get_coordinates(img=img, img_filename=image_path)

        if root_tips is None or len(root_tips) == 0:
            logger.error("No root tips detected")
            return 0

        # Step 3: Convert to mm then to meter coordinates
        logger.info(f"Converting coordinates for {len(root_tips)} plants...")
        targets = self._compute_targets(
            img, root_tips, controller_config, starting_point
        )

        if not targets:
            logger.error("Failed to compute targets")
            return 0

        # Step 4: Execute robotic manipulation
        logger.info(f"Starting robotic manipulation for {len(targets)} targets...")
        return self._execute_manipulation(
            targets, controller_config, pid_x, pid_y, pid_z
        )

    def _compute_targets(
        self,
        img: np.ndarray,
        root_tips: Dict[int, List[Tuple[int, int]]],
        controller_config: ControllerConfig,
        starting_point: Tuple[float, float],
    ) -> List[List[float]]:
        """Convert root tips to robot target coordinates."""
        targets = []
        for plant_id, coords in root_tips.items():
            end_y, end_x = coords[1]  # Get end point (root tip)

            # Convert from pixels to mm
            tip_mm = self._convert_px_to_mm(
                img=img,
                points=[(end_x, end_y)],
                size_mm=controller_config.dish_size,
            )

            # Convert to meters with offset
            x_mm, y_mm = tip_mm[0]
            x_m = starting_point[0] + (y_mm / 1000.0)
            y_m = starting_point[1] + (x_mm / 1000.0)
            z_m = 0.1215  # Fixed z height

            targets.append([x_m, y_m, z_m])
            logger.debug(
                f"Plant {plant_id}: Root tip at ({x_m:.4f}, {y_m:.4f}, {z_m:.4f})"
            )

        return targets

    def _execute_manipulation(
        self,
        targets: List[List[float]],
        controller_config: ControllerConfig,
        pid_x: PIDController,
        pid_y: PIDController,
        pid_z: PIDController,
    ) -> int:
        """Execute the robotic manipulation loop."""
        unwrapped_env = self.env.unwrapped
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
                        f"Dropping at target {target_idx} (step {step}), "
                        f"distance: {distance:.6f}"
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

        logger.info(f"Finished! Dropped at {target_idx + 1} targets")
        return target_idx


def main():
    pipeline = RoboticsPipeline(
        inference_config=INFERENCE_CONFIG,
        segmentation_config=SEGMENTATION_CONFIG,
        env_config=ENV_CONFIG,
        controller_config=CONTROLLER_CONFIG,
    )

    num_drops = pipeline.run_robotic_manipulation(
        controller_config=CONTROLLER_CONFIG,
        starting_point=(0.10775, 0.062),
    )


if __name__ == "__main__":
    main()