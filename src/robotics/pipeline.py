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
from src.configs.thresholds import ADAPTIVE_THRESHOLD_CONFIG
from src.helpers.extract_petri_dish import extract_petri_dish
import src.robotics.ot2_env
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

# Import the new controller abstraction
from src.robotics.controllers import BaseController, PIDRobotController

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
        controller: Optional[BaseController] = None,
    ):
        self.env_config = env_config
        self.controller_config = controller_config
        self.length_config = length_config or LENGTH_CALCULATOR_CONFIG

        self._init_plant_master(
            inference_config, segmentation_config, self.length_config, thresholds_config
        )
        self._init_robotics_env()

        # Initialize controller (default to PID if not provided)
        self.controller = controller or PIDRobotController()

    def _init_plant_master(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: SegmentationConfig,
        length_config: LengthCalculatorConfig,
        thresholds_config: Optional[AdaptiveThresholdConfig] = None,
    ) -> None:
        """Initialize PlantMaster for CV operations."""
        try:
            thresholds_config = thresholds_config or ADAPTIVE_THRESHOLD_CONFIG
            self.plant_master = PlantMaster(
                inference_config=inference_config,
                segmentation_config=segmentation_config,
                calculator_config=length_config,
                thresholds_config=thresholds_config,
                crop_to_dish=True,
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

    def set_controller(self, controller: BaseController) -> None:
        """Swap the controller at runtime."""
        self.controller = controller
        logger.info(f"Controller set to {type(controller).__name__}")

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
            raise ValueError("Petri dish size detection failed")
        factor = size_mm / dish_px
        points_arr = np.array(points if isinstance(points, list) else [points])
        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(1, 2)
        return points_arr * factor

    def _extract_root_tips(
        self, result: ImageResult
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Extract root tip coordinates from PlantMaster result."""
        tips = {}
        for plant_id, plant_result in result.plant_results.items():
            if (
                plant_result.detected
                and plant_result.start_point
                and plant_result.end_point
            ):
                tips[plant_id] = [plant_result.start_point, plant_result.end_point]
                logger.debug(
                    f"Plant {plant_id}: start={plant_result.start_point}, end={plant_result.end_point}"
                )
        return tips

    def get_coordinates(
        self, img: np.ndarray, img_filename: str = "unknown"
    ) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        """Get root tip coordinates from an image."""
        try:
            result = self.plant_master.process_image(img, img_filename)
            tips = self._extract_root_tips(result)
            return tips if tips else None
        except Exception as e:
            logger.exception(f"Error processing image {img_filename}: {e}")
            return None

    def get_full_analysis(
        self, img: np.ndarray, img_filename: str = "unknown"
    ) -> Optional[ImageResult]:
        """Get full analysis result from PlantMaster."""
        try:
            return self.plant_master.process_image(img, img_filename)
        except Exception as e:
            logger.exception(f"Error processing image {img_filename}: {e}")
            return None

    def _compute_targets(
        self,
        img: np.ndarray,
        root_tips: Dict[int, List[Tuple[int, int]]],
        starting_point: Tuple[float, float],
    ) -> List[List[float]]:
        """Convert root tips to robot target coordinates."""
        targets = []
        for plant_id, coords in root_tips.items():
            end_y, end_x = coords[1]
            tip_mm = self._convert_px_to_mm(
                img=img,
                points=[(end_x, end_y)],
                size_mm=self.controller_config.dish_size,
            )
            x_mm, y_mm = tip_mm[0]
            x_m = starting_point[0] + (y_mm / 1000.0)
            y_m = starting_point[1] + (x_mm / 1000.0)
            z_m = 0.1215
            targets.append([x_m, y_m, z_m])
            logger.debug(
                f"Plant {plant_id}: Root tip at ({x_m:.4f}, {y_m:.4f}, {z_m:.4f})"
            )
        return targets

    def run_robotic_manipulation(
            self,
            starting_point: Tuple[float, float] = (0.0, 0.0),
            starting_z: float = 0.1215,
            max_steps: Optional[int] = None,
            controller: Optional[BaseController] = None,
        ) -> int:
            """
            Run complete robotic manipulation pipeline.

            Args:
                starting_point: (x, y) offset in meters for coordinate conversion
                starting_z: z coordinate for pipette starting position
                max_steps: Override max steps (uses controller_config.max_steps if None)
                controller: Optional controller override for this run only

            Returns:
                Number of targets successfully reached
            """
            active_controller = controller or self.controller
            max_steps = max_steps or self.controller_config.max_steps

            # Step 1: Get image from robot camera
            logger.info("Capturing image from robot camera...")
            unwrapped_env = self.env.unwrapped
            image_path = unwrapped_env.sim.get_plate_image()
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image from {image_path}")
                return 0

            # Step 2: Detect root tips
            logger.info("Detecting root tips...")
            root_tips = self.get_coordinates(img=img, img_filename=image_path)
            if not root_tips:
                logger.error("No root tips detected")
                return 0

            # Step 3: Compute targets
            logger.info(f"Converting coordinates for {len(root_tips)} plants...")
            targets = self._compute_targets(img, root_tips, starting_point)
            if not targets:
                logger.error("Failed to compute targets")
                return 0

            # Step 4: Execute manipulation loop
            logger.info(
                f"Starting manipulation for {len(targets)} targets using {type(active_controller).__name__}"
            )
            starting_position = (starting_point[0], starting_point[1], starting_z)
            return self._execute_manipulation(targets, active_controller, max_steps, starting_position)

    def _execute_manipulation(
            self, targets: List[List[float]], controller: BaseController, max_steps: int,
            starting_position: Tuple[float, float, float] = None,
        ) -> int:
            """Execute the robotic manipulation loop using the provided controller."""
            unwrapped_env = self.env.unwrapped
            obs, info = self.env.reset(seed=42)
            controller.reset()

            # Disable env's single-target termination for multi-target manipulation
            unwrapped_env.dwell_steps = max_steps + 1

            # Set pipette to starting position
            if starting_position:
                unwrapped_env.sim.set_start_position(*starting_position)
                states = unwrapped_env.sim.get_states()
                obs = unwrapped_env._get_obs(states)
                info = unwrapped_env._get_info(states)

            if targets:
                unwrapped_env.target = np.array(targets[0], dtype=np.float32)

            # Debug: Check initial state
            pipette_pos = controller.get_pipette_position(info)
            logger.debug(f"Initial pipette position: ({pipette_pos[0]:.4f}, {pipette_pos[1]:.4f}, {pipette_pos[2]:.4f})")

            dt = self.controller_config.dt
            targets_completed = 0

            for step in range(max_steps):
                action, done, completed_idx = controller.step(obs, info, targets, dt)

                if completed_idx is not None:
                    targets_completed = completed_idx + 1
                    target = targets[completed_idx]
                    pipette = controller.get_pipette_position(info)
                    dist = np.linalg.norm(np.array(target[:2]) - pipette[:2])
                    logger.debug(
                        f"TARGET {completed_idx}: pipette=({pipette[0]:.4f}, {pipette[1]:.4f}), "
                        f"dist={dist:.6f}"
                    )
                    # Reset env's dwell counter when moving to next target
                    unwrapped_env._steps_at_target = {i: 0 for i in range(unwrapped_env.num_agents)}

                if controller.state.current_target_idx < len(targets):
                    unwrapped_env.target = np.array(
                        targets[controller.state.current_target_idx], dtype=np.float32
                    )

                obs, reward, terminated, truncated, info = self.env.step(action)

                if done:
                    for _ in range(20):
                        self.env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    break

                if terminated or truncated:
                    pipette_pos = controller.get_pipette_position(info)
                    logger.debug(
                        f"Episode ended at step {step}: terminated={terminated}, truncated={truncated}, "
                        f"pipette=({pipette_pos[0]:.4f}, {pipette_pos[1]:.4f}, {pipette_pos[2]:.4f})"
                    )
                    break

            logger.info(f"Finished! Completed {targets_completed}/{len(targets)} targets")
            return targets_completed


def main():
    pipeline = RoboticsPipeline(
        inference_config=INFERENCE_CONFIG,
        segmentation_config=SEGMENTATION_CONFIG,
        env_config=ENV_CONFIG,
        controller_config=CONTROLLER_CONFIG,
    )

    num_drops = pipeline.run_robotic_manipulation(starting_point=(0.10775, 0.062))


if __name__ == "__main__":
    main()
