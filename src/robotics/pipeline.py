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
import gymnasium as gym
import logging
from typing import Tuple, List, Dict

# Import pipeline components
from src.inference import InferenceManager
from src.core.segmenation import PlantSegmentor
from src.core.root_length import calculate_geodesic_length
from src.configs.inference import INFERENCE_CONFIG
from src.configs.segmentation import SEGMENTATION_CONFIG
from src.configs.robotics import ENV_CONFIG, CONTROLLER_CONFIG
from src.helpers.extract_petri_dish import extract_petri_dish
import src.robotics.ot2_env  # needed to trigger registration
from src.robotics.pid import PIDController
from src.models import (
    InferenceError,
    InitializationError,
    SegmentationConfig,
    ModelConfig,
    EnvConfig,
    ControllerConfig,
)

# Configure logging
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
    """Main pipeline for plant root analysis and robotic manipulation.

    This class integrates:
    - Deep learning models for root and anchor detection
    - Plant segmentation for zone-based analysis
    - Gymnasium robotics environment for OT-2 robot control
    - Coordinate conversion between pixel and millimeter spaces

    Attributes:
        inference_config: Configuration tuple for root and anchor models
        segmentation_config: Configuration for plant segmentation
        env_config: Configuration for robotics environment
        root_manager: Inference manager for root detection model
        anchor_manager: Inference manager for anchor detection model
        segmentor: Plant segmentation processor
        env: Gymnasium robotics environment
    """

    def __init__(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: Tuple[dict, SegmentationConfig],
        env_config: EnvConfig,
        controller_config: ControllerConfig,
    ):
        """Initialize the robotics pipeline.

        Args:
            inference_config: Tuple of (root_config, anchor_config)
            segmentation_config: Tuple of (plant_zones, segmentation_config)
            env_config: Environment configuration
        """
        self.inference_config = inference_config
        self.segmentation_config = segmentation_config
        self.env_config = env_config
        self.controller_config = controller_config
        self._init_components(
            self.inference_config, self.segmentation_config, self.env_config
        )

    def _init_components(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: Tuple[dict, SegmentationConfig],
        env_config: EnvConfig,
    ):
        """Initialize all pipeline components with error handling.

        Loads deep learning models, segmentor, and robotics environment.
        Each component is loaded independently with specific error handling.

        Args:
            inference_config: Configuration for inference models
            segmentation_config: Configuration for plant segmentation
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
                self.segmentor = PlantSegmentor(segmentation_config)
                logger.debug("✓ Segmentor instance loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Segmentor. Exception: {e}")
                raise InitializationError

            # Initialize robotics environment
            try:
                self.env = gym.make(
                    "OT2ENV-v0",
                    num_agents=env_config.num_agents,
                    render_mode=env_config.render_mode,
                    target=env_config.target,
                    max_steps=env_config.max_steps,
                )
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
        points: List[Tuple[float, float]] | np.ndarray,
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
        # Get Petri Dish size in PX
        dish_px = self._get_dish_size_px(img)
        if dish_px == 0:
            logger.error(("Could not detect Petri Dish size."))
            raise ValueError

        # Calculate mm per pixel conversion factor
        factor = size_mm / dish_px

        # Ensure points is a list
        if not isinstance(points, list):
            points = [points]

        # Convert each point from pixels to millimeters
        points_arr = np.array(points)
        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(1, 2)

        # Apply scaling
        points_mm = points_arr * factor

        return points_mm

    def _unpadd_mask(
        self, manager: InferenceManager, padded_mask: np.ndarray, patch_size: int
    ) -> np.ndarray:
        if manager.crop_coords is None:
            raise ValueError(
                "restore_mask requires self.crop_coords and self.img_shape to be set by patch_image()."
            )
        # 1. Unpack coordinates from extract_petri_dish (in original image coordinates)
        x_start, y_start, x_end, y_end = manager.crop_coords

        # Calculate the dimensions of the crop BEFORE padding
        h_crop = y_end - y_start
        w_crop = x_end - x_start

        # 2. Re-calculate the padding that was added
        # (This logic MUST mirror the 'padder' utility function)
        height_padding = ((h_crop // patch_size) + 1) * patch_size - h_crop
        width_padding = ((w_crop // patch_size) + 1) * patch_size - w_crop

        # Calculate exactly how much was added to top and left
        top_pad = height_padding // 2
        left_pad = width_padding // 2

        # 3. Remove the padding (Un-pad)
        # Extract the region starting at top/left pad, with the size of the original crop.
        unpadded_mask = padded_mask[
            top_pad : top_pad + h_crop, left_pad : left_pad + w_crop
        ]

        return unpadded_mask

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

        unpadded_mask = self._unpadd_mask(
            manager, padded_mask=padded_mask, patch_size=patch_size
        )

        # Apply threshold to create binary mask
        binary_mask = (unpadded_mask > threshold).astype(np.uint8) * 255
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
        root_config: ModelConfig | None = None,
        anchor_config: ModelConfig | None = None,
        segmentation_config: SegmentationConfig | None = None,
    ) -> Dict[int, List[Tuple[int, int]]]:
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
            segmentation_config: Configuration for segmentation algorithm

        Returns:
            Dictionary mapping plant_id to [(start_y, start_x), (end_y, end_x)]
            Returns None if processing fails
        """
        # Set default ModelConfig parameters if not provided
        if root_config is None:
            root_config = self.inference_config[0]
        if anchor_config is None:
            anchor_config = self.inference_config[1]

        # Set default SegmentationConfig parameters if not provided
        if segmentation_config is None:
            segmentation_config = self.segmentation_config

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

        dish = extract_petri_dish(img)

        # Validate anchor mask dimensions
        validate = self._validate_mask_dimensions(dish, anchor_mask, img_filename)
        if not bool(validate):
            return None

        # cv2.imwrite(str(debug_dir / "anchors_mask.png"), anchor_mask)
        # logger.debug(f"Saved captured image to {debug_dir / 'anchors_mask.png'}")

        # Step 2: Extract and filter anchors by zone
        all_anchors = self.segmentor.get_anchors(
            anchor_mask, min_area=segmentation_config.min_anchor_area, y_ratio=0.25
        )

        anchors = self.segmentor.get_largest_anchor_by_zone(all_anchors)

        anchors = self.segmentor.create_anchor_bboxes(
            dish.shape,
            anchors,
            bbox_size=segmentation_config.anchor_bbox_size,
            vertical_offset=segmentation_config.vertical_offset,
        )
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
        validate = self._validate_mask_dimensions(dish, root_mask, img_filename)
        if not bool(validate):
            return None

        # cv2.imwrite(str(debug_dir / "root_mask.png"), root_mask)
        # logger.debug(f"Saved captured image to {debug_dir / 'root_mask.png'}")

        # Step 4: Associate roots with anchors by spatial regions
        _, labels = self.segmentor.process_roots_by_anchor(root_mask, anchors)

        # Step 5: Extract root tip coordinates for each plant
        unique = [l for l in np.unique(labels) if l > 0]
        if not unique:
            logger.warning("No plants detected")
            # return None

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

            max_gap = segmentation_config.root_merge_distance

            # Calculate geodesic path along the root
            result = calculate_geodesic_length(crop > 0, max_gap=max_gap)
            plant_id = label_id - 1

            path = result["path"]
            if path:
                # Extract start point (anchor/seed location) and end point (root tip)
                sy, sx = int(path[0][0]), int(round(path[0][1]))
                ey, ex = int(path[-1][0]), int(round(path[-1][1]))
                # Convert back to full image coordinates
                sy += y_min
                ey += y_min
                sx += x_min
                ex += x_min
                all_results[plant_id] = [(sy, sx), (ey, ex)]

        return all_results

    def get_coordinates(
        self,
        img: np.ndarray,
        img_filename: str,
        inference_config: Tuple[ModelConfig, ModelConfig] | None = None,
        segmentation_config: SegmentationConfig | None = None,
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Public method to get root tip coordinates from an image.

        Convenience wrapper around _get_root_tip_coordinates.

        Args:
            img: Input image containing plants
            img_filename: Filename for logging
            inference_config: Tuple of (root_config, anchor_config)
            segmentation_config: Configuration for segmentation algorithm

        Returns:
            Dictionary mapping plant_id to [(start_coords), (end_coords)]
        """
        root_config, anchor_config = inference_config
        return self._get_root_tip_coordinates(
            img, img_filename, root_config, anchor_config, segmentation_config
        )

    def run_robotic_manipulation(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig] | None = None,
        segmentation_config: SegmentationConfig | None = None,
        controller_config: ControllerConfig | None = None,
        starting_point: Tuple[float, float] = (0.0, 0.0),
        pid_params: Dict = None,
    ):
        """Run complete robotic manipulation pipeline.

        This method:
        1. Captures an image from the robot camera
        2. Detects plant root tips using computer vision
        3. Converts pixel coordinates to real-world millimeter positions
        4. Uses PID control to navigate the robot to each target
        5. Drops at each root tip location

        Args:
            inference_config: Tuple of (root_config, anchor_config)
            segmentation_config: Configuration for segmentation algorithm
            controller_config: Controller configuration with PID and drop parameters
            starting_point: Starting point of the dish
            pid_params: Dictionary with PID parameters for x, y, z axes
                       Format: {'x': {'kp': 9.0, 'ki': 0.048, 'kd': 0.0}, ...}

        Returns:
            Number of successful drops completed
        """
        # Set default InferenceConfig parameters if not provided
        if inference_config is None:
            inference_config = self.inference_config

        # Set default ControllerConfig parameters if not provided
        if controller_config is None:
            controller_config = self.controller_config

        # Set default PID parameters if not provided
        if pid_params is None:
            pid_params = {
                "x": {"kp": 9.0, "ki": 0.048, "kd": 0.0},
                "y": {"kp": 9.0, "ki": 0.048, "kd": 0.0},
                "z": {"kp": 5.0, "ki": 0.024, "kd": 0.0},
            }

        # Initialize PID controllers for each axis
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

        # # Debug: Save the captured image
        # debug_dir = Path("debug_output")
        # debug_dir.mkdir(exist_ok=True)
        # cv2.imwrite(str(debug_dir / "captured_image.png"), img)
        # logger.debug(f"Captured image shape: {img.shape}")
        # logger.debug(f"Saved captured image to {debug_dir / 'captured_image.png'}")

        # Step 2: Detect root tips and get coordinates
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

        # Step 3: Convert pixel coordinates to millimeter coordinates
        logger.debug(f"Converting coordinates for {len(root_tips)} plants...")
        targets = []
        for plant_id, coords in root_tips.items():
            # coords = [(start_y, start_x), (end_y, end_x)]
            # We want the end point (root tip)
            end_y, end_x = coords[1]

            # Convert to millimeters
            tip_mm = self._convert_mm2px(
                img=img,
                points=[(end_x, end_y)],  # Note: converting (x, y) format
                size_mm=controller_config.dish_size,
            )

            x_mm, y_mm = tip_mm[0]
            # Convert mm to meters, then add starting point (which is in meters)
            dish_size_mm = controller_config.dish_size  # 150

            # If starting_point is the CENTER of the dish, offset from center
            x_m = starting_point[0] + ((y_mm) / 1000.0)
            y_m = starting_point[1] + ((x_mm) / 1000.0)
            z_m = 0.1215

            targets.append([x_m, y_m, z_m])
            logger.debug(
                f"Plant {plant_id}: Root tip at ({x_m:.4f}, {y_m:.4f}, {z_m:.4f})"
            )
        # Step 4: Execute robotic manipulation
        logger.info(f"Starting robotic manipulation for {len(targets)} targets...")

        # Reset environment and set first target
        obs, _ = self.env.reset(seed=42)

        # Set the first target in the unwrapped environment
        if len(targets) > 0:
            unwrapped_env.target = np.array(targets[0], dtype=np.float32)

        target_idx = 0
        dropped_at_target = False
        cooldown_counter = 0

        logger.debug(f"Moving to target {target_idx}")

        for step in range(controller_config.max_steps):
            # Check if we've completed all targets
            if target_idx >= len(targets):
                logger.info(f"All targets completed at step {step}!")
                break

            target = targets[target_idx]

            # Extract current pipette position (indices 6, 7, 8 in observation)
            x, y, z = obs[6:9]
            target_x, target_y, target_z = target

            # Calculate error for each axis
            error_x = target_x - x
            error_y = target_y - y
            error_z = target_z - z

            # Calculate distance
            distance = np.sqrt(error_x**2 + error_y**2 + error_z**2)

            dt = controller_config.dt

            # Compute PID control for each axis
            control_x = pid_x.compute(error_x, dt, output_limits=(-1.0, 1.0))
            control_y = pid_y.compute(error_y, dt, output_limits=(-1.0, 1.0))
            control_z = pid_z.compute(error_z, dt, output_limits=(-1.0, 1.0))

            # State machine logic
            if not dropped_at_target:
                # Still approaching or at target, haven't dropped yet
                if distance <= controller_config.drop_threshold:
                    # Close enough - drop the droplet
                    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    dropped_at_target = True
                    cooldown_counter = 0
                    logger.debug(
                        f"Dropping at target {target_idx} (step {step}), distance: {distance:.6f}"
                    )
                    logger.debug(f"Position: ({target_x:.4f}, {target_y:.4f})")
                else:
                    # Still approaching - keep moving
                    action = np.array(
                        [control_x, control_y, control_z, 0.0], dtype=np.float32
                    )
            else:
                # Already dropped, in cooldown phase
                cooldown_counter += 1
                action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

                if cooldown_counter >= controller_config.drop_cooldown:
                    # Cooldown complete - move to next target
                    target_idx += 1
                    dropped_at_target = False
                    cooldown_counter = 0

                    # Reset PID controllers for the new target
                    pid_x.reset()
                    pid_y.reset()
                    pid_z.reset()

                    if target_idx < len(targets):
                        logger.debug(f"Moving to target {target_idx}")
                        # Update the environment's target
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
    # Initialize pipeline
    pipeline = RoboticsPipeline(
        inference_config=INFERENCE_CONFIG,
        segmentation_config=SEGMENTATION_CONFIG,
        env_config=ENV_CONFIG,
        controller_config=CONTROLLER_CONFIG,
    )

    # Run the complete pipeline
    num_drops = pipeline.run_robotic_manipulation(
        INFERENCE_CONFIG,
        SEGMENTATION_CONFIG,
        CONTROLLER_CONFIG,
        starting_point=(0.10775, 0.062),
    )

    print(f"Successfully dropped at {num_drops} root tips")


if __name__ == "__main__":
    main()
