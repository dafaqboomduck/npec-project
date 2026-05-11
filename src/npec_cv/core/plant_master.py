"""Centralized plant image processing pipeline."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from npec_cv.core.plant_segmentation import PlantSegmentation
from npec_cv.core.root_lengths import LengthCalculator
from npec_cv.exceptions import ConfigurationError
from npec_cv.helpers.extract_petri_dish import extract_petri_dish
from npec_cv.inference import InferenceManager
from npec_cv.models import (
    AdaptiveThresholdConfig,
    Anchor,
    ImageResult,
    LengthCalculatorConfig,
    ModelConfig,
    PlantResult,
    SegmentationConfig,
)

logger = logging.getLogger(__name__)


class PlantMaster:
    """Central orchestrator for plant root analysis pipeline.

    Handles:
    - Inference (root and anchor detection)
    - Shoot penalty application
    - Adaptive threshold detection
    - Root segmentation and assignment
    - Length calculation

    Can operate in two modes:
    - Full image mode (crop_to_dish=False): processes entire image
    - Dish mode (crop_to_dish=True): extracts and processes petri dish only
    """

    def __init__(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: SegmentationConfig,
        calculator_config: LengthCalculatorConfig,
        thresholds_config: AdaptiveThresholdConfig,
        plant_zones: Optional[Dict[int, Tuple[int, int]]] = None,
        crop_to_dish: bool = False,
    ):
        """Initialize PlantMaster."""
        self._deserialize_configs(
            inference_config,
            segmentation_config,
            calculator_config,
            thresholds_config,
            plant_zones,
        )
        self.crop_to_dish = crop_to_dish
        self._init_components()
        self.num_plants = len(self.plant_zones)
        logger.debug(f"PlantMaster initialized for {self.num_plants} plant zones " f"(crop_to_dish={crop_to_dish})")

    def _deserialize_configs(
        self,
        inference_config: Tuple[ModelConfig, ModelConfig],
        segmentation_config: SegmentationConfig,
        calculator_config: LengthCalculatorConfig,
        thresholds_config: AdaptiveThresholdConfig,
        plant_zones: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> None:
        self.inference_config = inference_config
        self.segmentation_config = segmentation_config
        self.calculator_config = calculator_config
        self.thresholds_config = thresholds_config
        self.plant_zones = plant_zones if plant_zones is not None else segmentation_config.plant_zones

    def _init_components(self) -> None:
        """Initialize all pipeline components with error handling."""
        logger.debug("Initializing pipeline components...")
        try:
            try:
                self.root_manager = InferenceManager(self.inference_config[0])
                logger.debug("✓ Root model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Root model. Exception: {e}")
                raise ConfigurationError

            try:
                self.anchor_manager = InferenceManager(self.inference_config[1])
                logger.debug("✓ Anchor model loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Anchor model. Exception: {e}")
                raise ConfigurationError

            try:
                self.segmentor = PlantSegmentation(self.segmentation_config)
                logger.debug("✓ Segmentor instance loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Segmentor. Exception: {e}")
                raise ConfigurationError

            try:
                self.calculator = LengthCalculator(self.calculator_config)
                logger.debug("✓ Calculator instance loaded")
            except Exception as e:
                logger.exception(f"⚠ Failed to load Calculator. Exception: {e}")
                raise ConfigurationError

            logger.debug("Pipeline Components initialized successfully!")
        except Exception as e:
            logger.exception(
                f"UNEXPECTED ERROR: Encountered an unexpected error while "
                f"initializing the pipeline components. Exception: {e}"
            )
            raise ConfigurationError

    def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """Preprocess image based on crop mode.

        Returns:
            Tuple of (processed_image, crop_offset)
            crop_offset is (y_offset, x_offset) if cropped, None otherwise
        """
        if self.crop_to_dish:
            dish = extract_petri_dish(img)
            # Calculate offset for coordinate mapping if needed later
            # Note: extract_petri_dish returns just the cropped region
            # If you need the offset, you may need to modify extract_petri_dish
            # to return it, or calculate it here
            return dish, None  # For now, offset tracking not needed since we stay in dish coords
        return img, None

    def _run_inference(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run both root and anchor inference models.

        Returns:
            Tuple of (root_probabilities, anchor_probabilities)
        """
        # Root inference - keep as probabilities
        root_patches = self.root_manager.patch_image(
            img, patch_size=self.inference_config[0].patch_size, normalize=True
        )
        root_padded = self.root_manager.create_img_mask(root_patches)
        root_probs = self.root_manager.restore_mask(root_padded, patch_size=self.inference_config[0].patch_size)

        # Anchor inference
        anchor_patches = self.anchor_manager.patch_image(
            img, patch_size=self.inference_config[1].patch_size, normalize=True
        )
        anchor_padded = self.anchor_manager.create_img_mask(anchor_patches)
        anchor_probs = self.anchor_manager.restore_mask(anchor_padded, patch_size=self.inference_config[1].patch_size)

        # If in crop mode, extract dish region from restored masks
        if self.crop_to_dish:
            if self.root_manager.crop_coords is not None:
                x1, y1, x2, y2 = self.root_manager.crop_coords
                root_probs = root_probs[y1:y2, x1:x2]
            if self.anchor_manager.crop_coords is not None:
                x1, y1, x2, y2 = self.anchor_manager.crop_coords
                anchor_probs = anchor_probs[y1:y2, x1:x2]

        return root_probs, anchor_probs

    def _apply_shoot_penalty(self, root_probs: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
        """Apply confidence penalty to root predictions overlapping with shoots."""
        adjusted = root_probs.copy()
        overlap = anchor_mask > 0

        penalty = self.thresholds_config.shoot_overlap_penalty
        min_conf = self.thresholds_config.shoot_overlap_min_conf

        adjusted[overlap] = np.where(root_probs[overlap] >= min_conf, root_probs[overlap] - penalty, 0)
        return np.clip(adjusted, 0, 1)

    def _find_endpoints_at_threshold(
        self,
        probs: np.ndarray,
        anchors: List[Anchor],
        threshold: float,
    ) -> Dict[int, Dict]:
        """Find root start/end points at a specific threshold."""
        results = {zid: {"start": None, "end": None, "detected": False} for zid in self.plant_zones.keys()}

        binary_mask = (probs > threshold).astype(np.uint8) * 255

        for anchor in anchors:
            if anchor.zone_id is None or anchor.bbox is None:
                continue

            zid = anchor.zone_id
            x_min, y_min, x_max, y_max = anchor.bbox
            zone_x_min, zone_x_max = self.plant_zones[zid]

            zone_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            zone_mask[:, zone_x_min:zone_x_max] = binary_mask[:, zone_x_min:zone_x_max]

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zone_mask, connectivity=8)

            best_comp = None
            best_score = -1
            bbox_tol = self.segmentation_config.bbox_tolerance

            for lid in range(1, num_labels):
                area = stats[lid, cv2.CC_STAT_AREA]
                comp_mask = labels == lid
                ys, xs = np.where(comp_mask)

                if len(ys) == 0:
                    continue

                top_y = ys.min()
                top_x = int(np.mean(xs[ys == top_y]))

                if x_min - bbox_tol <= top_x <= x_max + bbox_tol and y_min - bbox_tol <= top_y <= y_max + bbox_tol * 2:
                    ax, ay = anchor.centroid
                    dist = np.sqrt((top_x - ax) ** 2 + (top_y - ay) ** 2)
                    score = area / (dist + 1)

                    if score > best_score:
                        best_score = score
                        best_comp = {
                            "mask": comp_mask,
                            "start": (top_y, top_x),
                            "area": area,
                        }

            if best_comp is None:
                continue

            ys, xs = np.where(best_comp["mask"])
            y_range = ys.max() - ys.min()
            bottom_thresh = ys.min() + y_range * (1 - self.segmentation_config.y_bot_ratio)

            bottom_ys = ys[ys >= bottom_thresh]
            bottom_xs = xs[ys >= bottom_thresh]

            end_y = bottom_ys.max()
            end_x = int(np.mean(bottom_xs[bottom_ys == end_y]))

            results[zid] = {
                "start": best_comp["start"],
                "end": (end_y, end_x),
                "detected": True,
                "component_mask": best_comp["mask"],
            }

        return results

    def detect_endpoints_adaptive(
        self,
        root_probs: np.ndarray,
        anchors: List[Anchor],
    ) -> Tuple[float, Dict[int, Dict], bool]:
        """Iteratively lower threshold until all plants have endpoints."""
        cfg = self.thresholds_config
        thresholds = np.arange(
            cfg.adaptive_threshold_start,
            cfg.adaptive_threshold_end - cfg.adaptive_threshold_step / 2,
            -cfg.adaptive_threshold_step,
        )

        for thresh in thresholds:
            results = self._find_endpoints_at_threshold(root_probs, anchors, thresh)
            found_count = sum(1 for r in results.values() if r["detected"])

            logger.debug(f"Threshold {thresh:.2f}: {found_count}/{self.num_plants} detected")

            if found_count == self.num_plants:
                return thresh, results, True

        results = self._find_endpoints_at_threshold(root_probs, anchors, cfg.adaptive_threshold_end)
        found_count = sum(1 for r in results.values() if r["detected"])

        return cfg.adaptive_threshold_end, results, found_count == self.num_plants

    def _segment_roots(
        self,
        root_probs: np.ndarray,
        endpoint_info: Dict[int, Dict],
        optimal_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment roots using endpoint info and merge components."""
        root_mask = (root_probs > optimal_threshold).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(root_mask, connectivity=8)

        filtered_mask = np.zeros_like(root_mask)
        final_labels = np.zeros(root_mask.shape, dtype=np.int32)

        min_area = self.segmentation_config.min_root_area
        merge_dist = self.segmentation_config.root_merge_distance

        for zid, ep_data in endpoint_info.items():
            if not ep_data["detected"]:
                continue

            start_y, start_x = ep_data["start"]
            zone_x_min, zone_x_max = self.plant_zones[zid]

            zone_components = []
            for lid in range(1, num_labels):
                area = stats[lid, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue

                comp_mask = labels == lid
                ys, xs = np.where(comp_mask)

                if not (xs.min() < zone_x_max and xs.max() > zone_x_min):
                    continue

                start_region = comp_mask[
                    max(0, start_y - 30) : start_y + 50,
                    max(0, start_x - 30) : min(comp_mask.shape[1], start_x + 30),
                ]
                if start_region.any():
                    zone_components.append({"mask": comp_mask, "area": area})

            if not zone_components:
                if "component_mask" in ep_data:
                    merged = ep_data["component_mask"]
                else:
                    continue
            else:
                merged = zone_components[0]["mask"].copy()
                for comp in zone_components[1:]:
                    merged = merged | comp["mask"]

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_dist // 2, merge_dist // 2))
            dilated = cv2.dilate(merged.astype(np.uint8), kernel, iterations=1)

            for lid in range(1, num_labels):
                comp_mask = labels == lid
                if np.any(dilated & comp_mask.astype(np.uint8)):
                    ys, xs = np.where(comp_mask)
                    if xs.min() < zone_x_max and xs.max() > zone_x_min:
                        merged = merged | comp_mask

            filtered_mask[merged] = 255
            final_labels[merged] = zid + 1

            ys, xs = np.where(merged)
            if len(ys) > 0:
                end_y = ys.max()
                end_x = int(np.mean(xs[ys == end_y]))
                endpoint_info[zid]["end"] = (end_y, end_x)
                endpoint_info[zid]["threshold_used"] = optimal_threshold

        return filtered_mask, final_labels

    def process_image(
        self,
        img: np.ndarray,
        image_name: str = "unknown",
    ) -> ImageResult:
        """Process a single image through the complete pipeline.

        Args:
            img: BGR image array
            image_name: Name for logging/results

        Returns:
            ImageResult with all detection and measurement data

        Note:
            If crop_to_dish=True, all coordinates in the result are
            relative to the cropped dish region.
        """
        logger.debug(f"Processing: {image_name} (crop_to_dish={self.crop_to_dish})")

        # Preprocess (crop if needed)
        processed_img, _ = self._preprocess_image(img)

        # Initialize result
        result = ImageResult(
            image_name=image_name,
            plant_results={i: PlantResult(plant_id=i) for i in range(self.num_plants)},
        )

        # Step 1: Run inference
        logger.debug("Running inference...")
        root_probs, anchor_probs = self._run_inference(processed_img)
        result.root_probs = root_probs
        anchor_mask = (anchor_probs > self.inference_config[1].threshold).astype(np.uint8) * 255
        result.anchor_mask = anchor_mask

        # Step 2: Extract anchors
        logger.debug("Extracting anchors...")
        anchors = self.segmentor.extract_anchors(anchor_mask)
        anchors = self.segmentor.create_anchor_bboxes(
            anchors,
            processed_img.shape,
            self.segmentation_config.anchor_bbox_size,
            self.segmentation_config.vertical_offset,
        )
        result.anchors = anchors
        logger.debug(f"  Found {len(anchors)} anchors")

        # Step 3: Apply shoot penalty
        logger.debug("Applying shoot penalty...")
        adjusted_probs = self._apply_shoot_penalty(root_probs, (anchor_mask > 0).astype(np.uint8))

        # Step 4: Adaptive endpoint detection
        logger.debug("Detecting endpoints adaptively...")
        optimal_thresh, endpoint_info, all_found = self.detect_endpoints_adaptive(adjusted_probs, anchors)
        result.endpoint_info = endpoint_info
        logger.debug(f"  Optimal threshold: {optimal_thresh:.2f}, All found: {all_found}")

        # Step 5: Segment roots
        logger.debug("Segmenting roots...")
        filtered_mask, labels = self._segment_roots(adjusted_probs, endpoint_info, optimal_thresh)
        result.filtered_mask = filtered_mask
        result.labels = labels

        # Step 6: Calculate lengths
        logger.debug("Calculating lengths...")
        length_results = self.calculator.analyze_plant_lengths(
            labels,
            num_plants=self.num_plants,
            max_gap=self.calculator_config.max_gap,
            max_path=self.calculator_config.max_path,
            endpoint_info=endpoint_info,
        )

        # Populate plant results
        for pid in range(self.num_plants):
            pr = result.plant_results[pid]

            if pid in length_results:
                lr = length_results[pid]
                pr.detected = lr.detected
                pr.length = lr.length if lr.detected else None
                pr.pixel_count = lr.pixel_count if lr.detected else None
                if lr.path_result:
                    pr.skeleton = lr.path_result.skeleton
                    pr.path = lr.path_result.path

            if pid in endpoint_info and endpoint_info[pid]["detected"]:
                ep = endpoint_info[pid]
                pr.start_point = ep["start"]
                pr.end_point = ep["end"]
                pr.threshold_used = ep.get("threshold_used")
                pr.mask = labels == pid + 1

        # Log summary
        for pid, pr in result.plant_results.items():
            status = "✓" if pr.detected else "✗"
            length_str = f"{pr.length}px" if pr.length else "N/A"
            thresh_str = f" (t={pr.threshold_used:.2f})" if pr.threshold_used else ""
            logger.debug(f"  Plant {pid}: {status} {length_str}{thresh_str}")

        return result
