"""Module for plant root segmentation and anchor extraction."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.npec_cv.exceptions import ConfigurationError
from src.npec_cv.models import Anchor, SegmentationConfig, SegmentationResult

logger = logging.getLogger(__name__)


class PlantSegmentation:
    """Handles anchor detection and root segmentation for plant images.

    This class processes probability maps from inference to identify
    individual plants based on anchor points (e.g., seed locations)
    and assigns root components to each plant zone.
    """

    def __init__(self, segmentation_config: SegmentationConfig):
        """Initialize the segmentor.

        Args:
            segmentation_config: Configuration with plant zones and threshold values.

        Raises:
            ConfigurationError: If config is None or missing required fields.
        """
        if segmentation_config is None:
            raise ConfigurationError("segmentation_config is required")
        if not segmentation_config.plant_zones:
            raise ConfigurationError("plant_zones must be defined in config")

        self.config = segmentation_config
        logger.debug(f"PlantSegmentation initialized with {len(self.config.plant_zones)} zones")

    def _detect_anchor_components(
        self,
        anchor_mask: np.ndarray,
        min_area: Optional[int] = None,
        max_y_ratio: Optional[float] = None,
    ) -> List[Anchor]:
        """Detect anchor components from binary mask.

        Finds connected components in the anchor mask and filters by
        minimum area and vertical position (anchors should be near top).

        Args:
            anchor_mask: Binary mask of anchor predictions.
            min_area: Minimum component area in pixels.
            max_y_ratio: Maximum y-position as ratio of image height (0-1).

        Returns:
            List of detected Anchor objects.
        """
        min_area = min_area or self.config.min_anchor_area
        max_y_ratio = max_y_ratio or self.config.y_top_ratio

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            anchor_mask.astype(np.uint8), connectivity=8
        )

        max_y = int(anchor_mask.shape[0] * max_y_ratio)
        anchors = []

        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            cx, cy = int(centroids[label_id][0]), int(centroids[label_id][1])
            if cy > max_y:
                continue

            component_mask = (labels == label_id).astype(np.uint8)

            anchors.append(
                Anchor(
                    label_id=label_id,
                    centroid=(cx, cy),
                    area=area,
                    mask=component_mask,
                )
            )

        logger.debug(f"Detected {len(anchors)} anchor components (min_area={min_area})")
        return anchors

    def _assign_anchors_to_zones(self, anchors: List[Anchor]) -> List[Anchor]:
        """Assign each anchor to a plant zone based on x-coordinate.

        Each anchor is assigned to the zone containing its centroid.
        Only the largest anchor per zone is kept.

        Args:
            anchors: List of detected anchors.

        Returns:
            List of anchors with zone_id set (one per zone, largest only).
        """
        # Assign zone IDs based on x-coordinate
        for anchor in anchors:
            x = anchor.centroid[0]
            for zone_id, (x_min, x_max) in self.config.plant_zones.items():
                if x_min <= x <= x_max:
                    anchor.zone_id = zone_id
                    break

        # Keep only largest anchor per zone
        zone_anchors: Dict[int, Anchor] = {}
        for anchor in anchors:
            if anchor.zone_id is None:
                continue
            if anchor.zone_id not in zone_anchors or anchor.area > zone_anchors[anchor.zone_id].area:
                zone_anchors[anchor.zone_id] = anchor

        result = list(zone_anchors.values())
        logger.debug(f"Assigned anchors to {len(result)} zones: " f"{[a.zone_id for a in result]}")
        return result

    def create_anchor_bboxes(
        self,
        anchors: List[Anchor],
        img_shape: Tuple[int, ...],
        bbox_size: Optional[int] = None,
        vertical_offset: Optional[int] = None,
    ) -> List[Anchor]:
        """Create bounding boxes centered on anchor centroids.

        Args:
            anchors: List of anchors to add bboxes to.
            img_shape: Image dimensions (H, W, ...) for boundary clamping.
            bbox_size: Size of square bounding box.
            vertical_offset: Vertical offset to shift bbox down.

        Returns:
            Same anchors with bbox field populated.

        Raises:
            ValueError: If img_shape has fewer than 2 dimensions.
        """
        if len(img_shape) < 2:
            raise ValueError(f"img_shape must have at least 2 dims, got {img_shape}")

        bbox_size = bbox_size or self.config.anchor_bbox_size
        vertical_offset = vertical_offset or self.config.vertical_offset
        half = bbox_size // 2
        h, w = img_shape[:2]

        for anchor in anchors:
            x, y = anchor.centroid
            anchor.bbox = (
                max(0, x - half),
                max(0, y - half + vertical_offset),
                min(w, x + half),
                min(h, y + half + vertical_offset),
            )

        return anchors

    def extract_anchors(
        self,
        anchor_mask: np.ndarray,
    ) -> List[Anchor]:
        """Full anchor extraction pipeline: detect, assign zones, create bboxes.

        Args:
            anchor_mask: Binary mask of anchor predictions.
            img_shape: Original image shape for bbox boundary clamping.

        Returns:
            List of Anchor objects with zone_id and bbox populated.
        """
        anchors = self._detect_anchor_components(anchor_mask)
        anchors = self._assign_anchors_to_zones(anchors)

        logger.debug(f"Extracted {len(anchors)} anchors for zones: {[a.zone_id for a in anchors]}")
        return anchors

    def segment_roots(
        self,
        root_mask: np.ndarray,
        anchors: List[Anchor],
        min_area: Optional[int] = None,
        merge_distance: Optional[int] = None,
        bbox_tolerance: int = 30,
    ) -> SegmentationResult:
        """Segment roots by assigning components to anchor zones.

        For each anchor zone, finds root components that start within the
        anchor's bounding box, then merges nearby fragments.

        Args:
            root_mask: Binary mask of root predictions.
            anchors: List of anchors with bbox and zone_id set.
            min_area: Minimum root component area.
            merge_distance: Maximum gap to merge nearby components.
            bbox_tolerance: Tolerance around bbox for component matching.

        Returns:
            SegmentationResult with filtered mask, labels, and zone masks.
        """
        min_area = min_area or self.config.min_root_area
        merge_distance = merge_distance or self.config.root_merge_distance

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(root_mask.astype(np.uint8), connectivity=8)

        zone_masks: Dict[int, np.ndarray] = {}

        for anchor in anchors:
            if anchor.zone_id is None or anchor.bbox is None:
                continue

            x_min, y_min, x_max, y_max = anchor.bbox

            # Find components starting in anchor bbox
            zone_components = []
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue

                comp_mask = labels == label_id
                y_coords, x_coords = np.where(comp_mask)
                if len(y_coords) == 0:
                    continue

                # Get top-most point of component
                top_y = y_coords.min()
                top_x = int(np.mean(x_coords[y_coords == top_y]))

                # Check if top point is within anchor bbox (with tolerance)
                if (
                    x_min - bbox_tolerance <= top_x <= x_max + bbox_tolerance
                    and y_min - bbox_tolerance <= top_y <= y_max + bbox_tolerance * 2
                ):
                    zone_components.append(
                        {
                            "label_id": label_id,
                            "mask": comp_mask,
                            "area": area,
                        }
                    )

            if not zone_components:
                logger.debug(f"No root components found for zone {anchor.zone_id}")
                continue

            # Start with largest component
            largest = max(zone_components, key=lambda c: c["area"])
            merged_mask = largest["mask"].copy()

            # Get bounds for merge distance calculation
            ly, lx = np.where(largest["mask"])
            ly_min, ly_max = ly.min(), ly.max()
            lx_min, lx_max = lx.min(), lx.max()

            # Merge nearby components
            for comp in zone_components:
                if comp["label_id"] == largest["label_id"]:
                    continue

                cy, cx = np.where(comp["mask"])
                cy_min, cy_max = cy.min(), cy.max()
                cx_min, cx_max = cx.min(), cx.max()

                y_gap = max(0, max(cy_min - ly_max, ly_min - cy_max))
                x_gap = max(0, max(cx_min - lx_max, lx_min - cx_max))

                if y_gap <= merge_distance and x_gap <= merge_distance:
                    merged_mask = merged_mask | comp["mask"]
                    logger.debug(f"Merged component {comp['label_id']} into zone {anchor.zone_id}")

            zone_masks[anchor.zone_id] = merged_mask

        # Build output masks
        filtered_mask = np.zeros_like(root_mask)
        final_labels = np.zeros(root_mask.shape, dtype=np.int32)

        for zone_id, mask in zone_masks.items():
            filtered_mask[mask] = 255
            final_labels[mask] = zone_id + 1

        logger.debug(f"Segmented roots for {len(zone_masks)} zones")

        return SegmentationResult(
            filtered_mask=filtered_mask,
            labels=final_labels,
            anchors=anchors,
            zone_masks=zone_masks,
        )
