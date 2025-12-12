# import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# from scipy.ndimage import find_objects
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components
# from scipy.spatial.distance import cdist

from src.models import SegmentationConfig


class PlantSegmentor:
    def __init__(
        self,
        config: Optional[SegmentationConfig] = None,
    ):
        """
        Initialize the segmentor.

        Args:
            config: SegmentationConfig object with plant zones and threshold values.
        """
        self.config = config

    def get_anchors(
        self, anchor_mask: np.ndarray, min_area: int = None, y_ratio: float = None
    ) -> list:
        """Extract anchor info using centroid. Filter by area and y-position."""
        if min_area is None:
            min_area = self.config.min_anchor_area
        if y_ratio is None:
            y_ratio = self.config.y_ratio
        self.anchors = []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            anchor_mask.astype(np.uint8), connectivity=8
        )

        max_y = int(anchor_mask.shape[0] * y_ratio)

        # Iterate through the labels starting from 1 (skipping the background)
        for label_id in range(1, num_labels):
            # Get the area of the component
            area = stats[label_id, cv2.CC_STAT_AREA]
            # Check if area meets the minimum area requirement else continue
            if area < min_area:
                continue

            # Get the x and y of the component's centroid
            centroid_x, centroid_y = int(centroids[label_id][0]), int(
                centroids[label_id][1]
            )
            # Check if the y position of the centroid meets the y position requirement else continue
            if centroid_y > max_y:
                continue

            # Create binary mask for the component
            component_mask = (labels == label_id).astype(np.uint8)

            self.anchors.append(
                {
                    "label_id": label_id,
                    "centroid": (centroid_x, centroid_y),
                    "area": area,
                    "mask": component_mask,
                }
            )
            
        return self.anchors

    def get_largest_anchor_by_zone(self, anchors: list = None) -> list:
        """Keep only the largest anchor in each zone."""
        if anchors is None:
            anchors = self.anchors
        for anchor in anchors:
            x = anchor["centroid"][0]
            anchor["zone_id"] = None
            for zone_id, (x_min, x_max) in self.config.plant_zones.items():
                if x_min <= x <= x_max:
                    anchor["zone_id"] = zone_id
                    break

        zone_anchors = {}
        for anchor in anchors:
            zone_id = anchor["zone_id"]
            if zone_id is None:
                continue
            if (
                zone_id not in zone_anchors
                or anchor["area"] > zone_anchors[zone_id]["area"]
            ):
                zone_anchors[zone_id] = anchor

        return list(zone_anchors.values())

    def create_anchor_bboxes(
        self,
        img_shape: tuple,
        anchors: list = None,
        bbox_size: int = None,
        vertical_offset: int = None,
    ) -> list:
        """Create bounding boxes centered on anchor centroid."""
        if anchors is None:
            anchors = self.anchors
        if bbox_size is None:
            bbox_size = self.config.anchor_bbox_size
        if vertical_offset is None:
            vertical_offset = self.config.vertical_offset
        half_size = bbox_size // 2

        if len(img_shape) < 2:
            raise ValueError(
                f"Expected img_shape with at least 2 elements, got {len(img_shape)}: {img_shape}"
            )
        h, w = img_shape[:2]

        for anchor in anchors:
            x, y = anchor["centroid"]
            anchor["bbox"] = (
                max(0, x - half_size),
                max(0, y - half_size + vertical_offset),
                min(w, x + half_size),
                min(h, y + half_size + vertical_offset),
            )
        return anchors

    def process_roots_by_anchor(
        self,
        root_mask: np.ndarray,
        anchors: list = None,
        min_area: int = None,
        merge_distance: int = None,
    ) -> tuple:
        """
        For each anchor zone, find root components and merge nearby fragments.
        Returns filtered_mask and final_labels.
        """
        if anchors is None:
            anchors = self.anchors
        if min_area is None:
            min_area = self.config.min_root_area
        if merge_distance is None:
            merge_distance = self.config.root_merge_distance

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            root_mask.astype(np.uint8), connectivity=8
        )

        zone_masks = {}

        for anchor in anchors:
            zone_id = anchor.get("zone_id")
            if zone_id is None:
                continue

            x_min, y_min, x_max, y_max = anchor["bbox"]
            bbox_tolerance = 30

            # Find components starting in anchor bbox
            zone_componenets = []
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue

                comp_mask = labels == label_id
                y_coords, x_coords = np.where(comp_mask)
                if len(y_coords) == 0:
                    continue

                top_y = y_coords.min()
                top_x = int(np.mean(x_coords[y_coords == top_y]))

                if (
                    x_min - bbox_tolerance <= top_x <= x_max + bbox_tolerance
                    and y_min - bbox_tolerance <= top_y <= y_max + bbox_tolerance * 2
                ):
                    zone_componenets.append(
                        {"label_id": label_id, "mask": comp_mask, "area": area}
                    )

            if not zone_componenets:
                continue

            # Start with the largest component
            largest = max(zone_componenets, key=lambda c: c["area"])
            merged_mask = largest["mask"].copy()

            # Get bounds of largest for merge distance check
            ly, lx = np.where(largest["mask"])
            ly_min, ly_max = ly.min(), ly.max()
            lx_min, lx_max = lx.min(), lx.max()

            # Merge nearby components
            for comp in zone_componenets:
                if comp["label_id"] == largest["label_id"]:
                    continue

                cy, cx = np.where(comp["mask"])
                cy_min, cy_max = cy.min(), cy.max()
                cx_min, cx_max = cx.min(), cx.max()

                y_gap = max(0, max(cy_min - ly_max, ly_min - cy_max))
                x_gap = max(0, max(cx_min - lx_max, lx_min - cx_max))

                if y_gap <= merge_distance and x_gap <= merge_distance:
                    merged_mask = merged_mask | comp["mask"]

            zone_masks[zone_id] = merged_mask

        # Create output masks
        filtered_mask = np.zeros_like(root_mask)
        final_labels = np.zeros(root_mask.shape, dtype=np.int32)

        for zone_id, mask in zone_masks.items():
            filtered_mask[mask] = 255
            final_labels[mask] = zone_id + 1

        return filtered_mask, final_labels
