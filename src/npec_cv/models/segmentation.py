"""Data models and configuration dataclasses for segmentation."""


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# SEGMENTATION MODELS
# =============================================================================


@dataclass
class SegmentationConfig:
    """Configuration for plant segmentation and anchor detection.

    Attributes:
        plant_zones: Dict mapping zone_id to (x_min, x_max) pixel boundaries.
        min_anchor_area: Minimum area in pixels for valid anchor components.
        min_root_area: Minimum area in pixels for valid root components.
        y_top_ratio: Maximum y-position ratio (0-1) for anchor detection.
        y_bot_ratio: Minimum y-position ratio (0-1) for endpoint detection.
        anchor_bbox_size: Size of square bounding box around anchors (pixels).
        bbox_tolerance: Tolerance for roots outside the bounding boxes (pixels).
        vertical_offset: Vertical offset to shift anchor bbox down (pixels).
        root_merge_distance: Maximum gap in pixels to merge nearby root fragments.
    """

    plant_zones: dict
    min_anchor_area: int
    min_root_area: int
    y_top_ratio: float
    y_bot_ratio: float
    anchor_bbox_size: int
    bbox_tolerance: int
    vertical_offset: int
    root_merge_distance: int


@dataclass
class Anchor:
    """Represents a detected anchor point for a plant.

    Attributes:
        label_id: Unique identifier from connected components labeling.
        centroid: Center point coordinates as (x, y) in pixels.
        area: Total area of the anchor component in pixels.
        mask: Binary mask of the anchor component.
        zone_id: Assigned plant zone identifier, or None if unassigned.
        bbox: Bounding box as (x_min, y_min, x_max, y_max), or None if not computed.
    """

    label_id: int
    centroid: Tuple[int, int]
    area: int
    mask: np.ndarray
    zone_id: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class SegmentationResult:
    """Result of root segmentation for a single image.

    Attributes:
        filtered_mask: Binary mask containing only assigned root pixels.
        labels: Integer array where each pixel's value indicates its zone_id + 1.
        anchors: List of Anchor objects used for segmentation.
        zone_masks: Dict mapping zone_id to binary mask for that zone's roots.
    """

    filtered_mask: np.ndarray
    labels: np.ndarray
    anchors: List[Anchor]
    zone_masks: Dict[int, np.ndarray]
