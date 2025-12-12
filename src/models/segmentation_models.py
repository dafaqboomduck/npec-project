from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    """Configuration for plant segmentation thresholds."""

    plant_zones: dict
    min_anchor_area: int
    min_root_area: int
    y_ratio: float
    anchor_bbox_size: int
    vertical_offset: int
    root_merge_distance: int
