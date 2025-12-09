from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    """Configuration for plant segmentation thresholds."""

    min_anchor_area: int = 100
    min_root_area: int = 30
    y_ratio: float = 0.25
    anchor_bbox_size: int = 80
    vertical_offset: int = 10
    root_merge_distance: int = 10
