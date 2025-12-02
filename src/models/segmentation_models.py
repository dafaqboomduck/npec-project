from dataclasses import dataclass

@dataclass
class SegmentationConfig:
    """Configuration for plant segmentation thresholds."""
    min_area: int = 200
    min_height_ratio: float = 0.01
    top_region_ratio: float = 0.35
    min_top_pixels: int = 30
    min_aspect_ratio: float = 0.3
    max_width_ratio: float = 0.15