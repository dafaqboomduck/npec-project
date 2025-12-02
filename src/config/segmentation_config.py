from dataclasses import dataclass

PLANT_ZONES = {
    0: (0, 1190),     # Plant 0: center ~940, boundary at midpoint to plant 1
    1: (1190, 1690),  # Plant 1: center ~1440
    2: (1690, 2200),  # Plant 2: center ~1940
    3: (2200, 2760),  # Plant 3: center ~2470
    4: (2760, 4000),  # Plant 4: center ~3060, extends to right edge
}

@dataclass
class SegmentationConfig:
    """Configuration for plant segmentation thresholds."""
    min_area: int = 200
    min_height_ratio: float = 0.01
    top_region_ratio: float = 0.35
    min_top_pixels: int = 30
    min_aspect_ratio: float = 0.3
    max_width_ratio: float = 0.15
