from src.models import SegmentationConfig

PLANT_ZONES = {
    0: (0, 1190),     # Plant 0: center ~940, boundary at midpoint to plant 1
    1: (1190, 1690),  # Plant 1: center ~1440
    2: (1690, 2200),  # Plant 2: center ~1940
    3: (2200, 2760),  # Plant 3: center ~2470
    4: (2760, 4000),  # Plant 4: center ~3060, extends to right edge
}
SEGMENTATION_CONFIG = SegmentationConfig()