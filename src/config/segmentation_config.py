## VARIABLES
PLANT_ZONES = {
    0: (0, 1190),     # Plant 0: center ~940, boundary at midpoint to plant 1
    1: (1190, 1690),  # Plant 1: center ~1440
    2: (1690, 2200),  # Plant 2: center ~1940
    3: (2200, 2760),  # Plant 3: center ~2470
    4: (2760, 4000),  # Plant 4: center ~3060, extends to right edge
}
# Configurable thresholds
MIN_AREA = 200
MIN_HEIGHT_RATIO = 0.01
TOP_REGION_RATIO = 0.35
MIN_TOP_PIXELS = 30
MIN_ASPECT_RATIO = 0.3
MAX_WIDTH_RATIO = 0.15
