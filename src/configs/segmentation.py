from src.models import SegmentationConfig

PLANT_ZONES = {
    0: (890, 1390),
    1: (1390, 1890),
    2: (1890, 2400),
    3: (2400, 2900),
    4: (2900, 3400),
}
SEGMENTATION_CONFIG = SegmentationConfig(PLANT_ZONES)