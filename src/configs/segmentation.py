from src.models import SegmentationConfig

PLANT_ZONES = {
    0: (890, 1390),
    1: (1390, 1890),
    2: (1890, 2400),
    3: (2400, 2900),
    4: (2900, 3400),
}

SEGMENTATION_CONFIG = SegmentationConfig(
    PLANT_ZONES,
    min_anchor_area=100,
    min_root_area=50,
    y_top_ratio=0.35,
    y_bot_ratio=0.5,
    anchor_bbox_size=150,
    bbox_tolerance=30,
    vertical_offset=10,
    root_merge_distance=30,
)
