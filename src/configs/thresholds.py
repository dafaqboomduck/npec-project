from src.models import AdaptiveThresholdConfig

ADAPTIVE_THRESHOLD_CONFIG = AdaptiveThresholdConfig(
    adaptive_threshold_start=0.50,
    adaptive_threshold_end=0.05,
    adaptive_threshold_step=0.05,
    root_threshold_final=0.07,
    shoot_overlap_penalty=0.4,
    shoot_overlap_min_conf=0.5,
)
