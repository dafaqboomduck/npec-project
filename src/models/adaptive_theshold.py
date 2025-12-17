from dataclasses import dataclass


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive thresholding in root detection.

    Attributes:
        adaptive_threshold_start: Initial (high) confidence threshold (0-1).
        adaptive_threshold_end: Final (low) confidence threshold (0-1).
        adaptive_threshold_step: Step size for threshold iteration.
        shoot_overlap_penalty: Confidence penalty applied in shoot overlap regions.
        shoot_overlap_min_conf: Minimum confidence required in overlap regions.
    """

    adaptive_threshold_start: float
    adaptive_threshold_end: float
    adaptive_threshold_step: float
    shoot_overlap_penalty: float
    shoot_overlap_min_conf: float
