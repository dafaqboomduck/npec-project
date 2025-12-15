from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np
from .inference_models import ModelConfig


@dataclass
class SegmentationConfig:
    """Configuration for plant segmentation thresholds."""

    plant_zones: dict
    min_anchor_area: int
    min_root_area: int
    y_top_ratio: float
    y_bot_ratio: float
    anchor_bbox_size: int
    vertical_offset: int
    root_merge_distance: int


@dataclass
class Anchor:
    """Represents a detected anchor point for a plant."""

    label_id: int
    centroid: Tuple[int, int]  # (x, y)
    area: int
    mask: np.ndarray
    zone_id: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)


@dataclass
class SegmentationResult:
    """Result of root segmentation for a single image."""

    filtered_mask: np.ndarray
    labels: np.ndarray
    anchors: List[Anchor]
    zone_masks: Dict[int, np.ndarray]


@dataclass
class AdaptiveThresholdConfig:
    adaptive_theshold_start: float
    adaptive_theshold_end: float
    adaptive_theshold_step: float
    root_threshold_final: float
    shoot_overlap_penalty: float
    shoot_overlap_min_conf: float


@dataclass
class PlantMasterConfig:
    inference_config = Tuple[ModelConfig, ModelConfig]
    segmentation_config = SegmentationConfig
    adaptive_theshold_config = AdaptiveThresholdConfig
