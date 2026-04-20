from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import numpy as np
from npec_cv.models.inference import ModelConfig
from npec_cv.models.segmentation import SegmentationConfig, Anchor
from npec_cv.models.length import LengthCalculatorConfig
from npec_cv.models.adaptive_threshold import AdaptiveThresholdConfig


@dataclass
class PlantMasterConfig:
    """Master configuration for the full plant phenotyping pipeline.

    Attributes:
        inference_configs: Tuple of ModelConfigs for (root_model, anchor_model).
        segmentation_config: Configuration for plant segmentation.
        length_calculator_config: Configuration for root length calculation.
        adaptive_threshold_config: Configuration for adaptive thresholding.
    """

    inference_configs: Tuple[ModelConfig, ModelConfig]
    segmentation_config: SegmentationConfig
    length_calculator_config: LengthCalculatorConfig
    adaptive_threshold_config: AdaptiveThresholdConfig


@dataclass
class PlantResult:
    """Results for a single plant."""

    plant_id: int
    detected: bool = False
    length: Optional[int] = None
    pixel_count: Optional[int] = None
    start_point: Optional[Tuple[int, int]] = None
    end_point: Optional[Tuple[int, int]] = None
    threshold_used: Optional[float] = None
    mask: Optional[np.ndarray] = None
    skeleton: Optional[np.ndarray] = None
    path: Optional[List[Tuple[int, int]]] = None


@dataclass
class ImageResult:
    """Complete results for a single image."""

    image_name: str
    plant_results: Dict[int, PlantResult] = field(default_factory=dict)
    anchors: List[Anchor] = field(default_factory=list)
    filtered_mask: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    endpoint_info: Dict = field(default_factory=dict)

    # Raw outputs for visualization
    root_probs: Optional[np.ndarray] = None
    anchor_mask: Optional[np.ndarray] = None

    @property
    def num_detected(self) -> int:
        return sum(1 for p in self.plant_results.values() if p.detected)

    @property
    def total_plants(self) -> int:
        return len(self.plant_results)
