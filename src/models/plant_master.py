from dataclasses import dataclass
from typing import Tuple
from src.models import (
    ModelConfig,
    SegmentationConfig,
    LengthCalculatorConfig,
    AdaptiveThresholdConfig,
)


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
