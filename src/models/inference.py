from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for deep learning model inference.

    Attributes:
        checkpoint: Path to the model checkpoint file
        threshold: Confidence threshold for binary predictions
        patch_size: Size of image patches for inference (pixels)
        batch_size: Size of the batches used during inference
        custom_objects: Dictionary of custom loss/metric functions
    """

    checkpoint: str | Path
    threshold: float
    patch_size: int
    batch_size: int
    custom_objects: dict


class MaskCombineStrategy(Enum):
    """Strategy for combining masks from multiple models."""

    UNION = "union"  # OR - pixels detected by ANY model
    INTERSECTION = "intersection"  # AND - pixels detected by ALL models
