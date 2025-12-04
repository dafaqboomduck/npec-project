from dataclasses import dataclass
from enum import Enum
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a single inference model."""
    checkpoint: Path
    patch_size: int
    name: str

class MaskCombineStrategy(Enum):
    """Strategy for combining masks from multiple models."""
    UNION = "union"           # OR - pixels detected by ANY model
    INTERSECTION = "intersection"  # AND - pixels detected by ALL models

