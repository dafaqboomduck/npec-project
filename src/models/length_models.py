from dataclasses import dataclass
import numpy as np


@dataclass
class LengthCalculatorConfig:
    num_plant: int
    max_gap: int
    max_path: int
    use_farthest_endpoint: bool


@dataclass
class PathResult:
    length: float
    path: list
    skeleton: np.ndarray | None
    start: tuple | None
    end: tuple | None


@dataclass
class LengthResult:
    detected: bool
    length: int
    pixel_count: int
