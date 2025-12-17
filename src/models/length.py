from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class LengthCalculatorConfig:
    """Configuration for root length calculation.

    Attributes:
        num_plants: Expected number of plants in each image.
        max_gap: Maximum gap in pixels to bridge when tracing paths.
        max_path: Maximum path length to consider (prevents runaway searches).
        use_farthest_endpoint: If True, use farthest point as endpoint; else use lowest.
    """

    num_plants: int
    max_gap: int
    max_path: int
    use_farthest_endpoint: bool


@dataclass
class PathResult:
    """Result of path tracing for a single root.

    Attributes:
        length: Total path length in pixels (Euclidean distance).
        path: List of (y, x) coordinates along the traced path.
        skeleton: Skeletonized binary mask of the root, or None if not computed.
        start: Starting point (y, x) of the path, or None if not found.
        end: Ending point (y, x) of the path, or None if not found.
    """

    length: float
    path: list
    skeleton: Optional[np.ndarray]
    start: Optional[Tuple[int, int]]
    end: Optional[Tuple[int, int]]


@dataclass
class LengthResult:
    """Summary result of length measurement for a single plant.

    Attributes:
        detected: Whether a valid root was detected for this plant.
        length: Measured root length in pixels (0 if not detected).
        pixel_count: Total number of pixels in the root mask.
    """

    detected: bool
    length: int
    pixel_count: int
    path_result: Optional[PathResult] = None
