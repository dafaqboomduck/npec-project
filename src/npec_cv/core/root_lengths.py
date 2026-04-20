import logging
from collections import deque
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from npec_cv.exceptions import ConfigurationError

from npec_cv.configs.lengths import (
    EMPTY_RESULT,
    RETURN_EMPTY_MASK,
    RETURN_INSUFFICIENT_PIXELS,
    RETURN_NO_PATH,
    RETURN_SUCCESS,
)
from npec_cv.models.length import LengthCalculatorConfig, LengthResult, PathResult

logger = logging.getLogger(__name__)

# Type aliases for clarity
Point = tuple[int, int]
PixelSet = set[Point]


class LengthCalculator:
    """Calculates geodesic lengths of plant roots from binary masks."""

    # 8-connectivity neighbor offsets
    _NEIGHBOR_OFFSETS = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    def __init__(self, config: LengthCalculatorConfig):
        self.config = config
        if config is None:
            raise ConfigurationError("config is required")

    def _get_neighbors(self, point: Point, pixel_set: PixelSet) -> list[Point]:
        """Get all valid 8-connected neighbors of a point."""
        y, x = point
        return [
            (y + dy, x + dx)
            for dy, dx in self._NEIGHBOR_OFFSETS
            if (y + dy, x + dx) in pixel_set
        ]

    def _find_endpoints(self, skeleton: np.ndarray) -> list[Point]:
        """Find endpoint pixels in a skeleton (pixels with exactly 1 neighbor)."""
        pixels = np.argwhere(skeleton)
        pixel_set: PixelSet = set(map(tuple, pixels))
        return [
            (y, x)
            for y, x in pixels
            if len(self._get_neighbors((y, x), pixel_set)) == 1
        ]

    def _are_connected(
        self, pixel_set: PixelSet, pt1: Point, pt2: Point, max_depth: int
    ) -> bool:
        """Check if two points are connected within max_depth steps using BFS."""
        if pt1 not in pixel_set or pt2 not in pixel_set:
            return False
        if pt1 == pt2:
            return True

        visited = {pt1}
        queue = deque([(pt1, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self._get_neighbors(current, pixel_set):
                if neighbor == pt2:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return False

    def _draw_line(self, img: np.ndarray, pt1: Point, pt2: Point) -> np.ndarray:
        """Draw a line between two points using Bresenham's algorithm."""
        y0, x0 = int(pt1[0]), int(pt1[1])
        y1, x1 = int(pt2[0]), int(pt2[1])
        h, w = img.shape

        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy

        while True:
            if 0 <= y0 < h and 0 <= x0 < w:
                img[y0, x0] = True
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return img

    def _find_nearest_skeleton_point(
        self, pixels: np.ndarray, target: Point, max_search: float
    ) -> Optional[Point]:
        """Find the nearest skeleton pixel to a target point within max_search radius."""
        if len(pixels) == 0:
            return None

        ty, tx = target
        distances = np.hypot(pixels[:, 0] - ty, pixels[:, 1] - tx)
        min_idx = np.argmin(distances)

        if distances[min_idx] <= max_search:
            return tuple(pixels[min_idx])
        return None

    def _find_farthest_point_from(
        self, G: nx.Graph, start_node: Point
    ) -> tuple[Point, float]:
        """Find the farthest reachable point from start using Dijkstra's algorithm."""
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, start_node, weight="weight"
            )
            if not lengths:
                return start_node, 0.0
            farthest_node, distance = max(lengths.items(), key=lambda x: x[1])
            return farthest_node, distance
        except nx.NetworkXError:
            return start_node, 0.0

    def _build_skeleton_graph(self, pixels: np.ndarray) -> nx.Graph:
        """Build a weighted graph from skeleton pixels with 8-connectivity."""
        G = nx.Graph()
        pixel_set: PixelSet = set(map(tuple, pixels))

        for y, x in pixels:
            node = (y, x)
            G.add_node(node)
            for dy, dx in self._NEIGHBOR_OFFSETS:
                neighbor = (y + dy, x + dx)
                if neighbor in pixel_set:
                    dist = np.hypot(dy, dx)  # 1.0 for cardinal, ~1.414 for diagonal
                    G.add_edge(node, neighbor, weight=dist)
        return G

    def _resolve_endpoint(
        self,
        pixels: np.ndarray,
        valid_nodes: set[Point],
        sorted_by_y: list[Point],
        provided_point: Optional[Point],
        max_gap: float,
        use_top: bool,
    ) -> Point:
        """Resolve an endpoint from a provided point or default to top/bottom."""
        if provided_point is not None:
            nearest = self._find_nearest_skeleton_point(pixels, provided_point, max_gap)
            if nearest is not None and nearest in valid_nodes:
                return nearest
        return sorted_by_y[0] if use_top else sorted_by_y[-1]

    def _expand_skeleton(self, skel_crop, full_shape, y_min, y_max, x_min, x_max):
        if skel_crop is None:
            return None
        full = np.zeros(full_shape, dtype=bool)
        full[y_min : y_max + 1, x_min : x_max + 1] = skel_crop
        return full

    def _find_alternative_endpoint(
        self,
        skeleton: np.ndarray,
        valid_nodes: set[Point],
        start_node: Point,
        sorted_by_y: list[Point],
    ) -> Point:
        """Find an alternative endpoint when start equals end."""
        endpoints = self._find_endpoints(skeleton)
        candidates = [e for e in endpoints if e in valid_nodes and e != start_node]

        if candidates:
            return max(
                candidates,
                key=lambda e: np.hypot(e[0] - start_node[0], e[1] - start_node[1]),
            )

        return sorted_by_y[-1] if sorted_by_y[-1] != start_node else sorted_by_y[0]

    def _process_single_plant(
        self,
        labels: np.ndarray,
        label_id: int,
        plant_id: int,
        max_gap: int,
        max_path: int,
        endpoint_info: Optional[dict],
    ) -> tuple[LengthResult, int]:
        """Process a single plant and return its length result."""
        y_idx, x_idx = np.where(labels == label_id)
        if len(y_idx) == 0:
            logger.warning(f"Plant {plant_id}: No pixels found")
            return EMPTY_RESULT, RETURN_EMPTY_MASK

        y_min, y_max = y_idx.min(), y_idx.max()
        x_min, x_max = x_idx.min(), x_idx.max()
        pixel_count = len(y_idx)

        mask_crop = labels[y_min : y_max + 1, x_min : x_max + 1] == label_id

        # Convert provided endpoints to crop coordinates
        start_point, end_point = self._get_crop_endpoints(
            endpoint_info, plant_id, y_min, x_min
        )

        path_result, return_code = self.calculate_geodesic_length(
            mask_crop,
            start_point=start_point,
            end_point=end_point,
            max_gap=max_gap,
            max_path=max_path,
        )

        # Convert path_result to full image coordinates
        full_path_result = PathResult(
            length=path_result.length,
            path=(
                [(y + y_min, x + x_min) for y, x in path_result.path]
                if path_result.path
                else []
            ),
            skeleton=self._expand_skeleton(
                path_result.skeleton, labels.shape, y_min, y_max, x_min, x_max
            ),
            start=(
                (path_result.start[0] + y_min, path_result.start[1] + x_min)
                if path_result.start
                else None
            ),
            end=(
                (path_result.end[0] + y_min, path_result.end[1] + x_min)
                if path_result.end
                else None
            ),
        )

        self._log_result(plant_id, path_result, return_code)

        return (
            LengthResult(
                detected=True,
                length=int(path_result.length),
                pixel_count=pixel_count,
                path_result=full_path_result,
            ),
            return_code,
        )

    def _get_crop_endpoints(
        self,
        endpoint_info: Optional[dict],
        plant_id: int,
        y_offset: int,
        x_offset: int,
    ) -> tuple[Optional[Point], Optional[Point]]:
        """Extract and convert endpoint coordinates to crop-relative coordinates."""
        if not endpoint_info or plant_id not in endpoint_info:
            return None, None

        ep = endpoint_info[plant_id]
        if not ep.get("detected"):
            return None, None

        start = end = None
        if ep.get("start"):
            start = (ep["start"][0] - y_offset, ep["start"][1] - x_offset)
        if ep.get("end"):
            end = (ep["end"][0] - y_offset, ep["end"][1] - x_offset)
        return start, end

    def _log_result(self, plant_id: int, result: PathResult, return_code: int) -> None:
        """Log the result of a plant length calculation."""
        messages = {
            RETURN_SUCCESS: f"Plant {plant_id}: length={result.length:.2f}px, path_nodes={len(result.path)}",
            RETURN_EMPTY_MASK: f"Plant {plant_id}: Empty mask",
            RETURN_INSUFFICIENT_PIXELS: f"Plant {plant_id}: Insufficient pixels, estimate={result.length}",
            RETURN_NO_PATH: f"Plant {plant_id}: No path found between {result.start} and {result.end}",
        }
        level = logging.DEBUG if return_code == RETURN_SUCCESS else logging.WARNING
        logger.log(
            level,
            messages.get(return_code, f"Plant {plant_id}: Unknown error {return_code}"),
        )

    def bridge_skeleton_gaps(
        self, skeleton: np.ndarray, max_gap: int, max_path: int
    ) -> np.ndarray:
        """Bridge small gaps in a skeleton by connecting nearby endpoints."""
        skeleton = skeleton.copy().astype(bool)
        endpoints = self._find_endpoints(skeleton)

        if len(endpoints) < 2:
            return skeleton

        pixel_set: PixelSet = set(map(tuple, np.argwhere(skeleton)))
        tree = cKDTree(endpoints)

        pairs_to_bridge = []
        used: set[int] = set()

        for i, pt in enumerate(endpoints):
            if i in used:
                continue
            distances, indices = tree.query(
                pt, k=min(5, len(endpoints)), distance_upper_bound=max_gap
            )

            for dist, j in zip(distances, indices):
                if j >= len(endpoints) or i == j or j in used:
                    continue
                if 0 < dist <= max_gap:
                    if not self._are_connected(
                        pixel_set, endpoints[i], endpoints[j], max_path
                    ):
                        pairs_to_bridge.append((endpoints[i], endpoints[j], dist))
                        used.update([i, j])
                        break

        # Bridge closest pairs first
        for pt1, pt2, _ in sorted(pairs_to_bridge, key=lambda x: x[2]):
            skeleton = self._draw_line(skeleton, pt1, pt2)

        return skeleton

    def calculate_geodesic_length(
        self,
        binary_mask: np.ndarray,
        start_point: Optional[Point] = None,
        end_point: Optional[Point] = None,
        use_farthest_endpoint: bool = True,
        max_gap: int | None = None,
        max_path: int | None = None,
    ) -> tuple[PathResult, int]:
        """
        Calculate the geodesic length of a root by skeletonizing and finding the longest path.

        Args:
            binary_mask: Binary array where root pixels are True/1.
            start_point: Optional (y, x) start location. Defaults to topmost point.
            end_point: Optional (y, x) end location. Defaults to farthest/bottommost.
            use_farthest_endpoint: If True and end_point is None, find farthest point from start.
            max_gap: Maximum gap distance (pixels) to bridge in skeleton.
            max_path: Maximum path length to check for existing connections when bridging.

        Returns:
            Tuple of (PathResult, return_code).
        """
        empty_result = PathResult(0.0, [], None, None, None)

        if np.sum(binary_mask) == 0:
            return empty_result, RETURN_EMPTY_MASK

        skeleton = skeletonize(binary_mask)
        skeleton_bridged = self.bridge_skeleton_gaps(skeleton, max_gap, max_path)
        pixels = np.argwhere(skeleton_bridged)

        if len(pixels) < 2:
            result = PathResult(float(len(pixels)), [], skeleton_bridged, None, None)
            return result, RETURN_INSUFFICIENT_PIXELS

        G = self._build_skeleton_graph(pixels)

        # Use largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        valid_nodes = set(G.nodes)
        sorted_by_y = sorted(valid_nodes, key=lambda p: p[0])

        # Determine start node
        start_node = self._resolve_endpoint(
            pixels, valid_nodes, sorted_by_y, start_point, max_gap, use_top=True
        )

        # Determine end node
        if end_point is not None:
            end_node = self._resolve_endpoint(
                pixels, valid_nodes, sorted_by_y, end_point, max_gap, use_top=False
            )
        elif use_farthest_endpoint:
            end_node, _ = self._find_farthest_point_from(G, start_node)
        else:
            end_node = sorted_by_y[-1]

        # Ensure start and end differ
        if start_node == end_node:
            end_node = self._find_alternative_endpoint(
                skeleton_bridged, valid_nodes, start_node, sorted_by_y
            )

        try:
            length, path = nx.single_source_dijkstra(
                G, source=start_node, target=end_node, weight="weight"
            )
            return (
                PathResult(length, path, skeleton_bridged, start_node, end_node),
                RETURN_SUCCESS,
            )
        except nx.NetworkXNoPath:
            return (
                PathResult(0.0, [], skeleton_bridged, start_node, end_node),
                RETURN_NO_PATH,
            )

    def analyze_plant_lengths(
        self,
        labels: np.ndarray,
        num_plants: int = 5,
        max_gap: int | None = None,
        max_path: int | None = None,
        endpoint_info: Optional[dict] = None,
    ) -> dict[int, LengthResult]:
        """
        Calculate lengths for all labeled plant components.

        Args:
            labels: 2D array with labeled components (0 = background).
            num_plants: Expected number of plants.
            max_gap: Maximum gap to bridge in skeleton.
            max_path: Maximum path length for connection checks.
            endpoint_info: Optional dict mapping plant_id to {'start': (y,x), 'end': (y,x)}.

        Returns:
            Dict mapping plant_id (0 to num_plants-1) to LengthResult.
        """
        logger.debug(f"Starting plant length analysis for {num_plants} plants")
        results: dict[int, LengthResult] = {i: EMPTY_RESULT for i in range(num_plants)}
        stats = {"success": 0, "failed": 0}

        for label_id in np.unique(labels):
            if label_id == 0:
                continue

            plant_id = int(label_id) - 1
            if not 0 <= plant_id < num_plants:
                logger.warning(
                    f"Label {label_id} maps to invalid plant_id {plant_id}, skipping"
                )
                continue

            result, return_code = self._process_single_plant(
                labels, label_id, plant_id, max_gap, max_path, endpoint_info
            )
            results[plant_id] = result

            if return_code == RETURN_SUCCESS:
                stats["success"] += 1
            else:
                stats["failed"] += 1

        logger.debug(
            f"Analysis complete: {stats['success']} successful, {stats['failed']} failed, "
            f"{num_plants - stats['success'] - stats['failed']} not detected"
        )
        return results
