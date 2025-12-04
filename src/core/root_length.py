import numpy as np
import networkx as nx
from skimage.morphology import skeletonize, binary_dilation, disk
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


def bridge_skeleton_gaps(skeleton: np.ndarray, max_gap: int = 30) -> np.ndarray:
    """
    Bridge small gaps in a skeleton by connecting nearby endpoints.
    
    Args:
        skeleton: Binary skeleton array
        max_gap: Maximum gap distance (in pixels) to bridge
        
    Returns:
        Skeleton with gaps bridged
    """
    skeleton = skeleton.copy().astype(bool)
    
    # Find endpoint pixels (pixels with only 1 neighbor)
    endpoints = find_endpoints(skeleton)
    
    if len(endpoints) < 2:
        return skeleton
    
    # Build KD-tree for fast nearest neighbor lookup
    endpoint_array = np.array(endpoints)
    tree = cKDTree(endpoint_array)
    
    # Find pairs of endpoints within max_gap distance
    pairs_to_bridge = []
    used = set()
    
    for i, pt in enumerate(endpoints):
        if i in used:
            continue
            
        # Find nearby endpoints
        distances, indices = tree.query(pt, k=min(5, len(endpoints)), distance_upper_bound=max_gap)
        
        for dist, j in zip(distances, indices):
            if j >= len(endpoints) or i == j or j in used:
                continue
            if dist <= max_gap and dist > 0:
                # Check they're not already connected via the skeleton
                if not are_connected(skeleton, endpoints[i], endpoints[j]):
                    pairs_to_bridge.append((endpoints[i], endpoints[j], dist))
                    used.add(i)
                    used.add(j)
                    break
    
    # Sort by distance (bridge shortest gaps first)
    pairs_to_bridge.sort(key=lambda x: x[2])
    
    # Bridge each pair by drawing a line
    for pt1, pt2, _ in pairs_to_bridge:
        skeleton = draw_line(skeleton, pt1, pt2)
    
    return skeleton


def find_endpoints(skeleton: np.ndarray) -> list:
    """Find endpoint pixels in a skeleton (pixels with exactly 1 neighbor)."""
    endpoints = []
    pixels = np.argwhere(skeleton)
    pixel_set = set(map(tuple, pixels))
    
    for y, x in pixels:
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in pixel_set:
                    neighbors += 1
        if neighbors == 1:
            endpoints.append((y, x))
    
    return endpoints


def are_connected(skeleton: np.ndarray, pt1: tuple, pt2: tuple, max_path: int = 50) -> bool:
    """Check if two points are connected via the skeleton within a short path."""
    pixels = np.argwhere(skeleton)
    if len(pixels) < 2:
        return False
    
    pixel_set = set(map(tuple, pixels))
    if pt1 not in pixel_set or pt2 not in pixel_set:
        return False
    
    # BFS with limited depth
    visited = {pt1}
    queue = [(pt1, 0)]
    
    while queue:
        current, depth = queue.pop(0)
        if depth > max_path:
            continue
        if current == pt2:
            return True
            
        y, x = current
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor in pixel_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    
    return False


def draw_line(img: np.ndarray, pt1: tuple, pt2: tuple) -> np.ndarray:
    """Draw a line between two points using Bresenham's algorithm."""
    y0, x0 = int(pt1[0]), int(pt1[1])
    y1, x1 = int(pt2[0]), int(pt2[1])
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
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


def calculate_geodesic_length(binary_mask: np.ndarray, max_gap: int = 30) -> dict:
    """
    Calculates the curved length of a root by skeletonizing it, bridging gaps,
    and finding the shortest path from top to bottom.
    
    Args:
        binary_mask: Boolean or binary numpy array where the root is 1/True.
        max_gap: Maximum gap distance (in pixels) to bridge.
        
    Returns:
        dict: Contains 'length', 'path', 'skeleton', 'start', 'end' keys.
    """
    empty_result = {'length': 0.0, 'path': [], 'skeleton': None, 'start': None, 'end': None}
    
    if np.sum(binary_mask) == 0:
        return empty_result
    
    # Skeletonize
    skeleton = skeletonize(binary_mask)
    
    # Bridge gaps in the skeleton
    skeleton_bridged = bridge_skeleton_gaps(skeleton, max_gap=max_gap)
    
    pixels = np.argwhere(skeleton_bridged)
    
    if len(pixels) < 2:
        return {'length': float(len(pixels)), 'path': [], 'skeleton': skeleton_bridged, 'start': None, 'end': None}

    # Build graph from bridged skeleton
    G = nx.Graph()
    pixel_set = set(map(tuple, pixels))
    
    for y, x in pixels:
        node = (y, x)
        G.add_node(node)
        neighbors = [
            (y-1, x-1), (y-1, x), (y-1, x+1),
            (y, x-1),             (y, x+1),
            (y+1, x-1), (y+1, x), (y+1, x+1)
        ]
        for ny, nx_coord in neighbors:
            if (ny, nx_coord) in pixel_set:
                dist = np.sqrt((y - ny)**2 + (x - nx_coord)**2)
                G.add_edge(node, (ny, nx_coord), weight=dist)

    # Use largest connected component
    if not nx.is_connected(G):
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy()
        
    valid_nodes = list(G.nodes)
    valid_nodes.sort(key=lambda p: p[0])
    
    start_node = valid_nodes[0]
    end_node = valid_nodes[-1]
    
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
        return {
            'length': length,
            'path': path,
            'skeleton': skeleton_bridged,
            'start': start_node,
            'end': end_node
        }
    except nx.NetworkXNoPath:
        return {'length': 0.0, 'path': [], 'skeleton': skeleton_bridged, 'start': start_node, 'end': end_node}


def analyze_plant_lengths(labels: np.ndarray, num_plants: int = 5, max_gap: int = 30) -> dict:
    """
    Process a full labels array and calculate lengths for all plants.
    
    Args:
        labels: 2D numpy array with labeled components.
        num_plants: Expected number of plant zones (default 5).
        max_gap: Maximum gap to bridge in skeleton (default 30px).
        
    Returns:
        dict: Keys 0..4 containing detection stats.
    """
    results = {
        i: {'detected': False, 'length': 0, 'pixel_count': 0}
        for i in range(num_plants)
    }

    unique_labels = np.unique(labels)
    
    for label_id in unique_labels:
        if label_id == 0:
            continue
        
        plant_id = int(label_id) - 1
        
        if plant_id >= num_plants or plant_id < 0:
            continue
        
        y_indices, x_indices = np.where(labels == label_id)
        if len(y_indices) == 0:
            continue
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        mask_crop = (labels[y_min:y_max+1, x_min:x_max+1] == label_id)
        
        length_result = calculate_geodesic_length(mask_crop, max_gap=max_gap)
        pixel_count = len(y_indices)
        
        results[plant_id] = {
            'detected': True,
            'length': int(length_result['length']),
            'pixel_count': pixel_count,
        }
        
    return results