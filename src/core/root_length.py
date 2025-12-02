import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import logging

logger = logging.getLogger(__name__)

def calculate_geodesic_length(binary_mask: np.ndarray) -> float:
    """
    Calculates the curved length of a root by skeletonizing it and finding
    the shortest path through the skeleton from the top-most to bottom-most pixel.
    
    Args:
        binary_mask: Boolean or binary numpy array where the root is 1/True.
        
    Returns:
        float: The length in pixels. Returns 0.0 if mask is empty.
    """
    # 1. Validate Input
    if np.sum(binary_mask) == 0:
        return 0.0
        
    # 2. Skeletonize
    # Reduces the root to a 1-pixel wide line preserving topology
    skeleton = skeletonize(binary_mask)
    pixels = np.argwhere(skeleton)
    
    if len(pixels) < 2:
        return float(len(pixels))

    # 3. Build Graph from Skeleton
    # Nodes are (y, x) coordinates. Edges connect 8-connected neighbors.
    G = nx.Graph()
    
    # Fast graph construction
    # Create a mapping of coordinate tuples to simplify lookups
    pixel_set = set(map(tuple, pixels))
    
    for y, x in pixels:
        node = (y, x)
        G.add_node(node)
        
        # Check all 8 neighbors
        # We only need to check half (e.g., forward/down) to avoid duplicates, 
        # but checking all is safer for edge cases in grid graphs.
        neighbors = [
            (y-1, x-1), (y-1, x), (y-1, x+1),
            (y, x-1),             (y, x+1),
            (y+1, x-1), (y+1, x), (y+1, x+1)
        ]
        
        for ny, nx_coord in neighbors:
            if (ny, nx_coord) in pixel_set:
                # Weight is sqrt(2) for diagonals, 1.0 for orthogonal
                dist = np.sqrt((y-ny)**2 + (x-nx_coord)**2)
                G.add_edge(node, (ny, nx_coord), weight=dist)

    # 4. Find Start and End Points
    # Robustness: Use the largest connected component to ignore noise
    if not nx.is_connected(G):
        # Get largest component (subgraph)
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy()
        
    # Re-fetch nodes from the clean graph
    valid_nodes = list(G.nodes)
    
    # Sort by Y coordinate (vertical depth)
    # Start = minimum Y (top of image)
    # End = maximum Y (bottom of image)
    valid_nodes.sort(key=lambda p: p[0]) 
    
    start_node = valid_nodes[0]
    end_node = valid_nodes[-1]
    
    # 5. Calculate Shortest Path
    try:
        length = nx.shortest_path_length(
            G, 
            source=start_node, 
            target=end_node, 
            weight='weight'
        )
        return length
    except nx.NetworkXNoPath:
        # Fallback for disconnected graph edge cases
        return 0.0

def analyze_plant_lengths(labels: np.ndarray, num_plants: int = 5) -> dict:
    """
    Process a full labels array and calculate lengths for all plants.
    
    Args:
        labels: 2D numpy array with labeled components.
        num_plants: Expected number of plant zones (default 5).
        
    Returns:
        dict: Keys 0..4 containing detection stats.
              Includes 'detected' and 'length' for backward compatibility.
    """
    # 1. Initialize default structure for ALL expected plants
    # This prevents KeyError in pipelines that iterate through range(num_plants)
    results = {
        i: {
            'detected': False, 
            'length': 0, 
            'pixel_count': 0
        } 
        for i in range(num_plants)
    }

    unique_labels = np.unique(labels)
    
    for label_id in unique_labels:
        if label_id == 0: continue
        
        # Plant ID is 0-indexed, Label ID is 1-indexed
        plant_id = int(label_id) - 1
        
        # Safety check: if we found more labels than expected zones, ignore or log
        if plant_id >= num_plants or plant_id < 0:
            continue
        
        # Extract binary mask for this specific plant
        y_indices, x_indices = np.where(labels == label_id)
        if len(y_indices) == 0: continue
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # Create crop
        mask_crop = (labels[y_min:y_max+1, x_min:x_max+1] == label_id)
        
        # Calculate Metrics
        true_length = calculate_geodesic_length(mask_crop)
        pixel_count = len(y_indices)
        
        # Populate result for this plant
        results[plant_id] = {
            # Legacy fields (required for pipeline compatibility)
            'detected': True,
            'length': int(true_length), # Updating 'length' to be the curve length
            'pixel_count': pixel_count,
        }
        
    return results