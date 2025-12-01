import numpy as np
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


def merge_components_by_distance(labels: np.ndarray, distance_thresh: float) -> np.ndarray:
    """
    Merge connected components based on the distance between their highest points.
    
    This function uses a union-find algorithm to group components whose highest points
    (topmost pixels) are within a specified distance threshold. Components are merged
    by finding their topmost point (smallest y-coordinate, then smallest x-coordinate)
    and calculating pairwise distances between these points.
    
    Args:
        labels (np.ndarray): 2D array where each unique non-zero integer represents 
                           a distinct connected component. Background is labeled as 0.
        distance_thresh (float): Maximum Euclidean distance between highest points 
                               for components to be merged.
    
    Returns:
        np.ndarray: New label array with merged components renumbered sequentially 
                   starting from 1. Background remains 0.
    """
    # Extract all unique component labels, excluding background (label 0)
    unique_labels = np.array([l for l in np.unique(labels) if l != 0])
    n = len(unique_labels)
    logger.debug(f"The number of original objects detected in the image is: {n}")
    
    # Handle edge case: no components to merge
    if n == 0:
        return labels
    
    # Find the highest (topmost) point for each component
    highest_points = []
    for label_value in unique_labels:
        # Get all pixel coordinates belonging to this component
        y_coords, x_coords = np.where(labels == label_value)
        
        # Find minimum y-coordinate (topmost row)
        y_min = y_coords.min()
        
        # Among pixels in the topmost row, find the leftmost one
        mask = y_coords == y_min
        x_min = x_coords[mask].min()
        
        highest_points.append([y_min, x_min])
    
    highest_points = np.array(highest_points)
    
    # Initialize union-find structure
    # Each component starts as its own parent
    merged = np.arange(n)
    
    def find(x):
        """
        Find operation with path compression for union-find.
        Returns the root parent of component x.
        """
        while merged[x] != x:
            # Path compression: point directly to grandparent
            merged[x] = merged[merged[x]]
            x = merged[x]
        return x
    
    def union(a, b):
        """
        Union operation for union-find.
        Merges the sets containing components a and b.
        """
        ra, rb = find(a), find(b)
        if ra != rb:
            # Make rb point to ra (merge sets)
            merged[rb] = ra
    
    # Calculate pairwise Euclidean distances between all highest points
    D = cdist(highest_points, highest_points)
    
    # Merge components whose highest points are within the distance threshold
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= distance_thresh:
                union(i, j)
    
    # Group components by their root
    groups = {}
    for i in range(n):
        r = find(i)
        if r not in groups:
            groups[r] = []
        groups[r].append(i)
    
    # For each group, find the leftmost x-coordinate
    group_left_x = {}
    for root, indices in groups.items():
        min_x = float('inf')
        for i in indices:
            label_value = unique_labels[i]
            x_coords = np.where(labels == label_value)[1]
            min_x = min(min_x, x_coords.min())
        group_left_x[root] = min_x
    
    # Sort groups by their leftmost x-coordinate
    sorted_roots = sorted(groups.keys(), key=lambda r: group_left_x[r])
    
    # Create mapping: sorted position -> new label (starting from 1, since 0 is background)
    root_map = {root: idx + 1 for idx, root in enumerate(sorted_roots)}
    
    out = np.zeros_like(labels)
    for i in range(n):
        label_value = unique_labels[i]
        merged_id = root_map[find(i)]
        out[labels == label_value] = merged_id
    
    logger.debug(f"The merging resulted in {len(groups)} component(s)")
    
    return out


def get_bounding_boxes(labels: np.ndarray) -> dict:
    """
    Extract axis-aligned bounding boxes for each labeled component.
    
    Args:
        labels (np.ndarray): 2D array where each unique non-zero integer represents 
                           a distinct connected component. Background is labeled as 0.
    
    Returns:
        dict: Dictionary mapping label_id (int) to bounding box tuple 
             (x_min, y_min, x_max, y_max), where:
             - x_min, y_min: top-left corner coordinates
             - x_max, y_max: bottom-right corner coordinates
    """
    bboxes = {}
    # Get all non-background labels
    unique_labels = [l for l in np.unique(labels) if l != 0]
    
    for label_id in unique_labels:
        # Find all pixels belonging to this component
        y_coords, x_coords = np.where(labels == label_id)
        
        # Calculate bounding box extents
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        bboxes[label_id] = (x_min, y_min, x_max, y_max)
    
    return bboxes


def crop_components(image: np.ndarray, labels: np.ndarray, padding: int = 10) -> dict:
    """
    Extract cropped regions for each labeled component with optional padding.
    
    This function crops the original image around each component's bounding box,
    creating isolated views of individual components. It also generates a binary
    mask for each component showing only that component's pixels.
    
    Args:
        image (np.ndarray): Original image (grayscale or color).
        labels (np.ndarray): 2D label array with same height/width as image, 
                           where each unique non-zero integer represents a 
                           distinct component.
        padding (int, optional): Number of pixels to add around each bounding box.
                               Defaults to 10. Padding is clipped to image boundaries.
    
    Returns:
        dict: Dictionary mapping label_id to a dictionary containing:
            - 'image' (np.ndarray): Cropped image region
            - 'mask' (np.ndarray): Binary mask (255 for component, 0 for background)
            - 'bbox' (tuple): Padded bounding box (x_min, y_min, x_max, y_max)
    """
    # Get bounding boxes for all components
    bboxes = get_bounding_boxes(labels)
    cropped_images = {}
    
    # Get image dimensions for boundary checking
    h, w = image.shape[:2]
    
    for label_id, (x_min, y_min, x_max, y_max) in bboxes.items():
        # Expand bounding box by padding, ensuring we don't go out of bounds
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop the image region (note: +1 on max to include the boundary pixel)
        cropped = image[y_min:y_max+1, x_min:x_max+1]
        
        # Extract the corresponding label region
        cropped_mask = labels[y_min:y_max+1, x_min:x_max+1]
        
        # Create binary mask: 255 for this component's pixels, 0 for everything else
        cropped_mask = (cropped_mask == label_id).astype(np.uint8) * 255
        
        # Store all extracted data for this component
        cropped_images[label_id] = {
            'image': cropped,
            'mask': cropped_mask,
            'bbox': (x_min, y_min, x_max, y_max)
        }
    
    return cropped_images