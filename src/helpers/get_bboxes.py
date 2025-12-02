import numpy as np

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