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