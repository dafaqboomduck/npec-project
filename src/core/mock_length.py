import numpy as np

def get_plant_results(labels):
    """
    Get detection results for all 5 plant positions.
    Returns dict with plant_id: {'detected': bool, 'pixel_count': int, 'length': int}
    Length is calculated as vertical distance from top to bottom of the component.
    """
    results = {i: {'detected': False, 'pixel_count': 0, 'length': 0} for i in range(5)}
    
    for plant_pos in range(5):
        label_id = plant_pos + 1  # Labels are 1-indexed
        y_coords = np.where(labels == label_id)[0]
        
        if len(y_coords) > 0:
            results[plant_pos]['detected'] = True
            results[plant_pos]['pixel_count'] = len(y_coords)
            # Length = vertical extent (bottom - top)
            y_min, y_max = y_coords.min(), y_coords.max()
            results[plant_pos]['length'] = y_max - y_min
    
    return results