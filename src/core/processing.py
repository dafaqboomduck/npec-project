import numpy as np
from scipy.spatial.distance import cdist

import logging
logger = logging.getLogger(__name__)

from src.config.segmentation_config import *

class PlantSegmentor:
    def __init__(self, zones: dict = PLANT_ZONES):
        self.zones = zones

    def get_plant_position_by_zone(self, x_coord):
        """
        Determine which plant position (0-4) a component belongs to based on its X coordinate.
        Returns None if the coordinate doesn't fall within any defined zone.
        """
        for plant_id, (x_min, x_max) in self.zones.items():
            if x_min <= x_coord <= x_max:
                return plant_id
        return None
    
    def filter_plant_components(self, labels, img_height, img_width):
        """
        Filter connected components to keep only valid plant roots.
        """
        filtered_labels = labels.copy()
        unique_labels = [l for l in np.unique(labels) if l != 0]
        top_region_height = int(img_height * TOP_REGION_RATIO)
        
        for label_id in unique_labels:
            component_mask = (labels == label_id)
            y_coords, x_coords = np.where(component_mask)
            
            if len(y_coords) == 0:
                continue
                
            area = len(y_coords)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            comp_height = y_max - y_min
            comp_width = x_max - x_min
            
            if area < MIN_AREA:
                logger.debug(f"  Component {label_id}: REJECTED - too small ({area}px)")
                filtered_labels[filtered_labels == label_id] = 0
                continue
            
            if comp_height < img_height * MIN_HEIGHT_RATIO:
                logger.debug(f"  Component {label_id}: REJECTED - insufficient height ({comp_height}px, {comp_height/img_height:.1%})")
                filtered_labels[filtered_labels == label_id] = 0
                continue
            
            top_pixels = np.sum(component_mask[:top_region_height, :])
            if top_pixels < MIN_TOP_PIXELS:
                logger.debug(f"  Component {label_id}: REJECTED - no top attachment (top_pixels={top_pixels})")
                filtered_labels[filtered_labels == label_id] = 0
                continue
            
            aspect_ratio = comp_height / (comp_width + 1e-8)
            if aspect_ratio < MIN_ASPECT_RATIO:
                logger.debug(f"  Component {label_id}: REJECTED - too horizontal (aspect={aspect_ratio:.2f})")
                filtered_labels[filtered_labels == label_id] = 0
                continue
            
            if comp_width > img_width * MAX_WIDTH_RATIO:
                logger.debug(f"  Component {label_id}: REJECTED - too wide ({comp_width}px, {comp_width/img_width:.1%})")
                filtered_labels[filtered_labels == label_id] = 0
                continue
            
            if y_min > top_region_height:
                logger.debug(f"  Component {label_id}: REJECTED - doesn't start in top region (y_min={y_min})")
                filtered_labels[filtered_labels == label_id] = 0
                continue
            
            logger.debug(f"  Component {label_id}: ACCEPTED (area={area}, h={comp_height}, w={comp_width}, aspect={aspect_ratio:.2f})")
        
        return filtered_labels

    def merge_components_by_distance(self, labels: np.ndarray, distance_thresh: float) -> np.ndarray:
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

        logger.debug(f"The number of original objects detected in the iamge is: {n}")
        
        # Find the highest (topmost) point for each component
        highest_points = []
        for label_value in unique_labels:
            # Get all pixel coordinates belonging to this component
            y_coords, x_coords = np.where(labels == label_value)
            # Find minimum y-coordinate (topmost row)
            y_min = y_coords.min()
            mask = y_coords == y_min
            # Among pixels in the topmost row, find the leftmost one
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
        
        # For each group, find the highest point's X coordinate to determine plant zone
        group_tip_x = {}
        for root, indices in groups.items():
            # Collect all coordinates from all components in this group
            all_y = []
            all_x = []
            for i in indices:
                label_value = unique_labels[i]
                y_coords, x_coords = np.where(labels == label_value)
                all_y.extend(y_coords)
                all_x.extend(x_coords)
            
            # Convert to numpy arrays
            all_y = np.array(all_y)
            all_x = np.array(all_x)
            
            # Find the topmost point across all components in the group
            min_y = all_y.min()
            mask = (all_y == min_y)
            tip_x = all_x[mask].min()
            group_tip_x[root] = tip_x

         # Assign plant positions based on zones
        out = np.zeros_like(labels)
        assigned_plants = {}

        for root, tip_x in group_tip_x.items():
            plant_pos = self.get_plant_position_by_zone(tip_x)
            if plant_pos is not None:
                assigned_plants[root] = plant_pos
                logger.debug(f"Group with tip at x={tip_x} assigned to plant position {plant_pos}")
            else:
                logger.warning(f"Warning: Group with tip at x={tip_x} doesn't fall in any defined zone")

        # Apply labels: use plant_position + 1 as label (0 is background)
        for i in range(n):
            label_value = unique_labels[i]
            root = find(i)
            if root in assigned_plants:
                plant_label = assigned_plants[root] + 1  # +1 because 0 is background
                out[labels == label_value] = plant_label
        
        detected_positions = sorted(assigned_plants.values())
        missing_positions = [p for p in range(5) if p not in detected_positions]
        
        
        logger.debug(f"The merging resulted in {len(groups)} component(s)")
        logger.debug(f"Detected plants at positions: {detected_positions}")
        logger.debug(f"Missing plants at positions: {missing_positions}")
        
        return out