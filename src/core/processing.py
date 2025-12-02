import numpy as np
import logging
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.ndimage import find_objects
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.config.segmentation_config import PLANT_ZONES, SegmentationConfig

# Configure logger
logger = logging.getLogger(__name__)


class PlantSegmentor:
    def __init__(self, 
                 zones: Optional[Dict[int, Tuple[int, int]]] = None, 
                 config: Optional[SegmentationConfig] = None):
        """
        Initialize the segmentor.
        
        Args:
            zones: Dictionary mapping plant ID to (x_min, x_max) tuples.
            config: SegmentationConfig object with threshold values.
        """
        self.zones = zones if zones is not None else PLANT_ZONES
        self.config = config if config is not None else SegmentationConfig()

    def get_plant_position_by_zone(self, x_coord: float) -> Optional[int]:
        """
        Determine which plant position a coordinate belongs to.
        """
        for plant_id, (x_min, x_max) in self.zones.items():
            if x_min <= x_coord <= x_max:
                return plant_id
        return None

    def filter_plant_components(self, labels: np.ndarray) -> np.ndarray:
        """
        Filter connected components to keep only valid plant roots based on geometric properties.
        
        Optimized using scipy.ndimage.find_objects to avoid full-image scans.
        """
        if labels.max() == 0:
            return labels

        img_height, img_width = labels.shape
        filtered_labels = labels.copy()
        
        # find_objects returns a list of slices (slice_y, slice_x) for labels 1..N
        # It returns None for missing labels in the sequence
        slices = find_objects(labels)
        
        top_region_height = int(img_height * self.config.top_region_ratio)
        kept_count = 0
        rejected_count = 0

        for idx, sl in enumerate(slices):
            if sl is None:
                continue
                
            label_id = idx + 1 # find_objects is 0-indexed for label 1
            
            # Extract only the relevant crop for this component
            # This is significantly faster than (labels == label_id) on the whole image
            slice_y, slice_x = sl
            component_crop = labels[slice_y, slice_x]
            component_mask = (component_crop == label_id)
            
            # Calculate geometric properties
            area = np.sum(component_mask)
            y_min = slice_y.start
            # Height/Width comes directly from slice dimensions
            comp_height = slice_y.stop - slice_y.start
            comp_width = slice_x.stop - slice_x.start

            # 1. Area Check
            if area < self.config.min_area:
                self._reject(filtered_labels, sl, label_id, f"too small ({area}px)")
                rejected_count += 1
                continue

            # 2. Height Check
            if comp_height < img_height * self.config.min_height_ratio:
                self._reject(filtered_labels, sl, label_id, f"insufficient height ({comp_height}px)")
                rejected_count += 1
                continue

            # 3. Top Attachment Check (Relative to the crop, top region is adjusted)
            # We need to intersect the component's slice with the global top region
            local_top_limit = max(0, top_region_height - slice_y.start)
            if local_top_limit > 0:
                top_pixels = np.sum(component_mask[:local_top_limit, :])
            else:
                top_pixels = 0

            if top_pixels < self.config.min_top_pixels:
                self._reject(filtered_labels, sl, label_id, f"no top attachment ({top_pixels}px)")
                rejected_count += 1
                continue

            # 4. Aspect Ratio Check
            aspect_ratio = comp_height / (comp_width + 1e-8)
            if aspect_ratio < self.config.min_aspect_ratio:
                self._reject(filtered_labels, sl, label_id, f"too horizontal (AR={aspect_ratio:.2f})")
                rejected_count += 1
                continue

            # 5. Max Width Check
            if comp_width > img_width * self.config.max_width_ratio:
                self._reject(filtered_labels, sl, label_id, f"too wide ({comp_width}px)")
                rejected_count += 1
                continue

            # 6. Start Position Check
            if y_min > top_region_height:
                self._reject(filtered_labels, sl, label_id, f"starts too low (y={y_min})")
                rejected_count += 1
                continue

            # Accepted
            logger.debug(f"Component {label_id}: ACCEPTED (area={area}, h={comp_height})")
            kept_count += 1

        logger.info(f"Filter Summary: Kept {kept_count}, Rejected {rejected_count}")
        return filtered_labels

    def _reject(self, label_arr: np.ndarray, sl: Tuple[slice, slice], label_id: int, reason: str):
        """Helper to zero out a label efficiently."""
        logger.debug(f"Component {label_id}: REJECTED - {reason}")
        # Only modify the specific slice in the array
        crop = label_arr[sl]
        crop[crop == label_id] = 0

    def merge_components_by_distance(self, labels: np.ndarray, distance_thresh: float) -> np.ndarray:
        """
        Merge connected components based on the distance between their highest points.
        Uses scipy.sparse.csgraph for efficient connected component analysis.
        """
        unique_labels = np.array([l for l in np.unique(labels) if l != 0])
        n_components = len(unique_labels)
        
        logger.debug(f"Merging {n_components} original objects...")

        if n_components == 0:
            return labels

        # 1. Find the highest (topmost) point for each component efficiently
        # We can leverage find_objects again to find the start of the component
        slices = find_objects(labels)
        highest_points = []
        
        # Map sequential index 0..N back to original label ID
        idx_to_label = {} 

        valid_idx = 0
        for original_label in unique_labels:
            sl = slices[original_label - 1] # find_objects is 0-indexed vs label 1-indexed
            y_start, x_start = sl[0].start, sl[1].start
            
            # Refine x: find the first x pixel in the top row of this component
            # Extract just the top row relative to the slice
            crop = labels[sl]
            mask = (crop == original_label)
            
            # Get relative indices of pixels
            rel_y, rel_x = np.where(mask)
            
            # Find the minimum y (which is 0 relative to the slice usually, but not always if shape is complex)
            min_rel_y = rel_y.min()
            
            # Find the min x corresponding to that min y
            top_row_mask = (rel_y == min_rel_y)
            min_rel_x = rel_x[top_row_mask].min()
            
            # Convert back to absolute coordinates
            abs_y = y_start + min_rel_y
            abs_x = x_start + min_rel_x
            
            highest_points.append([abs_y, abs_x])
            idx_to_label[valid_idx] = original_label
            valid_idx += 1

        highest_points = np.array(highest_points)

        # 2. Compute Distance Matrix
        dist_matrix = cdist(highest_points, highest_points)
        
        # 3. Create Adjacency Matrix
        # adj[i, j] = 1 if distance <= threshold
        adjacency = (dist_matrix <= distance_thresh).astype(int)
        
        # 4. Connected Components (Union-Find optimization)
        # scipy's implementation is C-optimized and very fast
        graph = csr_matrix(adjacency)
        n_groups, group_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        # 5. Process Groups and Assign Zones
        out = np.zeros_like(labels)
        assigned_plants_map = {} # Maps group_id -> plant_position_id

        # Group indices by their new group label
        for group_id in range(n_groups):
            # Get indices of components in this group
            component_indices = np.where(group_labels == group_id)[0]
            
            # Find the "tip" of the group (min y, then min x) among all components in group
            # We look at highest_points array we already built
            group_points = highest_points[component_indices]
            
            # Sort by Y (primary) then X (secondary) to find top-most, left-most point
            # Lexsort keys are provided in reverse order (X then Y)
            sort_order = np.lexsort((group_points[:, 1], group_points[:, 0]))
            top_point = group_points[sort_order[0]]
            tip_x = top_point[1]
            
            plant_pos = self.get_plant_position_by_zone(tip_x)
            
            if plant_pos is not None:
                assigned_plants_map[group_id] = plant_pos
            else:
                logger.warning(f"Group tip at x={tip_x} outside defined zones.")

        # 6. Reconstruct Output Image
        # Iterate original labels and map them to new plant IDs
        # We assume output labels should be (plant_pos + 1) to reserve 0 for background
        for i, original_label in idx_to_label.items():
            group_id = group_labels[i]
            if group_id in assigned_plants_map:
                new_val = assigned_plants_map[group_id] + 1
                # Apply change only to the slice of the original component
                sl = slices[original_label - 1]
                crop_out = out[sl]
                crop_lbl = labels[sl]
                crop_out[crop_lbl == original_label] = new_val

        # Logging Statistics
        detected = sorted(list(set(assigned_plants_map.values())))
        # Safely determine expected zones from the keys of the zones dict
        all_possible_zones = sorted(self.zones.keys())
        missing = [z for z in all_possible_zones if z not in detected]

        logger.info(f"Merged into {len(assigned_plants_map)} plants. Positions: {detected}")
        if missing:
            logger.info(f"Missing positions: {missing}")

        return out