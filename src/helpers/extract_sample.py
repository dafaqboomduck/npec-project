import os
import random
import shutil
from typing import Literal, Optional, List, Tuple
from pathlib import Path
# from collections import defaultdict
from src.helpers.dataset_manager import DatasetManager

def extract_test_data(img_dir: str, 
                      mask_dir: str, 
                      output_dir: Optional[str] = None, 
                      num_images: int = 10, 
                      patched: bool = False,
                      shuffle: bool = False, 
                      type: Literal["train", "val", "test"] = "train"):
    
    if patched:
        print("Error: pathed=True not supported ATM. For now, please use patched=False and patch the images using the DatasetManager.")
        return
    
    manager = DatasetManager()
    manager.load_images(img_dir)
    manager.load_masks(mask_dir)
    
    # 1. Infer output_dir if None
    if output_dir is None:
        try:
            # Assumes img_dir structure: <root>/<split>_images/<split>
            dataset_root = os.path.dirname(os.path.dirname(img_dir)) 
            dataset_root_name = os.path.basename(dataset_root)     
            output_dir = os.path.join(
                os.path.dirname(dataset_root),
                f"{dataset_root_name}_sample"
            )
        except Exception:
            print(f"Error inferring output_dir from '{img_dir}'. Please provide output_dir explicitly.")
            return

    # 2. Build map for robust, case-insensitive lookup, storing original mask filename
    mask_map = {m.lower(): m for m in manager.mask_files}

    # 3. Build list of valid pairs or patch groups
    valid_pairs: List[Tuple[str, str]] | List[List[Tuple[str, str]]] = []
    
    if not patched:
        # NON-PATCHED: List of (img_file, mask_file) tuples
        for img_file in manager.img_files:
            img_key = img_file.lower()
            mask_key = img_key.split(".png")[0] + "_root_mask.tif"
            if mask_key in mask_map:
                corr_mask_file = mask_map[mask_key] 
                valid_pairs.append((img_file, corr_mask_file))
    else:
        # # PATCHED: List of lists, where each inner list is a group of patches for one original image.
        # # Group all image/mask patch pairs by their original image base name.
        # patch_groups = defaultdict(list)
        
        # for img_file in manager.img_files:
        #     img_key = img_file.lower()
            
        #     # Find the corresponding mask file
        #     if img_key not in mask_map:
        #         continue # Skip images without a mask
                
        #     corr_mask_file = mask_map[img_key]
            
        #     # Extract base name (e.g., 'image_01_patch_00' -> 'image_01')
        #     base_name, _ = os.path.splitext(img_key)
        #     base_parts = base_name.split('_patch')
        #     base_img = base_parts[0] # Assumes base is always before '_patch'
        #     print(base_img)
            
        #     patch_groups[base_img].append((img_file, corr_mask_file))

        # # Convert the defaultdict values (lists of patches) into the final valid_pairs list
        # valid_pairs = [group for group in patch_groups.values() if group]
        print("Error: pathed=True not supported ATM. For now, please use patched=False and patch the images using the DatasetManager.")
        return
        
    # Check for found pairs/groups
    if not valid_pairs:
        print(f"Warning: No matching image/mask pairs/groups found.")
        return 
    
    # Check requested number
    available_count = len(valid_pairs)
    if num_images > available_count:
        print(f"Warning: Requested {num_images} samples, but only {available_count} valid {'groups' if patched else 'pairs'} available. Selecting all available.")
        num_images = available_count

    if shuffle:
        random.shuffle(valid_pairs)

    # 4. Select N samples (either pairs or groups)
    selected = valid_pairs[:num_images]

    # 5. Prepare output structure
    out_img_dir = os.path.join(output_dir, f"{type}_images", type)
    out_mask_dir = os.path.join(output_dir, f"{type}_masks", type)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    moved = []

    # 6. Move selected pairs/groups
    for item in selected:
        if patched:
            # item is a list of (img_file, mask_file) tuples (a group)
            for img_file, mask_file in item:
                src_img = os.path.join(img_dir, img_file)
                src_mask = os.path.join(mask_dir, mask_file)

                dst_img = os.path.join(out_img_dir, img_file)
                dst_mask = os.path.join(out_mask_dir, mask_file)

                shutil.move(src_img, dst_img)
                shutil.move(src_mask, dst_mask)
                moved.append((img_file, mask_file))
        else:
            # item is a single (img_file, mask_file) tuple
            img_file, mask_file = item
            src_img = os.path.join(img_dir, img_file)
            src_mask = os.path.join(mask_dir, mask_file)

            dst_img = os.path.join(out_img_dir, img_file)
            dst_mask = os.path.join(out_mask_dir, mask_file)

            shutil.move(src_img, dst_img)
            shutil.move(src_mask, dst_mask)
            moved.append((img_file, mask_file))

    print(f"Successfully moved {len(moved)} individual files ({num_images} {'groups' if patched else 'pairs'}) to: {output_dir}")
    return moved

if __name__ == "__main__":
    type = "train"
    # Get the directory two levels up from the script's location
    base_root = Path(os.path.realpath(__file__)).parents[2] / "dataset"
    img_dir = base_root / f"{type}_images" / type
    mask_dir = base_root / f"{type}_masks" / type
    extract_test_data(img_dir, mask_dir, num_images=50, shuffle=True)