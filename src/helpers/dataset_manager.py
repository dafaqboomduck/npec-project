import os
import argparse
from typing import Literal
from .padder import padder 
from .extract_petri_dish import extract_petri_dish

class DatasetManager:
    """
    Manages the loading, processing, and patching of image and corresponding
    root mask datasets. It handles file discovery, image loading, petri dish
    extraction, image padding, and patch generation.
    """
    def __init__(self):
        """
        Initializes the DatasetManager with empty lists for image and mask files
        and an empty set to store required mask filenames.
        """
        self.img_files = []
        self.mask_files = []
        self.required_masks = set()

    def load_images(self, img_dir: str):
        """
        Loads the list of image files (ending with .png) from the specified directory.
        It also populates the required_masks set based on the image filenames.

        Args:
            img_dir (str): The path to the directory containing the image files.
        """
        files = sorted(os.listdir(img_dir))
        self.img_files = [f for f in files if f.lower().endswith(".png")]

        self.required_masks = {
            os.path.splitext(img)[0].lower() + "_root_mask.tif"
            for img in self.img_files
        }

    def load_masks(self, mask_dir: str):
        """
        Loads the list of mask files (ending with .tif) from the specified directory.

        Args:
            mask_dir (str): The path to the directory containing the mask files.
        """
        files = sorted(os.listdir(mask_dir))
        self.mask_files = [f for f in files if f.lower().endswith(".tif")]

    def create_and_save_patches(
        self,
        img_dir: str,
        mask_dir: str,
        patch_dir: str | None = None,
        patch_size: int = 256,
        scaling_factor: float = 1.0,
        type: Literal["train", "val", "test"] = "train"):
        """
        Splits images and corresponding root masks into uniform-sized patches
        and saves them to a structured output directory.

        The process involves:
        1. Scaling images/masks if required.
        2. Padding the scaled images/masks to ensure they are fully divisible by
            the patch size.
        3. Cropping the padded images/masks to the area of the petri dish.
        4. Generating and saving the patches.

        Args:
            img_dir (str): The directory containing the source images (.png).
            mask_dir (str): The directory containing the source masks (.tif).
            patch_dir (str | None, optional): The base directory to save the patches.
                                                If None, it defaults to a path relative
                                                to the input directories (e.g.,
                                                'test_data_patches'). Defaults to None.
            patch_size (int, optional): The side length (in pixels) for the square patches.
                                        Defaults to 128.
            scaling_factor (float, optional): A factor by which to resize the images/masks
                                                before padding and patching (e.g., 0.5 for
                                                half resolution). Defaults to 1.0 (no scaling).
        """
        import cv2
        import numpy as np
        from patchify import patchify

        print(f"--- Starting Patch Generation ---")
        print(f"Parameters: Patch Size={patch_size}, Scaling Factor={scaling_factor}")
        print(f"Source Image Dir: {img_dir}")
        print(f"Source Mask Dir: {mask_dir}")
        print("-" * 35)

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print("Error: Input directories not found.")
            return

        if patch_dir is None:
            # img_dir example: test_data/train_images/train
            dataset_root = os.path.dirname(os.path.dirname(img_dir))   
            dataset_root_name = os.path.basename(dataset_root)         
            patch_dir = os.path.join(
                os.path.dirname(dataset_root),
                f"{dataset_root_name}_patches"
            )

        print(f"Destination Patch Dir: {patch_dir}")
        os.makedirs(patch_dir, exist_ok=True)

        self.load_images(img_dir)
        self.load_masks(mask_dir)
        
        print(f"Found {len(self.img_files)} images and {len(self.mask_files)} masks.")

        mask_set = {m.lower() for m in self.mask_files}

        valid_pairs = []
        for img_file in self.img_files:
            base = os.path.splitext(img_file)[0].lower()
            corr_mask = base + "_root_mask.tif"
            if corr_mask in mask_set:
                valid_pairs.append((img_file, corr_mask))

        print(f"Identified {len(valid_pairs)} valid image/mask pairs for processing.")
        print("-" * 35)

        # Prepare output folders
        save_img_dir = os.path.join(patch_dir, f"{type}_images", type)
        save_mask_dir = os.path.join(patch_dir, f"{type}_masks", type)
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)
        print(f"Saving patches to:\n  - {save_img_dir}\n  - {save_mask_dir}")

        for i, (img_file, mask_file) in enumerate(valid_pairs):
            print(f"\n[{i+1}/{len(valid_pairs)}] Processing: {img_file}")
            im_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            im = cv2.imread(im_path)
            # Load mask as grayscale (0 flag)
            mask = cv2.imread(mask_path, 0)

            if im is None or mask is None:
                print(f"  Warning: Could not read image/mask. Skipping.")
                continue

            # Initial shape check
            print(f"  Original shape: Image={im.shape}, Mask={mask.shape}")

            # Scale if required
            if scaling_factor != 1.0:
                scaled_im = cv2.resize(im, (0, 0), fx=scaling_factor, fy=scaling_factor)
                # Use nearest neighbor for mask to preserve discrete values
                scaled_mask = cv2.resize(mask, (0, 0), fx=scaling_factor, fy=scaling_factor,
                                            interpolation=cv2.INTER_NEAREST)
            else:
                scaled_im = im
                scaled_mask = mask

            # Pad
            padded_im = padder(scaled_im, patch_size)
            padded_mask = padder(scaled_mask, patch_size)

            # Ensure mask has channel dimension for patchify
            if padded_mask.ndim == 2:
                padded_mask = np.expand_dims(padded_mask, axis=-1)

            # Extract Petri Dish area - returns coordinates
            x_start, y_start, x_end, y_end = extract_petri_dish(padded_im, return_coords = True)
            
            cropped_im = padded_im[y_start:y_end, x_start:x_end]
            cropped_mask = padded_mask[y_start:y_end, x_start:x_end]
            print(f"  Cropped image shape: {cropped_im.shape}")

            # Patchify the cropped square area
            patches_im = patchify(cropped_im, (patch_size, patch_size, 3), step=patch_size)
            patches_mask = patchify(cropped_mask, (patch_size, patch_size, 1), step=patch_size)

            # Flatten patch arrays: (N_h, N_w, 1, patch_h, patch_w, N_c) -> (N, patch_h, patch_w, N_c)
            patches_im = patches_im.reshape(-1, patch_size, patch_size, 3)
            patches_mask = patches_mask.reshape(-1, patch_size, patch_size, 1)
            
            num_patches = len(patches_im)
            print(f"  Generated {num_patches} patches (size {patch_size}x{patch_size})")

            image_base_name = os.path.splitext(img_file)[0]
            
            # Save image and mask patches
            for j, patch in enumerate(patches_im):
                save_name = f"{image_base_name}_patch_{j}.png"
                cv2.imwrite(os.path.join(save_img_dir, save_name), patch)

            for j, patch in enumerate(patches_mask):
                # Remove the channel dimension (1) before saving the mask
                patch_to_save = patch.squeeze(-1)
                # save_name = f"{image_base_name}_patch_{j}_root_mask.png"
                save_name = f"{image_base_name}_patch_{j}.png"
                cv2.imwrite(os.path.join(save_mask_dir, save_name), patch_to_save)
            
            print(f"  Successfully saved {num_patches} patches for {img_file}.")

        print("\n--- Patch Generation Complete ---")

def main():
    """
    Parses command-line arguments and runs the patch generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generates fixed-size patches from image and mask directories."
    )
    
    # Required Arguments
    parser.add_argument(
        "img_dir",
        type=str,
        help="Path to the directory containing source images (.png)."
    )
    parser.add_argument(
        "mask_dir",
        type=str,
        help="Path to the directory containing source masks (.tif)."
    )
    
    # Optional Arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional destination path for the patches. If not provided, a default path like '<root_dir>_patches' will be used."
    )
    parser.add_argument(
        "--patch_size",
        "-p",
        type=int,
        default=256,
        help="The side length (in pixels) for the square patches (default: 128)."
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="A scaling factor to resize images before patching (e.g., 0.5 for half resolution) (default: 1.0)."
    )

    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="train"
    )


    args = parser.parse_args()

    manager = DatasetManager()
    
    # Call the main processing function with parsed arguments
    manager.create_and_save_patches(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        patch_dir=args.output,
        patch_size=args.patch_size,
        scaling_factor=args.scale,
        type=args.type
    )


if __name__ == "__main__":
    main()