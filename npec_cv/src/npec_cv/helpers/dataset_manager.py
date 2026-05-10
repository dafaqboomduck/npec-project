"""Dataset management and patch generation utilities."""

import argparse
import os
from typing import Literal

from tqdm.auto import tqdm

from .extract_petri_dish import extract_petri_dish
from .padder import padder


class DatasetManager:
    """DatasetManager for handling image and patch operations.

    Manages the loading, processing, and patching of image and corresponding
    root mask datasets. It handles file discovery, image loading, petri dish
    extraction, image padding, and patch generation.
    """

    def __init__(self):
        """Initialize the DatasetManager.

        Initializes the DatasetManager with empty lists for image and mask files
        and an empty set to store required mask filenames.
        """
        self.img_files = []
        self.mask_files = []
        self.required_masks = set()

    def load_images(self, img_dir: str):
        """Load image files from the specified directory.

        Loads the list of image files (ending with .png) from the specified directory.
        It also populates the required_masks set based on the image filenames.

        Args:
            img_dir (str): The path to the directory containing the image files.
        """
        files = sorted(os.listdir(img_dir))
        self.img_files = [f for f in files if f.lower().endswith(".png")]

        self.required_masks = {os.path.splitext(img)[0].lower() + "_root_mask.tif" for img in self.img_files}

    def load_masks(self, mask_dir: str):
        """Loads the list of mask files (ending with .tif) from the specified directory.

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
        type: Literal["train", "val", "test"] = "train",
    ):
        """Split images and corresponding root masks into uniform-sized patches.

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
            type (Literal['train', 'val', 'test']): The type of patches to generate.
        """
        import cv2
        import numpy as np

        from npec_cv.helpers.patchify import patchify

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print("Error: Input directories not found.")
            return

        if patch_dir is None:
            # img_dir example: test_data/train_images/train
            dataset_root = os.path.dirname(os.path.dirname(img_dir))
            dataset_root_name = os.path.basename(dataset_root)
            patch_dir = os.path.join(os.path.dirname(dataset_root), f"{dataset_root_name}_patches_{patch_size}px")

        os.makedirs(patch_dir, exist_ok=True)

        self.load_images(img_dir)
        self.load_masks(mask_dir)

        # Create a lookup dictionary: lowercase name -> actual filename
        mask_lookup = {m.lower(): m for m in self.mask_files}

        valid_pairs = []
        for img_file in self.img_files:
            base = os.path.splitext(img_file)[0].lower()
            corr_mask = base + "_root_mask.tif"
            if corr_mask in mask_lookup:
                mask_file = mask_lookup[corr_mask]  # Use the actual filename
                valid_pairs.append((img_file, mask_file))

        # Prepare output folders
        save_img_dir = os.path.join(patch_dir, f"{type}_images", type)
        save_mask_dir = os.path.join(patch_dir, f"{type}_masks", type)
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)

        total_patches_saved = 0

        # Main progress bar for processing images
        with tqdm(total=len(valid_pairs), desc="Processing images", unit="img") as pbar:
            for img_file, mask_file in valid_pairs:
                im_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)

                im = cv2.imread(im_path)
                # Load mask as grayscale (0 flag)
                mask = cv2.imread(mask_path, 0)

                if im is None or mask is None:
                    pbar.set_postfix_str(f"Skipped: {img_file}")
                    pbar.update(1)
                    continue

                # Scale if required
                if scaling_factor != 1.0:
                    scaled_im = cv2.resize(im, (0, 0), fx=scaling_factor, fy=scaling_factor)
                    # Use nearest neighbor for mask to preserve discrete values
                    scaled_mask = cv2.resize(
                        mask, (0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST
                    )
                else:
                    scaled_im = im
                    scaled_mask = mask

                # Extract Petri Dish area - returns coordinates
                x_start, y_start, x_end, y_end = extract_petri_dish(scaled_im, return_coords=True)

                cropped_im = scaled_im[y_start:y_end, x_start:x_end]
                cropped_mask = scaled_mask[y_start:y_end, x_start:x_end]

                # Pad
                padded_im = padder(cropped_im, patch_size)
                padded_mask = padder(cropped_mask, patch_size)

                # Ensure mask has channel dimension for patchify
                if padded_mask.ndim == 2:
                    padded_mask = np.expand_dims(padded_mask, axis=-1)

                # Patchify the cropped square area
                patches_im = patchify(padded_im, (patch_size, patch_size, 3), step=patch_size)
                patches_mask = patchify(padded_mask, (patch_size, patch_size, 1), step=patch_size)

                # Flatten patch arrays: (N_h, N_w, 1, patch_h, patch_w, N_c) -> (N, patch_h, patch_w, N_c)
                patches_im = patches_im.reshape(-1, patch_size, patch_size, 3)
                patches_mask = patches_mask.reshape(-1, patch_size, patch_size, 1)

                num_patches = len(patches_im)
                image_base_name = os.path.splitext(img_file)[0]

                # Save image and mask patches
                for j, patch in enumerate(patches_im):
                    save_name = f"{image_base_name}_patch_{j}.png"
                    cv2.imwrite(os.path.join(save_img_dir, save_name), patch)

                for j, patch in enumerate(patches_mask):
                    # Remove the channel dimension (1) before saving the mask
                    patch_to_save = patch.squeeze(-1)
                    save_name = f"{image_base_name}_patch_{j}.png"
                    cv2.imwrite(os.path.join(save_mask_dir, save_name), patch_to_save)

                total_patches_saved += num_patches
                pbar.set_postfix_str(f"{num_patches} patches | Total: {total_patches_saved}")
                pbar.update(1)

        print(f"\n✓ Complete! Generated {total_patches_saved} patches total")
        print(f"  Images: {save_img_dir}")
        print(f"  Masks: {save_mask_dir}")


def main():
    """Parses command-line arguments and runs the patch generation process."""
    parser = argparse.ArgumentParser(description="Generates fixed-size patches from image and mask directories.")

    # Required Arguments
    parser.add_argument("img_dir", type=str, help="Path to the directory containing source images (.png).")
    parser.add_argument("mask_dir", type=str, help="Path to the directory containing source masks (.tif).")

    # Optional Arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional destination path for the patches. "
        "If not provided, a default path like '<root_dir>_patches' will be used.",
    )
    parser.add_argument(
        "--patch_size",
        "-p",
        type=int,
        default=256,
        help="The side length (in pixels) for the square patches (default: 128).",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="A scaling factor to resize images before patching (e.g., 0.5 for half resolution) (default: 1.0).",
    )

    parser.add_argument("--type", "-t", type=str, default="train")

    args = parser.parse_args()

    manager = DatasetManager()

    # Call the main processing function with parsed arguments
    manager.create_and_save_patches(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        patch_dir=args.output,
        patch_size=args.patch_size,
        scaling_factor=args.scale,
        type=args.type,
    )


if __name__ == "__main__":
    main()
