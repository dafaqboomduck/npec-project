import os

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

    
    def extract_petri_dish(self, img, padding=30, return_coords: bool = False):
        """
        Identifies the largest object (assumed to be the petri dish) in an image
        using Otsu's thresholding and contour finding, and then extracts a
        padded square region containing it.

        The extracted region is guaranteed to be a square and is clamped to the
        image boundaries.

        Args:
            img (np.ndarray): The input image (BGR or grayscale).
            padding (int, optional): Extra pixels to add around the bounding box
                                     to form the final square side. Defaults to 30.
            return_coords (bool, optional): If True, returns the coordinates (x_start,
                                            y_start, x_end, y_end) instead of the
                                            cropped image. Defaults to False.

        Returns:
            np.ndarray or tuple: The cropped square image, or a tuple of
                                 (x_start, y_start, x_end, y_end) if
                                 return_coords is True.
        """
        import cv2

        def bgr2gray(img):
            """
            Converts a BGR image to grayscale. Returns a copy if the image is
            already 2-dimensional (grayscale).

            Args:
                img (np.ndarray): The input image.

            Returns:
                np.ndarray: The grayscale image.
            """
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            return gray
        
        def calculate_square(img, rect, padding):
            """
            Calculates the coordinates for a padded square crop based on a bounding box.
            The resulting square is centered on the bounding box and its side length
            is max(width, height) + 2*padding. The coordinates are clamped to the
            image boundaries (0 to img.shape[0/1]).

            Args:
                img (np.ndarray): The full image array.
                rect (tuple): The bounding box (x, y, w, h) from cv2.boundingRect.
                padding (int): The padding to add on all sides.

            Returns:
                tuple: A tuple (x_start, y_start, x_end, y_end) representing the
                       final square crop coordinates.
            """
            x, y, w, h = rect

            # 1. Determine the largest dimension (W or H) for the square's side
            side = max(w, h)

            # 2. Calculate the center point of the original bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # 3. Calculate the new square coordinates based on the center and the side length
            padded_side = side + 2 * padding
            half_side = padded_side // 2

            # New top-left corner without image bounds check yet
            x_new = center_x - half_side
            y_new = center_y - half_side

            # 4. Define the final crop coordinates (x_start, y_start, x_end, y_end)
            # Calculate final coordinates, clamping them to the image boundaries
            x_start = max(0, x_new)
            y_start = max(0, y_new)
            
            # Calculate the *desired* bottom-right corner
            x_desired_end = x_new + padded_side
            y_desired_end = y_new + padded_side

            # Clamp the end coordinates to the image width/height
            x_end = min(img.shape[1], x_desired_end)
            y_end = min(img.shape[0], y_desired_end)

            # 5. Handle cases where clamping truncated the square (e.g., if the dish is at an edge)
            # Calculate the actual dimensions after clamping
            actual_w = x_end - x_start
            actual_h = y_end - y_start
            
            # Use the *smaller* of the actual dimensions as the final side length
            final_side = min(actual_w, actual_h)
            
            # Recalculate the end points to form a perfect square using 'final_side'
            # This part adjusts the *start* point if the image boundary cut off the *end* point.
            
            # If the X dimension was cut short, adjust x_start to maintain the square size
            if actual_w < padded_side and x_end == img.shape[1]:
                x_start = max(0, x_end - final_side)
                
            # If the Y dimension was cut short, adjust y_start to maintain the square size
            if actual_h < padded_side and y_end == img.shape[0]:
                y_start = max(0, y_end - final_side)

            # Ensure the final square uses the calculated 'final_side'
            # (This is mostly a safety check after the boundary adjustments)
            x_end = x_start + final_side
            y_end = y_start + final_side

            return x_start, y_start, x_end, y_end
    
        
        # Convert to grayscale for easier processing
        gray = bgr2gray(img)

        # Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the Largest Contour
        # Safety: handle case where no contours are found (shouldn't happen with Otsu)
        if not contours:
            return None if not return_coords else (0, 0, img.shape[1], img.shape[0])

        largest_contour = max(contours, key=cv2.contourArea)

        # Get the Bounding Box
        # x, y are the top-left coordinates, w is width, h is height
        rect = cv2.boundingRect(largest_contour)
        
        # Calculate padded coordinates, ensuring they don't go out of image bounds
        x_start, y_start, x_end, y_end = calculate_square(img, rect, padding)
        
        cropped_image = img[y_start:y_end, x_start:x_end]
        if not return_coords:
            return cropped_image
        else:
            return x_start, y_start, x_end, y_end

    def create_and_save_patches(
            self,
            img_dir: str,
            mask_dir: str,
            patch_dir: str | None = None,
            patch_size: int = 128,
            scaling_factor: float = 1.0
        ):
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

            def padder(image, patch_size):
                """
                Pads an image (or mask) so that its height and width are the
                smallest multiple of `patch_size` that is greater than or equal
                to the original dimensions. Padding is distributed evenly on all sides.

                Args:
                    image (np.ndarray): The input image or mask.
                    patch_size (int): The desired patch size.

                Returns:
                    np.ndarray: The padded image/mask.
                """
                h, w = image.shape[:2]

                # Calculate the required padding to make dimensions a multiple of patch_size
                height_padding = ((h // patch_size) + 1) * patch_size - h
                width_padding  = ((w // patch_size) + 1) * patch_size - w

                # Distribute padding to top/bottom, left/right
                top = height_padding // 2
                bottom = height_padding - top
                left = width_padding // 2
                right = width_padding - left

                # Handle channels for pad value
                if image.ndim == 2:        # mask (grayscale)
                    pad_value = 0
                else:                      # image (color)
                    pad_value = [0, 0, 0]

                return cv2.copyMakeBorder(
                    image, top, bottom, left, right,
                    cv2.BORDER_CONSTANT,
                    value=pad_value
                )

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
                dataset_root = os.path.dirname(os.path.dirname(img_dir))   # test_data
                dataset_root_name = os.path.basename(dataset_root)         # test_data
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
            save_img_dir = os.path.join(patch_dir, "images")
            save_mask_dir = os.path.join(patch_dir, "masks")
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
                x_start, y_start, x_end, y_end = self.extract_petri_dish(padded_im, return_coords = True)
                
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
                    save_name = f"{image_base_name}_root_mask_patch_{j}.png"
                    cv2.imwrite(os.path.join(save_mask_dir, save_name), patch_to_save)
                
                print(f"  Successfully saved {num_patches} patches for {img_file}.")

            print("\n--- Patch Generation Complete ---")

                

if __name__ == "__main__":
    # Example usage:
    manager = DatasetManager()
    img_dir = "test_data/train_images/train"
    mask_dir = "test_data/train_masks/train"

    manager.create_and_save_patches(img_dir, mask_dir)