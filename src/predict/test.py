import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from unittest.mock import MagicMock, patch
import logging
# --- 1. SETUP DUMMY ENVIRONMENT AND MOCKING ---

# Define Paths relative to the script location
BASE_DIR = Path(__file__).parent.parent.parent
RAW_IMG_DIR = BASE_DIR / "dataset_sample" / "train_images" / "train"
PATCH_SIZE = 256
MODEL_PATH = BASE_DIR / "src" / "artifacts" / "models" / "unet_model_256px_v1.keras"

logging.basicConfig(
    level=logging.DEBUG)

# Import your classes now that mocks are active
from .preprocessing import ImagePreprocessor
from .inference_manager import InferenceManager

# --- 3. RUN TESTS ---

def run_tests():
    """Executes the full preprocessing and inference pipeline."""
    patch_input_dir = None
    
    try:
        ## A. TEST IMAGE PREPROCESSOR
        print("\n--- 1. Testing ImagePreprocessor ---")
        processor = ImagePreprocessor()
        
        # 1. Load Images
        processor.load_images(RAW_IMG_DIR)
        assert len(processor.img_files) == 50, "ImagePreprocessor failed to load the correct number of images."
        print(f"✅ Load Images: Loaded {len(processor.img_files)} images.")

        # 2. Create and Save Patches
        # Note: The output path will be BASE_DIR / "data_patches_256px" / "test_subset_images" / "test_subset"
        patch_dir_root = processor.create_and_save_patches(
            img_dir=RAW_IMG_DIR,
            patch_size=PATCH_SIZE,
            subset_name="test_subset"
        )
        
        # The exact location where patches are saved
        patch_input_dir = patch_dir_root / "test_subset_images" / "test_subset"
        
        patch_files = list(patch_input_dir.glob("*.png"))
        expected_patches = 16 * 144 + 34 * 121
        assert len(patch_files) == expected_patches, f"Patch count mismatch. Expected {expected_patches}, got {len(patch_files)}"
        print(f"✅ Create Patches: Successfully generated {len(patch_files)} patches.")


        ## B. TEST INFERENCE MANAGER
        print("\n--- 2. Testing InferenceManager ---")
        
        manager = InferenceManager(str(MODEL_PATH), patch_size=PATCH_SIZE)
        
        # 1. Process Directory (Predict and Unpatch)
        manager.process_directory(str(patch_input_dir), cleanup=False)
        
        # 2. Validation
        output_dir = patch_input_dir.parent / "predicted_masks"
        
        assert output_dir.exists(), "Predicted mask directory not created."
        masks = list(output_dir.glob("*.tif"))
        
        assert len(masks) == 50, f"Mask count mismatch. Expected 50 masks, got {len(masks)}"
        print(f"✅ Process Directory: Generated {len(masks)} final masks.")
        
        # 3. Check Mask Dimensions (Must match the original padded image size)
        mask_sample = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
        print(mask_sample.shape)
        
        ## C. TEST CLEANUP
        # print("\n--- 3. Testing Cleanup ---")
        # processor.delete_patch_directory(patch_dir_root)
        # assert not patch_dir_root.exists(), "Cleanup failed to delete the patch root directory."
        # print("✅ Cleanup: Successfully deleted patch directory.")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNHANDLED EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
    # cleanup() # Uncomment this line to auto-delete the 'test_env' folder after completion