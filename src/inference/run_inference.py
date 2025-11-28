# Setup logging
import logging
logging.basicConfig(level=logging.WARNING)

# Import the module config 
from ..config.inference_config import RAW_IMG_DIR, PATCH_SIZE, MODEL_PATH

# Import the custom classes
from .preprocessing import ImagePreprocessor
from .inference_manager import InferenceManager

# --- RUN INFERENCE ---

def run_inference():
    """Executes the full preprocessing and inference pipeline."""
    patch_input_dir = None
    
    try:
        # Initialize the ImagePreprocessor
        processor = ImagePreprocessor()
        
        # 1. Load Images
        processor.load_images(RAW_IMG_DIR)

        # 2. Create and Save Patches
        # Note: The output path will be BASE_DIR / "data_patches_256px" / "patches"
        patch_dir_root = processor.create_and_save_patches(
            img_dir=RAW_IMG_DIR,
            patch_size=PATCH_SIZE,
            subset_name="patches"
        )

        # The exact location where patches are saved
        patch_input_dir = patch_dir_root / "patches"
        
        # Initialize the InferenceManager
        manager = InferenceManager(str(MODEL_PATH), patch_size=PATCH_SIZE)
        
        # Process Directory (Predict and Unpatch)
        manager.process_directory(str(patch_input_dir), cleanup=False)

    except Exception as e:
        Exception(f" UNHANDLED EXCEPTION: {e}")

if __name__ == "__main__":
    run_inference()