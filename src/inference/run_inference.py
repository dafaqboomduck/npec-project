# Setup logging
import logging
logging.basicConfig(level=logging.WARNING)

# Import the module config 
from ..config.inference_config import RAW_IMG_DIR, MODEL_PATH

# Import the custom classes
from .inference_manager import InferenceManager
from .custom_functions import f1, iou, dice_bce_loss

# --- RUN INFERENCE ---

def run_inference():
    """Executes the full preprocessing and inference pipeline."""
    patch_input_dir = None
    
    try:
        custom_objects={"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}
        patch_dir_root = RAW_IMG_DIR.parent
        # The exact location where patches are saved
        patch_input_dir = patch_dir_root / "patches"
        
        # Initialize the InferenceManager
        manager = InferenceManager(str(MODEL_PATH), custom_objects=custom_objects)
        
        # Process Directory (Predict and Unpatch)
        manager.process_directory(str(patch_input_dir))

    except Exception as e:
        Exception(f" UNHANDLED EXCEPTION: {e}")

if __name__ == "__main__":
    run_inference()