from pathlib import Path

# Define Paths relative to the script location
BASE_DIR = Path(__file__).parent.parent.parent
RAW_IMG_DIR = BASE_DIR / "dataset_sample" / "train_images" / "train"
PATCH_SIZE = 256
MODEL_PATH = BASE_DIR / "src" / "artifacts" / "models" / "unet_model_256px_v1.keras"