from pathlib import Path

# Get the directory where config.py is located (src/)
CONFIG_DIR = Path(__file__).resolve().parent

# Go up one level to get project root
BASE_DIR = CONFIG_DIR.parent
IMG_DIR = BASE_DIR / "kaggle_dataset/images"

