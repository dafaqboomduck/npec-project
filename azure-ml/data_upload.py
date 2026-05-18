import os
import subprocess
from datetime import datetime
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Helper Functions ---

# Set your Azure ML workspace details from environment variables
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_ML_WORKSPACE")


def get_git_revision_short_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def count_dataset_files(root_path: str):
    """Recursively counts images and masks within the dataset directory."""
    root = Path(root_path)

    # Define what extensions we consider valid (adjust if you use .tiff, .jpg, etc.)
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    img_count = 0
    mask_count = 0

    # rglob('*') recursively searches all files in all subdirectories
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            # Check if file is in 'masks' folder or has 'mask' in filename
            if "masks" in file_path.parts or "mask" in file_path.name.lower():
                mask_count += 1
            else:
                img_count += 1

    return str(img_count), str(mask_count)


# --- 2. Setup and Execution ---

# Set this to the TOP-LEVEL folder containing your test images
LOCAL_DATASET_ROOT = "C:/Users/nemev/Documents/TestImages"

print(f"Scanning {LOCAL_DATASET_ROOT} for files...")
num_images, num_masks = count_dataset_files(LOCAL_DATASET_ROOT)
print(f"Found {num_images} images and {num_masks} masks.")

ml_client = MLClient(
    credential=InteractiveBrowserCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace,
)

# Dynamically populated tags
tags = {
    "source": "test",
    "is_test_set": "true",
    "is_raw": "true",
    "num_images": num_images,  # <--- Dynamically set
    "num_masks": num_masks,  # <--- Dynamically set
    "created_by": "VinceNemeth242058",
    "created_date": datetime.today().strftime("%Y-%m-%d"),
    "git_commit": get_git_revision_short_hash(),
    "notes": "Test image dataset for validation and development.",
}

description = (
    "Test image dataset for validation and development.\n"
    "Source: Local test images directory.\n"
    "Used for pipeline testing and validation."
)

# Create the Data Asset definition
test_images_data = Data(
    name="cv5_test_images",
    version="1.0.0",
    description=description,
    path=LOCAL_DATASET_ROOT,  # <--- The test images root folder
    type=AssetTypes.URI_FOLDER,
    tags=tags,
)

print(f"Registering {test_images_data.name} v{test_images_data.version}...")
registered_data = ml_client.data.create_or_update(test_images_data)

print(f"Successfully registered: {registered_data.name}")
print(f"Version: {registered_data.version}")
