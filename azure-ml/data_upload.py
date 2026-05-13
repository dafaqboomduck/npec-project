import subprocess
from datetime import datetime
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import InteractiveBrowserCredential

# --- 1. Helper Functions ---

# Set your Azure ML workspace details
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace = "team-CV5-2026"


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
            # Check if this file lives inside an 'images' or 'masks' folder anywhere in its path
            if "images" in file_path.parts:
                img_count += 1
            elif "masks" in file_path.parts:
                mask_count += 1

    return str(img_count), str(mask_count)


# --- 2. Setup and Execution ---

# Set this to the TOP-LEVEL folder containing your cohort folders (Y2B_23, Y2B_24, etc.)
LOCAL_DATASET_ROOT = "C:/Users/nemev/Documents/GitHub/2025-26b-fai2-adsai-VinceNemeth242058/data/raw/Y2B_24/"

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
    "source": "hades",
    "cohorts": "Y2B_24",  # Update this if you are uploading multiple cohorts now!
    "is_raw": "true",
    "num_images": num_images,  # <--- Dynamically set
    "num_masks": num_masks,  # <--- Dynamically set
    "created_by": "<your_github_username>",
    "created_date": datetime.today().strftime("%Y-%m-%d"),
    "git_commit": get_git_revision_short_hash(),
    "excluded_paths": "Kaggle/,*.txt",
    "notes": "Uploading the full Y2B_24 cohort with automated file counting.",
}

description = (
    "Raw HADES/NPEC plant segmentation dataset, mirrored from on-disk delivery.\n"
    "Source pointer: Local on-disk delivery.\n"
    "Excluded: Kaggle/ folder and lone .txt files.\n"
    "Content details: docs/data/data_inventory.md\n"
    "Consumer contract: docs/data/preprocessing_spec.md"
)

# Create the Data Asset definition
hades_raw_data = Data(
    name="cv5_hades_raw",
    version="1.1.0",  # Bumped version since the data footprint changed!
    description=description,
    path=LOCAL_DATASET_ROOT,  # <--- The single common root folder
    type=AssetTypes.URI_FOLDER,
    tags=tags,
)

print(f"Registering {hades_raw_data.name} v{hades_raw_data.version}...")
registered_data = ml_client.data.create_or_update(hades_raw_data)

print(f"Successfully registered: {registered_data.name}")
print(f"Version: {registered_data.version}")
