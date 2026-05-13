import subprocess
from datetime import datetime

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential


# Function to grab the short SHA automatically
def get_git_revision_short_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


# Set your Azure ML workspace details
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace = "team-CV5-2026"

# Authenticate and initialize the client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace,
)


# 2. Define the tags based on Section 4 of your strategy
tags = {
    "source": "hades",
    "cohorts": "Y2B_24",  # Adjust if these 5 images belong to a different cohort
    "is_raw": "true",
    "num_images": "5",
    "num_masks": "0",  # No masks in this initial 5-image test
    "created_by": "VinceNemeth242058",
    "created_date": datetime.today().strftime("%Y-%m-%d"),
    "git_commit": get_git_revision_short_hash(),
    "excluded_paths": "Kaggle/,*.txt",
    "notes": "Initial test upload of 5 raw images to establish the dataset pipeline.",
}

# 3. Define the description based on Section 5 of your strategy
description = (
    "Raw HADES/NPEC plant segmentation dataset, mirrored from on-disk delivery.\n"
    "Source pointer: Local on-disk delivery (Initial 5-image test).\n"
    "Excluded: Kaggle/ folder and lone .txt files.\n"
    "Content details: docs/data/data_inventory.md\n"
    "Consumer contract: docs/data/preprocessing_spec.md"
)

# 4. Create the Data Asset definition
hades_raw_data = Data(
    name="cv5_hades_raw",  # Strict naming convention (Section 2.1)
    version="1.0.0",  # Initial major release (Section 3)
    description=description,
    path="C:/Users/nemev/Documents/GitHub/2025-26b-fai2-adsai-VinceNemeth242058/data/raw/Y2B_24/images",  # Path to the local folder containing your images
    type=AssetTypes.URI_FOLDER,  # Preserves folder structure (Section 2.2)
    tags=tags,
)

# 5. Register the data asset in Azure ML
print(f"Registering {hades_raw_data.name} v{hades_raw_data.version}...")
registered_data = ml_client.data.create_or_update(hades_raw_data)

print(f"Successfully registered: {registered_data.name}")
print(f"Version: {registered_data.version}")
