"""Initialise the cv5_corrections destination for prediction corrections.

Stages a folder locally containing a README, then registers it as a
URI_FOLDER asset (cv5_corrections). The Azure ML SDK uploads the staged
contents to workspaceblobstore under a LocalUpload/<guid>/ path it
chooses. The asset's .path attribute resolves to that location; consumers
load the asset by name+version and never need to care about the GUID.

This is the workaround for the case where direct blob writes to
arbitrary datastore paths are blocked by RBAC. The trade-off is a less
clean URI, hidden behind the asset name.

Usage:
    python scripts/setup_corrections_asset.py --version 1.0.0
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

ASSET_NAME = "cv5_corrections"

README_CONTENT = """# cv5_corrections

Destination for user-submitted prediction corrections.

## Producer

The API writes corrected annotation masks here when a user confirms a
prediction via `POST /annotations/{annotation_id}/confirm`.

## Consumer

The retraining pipeline reads this folder to build new training data
snapshots (`cv5_train`/`cv5_val`/`cv5_test` v2.0.0 and later).

## File format

- One file per correction: `<annotation_id>.tif`
- Single-frame TIFF, `uint8`, binary mask (`{0, 1}`)
- Filename matches the `Annotation.annotation_id` in the database

## Conventions

- **Append-only.** Never modify or delete existing files. If a correction
  needs to be revoked, mark the annotation as `review_required` in the
  database and exclude it at retraining time, but keep the file.
- The `Annotation.mask_url` column in the database stores the full
  `azureml://...` URI to the file inside this folder. Producers obtain
  the folder URI by loading the `cv5_corrections` asset and writing to
  `<asset.path>/<annotation_id>.tif`.

## Asset

This folder is registered as the `cv5_corrections` Azure ML data asset.
The underlying storage path uses an auto-generated `LocalUpload/<guid>/`
prefix; consumers should never hardcode it -- always resolve via
`ml_client.data.get(name="cv5_corrections", version=...)` and use
`asset.path`.

See `docs/data/retraining_pipeline.md` (pending) for the full flow.
"""


def main() -> int:
    """Stage the corrections folder locally and register it as an asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Asset version, e.g. 1.0.0")
    args = parser.parse_args()

    load_dotenv()
    subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZUREML_RESOURCE_GROUP")
    workspace_name = os.getenv("AZUREML_WORKSPACE")
    if not all([subscription_id, resource_group, workspace_name]):
        print(
            "error: AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP / " "AZURE_ML_WORKSPACE missing from environment",
            file=sys.stderr,
        )
        return 1

    ml_client = MLClient(
        credential=InteractiveBrowserCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    with tempfile.TemporaryDirectory(prefix="cv5_corrections_init_") as tmp:
        staging = Path(tmp) / "corrections"
        staging.mkdir(parents=True)
        (staging / "README.md").write_text(README_CONTENT, encoding="utf-8")
        print(f"staged README at {staging / 'README.md'}")

        asset = Data(
            name=ASSET_NAME,
            version=args.version,
            description=(
                "Destination for user-submitted prediction corrections.\n"
                "The API writes corrected annotation masks here as users "
                "confirm predictions via POST /annotations/{annotation_id}"
                "/confirm.\n"
                "Filename convention: <annotation_id>.tif (single-frame "
                "TIFF, uint8 binary mask). The retraining pipeline reads "
                "this folder to build new training data snapshots.\n"
                "Append-only: never modify or delete existing files. The "
                "asset is a pointer to the folder; contents grow over "
                "time without asset version bumps.\n"
                "Underlying storage path uses an auto-generated "
                "LocalUpload/<guid>/ prefix -- always resolve via the "
                "asset, never hardcode.\n"
                "See the README.md inside the folder for full conventions."
            ),
            path=str(staging),
            type=AssetTypes.URI_FOLDER,
            tags={
                "role": "corrections",
                "purpose": "user_feedback_storage",
                "consumer": "retraining_pipeline",
                "producer": "api_backend",
                "format": "tiff_mask_uint8_binary",
                "filename_pattern": "<annotation_id>.tif",
                "append_only": "true",
            },
        )

        print(f"registering {ASSET_NAME} v{args.version}...")
        registered = ml_client.data.create_or_update(asset)
        print(f"-> {registered.name} v{registered.version}")
        print(f"   path: {registered.path}")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
