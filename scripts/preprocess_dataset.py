"""Patchify cv5_train + cv5_val + cv5_test into a new versioned `cv5_patches` data asset.

Runs as a pipeline step on Azure ML compute. Reads three mounted source
folders (train/val/test, each with `images/` and `masks/` subdirs that may
sit under a `Y2B_24/` parent), pads every image and mask to a multiple of
`--patch_size`, splits each one into non-overlapping square patches, writes
them to a local temp dir, and registers the temp dir as a NEW VERSION of
the `cv5_patches` data asset on the workspace.

Output asset layout:

    cv5_patches:<auto-incremented version>/
    ├── train/
    │   ├── images/  (PNG patches)
    │   └── masks/   (PNG patches, single-channel uint8)
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

Each run of the pipeline auto-increments the version (1 → 2 → 3 …) by
listing existing versions of `cv5_patches` and adding one.

Usage (called by the pipeline, not manually):
    python scripts/preprocess_dataset.py
        --train_dir <mounted cv5_train>
        --val_dir   <mounted cv5_val>
        --test_dir  <mounted cv5_test>
        [--patch_size 512]
        [--asset_name cv5_patches]
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("preprocess_dataset")
    parser.add_argument("--train_dir", required=True, help="Mounted path to cv5_train")
    parser.add_argument("--val_dir", required=True, help="Mounted path to cv5_val")
    parser.add_argument("--test_dir", required=True, help="Mounted path to cv5_test")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--asset_name", default="cv5_patches")
    return parser.parse_args()


def get_client() -> MLClient:
    """Create MLClient using the cluster's managed identity."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )


def get_next_version(client: MLClient, name: str) -> str:
    """Return the next integer version for the named data asset.

    Lists every registered version of `name`, parses each as int (skipping
    non-integer versions like '1.0.0'), and returns max + 1 as a string.
    Returns '1' if nothing is registered yet.
    """
    try:
        existing = list(client.data.list(name=name))
    except Exception:
        return "1"
    if not existing:
        return "1"
    max_v = 0
    for d in existing:
        if d.version and d.version.isdigit():
            max_v = max(max_v, int(d.version))
    return str(max_v + 1) if max_v > 0 else "1"


def pad_to_multiple(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Zero-pad on the bottom and right so both dims are multiples of patch_size.

    Same convention as `train_root_detection._pad_to_multiple` so patches
    produced here are byte-for-byte identical to what training would produce.
    """
    h, w = arr.shape[:2]
    ph = ((h + patch_size - 1) // patch_size) * patch_size
    pw = ((w + patch_size - 1) // patch_size) * patch_size
    if ph == h and pw == w:
        return arr
    out = np.zeros((ph, pw) + arr.shape[2:], dtype=arr.dtype)
    out[:h, :w] = arr
    return out


def find_split_dirs(split_dir: str) -> Tuple[str, str]:
    """Locate `images/` and `masks/` for a split, handling the Y2B_24 wrapper."""
    for subdir in ["Y2B_24", ""]:
        base = os.path.join(split_dir, subdir) if subdir else split_dir
        img_d = os.path.join(base, "images")
        msk_d = os.path.join(base, "masks")
        if os.path.isdir(img_d) and os.path.isdir(msk_d):
            return img_d, msk_d
    raise FileNotFoundError(
        f"Cannot find images/ + masks/ inside {split_dir} (or its Y2B_24 subfolder). "
        f"Contents: {os.listdir(split_dir)}"
    )


def patchify_split(
    split_dir: str,
    out_img_dir: str,
    out_msk_dir: str,
    patch_size: int,
) -> int:
    """Patchify one split. Returns total patches written."""
    img_dir, mask_dir = find_split_dirs(split_dir)
    print(f"  Images: {img_dir}")
    print(f"  Masks : {mask_dir}")

    img_files = {Path(f).stem.lower(): f for f in os.listdir(img_dir) if Path(f).suffix.lower() in IMG_EXTS}

    # Walk the mask dir recursively because some splits nest masks inside
    # per-image subfolders.
    msk_files: dict[str, str] = {}
    for root, _, files in os.walk(mask_dir):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                msk_files[Path(f).stem.lower()] = os.path.join(root, f)

    # Match by stem, then fall back to `<stem>_root_mask` for the
    # naming convention used by some of the labelled exports.
    pairs: list[tuple[str, str]] = []
    for stem, img_fname in sorted(img_files.items()):
        if stem in msk_files:
            pairs.append((img_fname, msk_files[stem]))
        elif (stem + "_root_mask") in msk_files:
            pairs.append((img_fname, msk_files[stem + "_root_mask"]))

    print(f"  Image/mask pairs: {len(pairs)}")
    if not pairs:
        raise ValueError(f"No image/mask pairs matched in {split_dir}.")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    total = 0
    for img_fname, msk_path in pairs:
        img = cv2.imread(os.path.join(img_dir, img_fname))
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print(f"    Skipping {img_fname} (read failed)")
            continue

        img = pad_to_multiple(img, patch_size)
        msk = pad_to_multiple(msk, patch_size)
        h, w = img.shape[:2]
        stem = Path(img_fname).stem
        idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                name = f"{stem}_patch_{idx}.png"
                cv2.imwrite(os.path.join(out_img_dir, name), img[i : i + patch_size, j : j + patch_size])
                cv2.imwrite(os.path.join(out_msk_dir, name), msk[i : i + patch_size, j : j + patch_size])
                idx += 1
        total += idx

    print(f"  Patches written: {total}")
    return total


def main() -> None:
    """Patchify all three splits and register as a new version of cv5_patches."""
    args = parse_args()

    client = get_client()
    version = get_next_version(client, args.asset_name)
    print(f"Target asset       : {args.asset_name} v{version}")
    print(f"Patch size         : {args.patch_size}")

    # Build the full output structure under one temp dir, then upload the
    # whole tree to Azure ML in one shot via Data(path=...).
    with tempfile.TemporaryDirectory(prefix="patches_") as temp_dir:
        temp_dir_path = Path(temp_dir)

        totals: dict[str, int] = {}
        for split_name, source_dir in [
            ("train", args.train_dir),
            ("val", args.val_dir),
            ("test", args.test_dir),
        ]:
            print(f"\nPatchifying {split_name} from {source_dir}")
            out_img = temp_dir_path / split_name / "images"
            out_msk = temp_dir_path / split_name / "masks"
            totals[split_name] = patchify_split(source_dir, str(out_img), str(out_msk), args.patch_size)

        print(f"\nTotal patches      : {sum(totals.values())}  ({totals})")

        # Register as a new version. Passing a local path makes the SDK
        # upload the whole folder to the workspace's default datastore
        # and create a managed asset reference.
        description = (
            f"Patched + zero-padded version of cv5_train / cv5_val / cv5_test "
            f"(patch_size={args.patch_size}, non-overlapping, bottom-right padding). "
            f"Counts — train: {totals['train']}, val: {totals['val']}, test: {totals['test']}."
        )
        data_asset = Data(
            name=args.asset_name,
            version=version,
            path=str(temp_dir_path),
            type=AssetTypes.URI_FOLDER,
            description=description,
        )
        print(f"\nRegistering {args.asset_name} v{version} (uploading {sum(totals.values())} patches)...")
        registered = client.data.create_or_update(data_asset)
        print(f"Registered         : {registered.name} v{registered.version}")
        print(f"Asset path         : {registered.path}")


if __name__ == "__main__":
    main()
