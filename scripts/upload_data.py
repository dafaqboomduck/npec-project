"""Upload HADES root-segmentation data as flat raw + train/val/test assets.

Walks the raw NPEC delivery, flattens cohorts and annotators into a single
images/masks structure, applies an 80/10/10 deterministic split for the
training subset, and registers four versioned data assets in the workspace:
cv5_raw, cv5_train, cv5_val, cv5_test.

Asset layout (identical for all four assets):

    <asset_root>/
        images/         # all images, flat, original filenames
        masks/          # all masks, flat, original filenames
        manifest.csv    # only in train/val/test; maps image to its masks

Inclusion rules:
  - cv5_raw: the splittable working set (images with a _root_mask.tif),
    pooled across cohorts, flattened, unsplit. Excludes Kaggle/, *.txt,
    *_occluded_root_mask.tif, and images with no root mask.
  - cv5_train/val/test: 80/10/10 deterministic split of cv5_raw by image
    stem, seed=42. An image and its 1-3 masks travel together.

Subset smoke test first (recommended):

    python scripts/upload_data.py --raw-root raw/ --subset 20 --version 0.1.0

Full upload after PO sign-off:

    python scripts/upload_data.py --raw-root raw/ --version 1.0.0
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

COHORTS_IN_SCOPE = ("Y2B_23", "Y2B_24", "Y2B_25")
IMAGE_EXTS = {".png"}
MASK_SUFFIXES = ("_root_mask.tif", "_shoot_mask.tif", "_seed_mask.tif")
REQUIRED_MASK_SUFFIX = "_root_mask.tif"
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
ASSET_NAMES = {
    "raw": "cv5_raw",
    "train": "cv5_train",
    "val": "cv5_val",
    "test": "cv5_test",
}

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class ImageRecord:
    """One image and any masks discovered alongside it."""

    stem: str
    image_path: Path
    masks: dict[str, Path] = field(default_factory=dict)

    @property
    def has_root_mask(self) -> bool:
        """True if a _root_mask.tif was found for this image."""
        return REQUIRED_MASK_SUFFIX in self.masks


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def discover_records(
    raw_root: Path,
    cohorts: tuple[str, ...] = COHORTS_IN_SCOPE,
) -> list[ImageRecord]:
    """Walk every in-scope cohort and pair images with masks.

    Returns a list sorted by stem for stable ordering across machines.
    Raises RuntimeError if two cohorts contain a file with the same name.
    """
    records: dict[str, ImageRecord] = {}
    seen_image_filenames: dict[str, Path] = {}
    seen_mask_filenames: dict[str, Path] = {}

    for cohort in cohorts:
        cohort_root = raw_root / cohort
        if not cohort_root.is_dir():
            print(f"  warn: cohort folder missing: {cohort_root}", file=sys.stderr)
            continue

        masks_by_stem: dict[str, dict[str, Path]] = defaultdict(dict)
        masks_dir = cohort_root / "masks"
        if masks_dir.is_dir():
            for mask_path in masks_dir.rglob("*.tif"):
                name = mask_path.name
                for suffix in MASK_SUFFIXES:
                    if name.endswith(suffix):
                        stem = name[: -len(suffix)]
                        masks_by_stem[stem][suffix] = mask_path
                        if name in seen_mask_filenames:
                            raise RuntimeError(
                                f"mask filename collision: {name} "
                                f"in {mask_path.parent} and "
                                f"{seen_mask_filenames[name].parent}"
                            )
                        seen_mask_filenames[name] = mask_path
                        break

        images_dir = cohort_root / "images"
        if not images_dir.is_dir():
            print(f"  warn: images folder missing: {images_dir}", file=sys.stderr)
            continue

        for img_path in images_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS or not img_path.is_file():
                continue
            if img_path.name in seen_image_filenames:
                raise RuntimeError(
                    f"image filename collision: {img_path.name} "
                    f"in {img_path.parent} and "
                    f"{seen_image_filenames[img_path.name].parent}"
                )
            seen_image_filenames[img_path.name] = img_path

            stem = img_path.stem
            rec = ImageRecord(
                stem=stem,
                image_path=img_path,
                masks=masks_by_stem.get(stem, {}),
            )
            records[stem] = rec

    all_records = sorted(records.values(), key=lambda r: r.stem)
    splittable = [r for r in all_records if r.has_root_mask]
    print(f"discovered {len(all_records)} images total; " f"{len(splittable)} have a root mask and are splittable")
    return all_records


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def split_for_training(records: list[ImageRecord], seed: int) -> dict[str, list[ImageRecord]]:
    """80/10/10 deterministic split. Caller must pre-filter to splittable records."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    for name, recs in splits.items():
        print(f"  {name}: {len(recs)} images")
    return splits


# --------------------------------------------------------------------------- #
# Staging
# --------------------------------------------------------------------------- #


def stage_flat(
    records: list[ImageRecord],
    out_dir: Path,
    write_manifest: bool,
) -> tuple[int, int]:
    """Copy images and masks into a flat images/ + masks/ layout."""
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    num_masks = 0

    for rec in records:
        shutil.copy2(rec.image_path, images_dir / rec.image_path.name)

        row = {
            "image_filename": rec.image_path.name,
            "root_mask_filename": "",
            "shoot_mask_filename": "",
            "seed_mask_filename": "",
        }
        for suffix in MASK_SUFFIXES:
            mask_path = rec.masks.get(suffix)
            if mask_path is None:
                continue
            shutil.copy2(mask_path, masks_dir / mask_path.name)
            num_masks += 1
            col = {
                "_root_mask.tif": "root_mask_filename",
                "_shoot_mask.tif": "shoot_mask_filename",
                "_seed_mask.tif": "seed_mask_filename",
            }[suffix]
            row[col] = mask_path.name
        manifest_rows.append(row)

    if write_manifest:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_filename",
                    "root_mask_filename",
                    "shoot_mask_filename",
                    "seed_mask_filename",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    return len(records), num_masks


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #


def git_short_sha() -> str:
    """Return the short SHA of the current git HEAD, or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def upload_asset(
    ml_client: MLClient,
    name: str,
    path: Path,
    version: str,
    description: str,
    tags: dict[str, str],
) -> None:
    """Register a URI_FOLDER data asset with the given name, version, and tags."""
    asset = Data(
        name=name,
        version=version,
        description=description,
        path=str(path),
        type=AssetTypes.URI_FOLDER,
        tags=tags,
    )
    print(f"  registering {name} v{version}...")
    registered = ml_client.data.create_or_update(asset)
    print(f"  -> {registered.name} v{registered.version}")


# --------------------------------------------------------------------------- #
# Register raw and splits
# --------------------------------------------------------------------------- #


def register_raw_and_splits(
    ml_client: MLClient,
    raw_root: Path,
    version: str,
    *,
    cohorts: tuple[str, ...] = COHORTS_IN_SCOPE,
    seed: int = 42,
    subset: int | None = None,
    dry_run: bool = False,
    created_by: str = "unknown",
    source_note: str = "",
    extra_tags: dict[str, str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Discover, split, stage, and register cv5_raw + cv5_train/val/test at `version`.

    Returns {asset_role: (n_images, n_masks)}. Reused by main() and by
    export_feedback_to_raw.py (which passes cohorts including 'feedback').
    """
    sha = git_short_sha()
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_subset = subset is not None

    all_records = discover_records(raw_root, cohorts=cohorts)
    if not all_records:
        raise RuntimeError("no records discovered; aborting")

    splittable = [r for r in all_records if r.has_root_mask]
    print(f"splittable subset (root mask present): {len(splittable)} images")

    if is_subset:
        rng = random.Random(seed)
        splittable = sorted(rng.sample(splittable, min(subset, len(splittable))), key=lambda r: r.stem)
        print(f"subset of {len(splittable)} images selected")

    splits = split_for_training(splittable, seed=seed)
    counts: dict[str, tuple[int, int]] = {}

    with tempfile.TemporaryDirectory(prefix="cv5_staging_") as tmp:
        staging = Path(tmp)
        raw_n_img, raw_n_mask = stage_flat(splittable, staging / "raw", write_manifest=False)
        counts["raw"] = (raw_n_img, raw_n_mask)
        print(f"  staged raw: {raw_n_img} images, {raw_n_mask} masks")

        for split in ("train", "val", "test"):
            counts[split] = stage_flat(splits[split], staging / split, write_manifest=True)
            print(f"  staged {split}: {counts[split][0]} images, {counts[split][1]} masks")

        if dry_run:
            print("dry run, skipping upload")
            return counts

        base_tags = {
            "source": "hades",
            "role": "raw",
            "mask_classes": "root,shoot,seed",
            "num_images": str(raw_n_img),
            "num_masks": str(raw_n_mask),
            "is_subset": str(is_subset).lower(),
            "created_by": created_by,
            "created_date": created,
            "git_commit": sha,
            "cohorts": ",".join(cohorts),
        }
        if extra_tags:
            base_tags.update(extra_tags)

        upload_asset(
            ml_client,
            name=ASSET_NAMES["raw"],
            path=staging / "raw",
            version=version,
            description=(
                f"HADES root-segmentation working set, pooled and flattened.\n"
                f"Cohorts {','.join(cohorts)} merged into one images/ + masks/ layout.\n"
                f"{source_note}\n"
                f"Git commit: {sha}. Subset: {str(is_subset).lower()}."
            ),
            tags=base_tags,
        )
        for split in ("train", "val", "test"):
            n_img, n_mask = counts[split]
            upload_asset(
                ml_client,
                name=ASSET_NAMES[split],
                path=staging / split,
                version=version,
                description=(
                    f"HADES root-segmentation {split} split.\n"
                    f"Derived from cv5_raw v{version} by 80/10/10 split, seed={seed}.\n"
                    f"{source_note} Git commit: {sha}."
                ),
                tags={**base_tags, "role": split, "num_images": str(n_img), "num_masks": str(n_mask)},
            )
    return counts


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    """Run the upload pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Path to raw delivery root")
    parser.add_argument("--version", required=True, help="Asset version, e.g. 1.0.0")
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, sample this many images for a smoke test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage locally, skip upload",
    )
    args = parser.parse_args()

    if not args.raw_root.is_dir():
        print(f"error: raw root not found: {args.raw_root}", file=sys.stderr)
        return 1

    load_dotenv()
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE")
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
    created_by = os.getenv("USER") or os.getenv("USERNAME") or "unknown"

    register_raw_and_splits(
        ml_client,
        args.raw_root,
        args.version,
        seed=args.seed,
        subset=args.subset,
        dry_run=args.dry_run,
        created_by=created_by,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
