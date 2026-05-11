"""Inventory the HADES/NPEC raw dataset.

Walks a root directory, counts files by extension and folder, samples a
configurable number of images + masks to extract dimensions, dtype, channel
count, and mask value distribution, and flags unreadable files.

Usage:
    python inventory_dataset.py <root_path> [--sample N] [--out report.md]

Output is both printed and (optionally) written to a markdown report that can
be pasted straight into docs/data/data_inventory.md.

Dependencies: Pillow, numpy. Optional: tifffile (for .tif/.tiff).
Install:  pip install pillow numpy tifffile
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile  # optional, only needed for TIFFs

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_HINTS = ("mask", "label", "annotation", "gt", "segment")


def is_mask_path(p: Path) -> bool:
    """Check if a path corresponds to a mask based on filename hints.

    Args:
        p: The path to check.

    Returns:
        True if the path is likely a mask, False otherwise.
    """
    s = str(p).lower()
    return any(h in s for h in MASK_HINTS)


def load_image(path: Path) -> np.ndarray | None:
    """Load as numpy array, returning None on failure."""
    try:
        if path.suffix.lower() in {".tif", ".tiff"} and HAS_TIFFFILE:
            return tifffile.imread(str(path))
        with Image.open(path) as im:
            im.load()
            return np.array(im)
    except Exception:
        return None


def summarise_array(arr: np.ndarray) -> dict:
    """Extract metadata and statistics from a numpy array.

    Args:
        arr: The numpy array to summarise.

    Returns:
        A dictionary containing shape, dtype, min, max, and channels.
    """
    info = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    if arr.ndim == 2:
        info["channels"] = 1
    elif arr.ndim == 3:
        info["channels"] = arr.shape[-1]
    return info


def analyse_pairing(image_paths: list[Path], mask_paths: list[Path]) -> dict:
    """Try to match images to masks by stem, report the ratio and orphans.

    Heuristic: strip common mask suffixes (_mask, _label, etc.) from mask stems
    and see how many match image stems exactly. Then also check what other
    patterns (prefixes, parent folders) might be grouping them.
    """
    image_stems = {p.stem: p for p in image_paths}
    mask_stems_raw = {p.stem: p for p in mask_paths}

    # strip common suffixes from mask stems, then try progressively shorter prefixes
    suffix_patterns = ["_mask", "_label", "_gt", "_annotation", "_segment", "-mask", "-label"]
    mask_stem_to_image_stem: dict[str, str] = {}
    for mstem in mask_stems_raw:
        stripped = mstem
        # first pass: remove mask suffix
        for sfx in suffix_patterns:
            if stripped.lower().endswith(sfx):
                stripped = stripped[: -len(sfx)]
                break
        # second pass: if stripped doesn't match an image, try peeling underscore-delimited
        # tokens from the end (e.g. "img_001_leaf" -> "img_001") until we find an image
        candidate = stripped
        if candidate not in image_stems and "_" in candidate:
            parts = candidate.split("_")
            for i in range(len(parts) - 1, 0, -1):
                shortened = "_".join(parts[:i])
                if shortened in image_stems:
                    candidate = shortened
                    break
        mask_stem_to_image_stem[mstem] = candidate

    # counts
    matched_image_stems: set[str] = set()
    masks_per_image: Counter[str] = Counter()
    masks_with_no_image: list[str] = []

    for mstem, target in mask_stem_to_image_stem.items():
        if target in image_stems:
            matched_image_stems.add(target)
            masks_per_image[target] += 1
        else:
            # also try parent-folder-based match: image might be in images/plant_X/foo.png
            # and mask in masks/plant_X/foo_mask.png
            masks_with_no_image.append(mstem)

    images_with_no_mask = [s for s in image_stems if s not in matched_image_stems]

    ratio_dist = Counter(masks_per_image.values())

    return {
        "num_images": len(image_paths),
        "num_masks": len(mask_paths),
        "ratio_masks_per_image": round(len(mask_paths) / len(image_paths), 3) if image_paths else 0,
        "matched_pairs": len(matched_image_stems),
        "images_with_at_least_one_mask": len(matched_image_stems),
        "images_with_no_mask_count": len(images_with_no_mask),
        "images_with_no_mask_examples": images_with_no_mask[:5],
        "masks_with_no_image_count": len(masks_with_no_image),
        "masks_with_no_image_examples": masks_with_no_image[:5],
        "masks_per_image_distribution": {f"{k}_masks": v for k, v in sorted(ratio_dist.items())},
        "suffix_patterns_checked": suffix_patterns,
    }


def walk_dataset(root: Path) -> dict:
    """Walk the root directory and categorise files by extension and folder.

    Args:
        root: The root directory to walk.

    Returns:
        A dictionary containing extension counts, folder counts, and file paths.
    """
    ext_counts: Counter[str] = Counter()
    folder_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_image_paths: list[Path] = []
    all_mask_paths: list[Path] = []
    total_bytes = 0

    for dirpath, _, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            ext_counts[ext] += 1
            folder_counts[rel_dir][ext] += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
            if ext in IMAGE_EXTS:
                if is_mask_path(p):
                    all_mask_paths.append(p)
                else:
                    all_image_paths.append(p)

    return {
        "ext_counts": dict(ext_counts),
        "folder_counts": {k: dict(v) for k, v in folder_counts.items()},
        "image_paths": all_image_paths,
        "mask_paths": all_mask_paths,
        "total_bytes": total_bytes,
    }


def sample_and_inspect(paths: list[Path], n: int, label: str) -> dict:
    """Sample a subset of files and inspect their image metadata.

    Args:
        paths: List of paths to sample from.
        n: Number of samples to take.
        label: Label for the report (e.g., "images" or "masks").

    Returns:
        A dictionary containing distribution stats and unreadable file counts.
    """
    if not paths:
        return {"label": label, "count": 0, "samples": [], "unique_values": None}

    random.seed(42)
    sample = random.sample(paths, min(n, len(paths)))
    samples_info = []
    shape_counter: Counter[tuple] = Counter()
    dtype_counter: Counter[str] = Counter()
    all_unique_vals: set[int] = set()
    unreadable: list[str] = []

    for p in sample:
        arr = load_image(p)
        if arr is None:
            unreadable.append(str(p))
            continue
        info = summarise_array(arr)
        info["path"] = str(p.name)
        samples_info.append(info)
        shape_counter[tuple(info["shape"])] += 1
        dtype_counter[info["dtype"]] += 1
        if label == "masks":
            # capture unique values for mask encoding analysis
            try:
                uniq = np.unique(arr)
                if uniq.size <= 32:  # only bother if low cardinality
                    all_unique_vals.update(uniq.tolist())
            except Exception:
                pass

    return {
        "label": label,
        "count": len(paths),
        "sampled": len(samples_info),
        "samples": samples_info,
        "shape_distribution": {str(k): v for k, v in shape_counter.most_common()},
        "dtype_distribution": dict(dtype_counter),
        "unique_values_across_samples": sorted(all_unique_vals) if label == "masks" else None,
        "unreadable": unreadable,
    }


def format_report(result: dict, root: Path) -> str:
    """Generate a markdown report from the inventory results.

    Args:
        result: The inventory results dictionary.
        root: The root directory that was inventoried.

    Returns:
        A formatted markdown string.
    """
    walk = result["walk"]
    img_rep = result["image_report"]
    mask_rep = result["mask_report"]

    lines: list[str] = []
    lines.append("# Dataset inventory report\n")
    lines.append(f"**Root:** `{root}`  ")
    lines.append(f"**Total size:** {walk['total_bytes'] / 1e9:.2f} GB  ")
    lines.append(f"**Total files:** {sum(walk['ext_counts'].values())}\n")

    lines.append("## File counts by extension\n")
    lines.append("| Extension | Count |")
    lines.append("|---|---|")
    for ext, c in sorted(walk["ext_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{ext or '(none)'}` | {c} |")

    lines.append("\n## Top-level folders\n")
    lines.append("| Folder | File count |")
    lines.append("|---|---|")
    folder_totals = [(f, sum(v.values())) for f, v in walk["folder_counts"].items()]
    for folder, total in sorted(folder_totals, key=lambda x: -x[1])[:20]:
        lines.append(f"| `{folder}` | {total} |")

    for rep, heading in [(img_rep, "Images"), (mask_rep, "Masks")]:
        lines.append(f"\n## {heading}\n")
        lines.append(f"- **Total found:** {rep['count']}")
        if rep["count"] == 0:
            continue
        lines.append(f"- **Sampled:** {rep['sampled']}")
        lines.append(f"- **Shape distribution (sampled):** `{rep['shape_distribution']}`")
        lines.append(f"- **Dtype distribution (sampled):** `{rep['dtype_distribution']}`")
        if rep["unreadable"]:
            lines.append(f"- **Unreadable files in sample:** {len(rep['unreadable'])}")
            for u in rep["unreadable"][:5]:
                lines.append(f"    - `{u}`")
        if heading == "Masks" and rep.get("unique_values_across_samples") is not None:
            lines.append(f"- **Unique mask values across samples:** `{rep['unique_values_across_samples']}`")

    pairing = result["pairing"]
    lines.append("\n## Image ↔ mask pairing\n")
    lines.append(f"- **Images:** {pairing['num_images']}")
    lines.append(f"- **Masks:** {pairing['num_masks']}")
    lines.append(f"- **Ratio:** {pairing['ratio_masks_per_image']} masks per image")
    lines.append(f"- **Images with ≥ 1 matching mask:** {pairing['images_with_at_least_one_mask']}")
    lines.append(f"- **Images with no matching mask:** {pairing['images_with_no_mask_count']}")
    if pairing["images_with_no_mask_examples"]:
        lines.append("    - examples:")
        for s in pairing["images_with_no_mask_examples"]:
            lines.append(f"        - `{s}`")
    lines.append(f"- **Masks with no matching image:** {pairing['masks_with_no_image_count']}")
    if pairing["masks_with_no_image_examples"]:
        lines.append("    - examples:")
        for s in pairing["masks_with_no_image_examples"]:
            lines.append(f"        - `{s}`")
    lines.append(f"- **Distribution of masks per image:** `{pairing['masks_per_image_distribution']}`")
    lines.append(f"- **Suffix patterns checked when matching:** `{pairing['suffix_patterns_checked']}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the inventory script.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Path to the dataset root")
    ap.add_argument("--sample", type=int, default=150, help="Number of images/masks to sample-inspect (default 150)")
    ap.add_argument("--out", type=Path, default=None, help="Optional output .md file")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional raw JSON output")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"ERROR: {args.root} does not exist", file=sys.stderr)
        return 1

    print(f"Walking {args.root} ...")
    walk = walk_dataset(args.root)
    print(f"  found {len(walk['image_paths'])} images, {len(walk['mask_paths'])} masks")
    print(f"Sampling {args.sample} images and {args.sample} masks ...")
    image_report = sample_and_inspect(walk["image_paths"], args.sample, "images")
    mask_report = sample_and_inspect(walk["mask_paths"], args.sample, "masks")
    print("Analysing image ↔ mask pairing ...")
    pairing = analyse_pairing(walk["image_paths"], walk["mask_paths"])
    print(f"  ratio: {pairing['ratio_masks_per_image']} masks per image")
    print(f"  distribution: {pairing['masks_per_image_distribution']}")

    result = {"walk": walk, "image_report": image_report, "mask_report": mask_report, "pairing": pairing}

    # strip Path objects before serialising
    walk_serialisable = {k: v for k, v in walk.items() if k not in {"image_paths", "mask_paths"}}
    result_serialisable = {
        "walk": walk_serialisable,
        "image_report": image_report,
        "mask_report": mask_report,
        "pairing": pairing,
    }

    report = format_report(result, args.root)
    print("\n" + report)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.out}")

    if args.json_out:
        args.json_out.write_text(json.dumps(result_serialisable, indent=2, default=str), encoding="utf-8")
        print(f"Raw JSON written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
