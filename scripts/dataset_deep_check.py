"""Deep-dive checks on the HADES/NPEC dataset, run AFTER inventory_dataset.py.

Answers three questions the basic inventory can't:

1. Are the "unreadable" TIFFs genuinely corrupt, or just a PIL limitation?
   → retries with tifffile, reports which are truly dead.

2. Which images are missing which class mask (root / shoot / seed)?
   → per-image coverage map and summary, focus on root for training.

3. Is the annotator partition clean (no image belongs to two annotators)?
   → confirms the "disjoint slice" assumption.

Usage:
    python dataset_deep_check.py <root_path> [--out report.md]

Assumes the layout seen in the real inventory:
    Y2B_*/images/...      or Y2B_*/images/{train,test}/...
    Y2B_*/masks/{annotator}/...   (or Y2B_23/masks/ flat)
    mask filenames end in _root_mask.tif / _shoot_mask.tif / _seed_mask.tif
    (also handles _occluded_root_mask which is treated as a distinct class here)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# Class tokens we care about. Order matters for reporting.
CLASS_TOKENS = ("root", "shoot", "seed", "occluded_root")
MASK_SUFFIX_RE = re.compile(r"_(occluded_root|root|shoot|seed)_mask\.tif$", re.IGNORECASE)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def try_read(path: Path) -> tuple[bool, str]:
    """Try PIL first, then tifffile. Return (readable, reader_used)."""
    try:
        with Image.open(path) as im:
            im.load()
        return True, "PIL"
    except Exception:
        pass
    if HAS_TIFFFILE:
        try:
            tifffile.imread(str(path))
            return True, "tifffile"
        except Exception:
            pass
    return False, "none"


def parse_mask_name(mask_path: Path) -> tuple[str, str] | None:
    """Return (class_token, image_stem) or None if the filename doesn't match.

    class_token is one of CLASS_TOKENS. image_stem is everything before the class suffix.
    """
    m = MASK_SUFFIX_RE.search(mask_path.name)
    if not m:
        return None
    class_token = m.group(1).lower()
    stem = mask_path.name[: m.start()]
    return class_token, stem


def cohort_of(path: Path, root: Path) -> str:
    """Return top-level cohort folder name (e.g. 'Y2B_23') or 'other'."""
    rel = path.relative_to(root)
    parts = rel.parts
    if parts and parts[0].startswith("Y2B_"):
        return parts[0]
    return parts[0] if parts else "other"


def annotator_of(mask_path: Path, root: Path) -> str:
    """Return annotator folder name if the mask lives under masks/<annotator>/."""
    rel = mask_path.relative_to(root)
    parts = rel.parts
    # looking for .../masks/<annotator>/file.tif
    for i, part in enumerate(parts):
        if part.lower() == "masks" and i + 1 < len(parts) - 0:
            # next part is either an annotator folder or the file itself
            nxt = parts[i + 1]
            if "." not in nxt:  # folder, not file
                return nxt
    return "(flat)"  # masks sit directly in masks/


def walk(root: Path) -> tuple[list[Path], list[Path]]:
    images, masks = [], []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            # it's a mask if filename matches the mask suffix pattern
            if MASK_SUFFIX_RE.search(fn):
                masks.append(p)
            else:
                images.append(p)
    return images, masks


def check_unreadable(masks: list[Path]) -> dict:
    """Test every mask, not just a sample. Report recovery via tifffile."""
    pil_fail_tifffile_ok: list[str] = []
    pil_fail_tifffile_fail: list[str] = []
    pil_ok = 0

    for i, p in enumerate(masks, 1):
        if i % 200 == 0:
            print(f"  checked {i}/{len(masks)} masks ...", file=sys.stderr)
        try:
            with Image.open(p) as im:
                im.load()
            pil_ok += 1
            continue
        except Exception:
            pass
        if HAS_TIFFFILE:
            try:
                tifffile.imread(str(p))
                pil_fail_tifffile_ok.append(str(p))
                continue
            except Exception:
                pass
        pil_fail_tifffile_fail.append(str(p))

    return {
        "total": len(masks),
        "pil_ok": pil_ok,
        "pil_fail_tifffile_ok": pil_fail_tifffile_ok,
        "pil_fail_tifffile_fail": pil_fail_tifffile_fail,
        "recovery_rate": (len(pil_fail_tifffile_ok) / max(1, len(pil_fail_tifffile_ok) + len(pil_fail_tifffile_fail))),
    }


def class_coverage(masks: list[Path], root: Path) -> dict:
    """For each image stem, which class masks exist? Per cohort summary."""
    per_stem: dict[str, set[str]] = defaultdict(set)
    stem_to_cohort: dict[str, str] = {}
    unmatched: list[str] = []

    for m in masks:
        parsed = parse_mask_name(m)
        if parsed is None:
            unmatched.append(str(m))
            continue
        cls, stem = parsed
        per_stem[stem].add(cls)
        stem_to_cohort.setdefault(stem, cohort_of(m, root))

    # summarise
    per_cohort: dict[str, Counter] = defaultdict(Counter)
    root_mask_count_by_cohort: Counter = Counter()
    occluded_root_by_cohort: Counter = Counter()
    for stem, classes in per_stem.items():
        coh = stem_to_cohort[stem]
        per_cohort[coh][frozenset(classes)] += 1
        if "root" in classes:
            root_mask_count_by_cohort[coh] += 1
        if "occluded_root" in classes:
            occluded_root_by_cohort[coh] += 1

    # readable summary
    coverage_table = []
    for coh, c in per_cohort.items():
        for combo, n in c.most_common():
            coverage_table.append(
                {
                    "cohort": coh,
                    "classes_present": sorted(combo),
                    "num_images": n,
                }
            )

    return {
        "unique_image_stems_with_masks": len(per_stem),
        "unmatched_mask_filenames": unmatched[:10],
        "num_unmatched": len(unmatched),
        "root_mask_count_by_cohort": dict(root_mask_count_by_cohort),
        "occluded_root_by_cohort": dict(occluded_root_by_cohort),
        "total_root_masks": sum(root_mask_count_by_cohort.values()),
        "total_occluded_root_masks": sum(occluded_root_by_cohort.values()),
        "coverage_by_cohort": coverage_table,
    }


def annotator_partition_check(masks: list[Path], root: Path) -> dict:
    """For each image stem, which annotators produced masks for it?

    If partitioning is clean, every stem should have exactly one annotator.
    """
    stem_to_annotators: dict[str, set[str]] = defaultdict(set)
    annotator_counts: Counter = Counter()
    for m in masks:
        parsed = parse_mask_name(m)
        if parsed is None:
            continue
        _, stem = parsed
        ann = annotator_of(m, root)
        stem_to_annotators[stem].add(ann)
        annotator_counts[ann] += 1

    overlaps = {s: sorted(a) for s, a in stem_to_annotators.items() if len(a) > 1}

    return {
        "annotator_mask_counts": dict(annotator_counts),
        "num_stems_with_single_annotator": sum(1 for a in stem_to_annotators.values() if len(a) == 1),
        "num_stems_with_multiple_annotators": len(overlaps),
        "example_overlaps": dict(list(overlaps.items())[:5]),
    }


def format_report(unread: dict, cov: dict, part: dict) -> str:
    L: list[str] = []
    L.append("# Dataset deep-check report\n")

    L.append("## 1. TIFF readability (full scan)\n")
    L.append(f"- **Masks checked:** {unread['total']}")
    L.append(f"- **Readable by PIL:** {unread['pil_ok']}")
    L.append(f"- **PIL fails, `tifffile` recovers:** {len(unread['pil_fail_tifffile_ok'])}")
    L.append(f"- **Genuinely unreadable (both fail):** {len(unread['pil_fail_tifffile_fail'])}")
    if unread["pil_fail_tifffile_fail"]:
        L.append("\n**Dead files:**")
        for p in unread["pil_fail_tifffile_fail"]:
            L.append(f"  - `{p}`")

    L.append("\n## 2. Class mask coverage\n")
    L.append(f"- **Image stems that have at least one mask:** {cov['unique_image_stems_with_masks']}")
    L.append(f"- **Total root masks across all cohorts:** {cov['total_root_masks']}")
    L.append(f"- **Total occluded_root masks (excluded from training):** {cov['total_occluded_root_masks']}")
    if cov["num_unmatched"]:
        L.append(f"- **Mask filenames that didn't match the expected pattern:** {cov['num_unmatched']}")
        for f in cov["unmatched_mask_filenames"]:
            L.append(f"    - `{f}`")

    L.append("\n### Root masks available per cohort\n")
    L.append("| Cohort | Images with a root mask | Images with occluded_root mask |")
    L.append("|---|---|---|")
    cohorts = sorted(set(list(cov["root_mask_count_by_cohort"].keys()) + list(cov["occluded_root_by_cohort"].keys())))
    for coh in cohorts:
        L.append(
            f"| {coh} | {cov['root_mask_count_by_cohort'].get(coh, 0)} | {cov['occluded_root_by_cohort'].get(coh, 0)} |"
        )

    L.append("\n### Class combinations present per cohort\n")
    L.append("| Cohort | Classes present | Image count |")
    L.append("|---|---|---|")
    for row in sorted(cov["coverage_by_cohort"], key=lambda r: (r["cohort"], -r["num_images"])):
        L.append(f"| {row['cohort']} | {', '.join(row['classes_present'])} | {row['num_images']} |")

    L.append("\n## 3. Annotator partition check\n")
    L.append(f"- **Image stems with masks from a single annotator:** {part['num_stems_with_single_annotator']}")
    L.append(
        f"- **Image stems with masks from multiple annotators (overlap):** {part['num_stems_with_multiple_annotators']}"
    )
    if part["num_stems_with_multiple_annotators"] == 0:
        L.append("- ✅ **Partition is clean** — annotators can be concatenated safely.")
    else:
        L.append("- ⚠️ **Partition is NOT clean** — examples:")
        for stem, anns in part["example_overlaps"].items():
            L.append(f"    - `{stem}` → {anns}")

    L.append("\n**Masks per annotator:**\n")
    L.append("| Annotator | Mask count |")
    L.append("|---|---|")
    for ann, n in sorted(part["annotator_mask_counts"].items(), key=lambda x: -x[1]):
        L.append(f"| {ann} | {n} |")

    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-readability", action="store_true", help="skip the full TIFF read scan (slow)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"ERROR: {args.root} does not exist", file=sys.stderr)
        return 1
    if not HAS_TIFFFILE:
        print(
            "WARNING: tifffile is not installed, recovery check will be skipped. Install with: pip install tifffile",
            file=sys.stderr,
        )

    print(f"Walking {args.root} ...")
    images, masks = walk(args.root)
    print(f"  found {len(images)} images, {len(masks)} masks")

    if args.skip_readability:
        unread = {"total": 0, "pil_ok": 0, "pil_fail_tifffile_ok": [], "pil_fail_tifffile_fail": [], "recovery_rate": 0}
    else:
        print("Checking TIFF readability (this scans every mask) ...")
        unread = check_unreadable(masks)

    print("Analysing class coverage ...")
    cov = class_coverage(masks, args.root)

    print("Checking annotator partition ...")
    part = annotator_partition_check(masks, args.root)

    report = format_report(unread, cov, part)
    print("\n" + report)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
