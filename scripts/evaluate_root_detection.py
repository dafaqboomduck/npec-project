"""Azure ML evaluation script — root detection U-Net patch-based inference.

Usage (submitted by Azure ML, not run manually):
    python scripts/evaluate_root_detection.py
        --data_dir   <mounted path to cv5_test_images>
        --model_dir  <mounted path to registered root-detection model>
        --output_dir <path where results are written>
        [--patch_size 512]
        [--threshold 0.5]
"""

import argparse
import glob as glob_module
import os
import sys
import types
from pathlib import Path

import cv2
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

# Make repo packages importable (Azure ML sets CWD to the uploaded code root).
# cv2/tf/sklearn are in the conda env and don't need this; only local repo
# modules do. The npec_cv stub prevents its heavy __init__.py from running.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO_ROOT))

_npec_stub = types.ModuleType("npec_cv")
_npec_stub.__path__ = [str(_SRC / "npec_cv")]
_npec_stub.__package__ = "npec_cv"
sys.modules.setdefault("npec_cv", _npec_stub)

_IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation job."""
    parser = argparse.ArgumentParser("evaluate_root_detection")
    parser.add_argument("--data_dir", required=True, help="Mounted path to cv5_test_images")
    parser.add_argument("--model_dir", required=True, help="Mounted path to root-detection model dir")
    parser.add_argument("--output_dir", default="outputs", help="Directory to write results")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


# ── Directory detection ────────────────────────────────────────────────────────


def _find_test_dirs(data_dir: str) -> tuple[str, str | None]:
    """Return (image_dir, mask_dir_or_None).

    Checks common subfolder layouts first. If only images are found (no
    separate mask folder), returns None for mask_dir so the caller can
    fall back to inference-only mode.
    """
    all_entries = os.listdir(data_dir)
    print(f"Test data root: {data_dir}")
    print(f"Contents ({len(all_entries)} entries): {all_entries[:30]}")

    # Check subdir layouts
    for img_sub, msk_sub in [
        ("images", "masks"),
        ("cropped_images", "cropped_masks"),
        ("test_images", "test_masks"),
    ]:
        img_d = os.path.join(data_dir, img_sub)
        msk_d = os.path.join(data_dir, msk_sub)
        if os.path.isdir(img_d) and os.path.isdir(msk_d):
            return img_d, msk_d

    # Flat layout: split by naming convention (files with 'mask' in stem are masks)
    img_files = [f for f in all_entries if Path(f).suffix.lower() in _IMG_EXTS and "mask" not in f.lower()]
    msk_files = [f for f in all_entries if Path(f).suffix.lower() in _IMG_EXTS and "mask" in f.lower()]

    print(f"Flat layout — image files: {len(img_files)}, mask files: {len(msk_files)}")

    if img_files and msk_files:
        # Both found in the same flat folder
        return data_dir, data_dir

    if img_files and not msk_files:
        # Images only — inference-only mode
        print("No mask files found. Running in inference-only mode (no F1 metrics).")
        return data_dir, None

    raise FileNotFoundError(f"Cannot find any image files inside {data_dir}.\n" f"Contents: {all_entries}")


def _find_model_file(model_dir: str) -> str:
    """Find the .keras or .h5 model file inside the model directory."""
    for ext in ("*.keras", "*.h5"):
        matches = glob_module.glob(os.path.join(model_dir, "**", ext), recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No .keras or .h5 file found in {model_dir}.\n" f"Contents: {os.listdir(model_dir)}")


def _list_images(folder: str) -> list[str]:
    """Return sorted list of image file paths from a folder."""
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if Path(f).suffix.lower() in _IMG_EXTS)


# ── Inference helpers ─────────────────────────────────────────────────────────


def _pad(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Zero-pad image so H and W are multiples of patch_size."""
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def predict_full_mask(image: np.ndarray, model: tf.keras.Model, patch_size: int) -> np.ndarray:
    """Run patch-based inference and stitch back to original image size."""
    original_h, original_w = image.shape[:2]
    padded = _pad(image, patch_size)
    h, w = padded.shape[:2]
    full_pred = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = padded[y : y + patch_size, x : x + patch_size].astype(np.float32) / 255.0
            patch = np.expand_dims(patch, axis=0)
            full_pred[y : y + patch_size, x : x + patch_size] = model.predict(patch, verbose=0)[0, :, :, 0]

    return full_pred[:original_h, :original_w]


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float, float]:
    """Return (f1, precision, recall) for a single image pair."""
    gt = (gt_mask > 0).astype(np.uint8).flatten()
    pred = (pred_mask > 0).astype(np.uint8).flatten()
    return (
        float(f1_score(gt, pred, zero_division=0)),
        float(precision_score(gt, pred, zero_division=0)),
        float(recall_score(gt, pred, zero_division=0)),
    )


# ── Pair matching ─────────────────────────────────────────────────────────────


def _match_pairs(img_dir: str, msk_dir: str) -> list[tuple[str, str]]:
    """Match image files to mask files by stem name.

    Tries exact stem match first, then stem + '_root_mask' suffix
    (same logic as the training patchify).
    """
    img_files = {
        Path(f).stem.lower(): os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if Path(f).suffix.lower() in _IMG_EXTS
    }

    msk_files: dict[str, str] = {}
    for root, _, files in os.walk(msk_dir):
        for f in files:
            if Path(f).suffix.lower() in _IMG_EXTS:
                msk_files[Path(f).stem.lower()] = os.path.join(root, f)

    print(f"Image files found : {len(img_files)}  e.g. {list(img_files.keys())[:3]}")
    print(f"Mask  files found : {len(msk_files)}  e.g. {list(msk_files.keys())[:3]}")

    pairs = []
    for stem, img_path in sorted(img_files.items()):
        if stem in msk_files:
            pairs.append((img_path, msk_files[stem]))
        elif stem + "_root_mask" in msk_files:
            pairs.append((img_path, msk_files[stem + "_root_mask"]))

    print(f"Matched pairs     : {len(pairs)}")
    if not pairs:
        raise ValueError(
            "No image-mask pairs matched.\n"
            f"Image stems (sample): {list(img_files.keys())[:5]}\n"
            f"Mask  stems (sample): {list(msk_files.keys())[:5]}"
        )
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────


def _list_image_paths(folder: str) -> list[str]:
    """Return sorted list of image paths that are not mask files."""
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in _IMG_EXTS and "mask" not in f.lower()
    )


def main() -> None:
    """Run patch-based evaluation (with metrics) or inference-only (no GT masks)."""
    args = parse_args()

    img_dir, msk_dir = _find_test_dirs(args.data_dir)
    inference_only = msk_dir is None

    model_path = _find_model_file(args.model_dir)
    print(f"Model : {model_path}")
    print(f"Images: {img_dir}")
    print(f"Masks : {'N/A — inference-only mode' if inference_only else msk_dir}")

    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    if inference_only:
        print("U-NET ROOT SEGMENTATION — INFERENCE ONLY (no GT masks found)")
    else:
        print("U-NET ROOT SEGMENTATION EVALUATION")
    print("=" * 60)

    with mlflow.start_run():
        mlflow.log_param("patch_size", args.patch_size)
        mlflow.log_param("threshold", args.threshold)

        if inference_only:
            # ── No GT masks: just predict and save ───────────────────────────
            img_paths = _list_image_paths(img_dir)
            mlflow.log_param("mode", "inference_only")
            mlflow.log_param("n_images", len(img_paths))
            print(f"Running inference on {len(img_paths)} images...")

            for img_path in img_paths:
                name = Path(img_path).name
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  SKIP {name} — failed to load")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pred_prob = predict_full_mask(image, model, args.patch_size)
                pred_mask = ((pred_prob > args.threshold) * 255).astype(np.uint8)
                print(f"  {name}  max_pred={pred_mask.max()}")

            print("\nInference complete. No metrics logged (no GT masks).")
            print("NOTE: To get F1/precision/recall, add a masks/ subfolder to cv5_test_images.")

        else:
            # ── Full evaluation with GT masks ─────────────────────────────────
            pairs = _match_pairs(img_dir, msk_dir)
            mlflow.log_param("mode", "evaluation")
            mlflow.log_param("n_images", len(pairs))

            f1_scores, precision_scores, recall_scores = [], [], []

            for img_path, msk_path in pairs:
                name = Path(img_path).name

                image = cv2.imread(img_path)
                if image is None:
                    print(f"  SKIP {name} — failed to load image")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                gt_mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is None:
                    print(f"  SKIP {name} — failed to load mask")
                    continue
                gt_mask = (gt_mask > 0).astype(np.uint8)

                pred_prob = predict_full_mask(image, model, args.patch_size)
                pred_mask = (pred_prob > args.threshold).astype(np.uint8)

                f1, precision, recall = compute_metrics(gt_mask, pred_mask)
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)
                print(f"{name:50s}  F1={f1:.3f}  P={precision:.3f}  R={recall:.3f}")

            if not f1_scores:
                raise RuntimeError("No images were evaluated — check data paths and file formats.")

            mean_f1 = float(np.mean(f1_scores))
            mean_precision = float(np.mean(precision_scores))
            mean_recall = float(np.mean(recall_scores))

            mlflow.log_metric("mean_f1", mean_f1)
            mlflow.log_metric("mean_precision", mean_precision)
            mlflow.log_metric("mean_recall", mean_recall)

            print("-" * 60)
            print(f"MEAN F1       : {mean_f1:.4f}")
            print(f"MEAN PRECISION: {mean_precision:.4f}")
            print(f"MEAN RECALL   : {mean_recall:.4f}")
            print("=" * 60)

            results_path = os.path.join(args.output_dir, "eval_results.txt")
            with open(results_path, "w") as fh:
                fh.write(f"mean_f1={mean_f1:.6f}\n")
                fh.write(f"mean_precision={mean_precision:.6f}\n")
                fh.write(f"mean_recall={mean_recall:.6f}\n")
                fh.write(f"n_images={len(f1_scores)}\n")
            print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
