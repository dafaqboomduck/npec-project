"""Run the root-detection model locally on a single image.

Bypasses PlantMaster, anchors, zones, dish detection — just feeds the image
to the root .keras model and saves the raw probability + mask outputs.
Useful for iterating on preprocessing changes without redeploying.

Mirrors the preprocessing the deployed score.py currently does:
    BGR -> RGB  ->  /255  ->  pad bottom/right  ->  512x512 patches (no overlap)
    -> model.predict  ->  stitch  ->  crop to original size

Outputs (saved next to the input image):
    <stem>_heatmap.png  — raw probabilities as uint8 (0=black, 255=high conf)
    <stem>_mask.png     — binary mask at the chosen threshold
    <stem>_overlay.png  — red mask blended onto the original image

Prints a left/right mean comparison so you can spot bias quickly.

Usage:
    python scripts/test_model_local.py --image path/to/img.png
    python scripts/test_model_local.py --image path/to/img.png --threshold 0.3
    python scripts/test_model_local.py --image path/to/img.png --no_rgb
    python scripts/test_model_local.py --image path/to/img.png --clahe

Flags exist for experimenting:
    --no_rgb     Skip BGR->RGB (reproduces the pre-fix behaviour)
    --clahe      Apply CLAHE before /255 (for testing the contrast-norm idea)
    --threshold  Binarisation threshold for the saved mask (default 0.5)
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "scripts" / "root_det_model_512px.keras"
PATCH_SIZE = 512
ROOT_MODEL_NAME = "root-detection"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("test_model_local")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image (PNG/TIFF/JPEG)")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help=f"Path to .keras model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Binarisation threshold for the saved mask (default 0.5)"
    )
    parser.add_argument(
        "--patch_size", type=int, default=PATCH_SIZE, help=f"Patch size in pixels (default {PATCH_SIZE})"
    )
    parser.add_argument("--no_rgb", action="store_true", help="Skip BGR->RGB conversion (reproduces pre-fix behaviour)")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE for contrast normalisation before /255")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the latest registered root-detection model " "from Azure ML to scripts/ before running",
    )
    return parser.parse_args()


def download_latest_root_model(dest: Path) -> None:
    """Pull the latest registered root-detection model from Azure ML."""
    import json
    import shutil

    _az_wbin = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
    if _az_wbin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _az_wbin + os.pathsep + os.environ.get("PATH", "")

    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    config = json.loads((REPO_ROOT / "config.json").read_text())
    client = MLClient(
        credential=AzureCliCredential(process_timeout=30),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )

    versions = list(client.models.list(name=ROOT_MODEL_NAME))
    if not versions:
        raise RuntimeError(f"No registered versions found for '{ROOT_MODEL_NAME}'.")
    latest = max(versions, key=lambda m: int(m.version))
    print(f"Downloading {ROOT_MODEL_NAME} v{latest.version} from Azure ML...")

    tmp_dir = dest.parent / f"_{ROOT_MODEL_NAME}_dl"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    client.models.download(name=ROOT_MODEL_NAME, version=latest.version, download_path=str(tmp_dir))

    # Find the .keras file inside (Azure nests it under <model_name>/...)
    keras_files = list(tmp_dir.rglob("*.keras"))
    if not keras_files:
        raise FileNotFoundError(f"No .keras file inside downloaded model at {tmp_dir}")
    shutil.move(str(keras_files[0]), str(dest))
    shutil.rmtree(tmp_dir)
    print(f"Saved to: {dest}")


def load_model(path: Path):
    """Load the .keras model, trying Keras 3 first then tf_keras (Keras 2)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run with --download to pull the latest version from Azure ML, "
            f"or pass --model with a valid path."
        )

    last_err = None
    for loader_name in ("keras", "tf_keras", "tensorflow.keras"):
        try:
            if loader_name == "keras":
                import keras as _k
            elif loader_name == "tf_keras":
                import tf_keras as _k
            else:
                from tensorflow import keras as _k
            model = _k.models.load_model(str(path), compile=False)
            print(f"  Loaded with {loader_name}")
            return model
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Could not load model with any backend. Last error: {last_err}")


def pad_bottom_right(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Zero-pad on bottom + right only. Matches train_root_detection.py."""
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return image
    if image.ndim == 2:
        return np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def apply_clahe(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """CLAHE on luminance, then convert back to RGB."""
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def predict_full_mask(img_norm: np.ndarray, model, patch_size: int, batch_size: int = 8) -> tuple:
    """Patchify (non-overlapping), predict in one batched call, stitch back."""
    padded = pad_bottom_right(img_norm, patch_size)
    h, w = padded.shape[:2]

    coords = [(y, x) for y in range(0, h, patch_size) for x in range(0, w, patch_size)]
    patches = np.stack(
        [padded[y : y + patch_size, x : x + patch_size] for (y, x) in coords],
        axis=0,
    )
    print(f"  Predicting {len(patches)} patches in one batched call (batch_size={batch_size})...")
    preds = model.predict(patches, batch_size=batch_size, verbose=1)

    full = np.zeros((h, w), dtype=np.float32)
    for (y, x), pred in zip(coords, preds):
        full[y : y + patch_size, x : x + patch_size] = pred[:, :, 0]
    return full, len(patches)


def main() -> None:
    """Entry point."""
    args = parse_args()

    # ── Optionally fetch the latest model from Azure ML ──────────────────────
    if args.download or not args.model.exists():
        if args.download:
            print(f"--download set; fetching latest {ROOT_MODEL_NAME} from Azure ML")
        else:
            print(f"Model not found at {args.model}; attempting download")
        download_latest_root_model(args.model)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    # ── Load + preprocess image (mirror score.py) ────────────────────────────
    print(f"Loading image: {args.image}")
    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {args.image}")
    h, w = img_bgr.shape[:2]
    print(f"  Resolution: {w}x{h}")

    if args.no_rgb:
        print("  Skipping BGR->RGB conversion (--no_rgb)")
        img_rgb = img_bgr
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if args.clahe:
        print("  Applying CLAHE (luminance, clipLimit=2.0, tile=8x8)")
        img_rgb = apply_clahe(img_rgb)

    img_norm = img_rgb.astype(np.float32) / 255.0

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"Running inference (patch_size={args.patch_size})...")
    probs, n_patches = predict_full_mask(img_norm, model, args.patch_size)
    probs = probs[:h, :w]  # crop padding back off
    print(f"  {n_patches} patches predicted")

    # ── Diagnostic: bias check ───────────────────────────────────────────────
    half_w = w // 2
    left, right = probs[:, :half_w], probs[:, half_w:]
    ratio = (right.mean() + 1e-9) / (left.mean() + 1e-9)
    print()
    print("DIAGNOSTIC (root_probs)")
    print(f"  left   : max={left.max():.4f}  mean={left.mean():.5f}")
    print(f"  right  : max={right.max():.4f}  mean={right.mean():.5f}")
    print(f"  ratio  : right/left = {ratio:.2f}x   " f"({'symmetric' if 0.5 < ratio < 2.0 else 'biased'})")

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir = args.image.parent
    stem = args.image.stem

    heatmap_path = out_dir / f"{stem}_heatmap.png"
    cv2.imwrite(str(heatmap_path), (probs * 255).astype(np.uint8))

    mask = (probs > args.threshold).astype(np.uint8) * 255
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    overlay = img_bgr.copy()
    overlay[mask > 0] = [0, 0, 255]
    blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), blended)

    print()
    print(f"Heatmap  : {heatmap_path}")
    print(f"Mask     : {mask_path}   (threshold > {args.threshold})")
    print(f"Overlay  : {overlay_path}")


if __name__ == "__main__":
    main()
