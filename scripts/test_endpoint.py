"""Test the root-detection managed online endpoint.

Two modes:

  Local (recommended for debugging score.py):
      python scripts/test_endpoint.py --image <path/to/image.png> --local

  Both scripts/root_det_model_512px.keras AND scripts/anchor_det_model_512px.keras
  must be present locally. score.py picks up the anchor file next to itself,
  so no Azure call is made. Pass --root_model to point at a different
  root .keras file.

  Remote (default — hits the deployed endpoint):
      python scripts/test_endpoint.py --image <path/to/image.png>

Options:
  --save_mask   Decode the base64 segmentation mask and save it next to the
                input image as <name>_mask.png. Also saves an overlay PNG
                showing the mask blended on the original image.
  --pretty      Pretty-print the full JSON response.
  --root_model  (Local mode only) Path to root .keras file. Default:
                scripts/root_det_model_512px.keras
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_ROOT_MODEL = SCRIPTS_DIR / "root_det_model_512px.keras"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("test_endpoint")
    parser.add_argument("--image", required=True, help="Path to a test image (PNG / TIFF / JPEG)")
    parser.add_argument(
        "--save_mask", action="store_true", help="Save returned segmentation mask + overlay PNG next to the image"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the full JSON response")
    parser.add_argument(
        "--local", action="store_true", help="Run score.py locally instead of hitting the deployed endpoint"
    )
    parser.add_argument(
        "--root_model", type=Path, default=DEFAULT_LOCAL_ROOT_MODEL, help="(Local mode only) path to root .keras file"
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def encode_image(image_path: Path) -> str:
    """Read image from disk and return base64-encoded string."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def build_payload(image_path: Path) -> str:
    """Build the JSON request body."""
    return json.dumps(
        {
            "image_b64": encode_image(image_path),
            "filename": image_path.name,
        }
    )


# ── Call modes ────────────────────────────────────────────────────────────────


def call_local(image_path: Path, root_model: Path) -> dict:
    """Import score.py and call init() / run() directly.

    Requirements:
      - scripts/anchor_det_model_512px.keras must exist (so score.py finds it
        locally and skips the registry download).
      - root_model file must exist; we set AZUREML_MODEL_DIR to its parent so
        score.py's recursive search finds it.
    """
    import importlib.util

    anchor_path = SCRIPTS_DIR / "anchor_det_model_512px.keras"
    if not anchor_path.exists():
        sys.exit(
            f"Anchor model not found at {anchor_path}. " f"Local mode requires the anchor .keras file next to score.py."
        )
    if not root_model.exists():
        sys.exit(f"Root model not found at {root_model}. " f"Pass --root_model with a valid path.")

    # Point AZUREML_MODEL_DIR at the root model's directory so score.py's
    # _find_root_model() recursive search picks it up.
    os.environ["AZUREML_MODEL_DIR"] = str(root_model.parent)
    os.environ.setdefault("MODEL_VERSION", "local-test")

    score_path = SCRIPTS_DIR / "score.py"
    spec = importlib.util.spec_from_file_location("score", score_path)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)

    print("Running init() ...")
    t_init = time.perf_counter()
    score.init()
    print(f"init() done in {time.perf_counter() - t_init:.1f}s")

    payload = build_payload(image_path)
    print("Running run() ...")
    t0 = time.perf_counter()
    result = score.run(payload)
    print(f"run() returned in {int((time.perf_counter() - t0) * 1000)} ms")

    return json.loads(result)


def call_remote(image_path: Path) -> dict:
    """Send a scored request to the deployed Azure ML endpoint."""
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    _az_wbin = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
    if _az_wbin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _az_wbin + os.pathsep + os.environ.get("PATH", "")

    config = json.loads((REPO_ROOT / "config.json").read_text())
    client = MLClient(
        credential=AzureCliCredential(process_timeout=30),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )

    # invoke() expects a file path, not a string body — write payload to temp file.
    import tempfile

    payload = build_payload(image_path)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write(payload)
        payload_file = fh.name

    print("Sending request to endpoint ...")
    t0 = time.perf_counter()
    try:
        response_str = client.online_endpoints.invoke(
            endpoint_name="root-detection-endpoint",
            deployment_name="current",
            request_file=payload_file,
        )
    finally:
        os.unlink(payload_file)
    print(f"Round-trip time: {int((time.perf_counter() - t0) * 1000)} ms")

    # The SDK sometimes wraps the response in an extra JSON layer (string
    # containing JSON). Unwrap until we get a dict.
    parsed = response_str
    for _ in range(3):
        if isinstance(parsed, dict):
            return parsed
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            break
    if isinstance(parsed, dict):
        return parsed
    return {"status": "error", "error_code": "NON_OBJECT_RESPONSE", "message": str(parsed)[:500]}


# ── Output ────────────────────────────────────────────────────────────────────


def print_summary(response: dict) -> None:
    """Print a human-readable summary of the endpoint response."""
    print("\n" + "=" * 60)
    status = response.get("status")
    print(f"Status             : {status}")

    if status == "error":
        print(f"Error code         : {response.get('error_code')}")
        print(f"Message            : {response.get('message')}")
        print("=" * 60)
        return

    meta = response.get("metadata", {})
    print(f"Model version      : {meta.get('model_version')}")
    print(f"Inference timestamp: {meta.get('inference_timestamp')}")
    print(f"Processing time    : {meta.get('processing_time_ms')} ms")
    print(f"Input resolution   : {meta.get('input_resolution')}")
    print(f"Patch count        : {meta.get('patch_count')}")
    print(f"Overall confidence : {response.get('overall_confidence')}")
    print(f"Plants detected    : {len(response.get('plants', []))}")

    for plant in response.get("plants", []):
        print(
            f"  Plant {plant['plant_id']}: "
            f"tip=({plant['tip_x']}, {plant['tip_y']})  "
            f"length={plant['root_length_px']:.1f} px / {plant['root_length_mm']} mm  "
            f"conf={plant['confidence']:.4f}"
        )

    warnings = response.get("warnings", [])
    if warnings:
        print("Warnings           :")
        for w in warnings:
            print(f"  - {w}")

    has_mask = bool(response.get("segmentation_mask_b64"))
    print(f"Mask in response   : {'yes' if has_mask else 'no'}")
    print("=" * 60)


def save_mask(response: dict, image_path: Path) -> None:
    """Decode the base64 mask and save it + an overlay PNG next to the image."""
    mask_b64 = response.get("segmentation_mask_b64", "")
    if not mask_b64:
        print("No segmentation mask in response — nothing to save.")
        return

    mask_bytes = base64.b64decode(mask_b64)
    arr = np.frombuffer(mask_bytes, np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    mask_path = image_path.parent / (image_path.stem + "_mask.png")
    cv2.imwrite(str(mask_path), mask)
    print(f"Mask saved: {mask_path}")

    # Overlay (red mask on original image) for quick visual inspection.
    original = cv2.imread(str(image_path))
    if original is None or original.shape[:2] != mask.shape:
        return
    overlay = original.copy()
    overlay[mask > 0] = [0, 0, 255]
    blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    # Draw tip markers
    for plant in response.get("plants", []):
        tx, ty = plant["tip_x"], plant["tip_y"]
        cv2.circle(blended, (tx, ty), 12, (0, 255, 255), 2)
        cv2.putText(blended, str(plant["plant_id"]), (tx + 14, ty + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    overlay_path = image_path.parent / (image_path.stem + "_overlay.png")
    cv2.imwrite(str(overlay_path), blended)
    print(f"Overlay saved: {overlay_path}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    image_path = Path(args.image).resolve()

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"Image : {image_path}  ({image_path.stat().st_size / 1024:.1f} KB)")
    print(f"Mode  : {'LOCAL' if args.local else 'REMOTE'}")

    if args.local:
        response = call_local(image_path, args.root_model)
    else:
        response = call_remote(image_path)

    print_summary(response)

    if args.pretty:
        # Hide the base64 mask blob to keep output readable
        printable = {k: v for k, v in response.items() if k != "segmentation_mask_b64"}
        printable["segmentation_mask_b64"] = "<base64 PNG omitted>" if response.get("segmentation_mask_b64") else ""
        print("\nFull response:")
        print(json.dumps(printable, indent=2))

    if args.save_mask and response.get("status") == "ok":
        save_mask(response, image_path)


if __name__ == "__main__":
    main()
