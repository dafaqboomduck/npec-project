"""Azure ML job script — combine root + anchor model files into one artifact.

Runs INSIDE Azure ML (not from your local machine). The job uploads only
this directory (which is why it lives in scripts/_combine/ in isolation —
keeping the upload to ~1 KB, no large .keras files).

Inputs (mounted by Azure ML at runtime):
    --root_model    folder containing root_det_model_512px.keras (any depth)
    --anchor_model  folder containing the anchor .keras file (any depth)
    --output        folder to write both files with canonical names

Output files:
    root_det_model_512px.keras
    anchor_det_model_512px.keras
"""

import argparse
import shutil
from pathlib import Path


def find_keras(path: Path, label: str) -> Path:
    """Return the .keras file at `path`.

    Azure ML mounts a custom_model input either as a directory (when the
    model was registered with a folder) or as a single file (when registered
    with one .keras file directly). Handle both.
    """
    if path.is_file():
        return path
    if path.is_dir():
        matches = list(path.rglob("*.keras"))
        if matches:
            return matches[0]
        # Fallback: list everything in the dir so the log shows what's there.
        contents = list(path.rglob("*"))
        raise FileNotFoundError(f"No .keras file found in {label} dir: {path}\n" f"Directory contents: {contents}")
    raise FileNotFoundError(f"{label} path does not exist: {path}")


parser = argparse.ArgumentParser()
parser.add_argument("--root_model", required=True)
parser.add_argument("--anchor_model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)

root_keras = find_keras(Path(args.root_model), "root")
anchor_keras = find_keras(Path(args.anchor_model), "anchor")

shutil.copy(root_keras, out_dir / "root_det_model_512px.keras")
shutil.copy(anchor_keras, out_dir / "anchor_det_model_512px.keras")

print(f"Root model   : {root_keras} -> root_det_model_512px.keras")
print(f"Anchor model : {anchor_keras} -> anchor_det_model_512px.keras")
print(f"Output dir   : {list(out_dir.iterdir())}")
