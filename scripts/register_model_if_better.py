"""Register the trained model if it beats the current best registered version.

This script runs as a pipeline step inside Azure ML compute. It uses the
managed identity of the cluster to authenticate — no config.json needed.

Conditions for registration (both must pass):
  1. mean_f1 >= --f1_threshold  (minimum quality bar)
  2. mean_f1 >  best F1 of all existing registered versions

The script writes a tiny `registered_version.txt` file into --output_dir,
which the downstream deploy step reads to decide whether to roll the
endpoint forward. Contents:
    <integer version>   — a new version was registered, deploy it
    none                — registration was skipped (model didn't pass), nothing to do

Usage (called by the pipeline, not manually):
    python scripts/register_model_if_better.py
        --model_dir  <mounted model output directory>
        --eval_dir   <mounted eval output directory>
        --output_dir <mounted output for registered_version.txt>
        [--model_name root-detection]
        [--f1_threshold 0.6]
"""

import argparse
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

MODEL_NAME_DEFAULT = "root-detection"
F1_THRESHOLD_DEFAULT = 0.6


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("register_model_if_better")
    parser.add_argument("--model_dir", required=True, help="Mounted path to trained model output")
    parser.add_argument("--eval_dir", required=True, help="Mounted path to eval output (contains eval_results.txt)")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Pipeline output dir to write registered_version.txt into. "
        "If omitted (manual runs), the version file isn't written.",
    )
    parser.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--f1_threshold", type=float, default=F1_THRESHOLD_DEFAULT)
    return parser.parse_args()


def write_version_marker(output_dir: str | None, version: str) -> None:
    """Drop registered_version.txt into output_dir for the downstream deploy step.

    Args:
        output_dir: Pipeline-mounted output directory, or None (no-op).
        version: The registered version string, or the literal "none" to
            signal that no new model was registered this run.
    """
    if output_dir is None:
        print(f"(no --output_dir provided — skipping registered_version.txt; would have been '{version}')")
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "registered_version.txt"
    marker.write_text(version + "\n")
    print(f"Wrote registered_version.txt: '{version}'  ({marker})")


def get_client() -> MLClient:
    """Create MLClient using the cluster's managed identity + Azure ML env vars."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )


def read_mean_f1(eval_dir: str) -> float:
    """Read mean_f1 from eval_results.txt in the eval output directory."""
    results_file = Path(eval_dir) / "eval_results.txt"
    if not results_file.exists():
        raise FileNotFoundError(f"eval_results.txt not found in {eval_dir}")

    for line in results_file.read_text().splitlines():
        if line.startswith("mean_f1="):
            return float(line.split("=")[1])

    raise ValueError("mean_f1 not found in eval_results.txt")


def get_best_registered_f1(client: MLClient, model_name: str) -> float:
    """Return the highest mean_f1 across all registered versions of model_name.

    Parses score from description field (format: '... mean_f1=X.XXXX').
    Returns 0.0 if no models registered or none have an F1 in their description.
    """
    versions = list(client.models.list(name=model_name))
    if not versions:
        return 0.0

    best = 0.0
    for m in versions:
        for part in (m.description or "").split():
            if part.startswith("mean_f1="):
                try:
                    best = max(best, float(part.split("=")[1]))
                except ValueError:
                    pass
    return best


def main() -> None:
    """Evaluate F1 conditions and register model if both pass."""
    args = parse_args()

    mean_f1 = read_mean_f1(args.eval_dir)
    print(f"New model mean F1  : {mean_f1:.4f}")
    print(f"Minimum threshold  : {args.f1_threshold}")

    if mean_f1 < args.f1_threshold:
        print(f"\nNOT registered — F1 {mean_f1:.4f} is below threshold {args.f1_threshold}.")
        write_version_marker(args.output_dir, "none")
        return

    client = get_client()
    best_f1 = get_best_registered_f1(client, args.model_name)
    print(f"Best registered F1 : {best_f1:.4f}")

    if mean_f1 <= best_f1:
        print(f"\nNOT registered — F1 {mean_f1:.4f} does not beat best registered F1 {best_f1:.4f}.")
        write_version_marker(args.output_dir, "none")
        return

    model = Model(
        path=args.model_dir,
        name=args.model_name,
        description=f"Pipeline-trained root-detection — mean_f1={mean_f1:.4f}",
        type=AssetTypes.CUSTOM_MODEL,
    )
    registered = client.models.create_or_update(model)
    print(f"\nRegistered: {args.model_name} v{registered.version}  (mean_f1={mean_f1:.4f})")
    print(f"Azure ID  : {registered.id}")
    write_version_marker(args.output_dir, str(registered.version))


if __name__ == "__main__":
    main()
