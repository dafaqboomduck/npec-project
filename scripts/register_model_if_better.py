"""Register the trained model if it beats the current best registered version.

This script runs as a pipeline step inside Azure ML compute. It uses the
managed identity of the cluster to authenticate — no config.json needed.

Conditions for registration (both must pass):
  1. mean_f1 >= --f1_threshold  (minimum quality bar)
  2. mean_f1 >  best F1 of all existing registered versions

Usage (called by the pipeline, not manually):
    python scripts/register_model_if_better.py
        --model_dir  <mounted model output directory>
        --eval_dir   <mounted eval output directory>
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
    parser.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--f1_threshold", type=float, default=F1_THRESHOLD_DEFAULT)
    return parser.parse_args()


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
        return

    client = get_client()
    best_f1 = get_best_registered_f1(client, args.model_name)
    print(f"Best registered F1 : {best_f1:.4f}")

    if mean_f1 <= best_f1:
        print(f"\nNOT registered — F1 {mean_f1:.4f} does not beat best registered F1 {best_f1:.4f}.")
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


if __name__ == "__main__":
    main()
