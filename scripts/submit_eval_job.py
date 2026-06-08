"""Submit a root-detection evaluation job to Azure ML, then conditionally register.

Run after a training job has completed:

    # Evaluate latest registered model (standalone):
    python scripts/submit_eval_job.py

    # Evaluate a specific training job's output and register if F1 >= threshold:
    python scripts/submit_eval_job.py --train_job <job_name>

The script will:
  1. Submit an evaluation Command job on cv5_test_images.
  2. Stream logs so you can follow progress.
  3. Download eval_results.txt and read mean F1.
  4. If --train_job was given: register the model only if mean F1 >= F1_THRESHOLD.

F1 / precision / recall are also visible in Azure ML studio under the job's Metrics tab.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment, JobResourceConfiguration, Model
from azure.identity import AzureCliCredential

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
ENV_YAML = REPO_ROOT / "environments" / "azure-train-env.yml"
COMPUTE_NAME = "lambda-4"
INSTANCE_TYPE = "gpu"
ENV_NAME = "root-detection-train"
MODEL_NAME = "root-detection"
F1_THRESHOLD = 0.6  # model is registered only if mean F1 meets this bar


def parse_args() -> argparse.Namespace:
    """Parse optional --train_job argument."""
    parser = argparse.ArgumentParser("submit_eval_job")
    parser.add_argument(
        "--train_job",
        default=None,
        help=(
            "Name of a completed training job whose output should be evaluated. "
            "If omitted, the latest registered model version is used instead."
        ),
    )
    return parser.parse_args()


def get_client() -> MLClient:
    """Create an authenticated MLClient from config.json."""
    _az_wbin = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
    if _az_wbin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _az_wbin + os.pathsep + os.environ.get("PATH", "")
    config = json.loads(CONFIG_PATH.read_text())
    return MLClient(
        credential=AzureCliCredential(process_timeout=30),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )


def ensure_environment(client: MLClient) -> str:
    """Register the training environment if it has changed; return 'name:version'."""
    env = Environment(
        name=ENV_NAME,
        description="TensorFlow + GPU environment for root-detection retraining",
        conda_file=str(ENV_YAML),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    registered = client.environments.create_or_update(env)
    label = f"{registered.name}:{registered.version}"
    print(f"Environment: {label}")
    return label


def get_latest_model_version(client: MLClient) -> str:
    """Return the latest registered version number of root-detection."""
    versions = list(client.models.list(name=MODEL_NAME))
    if not versions:
        raise RuntimeError(
            f"No registered model named '{MODEL_NAME}' found.\n"
            "Run submit_training_job.py first, then pass --train_job <name>."
        )
    latest = max(versions, key=lambda m: int(m.version))
    print(f"Using registered model: {MODEL_NAME} v{latest.version}")
    return latest.version


def submit_job(client: MLClient, env_label: str, train_job_name: str | None, model_version: str | None) -> str:
    """Build and submit the evaluation Command job; return the job name."""
    if train_job_name:
        model_input = Input(
            type=AssetTypes.URI_FOLDER,
            path=f"azureml://jobs/{train_job_name}/outputs/model_output",
            mode=InputOutputModes.RO_MOUNT,
        )
    else:
        model_input = Input(
            type=AssetTypes.CUSTOM_MODEL,
            path=f"azureml:{MODEL_NAME}:{model_version}",
            mode=InputOutputModes.RO_MOUNT,
        )

    job = command(
        display_name="root-detection-eval",
        description="Evaluate root-detection model on cv5_test_images",
        code=str(REPO_ROOT),
        command=(
            "python scripts/evaluate_root_detection.py "
            "--data_dir  ${{inputs.test_data}} "
            "--model_dir ${{inputs.model}} "
            "--output_dir ${{outputs.eval_output}}"
        ),
        environment=env_label,
        compute=COMPUTE_NAME,
        resources=JobResourceConfiguration(instance_type=INSTANCE_TYPE),
        inputs={
            "test_data": Input(
                type=AssetTypes.URI_FOLDER,
                path="azureml:cv5_test:1.0.0",
                mode=InputOutputModes.RO_MOUNT,
            ),
            "model": model_input,
        },
        outputs={
            "eval_output": Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
    )

    returned_job = client.jobs.create_or_update(job)
    print(f"\nJob submitted : {returned_job.name}")
    print(f"Monitor at    : {returned_job.studio_url}\n")
    return returned_job.name


def get_eval_f1(client: MLClient, eval_job_name: str) -> float:
    """Download eval_results.txt from the eval job output and return mean_f1."""
    download_dir = tempfile.mkdtemp()
    client.jobs.download(eval_job_name, download_path=download_dir, output_name="eval_output")

    results_file = Path(download_dir) / "named-outputs" / "eval_output" / "eval_results.txt"
    if not results_file.exists():
        raise FileNotFoundError(f"eval_results.txt not found in {download_dir}")

    for line in results_file.read_text().splitlines():
        if line.startswith("mean_f1="):
            return float(line.split("=")[1])

    raise ValueError("mean_f1 not found in eval_results.txt")


def get_best_registered_f1(client: MLClient) -> float:
    """Return the highest mean_f1 among all registered root-detection versions.

    Parses the score from the model description field (format: '... mean_f1=X.XXXX').
    Returns 0.0 if no models are registered or none have an F1 in their description.
    """
    versions = list(client.models.list(name=MODEL_NAME))
    if not versions:
        return 0.0

    best = 0.0
    for m in versions:
        desc = m.description or ""
        for part in desc.split():
            if part.startswith("mean_f1="):
                try:
                    best = max(best, float(part.split("=")[1]))
                except ValueError:
                    pass
    return best


def register_model(client: MLClient, train_job_name: str, f1: float) -> None:
    """Register the training job output as a new version of root-detection."""
    model = Model(
        path=f"azureml://jobs/{train_job_name}/outputs/model_output",
        name=MODEL_NAME,
        description=f"Retrained root-detection — job {train_job_name} — mean_f1={f1:.4f}",
        type=AssetTypes.CUSTOM_MODEL,
    )
    registered = client.models.create_or_update(model)
    print(f"\nRegistered: {MODEL_NAME} v{registered.version}  (mean_f1={f1:.4f})")
    print(f"Azure ID  : {registered.id}")


def main() -> None:
    """Evaluate the model and conditionally register if --train_job is provided."""
    args = parse_args()
    client = get_client()
    env_label = ensure_environment(client)

    model_version = None
    if not args.train_job:
        model_version = get_latest_model_version(client)

    eval_job_name = submit_job(client, env_label, args.train_job, model_version)

    print("Streaming job logs (Ctrl+C to detach — job keeps running in Azure):")
    client.jobs.stream(eval_job_name)

    mean_f1 = get_eval_f1(client, eval_job_name)
    print(f"\nEvaluation result — mean F1: {mean_f1:.4f}")

    if args.train_job:
        best_f1 = get_best_registered_f1(client)
        print(f"Best registered F1 : {best_f1:.4f}")
        print(f"Minimum threshold  : {F1_THRESHOLD}")

        if mean_f1 < F1_THRESHOLD:
            print(f"\nModel NOT registered — mean F1 {mean_f1:.4f} is below threshold {F1_THRESHOLD}.")
        elif mean_f1 <= best_f1:
            print(f"\nModel NOT registered — mean F1 {mean_f1:.4f} does not beat best registered F1 {best_f1:.4f}.")
        else:
            register_model(client, args.train_job, mean_f1)
            print("\nDone. View all model versions at: https://ml.azure.com")
    else:
        print("\nDone. View metrics at:")
        print(f"  https://ml.azure.com → Jobs → {eval_job_name} → Metrics")


if __name__ == "__main__":
    main()
