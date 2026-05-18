"""Submit a root-detection evaluation job to Azure ML.

Run this script from the repo root after a training job has completed:
    python scripts/submit_eval_job.py

The script will:
  1. Submit a Command job that runs patch-based inference on cv5_test_images.
  2. Load the latest registered 'root-detection' model from the model registry.
  3. Stream logs so you can follow progress.
  4. F1 / precision / recall scores are logged to the Azure ML experiment.

Results are visible in the Azure ML studio under the job's Metrics tab.
"""

import json
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment, JobResourceConfiguration
from azure.identity import AzureCliCredential

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
ENV_YAML = REPO_ROOT / "environments" / "azure-train-env.yml"
COMPUTE_NAME = "lambda-0"
INSTANCE_TYPE = "gpu"
ENV_NAME = "root-detection-train"  # reuse the same environment
MODEL_NAME = "root-detection"


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
    """Return the latest version number of the root-detection model."""
    versions = list(client.models.list(name=MODEL_NAME))
    if not versions:
        raise RuntimeError(
            f"No registered model named '{MODEL_NAME}' found.\n"
            "Run submit_training_job.py first to train and register the model."
        )
    latest = max(versions, key=lambda m: int(m.version))
    print(f"Using model: {MODEL_NAME} v{latest.version}")
    return latest.version


def submit_job(client: MLClient, env_label: str, model_version: str) -> str:
    """Build and submit the evaluation Command job; return the job name."""
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
                path="azureml:cv5_test_images:1.1.1",
                mode=InputOutputModes.RO_MOUNT,
            ),
            "model": Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=f"azureml:{MODEL_NAME}:{model_version}",
                mode=InputOutputModes.RO_MOUNT,
            ),
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


def main() -> None:
    """Run the full submit → stream pipeline."""
    client = get_client()

    env_label = ensure_environment(client)
    model_version = get_latest_model_version(client)
    job_name = submit_job(client, env_label, model_version)

    print("Streaming job logs (Ctrl+C to detach — job keeps running in Azure):")
    client.jobs.stream(job_name)

    print("\nDone. View metrics at:")
    print(f"  https://ml.azure.com → Jobs → {job_name} → Metrics")


if __name__ == "__main__":
    main()
