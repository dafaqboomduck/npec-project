"""Submit a root-detection retraining job to Azure ML.

Run this script from the repo root to kick off a training job:
    python scripts/submit_training_job.py

The script will:
  1. Register (or reuse) the training environment in Azure ML.
  2. Submit a Command job on the GPU compute cluster.
  3. Stream logs so you can follow progress.

After training completes, run submit_eval_job.py to evaluate and
conditionally register the model if it meets the F1 threshold.
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
INSTANCE_TYPE = "gpu"  # requests 1 GPU on lambda-0
ENV_NAME = "root-detection-train"
MAX_IMAGES = None  # set to None to train on the full dataset


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


def submit_job(client: MLClient, env_label: str) -> str:
    """Build and submit the training Command job; return the job name."""
    job = command(
        display_name="root-detection-retrain",
        description="Retrain root-detection model on cv5_hades_raw",
        # Upload the entire repo as job code so npec_cv is importable
        code=str(REPO_ROOT),
        command=(
            "python scripts/train_root_detection.py "
            "--data_dir ${{inputs.train_data}} "
            "--val_dir ${{inputs.val_data}} "
            "--output_dir ${{outputs.model_output}} " + (f"--max_images {MAX_IMAGES}" if MAX_IMAGES is not None else "")
        ),
        environment=env_label,
        compute=COMPUTE_NAME,
        resources=JobResourceConfiguration(instance_type=INSTANCE_TYPE),
        inputs={
            "train_data": Input(
                type=AssetTypes.URI_FOLDER,
                path="azureml:cv5_train:1.0.0",
                mode=InputOutputModes.RO_MOUNT,
            ),
            "val_data": Input(
                type=AssetTypes.URI_FOLDER,
                path="azureml:cv5_val:1.0.0",
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
        outputs={
            # Azure ML captures this directory after the job finishes
            "model_output": Output(
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
    """Submit training job and stream logs."""
    client = get_client()
    env_label = ensure_environment(client)
    job_name = submit_job(client, env_label)

    print("Streaming job logs (Ctrl+C to detach — job keeps running in Azure):")
    client.jobs.stream(job_name)

    print(f"\nTraining complete. Job name: {job_name}")
    print("Run submit_eval_job.py to evaluate and conditionally register the model.")
    print(f"  python scripts/submit_eval_job.py --train_job {job_name}")


if __name__ == "__main__":
    main()
