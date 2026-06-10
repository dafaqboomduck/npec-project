"""Submit (and optionally schedule) the preprocessing pipeline.

Builds a one-step pipeline that takes the three existing data assets
(`cv5_train`, `cv5_val`, `cv5_test`), patchifies every image and mask, and
registers the result as a NEW VERSION of the `cv5_patches` data asset on
the workspace.

Each run produces a brand-new version (versions auto-increment as 1, 2,
3, …). Existing pipelines (`submit_pipeline.py`, `submit_training_only_pipeline.py`)
are NOT touched — they still patch from scratch each time.

Run once:
    python scripts/submit_preprocessing_pipeline.py --run-now

Register a weekly schedule (e.g. to refresh patches after dataset updates):
    python scripts/submit_preprocessing_pipeline.py --schedule

Both:
    python scripts/submit_preprocessing_pipeline.py --schedule --run-now
"""

import argparse
import json
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (
    CommandComponent,
    CronTrigger,
    Environment,
    JobResourceConfiguration,
    JobSchedule,
)
from azure.identity import AzureCliCredential

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
ENV_YAML = REPO_ROOT / "environments" / "azure-train-env.yml"
COMPUTE_NAME = "lambda-4"
INSTANCE_TYPE = "gpu"  # reuses the existing training env / instance type; CPU would also work
ENV_NAME = "root-detection-train"
SCHEDULE_NAME = "cv5-preprocess-weekly"
SCHEDULE_CRON = "0 7 * * 1"  # Mondays 07:00 UTC — one hour before the training pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("submit_preprocessing_pipeline")
    parser.add_argument("--schedule", action="store_true", help="Register the Monday schedule")
    parser.add_argument("--run-now", action="store_true", help="Submit a pipeline run immediately")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch dimension (default 512)")
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
        description="TensorFlow + GPU environment (reused for the preprocessing pipeline; "
        "patching is CPU-only but the same env keeps deps consistent).",
        conda_file=str(ENV_YAML),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    registered = client.environments.create_or_update(env)
    label = f"{registered.name}:{registered.version}"
    print(f"Environment: {label}")
    return label


def build_component(env_label: str) -> CommandComponent:
    """Create the single preprocess component."""
    return CommandComponent(
        name="preprocess_dataset",
        display_name="Preprocess (Patchify) Dataset",
        description="Patches cv5_train/cv5_val/cv5_test and registers as a new version of cv5_patches.",
        inputs={
            "train_data": Input(type="uri_folder"),
            "val_data": Input(type="uri_folder"),
            "test_data": Input(type="uri_folder"),
            "patch_size": Input(type="integer", default=512),
        },
        code=str(REPO_ROOT),
        command=(
            "python scripts/preprocess_dataset.py "
            "--train_dir   ${{inputs.train_data}} "
            "--val_dir     ${{inputs.val_data}} "
            "--test_dir    ${{inputs.test_data}} "
            "--patch_size  ${{inputs.patch_size}}"
        ),
        environment=env_label,
        resources=JobResourceConfiguration(instance_type=INSTANCE_TYPE),
    )


def build_pipeline_func(comp: CommandComponent) -> pipeline:
    """Wire the single component into an Azure ML pipeline."""

    @pipeline(
        display_name="cv5-preprocess-pipeline",
        description="Patchify cv5_train/val/test and register as a NEW VERSION of cv5_patches.",
    )
    def preprocess_pipeline(
        train_data: Input(type="uri_folder"),
        val_data: Input(type="uri_folder"),
        test_data: Input(type="uri_folder"),
        patch_size: int = 512,
    ):
        step = comp(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            patch_size=patch_size,
        )
        step.compute = COMPUTE_NAME
        step.resources = JobResourceConfiguration(instance_type=INSTANCE_TYPE)

    return preprocess_pipeline


def main() -> None:
    """Build, optionally submit, optionally schedule the preprocess pipeline."""
    args = parse_args()

    if not args.schedule and not args.run_now:
        print("Nothing to do. Use --schedule, --run-now, or both.")
        return

    client = get_client()
    env_label = ensure_environment(client)
    comp = build_component(env_label)
    pipeline_func = build_pipeline_func(comp)

    pipeline_job = pipeline_func(
        train_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv5_train:1.0.0", mode=InputOutputModes.RO_MOUNT),
        val_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv5_val:1.0.0", mode=InputOutputModes.RO_MOUNT),
        test_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv5_test:1.0.0", mode=InputOutputModes.RO_MOUNT),
        patch_size=args.patch_size,
    )

    if args.run_now:
        submitted = client.jobs.create_or_update(pipeline_job)
        print(f"\nPipeline submitted : {submitted.name}")
        print(f"Monitor at         : {submitted.studio_url}")

    if args.schedule:
        schedule = JobSchedule(
            name=SCHEDULE_NAME,
            trigger=CronTrigger(expression=SCHEDULE_CRON, time_zone="UTC"),
            create_job=pipeline_job,
        )
        client.schedules.begin_create_or_update(schedule).result()
        print(f"\nSchedule registered: '{SCHEDULE_NAME}'")
        print(f"Runs every Monday at 07:00 UTC  (cron: {SCHEDULE_CRON})")


if __name__ == "__main__":
    main()
