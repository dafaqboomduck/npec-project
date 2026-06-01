"""Build, submit, and schedule the root-detection retraining pipeline.

Run once to register the schedule (runs every Monday at 08:00 UTC):
    python scripts/submit_pipeline.py --schedule

Run immediately without scheduling (for testing):
    python scripts/submit_pipeline.py --run-now

Both:
    python scripts/submit_pipeline.py --schedule --run-now

Pipeline steps:
  1. Train    — trains U-Net on cv5_train + cv5_val, outputs model
  2. Evaluate — runs patch inference on cv5_test, writes eval_results.txt
  3. Register — registers model if F1 >= 0.6 AND beats best registered version;
                writes registered_version.txt (version string or 'none')
  4. Deploy   — reads registered_version.txt; if a new version was registered,
                rolls the online endpoint forward to it. Otherwise a no-op.
"""

import argparse
import json
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (
    CommandComponent,
    CronTrigger,
    Environment,
    JobResourceConfiguration,
    JobSchedule,
)
from azure.ai.ml.sweep import Choice
from azure.identity import AzureCliCredential

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
ENV_YAML = REPO_ROOT / "environments" / "azure-train-env.yml"
COMPUTE_NAME = "lambda-0"
INSTANCE_TYPE = "gpu"
ENV_NAME = "root-detection-train"
SCHEDULE_NAME = "root-detection-weekly"
SCHEDULE_CRON = "0 8 * * 1"  # Every Monday at 08:00 UTC
MAX_IMAGES = None  # set to None to train on the full dataset
SWEEP_BATCH_SIZES = [8, 16, 32]  # batch sizes to try; 3 trials, one per size


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("submit_pipeline")
    parser.add_argument("--schedule", action="store_true", help="Register the Monday schedule")
    parser.add_argument("--run-now", action="store_true", help="Submit a pipeline run immediately")
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


def build_components(
    env_label: str,
) -> tuple[CommandComponent, CommandComponent, CommandComponent, CommandComponent]:
    """Create the four pipeline components (train, eval, register, deploy)."""
    resources = JobResourceConfiguration(instance_type=INSTANCE_TYPE)
    code = str(REPO_ROOT)

    train_component = CommandComponent(
        name="train_root_detection",
        display_name="Train Root Detection",
        description="Train U-Net from scratch on cv5_train + cv5_val",
        inputs={
            "train_data": Input(type="uri_folder"),
            "val_data": Input(type="uri_folder"),
            "patch_size": Input(type="integer", default=512),
            "epochs": Input(type="integer", default=30),
            "batch_size": Input(type="integer", default=16),
        },
        outputs={
            "model_output": Output(type="uri_folder"),
        },
        code=code,
        command=(
            "python scripts/train_root_detection.py "
            "--data_dir ${{inputs.train_data}} "
            "--val_dir  ${{inputs.val_data}} "
            "--output_dir ${{outputs.model_output}} "
            "--patch_size ${{inputs.patch_size}} "
            "--epochs ${{inputs.epochs}} "
            "--batch_size ${{inputs.batch_size}} " + (f"--max_images {MAX_IMAGES}" if MAX_IMAGES is not None else "")
        ),
        environment=env_label,
        resources=resources,
    )

    eval_component = CommandComponent(
        name="evaluate_root_detection",
        display_name="Evaluate Root Detection",
        description="Patch-based inference on cv5_test; logs F1/precision/recall",
        inputs={
            "test_data": Input(type="uri_folder"),
            "model_dir": Input(type="uri_folder"),
            "patch_size": Input(type="integer", default=512),
            "threshold": Input(type="number", default=0.5),
        },
        outputs={
            "eval_output": Output(type="uri_folder"),
        },
        code=code,
        command=(
            "python scripts/evaluate_root_detection.py "
            "--data_dir  ${{inputs.test_data}} "
            "--model_dir ${{inputs.model_dir}} "
            "--output_dir ${{outputs.eval_output}} "
            "--patch_size ${{inputs.patch_size}} "
            "--threshold  ${{inputs.threshold}}"
        ),
        environment=env_label,
        resources=resources,
    )

    register_component = CommandComponent(
        name="register_model_if_better",
        display_name="Register Model If Better",
        description="Registers new model version only if F1 >= threshold and beats best registered. "
        "Writes registered_version.txt to register_output (version or 'none').",
        inputs={
            "model_dir": Input(type="uri_folder"),
            "eval_dir": Input(type="uri_folder"),
            "f1_threshold": Input(type="number", default=0.6),
        },
        outputs={
            "register_output": Output(type="uri_folder"),
        },
        code=code,
        command=(
            "python scripts/register_model_if_better.py "
            "--model_dir   ${{inputs.model_dir}} "
            "--eval_dir    ${{inputs.eval_dir}} "
            "--output_dir  ${{outputs.register_output}} "
            "--f1_threshold ${{inputs.f1_threshold}} "
            "--train_dataset cv5_train:1.0.0 "  # NEW
            "--val_dataset   cv5_val:1.0.0 "  # NEW
            "--test_dataset  cv5_test:1.0.0"  # NEW
        ),
        environment=env_label,
        resources=resources,
    )

    deploy_component = CommandComponent(
        name="deploy_endpoint",
        display_name="Deploy Endpoint If New Model",
        description="Reads registered_version.txt from the register step and rolls the "
        "online endpoint forward to that version. No-op if the register step "
        "wrote 'none' (i.e. no new model was registered this run).",
        inputs={
            "register_output": Input(type="uri_folder"),
        },
        code=code,
        command=("python scripts/deploy_endpoint.py " "--registered-version-file ${{inputs.register_output}}"),
        environment=env_label,
        resources=resources,
    )

    return train_component, eval_component, register_component, deploy_component


def build_pipeline(
    train_comp: CommandComponent,
    eval_comp: CommandComponent,
    register_comp: CommandComponent,
    deploy_comp: CommandComponent,
) -> pipeline:
    """Wire the four components into an Azure ML pipeline."""

    @pipeline(
        display_name="root-detection-weekly-retrain",
        description="Weekly pipeline: sweep batch_size → evaluate best model "
        "→ conditionally register → conditionally redeploy endpoint",
    )
    def root_detection_pipeline(
        train_data: Input(type="uri_folder"),
        val_data: Input(type="uri_folder"),
        test_data: Input(type="uri_folder"),
    ):
        # -- Sweep: try each batch size, keep the best val_f1 --------------------
        train_step = train_comp(train_data=train_data, val_data=val_data)
        sweep_step = train_step.sweep(
            primary_metric="best_val_f1",
            goal="maximize",
            sampling_algorithm="grid",
        )
        sweep_step.set_limits(
            max_total_trials=len(SWEEP_BATCH_SIZES),
            max_concurrent_trials=1,
        )
        sweep_step.search_space = {"batch_size": Choice(SWEEP_BATCH_SIZES)}
        sweep_step.compute = COMPUTE_NAME
        sweep_step.resources = JobResourceConfiguration(instance_type=INSTANCE_TYPE)

        # -- Evaluate best model from sweep --------------------------------------
        eval_step = eval_comp(
            test_data=test_data,
            model_dir=sweep_step.outputs.model_output,
        )
        eval_step.compute = COMPUTE_NAME

        # -- Register if F1 >= threshold and beats best registered version -------
        register_step = register_comp(
            model_dir=sweep_step.outputs.model_output,
            eval_dir=eval_step.outputs.eval_output,
        )
        register_step.compute = COMPUTE_NAME

        # -- Deploy endpoint if a new model was registered -----------------------
        # The deploy step ALWAYS runs, but `deploy_endpoint.py` reads
        # registered_version.txt and exits cleanly when its contents are 'none'.
        # That way we don't need conditional execution in the pipeline DSL.
        deploy_step = deploy_comp(
            register_output=register_step.outputs.register_output,
        )
        deploy_step.compute = COMPUTE_NAME

    return root_detection_pipeline


def main() -> None:
    """Submit the pipeline and/or register the Monday schedule."""
    args = parse_args()

    if not args.schedule and not args.run_now:
        print("Nothing to do. Use --schedule, --run-now, or both.")
        return

    client = get_client()
    env_label = ensure_environment(client)

    train_comp, eval_comp, register_comp, deploy_comp = build_components(env_label)
    pipeline_func = build_pipeline(train_comp, eval_comp, register_comp, deploy_comp)

    # Bind the fixed data assets
    pipeline_job = pipeline_func(
        train_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv5_train:1.0.0", mode=InputOutputModes.RO_MOUNT),
        val_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv5_val:1.0.0", mode=InputOutputModes.RO_MOUNT),
        test_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv5_test:1.0.0", mode=InputOutputModes.RO_MOUNT),
    )

    if args.run_now:
        submitted = client.jobs.create_or_update(pipeline_job)
        print(f"\nPipeline submitted : {submitted.name}")
        print(f"Monitor at         : {submitted.studio_url}")

    if args.schedule:
        schedule = JobSchedule(
            name=SCHEDULE_NAME,
            trigger=CronTrigger(
                expression=SCHEDULE_CRON,
                time_zone="UTC",
            ),
            create_job=pipeline_job,
        )
        client.schedules.begin_create_or_update(schedule).result()
        print(f"\nSchedule registered: '{SCHEDULE_NAME}'")
        print(f"Runs every Monday at 08:00 UTC  (cron: {SCHEDULE_CRON})")
        print("View at: https://ml.azure.com → Pipelines → Schedules")


if __name__ == "__main__":
    main()
