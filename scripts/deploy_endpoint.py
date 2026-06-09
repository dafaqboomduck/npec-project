"""Create or update the root-detection Kubernetes online endpoint on lambda-4.

This script is idempotent — run it whenever a new root-detection model
version is registered to roll the endpoint forward to that version:

    python scripts/deploy_endpoint.py

What it does
------------
1. Create a Kubernetes online endpoint backed by the `lambda-4` compute
   (the lab's local server) if it doesn't exist. If a Managed (Azure VM)
   endpoint with the same name already exists from a previous deployment,
   it's deleted and recreated as a Kubernetes endpoint — the migration
   happens automatically the first time you run this version.
2. Pick up the latest registered version of the 'root-detection' model.
3. Submit a small Azure ML job that combines root + anchor models into one
   artifact (runs entirely inside Azure — only ~1 KB of code uploaded from
   your machine). Skipped if the combined artifact already exists for this
   root model version.
4. Register the combined artifact as 'root-detection-serving:{root_version}'.
5. Create or update the 'current' deployment on the K8s endpoint, referencing
   the combined serving model. AZUREML_MODEL_DIR will contain BOTH .keras files.
6. Route 100% of traffic to the new deployment.

Why Kubernetes / lambda-4 instead of a managed Azure VM
-------------------------------------------------------
A `ManagedOnlineEndpoint` runs on an Azure-provided VM (e.g. Standard_DS3_v2)
billed 24/7 (~€150/month). `KubernetesOnlineEndpoint` runs the same scoring
container on whatever Kubernetes compute is attached to the workspace —
in our case the lab's `lambda-4` server, at zero incremental Azure cost.
The contract for `score.py` is identical between the two endpoint kinds.

Why bake root + anchor into one combined artifact
-------------------------------------------------
We can't grant the endpoint's managed identity permission to read models
from the workspace (students lack User Access Administrator on the shared
workspace). Instead we bake both .keras files into a single registered
model artifact, so the deployment just mounts them — no runtime download
and no auth needed.

Neither .keras file is ever uploaded from your machine: the combine job
mounts both registered models from inside Azure, copies them into one
output folder (fast network), and registers that folder as a new model.
"""

import argparse
import json
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    KubernetesOnlineDeployment,
    KubernetesOnlineEndpoint,
    Model,
    OnlineRequestSettings,
    ProbeSettings,
)
from azure.identity import AzureCliCredential, DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"
ENV_YAML = REPO_ROOT / "environments" / "azure-endpoint-env.yml"
COMBINE_CODE_DIR = REPO_ROOT / "scripts" / "_combine"  # tiny isolated folder

ENDPOINT_NAME = "root-detection-endpoint"
DEPLOYMENT_NAME = "current"
ROOT_MODEL_NAME = "root-detection"
ANCHOR_MODEL_NAME = "shoot-detection"
ANCHOR_MODEL_VERSION = "2"  # fixed — never changes
SERVING_MODEL_NAME = "root-detection-serving"
ENV_NAME = "root-detection-endpoint"

# Kubernetes compute attached to the workspace. Same `lambda-4` cluster
# used by the training pipeline and the combine job.
COMPUTE_NAME = "lambda-4"

# Kubernetes "instance type" — a named CPU/memory/GPU profile defined on the
# K8s cluster's azureml extension (NOT an Azure VM SKU). The cluster admin
# defines these. "defaultinstancetype" exists on every Azure ML K8s extension
# install. If lambda-4 has a custom instance type with more RAM/CPU set up
# (e.g. "cpu-4", "gpu"), change this constant.
#   List available types:
#     az ml compute show -n lambda-4 --resource-group buas-y2 \
#       --workspace-name team-CV5-2026 --query "properties" -o json
INSTANCE_TYPE = "defaultinstancetype"
INSTANCE_COUNT = 1


def get_client() -> MLClient:
    """Build an MLClient that works both locally and inside Azure ML compute.

    Inside an Azure ML pipeline step, the compute injects
    AZUREML_ARM_SUBSCRIPTION / AZUREML_ARM_RESOURCEGROUP / AZUREML_ARM_WORKSPACE_NAME
    environment variables and exposes a managed identity. We use those plus
    DefaultAzureCredential so the same script works when invoked as a
    pipeline step OR as a local CLI call (where it falls back to az login).
    """
    if all(
        k in os.environ
        for k in (
            "AZUREML_ARM_SUBSCRIPTION",
            "AZUREML_ARM_RESOURCEGROUP",
            "AZUREML_ARM_WORKSPACE_NAME",
        )
    ):
        print("Authenticating via managed identity (pipeline step)")
        return MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
            resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
            workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
        )
    print("Authenticating via Azure CLI credential (local run)")
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


def get_or_create_endpoint(client: MLClient) -> None:
    """Ensure a Kubernetes online endpoint backed by lambda-4 exists.

    If an endpoint with this name already exists but is a Managed (Azure VM)
    endpoint left over from an earlier deployment, it's deleted and recreated
    as a Kubernetes endpoint. Azure ML doesn't support changing kind in place.
    """
    try:
        existing = client.online_endpoints.get(ENDPOINT_NAME)
        # `compute` is set on Kubernetes endpoints, None on Managed ones.
        # `kind` is a more reliable signal but it's only on some SDK versions.
        existing_kind = getattr(existing, "kind", None) or (
            "Kubernetes" if getattr(existing, "compute", None) else "Managed"
        )
        if existing_kind == "Kubernetes" and getattr(existing, "compute", None) == COMPUTE_NAME:
            print(f"Endpoint already exists : {ENDPOINT_NAME} (Kubernetes on {COMPUTE_NAME})")
            return
        print(
            f"Endpoint {ENDPOINT_NAME} exists but is {existing_kind} "
            f"(compute={getattr(existing, 'compute', None)}); deleting and recreating "
            f"as Kubernetes on {COMPUTE_NAME}."
        )
        client.online_endpoints.begin_delete(ENDPOINT_NAME).result()
    except Exception as exc:
        # Endpoint doesn't exist OR the get failed for another reason. If it
        # truly doesn't exist, create_or_update below will succeed. If it's a
        # transient error, the subsequent create will surface it clearly.
        print(f"No existing endpoint to migrate ({type(exc).__name__}). Will create fresh.")

    print(f"Creating endpoint       : {ENDPOINT_NAME} (Kubernetes on {COMPUTE_NAME})")
    endpoint = KubernetesOnlineEndpoint(
        name=ENDPOINT_NAME,
        compute=COMPUTE_NAME,
        description=(
            "Root detection inference endpoint running on the lambda-4 Kubernetes "
            "compute. Re-run deploy_endpoint.py to roll forward to the latest model."
        ),
        auth_mode="key",
    )
    client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint created        : {ENDPOINT_NAME}")


def get_latest_model_version(client: MLClient, model_name: str) -> str:
    """Return the latest registered version of model_name (highest int version)."""
    versions = list(client.models.list(name=model_name))
    if not versions:
        raise ValueError(
            f"No registered versions found for model '{model_name}'. " f"Register it in Azure ML before deploying."
        )
    latest = max(versions, key=lambda m: int(m.version))
    print(f"Latest {model_name:<22}: v{latest.version}")
    return latest.version


def ensure_environment(client: MLClient) -> str:
    """Register the endpoint conda environment; return 'name:version'."""
    env = Environment(
        name=ENV_NAME,
        description="CPU inference environment for the root-detection endpoint",
        conda_file=str(ENV_YAML),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    registered = client.environments.create_or_update(env)
    label = f"{registered.name}:{registered.version}"
    print(f"Environment             : {label}")
    return label


def prepare_serving_model(client: MLClient, root_version: str, env_label: str) -> str:
    """Return version of 'root-detection-serving' bundling root + anchor models.

    Reuses the existing serving model for this root version if already
    registered. Otherwise submits a small combine job inside Azure ML.
    """
    try:
        existing = client.models.get(name=SERVING_MODEL_NAME, version=root_version)
        print(f"Serving model exists    : {SERVING_MODEL_NAME} v{existing.version} (skip combine)")
        return existing.version
    except Exception:
        pass

    print(f"\nCombining root v{root_version} + anchor v{ANCHOR_MODEL_VERSION} (Azure ML job, ~5 min) ...")
    print(f"  Code dir uploaded: {COMBINE_CODE_DIR}  (only _combine_models.py — no .keras files)")

    combine_job = command(
        display_name=f"combine-models-root-v{root_version}",
        description="Combine root + anchor .keras files into one serving artifact",
        inputs={
            # CUSTOM_MODEL — the SDK shorthand 'azureml:name:version' resolves
            # against the model registry (not the data-asset registry).
            "root_model": Input(type=AssetTypes.CUSTOM_MODEL, path=f"azureml:{ROOT_MODEL_NAME}:{root_version}"),
            "anchor_model": Input(
                type=AssetTypes.CUSTOM_MODEL, path=f"azureml:{ANCHOR_MODEL_NAME}:{ANCHOR_MODEL_VERSION}"
            ),
        },
        outputs={"combined": Output(type=AssetTypes.URI_FOLDER)},
        code=str(COMBINE_CODE_DIR),  # ← isolated dir, only _combine_models.py
        command=(
            "python _combine_models.py "
            "--root_model ${{inputs.root_model}} "
            "--anchor_model ${{inputs.anchor_model}} "
            "--output ${{outputs.combined}}"
        ),
        environment=env_label,
        compute=COMPUTE_NAME,
    )

    submitted = client.jobs.create_or_update(combine_job)
    print(f"  Job name : {submitted.name}")
    print(f"  Monitor  : {submitted.studio_url}")
    client.jobs.stream(submitted.name)  # blocks until done

    completed = client.jobs.get(submitted.name)
    if completed.status != "Completed":
        raise RuntimeError(
            f"Combine job {submitted.name} ended with status '{completed.status}'. "
            f"Check logs: {submitted.studio_url}"
        )

    # Reference the job output as a registered model. The
    # 'azureml://jobs/{run_id}/outputs/{output_name}' URI is the canonical
    # way to register a job output as a model in Azure ML v2.
    output_uri = f"azureml://jobs/{submitted.name}/outputs/combined"
    serving_model = Model(
        name=SERVING_MODEL_NAME,
        version=root_version,
        path=output_uri,
        type=AssetTypes.CUSTOM_MODEL,
        description=(
            f"Combined root+anchor for endpoint serving. "
            f"root-detection v{root_version} + shoot-detection v{ANCHOR_MODEL_VERSION}"
        ),
    )
    registered = client.models.create_or_update(serving_model)
    print(f"Serving model registered: {registered.name} v{registered.version}")
    return registered.version


def deploy(client: MLClient, serving_version: str, env_label: str) -> None:
    """Create or update the K8s deployment and route all traffic to it."""
    deployment = KubernetesOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        # Deploy the combined artifact — AZUREML_MODEL_DIR contains both
        # root_det_model_512px.keras and anchor_det_model_512px.keras.
        model=f"azureml:{SERVING_MODEL_NAME}:{serving_version}",
        environment=env_label,
        code_configuration=CodeConfiguration(
            code=str(REPO_ROOT),
            scoring_script="scripts/score.py",
        ),
        environment_variables={
            "MODEL_VERSION": f"{ROOT_MODEL_NAME}-v{serving_version}",
            # Dish cropping is now ON. The previous reason for disabling it
            # (extract_petri_dish mis-cropping) has been addressed in
            # src/npec_cv/helpers/extract_petri_dish.py — the new
            # implementation uses border-vs-dish luminance with a sanity
            # check so it falls back to the full image when detection fails
            # rather than cropping wildly. Cropping is critical because the
            # U-Net was trained on dish-cropped patches; feeding the full
            # image with its black border destroys edge plant predictions.
            #
            # NOTE: explicitly set to "0" (not omitted) because Azure ML's
            # begin_create_or_update preserves env vars from the previous
            # deployment when a key is missing from the new spec — omitting
            # the key leaves the old "1" value in place. Passing "0" forces
            # the override.
            "DISABLE_DISH_CROP": "0",
            # Horizontal flip-merge in score.py runs the root + anchor models
            # TWICE per request (once on the original image, once on the
            # left-right mirror) and merges with elementwise max, to symmetrize
            # a model spatial bias we noticed earlier. With the dish-cropping
            # fix that bias is no longer the main failure mode, and the
            # flip-merge doubles inference latency (~90s -> ~180s) for marginal
            # quality improvement. Disabling by default; set DISABLE_FLIP_MERGE=0
            # explicitly to bring it back if the bias returns on a future model.
            "DISABLE_FLIP_MERGE": "1",
        },
        # K8s instance type — a named CPU/memory profile defined on the
        # cluster, not an Azure VM SKU. See INSTANCE_TYPE comment at top.
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        # Default scoring timeout is 5000 ms; image inference on CPU is
        # ~10-30 s per request, so bump it. max_concurrent_requests_per_instance
        # stays at 1 (synchronous TF inference doesn't benefit from concurrency).
        request_settings=OnlineRequestSettings(
            request_timeout_ms=180000,
            max_concurrent_requests_per_instance=1,
            max_queue_wait_ms=60000,
        ),
        # Generous probes — first request loads two .keras models (~60s).
        liveness_probe=ProbeSettings(failure_threshold=30, timeout=2, period=10, initial_delay=10),
        readiness_probe=ProbeSettings(failure_threshold=30, timeout=10, period=10, initial_delay=10),
    )

    print(f"\nDeploying {SERVING_MODEL_NAME} v{serving_version}")
    print(f"  → {ENDPOINT_NAME}/{DEPLOYMENT_NAME} on {COMPUTE_NAME} (instance_type={INSTANCE_TYPE})")
    print("(First run takes 5–10 min while the container image builds and is pulled to the cluster.)")
    client.online_deployments.begin_create_or_update(deployment).result()

    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()

    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    print("\nDeployment complete!")
    print(f"  Serving model : {SERVING_MODEL_NAME} v{serving_version}")
    print(f"  Compute       : {COMPUTE_NAME} (Kubernetes)")
    print(f"  Scoring URI   : {endpoint.scoring_uri}")
    print("\nTest:    python scripts/test_endpoint.py --image <path/to/image.png>")
    print("Update:  python scripts/deploy_endpoint.py   (picks up latest model automatically)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("deploy_endpoint")
    parser.add_argument(
        "--root-version",
        default=None,
        help="Pin a specific root-detection version (e.g. '1'). " "Default: latest registered version.",
    )
    parser.add_argument(
        "--registered-version-file",
        default=None,
        help="Pipeline-mode input. Path to a folder containing "
        "registered_version.txt produced by register_model_if_better. "
        "If the file says 'none' (no new model), the script exits "
        "without deploying. Otherwise the version inside is used as "
        "the root model version to deploy.",
    )
    return parser.parse_args()


def resolve_root_version_from_pipeline(version_dir: str) -> "str | None":
    """Read registered_version.txt from the upstream register step.

    Pipeline-mode contract:
        file contains '<int>'     -> roll endpoint forward to that version
        file contains 'none'/''   -> register skipped, return None
        file missing              -> treat as 'none'

    Returns:
        The version string if a redeploy is needed, else None.
    """
    marker = Path(version_dir) / "registered_version.txt"
    if not marker.exists():
        print(f"No registered_version.txt found at {marker} — nothing to deploy.")
        return None
    raw = marker.read_text().strip()
    if not raw or raw.lower() == "none":
        print("register_model_if_better did not register a new model — skipping deploy.")
        return None
    print(f"Pipeline mode: register step wrote version '{raw}'")
    return raw


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Pipeline-mode short-circuit: if no new model was registered this run,
    # exit cleanly without touching the endpoint. This makes the deploy step
    # safe to put in every weekly pipeline run.
    if args.registered_version_file:
        pipeline_version = resolve_root_version_from_pipeline(args.registered_version_file)
        if pipeline_version is None:
            return
        # Override any --root-version: the upstream register step is the
        # source of truth in pipeline mode.
        args.root_version = pipeline_version

    client = get_client()
    get_or_create_endpoint(client)
    if args.root_version:
        # Verify the requested version exists; raises if not registered.
        client.models.get(name=ROOT_MODEL_NAME, version=args.root_version)
        root_version = args.root_version
        print(f"Using pinned     {ROOT_MODEL_NAME:<22}: v{root_version}")
    else:
        root_version = get_latest_model_version(client, ROOT_MODEL_NAME)
    get_latest_model_version(client, ANCHOR_MODEL_NAME)  # fail fast if missing
    env_label = ensure_environment(client)
    serving_version = prepare_serving_model(client, root_version, env_label)
    deploy(client, serving_version, env_label)


if __name__ == "__main__":
    main()
