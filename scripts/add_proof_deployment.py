"""Add a second deployment slot on the existing endpoint at 0% traffic.

For demonstrating that an Azure ML online endpoint supports multiple
concurrent deployments (the foundation of blue-green / canary rollouts).
After this runs, the endpoint has TWO deployments configured:

    current  - 100% traffic (production, unchanged)
    proof    -   0% traffic (proof of multi-slot capability)

Both deployments run their own pod on the lambda-0 K8s cluster; both load
the same combined root+anchor model artifact; only `current` receives real
client requests. The `proof` deployment is purely there for the studio
graph and the `az ml online-deployment list` output to show two slots.

Usage:
    python scripts/add_proof_deployment.py

When the mentor / teacher has seen the proof, delete the second slot to
free K8s resources on lambda-0:

    az ml online-deployment delete --name proof ^
        --endpoint-name root-detection-endpoint ^
        --resource-group buas-y2 --workspace-name team-CV5-2026 --yes

This script does NOT modify endpoint.traffic — Azure ML automatically
leaves a brand-new deployment at 0% traffic until you explicitly assign
some, so `current` keeps serving 100% of requests throughout.
"""

import sys
from pathlib import Path

# Reuse all the constants + helpers from the main deploy script so this stays
# in sync if those change.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from azure.ai.ml.entities import (  # noqa: E402
    CodeConfiguration,
    KubernetesOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings,
)
from deploy_endpoint import (  # noqa: E402
    ENDPOINT_NAME,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    REPO_ROOT,
    ROOT_MODEL_NAME,
    SERVING_MODEL_NAME,
    ensure_environment,
    get_client,
    get_latest_model_version,
)

PROOF_DEPLOYMENT_NAME = "proof"


def main() -> None:
    """Create the second deployment slot on the existing endpoint."""
    client = get_client()

    # Resolve the latest serving model version (same as current uses).
    root_version = get_latest_model_version(client, ROOT_MODEL_NAME)
    try:
        serving = client.models.get(name=SERVING_MODEL_NAME, version=root_version)
        serving_version = serving.version
    except Exception:
        print(
            f"ERROR: serving model {SERVING_MODEL_NAME}:{root_version} not found. "
            f"Run `python scripts/deploy_endpoint.py` first to build it.",
            file=sys.stderr,
        )
        sys.exit(1)

    env_label = ensure_environment(client)

    print(f"Endpoint                : {ENDPOINT_NAME}")
    print(f"Adding deployment       : {PROOF_DEPLOYMENT_NAME}")
    print(f"Serving model           : {SERVING_MODEL_NAME} v{serving_version}")
    print(f"Environment             : {env_label}")
    print(f"Compute / instance_type : lambda-0 / {INSTANCE_TYPE}")
    print(f"Instance count          : {INSTANCE_COUNT}")
    print()

    deployment = KubernetesOnlineDeployment(
        name=PROOF_DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        # Same combined root+anchor artifact `current` uses.
        model=f"azureml:{SERVING_MODEL_NAME}:{serving_version}",
        environment=env_label,
        code_configuration=CodeConfiguration(
            code=str(REPO_ROOT),
            scoring_script="scripts/score.py",
        ),
        environment_variables={
            # MODEL_VERSION gets suffixed so logs make it obvious which pod
            # responded if you ever do call the proof slot directly via the
            # azureml-model-deployment header.
            "MODEL_VERSION": f"{ROOT_MODEL_NAME}-v{serving_version}-proof",
            "DISABLE_DISH_CROP": "0",
        },
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=180000,
            max_concurrent_requests_per_instance=1,
            max_queue_wait_ms=60000,
        ),
        liveness_probe=ProbeSettings(failure_threshold=30, timeout=2, period=10, initial_delay=10),
        readiness_probe=ProbeSettings(failure_threshold=30, timeout=10, period=10, initial_delay=10),
    )

    print(f"Deploying {PROOF_DEPLOYMENT_NAME} ...  (~5-10 min while the pod starts on lambda-0)")
    client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment {PROOF_DEPLOYMENT_NAME} created.")

    # IMPORTANT: do NOT update endpoint.traffic. By default the new
    # deployment receives 0% traffic, leaving `current` at 100%.

    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    traffic = endpoint.traffic or {}
    print()
    print("Endpoint state after deploy:")
    for dep_name in sorted({*traffic.keys(), PROOF_DEPLOYMENT_NAME, "current"}):
        pct = traffic.get(dep_name, 0)
        print(f"  {dep_name:<10} : {pct}% traffic")

    print()
    print("Verify:")
    print(
        f"  az ml online-deployment list --endpoint-name {ENDPOINT_NAME} "
        f"--resource-group buas-y2 --workspace-name team-CV5-2026 --output table"
    )
    print()
    print("Studio:")
    print(f"  https://ml.azure.com -> Endpoints -> {ENDPOINT_NAME} -> Deployments")
    print()
    print("Cleanup (when the demo is done):")
    print(
        f"  az ml online-deployment delete --name {PROOF_DEPLOYMENT_NAME} "
        f"--endpoint-name {ENDPOINT_NAME} "
        f"--resource-group buas-y2 --workspace-name team-CV5-2026 --yes"
    )


if __name__ == "__main__":
    main()
