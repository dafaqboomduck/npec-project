"""Register the shoot-detection and root-detection Keras models in Azure ML."""

import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import AzureCliCredential

# ── Fill in your local model paths here ──────────────────────────────────────
SHOOT_MODEL_PATH = r"C:\Users\koenm\Downloads\root_det_model_512px.keras"
ROOT_MODEL_PATH = r"C:\Users\koenm\Downloads\anchor_det_model_512px.keras"
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def get_client() -> MLClient:
    """Create and return an authenticated MLClient."""
    config = json.loads(CONFIG_PATH.read_text())
    return MLClient(
        credential=AzureCliCredential(process_timeout=30),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )


def register_model(client: MLClient, name: str, path: str, description: str) -> None:
    """Upload a local .keras file and register it in the Azure ML model registry."""
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Model file not found: {local_path}")

    print(f"Registering '{name}' from {local_path} ...")

    model = Model(
        path=str(local_path),
        name=name,
        description=description,
        type=AssetTypes.CUSTOM_MODEL,
    )

    registered = client.models.create_or_update(model)
    print(f"  Done — version {registered.version} registered.")
    print(f"  Azure ID: {registered.id}")


def main() -> None:
    """Register both models in the Azure ML workspace."""
    if not SHOOT_MODEL_PATH or not ROOT_MODEL_PATH:
        raise ValueError("Please fill in SHOOT_MODEL_PATH and ROOT_MODEL_PATH at the top of this script.")

    client = get_client()

    register_model(
        client=client,
        name="shoot-detection",
        path=SHOOT_MODEL_PATH,
        description="Keras model for shoot detection — Sprint 3 baseline",
    )

    register_model(
        client=client,
        name="root-detection",
        path=ROOT_MODEL_PATH,
        description="Keras model for root detection — Sprint 3 baseline",
    )

    print("\nBoth models registered. View them at:")
    print("  https://ml.azure.com > Assets > Models")


if __name__ == "__main__":
    main()
