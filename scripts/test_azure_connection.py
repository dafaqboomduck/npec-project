"""Verify connectivity to the Azure ML workspace."""

import json
import os
from pathlib import Path

# Ensure the Azure CLI wbin directory is on PATH so AzureCliCredential can find az.cmd.
# Needed when the process was started before the CLI was installed (PATH not yet refreshed).
_AZ_WBIN = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
if _AZ_WBIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _AZ_WBIN + os.pathsep + os.environ.get("PATH", "")

from azure.ai.ml import MLClient  # noqa: E402
from azure.identity import AzureCliCredential  # noqa: E402

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def main() -> None:
    """Connect to the Azure ML workspace and print its details."""
    config = json.loads(CONFIG_PATH.read_text())

    client = MLClient(
        credential=AzureCliCredential(process_timeout=30),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )

    ws = client.workspaces.get(config["workspace_name"])
    print(f"Connected to workspace: {ws.name}")
    print(f"  Location      : {ws.location}")
    print(f"  Resource group: {ws.resource_group}")


if __name__ == "__main__":
    main()
