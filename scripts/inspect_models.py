"""Print metadata about every registered version of root-detection and shoot-detection.

Useful for spotting mix-ups — e.g. when a 'root-detection' version was actually
uploaded from a shoot-model artifact path. Compare descriptions, tags, path
basenames, and registration timestamps across versions to find duplicates.

Usage:
    python scripts/inspect_models.py
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    os.environ["PATH"] = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin" + os.pathsep + os.environ.get("PATH", "")
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    cfg = json.loads((REPO_ROOT / "config.json").read_text())
    client = MLClient(
        credential=AzureCliCredential(process_timeout=60),
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )

    for name in ("root-detection", "shoot-detection", "root-detection-serving"):
        print("=" * 78)
        print(f"MODEL: {name}")
        print("=" * 78)
        try:
            versions = sorted(client.models.list(name=name), key=lambda m: int(m.version))
        except Exception as exc:
            print(f"  (could not list: {exc})\n")
            continue
        if not versions:
            print("  (no versions registered)\n")
            continue
        for stub in versions:
            full = client.models.get(name=name, version=stub.version)
            print(f"  v{full.version}")
            print(f"    type        : {full.type}")
            print(f"    created     : {getattr(full, 'creation_context', None) and full.creation_context.created_at}")
            print(f"    description : {(full.description or '').strip()[:200]}")
            print(f"    tags        : {full.tags}")
            print(f"    path tail   : ...{(full.path or '')[-100:]}")
            print()


if __name__ == "__main__":
    main()
