"""Print container logs from the deployed endpoint.

Usage:
    python scripts/get_logs.py              # last 200 lines, diagnostic-filtered
    python scripts/get_logs.py --all        # last 200 lines, unfiltered
    python scripts/get_logs.py --lines 500  # request more lines
"""

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Lines containing any of these substrings are kept in filtered mode.
KEYWORDS = ("DIAGNOSTIC", "Calibrated", "Disabled", "Injected", "ERROR", "Traceback", "Exception", "FAILED")


def main() -> None:
    parser = argparse.ArgumentParser("get_logs")
    parser.add_argument("--all", action="store_true", help="Show ALL log lines, not just diagnostic-related ones.")
    parser.add_argument("--lines", type=int, default=300, help="How many recent log lines to fetch (default: 300).")
    args = parser.parse_args()

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

    print(f"Fetching last {args.lines} log lines ...")
    logs = client.online_deployments.get_logs(
        name="current",
        endpoint_name="root-detection-endpoint",
        lines=args.lines,
    )

    if args.all:
        print(logs)
        return

    matches = [line for line in logs.splitlines() if any(kw in line for kw in KEYWORDS)]
    if not matches:
        print("No diagnostic lines found. Send a test request first, then re-run this.")
        print("Or use --all to dump everything.")
        return

    print(f"\n=== Filtered to {len(matches)} relevant line(s) ===")
    for line in matches:
        print(line)


if __name__ == "__main__":
    main()
