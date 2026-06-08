"""CLI wrapper around the in-process feedback export.

Thin entry point for manual/scheduled runs. The actual logic lives in
api/src/services/feedback_export.py (export_confirmed_feedback), which the
API also calls in-process — so the CLI and the app share one implementation.

    python scripts/export_feedback_to_raw.py            # export new confirmed pairs
    python scripts/export_feedback_to_raw.py --limit 50
    python scripts/export_feedback_to_raw.py --rebuild  # re-materialise ALL confirmed pairs (recovery)

Env (cloud): DATABASE_URL, DEPLOYMENT_ENV=cloud, AZURE_STORAGE_CONNECTION_STRING,
AZURE_STORAGE_CONTAINER, AZUREML_SUBSCRIPTION_ID/RESOURCE_GROUP/WORKSPACE,
NPEC_CORRECTIONS_DIR, and a credential (service principal or `az login`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from api.src.db.session import get_sessionmaker
from api.src.services.feedback_export import export_confirmed_feedback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max pairs to export this run.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore the watermark and re-materialise ALL confirmed pairs (recovery).",
    )
    args = parser.parse_args()
    load_dotenv()

    async def _run() -> int:
        async with get_sessionmaker()() as db:
            return await export_confirmed_feedback(db, limit=args.limit, rebuild=args.rebuild)

    added = asyncio.run(_run())
    print(f"Exported {added} confirmed pair(s) into cv5_corrections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
