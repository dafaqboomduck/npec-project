#!/usr/bin/env bash
# scripts/db-migrate.sh — thin wrapper around alembic that auto-loads .env.
#
# Usage (from the repo root):
#   ./scripts/db-migrate.sh upgrade head
#   ./scripts/db-migrate.sh revision --autogenerate -m "add foo column"
#   ./scripts/db-migrate.sh downgrade -1
#   ./scripts/db-migrate.sh current
#   ./scripts/db-migrate.sh history
#
# All arguments are passed through to `alembic` unchanged. The script's only
# job is to load .env (if present) and point alembic at db/alembic.ini.
#
# If DATABASE_URL is already exported in your shell, it takes precedence over
# what's in .env — that's standard env-var behaviour and keeps CI from being
# accidentally overridden by a stale .env file.

set -euo pipefail

# Resolve the repo root relative to this script so the wrapper works regardless
# of where it's invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load .env if it exists. `set -a` makes every assignment an automatic export.
# We use `set +a` to turn that back off afterwards so we don't pollute the
# rest of the script's environment.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Fail fast with a clear message if DATABASE_URL still isn't set. Alembic
# would also fail, but its error is less helpful than ours.
if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "ERROR: DATABASE_URL is not set." >&2
    echo "       Either export it in your shell, or copy .env.example to .env" >&2
    echo "       and fill it in. See db/session.py for the URL format." >&2
    exit 1
fi

exec alembic -c db/alembic.ini "$@"
