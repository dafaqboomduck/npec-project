#!/usr/bin/env bash
# scripts/db-migrate.sh — thin wrapper around alembic.
#
# Why this exists:
#   The DATABASE_URL lives in .env, but `alembic` doesn't auto-load .env files.
#   Without this wrapper, every team member has to remember to `source .env`
#   (or `export $(grep -v '^#' .env | xargs)`) before running alembic, which
#   nobody does, and then they get a confusing "DATABASE_URL is not set" error.
#
# Usage:
#   ./scripts/db-migrate.sh upgrade head
#   ./scripts/db-migrate.sh current
#   ./scripts/db-migrate.sh history
#   ./scripts/db-migrate.sh revision --autogenerate -m "add foo"
#
# All arguments are passed through to alembic verbatim.

set -euo pipefail

# Resolve repo root from this script's location so the wrapper works no matter
# where you invoke it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Load .env if present. We only export entries that look like VAR=value so
# stray comments or blank lines don't blow up the shell.
if [[ -f .env ]]; then
    # shellcheck disable=SC2046
    export $(grep -E '^[A-Z_][A-Z0-9_]*=' .env | xargs -0 -d '\n' 2>/dev/null || grep -E '^[A-Z_][A-Z0-9_]*=' .env)
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "error: DATABASE_URL is not set." >&2
    echo "       Copy .env.example to .env and fill it in, or export the var manually." >&2
    exit 1
fi

# Hand off to alembic with the db/ config and whatever args the caller gave us.
exec alembic -c db/alembic.ini "$@"