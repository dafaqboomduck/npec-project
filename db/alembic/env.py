"""Alembic environment configuration (async).

Wires up Alembic to the SQLAlchemy models. Key points:

- ``DATABASE_URL`` is read from the environment, not from alembic.ini. This
  keeps connection strings out of the codebase (task 6.3).
- We add the repo root to ``sys.path`` so ``from db import Base`` works when
  Alembic is run from the repo root.
- Importing ``db`` also imports ``db.orm``, which registers every model class
  on ``Base.metadata`` — that metadata is what Alembic compares against the
  live database when autogenerating migrations.
- The online migration path uses ``create_async_engine`` and runs the
  (synchronous) Alembic migration routine via ``connection.run_sync(...)``.
  Alembic's migration API itself is sync; ``run_sync`` bridges it to the
  async engine so we don't maintain two connection URLs.
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Make `db` importable regardless of where alembic is invoked from.
# ---------------------------------------------------------------------------
# This file lives at:  <repo_root>/db/alembic/env.py
# Repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db import Base  # noqa: E402  (import after sys.path manipulation)

# ---------------------------------------------------------------------------
# Alembic config object
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata Alembic uses for --autogenerate diffs.
target_metadata = Base.metadata

# Inject DATABASE_URL into the Alembic config dynamically.
# This happens after fileConfig() so we overwrite any sqlalchemy.url in the .ini.
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError(
        "DATABASE_URL is not set. Alembic needs it to know which database to "
        "migrate. Export it in your shell or load it from .env before running "
        "alembic."
    )
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emits SQL to stdout instead of executing it.

    Useful for generating a SQL file to apply manually (e.g. via a DBA).
    Offline mode doesn't actually connect, so there's no async work here.
    """
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # compare_type/compare_server_default make autogenerate pick up more
        # kinds of changes between the model and the live schema.
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure Alembic against a live sync connection and run migrations.

    Called from :func:`run_async_migrations` via ``connection.run_sync(...)``.
    The connection object passed in is a synchronous SQLAlchemy ``Connection``
    proxy that ``AsyncConnection.run_sync`` hands to us — Alembic's migration
    machinery is sync and this is the documented way to drive it from an
    async engine.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Build an async engine and run migrations over it."""
    # NullPool: migrations are a one-shot process. We don't want a pool hanging
    # around after the script exits, and pooling connections across a DDL run
    # can interact badly with some drivers.
    from sqlalchemy.pool import NullPool

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — executes against the target database.

    Alembic expects a plain sync entrypoint, so we kick off the async
    migration coroutine with :func:`asyncio.run` here.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
