"""Alembic environment configuration.

Wires up Alembic to the SQLAlchemy models. Key points:

- ``DATABASE_URL`` is read from the environment, not from alembic.ini. This
  keeps connection strings out of the codebase (task 6.3).
- We add the repo root to ``sys.path`` so ``from db import Base`` works when
  Alembic is run from the repo root.
- Importing ``db`` also imports ``db.orm``, which registers every model class
  on ``Base.metadata`` — that metadata is what Alembic compares against the
  live database when autogenerating migrations.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

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


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — executes against the target database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
