"""Database engine and session factory for the CV5 database.

Reads ``DATABASE_URL`` from the environment at import time. The URL format is
standard SQLAlchemy:

    postgresql+psycopg://user:pass@host:port/dbname

If you need to point at a different database (different env, test fixtures,
etc.) change the env var, not this file. See ``.env.example`` and
``docs/database/schema.md`` for supported URLs.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def _get_database_url() -> str:
    """Read ``DATABASE_URL`` from the environment.

    We intentionally do not fall back to a default connection string — a silent
    default leads to "works on my machine" bugs where the wrong database is
    written to. Better to fail fast at import time.

    Returns:
        The SQLAlchemy connection URL from the ``DATABASE_URL`` environment
        variable.

    Raises:
        RuntimeError: If ``DATABASE_URL`` is not set. The message tells the
            caller how to fix it.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Copy .env.example to .env and fill in your database URL, or "
            "export DATABASE_URL in your shell before starting the app."
        )
    return url


def build_engine(url: str | None = None) -> Engine:
    """Construct a SQLAlchemy engine.

    Exposed as a function (rather than a module-level constant) so tests can
    build throwaway engines pointed at SQLite or an ephemeral Postgres.

    Args:
        url: Optional SQLAlchemy connection URL. If ``None``, the URL is read
            from the ``DATABASE_URL`` environment variable via
            :func:`_get_database_url`.

    Returns:
        A SQLAlchemy engine with ``pool_pre_ping`` enabled.
    """
    return create_engine(
        url or _get_database_url(),
        # pool_pre_ping avoids "connection closed" errors after Postgres
        # restarts or long idle periods — cheap insurance.
        pool_pre_ping=True,
        future=True,
    )


# Module-level engine and session factory for the backend to import.
# These are built lazily on first access so that simply importing this module
# (e.g. for running Alembic) doesn't fail if DATABASE_URL isn't set yet.
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, building it on first access.

    Returns:
        The lazily-initialised SQLAlchemy engine bound to ``DATABASE_URL``.
    """
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide ``sessionmaker``, building it on first access.

    Returns:
        A configured ``sessionmaker`` bound to the engine from :func:`get_engine`.
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SessionLocal


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session for use as a FastAPI dependency.

    The session is closed when the generator is exhausted, so callers do not
    need to manage cleanup themselves. Typical use::

        @app.get("/predictions/{id}")
        def read(id: UUID, session: Session = Depends(get_session)):
            ...

    Yields:
        An open SQLAlchemy ``Session`` bound to the process-wide engine.
    """
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


__all__ = [
    "build_engine",
    "get_engine",
    "get_sessionmaker",
    "get_session",
]
