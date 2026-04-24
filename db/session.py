"""Async database engine and session factory for the CV5 database.

Reads ``DATABASE_URL`` from the environment at import time. The URL format is
standard SQLAlchemy — we use the async psycopg 3 driver:

    postgresql+psycopg://user:pass@host:port/dbname

psycopg 3 supports both sync and async under the same URL scheme; SQLAlchemy
picks the async path because we use :func:`create_async_engine`. No change to
the URL string is needed when going from sync to async.

If you need to point at a different database (different env, test fixtures,
etc.) change the env var, not this file. See ``.env.example`` and
``docs/database/schema.md`` for supported URLs.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


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


def build_engine(url: str | None = None) -> AsyncEngine:
    """Construct an async SQLAlchemy engine.

    Exposed as a function (rather than a module-level constant) so tests can
    build throwaway engines pointed at an ephemeral Postgres or aiosqlite.

    Args:
        url: Optional SQLAlchemy connection URL. If ``None``, the URL is read
            from the ``DATABASE_URL`` environment variable via
            :func:`_get_database_url`.

    Returns:
        An async SQLAlchemy engine with ``pool_pre_ping`` enabled.
    """
    return create_async_engine(
        url or _get_database_url(),
        # pool_pre_ping avoids "connection closed" errors after Postgres
        # restarts or long idle periods — cheap insurance.
        pool_pre_ping=True,
        future=True,
    )


# Module-level engine and session factory for the backend to import.
# These are built lazily on first access so that simply importing this module
# (e.g. for running Alembic) doesn't fail if DATABASE_URL isn't set yet.
_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async SQLAlchemy engine, building it on first access.

    Returns:
        The lazily-initialised async SQLAlchemy engine bound to ``DATABASE_URL``.
    """
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide ``async_sessionmaker``, building it on first access.

    Returns:
        A configured ``async_sessionmaker`` bound to the engine from
        :func:`get_engine`.
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = async_sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SessionLocal


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session for use as a FastAPI dependency.

    The session is closed when the generator is exhausted, so callers do not
    need to manage cleanup themselves. Typical use::

        @app.get("/predictions/{id}")
        async def read(id: UUID, session: AsyncSession = Depends(get_session)):
            ...

    Yields:
        An open ``AsyncSession`` bound to the process-wide engine.
    """
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        yield session


__all__ = [
    "build_engine",
    "get_engine",
    "get_sessionmaker",
    "get_session",
]
