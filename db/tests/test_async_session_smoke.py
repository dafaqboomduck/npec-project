"""Smoke test for the async session factory.

Verifies that:

1. The engine builds from ``DATABASE_URL``.
2. A session can be opened.
3. A trivial round-trip query succeeds.
4. A query against a real table (``alembic_version``) works — proving
   migrations have been applied to the target DB.

This is intentionally minimal. It is **not** a unit test of the ORM. It is
the equivalent of pinging the database from the same code path the backend
uses, so that "the database layer works end-to-end after the async migration"
has a single, runnable check.

How to run:

    # 1. Ensure a Postgres is up and migrations have been applied
    docker compose -f docker-compose.dev.yml up -d db
    docker compose -f docker-compose.dev.yml run --rm migrations

    # 2. With DATABASE_URL in .env (or exported):
    pytest db/tests/test_async_session_smoke.py -v

If the database is unreachable, the test is skipped rather than failed — this
keeps it safe to leave in CI without forcing every job to provision Postgres.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from db.session import build_engine, get_session, get_sessionmaker

pytestmark = pytest.mark.asyncio


def _database_url_or_skip() -> str:
    """Return DATABASE_URL or skip the test if it's not set."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; skipping async smoke test.")
    return url


async def test_engine_can_connect() -> None:
    """A bare engine should accept a connection and run SELECT 1."""
    _database_url_or_skip()
    engine = build_engine()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
    except OperationalError as exc:
        pytest.skip(f"Database not reachable: {exc}")
    finally:
        await engine.dispose()


async def test_session_factory_yields_working_session() -> None:
    """``get_sessionmaker`` should produce sessions that can execute queries."""
    _database_url_or_skip()
    SessionLocal = get_sessionmaker()
    try:
        async with SessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
    except OperationalError as exc:
        pytest.skip(f"Database not reachable: {exc}")


async def test_get_session_dependency() -> None:
    """``get_session`` (the FastAPI dependency) should yield a usable session."""
    _database_url_or_skip()
    try:
        # Drive the async generator the way FastAPI would.
        gen = get_session()
        session = await gen.__anext__()
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1
        # Exhaust the generator so the session is closed cleanly.
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    except OperationalError as exc:
        pytest.skip(f"Database not reachable: {exc}")


async def test_alembic_version_table_exists() -> None:
    """If migrations have run, the ``alembic_version`` table should be present.

    This is a sanity check that the database the test is pointing at is the
    one migrations were applied to — not, for example, an empty test database
    that was never bootstrapped.
    """
    _database_url_or_skip()
    SessionLocal = get_sessionmaker()
    try:
        async with SessionLocal() as session:
            result = await session.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables "
                    "  WHERE table_schema = 'public' "
                    "  AND table_name = 'alembic_version'"
                    ")"
                )
            )
            assert result.scalar_one() is True, (
                "alembic_version table not found — has `alembic upgrade head` " "been run against this database?"
            )
    except OperationalError as exc:
        pytest.skip(f"Database not reachable: {exc}")
