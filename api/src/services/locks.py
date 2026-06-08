"""Postgres advisory-lock helpers for cross-worker mutual exclusion.

Advisory locks are namespaced by a 64-bit key. Stable keys are derived from
a fixed namespace plus a UUID so the same logical resource always maps to
the same lock across every worker/replica.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Fixed keys
INFERENCE_SINGLEFLIGHT_KEY = 815_000_001
TRAINING_START_KEY = 815_000_02
_DATASET_NS = 815_100_000  # dataset batch guard namespace


def dataset_lock_key(dataset_id) -> int:
    """Stable signed-bigint advisory key for a dataset's batch guard.

    Args:
        dataset_id: UUID of the dataset.

    Retunrs:
        int dataset lock key.
    """
    h = hashlib.sha1(str(dataset_id).encode()).digest()[:6]  # 48 bits
    return _DATASET_NS + int.from_bytes(h, "big")


@contextlib.asynccontextmanager
async def advisory_xact_lock(db: AsyncSession, key: int) -> AsyncIterator[bool]:
    """Try to take a transaction-scoped advisory lock (auto-released on commit/rollback).

    Yields True if acquired, False if already held by someone else. Use for
    short critical sections such as 'check-then-enqueue'.
    """
    got = (await db.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": key})).scalar()
    yield bool(got)


@contextlib.asynccontextmanager
async def advisory_session_lock(db: AsyncSession, key: int) -> AsyncIterator[None]:
    """Hold a session-scoped advisory lock for the duration of the block.

    Blocks until acquired; released explicitly on exit. Use to serialise long
    operations (e.g. a single model evaluation) across workers.
    """
    await db.execute(text("SELECT pg_advisory_lock(:k)"), {"k": key})
    try:
        yield
    finally:
        await db.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
