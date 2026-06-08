"""Serialises deep-learning inference to prevent CUDA OOM / CPU starvation.

Baseline (always on): a per-process asyncio.Semaphore caps concurrent model
evaluations (default 1). For a single shared GPU across multiple replicas,
enable cluster single-flight to serialise across the whole deployment via a
Postgres advisory lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import AsyncIterator

from api.src.services.locks import INFERENCE_SINGLEFLIGHT_KEY, advisory_session_lock

logger = logging.getLogger(__name__)

_MAX_CONCURRENCY = int(os.getenv("INFERENCE_MAX_CONCURRENCY", "1"))
_CLUSTER_SINGLEFLIGHT = os.getenv("INFERENCE_CLUSTER_SINGLEFLIGHT", "false").lower() == "true"


class InferenceGate:
    """Bounds concurrent model evaluations within (and optionally across) workers."""

    def __init__(self) -> None:
        """Intialise the Inference Gate."""
        self._sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire an inference slot for the duration of the block."""
        async with self._sem:
            if not _CLUSTER_SINGLEFLIGHT:
                yield
                return
            # Cluster-wide single-flight: hold a Postgres advisory lock on a
            # dedicated connection for the inference duration.
            from api.src.db.session import get_sessionmaker

            async with get_sessionmaker()() as lock_db:
                async with advisory_session_lock(lock_db, INFERENCE_SINGLEFLIGHT_KEY):
                    yield


_gate: InferenceGate | None = None


def get_inference_gate() -> InferenceGate:
    """Return the process-wide inference gate singleton."""
    global _gate
    if _gate is None:
        _gate = InferenceGate()
    return _gate
