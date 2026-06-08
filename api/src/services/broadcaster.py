"""Broadcaster service for SSE events.

This module provides a mechanism to broadcast events to active SSE client connections.
"""

import asyncio
from typing import Set


class EventBroadcaster:
    """Thread-safe broadcaster to manage SSE client subscriptions and distribute messages."""

    def __init__(self) -> None:
        """Initialize an empty set of active client listener queues."""
        self._listeners: Set[asyncio.Queue[str]] = set()

    def subscribe(self) -> asyncio.Queue[str]:
        """Subscribe a new client by returning an active listener queue."""
        q: asyncio.Queue[str] = asyncio.Queue()
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        """Unsubscribe a client queue when their connection terminates."""
        self._listeners.discard(q)

    async def broadcast(self, event_type: str) -> None:
        """Broadcast a message/event type to all registered subscriber queues."""
        for q in list(self._listeners):
            await q.put(event_type)


# Global singleton instance
inference_broadcaster = EventBroadcaster()
