import asyncio

import pytest

from api.src.services.broadcaster import EventBroadcaster


@pytest.mark.asyncio
async def test_event_broadcaster_subscribe_unsubscribe():
    broadcaster = EventBroadcaster()

    # Initially no listeners
    assert len(broadcaster._listeners) == 0

    # Subscribe
    q = broadcaster.subscribe()
    assert isinstance(q, asyncio.Queue)
    assert q in broadcaster._listeners
    assert len(broadcaster._listeners) == 1

    # Unsubscribe
    broadcaster.unsubscribe(q)
    assert q not in broadcaster._listeners
    assert len(broadcaster._listeners) == 0


@pytest.mark.asyncio
async def test_event_broadcaster_broadcast():
    broadcaster = EventBroadcaster()

    q1 = broadcaster.subscribe()
    q2 = broadcaster.subscribe()

    await broadcaster.broadcast("test_event")

    # Both queues should have the event
    assert q1.qsize() == 1
    assert q2.qsize() == 1

    assert await q1.get() == "test_event"
    assert await q2.get() == "test_event"
