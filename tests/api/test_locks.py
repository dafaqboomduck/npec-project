import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.src.services.locks import (
    advisory_session_lock,
    advisory_xact_lock,
    dataset_lock_key,
)


def test_dataset_lock_key():
    ds_id = uuid.uuid4()
    key = dataset_lock_key(ds_id)
    assert isinstance(key, int)
    # Ensure it's stable
    assert dataset_lock_key(ds_id) == key


@pytest.mark.asyncio
async def test_advisory_xact_lock_acquired():
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = True
    mock_db.execute.return_value = mock_result

    async with advisory_xact_lock(mock_db, 123) as acquired:
        assert acquired is True

    # Ensure it was called with the correct SQL and key parameter
    mock_db.execute.assert_called_once()
    sql_arg = mock_db.execute.call_args[0][0]
    assert "pg_try_advisory_xact_lock" in str(sql_arg)
    assert mock_db.execute.call_args[0][1] == {"k": 123}


@pytest.mark.asyncio
async def test_advisory_xact_lock_failed():
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = False
    mock_db.execute.return_value = mock_result

    async with advisory_xact_lock(mock_db, 456) as acquired:
        assert acquired is False


@pytest.mark.asyncio
async def test_advisory_session_lock():
    mock_db = AsyncMock()

    async with advisory_session_lock(mock_db, 789):
        pass

    assert mock_db.execute.call_count == 2
    # First call is lock, second is unlock
    call1_sql = mock_db.execute.call_args_list[0][0][0]
    call2_sql = mock_db.execute.call_args_list[1][0][0]
    assert "pg_advisory_lock" in str(call1_sql)
    assert "pg_advisory_unlock" in str(call2_sql)
    assert mock_db.execute.call_args_list[0][0][1] == {"k": 789}
    assert mock_db.execute.call_args_list[1][0][1] == {"k": 789}
