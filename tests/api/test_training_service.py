from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.services.training import (
    reconcile_orphaned_training_runs,
)


@pytest.mark.asyncio
async def test_reconcile_orphaned_training_runs():
    mock_db = AsyncMock()
    # Mock the execute return
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    class Context:
        async def __aenter__(self):
            return mock_db

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_sessionmaker():
        def factory():
            return Context()

        return factory

    with patch("api.src.db.session.get_sessionmaker", new=mock_sessionmaker):
        await reconcile_orphaned_training_runs()

    mock_db.execute.assert_awaited_once()
