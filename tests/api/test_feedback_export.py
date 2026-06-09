import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.services.feedback_export import (
    _materialise,
    _next_version,
    _register,
    export_confirmed_feedback,
)


@pytest.mark.asyncio
async def test_export_confirmed_feedback_no_rows():
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_db.execute.return_value = mock_result

    with patch("api.src.services.feedback_export.Path"):
        res = await export_confirmed_feedback(mock_db)
        assert res == 0


@pytest.mark.asyncio
async def test_export_confirmed_feedback_with_rows():
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [(uuid.uuid4(), "s3://image1.png", "s3://mask1.png")]
    mock_db.execute.return_value = mock_result

    mock_storage = MagicMock()
    mock_storage.download.return_value = b"fake_bytes"

    with (
        patch("api.src.services.storage.get_storage_backend", return_value=mock_storage),
        patch("api.src.services.feedback_export._materialise") as mock_materialise,
        patch("api.src.services.feedback_export._register", return_value="1.0.1") as mock_register,
        patch("api.src.services.feedback_export.Path"),
    ):

        res = await export_confirmed_feedback(mock_db, rebuild=False)
        assert res == 1
        mock_materialise.assert_called_once()
        mock_register.assert_called_once()
        mock_db.commit.assert_awaited_once()


def test_next_version():
    mock_ml_client = MagicMock()
    # Mock current version as 1.2.3
    mock_ml_client.data.get.return_value.version = "1.2.3"
    assert _next_version(mock_ml_client) == "1.2.4"

    # Mock failure to get latest version
    mock_ml_client.data.get.side_effect = RuntimeError("not found")
    assert _next_version(mock_ml_client) == "1.0.0"


def test_materialise():
    import numpy as np

    mock_storage = MagicMock()
    mock_storage.download.side_effect = [b"img_bytes", b"mask_bytes"]

    mock_img_np = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_msk_np = np.zeros((10, 10), dtype=np.uint8)

    with (
        patch("PIL.Image.open") as mock_open,
        patch("numpy.array") as mock_array,
        patch("PIL.Image.fromarray") as mock_fromarray,
    ):

        mock_array.side_effect = [mock_img_np, mock_msk_np]

        _materialise(mock_storage, uuid.uuid4(), "s3://img", "s3://msk", Path("/tmp/img"), Path("/tmp/msk"))

        assert mock_open.call_count == 2
        assert mock_fromarray.call_count == 2


def test_register_flow():
    mock_ml_client = MagicMock()
    mock_ml_client.data.get.return_value.version = "1.2.3"

    fake_azure = {
        "azure": MagicMock(),
        "azure.ai": MagicMock(),
        "azure.ai.ml": MagicMock(),
        "azure.ai.ml.entities": MagicMock(),
        "azure.ai.ml.constants": MagicMock(),
        "azure.identity": MagicMock(),
    }

    with (
        patch.dict(sys.modules, fake_azure),
        patch("api.src.services.feedback_export._ml_client", return_value=mock_ml_client),
    ):
        version = _register(Path("/tmp/corrections"), 5)
        assert version == "1.2.4"
        mock_ml_client.data.create_or_update.assert_called_once()
