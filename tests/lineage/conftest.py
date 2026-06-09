import sys
from unittest.mock import MagicMock, patch

import pytest


class _FakeModel:
    """Stand-in for azure.ai.ml.entities.Model — stores constructor kwargs as attributes."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture(autouse=True, scope="session")
def _stub_azure_ml():
    """Inject fake azure.ai.ml modules so lineage tests run without the package installed."""
    try:
        import azure.ai.ml  # noqa: F401

        yield  # real package available — no patching needed
        return
    except ImportError:
        pass

    fake_entities = MagicMock()
    fake_entities.Model = _FakeModel

    with patch.dict(
        sys.modules,
        {
            "azure": MagicMock(),
            "azure.ai": MagicMock(),
            "azure.ai.ml": MagicMock(),
            "azure.ai.ml.entities": fake_entities,
            "azure.ai.ml.constants": MagicMock(),
            "azure.identity": MagicMock(),
        },
    ):
        yield
