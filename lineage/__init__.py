"""Lineage and audit logging for the CV5 deployment.

This module gives every prediction, training run, and registered model
a traceable identity. The chain it supports:

    prediction -> model_version -> training_job -> dataset_versions -> git_commit

There is one storage backend: the project's own PostgreSQL database
(via the existing db/ module). The same code path runs on all three
deployment targets (local, Portainer, Azure ML on lambda-0); each
deployment has its own database, but the schema and code are identical.

Public API
----------
LineageContext, DatasetRef, gather_lineage_context, LineageError
    Collect provenance facts from the current execution environment.
    Pure Python; no external dependencies beyond the standard library.

InferenceLogger
    Per-prediction audit logger. Writes one row to ``inference_audit_log``.
    Requires sqlalchemy (already a project dependency).

trace, TraceResult
    Walk the lineage chain backward from a prediction id or request id.
    Requires sqlalchemy (already a project dependency).

register_model_with_lineage, register_baseline_model, lineage_from_job
    Register Azure ML models with lineage tags. Requires ``azure-ai-ml``.
    Only used by ``scripts/submit_training_job.py``, which runs on a
    developer laptop or CI host — NOT in the API container. These names
    are exported lazily so importing ``lineage`` does not require the
    azure-ai-ml package to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Eager imports: only modules with dependencies the API container already has.
from lineage.context import (
    DatasetRef,
    LineageContext,
    LineageError,
    gather_lineage_context,
)
from lineage.inference_logger import InferenceLogger
from lineage.traceability import TraceResult, trace

# Lazy imports: anything that requires azure-ai-ml. Looked up only when
# the consumer actually references the name.
_LAZY_NAMES = {
    "lineage_from_job",
    "register_baseline_model",
    "register_model_with_lineage",
}


def __getattr__(name: str):
    """Lazily resolve Azure-ML-dependent symbols.

    Called by Python when ``lineage.<name>`` is accessed and not already
    bound in the module's namespace. Keeps ``azure-ai-ml`` out of the
    eager import chain so the API container (which does not install it)
    can still ``import lineage`` and use ``InferenceLogger``.
    """
    if name in _LAZY_NAMES:
        from lineage import registration  # imports azure-ai-ml

        value = getattr(registration, name)
        globals()[name] = value  # cache for future lookups
        return value
    raise AttributeError(f"module 'lineage' has no attribute {name!r}")


if TYPE_CHECKING:
    # Help static analyzers (mypy, Pyright, IDE autocomplete) see the
    # lazy names. Not executed at runtime.
    from lineage.registration import (  # noqa: F401
        lineage_from_job,
        register_baseline_model,
        register_model_with_lineage,
    )


__all__ = [
    "DatasetRef",
    "LineageContext",
    "LineageError",
    "gather_lineage_context",
    "lineage_from_job",
    "register_baseline_model",
    "register_model_with_lineage",
    "InferenceLogger",
    "trace",
    "TraceResult",
]
