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
LineageContext, DatasetRef, gather_lineage_context
    Collect provenance facts from the current execution environment.

register_model_with_lineage
    Wrapper around ``MLClient.models.create_or_update`` that refuses
    to register a model without a git commit and writes all lineage
    facts as model tags.

register_baseline_model
    Explicit escape hatch for pre-pipeline / externally-trained models
    that genuinely have no upstream training run. Tags the model as
    ``baseline=true`` so trace output reports it honestly rather than
    treating it as a broken lineage record.

lineage_from_job
    Build a LineageContext from a completed Azure ML job.

InferenceLogger
    Per-prediction audit logger. Writes one row to ``inference_audit_log``.

trace
    Walk the lineage chain backward from a prediction id or request id.
"""

from lineage.context import (
    DatasetRef,
    LineageContext,
    LineageError,
    gather_lineage_context,
)
from lineage.inference_logger import InferenceLogger
from lineage.registration import (
    lineage_from_job,
    register_baseline_model,
    register_model_with_lineage,
)
from lineage.traceability import TraceResult, trace

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
