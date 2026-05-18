"""Training run Pydantic schemas.

Used by the four /training/* endpoints and the TrainingRunRepository.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class TrainingRunStatus(str, Enum):
    """Training run lifecycle status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingStartRequest(BaseModel):
    """Optional hyperparameter overrides for POST /training/start."""

    learning_rate: float | None = None
    batch_size: int | None = None
    epochs: int | None = None


class TrainingRunSummary(BaseModel):
    """Returned by GET /training/runs (list item)."""

    run_id: UUID
    status: TrainingRunStatus
    started_at: datetime
    completed_at: datetime | None = None
    final_f1: float | None = None


class TrainingRunDetail(TrainingRunSummary):
    """Returned by GET /training/runs/{run_id}."""

    hyperparameters: dict[str, Any] | None = None
    model_version_id: UUID | None = None


class TrainingStartResponse(BaseModel):
    """Returned by POST /training/start."""

    run_id: UUID
    message: str
