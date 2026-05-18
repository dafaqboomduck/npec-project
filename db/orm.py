"""SQLAlchemy ORM models for the CV5 database.

Matches ``docs/database/schema.md`` v0.4. Nine tables:

- users
- refresh_tokens
- model_versions
- predictions
- feedback
- monitoring_metrics
- datasets
- annotations
- training_runs

Any change to these models must be matched by an update to the schema doc
and a new Alembic migration (never hand-edit an old migration).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# ---------------------------------------------------------------------------
# Shared column helpers
# ---------------------------------------------------------------------------

# JSONB is Postgres-specific; SQLite (used in unit tests) falls back to JSON.
JSONB_or_JSON = JSONB().with_variant(JSON(), "sqlite")


def _uuid_pk() -> Mapped[uuid.UUID]:
    """Primary key column: server-generated UUID via pgcrypto's gen_random_uuid()."""
    return mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class User(Base):
    """Authenticated account. Matches schema §2.1."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(254), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="user")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="user")

    __table_args__ = (CheckConstraint("role IN ('user', 'admin')", name="role_valid"),)


# ---------------------------------------------------------------------------
# Refresh tokens
# ---------------------------------------------------------------------------


class RefreshToken(Base):
    """Server-side rotating refresh token. Matches schema §2.2."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    replaced_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("refresh_tokens.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = _created_at()

    user: Mapped["User"] = relationship(back_populates="refresh_tokens")

    __table_args__ = (
        Index("ix_refresh_tokens_user_id", "user_id"),
        Index("ix_refresh_tokens_expires_at", "expires_at"),
    )


# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------


class ModelVersion(Base):
    """Registry of trained U-Net versions. Matches schema §2.3."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)

    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inference_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    artifact_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = _created_at()
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")

    __table_args__ = (
        CheckConstraint(
            "status IN ('deployed', 'registered', 'archived')",
            name="status_valid",
        ),
        CheckConstraint("f1_score IS NULL OR (f1_score BETWEEN 0 AND 1)", name="f1_range"),
        CheckConstraint(
            "precision IS NULL OR (precision BETWEEN 0 AND 1)",
            name="precision_range",
        ),
        CheckConstraint("recall IS NULL OR (recall BETWEEN 0 AND 1)", name="recall_range"),
        Index("ix_model_versions_status", "status"),
        # Partial unique index: at most one row with status='deployed'.
        Index(
            "one_deployed_model",
            "status",
            unique=True,
            postgresql_where="status = 'deployed'",
        ),
    )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class Prediction(Base):
    """One row per POST /predict call. Matches schema §2.4."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)

    # Input image metadata
    input_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    input_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    input_width_px: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_height_px: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_channels: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # Inference metadata
    normalisation_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    scale_factor_mm_per_px: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Output
    segmentation_mask_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tip_photo_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    plants: Mapped[list[dict[str, Any]]] = mapped_column(JSONB_or_JSON, nullable=False, server_default="[]")
    warnings: Mapped[list[str]] = mapped_column(JSONB_or_JSON, nullable=False, server_default="[]")

    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    inference_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_at: Mapped[datetime] = _created_at()

    user: Mapped["User"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="prediction", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("status IN ('ok', 'error')", name="status_valid"),
        Index("ix_predictions_user_id", "user_id"),
        Index("ix_predictions_model_version_id", "model_version_id"),
        Index("ix_predictions_user_created", "user_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class Feedback(Base):
    """Thumbs up/down on a prediction. Matches schema §2.5.

    Multiple rows per (user_id, prediction_id) are allowed — users can update
    their opinion, and each submission is a new row. Consumers that need
    "current feedback" should read the most recent row by created_at.
    """

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = _uuid_pk()
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    signal: Mapped[str] = mapped_column(String(16), nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = _created_at()

    prediction: Mapped["Prediction"] = relationship(back_populates="feedback")
    user: Mapped["User"] = relationship(back_populates="feedback")

    __table_args__ = (
        CheckConstraint("signal IN ('correct', 'incorrect')", name="signal_valid"),
        Index("ix_feedback_prediction_id", "prediction_id"),
        Index("ix_feedback_user_id", "user_id"),
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class Dataset(Base):
    """A named collection of root-scan images. Matches schema §2.7."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    species: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    growth_stage: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Denormalised count — updated by the application layer or a DB trigger.
    count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = _created_at()
    modified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    annotations: Mapped[list["Annotation"]] = relationship(back_populates="dataset")

    __table_args__ = (
        CheckConstraint(
            "growth_stage IS NULL OR growth_stage IN ('seedling', 'mature')",
            name="ck_datasets_growth_stage_valid",
        ),
        Index("ix_datasets_species", "species"),
    )


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


class Annotation(Base):
    """Human-reviewed segmentation mask for a single scan. Matches schema §2.8.

    Annotations come from two sources:

    - **Prediction-derived**: created by correcting a model output. In this
      case ``prediction_id`` points at the source prediction (and the input
      image URI can be looked up via that prediction).
    - **Upload-created**: created by uploading a mask directly, with no
      backing prediction. In this case ``prediction_id`` is NULL and
      ``image_uri`` carries the reference to the annotated image.
    """

    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # NULLable: upload-created annotations have no source prediction.
    prediction_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=True,
    )
    dataset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="'pending'")
    mask_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Reference to the annotated input image. Required (in practice) for
    # upload-created annotations; prediction-derived annotations can leave
    # this NULL and look the image up via prediction_id.
    image_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    species: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    age_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    treatment: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    growth_stage: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    dataset: Mapped[Optional["Dataset"]] = relationship(back_populates="annotations")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'annotated', 'review_required')",
            name="ck_annotations_status_valid",
        ),
        CheckConstraint(
            "growth_stage IS NULL OR growth_stage IN ('seedling', 'mature')",
            name="ck_annotations_growth_stage_valid",
        ),
        Index("ix_annotations_prediction_id", "prediction_id"),
        Index("ix_annotations_dataset_id", "dataset_id"),
        Index("ix_annotations_status", "status"),
    )


# ---------------------------------------------------------------------------
# Monitoring metrics
# ---------------------------------------------------------------------------


class MonitoringMetric(Base):
    """Time-series metrics store. Matches schema §2.6.

    Uses a BIGSERIAL primary key (not UUID) because this table sees a high
    insert rate and queries are time-ordered.
    """

    __tablename__ = "monitoring_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    model_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    labels: Mapped[dict[str, Any]] = mapped_column(JSONB_or_JSON, nullable=False, server_default="{}")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_monitoring_metrics_name_recorded", "metric_name", "recorded_at"),
        Index("ix_monitoring_metrics_model_version_id", "model_version_id"),
    )


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------


class TrainingRun(Base):
    """One row per training job. Matches schema §2.9.

    Inserted with status='running' when a job starts. Updated to
    'completed' or 'failed' when the subprocess exits. Can be moved
    to 'stopped' via DELETE /training/runs/{id}.
    """

    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    hyperparameters: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB_or_JSON, nullable=True)
    final_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'stopped')",
            name="ck_training_runs_status_valid",
        ),
        Index("ix_training_runs_status", "status"),
        Index("ix_training_runs_model_version_id", "model_version_id"),
        Index("ix_training_runs_started_at", "started_at"),
    )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "User",
    "RefreshToken",
    "ModelVersion",
    "Prediction",
    "Feedback",
    "MonitoringMetric",
    "Dataset",
    "Annotation",
    "TrainingRun",
]
