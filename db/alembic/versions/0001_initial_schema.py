"""Initial schema.

Revision ID: 0001
Revises:
Create Date: 2026-04-22

Creates the six Sprint 2 tables from scratch:

- users
- refresh_tokens
- model_versions
- predictions
- feedback
- monitoring_metrics

Tables are created in dependency order (parents before children) so the
downgrade step can drop them in reverse without FK violations.

The pgcrypto extension is enabled first because every table uses
gen_random_uuid() for default primary keys.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all Sprint 2 tables and the ``pgcrypto`` extension."""
    # -----------------------------------------------------------------------
    # Extensions
    # -----------------------------------------------------------------------
    # pgcrypto provides gen_random_uuid(). IF NOT EXISTS makes this idempotent.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # -----------------------------------------------------------------------
    # users
    # -----------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.CheckConstraint("role IN ('user', 'admin')", name="ck_users_role_valid"),
    )

    # -----------------------------------------------------------------------
    # refresh_tokens
    # -----------------------------------------------------------------------
    op.create_table(
        "refresh_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_refresh_tokens"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_refresh_tokens_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["replaced_by"],
            ["refresh_tokens.id"],
            name="fk_refresh_tokens_replaced_by_refresh_tokens",
        ),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_expires_at", "refresh_tokens", ["expires_at"])

    # -----------------------------------------------------------------------
    # model_versions
    # -----------------------------------------------------------------------
    op.create_table(
        "model_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("inference_latency_ms", sa.Integer(), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
        sa.UniqueConstraint("version", name="uq_model_versions_version"),
        sa.CheckConstraint(
            "status IN ('deployed', 'registered', 'archived')",
            name="ck_model_versions_status_valid",
        ),
        sa.CheckConstraint(
            "f1_score IS NULL OR (f1_score BETWEEN 0 AND 1)",
            name="ck_model_versions_f1_range",
        ),
        sa.CheckConstraint(
            "precision IS NULL OR (precision BETWEEN 0 AND 1)",
            name="ck_model_versions_precision_range",
        ),
        sa.CheckConstraint(
            "recall IS NULL OR (recall BETWEEN 0 AND 1)",
            name="ck_model_versions_recall_range",
        ),
    )
    op.create_index("ix_model_versions_status", "model_versions", ["status"])
    # Partial unique index: at most one row with status='deployed' at a time.
    op.create_index(
        "one_deployed_model",
        "model_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'deployed'"),
    )

    # -----------------------------------------------------------------------
    # predictions
    # -----------------------------------------------------------------------
    op.create_table(
        "predictions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("input_filename", sa.String(255), nullable=True),
        sa.Column("input_uri", sa.Text(), nullable=True),
        sa.Column("input_width_px", sa.Integer(), nullable=True),
        sa.Column("input_height_px", sa.Integer(), nullable=True),
        sa.Column("input_channels", sa.SmallInteger(), nullable=True),
        sa.Column(
            "normalisation_applied",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("scale_factor_mm_per_px", sa.Float(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("segmentation_mask_uri", sa.Text(), nullable=True),
        sa.Column("tip_photo_uri", sa.Text(), nullable=True),
        sa.Column(
            "plants",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "inference_timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_predictions"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_predictions_user_id_users",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_predictions_model_version_id_model_versions",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("status IN ('ok', 'error')", name="ck_predictions_status_valid"),
    )
    op.create_index("ix_predictions_user_id", "predictions", ["user_id"])
    op.create_index("ix_predictions_model_version_id", "predictions", ["model_version_id"])
    op.create_index("ix_predictions_user_created", "predictions", ["user_id", "created_at"])

    # -----------------------------------------------------------------------
    # feedback
    # -----------------------------------------------------------------------
    op.create_table(
        "feedback",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("signal", sa.String(16), nullable=False),
        sa.Column("comment", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feedback"),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name="fk_feedback_prediction_id_predictions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_feedback_user_id_users",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("signal IN ('correct', 'incorrect')", name="ck_feedback_signal_valid"),
    )
    op.create_index("ix_feedback_prediction_id", "feedback", ["prediction_id"])
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"])

    # -----------------------------------------------------------------------
    # monitoring_metrics
    # -----------------------------------------------------------------------
    op.create_table(
        "monitoring_metrics",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("metric_name", sa.String(64), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "labels",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_monitoring_metrics"),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_monitoring_metrics_model_version_id_model_versions",
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "ix_monitoring_metrics_name_recorded",
        "monitoring_metrics",
        ["metric_name", "recorded_at"],
    )
    op.create_index(
        "ix_monitoring_metrics_model_version_id",
        "monitoring_metrics",
        ["model_version_id"],
    )


def downgrade() -> None:
    """Drop all Sprint 2 tables in reverse dependency order.

    The ``pgcrypto`` extension is intentionally not dropped — another schema
    in the same database might still need it. Removing extensions should be
    an explicit operator decision, not a migration side effect.
    """
    # Drop in reverse dependency order. Indexes are dropped implicitly with
    # their tables, but the partial index needs no special handling because
    # it's owned by model_versions.
    op.drop_table("monitoring_metrics")
    op.drop_table("feedback")
    op.drop_table("predictions")
    op.drop_table("model_versions")
    op.drop_table("refresh_tokens")
    op.drop_table("users")
