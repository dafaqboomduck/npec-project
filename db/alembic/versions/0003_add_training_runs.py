"""Add training_runs table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-17
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the training_runs table."""
    op.create_table(
        "training_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hyperparameters", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("final_f1", sa.Float(), nullable=True),
        sa.Column(
            "model_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_training_runs"),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_training_runs_model_version_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'stopped')",
            name="ck_training_runs_status_valid",
        ),
    )
    op.create_index("ix_training_runs_status", "training_runs", ["status"])
    op.create_index("ix_training_runs_model_version_id", "training_runs", ["model_version_id"])
    op.create_index("ix_training_runs_started_at", "training_runs", ["started_at"])


def downgrade() -> None:
    """Drop the training_runs table."""
    op.drop_index("ix_training_runs_started_at", table_name="training_runs")
    op.drop_index("ix_training_runs_model_version_id", table_name="training_runs")
    op.drop_index("ix_training_runs_status", table_name="training_runs")
    op.drop_table("training_runs")
