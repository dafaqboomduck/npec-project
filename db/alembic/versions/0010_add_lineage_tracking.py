"""Add lineage tracking: columns on model_versions and training_runs, plus inference_audit_log.

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-22

Tickets: 2291, 2292, 2293.

Four changes:

1. ``model_versions``: add lineage columns linking a registered model
   back to its training job, code commit, and data. Plus a ``baseline``
   flag so pre-pipeline models can be honestly marked as such (and
   excluded from lineage assertions) rather than appearing "broken."

2. ``training_runs``: add columns capturing the Azure ML job name, git
   commit, environment, and dataset versions.

3. ``inference_audit_log``: new immutable, append-only table. One row
   per inference call. Survives deletion of the source prediction row
   (FK is SET NULL on delete) so the audit chain remains intact.

Note on revision numbering: an earlier draft of this migration was numbered
0006, but the dev branch advanced through 0006-0009 with other unrelated
schema changes (image hashes, created_by columns). This migration is
therefore renumbered to 0010 and rebased onto 0009.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply schema additions for lineage tracking."""
    # ── 1. model_versions ────────────────────────────────────────────
    op.add_column("model_versions", sa.Column("git_commit", sa.String(40), nullable=True))
    op.add_column("model_versions", sa.Column("git_branch", sa.String(255), nullable=True))
    op.add_column("model_versions", sa.Column("git_repo", sa.Text(), nullable=True))
    op.add_column("model_versions", sa.Column("training_job_name", sa.String(255), nullable=True))
    op.add_column("model_versions", sa.Column("pipeline_run_id", sa.String(255), nullable=True))
    op.add_column("model_versions", sa.Column("mlflow_run_id", sa.String(64), nullable=True))
    op.add_column(
        "model_versions",
        sa.Column("dataset_versions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("model_versions", sa.Column("environment", sa.String(255), nullable=True))
    op.add_column(
        "model_versions",
        sa.Column(
            "baseline",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "ix_model_versions_training_job_name",
        "model_versions",
        ["training_job_name"],
    )
    op.create_index(
        "ix_model_versions_git_commit",
        "model_versions",
        ["git_commit"],
    )
    op.create_index(
        "ix_model_versions_baseline",
        "model_versions",
        ["baseline"],
    )

    # ── 2. training_runs ─────────────────────────────────────────────
    op.add_column("training_runs", sa.Column("azureml_job_name", sa.String(255), nullable=True))
    op.add_column("training_runs", sa.Column("git_commit", sa.String(40), nullable=True))
    op.add_column("training_runs", sa.Column("environment", sa.String(255), nullable=True))
    op.add_column(
        "training_runs",
        sa.Column("dataset_versions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_training_runs_azureml_job_name",
        "training_runs",
        ["azureml_job_name"],
    )

    # ── 3. inference_audit_log ───────────────────────────────────────
    op.create_table(
        "inference_audit_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Denormalised so the audit row stays useful after FKs go NULL.
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("model_version_label", sa.String(64), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("input_ref", sa.Text(), nullable=False),
        sa.Column(
            "output_summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("endpoint", sa.String(64), nullable=False, server_default="api"),
        sa.Column("deployment", sa.String(32), nullable=False),
        sa.Column(
            "logged_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_inference_audit_log"),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name="fk_audit_prediction_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_audit_model_version_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "deployment IN ('local', 'portainer', 'azureml', 'unknown')",
            name="ck_audit_deployment_valid",
        ),
    )
    op.create_index("ix_audit_request_id", "inference_audit_log", ["request_id"])
    op.create_index("ix_audit_prediction_id", "inference_audit_log", ["prediction_id"])
    op.create_index("ix_audit_model_version_id", "inference_audit_log", ["model_version_id"])
    op.create_index("ix_audit_logged_at", "inference_audit_log", ["logged_at"])


def downgrade() -> None:
    """Revert lineage tracking additions."""
    op.drop_index("ix_audit_logged_at", table_name="inference_audit_log")
    op.drop_index("ix_audit_model_version_id", table_name="inference_audit_log")
    op.drop_index("ix_audit_prediction_id", table_name="inference_audit_log")
    op.drop_index("ix_audit_request_id", table_name="inference_audit_log")
    op.drop_table("inference_audit_log")

    op.drop_index("ix_training_runs_azureml_job_name", table_name="training_runs")
    op.drop_column("training_runs", "dataset_versions")
    op.drop_column("training_runs", "environment")
    op.drop_column("training_runs", "git_commit")
    op.drop_column("training_runs", "azureml_job_name")

    op.drop_index("ix_model_versions_baseline", table_name="model_versions")
    op.drop_index("ix_model_versions_git_commit", table_name="model_versions")
    op.drop_index("ix_model_versions_training_job_name", table_name="model_versions")
    op.drop_column("model_versions", "baseline")
    op.drop_column("model_versions", "environment")
    op.drop_column("model_versions", "dataset_versions")
    op.drop_column("model_versions", "mlflow_run_id")
    op.drop_column("model_versions", "pipeline_run_id")
    op.drop_column("model_versions", "training_job_name")
    op.drop_column("model_versions", "git_repo")
    op.drop_column("model_versions", "git_branch")
    op.drop_column("model_versions", "git_commit")
