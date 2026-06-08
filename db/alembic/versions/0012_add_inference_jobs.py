"""Durable inference job queue.

Revision ID: 0012
Revises: 0011
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create inference_jobs table."""
    op.create_table(
        "inference_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("kind", sa.String(16), nullable=False),  # 'single' | 'batch'
        sa.Column("dataset_id", UUID(as_uuid=True), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True),
        sa.Column(
            "annotation_id", UUID(as_uuid=True), sa.ForeignKey("annotations.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("image_uri", sa.Text, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("worker_id", sa.String(64), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("requested_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('queued','running','done','failed')", name="ck_inference_jobs_status_valid"),
        sa.CheckConstraint("kind IN ('single','batch')", name="ck_inference_jobs_kind_valid"),
    )
    op.create_index("ix_inference_jobs_status", "inference_jobs", ["status"])
    op.create_index("ix_inference_jobs_dataset_id", "inference_jobs", ["dataset_id"])
    op.create_index("ix_inference_jobs_queued_at", "inference_jobs", ["queued_at"])


def downgrade() -> None:
    """Remove inference_jobs table."""
    op.drop_index("ix_inference_jobs_queued_at", table_name="inference_jobs")
    op.drop_index("ix_inference_jobs_dataset_id", table_name="inference_jobs")
    op.drop_index("ix_inference_jobs_status", table_name="inference_jobs")
    op.drop_table("inference_jobs")
