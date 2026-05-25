"""Add image_hash column to annotations for deduplication.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"


def upgrade() -> None:
    """Add image hash to annotations."""
    op.add_column(
        "annotations",
        sa.Column("image_hash", sa.String(64), nullable=True),
    )
    op.add_column("predictions", sa.Column("input_hash", sa.String(64), nullable=True))
    # Partial unique index: only enforce uniqueness within a dataset
    op.create_index(
        "uq_annotations_dataset_hash",
        "annotations",
        ["dataset_id", "image_hash"],
        unique=True,
        postgresql_where="image_hash IS NOT NULL AND dataset_id IS NOT NULL",
    )


def downgrade() -> None:
    """Remove image hash to annotations."""
    op.drop_index("uq_annotations_dataset_hash", "annotations")
    op.drop_column("annotations", "image_hash")
    op.drop_column("predictions", "input_hash")
