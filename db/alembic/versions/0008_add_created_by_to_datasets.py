"""Add add created_by to datasets.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"


def upgrade() -> None:
    """Add add created_by to datasets."""
    op.add_column(
        "datasets",
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_datasets_created_by", "datasets", ["created_by"])


def downgrade() -> None:
    """Remove add created_by from datasets."""
    op.drop_index("ix_datasets_created_by")
    op.drop_column("datasets", "created_by")
