"""Add add created_by to annotations.

Revision ID: 0009
Revises: 0008
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0009"
down_revision = "0008"


def upgrade() -> None:
    """Add add created_by to annotations."""
    op.add_column(
        "annotations",
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_annotations_created_by", "annotations", ["created_by"])


def downgrade() -> None:
    """Remove add created_by from annotations."""
    op.drop_index("ix_annotations_created_by")
    op.drop_column("annotations", "created_by")
