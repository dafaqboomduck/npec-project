"""Add tip_photo_uri column to annotations table.

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"


def upgrade() -> None:
    """Add tip_photo_uri to annotations."""
    op.add_column(
        "annotations",
        sa.Column("tip_photo_uri", sa.Text, nullable=True),
    )


def downgrade() -> None:
    """Remove tip_photo_uri from annotations."""
    op.drop_column("annotations", "tip_photo_uri")
