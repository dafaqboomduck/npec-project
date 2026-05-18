"""Add image_uri to annotations and make prediction_id nullable.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-18
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add image_uri column and make prediction_id nullable."""
    op.add_column(
        "annotations",
        sa.Column("image_uri", sa.Text(), nullable=True),
    )
    op.alter_column(
        "annotations",
        "prediction_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )


def downgrade() -> None:
    """Remove image_uri and restore prediction_id NOT NULL.

    Note: downgrade will fail if any rows have prediction_id = NULL.
    Delete upload-created annotations before downgrading.
    """
    op.alter_column(
        "annotations",
        "prediction_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.drop_column("annotations", "image_uri")
