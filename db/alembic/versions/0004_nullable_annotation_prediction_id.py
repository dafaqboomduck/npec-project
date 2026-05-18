# db/alembic/versions/0005_nullable_annotation_prediction_id.py
"""Make annotations.prediction_id nullable for upload-created annotations.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-18
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow annotations.prediction_id to be NULL."""
    op.alter_column(
        "annotations",
        "prediction_id",
        existing_type=sa.dialects.postgresql.UUID(as_uuid=True),
        nullable=True,
    )


def downgrade() -> None:
    """Revert annotations.prediction_id to NOT NULL.

    Note: will fail if any rows have prediction_id = NULL.
    Delete upload-created annotations first.
    """
    op.alter_column(
        "annotations",
        "prediction_id",
        existing_type=sa.dialects.postgresql.UUID(as_uuid=True),
        nullable=False,
    )
