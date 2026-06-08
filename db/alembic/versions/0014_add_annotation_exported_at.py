"""Add annotations.exported_to_raw_at watermark.

Revision ID: 0014
Revises: 0013
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add exported_at column to annotatinos."""
    op.add_column(
        "annotations",
        sa.Column("exported_to_raw_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_annotations_exported_to_raw_at", "annotations", ["exported_to_raw_at"])


def downgrade() -> None:
    """Remove exported_at column to annotatinos."""
    op.drop_index("ix_annotations_exported_to_raw_at", table_name="annotations")
    op.drop_column("annotations", "exported_to_raw_at")
