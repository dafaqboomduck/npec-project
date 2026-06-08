"""Add annotations.exported_to_raw_at watermark.

Revision ID: 0013
Revises: 0012
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"  # or "0011" if you did not apply the queue migration
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
