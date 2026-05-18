"""Add image_uri to annotations and make prediction_id nullable.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-18

Two changes to the annotations table, motivated by the same feature
(upload-created annotations that aren't tied to a prediction):

1. ``prediction_id`` becomes NULLable. Annotations created by uploading a
   mask directly — rather than by correcting a model prediction — have no
   source prediction to reference.
2. ``image_uri`` is added. For upload-created annotations there is no
   prediction record to pull the input image URI from, so the annotation
   has to carry its own reference to the image it annotates.

Bundled together because they're a single logical change (supporting
upload-created annotations) and applying one without the other leaves the
table in a broken state.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ``image_uri`` and relax ``prediction_id`` to nullable."""
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
    """Restore ``prediction_id`` to NOT NULL and drop ``image_uri``.

    Will fail if any rows have ``prediction_id = NULL`` — the operator must
    delete or backfill upload-created annotations before downgrading.
    """
    op.alter_column(
        "annotations",
        "prediction_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.drop_column("annotations", "image_uri")
