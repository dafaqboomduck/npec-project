"""Add datasets and annotations tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-26

Adds two new tables introduced in the Library feature:

- datasets  — named collections of root-scan images
- annotations — human-reviewed segmentation masks linked to a prediction scan

datasets has no FK dependencies on existing tables and is created first.
annotations references both predictions (scan_id) and datasets (dataset_id),
so it is created after both of its parents.

Downgrade drops annotations first (child), then datasets (parent).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the datasets and annotations tables."""
    # -----------------------------------------------------------------------
    # datasets
    # -----------------------------------------------------------------------
    op.create_table(
        "datasets",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("species", sa.String(128), nullable=True),
        sa.Column("growth_stage", sa.String(32), nullable=True),
        sa.Column(
            "count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "modified_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_datasets"),
        sa.CheckConstraint(
            "growth_stage IS NULL OR growth_stage IN ('seedling', 'mature')",
            name="ck_datasets_growth_stage_valid",
        ),
    )
    op.create_index("ix_datasets_species", "datasets", ["species"])

    # -----------------------------------------------------------------------
    # annotations
    # -----------------------------------------------------------------------
    op.create_table(
        "annotations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        # FK → predictions.id  (the scan this annotation belongs to)
        sa.Column("scan_id", postgresql.UUID(as_uuid=True), nullable=False),
        # FK → datasets.id  (optional — an annotation may not be in a dataset yet)
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "status",
            sa.String(32),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("mask_uri", sa.Text(), nullable=True),
        sa.Column("species", sa.String(128), nullable=True),
        sa.Column("age_days", sa.Integer(), nullable=True),
        sa.Column("treatment", sa.String(128), nullable=True),
        sa.Column("growth_stage", sa.String(32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_annotations"),
        sa.ForeignKeyConstraint(
            ["scan_id"],
            ["predictions.id"],
            name="fk_annotations_scan_id_predictions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_annotations_dataset_id_datasets",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'annotated', 'review_required')",
            name="ck_annotations_status_valid",
        ),
        sa.CheckConstraint(
            "growth_stage IS NULL OR growth_stage IN ('seedling', 'mature')",
            name="ck_annotations_growth_stage_valid",
        ),
    )
    op.create_index("ix_annotations_scan_id", "annotations", ["scan_id"])
    op.create_index("ix_annotations_dataset_id", "annotations", ["dataset_id"])
    op.create_index("ix_annotations_status", "annotations", ["status"])


def downgrade() -> None:
    """Drop annotations first (child), then datasets (parent)."""
    # Drop indexes explicitly before tables for clarity (Postgres drops them
    # automatically with the table, but being explicit is good practice).
    op.drop_index("ix_annotations_status", table_name="annotations")
    op.drop_index("ix_annotations_dataset_id", table_name="annotations")
    op.drop_index("ix_annotations_scan_id", table_name="annotations")
    op.drop_table("annotations")

    op.drop_index("ix_datasets_species", table_name="datasets")
    op.drop_table("datasets")
