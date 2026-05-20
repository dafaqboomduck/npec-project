"""Add pid column to training_runs.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-19
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add pid column to training_runs table."""
    op.add_column(
        "training_runs",
        sa.Column("pid", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Remove the pid column from training_runs table."""
    op.drop_column("training_runs", "pid")
