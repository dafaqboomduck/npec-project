"""Drop the FK from inference_audit_log.prediction_id to predictions.id.

Revision ID: 0011
Revises: 0010

The audit log is informational with respect to predictions. The
prediction_id column records "which prediction was this audit row
written for" even when the prediction row itself has been deleted or
hasn't been inserted yet (e.g. when audit writes happen mid-transaction
before the prediction row is flushed).

Enforcing the FK creates a transaction-ordering constraint we don't
need: the audit logger has to wait for the prediction row to exist
before inserting. Dropping the FK removes that constraint while
preserving the column itself for the trace tool to read.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply schema additions for lineage tracking."""
    op.drop_constraint(
        "fk_audit_prediction_id",
        "inference_audit_log",
        type_="foreignkey",
    )


def downgrade() -> None:
    """Revert lineage tracking additions."""
    op.create_foreign_key(
        "fk_audit_prediction_id",
        "inference_audit_log",
        "predictions",
        ["prediction_id"],
        ["id"],
        ondelete="SET NULL",
    )
