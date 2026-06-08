"""Add system resource metrics to inference_audit_log.

All columns nullable, so this is non-blocking: no backfill, older rows
and CPU-only hosts simply carry NULLs.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    ("cpu_pct_system", sa.Float()),
    ("cpu_pct_process", sa.Float()),
    ("cpu_count", sa.Integer()),
    ("ram_used_mb", sa.Integer()),
    ("ram_available_mb", sa.Integer()),
    ("gpu_pct", sa.Float()),
    ("vram_used_mb", sa.Integer()),
    ("disk_free_mb", sa.Integer()),
)


def upgrade() -> None:
    """Add the nullable resource-metric columns."""
    # Check if columns already exist (for idempotency during redeploys)
    inspector = sa.inspect(op.get_bind())
    existing_columns = {col["name"] for col in inspector.get_columns("inference_audit_log")}

    for name, type_ in _COLUMNS:
        if name not in existing_columns:
            op.add_column("inference_audit_log", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    """Drop the resource-metric columns in reverse order."""
    for name, _ in reversed(_COLUMNS):
        op.drop_column("inference_audit_log", name)
