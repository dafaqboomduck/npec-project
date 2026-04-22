"""SQLAlchemy declarative base for the CV5 database.

Sets a naming convention on MetaData so Alembic generates predictable
constraint names (e.g. ``fk_predictions_user_id_users``) instead of
database-backend-specific hashes. This matters because:

- It keeps migrations readable on review.
- It prevents spurious diffs when Alembic compares existing and expected
  constraints across Postgres versions.
- It makes dropping constraints by name deterministic in future migrations.

See: https://alembic.sqlalchemy.org/en/latest/naming.html
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all ORM models in the CV5 database."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
