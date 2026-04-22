# `db/` — Database Layer

PostgreSQL schema, SQLAlchemy ORM models, and Alembic migrations for CV5. Imported by the backend; independent of `src/npec_cv/`.

---

## Layout

The database layer is mostly inside `db/`, but a few related files live elsewhere in the repo. This tree shows only the database-related files:

```
<repo-root>/
├── db/
│   ├── README.md                ← you are here
│   ├── __init__.py              Re-exports Base + all ORM classes
│   ├── base.py                  Declarative Base with constraint naming convention
│   ├── orm.py                   The six ORM models
│   ├── session.py               Engine + session factory (reads DATABASE_URL)
│   ├── alembic.ini              Alembic config
│   ├── alembic/
│   │   ├── env.py               Wires DATABASE_URL + Base.metadata into Alembic
│   │   ├── script.py.mako       Template for new migration files
│   │   └── versions/
│   │       └── 0001_initial_schema.py
│   ├── seeds/
│   │   └── dev_seed.sql         Dev-only seed (admin user + placeholder model)
│   └── Dockerfile               One-shot image for running migrations
│
├── docker-compose.yml           Local Postgres + migrations services
├── .env.example                 Template for DATABASE_URL and compose vars
│
├── scripts/
│   └── db-migrate.sh            Wrapper around alembic that auto-loads .env
│
└── docs/
    └── database/
        ├── schema.md            Authoritative schema (tables, columns, relationships)
        └── running.md           Operations doc — how to run migrations per environment
```

---

## What's in this directory

The tree above has a one-line purpose next to each file. A few notes that don't fit on one line:

- **`orm.py`** contains `User`, `RefreshToken`, `ModelVersion`, `Prediction`, `Feedback`, and `MonitoringMetric`.
- **`alembic/versions/`** holds migration files, numbered `0001`, `0002`, …. Never hand-edit an applied migration — create a new one instead.
- **`seeds/dev_seed.sql`** is idempotent (safe to re-run) but **never apply it to production** — it contains a known password hash.
- **`Dockerfile`** produces an image whose only job is running `alembic upgrade head`. Reused in CI, Portainer, and cloud one-shot jobs.

---

## Typical usage

**From the backend, reading/writing:**

```python
from db import User, Prediction
from db.session import get_session

def list_predictions_for_user(session, user_id):
    return session.query(Prediction).filter_by(user_id=user_id).all()
```

**From the CLI, running migrations:**

```bash
./scripts/db-migrate.sh upgrade head         # apply all pending migrations
./scripts/db-migrate.sh current              # show current revision
./scripts/db-migrate.sh revision --autogenerate -m "add foo"   # draft a new migration
```

The wrapper loads `.env` automatically and passes all arguments through to Alembic.

---

## Where to go for more

| If you want to know… | Read |
|---|---|
| What tables exist, what columns they have, and why | [`../docs/database/schema.md`](../docs/database/schema.md) |
| How to run Postgres + migrations in local / Docker / Portainer | [`../docs/database/running.md`](../docs/database/running.md) |
| What `DATABASE_URL` should look like in each environment | [`../.env.example`](../.env.example) |

---

## Conventions

- **One change per migration.** Multiple unrelated schema changes in one file make review and rollback messy.
- **Never hand-edit an applied migration.** Add a new one. An already-applied migration is part of the immutable history of the database.
- **Schema changes start in `schema.md`.** Then update `orm.py`, then generate/write the migration. Keeping the doc as the source-of-truth-by-convention prevents "the schema doc and the code disagree, which is right?" situations.
- **No hardcoded connection strings.** Everything goes through `DATABASE_URL`.