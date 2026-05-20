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
│   ├── orm.py                   The nine ORM models
│   ├── session.py               Engine + session factory (reads DATABASE_URL)
│   ├── alembic.ini              Alembic config
│   ├── alembic/
│   │   ├── env.py               Wires DATABASE_URL + Base.metadata into Alembic
│   │   ├── script.py.mako       Template for new migration files
│   │   └── versions/
│   │       └── 0001_initial_schema.py
│   ├── seeds/
│   │   └── dev_seed.sql         Dev-only seed (users + placeholder model_version)
│   └── Dockerfile               One-shot image for running migrations
│
├── docker-compose.dev.yml       Local Postgres + migrations services
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

## Canonical database name

The database is named **`cv5db`** in every environment — local Docker
Compose, on-premise Portainer, and cloud. There is no per-environment
variation.

- `POSTGRES_DB` defaults to `cv5db` in `docker-compose.dev.yml`.
- `.env.example` ships with `POSTGRES_DB=cv5db` and `DATABASE_URL=...:5432/cv5db`.
- The Portainer prod stack uses `cv5db`.

When connecting with `psql`, always use `-d cv5db`:

```bash
psql -U cv5 -d cv5db
```

> Historical note: an earlier compose default of `cv5` got out of sync
> with the Portainer stack (`cv5db`), which caused real confusion
> (`psql -d cv5` failing on prod with `database "cv5" does not exist`).
> The default has been changed to `cv5db` so all environments match. If
> you have an existing local volume from before this change, the old
> `cv5` database will still be inside it; drop the volume
> (`docker compose -f docker-compose.dev.yml down -v`) and re-run
> migrations to get a clean `cv5db`.

---

## What's in this directory

The tree above has a one-line purpose next to each file. A few notes that don't fit on one line:

- **`orm.py`** contains `User`, `RefreshToken`, `ModelVersion`, `Prediction`, `Feedback`, `Dataset`, `Annotation`, `MonitoringMetric`, and `TrainingRun`.
- **`alembic/versions/`** holds migration files, numbered `0001`, `0002`, …. Never hand-edit an applied migration — create a new one instead.
- **`seeds/dev_seed.sql`** is idempotent and dev-only — see the next section.
- **`Dockerfile`** produces an image whose only job is running `alembic upgrade head`. Reused in CI, Portainer, and cloud one-shot jobs.

---

## Scope of the dev seed

`db/seeds/dev_seed.sql` exists to give a fresh local or shared dev
database a baseline set of users and a placeholder `model_versions` row
to log in with and to reference during integration work. It is **not** a
substitute for the production admin and deploy flows.

- The placeholder `model_versions` row is inserted as
  `status = 'registered'` with `artifact_uri = NULL` and
  `deployed_at = NULL`. It is not a deployed model. Real models are
  registered and promoted to `deployed` by the deploy pipeline, as
  separate rows.
- The schema enforces "at most one deployed model at a time" via a
  partial unique index (`one_deployed_model` in `orm.py`). If the seed
  claimed `deployed`, the deploy flow couldn't promote a real model
  without first un-deploying the placeholder. Keeping the seed row in
  `registered` keeps that slot free.
- The health check requires `status = 'deployed' AND artifact_uri IS NOT NULL`.
  With the seed alone, the health check will correctly report no
  deployed model — that's the expected state until a real model is
  trained and deployed.
- The version string `unet-npec-v0.0.0-placeholder` is reserved for the
  seed row. Don't use it for anything else.
- The seed file contains a known password hash. **Never apply it to a
  production database.**

The seed is safe to re-run: users use `ON CONFLICT DO NOTHING` (so
local password changes aren't clobbered), and the placeholder
model_version uses `ON CONFLICT DO UPDATE` (so re-running the seed
forces the placeholder back to its intended registered state). The
upsert is safe specifically because the version string is a
placeholder — it can't collide with a real deploy. If you ever copy
this block for a real version, drop the `DO UPDATE`.

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