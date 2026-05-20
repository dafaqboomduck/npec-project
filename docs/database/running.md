# Running the database

How to start Postgres, apply migrations, and seed dev data in each environment. Operational companion to [`schema.md`](schema.md) (what the schema *is*) and [`../../db/README.md`](../../db/README.md) (what's in the `db/` directory).

---

## Prerequisites

- Docker Desktop running (for Docker / compose workflows)
- Python 3.10+ with the `db` package installable from `db/pyproject.toml` (for bare-metal workflows)
- A `.env` file at the repo root — copy from `.env.example` and edit if needed

The default `DATABASE_URL` in `.env.example` points at the Docker Compose Postgres on `db:5432`. If you're running Postgres on bare metal, change the host to `localhost`.

---

## Local development (Docker Compose) — recommended

The fastest path. Brings up Postgres + MinIO + the migrations container in one command.

```bash
# 1. Copy env template
cp .env.example .env

# 2. Start Postgres (and MinIO, pgadmin)
docker compose -f docker-compose.dev.yml up -d db

# 3. Run migrations against it
docker compose -f docker-compose.dev.yml run --rm migrations

# 4. (Optional) apply dev seed
docker compose -f docker-compose.dev.yml exec db \
    psql -U cv5 -d cv5 -f /seeds/dev_seed.sql
```

After step 3 you should see 9 tables in the `cv5` database:

```bash
docker compose -f docker-compose.dev.yml exec db psql -U cv5 -d cv5 -c '\dt'
```

Expected: `users`, `refresh_tokens`, `model_versions`, `predictions`, `feedback`, `monitoring_metrics`, `datasets`, `annotations`, `training_runs`, plus `alembic_version`.

### Resetting the database

```bash
docker compose -f docker-compose.dev.yml down -v   # delete the volume
docker compose -f docker-compose.dev.yml up -d db
docker compose -f docker-compose.dev.yml run --rm migrations
```

---

## Local development (bare metal)

For when you want Postgres running natively (e.g. for tighter feedback loops or debugging extensions).

```bash
# 1. Install Postgres 16 locally (brew, apt, installer — whatever)
# 2. Create a database
createdb cv5

# 3. Point DATABASE_URL at it in .env, e.g.
#    DATABASE_URL=postgresql+psycopg://you@localhost:5432/cv5

# 4. Install the db package
pip install -e ./db

# 5. Run migrations via the wrapper
./scripts/db-migrate.sh upgrade head

# 6. (Optional) apply dev seed
psql "$DATABASE_URL" -f db/seeds/dev_seed.sql
```

The wrapper loads `.env` automatically. If you prefer raw alembic:

```bash
export $(grep -v '^#' .env | xargs)
alembic -c db/alembic.ini upgrade head
```

---

## On-premise (Portainer)

The migrations image is the same one used locally; only the orchestration changes.

1. Push the `db/Dockerfile` image to a registry the Portainer host can pull from (GHCR, an internal registry, or build directly on the host).
2. Provision a Postgres 16 service in Portainer with a persistent volume.
3. Define a one-shot stack that runs the migrations image with `DATABASE_URL` pointed at the Postgres service. Set the restart policy to `no` — migrations should run once and exit.
4. Trigger the stack manually for each deploy, or wire it into a deploy hook.

Seed data is **not** applied automatically. If a dev seed is wanted on-prem (it usually isn't), run `psql` manually inside the Postgres container.

---

## Common alembic commands

All via the wrapper:

```bash
./scripts/db-migrate.sh upgrade head          # apply all pending migrations
./scripts/db-migrate.sh downgrade -1          # revert the last migration
./scripts/db-migrate.sh current               # show current revision
./scripts/db-migrate.sh history               # show full migration graph
./scripts/db-migrate.sh revision --autogenerate -m "add foo"   # draft a new migration
./scripts/db-migrate.sh stamp head            # mark DB as up-to-date without running migrations
```

`stamp head` is useful when an existing database already has the schema (e.g. restored from a backup) and just needs Alembic's bookkeeping table populated.

---

## Verifying portability (task 6.4)

The same migration should produce the same schema in every environment. Quick check:

```bash
# In each environment, after running migrations:
psql "$DATABASE_URL" -c "
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
"
```

Output should be identical across local, Docker, and Portainer.

For a stricter check:

```bash
pg_dump --schema-only --no-owner "$DATABASE_URL" > schema.local.sql
# compare against the Docker dump with diff
```

Small differences in `pg_dump` ordering or comments are expected across versions; constraint names should be stable because of the naming convention in `db/base.py`.

---

## Troubleshooting

**`DATABASE_URL is not set`**
The wrapper couldn't read `.env`. Check the file exists at the repo root and the variable isn't commented out.

**`could not translate host name "db" to address`**
You're running alembic outside Docker Compose, but `.env` still has the compose hostname. Change `db` to `localhost` in `DATABASE_URL`.

**`relation "alembic_version" does not exist` after manual changes**
Run `./scripts/db-migrate.sh stamp head` to teach Alembic the database is already at the latest revision.

**Pre-commit hook keeps rejecting the commit after `black` reformats files**
Re-`git add` the affected files after the first pass, then commit again. Black reformats in place; the unstaged reformatted version blocks the commit until it's re-added.

**Windows / Git Bash mangles `/seeds/dev_seed.sql` into a Windows path**
Use a double leading slash (`//seeds/dev_seed.sql`) or prefix the command with `MSYS_NO_PATHCONV=1`.
