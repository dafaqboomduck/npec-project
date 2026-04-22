# Running the Database

**Status:** Sprint 2
**Related:** [`docs/database/schema.md`](./schema.md)

How to bring up and migrate the CV5 database in each target environment. Same code, same migration, same resulting schema in all three.

---

## 1. Quick reference

| Environment | Postgres runs on | How you migrate |
|---|---|---|
| **Local (bare metal)** | Your host OS | `./scripts/db-migrate.sh upgrade head` |
| **Docker Compose** | `db` service in compose network | `docker compose run --rm migrations` |
| **Portainer / on-premise** | Managed Postgres service | One-shot container from the `cv5-migrations` image, triggered manually or by a deploy step (see §4) |

Seeds are separate from migrations and are **local-dev only** — see §5.

---

## 2. Local (bare metal)

Use this when you have Postgres installed on your host (e.g. via Homebrew, `apt`, or the official installer).

**One-time setup:**

```bash
# Create role and database (adjust credentials as you like)
createuser --pwprompt cv5
createdb --owner cv5 cv5

# Configure the app
cp .env.example .env
# Edit .env so DATABASE_URL matches what you just created, e.g.:
#   DATABASE_URL=postgresql+psycopg://cv5:cv5@localhost:5432/cv5
```

**Run migrations:**

```bash
./scripts/db-migrate.sh upgrade head
```

**Verify:**

```bash
psql "$DATABASE_URL" -c "\dt"          # should list 6 tables + alembic_version
psql "$DATABASE_URL" -c "SELECT * FROM alembic_version;"   # should show 0001
```

---

## 3. Docker Compose

Use this when you want a self-contained stack with no Postgres on your host.

**Start Postgres:**

```bash
docker compose up -d db
```

The `db` service has a healthcheck (`pg_isready`); it can take ~10 seconds to go from `starting` to `healthy` on first boot because Postgres initialises its data directory.

**Run migrations (one-shot):**

```bash
docker compose run --rm migrations
```

`migrations` waits until `db` reports healthy, runs `alembic upgrade head`, and exits. Re-run the command as often as you like — Alembic skips migrations that are already applied (idempotent).

**Tear down:**

```bash
docker compose down            # stops containers, keeps the volume
docker compose down -v         # also wipes the Postgres data volume
```

**Connect to the containerised Postgres from your host** (e.g. to open a `psql` shell or point DataGrip at it):

```bash
psql -h localhost -U cv5 -d cv5     # uses the port mapping in compose
```

---

## 4. Portainer / on-premise / cloud

For production-ish environments, **migrations are a one-shot job, not a startup hook**. Running migrations on every app container startup is risky: if you have multiple replicas, they race for the migration lock; if a migration fails, the app can't start. Run migrations **once, explicitly, before** the new app version is rolled out.

**Recommended pattern:**

1. Build the migrations image from `db/Dockerfile` as part of CI and push to a registry:
   ```bash
   docker build -f db/Dockerfile -t registry.example.com/cv5-migrations:$(git rev-parse --short HEAD) .
   docker push registry.example.com/cv5-migrations:$(git rev-parse --short HEAD)
   ```

2. In Portainer, create a one-shot container from that image with `DATABASE_URL` set as an environment variable (point it at your production Postgres). The container's default `CMD` is `alembic -c db/alembic.ini upgrade head` — it will run and exit with code 0 on success.

3. Only after that succeeds do you deploy the new backend version.

If you need a rollback, start another one-shot container with an override command:

```bash
docker run --rm -e DATABASE_URL=... cv5-migrations:<tag> \
    alembic -c db/alembic.ini downgrade -1
```

---

## 5. Seed data (dev only)

`db/seeds/dev_seed.sql` contains dev-only fixtures: one admin user and a placeholder deployed model version. **Never apply it to production** — it contains a known password hash.

**Apply seeds locally (bare metal):**

```bash
psql "$DATABASE_URL" -f db/seeds/dev_seed.sql
```

**Apply seeds in Docker Compose:**

```bash
docker compose exec db psql -U cv5 -d cv5 -f /seeds/dev_seed.sql
```

(The seeds directory is bind-mounted to `/seeds` in the `db` service.)

Seeds are idempotent — safe to re-run. They use `ON CONFLICT DO NOTHING` on unique columns.

**Dev credentials** (from `dev_seed.sql`):
- Email: `admin@dev.local`
- Password: `devpassword`

---

## 6. Verifying portability

If you ever suspect two environments have drifted, confirm they produce the same schema:

```bash
# From each environment, dump the schema:
pg_dump "$DATABASE_URL" --schema-only --no-owner --no-privileges \
    | grep -v -E '^\\(restrict|unrestrict)' \
    > schema_<envname>.sql

# Then diff:
diff schema_local.sql schema_docker.sql
# Expected output: nothing (identical) or only ordering differences
```

The `grep -v` filter removes `pg_dump`'s per-session tokens, which are random and meaningless for comparison.

This was verified as part of Task 6.4: migrating from a fresh database produces a byte-identical schema dump (405 lines, 6 tables + `alembic_version`, all indexes, partial unique index on `model_versions.status`).

---

## 7. Troubleshooting

**`DATABASE_URL is not set`**
You either have no `.env` and no exported var, or your `.env` has the line commented out. Copy `.env.example` to `.env` and uncomment one of the `DATABASE_URL=...` lines.

**Migration fails with `permission denied to create extension "pgcrypto"`**
The database user needs `SUPERUSER` (or someone with superuser has to run `CREATE EXTENSION pgcrypto;` manually once, then the migration will skip it because we use `CREATE EXTENSION IF NOT EXISTS`).

**Migration fails with `role "cv5" does not exist`**
The Postgres role/user wasn't created. See §2 "One-time setup."

**`docker compose up` hangs on `migrations` waiting for `db`**
Wait ~10 seconds — the healthcheck polls every 5s with 10 retries. If it still hangs, `docker compose logs db` to see what Postgres is complaining about (usually a port clash or a corrupt data volume — `docker compose down -v` resets it).

**Schema dump diffs show real differences between environments**
Real differences (not `\restrict` tokens) mean one of the environments has been hand-modified outside of Alembic. Find the divergence, decide which side is correct, and create a new migration to bring them back in sync. Never edit an old migration file — that breaks reproducibility for anyone else.

**`psql: error: C:/Program Files/Git/seeds/dev_seed.sql: No such file or directory` (Windows + Git Bash)**
Git Bash / MSYS2 rewrites Unix-style paths like `/seeds/...` to Windows paths when passing them to Docker. Use a double leading slash to opt out:
```bash
docker compose exec db psql -U cv5 -d cv5 -f //seeds/dev_seed.sql
```
Or set `MSYS_NO_PATHCONV=1` for the command. PowerShell, CMD, and WSL don't have this problem.
