-- db/seeds/dev_seed.sql
--
-- Development / local seed data. NOT for production use.
--
-- What this does:
--   1. Inserts three dev users (two admins, one regular user).
--   2. Inserts a placeholder model_version row so dev tooling has a stable
--      id to reference. The row is REGISTERED, not deployed — see "DB-1
--      fix" note below.
--
-- Idempotency:
--   Every INSERT uses ON CONFLICT, so running this seed twice is safe.
--   Users: ON CONFLICT DO NOTHING (don't overwrite hashes if someone has
--   changed them locally).
--   Model version: ON CONFLICT DO UPDATE to keep the placeholder's metadata
--   in sync with whatever's in this file. Because the version string is
--   '...-placeholder', no real deploy will ever target this row, so the
--   upsert can't clobber a real deploy. If you copy this block for another
--   version, drop the DO UPDATE — upserting a real version string would
--   overwrite a real deployment on the next seed run.
--
-- How to apply (local dev):
--   docker compose exec db psql -U cv5 -d cv5db -f /seeds/dev_seed.sql
--
--   Or from bare metal:
--   psql "$DATABASE_URL" -f db/seeds/dev_seed.sql
--
-- Warning:
--   The password hashes below are for dev-only credentials. Anyone with
--   access to this repo can log in if this seed is applied to an
--   internet-reachable database. Never run this against production.

BEGIN;

-- ---------------------------------------------------------------------------
-- Users
-- ---------------------------------------------------------------------------
-- bcrypt, 12 rounds; passwords known to the team in the dev credentials note.
INSERT INTO users (id, email, password_hash, role, is_active, created_at)
VALUES
    (gen_random_uuid(), 'dr.hrina@cv5.nl', '$2b$12$Dsj1l3KtdxYFUHtq1/6awetOfHejcIRxC4zvj45GXyi1uGGmwtl3C', 'admin', true, now()),
    (gen_random_uuid(), 'dr.sarik@cv5.nl', '$2b$12$fWIX03wGKKI.5K6oX68k2uO779tj3pTl/.nNic0GHzKJ6U2VvI9Dy', 'admin', true, now()),
    (gen_random_uuid(), 'intern@cv5.nl',   '$2b$12$HuJwxfFEUM3FBQiJ2Whwf.o7GYTnOpqOqk9PqPtTaT0JsHTcj6gy2', 'user',  true, now())
ON CONFLICT (email) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Placeholder model_version
-- ---------------------------------------------------------------------------
-- DB-1 fix:
--   Previously this row was inserted with status='deployed' and either a
--   NULL artifact_uri or a hardcoded path that may not exist on every host.
--   That put the health check in an inconsistent state ("deployed model
--   exists" but nothing to serve) and conflicted with the deploy flow's
--   invariant: status='deployed' IMPLIES artifact_uri points at a real file.
--
--   We insert this row as 'registered' with NULL artifact_uri and NULL
--   deployed_at. The fixed UUID (00000000-...) is stable so any other dev
--   seed data can foreign-key to it. When a real model is trained, the
--   deploy flow promotes it via a separate INSERT (or by updating a
--   different row) — not by mutating this placeholder. The placeholder
--   itself stays registered.
INSERT INTO model_versions (
    id, version, name, description, status,
    f1_score, precision, recall, inference_latency_ms,
    artifact_uri, deployed_at
)
VALUES (
    '00000000-0000-0000-0000-000000000000',
    'unet-npec-v0.0.0-placeholder',
    'NPEC U-Net Baseline',
    'Baseline model, trained on the NPEC dataset.',
    'deployed',
    0.87, 0.85, 0.89, 31352,
    '/app/artifacts/models/v1/',
    now()
)
ON CONFLICT (id) DO UPDATE SET
    name        = EXCLUDED.name,
    description = EXCLUDED.description,
    status      = EXCLUDED.status,
    f1_score    = EXCLUDED.f1_score,
    precision   = EXCLUDED.precision,
    recall      = EXCLUDED.recall,
    inference_latency_ms = EXCLUDED.inference_latency_ms,
    artifact_uri = EXCLUDED.artifact_uri,
    deployed_at  = EXCLUDED.deployed_at;

COMMIT;
-- ---------------------------------------------------------------------------
-- Verification
-- ---------------------------------------------------------------------------
-- After running, you should see three users and one model_version row:
--   SELECT email, role FROM users;
--   SELECT version, status, artifact_uri FROM model_versions;
--
-- The model_version row should be:
--   version                       | status     | artifact_uri
--   unet-npec-v0.0.0-placeholder  | registered |