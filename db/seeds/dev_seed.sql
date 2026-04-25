-- db/seeds/dev_seed.sql
--
-- Development / local seed data. NOT for production use.
--
-- What this does:
--   1. Inserts a dev admin user (email: admin@dev.local, password: devpassword)
--   2. Inserts a placeholder "deployed" model_version so POST /predict has
--      something to reference before a real model is registered.
--
-- Idempotency:
--   Every INSERT uses ON CONFLICT DO NOTHING, so running this seed twice is
--   safe and won't duplicate rows or fail. The unique constraints on
--   users.email and model_versions.version make this work.
--
-- How to apply (local dev):
--   docker compose exec db psql -U cv5 -d cv5 -f /seeds/dev_seed.sql
--
--   Or from bare metal:
--   psql "$DATABASE_URL" -f db/seeds/dev_seed.sql
--
-- Warning:
--   The password hash below is for the string "devpassword". Anyone with
--   access to this repo can log in as admin if this seed is applied to an
--   internet-reachable database. Never run this against production.

BEGIN;

-- ---------------------------------------------------------------------------
-- Users
-- ---------------------------------------------------------------------------
-- password: devpassword
-- hash: bcrypt, 12 rounds
INSERT INTO users (id, email, password_hash, role, is_active, created_at)
VALUES
    (gen_random_uuid(), 'dr.hrina@cv5.nl', '$2b$12$Dsj1l3KtdxYFUHtq1/6awetOfHejcIRxC4zvj45GXyi1uGGmwtl3C', 'admin', true, now()),
    (gen_random_uuid(), 'dr.sarik@cv5.nl', '$2b$12$fWIX03wGKKI.5K6oX68k2uO779tj3pTl/.nNic0GHzKJ6U2VvI9Dy', 'admin', true, now()),
    (gen_random_uuid(), 'intern@cv5.nl',  '$2b$12$HuJwxfFEUM3FBQiJ2Whwf.o7GYTnOpqOqk9PqPtTaT0JsHTcj6gy2', 'user',  true, now())
ON CONFLICT (email) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Placeholder deployed model_version
-- ---------------------------------------------------------------------------
-- Gives POST /predict a valid model_version_id to reference during local dev.
-- Replace with a real trained model via the training pipeline when available.
INSERT INTO model_versions (
    id, version, name, description, status,
    f1_score, precision, recall, inference_latency_ms,
    artifact_uri, deployed_at
)
VALUES (
    gen_random_uuid(),
    'unet-npec-v0.0.0-placeholder',
    'Placeholder U-Net',
    'Dev-only placeholder. Not a real trained model. Replace via training pipeline.',
    'deployed',
    NULL, NULL, NULL, NULL,
    NULL,
    NOW()
)
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ---------------------------------------------------------------------------
-- Verification
-- ---------------------------------------------------------------------------
-- After running, you should see one user and one model_version:
--   SELECT email, role FROM users;
--   SELECT version, status FROM model_versions;
