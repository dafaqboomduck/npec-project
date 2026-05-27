-- db/seeds/dev_seed.sql
--
-- Development / local seed data. NOT for production use.
--
-- What this does:
--   1. Inserts three dev users (two admins, one regular user).
--   2. Inserts a placeholder model_version row marked baseline=true.
--
-- Lineage note (migration 0010):
--   The model_versions row inserted here is a placeholder. It has no
--   git_commit, no training_job_name, and no dataset_versions because
--   no training pipeline produced it. We set baseline=true so the
--   lineage trace CLI reports it honestly as a baseline rather than as
--   a model with broken lineage. Once a real model is trained via
--   scripts/submit_training_job.py and registered through
--   register_model_with_lineage(), the deploy flow can promote that
--   real model and demote this placeholder.
--
-- Idempotency:
--   Every INSERT uses ON CONFLICT, so running this seed twice is safe.
--
-- How to apply (local dev):
--   docker compose exec db psql -U cv5 -d cv5db -f /seeds/dev_seed.sql
--
--   Or from bare metal:
--   psql "$DATABASE_URL" -f db/seeds/dev_seed.sql
--
-- Warning:
--   The password hashes below are for dev-only credentials. Never run
--   this against production.

BEGIN;

-- ---------------------------------------------------------------------------
-- Users
-- ---------------------------------------------------------------------------
INSERT INTO users (id, email, password_hash, role, is_active, created_at)
VALUES
    (gen_random_uuid(), 'dr.hrina@cv5.nl', '$2b$12$Dsj1l3KtdxYFUHtq1/6awetOfHejcIRxC4zvj45GXyi1uGGmwtl3C', 'admin', true, now()),
    (gen_random_uuid(), 'dr.sarik@cv5.nl', '$2b$12$fWIX03wGKKI.5K6oX68k2uO779tj3pTl/.nNic0GHzKJ6U2VvI9Dy', 'admin', true, now()),
    (gen_random_uuid(), 'intern@cv5.nl',   '$2b$12$HuJwxfFEUM3FBQiJ2Whwf.o7GYTnOpqOqk9PqPtTaT0JsHTcj6gy2', 'user',  true, now())
ON CONFLICT (email) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Placeholder model_version  (baseline, no upstream lineage)
-- ---------------------------------------------------------------------------
INSERT INTO model_versions (
    id, version, name, description, status,
    f1_score, precision, recall, inference_latency_ms,
    artifact_uri, deployed_at, baseline
)
VALUES (
    '00000000-0000-0000-0000-000000000000',
    'unet-npec-v0.0.0-placeholder',
    'NPEC U-Net Baseline',
    'Placeholder baseline shipped with the project. Pre-pipeline artifact — no upstream training run. See migration 0010 for the baseline flag semantics.',
    'deployed',
    0.8813, 0.0, 0.0, 31352,
    '/app/artifacts/models/v1/',
    now(),
    true
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
    deployed_at  = EXCLUDED.deployed_at,
    baseline     = EXCLUDED.baseline;

COMMIT;
-- ---------------------------------------------------------------------------
-- Verification
-- ---------------------------------------------------------------------------
--   SELECT email, role FROM users;
--   SELECT version, status, baseline, artifact_uri FROM model_versions;
--
-- Expected:
--   version                       | status   | baseline | artifact_uri
--   unet-npec-v0.0.0-placeholder  | deployed | t        | /app/artifacts/models/v1/
--
-- Tracing any prediction served by this model:
--   python -m lineage.traceability <prediction_id>
-- should report "BASELINE MODEL — no upstream training pipeline" in
-- the Model lineage section.
