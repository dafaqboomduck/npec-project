# Lineage and audit logging

System-level overview of how the CV5 project tracks data lineage and audit logging across its three deployment targets. For developer-facing API documentation and usage examples, see the [code-adjacent README](../../lineage/README.md).

This document satisfies the documentation half of tickets **2291**, **2292**, and **2293**.

## What lineage means here

Every prediction made by the system can be traced back through the chain:

```
prediction  ->  model version  ->  training job  ->  dataset versions  ->  git commit
            |
            +->  request id, input hash, output, latency, deployment, timestamp
```

Two halves:

- **Audit half** (one row per inference call): the `inference_audit_log` table. Records what happened at request time — which model served the call, what input was given, what output came back, how long it took.
- **Lineage half** (one row per model version): on the `model_versions` table and on the matching Azure ML model registry entry. Records how the model came to exist — git commit, training job, environment, dataset versions.

The two halves are joined by `model_version_id`.

## Baseline models

Not every model has upstream training lineage, and that's fine — as long as it's explicit. A **baseline model** is one that was *not* produced by the project's training pipeline. Examples:

- The `.keras` file bundled with the application's first release.
- A third-party reference checkpoint.
- A model trained before this lineage system existed.

Baseline models are registered with `baseline=True` on `model_versions` (and `baseline=true` tag in the Azure ML registry). Lineage tracing detects this flag and reports the model as "BASELINE MODEL — no upstream training pipeline" instead of treating the missing lineage fields as broken records.

Two paths produce baseline models:

1. `db/bootstrap.py` — runs on first container startup, registers the bundled artifact at `BASELINE_MODEL_URI` if no deployed model exists yet.
2. `db/seeds/dev_seed.sql` — inserts a fixed-UUID placeholder for local dev so other seed data has something to FK against.

Any model produced by `scripts/submit_training_job.py` is registered through `register_model_with_lineage` and gets `baseline=false`, plus the full set of lineage tags. This is the strict path; it refuses to register without a git commit.

## The three deployments

Each runs the same containers built from the same docker-compose files, against the same database schema. They do not communicate.

| Deployment | Runs on | Database | Inference logs | Model registry |
|-----------|---------|----------|---------------|----------------|
| Local | Developer laptop, `docker-compose.dev.yml` | Postgres container, local | `inference_audit_log` in the local DB | Local DB `model_versions` table |
| Portainer | BUas Portainer server, `docker-compose.prod.yaml` | Postgres container co-deployed | `inference_audit_log` in the Portainer DB | Portainer DB `model_versions` table |
| Azure ML | `lambda-4` Arc-connected Kubernetes cluster | Postgres container on the cluster | `inference_audit_log` in the Azure ML DB | Same DB + Azure ML model registry tags |

Each deployment's audit log only knows about predictions made on *that* deployment. There is no cross-deployment trace.

## How a prediction gets logged

1. User uploads via `POST /predict`.
2. The router validates, hashes the image, calls `UNetInferenceService.run(...)`.
3. The service looks up the currently deployed `ModelVersion` row.
4. Constructs an `InferenceLogger` bound to the request's DB session.
5. Inference runs inside `logger.measure(...)`.
6. Both the `predictions` row and the `inference_audit_log` row are added to the same session.
7. The router commits the transaction.

If a prediction is deleted via `DELETE /predictions/{id}`, the audit row survives — `prediction_id` becomes NULL by `ON DELETE SET NULL`. Everything else is preserved.

## How a pipeline-trained model gets registered with lineage

1. Developer runs `python scripts/submit_training_job.py`.
2. Azure ML runs `scripts/train_root_detection.py` on `lambda-4`.
3. MLflow autolog captures hyperparameters, metrics, environment, git commit.
4. The script writes the `.keras` to its output directory.
5. The submitter calls `lineage_from_job(client, job_name, ...)`.
6. The submitter calls `register_model_with_lineage(client, ...)`.

If lineage cannot be established (no git commit), registration fails loudly.

## Tracing a prediction

```bash
python -m lineage.traceability <prediction-id-or-request-id> \
    --database-url "postgresql+asyncpg://..."
```

For a pipeline-trained model:

```
Audit entry req-xyz
  prediction id   e3a1b4...
  served by       root-detection:v3
  ...

Model lineage
  git commit      5a10a63df5fc76a772f72514994f2e7fb9079943
  training job    coral_wing_lgn61syzng
  datasets        {'train': 'cv5_hades_raw:1.1.0'}
```

For a baseline model:

```
Audit entry req-xyz
  prediction id   e3a1b4...
  served by       NPEC U-Net Baseline:unet-npec-v0.0.0-placeholder
  ...

Model lineage
  BASELINE MODEL — no upstream training pipeline.
  reason          (none recorded)
  artifact_uri    /app/artifacts/models/v1/
  Lineage fields below are expected to be empty for baselines.
```

## Migration ordering

The lineage migration is `0010_add_lineage_tracking.py`, applied after the dev branch's `0006-0009` (image hashes, tip photos, created_by columns).

## Access

| Audience | What they need | How they get it |
|---------|---------------|-----------------|
| Workspace admins | Full lineage on Azure ML | Studio + `az ml model show` |
| DB admins | Full audit + lineage in the DB | Direct SQL or `python -m lineage.traceability` |
| Developers | Audit + lineage during debugging | Same as DB admins |
| End users | Their own predictions only | Existing `GET /predictions` endpoints |

## Known gaps

- **Pre-existing Azure ML registry entries** (`shoot-detection:1`, `root-detection:1` and `:2`) have empty tags. They can be re-registered via `lineage_from_job` if their training jobs still exist, or marked as baselines via `register_baseline_model` with an honest reason.
- **Audit log retention is unbounded.** For the project's scope this is fine; for production add an archival job.

## Schema reference

### `inference_audit_log` (new in migration 0010)

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | Server default `gen_random_uuid()` |
| prediction_id | UUID FK | `predictions.id`, SET NULL on delete |
| model_version_id | UUID FK | `model_versions.id`, SET NULL on delete |
| model_name | varchar(128) | Denormalised |
| model_version_label | varchar(64) | Denormalised |
| request_id | varchar(64) | Indexed |
| input_ref | text | `sha256:...` (compatible with `predictions.input_hash`) |
| output_summary | jsonb | Compact summary |
| latency_ms | integer | |
| error | text | Exception class + message |
| endpoint | varchar(64) | Default `api` |
| deployment | varchar(32) | `local` / `portainer` / `azureml` / `unknown` |
| logged_at | timestamptz | Default `now()` |

### Lineage columns on `model_versions` (added in migration 0010)

| Column | Type |
|--------|------|
| git_commit | varchar(40) |
| git_branch | varchar(255) |
| git_repo | text |
| training_job_name | varchar(255) |
| pipeline_run_id | varchar(255) |
| mlflow_run_id | varchar(64) |
| dataset_versions | jsonb |
| environment | varchar(255) |
| **baseline** | **boolean, default false** |

### Lineage columns on `training_runs` (added in migration 0010)

| Column | Type |
|--------|------|
| azureml_job_name | varchar(255) |
| git_commit | varchar(40) |
| environment | varchar(255) |
| dataset_versions | jsonb |
