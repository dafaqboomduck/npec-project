# Database Schema

**Version:** 0.4 (Sprint 2)
**Status:** Draft
**Related docs:** `docs/api_endpoints.md` · `docs/auth_schema.md` · `docs/IO_specifications.md`

---

## 1. Overview

This document is the authoritative schema for the CV5 PostgreSQL database for Sprint 2. It covers the nine tables needed to back the API at this stage:

- `users` — authenticated accounts
- `refresh_tokens` — server-side rotating refresh tokens (required by `auth_schema.md` §2.2)
- `model_versions` — registry of trained U-Net versions
- `predictions` — one row per `POST /predict` inference result
- `feedback` — user thumbs-up/down on predictions
- `monitoring_metrics` — time-series metrics for drift detection and dashboards
- `datasets` — named collections of root-scan images
- `annotations` — human-reviewed segmentation masks (prediction-derived or upload-created)
- `training_runs` — one row per training job, with hyperparameters and outcome

Tables for experiments, plates, and scans are mentioned in the API spec but are **deferred to a later sprint**. They are not created by these migrations.

### Conventions

- Primary keys are `UUID` (generated server-side with `gen_random_uuid()` from the `pgcrypto` extension) unless otherwise noted.
- All timestamps are `TIMESTAMPTZ` and default to `NOW()` at UTC.
- Soft deletes are not used in Sprint 2 — `DELETE` is hard.
- All foreign keys are indexed.
- Enum-like fields use `VARCHAR` + `CHECK` constraint rather than native `ENUM` types, because native Postgres enums are painful to migrate.

---

## 2. Tables

### 2.1 `users`

Backs `POST /auth/token` and ownership checks throughout the API.

| Column          | Type            | Constraints                                  | Notes |
|-----------------|-----------------|----------------------------------------------|-------|
| `id`            | `UUID`          | PK, default `gen_random_uuid()`              | Matches the JWT `sub` claim |
| `email`         | `VARCHAR(254)`  | UNIQUE, NOT NULL                             | RFC 5321 max length |
| `password_hash` | `VARCHAR(255)`  | NOT NULL                                     | bcrypt / argon2 hash — never plaintext |
| `role`          | `VARCHAR(16)`   | NOT NULL, CHECK `role IN ('user','admin')`   | From `auth_schema.md` §5 |
| `is_active`     | `BOOLEAN`       | NOT NULL, default `TRUE`                     | Soft-disable without deleting |
| `created_at`    | `TIMESTAMPTZ`   | NOT NULL, default `NOW()`                    | |
| `updated_at`    | `TIMESTAMPTZ`   | NOT NULL, default `NOW()`                    | Updated by app or trigger |

**Indexes:** unique index on `email` (implicit from UNIQUE constraint).

---

### 2.2 `refresh_tokens`

Server-side store for rotating refresh tokens. Required because `auth_schema.md` §2.2 specifies refresh tokens are opaque, stored server-side, rotated on every use, and must be invalidatable on logout.

| Column         | Type           | Constraints                                               | Notes |
|----------------|----------------|-----------------------------------------------------------|-------|
| `id`           | `UUID`         | PK, default `gen_random_uuid()`                           | |
| `user_id`      | `UUID`         | FK → `users.id` ON DELETE CASCADE, NOT NULL               | |
| `token_hash`   | `VARCHAR(255)` | UNIQUE, NOT NULL                                          | SHA-256 of the opaque token — we never store the token itself |
| `expires_at`   | `TIMESTAMPTZ`  | NOT NULL                                                  | 24h after issue, per spec |
| `revoked_at`   | `TIMESTAMPTZ`  | NULL                                                      | Set on logout or rotation |
| `replaced_by`  | `UUID`         | FK → `refresh_tokens.id`, NULL                            | Audit trail for rotation |
| `created_at`   | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                 | |

**Indexes:** `user_id`, `token_hash` (unique), `expires_at` (for cleanup jobs).

**Behaviour:** a token is valid iff `revoked_at IS NULL AND expires_at > NOW()`. Rotation sets the old row's `revoked_at` and `replaced_by`.

---

### 2.3 `model_versions`

Registry of trained U-Net versions. Every prediction references exactly one model version.

| Column               | Type            | Constraints                                                      | Notes |
|----------------------|-----------------|------------------------------------------------------------------|-------|
| `id`                 | `UUID`          | PK, default `gen_random_uuid()`                                  | |
| `version`            | `VARCHAR(64)`   | UNIQUE, NOT NULL                                                 | e.g. `unet-npec-v1.2.0` (IO_specifications §3.5) |
| `name`               | `VARCHAR(128)`  | NOT NULL                                                         | Human-readable |
| `description`        | `TEXT`          | NULL                                                             | |
| `status`             | `VARCHAR(16)`   | NOT NULL, CHECK `status IN ('deployed','registered','archived')` | From `api_spec.yaml` `ModelSummary` |
| `f1_score`           | `FLOAT`         | NULL, CHECK `f1_score BETWEEN 0 AND 1`                           | |
| `precision`          | `FLOAT`         | NULL, CHECK `precision BETWEEN 0 AND 1`                          | |
| `recall`             | `FLOAT`         | NULL, CHECK `recall BETWEEN 0 AND 1`                             | |
| `inference_latency_ms` | `INTEGER`     | NULL                                                             | |
| `artifact_uri`       | `TEXT`          | NULL                                                             | Path/URI to weights file (S3, blob, local) |
| `created_at`         | `TIMESTAMPTZ`   | NOT NULL, default `NOW()`                                        | |
| `deployed_at`        | `TIMESTAMPTZ`   | NULL                                                             | Set when status transitions to `deployed` |

**Indexes:** unique on `version`, index on `status` (for "find the deployed model" queries).

**Invariant:** at most one row has `status = 'deployed'` at a time. Enforced with a **partial unique index**:
`CREATE UNIQUE INDEX one_deployed_model ON model_versions ((status)) WHERE status = 'deployed';`

---

### 2.4 `predictions`

One row per `POST /predict` call. Matches the `PredictionResponse` envelope in `api_spec.yaml`.

| Column                    | Type           | Constraints                                                  | Notes |
|---------------------------|----------------|--------------------------------------------------------------|-------|
| `id`                      | `UUID`         | PK, default `gen_random_uuid()`                              | This is the `prediction_id` returned to the API |
| `user_id`                 | `UUID`         | FK → `users.id` ON DELETE RESTRICT, NOT NULL                 | Ownership — enforced per `auth_schema.md` §5 |
| `model_version_id`        | `UUID`         | FK → `model_versions.id` ON DELETE RESTRICT, NOT NULL        | Reproducibility |
| `status`                  | `VARCHAR(16)`  | NOT NULL, CHECK `status IN ('ok','error')`                   | From `PredictionResponse.status` |
| `input_filename`          | `VARCHAR(255)` | NULL                                                         | Original uploaded filename, for display only (e.g. `plate_03_day07.png`) |
| `input_uri`               | `TEXT`         | NULL                                                         | Pointer to the persisted input image (e.g. `s3://npec-inputs/{id}.png`, local path). NULL if input was not persisted. Needed for audit and retraining — see §2.4.2 |
| `input_width_px`          | `INTEGER`      | NULL                                                         | `metadata.input_resolution[0]` |
| `input_height_px`         | `INTEGER`      | NULL                                                         | `metadata.input_resolution[1]` |
| `input_channels`          | `SMALLINT`     | NULL                                                         | Usually 1 after conversion |
| `normalisation_applied`   | `BOOLEAN`      | NOT NULL, default `FALSE`                                    | |
| `scale_factor_mm_per_px`  | `FLOAT`        | NULL                                                         | |
| `processing_time_ms`      | `INTEGER`      | NULL                                                         | |
| `segmentation_mask_uri`   | `TEXT`         | NULL                                                         | Presigned URL or local path |
| `tip_photo_uri`           | `TEXT`         | NULL                                                         | Optional rendered overlay showing detected tip positions on the input image |
| `plants`                  | `JSONB`        | NOT NULL, default `'[]'::jsonb`                              | Array of `{plant_id, tip_x, tip_y, confidence, root_length_px, root_length_mm}` — see §2.4.1 |
| `warnings`                | `JSONB`        | NOT NULL, default `'[]'::jsonb`                              | Array of warning code strings |
| `confirmed_at`            | `TIMESTAMPTZ`  | NULL                                                         | Set by `POST /predictions/{id}/confirm` |
| `inference_timestamp`     | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                    | `metadata.inference_timestamp` |
| `created_at`              | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                    | DB insert time |

#### 2.4.1 Why `plants` is JSONB, not a separate `plant_detections` table

Per-plant detections are always read and written together with their parent prediction — there are no API endpoints that query plants independently (no "find all detections with confidence > X across predictions"). Normalising to a separate table would add joins on every read without unlocking any query. JSONB keeps the read/write path simple. If future requirements need per-plant querying, we can add a derived table in a later migration.

#### 2.4.2 Why `input_uri` is nullable

The product has not yet decided whether every prediction's input image is persisted server-side (the I/O spec and API spec are silent on this). We store a URI when it exists so audit and retraining workflows have a pointer, but keep the column nullable to avoid a painful backfill later. If persistence becomes a hard requirement, tightening to `NOT NULL` is a single-line migration because the column will already be populated.

`input_filename` is kept separately because it's a human label (shown in the UI next to the prediction) — URIs are typically opaque or ugly.

**Indexes:** `user_id`, `model_version_id`, `(user_id, created_at DESC)` for listing a user's recent predictions.

---

### 2.5 `feedback`

Thumbs-up/down from `POST /feedback`.

| Column           | Type           | Constraints                                                   | Notes |
|------------------|----------------|---------------------------------------------------------------|-------|
| `id`             | `UUID`         | PK, default `gen_random_uuid()`                               | |
| `prediction_id`  | `UUID`         | FK → `predictions.id` ON DELETE CASCADE, NOT NULL             | |
| `user_id`        | `UUID`         | FK → `users.id` ON DELETE RESTRICT, NOT NULL                  | Who gave the feedback |
| `signal`         | `VARCHAR(16)`  | NOT NULL, CHECK `signal IN ('correct','incorrect')`           | From `FeedbackRequest.signal` |
| `comment`        | `VARCHAR(500)` | NULL                                                          | Matches `maxLength: 500` in spec |
| `created_at`     | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                     | |

**Indexes:** `prediction_id`, `user_id`.

**Note:** multiple feedback rows per `(user_id, prediction_id)` are allowed — users can change their mind or update a comment, and each submission is recorded as a new row. No unique constraint on the pair. The most recent row (by `created_at`) represents the user's current opinion.

---

### 2.6 `monitoring_metrics`

Time-series store for dashboard KPIs, drift detection, and operational metrics. Generic key-value shape so we can add new metric types without schema changes.

| Column           | Type            | Constraints                                         | Notes |
|------------------|-----------------|-----------------------------------------------------|-------|
| `id`             | `BIGSERIAL`     | PK                                                  | Serial, not UUID — high insert rate, ordered reads |
| `metric_name`    | `VARCHAR(64)`   | NOT NULL                                            | e.g. `prediction_latency_ms`, `drift_score`, `f1_validation` |
| `metric_value`   | `DOUBLE PRECISION` | NOT NULL                                         | |
| `model_version_id` | `UUID`        | FK → `model_versions.id` ON DELETE SET NULL, NULL   | NULL for system-wide metrics |
| `labels`         | `JSONB`         | NOT NULL, default `'{}'::jsonb`                     | Prometheus-style labels e.g. `{"endpoint":"/predict"}` |
| `recorded_at`    | `TIMESTAMPTZ`   | NOT NULL, default `NOW()`                           | |

**Indexes:** `(metric_name, recorded_at DESC)` for time-series reads, `model_version_id`.

**Why not one table per metric?** We don't know in advance which metrics we'll need, and adding a column per new metric is expensive. A generic `(name, value, labels, time)` shape is the standard monitoring pattern (Prometheus, InfluxDB, etc.).

---

### 2.7 `datasets`

Named collections of root-scan images. Backs `GET /datasets` and `GET /datasets/{id}`.

| Column          | Type           | Constraints                                                        | Notes |
|-----------------|----------------|--------------------------------------------------------------------|-------|
| `id`            | `UUID`         | PK, default `gen_random_uuid()`                                    | |
| `name`          | `VARCHAR(128)` | NOT NULL                                                           | Human-readable label |
| `description`   | `TEXT`         | NULL                                                               | |
| `species`       | `VARCHAR(128)` | NULL                                                               | Filter target for `GET /datasets?species=` |
| `growth_stage`  | `VARCHAR(32)`  | NULL, CHECK `growth_stage IS NULL OR growth_stage IN ('seedling','mature')` | Filter target for `GET /datasets?growth_stage=` |
| `count`         | `INTEGER`      | NOT NULL, default `0`                                              | Denormalised — number of annotations in the dataset. Updated by the application layer |
| `created_at`    | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                          | |
| `modified_at`   | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                          | Updated when contents change |

**Indexes:** `species` (for the species filter).

#### 2.7.1 Why `count` is denormalised

`GET /datasets` returns dataset summaries with a count of items. A live `SELECT COUNT(*)` per row in the list response is fine at our scale but adds an N+1 join. Storing the count and updating it on annotation insert/delete is cheap and removes the join. If consistency drifts (e.g. a crash between annotation insert and count increment), a periodic reconciliation job can resync.

---

### 2.8 `annotations`

Human-reviewed segmentation masks. Annotations come from two sources:

- **Prediction-derived** — created by correcting a model output via `POST /annotations/{id}/confirm`. `prediction_id` references the source prediction; the annotated input image is reachable through that prediction.
- **Upload-created** — created by uploading a mask and its input image directly, with no backing prediction. `prediction_id` is NULL and `image_uri` carries the reference to the annotated image.

| Column          | Type           | Constraints                                                                                  | Notes |
|-----------------|----------------|----------------------------------------------------------------------------------------------|-------|
| `id`            | `UUID`         | PK, default `gen_random_uuid()`                                                              | |
| `prediction_id` | `UUID`         | FK → `predictions.id` ON DELETE CASCADE, NULL                                                | NULL for upload-created annotations — see introduction above |
| `dataset_id`    | `UUID`         | FK → `datasets.id` ON DELETE SET NULL, NULL                                                  | Optional — an annotation may not be in a dataset yet |
| `status`        | `VARCHAR(32)`  | NOT NULL, default `'pending'`, CHECK `status IN ('pending','annotated','review_required')`   | From `AnnotationResponse.status` |
| `mask_uri`      | `TEXT`         | NULL                                                                                         | Presigned URL or local path to the `.tiff` mask |
| `image_uri`     | `TEXT`         | NULL                                                                                         | URI of the annotated input image. Required in practice for upload-created annotations; prediction-derived annotations leave this NULL and resolve the image via `prediction_id` |
| `species`       | `VARCHAR(128)` | NULL                                                                                         | |
| `age_days`      | `INTEGER`      | NULL                                                                                         | |
| `treatment`     | `VARCHAR(128)` | NULL                                                                                         | |
| `growth_stage`  | `VARCHAR(32)`  | NULL, CHECK `growth_stage IS NULL OR growth_stage IN ('seedling','mature')`                  | |
| `created_at`    | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                                                    | |
| `updated_at`    | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                                                    | |

**Indexes:** `prediction_id`, `dataset_id`, `status`.

#### 2.8.1 Why no XOR constraint between `prediction_id` and `image_uri`

A CHECK enforcing "exactly one of `prediction_id` or `image_uri` is set" would be semantically tighter, but it would also reject legitimate intermediate states (e.g. an upload row inserted before its image_uri is populated). For Sprint 2 we leave the constraint out and let the application layer enforce the invariant; we can add it later if the ambiguity causes real bugs.

---

### 2.9 `training_runs`

One row per training job. Backs `GET /training/runs` and `GET /training/runs/{id}`.

| Column              | Type           | Constraints                                                                            | Notes |
|---------------------|----------------|----------------------------------------------------------------------------------------|-------|
| `id`                | `UUID`         | PK, default `gen_random_uuid()`                                                        | |
| `status`            | `VARCHAR(16)`  | NOT NULL, CHECK `status IN ('running','completed','failed','stopped')`                 | From `TrainingRunSummary.status` |
| `started_at`        | `TIMESTAMPTZ`  | NOT NULL, default `NOW()`                                                              | |
| `completed_at`      | `TIMESTAMPTZ`  | NULL                                                                                   | Set when status leaves `running` |
| `hyperparameters`   | `JSONB`        | NULL                                                                                   | Whatever subset of `TrainingStartRequest` was passed |
| `final_f1`          | `FLOAT`        | NULL                                                                                   | Populated on `completed` |
| `model_version_id`  | `UUID`         | FK → `model_versions.id` ON DELETE SET NULL, NULL                                      | The model registered from this run, once registration happens |

**Indexes:** `status`, `model_version_id`, `started_at`.

**Lifecycle:** rows are inserted with `status='running'` when a job starts. The training subprocess updates `status` to `completed` (with `final_f1` and `model_version_id` populated) or `failed` on exit. `DELETE /training/runs/{id}` sets status to `stopped` — it does not delete the row, so the audit trail is preserved.

---

## 3. Relationships (ER summary)

```
users ──1:N──> refresh_tokens
users ──1:N──> predictions
users ──1:N──> feedback

model_versions ──1:N──> predictions
model_versions ──1:N──> monitoring_metrics  (nullable FK)
model_versions ──1:N──> training_runs       (nullable FK)

predictions ──1:N──> feedback
predictions ──1:N──> annotations            (nullable FK — upload-created annotations have no prediction)

datasets ──1:N──> annotations               (nullable FK — annotation may not yet belong to a dataset)
```

### 3.1 Cascade rules

- Deleting a user cascades to their `refresh_tokens` (auth cleanup) but **restricts** if they have predictions or feedback (we want audit trail — deactivate via `is_active` instead).
- Deleting a prediction cascades to its feedback and to any prediction-derived annotations.
- Deleting a model version is restricted if predictions reference it (archive instead). For `monitoring_metrics` and `training_runs` the FK is `SET NULL` — metrics and runs survive their model.
- Deleting a dataset sets `dataset_id = NULL` on its annotations (annotations outlive their dataset assignment).

---

## 4. Decisions & Open Questions

**Decided:**
- ✅ **Feedback uniqueness:** multiple feedback rows per `(user_id, prediction_id)` are allowed. Users can update their opinion; each submission is a new row. See §2.5.
- ✅ **Monitoring retention:** no TTL or partitioning in Sprint 2. `monitoring_metrics` grows unboundedly for now; retention strategy will be revisited when table size becomes a problem.
- ✅ **Annotation source duality:** `annotations.prediction_id` is nullable so upload-created annotations (no backing prediction) and prediction-derived annotations share one table. `image_uri` carries the image reference when `prediction_id` is NULL. See §2.8.
- ✅ **Dataset count denormalisation:** `datasets.count` is maintained by the application layer rather than computed live. See §2.7.1.
- ✅ **Training run lifecycle:** stopped runs are kept (status='stopped') rather than deleted, preserving the audit trail.

**Open:**
- [ ] **Deferred tables:** `experiments`, `plates`, `scans` are in the API spec but out of scope for these migrations. They will need their own schema doc update in a future task.
- [ ] **XOR constraint on annotations:** whether to enforce "exactly one of `prediction_id` or `image_uri` set" at the DB layer. See §2.8.1.

---

## 5. Version History

| Version | Date       | Author | Changes |
|---------|------------|--------|---------|
| 0.1     | 2026-04-22 | —      | Initial draft — Sprint 2, six tables |
| 0.2     | 2026-04-22 | —      | Added nullable `predictions.input_uri` for persisted input pointer |
| 0.3     | 2026-04-26 | —      | Added Mermaid ER diagram (subsection 3.1) |
| 0.4     | 2026-05-18 | —      | Added `datasets`, `annotations`, `training_runs`; added `predictions.tip_photo_uri`; made `annotations.prediction_id` nullable and added `annotations.image_uri` for upload-created annotations |