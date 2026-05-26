# `lineage/` — module guide

Developer reference for the lineage module. For the system-level overview, see [`docs/lineage/index.md`](../docs/lineage/index.md).

## What lives here

| File | Purpose |
|------|---------|
| `context.py` | `LineageContext`, `DatasetRef`, `gather_lineage_context()` — collect provenance facts. No DB / Azure ML deps. |
| `registration.py` | `register_model_with_lineage()`, `register_baseline_model()`, `lineage_from_job()` |
| `inference_logger.py` | `InferenceLogger` — async DB-backed audit logger. |
| `traceability.py` | `trace()` and the CLI. Detects baseline models and reports honestly. |

## Quick usage

### Register a pipeline-trained model (strict path)

```python
from lineage import (
    DatasetRef,
    lineage_from_job,
    register_model_with_lineage,
)

datasets = (DatasetRef(role="train", name="cv5_hades_raw", version="1.1.0"),)
context = lineage_from_job(client, job_name, datasets=datasets, environment=env_label)

register_model_with_lineage(
    client,
    name="root-detection",
    path=f"azureml://jobs/{job_name}/outputs/model_output",
    description=f"Retrained — job {job_name}",
    datasets=datasets,
    context=context,
)
```

Fails with `LineageError` if no git commit is available.

### Register a baseline / external model (explicit path)

```python
from lineage import register_baseline_model

register_baseline_model(
    client,
    name="root-detection",
    path="./shipped_model/root.keras",
    description="Baseline shipped with v1.0",
    reason="shipped_with_app_v1",
)
```

Tags the model `baseline=true` plus `baseline_reason`. Tracing will report it as a baseline rather than a broken-lineage record.

The local DB equivalent is to set `baseline=True` on the `ModelVersion` row — `db/bootstrap.py` and `db/seeds/dev_seed.sql` both do this for their placeholder models.

### Log a prediction

```python
from lineage import InferenceLogger

logger = InferenceLogger(
    db,
    model_version_id=deployed_mv.id,
    model_name="root-detection",
    model_version_label=deployed_mv.version,
)

async with logger.measure(
    input_ref=InferenceLogger.hash_input(image_bytes),
) as rec:
    result = await run_pipeline()
    rec.set_output({"num_plants_detected": len(result.plants)})
    rec.set_prediction_id(prediction.id)
```

The audit row is added to the session but not committed. The caller's transaction commits both rows together.

### Trace a prediction

```bash
python -m lineage.traceability <request-id-or-prediction-uuid> \
    --database-url "postgresql+asyncpg://user:pw@host/db"
```

Or programmatically:

```python
from lineage import trace
result = await trace(db, "req-abc-123")
print(result.pretty())
```

## Required environment

| Variable | Set by | Used for |
|----------|--------|----------|
| `AZUREML_GIT_COMMIT` | Azure ML | `gather_lineage_context` |
| `AZUREML_RUN_ID` | Azure ML | Training job id |
| `MLFLOW_RUN_ID` | Azure ML | MLflow run id |
| `NPEC_DEPLOYMENT` | Container start | Tags audit rows: `local` / `portainer` / `azureml` |
| `DATABASE_URL` | Compose | The trace CLI |

Add to each compose file:

```yaml
# docker-compose.dev.yml         NPEC_DEPLOYMENT: local
# docker-compose.prod.yaml       NPEC_DEPLOYMENT: portainer
# (Azure ML compose, in progress) NPEC_DEPLOYMENT: azureml
```

## Design choices

- **One backend.** PostgreSQL via the existing `db/` schema, per-deployment.
- **Append-only audit.** `ON DELETE SET NULL` keeps the audit row alive past prediction deletion.
- **Strict lineage enforcement.** `register_model_with_lineage` refuses to register without `git_commit`.
- **Explicit baselines.** `register_baseline_model` + `baseline=True` column → tracing distinguishes "no lineage by design" from "lineage broken."
- **Same code, three deployments.** Deployment identity is just an env var.

## Tests

```bash
python tests/test_lineage_smoke.py
```

11 tests, in-memory SQLite, no Azure ML credentials needed.
