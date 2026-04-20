## API Endpoints
**Version:** 0.2  
**Author:** Erik Hrina (230395)

---
### Domain
- **Experiment** — a single Hades tray session. Scientists register one experiment per run.
- **Plate** — one Petri dish within an experiment. Contains 5 Arabidopsis seedlings.
- **Scan** — one daily photograph of a plate. A plate accumulates one scan per day as seeds germinate and roots grow.
- **Prediction** — what the model produces automatically when a scan is processed: segmentation masks, tip (x, y) locations, confidence scores.
- **Annotation** — a human-verified or human-corrected label for a scan. Stored as a .tiff mask. Used as ground truth for retraining. Distinct from prediction — the training pipeline consumes annotations, not predictions.
- **Image** — the raw scan file. Never embedded in JSON responses; returned as binary or a presigned URL via a dedicated endpoint.

---
### Overview
This document lists every endpoint exposed by the backend. It is the single source of truth for URI design and serves as the input to `api_spec.yaml`. Every endpoint is traceable to a specific UI element or data fetch in the frontend.

**Base URL (local):** `http://localhost:8000/api/v1`  
**Base URL (on-premise):** `pending`  
**Base URL (cloud):** `pending`

---
### Auth

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| POST | `/auth/token` | Obtain a JWT access token via username + password | No | Public |
| POST | `/auth/logout` | Invalidate the current access token | Yes | User |
| POST | `/auth/refresh` | Retrieves a new access token without requiring credentials | Yes | User |

---
### Health

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| GET | `/health` | Service liveness check - returns status and API version | No | Public |

---
### Home / Overview

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| GET | `/overview/status` | Returns total experiments, active model count, and data collected | Yes | User |
| GET | `/overview/activity` | Paginated recent activity feed | Yes | User |
| GET | `/overview/model-health` | Current accuracy, latency (ms), and drift status for the active model | Yes | User |
| GET | `/overview/environment` | GPU temp, memory usage, and system status | Yes | Admin |

> TODO: Confirm whether Experiments need their own endpoint group (see Open Questions).

---
### Dashboard

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| GET | `/dashboard/metrics` | Top-level KPIs | Yes | User |
| GET | `/dashboard/accuracy-history` | Time-series of validation accuracy across training epochs for charting | Yes | User |
| GET | `/dashboard/data-distribution` | Sample counts per species for the distribution bar chart | Yes | User |

---
### Library

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| GET | `/datasets` | List all datasets; supports `?species=` and `?growth_stage=` filters | Yes | User |
| GET | `/datasets/{dataset_id}` | Single dataset detail (name, created date, modified date, count) | Yes | User |
| GET | `/annotations` | Paginated list of all annotations; supports `?species=` and `?growth_stage=` filters | Yes | User |
| GET | `/annotations/{annotation_id}` | Single annotation detail (metadata, annotation status, species, age, treatment, image) | Yes | User |
| GET | `/models` | List all models | Yes | User |
| GET | `/models/{model_id}` | Single model detail (name, created date, parameters, metrics, status); filtered by auth token | Yes | User/Admin |
| POST | `/models/{model_id}/deploy` | Trigger deployment of a registered model | Yes | Admin |
| DELETE | `/models/{model_id}` | Delete a registered model | Yes | Admin |

---
### Inference

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| POST | `/predict` | Submit a root scan image for inference; returns prediction ID and initial result | Yes | User |
| GET | `/predictions/{prediction_id}` | Retrieve full result for a prediction (mask URL, tip locations, confidence scores, metadata) | Yes | User |
| POST | `/predictions/{prediction_id}/confirm` | Confirm and save a prediction result to the library | Yes | User |
| DELETE | `/predictions/{prediction_id}` | Discard an unsaved prediction result | Yes | User |
| GET | `/annotations/{annotation_id}` | Retrieve annotation metadata and .tiff mask URL for manual review | Yes | User |
| POST | `/annotations/{annotation_id}/confirm` | Submit an updated .tiff mask; returns the updated annotation object | Yes | User |
| DELETE | `/annotations/{annotation_id}` | Delete an annotation | Yes | User |

**Notes on `POST /predict`:**
- Accepts `multipart/form-data` with the image file
- Optional parameters: `confidence_threshold` (float, 0.50–0.99)
- Supported formats: `.png`, `.tiff`, `.tif`, `.jpg`, `.jpeg`
- Maximum file size: **50 MB** (files above this are rejected before decoding with HTTP 413)
- Minimum resolution: 512 × 512 px — maximum: 4096 × 4096 px (violations return HTTP 422)
- Multi-page TIFFs are rejected with HTTP 422
- JPEG is accepted but not recommended — compression artefacts may degrade segmentation quality

**Notes on `POST /annotations/{annotation_id}/confirm`:**
- Accepts `multipart/form-data` with a `.tiff` mask file
- Returns the updated annotation object (metadata + mask URL)
- The submitted mask becomes ground truth for retraining

**Response shape for `GET /predictions/{prediction_id}`** (from I/O spec v1.0.0):
```json
{
  "status": "ok",
  "metadata": {
    "model_version": "unet-npec-v1.2.0",
    "inference_timestamp": "2026-04-17T10:23:45Z",
    "processing_time_ms": 843,
    "input_resolution": [2775, 2048],
    "input_channels": 1,
    "normalisation_applied": true,
    "scale_factor_mm_per_px": 0.05405
  },
  "segmentation_mask_url": "s3://npec-outputs/{job_id}_mask.png",
  "plants": [
    {
      "plant_id": 1,
      "tip_x": 412,
      "tip_y": 1803,
      "confidence": 0.94,
      "root_length_px": 1643.2,
      "root_length_mm": 88.7
    }
  ],
  "overall_confidence": 0.91,
  "warnings": []
}
```

---
### Training

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| POST | `/training/start` | Trigger a training run with optional hyperparameter overrides | Yes | Admin |
| GET | `/training/runs` | List all training runs with status and metrics | Yes | Admin |
| GET | `/training/runs/{run_id}` | Detail for a single training run | Yes | Admin |
| DELETE | `/training/runs/{run_id}` | Stop a running training process | Yes | Admin |

---
### Feedback

| Method | URI | Description | Auth Required | Role |
|--------|-----|-------------|---------------|------|
| POST | `/feedback` | Submit a correction or quality signal on a prediction | Yes | User |

---
### Summary

| Count | Scope |
|-------|-------|
| 2 | Authentication |
| 1 | Health |
| 4 | Home |
| 3 | Dashboard |
| 8 | Library |
| 7 | Inference |
| 4 | Training |
| 1 | Feedback |
| **30** | **Total** |

---
### Versioning Strategy
All endpoints are prefixed with `/api/v1`. Breaking changes require a version update to `/api/v2`. Additive changes (new optional fields, new endpoints) do not require a version update.

---
### Open Questions

- [ ] **Experiments:** The domain model includes Experiments but no endpoint group exists yet.
- [ ] **Batch inference:** The Inference Lab UI mentions "batch process directories." `POST /predict` currently accepts a single file. Resolve before writing `api_spec.yaml`.
- [ ] **Automated scan ingestion:** How do daily Hades scans enter the system — does Hades push to the API, or does a pipeline job handle ingestion? If the latter, a `POST /scans` endpoint (Admin only) is needed.
- [ ] **JPEG rejection:** The I/O spec notes JPEG compression may degrade segmentation. Decide whether JPEG should be fully rejected (HTTP 415) or accepted with a warning in the response. Currently accepted with a note.
- [ ] **Normalisation parameters:** Training mean and std are pending (`[ask razvan]` in I/O spec §2.5). Must be filled in before Sprint 2 inference is implemented.