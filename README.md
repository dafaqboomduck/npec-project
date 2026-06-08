# NPEC Root Inoculation & Computer Vision Platform

An automated system for plant root detection, segmentation, and morphological measurement. This platform integrates deep learning (TensorFlow/Keras U-Net) with classical image processing to perform precise root phenotyping, co-deployed with an OT-2 robotics control layer for automated lab experiments.

This document serves as the central documentation index for the repository, detailing how the implementation satisfies the project requirements and Intended Learning Outcomes (ILOs).

---

## Table of Contents
1. [System Architecture & Data Pipeline (ILO 4.4)](#1-system-architecture--data-pipeline-ilo-44)
2. [Financial Cost & Resource Evaluation (ILO 4.4)](#2-financial-cost--resource-evaluation-ilo-44)
3. [Model Training & Experiment Tracking (ILO 8.9)](#3-model-training--experiment-tracking-ilo-89)
4. [Data Solutions & Pipeline Scheduling (ILO 9.4)](#4-data-solutions--pipeline-scheduling-ilo-94)
5. [Inference Deployment & CI/CD (ILO 9.5)](#5-inference-deployment--cicd-ilo-95)
6. [Observability, Traceability & Frontend (ILO 10.3)](#6-observability-traceability--frontend-ilo-103)
7. [Code Quality, Linting & Testing (ILO 9.3)](#7-code-quality-linting--testing-ilo-93)
8. [Installation & Local Setup Guide](#8-installation--local-setup-guide)
9. [CLI & API Usage Examples](#9-cli--api-usage-examples)
10. [Packaging & Publishing Wheels](#10-packaging--publishing-wheels)

---

## 1. System Architecture & Data Pipeline (ILO 4.4)

### Proposed ML Application Architecture
The platform is designed as a modular, decoupled set of services to separate concerns, enable independent deployment, and minimize runtime dependencies (e.g., preventing CLI calls from loading heavy TensorFlow models unless required).

```
                      +-------------------+
                      |   React Frontend  |
                      |   (Vite + TS)     |
                      +---------+---------+
                                | REST / WebSockets
                                v
                      +-------------------+
                      |    FastAPI API    | <---+
                      |   (src/api/)      |     |
                      +----+---------+----+     |
      Read/Write           |         |          | Lineage Trace
      Metadata             |         |          |
                           v         v          |
  +--------------+    +----+----+  +-+----------+----+
  |  PostgreSQL  |<---| SQLAlchemy |  |   Lineage Core   |
  |   (cv5db)    |    |   ORM   |  |   (lineage/)    |
  +--------------+    +---------+  +-----------------+
                                            |
                                            | Managed Models
                                            v
                               +------------+------------+
                               |     Azure ML Workspace  |
                               |  - Model Registry       |
                               |  - Kubernetes Endpoint  |
                               +-------------------------+
```

*   **npec_cv** (`src/npec_cv/`): The core computer vision package. It wraps the U-Net segmentation, image pre-processing, post-processing, and morphological measurement calculations.
*   **FastAPI API** (`src/api/`): The serving layer that exposes REST endpoints for health check, predictions, and training history, running background workers for image validation and queueing.
*   **PostgreSQL Database (`cv5db`)**: Persists predictions, user accounts, model versions, audit logs, and training run history. No raw image data is stored here; it is queried by SQLAlchemy and managed via Alembic migrations.
*   **lineage** (`lineage/`): Dedicated package handling inference audit logging (`inference_audit_log` table) and tracking strict model provenance (Git commits, datasets, training run IDs).

### Data Pipeline Flow
The computer vision pipeline processes large Petri dish scans (up to 5120 × 4096 px) through the following stages:

```
[ Raw Image ] 
      │
      ▼
[ Dish Cropping ]  ──► (Extract Petri dish boundary, falls back to full if detection fails)
      │
      ▼
[ Patch division ] ──► (Divide image into overlapping 512×512 patches via patchify)
      │
      ▼
[ Batch Inference] ──► (Run patches through U-Net model on CPU/GPU)
      │
      ▼
[ Mask Stitching ] ──► (Reassemble patch probabilities using overlap-average blending)
      │
      ▼
[ Post-Processing] ──► (Binarise, skeletonise, and extract NetworkX topology graph)
      │
      ▼
[ Morphological ]  ──► (Calculate total root length, lateral root counts, and tip coordinates)
```

---

## 2. Financial Cost & Resource Evaluation (ILO 4.4)

A core requirement is to balance budget constraints with system performance. 

### Cost-Effective Inference Deployment
Azure ML standard online deployments utilize **Managed Online Endpoints**, running on dedicated Azure-managed VMs (e.g., `Standard_DS3_v2`) billed 24/7. This would incur a fixed cost of **~€150/month**.
*   **Our Solution**: We employ a **`KubernetesOnlineEndpoint`** deployed on the local lab server **`lambda-0`** (Arc-connected Kubernetes cluster). By hosting the scoring container on local on-premise compute attached to the Azure ML workspace, we achieve **zero incremental Azure compute costs** for inference.

### Resource Allocation Strategy
*   **Training (GPU-Bound)**: Done on Azure ML compute instances utilizing temporary GPU resources. Command jobs are scheduled and run only when retraining is triggered, minimizing GPU runtime costs.
*   **Inference (CPU-Bound)**: The deployed Kubernetes service uses a CPU-bound instance type (`defaultinstancetype`) on `lambda-0`. TensorFlow runs CPU-based inference taking ~10-30s per image, optimized via overlapping patches.
*   **Concurrency Control**: Concurrency per instance is pinned to 1. Requests queue for up to 60 seconds. This prevents CPU thread thrashing and memory exhaustion under load.

---

## 3. Model Training & Experiment Tracking (ILO 8.9)

### Training Pipeline
Cloud training is orchestrated via `scripts/submit_training_job.py`. It packages the repository, provisions the training environment `root-detection-train` (configured in `environments/azure-train-env.yml`), and submits a Command job to the GPU cluster on `lambda-0` running `scripts/train_root_detection.py`.

### Metric Logging & Experiment Tracking
The training script utilizes **MLflow autologging**. During training, the following are automatically pushed to the Azure ML Studio:
*   **Training/Validation Metrics**: Loss curves, learning rates, and validation F1 score.
*   **Artifacts**: Saved checkpoints, model configurations, and sample test predictions.
*   **Environment Metadata**: Python dependencies, environment variables, and the specific Git commit hash from which training was executed.

### Automated Hyperparameter Tuning
In the weekly retraining pipeline, we configure an automated **hyperparameter sweep** using grid search (`Choice([8, 16, 32])` for batch size) to maximize the `best_val_f1` metric.

### Conditional Model Registration
To protect production deployments from degradation, we implement a strict registration check in `scripts/register_model_if_better.py`:
1. The trained model is evaluated on the test dataset `azureml:cv5_test:1.0.0` (independent of the training/validation sets).
2. The `mean_f1` score is extracted.
3. The model is registered in the Azure ML Registry **only if**:
   *   The `mean_f1 >= F1_THRESHOLD (0.6)`.
   *   The `mean_f1` exceeds the highest F1 score of currently registered versions.

---

## 4. Data Solutions & Pipeline Scheduling (ILO 9.4)

### Cloud Data Storage & Versioning
Data assets are version-controlled and stored in the Azure ML Datastore. Three primary data assets are registered:
*   `azureml:cv5_train:1.0.0` - Root scans for model training.
*   `azureml:cv5_val:1.0.0` - Scans for model validation during training.
*   `azureml:cv5_test:1.0.0` - Scans for testing and final threshold validation.

Data operations (upload, registration, and version updates) are performed programmatically via `scripts/upload_data.py`.

### Scheduled Pipeline Orchestration
The retraining pipeline (`scripts/submit_pipeline.py`) integrates training, parameter sweeping, model evaluation, and conditional registration into a single directed acyclic graph (DAG) executed on Azure ML.

```
[ Trigger (Cron / Manual) ]
            │
            ▼
┌───────────────────────┐
│     Sweep Step        │ ──► Sweeps batch sizes [8, 16, 32]
└──────────┬────────────┘
           │ Best Checkpoint
           ▼
┌───────────────────────┐
│    Evaluate Step      │ ──► Evaluates on cv5_test:1.0.0
└──────────┬────────────┘
           │ eval_results.txt
           ▼
┌───────────────────────┐
│    Register Step      │ ──► Registers if F1 >= 0.6 & beats best version
└──────────┬────────────┘
           │ registered_version.txt
           ▼
┌───────────────────────┐
│     Deploy Step       │ ──► Rolls endpoint forward if new version was registered
└───────────────────────┘
```

*   **Automation & Schedule**: The pipeline is registered with a `JobSchedule` using a Cron Trigger set to run **every Monday at 08:00 UTC** (`0 8 * * 1`), achieving fully automated model lifecycle management.

---

## 5. Inference Deployment & CI/CD (ILO 9.5)

### Deployment Environments
1.  **Local Dev**: Containerized setup via `docker-compose.dev.yml` linking FastAPI, Postgres, and migrations.
2.  **On-Premise (Production)**: Portainer-managed stack using `docker-compose.prod.yaml` for stable, long-running lab access.
3.  **Cloud Serving**: Kubernetes online endpoint (`root-detection-endpoint`) hosted on `lambda-0` for scalable, authenticated inference.

### Branching Strategy & CI/CD Workflows
We follow a Git branching strategy where development is conducted in feature branches, merged to `main` via Pull Requests.
*   **PR Checks (`py-tests.yml`, `py-lint.yml`, `frontend-ci.yml`)**: Triggered on pull requests to `main`. Automatically runs code formatting checks, Ruff linting, and pytest suites. Merges are blocked unless all checks pass.
*   **Release CD (`build-push-prod.yml`)**: Triggered on push to `main`. Builds Docker images for the migration helper, API, and frontend, tagging them with the Git SHA, and pushes them to GitHub Container Registry (GHCR).

### Serving Model & Endpoint Versioning
To serve predictions, we implement an online endpoint running `scripts/score.py`. 
*   **The Combined Serving Model**: We pack the root segmentation (`root-detection`) and anchor detection (`shoot-detection`) models into a single registered serving artifact `root-detection-serving`. This mounts both models inside the container, bypassing workspace identity permission limits.
*   **Deploy Script**: `scripts/deploy_endpoint.py` is called at the end of the retraining pipeline. It reads the output of the registration step and rolls the Kubernetes endpoint forward to the new version with 100% traffic, using an automated rolling deployment.

---

## 6. Observability, Traceability & Frontend (ILO 10.3)

### Database-Backed Audit Logging
Every inference request sent to the API is monitored. When an image is processed, an entry is added to `inference_audit_log` within the database. This logs the request ID, input image SHA-256 hash, latency, target deployment environment, the serving model version, and any warning or error messages.

### Lineage Traceability CLI
We provide a traceability module (`lineage/traceability.py`) to maintain an auditable chain of custody. Given a prediction UUID or request ID, the CLI traces the exact lineage path:

```bash
python -m lineage.traceability <prediction-id> --database-url "postgresql+asyncpg://..."
```

**Expected output trace:**
```
Audit entry: req-908234-cv
  prediction id : e3a1b412-88ef-410a-9d90-ab08b792ffef
  served by     : root-detection-serving:v4
  latency       : 34211 ms
  environment   : azureml

Model lineage:
  git commit    : 5a10a63df5fc76a772f72514994f2e7fb9079943
  training job  : training-run-coral-wing-lgn61
  datasets      : {'train': 'cv5_train:1.0.0', 'val': 'cv5_val:1.0.0'}
```
For pre-existing baseline models, it detects the `baseline` flag and reports the model's metadata honestly, indicating it was imported prior to the training pipeline's creation.

### React Dashboard Frontend
A React/Vite/TypeScript frontend dashboard (`frontend/`) visualizes the system's operational outcomes for stakeholders:
*   **Image Uploader**: Drag-and-drop tool with client-side file size and format validation.
*   **Interactive Segmentation Overlay**: Displays Petri dish images with overlayable segmentation masks (255=root, 0=background).
*   **Live Monitoring**: Displays active job statuses, OT-2 robot connectivity and coordinate tracking, and historical prediction tables.

---

## 7. Code Quality, Linting & Testing (ILO 9.3)

### Readability, docstrings & style
*   **Style**: PEP 8 compliance, formatted with `black` and sorted imports via `isort`.
*   **Linting**: Enforced via `ruff` checks (`E`, `F`, `W`, `I`, `D` rules) configured in `pyproject.toml`.
*   **Docstrings**: All functions and modules have docstrings conforming to the **Google Style guide**.
*   **Pre-commit Hooks**: Checked locally on every commit (`.pre-commit-config.yaml`).

### Unit Testing & Code Coverage
The test suite resides in `tests/`. Testing targets include:
*   Preprocessing and post-processing modules.
*   Model loading, custom Keras functions, and tiling (patchify).
*   API routes, error handling, and authorization.

Tests are configured in `pyproject.toml` to automatically run non-slow tests and enforce a coverage threshold:

```bash
pytest
```
*   **Coverage Target**: Code coverage is measured for `src/npec_cv`. The test execution fails automatically if the statement coverage drops below **70%** (`--cov-fail-under=70`).

---

## 8. Installation & Local Setup Guide

### 1. Backend & CLI Setup
First, ensure you have Python 3.10+ active. Create and activate a virtual environment, then install dependencies:

```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Register pre-commit hooks
pre-commit install

# Install the npec_cv package in editable mode
pip install -e src/npec_cv

# Optional: Install with CUDA support for GPU-accelerated local runs
pip install -e src/npec_cv[cuda]
```

Verify the CLI works:
```bash
npec-cv --help
```

### 2. Database Setup
Ensure you have Docker running. Set up your `.env` file from the template:

```bash
cp .env.example .env
```
Start the local database and run the migrations:
```bash
# Start Postgres container in background
docker compose -f docker-compose.dev.yml up -d db

# Run Alembic migrations to build tables
./scripts/db-migrate.sh upgrade head
```

### 3. Frontend Setup
Navigate to the frontend folder, install npm packages, and run the development server:

```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the application dashboard.

---

## 9. CLI & API Usage Examples

### Running Inference via CLI
Analyze a single plant image:
```bash
npec-cv predict --image data/samples/plant.png --output results/
```
Batch-process a directory of images:
```bash
npec-cv predict data/samples/ --confidence 0.05 --output results.json
```

### Calling the API endpoint (FastAPI)
Run the local FastAPI server:
```bash
uvicorn api.src.main:app --host 0.0.0.0 --port 8000 --reload
```
Perform a prediction using `curl`:
```bash
curl -X POST http://localhost:8000/api/v1/inference/predict \
     -H "Content-Type: application/json" \
     -d '{"image_b64": "<base64-string>", "filename": "sample.png"}'
```

---

## 10. Packaging & Publishing Wheels

We configure the computer vision core as an installable Python package. To package and distribute it:

### 1. Build the Wheels
Run the Python build module from the root directory:
```bash
python -m build
```
This generates the source distribution (`.tar.gz`) and binary distribution (`.whl`) files inside the `dist/` directory.

### 2. Publish to TestPyPI
Upload the distribution files using `twine`:
```bash
python -m twine upload --repository testpypi dist/*
```

### 3. Install from TestPyPI
To install the package in an external environment:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ npec-cv
```

---
*Developed by the NPEC CV5 Team (2026).*