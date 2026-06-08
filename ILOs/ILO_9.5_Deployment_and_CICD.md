# ILO 9.5: Deployment, CI/CD, and Monitoring

**Requirement**: Deploy locally, on-premise and to cloud. Automate linting, testing, and deployment via CI/CD. Deploy registered models with versioning and appropriate strategies. Implement automated retraining and monitoring.

## Evidence in the Project

### 1. Multi-Environment Deployment
The project supports deployment across all three required environments:
- **Local Dev**: Configured via `docker-compose.dev.yml` to spin up the API and Database for local development.
- **On-Premise**: Configured via `docker-compose.prod.yaml`, designed to be deployed to a server using Portainer.
- **Cloud**: The backend and frontend can be deployed as Azure Container Apps, while the inference endpoint is deployed as a Kubernetes online endpoint (`root-detection-endpoint`).

### 2. CI/CD Pipelines
Automated workflows are defined in `.github/workflows/`:
- **Continuous Integration (PR Checks)**: `.github/workflows/py-tests.yml`, `py-lint.yml`, and `frontend-ci.yml` run formatting checks, linting, and tests. Merges are blocked if these fail.
- **Continuous Deployment (Release)**: `.github/workflows/build-push-prod.yml` automatically builds Docker images for the API, database migrations, and frontend upon push to `main`, tags them with the Git SHA, and pushes them to GHCR.

### 3. Model Deployment Strategy
- The Azure ML deployment packages the models into a single registered serving artifact (`root-detection-serving`).
- **Rolling Deployment**: `scripts/deploy_endpoint.py` automates the roll-forward process. When a new model version is registered by the training pipeline, this script updates the endpoint to use the new version while seamlessly routing traffic.

### 4. Retraining and Monitoring
- **Retraining Pipeline**: Triggered via a Cron schedule (every Monday), creating a data flywheel that continuously evaluates and conditionally registers better models.
- **Operational Monitoring**: Request latencies, payload hashes, and serving model versions are tracked via the database-backed audit log. Concurrency is strictly managed to prevent resource exhaustion.

### Summary
The repository demonstrates advanced MLOps deployment practices, featuring full containerization, multi-environment compose files, automated CI/CD checks, and sophisticated rollout strategies for the model endpoints.
