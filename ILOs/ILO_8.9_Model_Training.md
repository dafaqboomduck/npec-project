# ILO 8.9: Model Training and Experiment Tracking

**Requirement**: Train models locally and in the cloud with experiment tracking. Build training/evaluation pipelines with automated hyperparameter tuning and conditional model registration.

## Evidence in the Project

### 1. Cloud Training Pipeline
The repository contains a robust set of scripts for automating training on Azure ML:
- **`scripts/submit_training_job.py`**: Orchestrates the packaging of the repository and provisions the training environment (`root-detection-train` via `environments/azure-train-env.yml`). It submits a Command job to the GPU cluster.
- **`scripts/train_root_detection.py`**: The core training script executed on the cloud compute.
- **Local Training**: Supported and documented via `scripts/train_local_and_report.py`.

### 2. Experiment Tracking (MLflow)
The training scripts leverage MLflow autologging to track experiments natively within the Azure ML Studio:
- Tracks Training/Validation Metrics (Loss curves, learning rates, F1 score).
- Logs Artifacts (saved checkpoints, configurations, sample predictions).
- Captures Environment Metadata (dependencies, environment variables, Git commit hashes).

### 3. Automated Hyperparameter Tuning
The pipeline includes hyperparameter sweeps to optimize the model:
- Configured during the retraining pipeline, it utilizes a grid search (e.g., sweeping batch sizes `[8, 16, 32]`) to maximize the `best_val_f1` metric.

### 4. Conditional Model Registration
To ensure production stability, the pipeline does not blindly register newly trained models.
- **`scripts/evaluate_root_detection.py`**: Evaluates the best model from the sweep against a dedicated test dataset (`azureml:cv5_test:1.0.0`).
- **`scripts/register_models.py`** (or `register_model_if_better` logic): Only registers the model to the Azure ML Registry if the `mean_f1` score exceeds a strict threshold (0.6) AND beats the highest F1 score of the currently registered versions.

### Summary
The implementation fully satisfies the requirements by providing an automated, cloud-native training pipeline that features hyperparameter sweeping, comprehensive MLflow tracking, and a safeguard mechanism (conditional registration) before models are marked for deployment.
