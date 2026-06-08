# ILO 9.4: Data Solutions and Pipeline Scheduling

**Requirement**: Store and version data in the cloud. Convert preprocessing code into automated data pipelines using industry-standard tools, scheduled or triggered automatically.

## Evidence in the Project

### 1. Cloud Data Storage & Versioning
Data assets are explicitly version-controlled and hosted securely in the Azure ML Datastore.
- **Registered Assets**: 
  - `azureml:cv5_train:1.0.0`
  - `azureml:cv5_val:1.0.0`
  - `azureml:cv5_test:1.0.0`
- **Data Management**: Scripts like [scripts/upload_data.py](../scripts/upload_data.py), [scripts/inventory_dataset.py](../scripts/inventory_dataset.py), and [scripts/dataset_deep_check.py](../scripts/dataset_deep_check.py) handle data ingestion, validation, and programmatic registration.

### 2. Orchestrated Data Pipelines
The project moves beyond static scripts by utilizing Azure ML pipelines to construct a complete Directed Acyclic Graph (DAG):
- The pipeline ([scripts/submit_pipeline.py](../scripts/submit_pipeline.py)) unifies preprocessing, training (with parameter sweeps), evaluation, conditional registration, and deployment into a single automated workflow.

### 3. Pipeline Scheduling
- **Automation Trigger**: The retraining pipeline is registered as an Azure ML `JobSchedule`.
- **Cron Evidence**: It is configured with a Cron Trigger to run automatically every Monday at 08:00 UTC (`0 8 * * 1`). This achieves fully automated, hands-off model lifecycle management.

### Summary
The project effectively manages data assets in the cloud and orchestrates the complex sequence of MLOps tasks into a robust, scheduled pipeline, eliminating the need for manual intervention in the retraining process.
