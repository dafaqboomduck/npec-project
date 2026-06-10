# ILO 8.9: Modeling

**ILO 8.9**: The student demonstrates the ability to train machine learning models on multiple platforms both locally and in the cloud making use of appropriate resources, platforms, and automation tools.

| Pre-requisites | A: 4 points | B: 6 points |
|--|--|--|
| The student submits a complete learning log, work log, and self-assessment rubric. The student exhibits professional behaviour during DataLab sessions. | The student demonstrates the ability to train models both locally and on an appropriate cloud platform using industry-standard libraries to track key training and evaluation metrics. The student demonstrates the ability to use version control techniques and tools in the context of ML models. | The student trains and evaluates multiple model architectures including custom and existing open-source models on cloud ML platforms. The student builds training and evaluation pipelines incorporating automated hyperparameter tuning and conditional model registration. Experiment tracking with advanced automated metric logging and visualisation is used to log key training and evaluation metrics, including learning curves and example outputs. The student demonstrates the ability to schedule or trigger pipelines based on appropriate criteria. |

## Pre-requisites

Daily learning log, all tasks tracked in Azure DevOps Boards, self-assessment rubric updated weekly. Professional conduct in every DataLab session.

## Criterion A (4 points)

Trained the U-Net locally via train_root_detection.py and on Azure ML via submit_pipeline.py (compute: lambda-0). Both code paths share the same training script. MLflow logs parameters, per-epoch loss, accuracy, and F1 via MlflowEpochMetricsCallback. Models are versioned in the Azure ML model registry (root-detection:N), code is versioned in Git with feature branches and pull request reviews.

**Possible evidence:**
- [Training code local and cloud (jobs in Azure ML)- Experiment tracking screenshots or links Code showing metric logging](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/ILO_8.9_Model_Training.md) [local here](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/train_local_and_report.py)
- [Screenshots or links showing models registered in a model registry](https://ml.azure.com/model/list?wsid=/subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourceGroups/buas-y2/providers/Microsoft.MachineLearningServices/workspaces/team-CV5-2026&tid=0a33589b-0036-4fe8-a829-3ed0926af886)


## Criterion B (6 points)

Multiple architectures evaluated: custom U-Net built from scratch and an alternative branched root model. Pipeline (submit_pipeline.py) chains four steps: train with hyperparameter sweep over batch sizes (8, 16, 32), evaluate on cv5_test, conditionally register if F1 beats threshold and current best (register_model_if_better.py), conditionally deploy. Per-epoch metrics render as line charts in Studio. Pipeline is scheduled weekly via cron and can be triggered on demand by the FastAPI backend (/pipeline/run endpoint reuses the stored schedule definition).

**Possible evidence:**
- [Links to cloud training jobs, pipeline definitions, or orchestration files](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/scripts)
- [Evidence of successful training, evaluation, and model registration pipelines](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/azure-pipeline-completed.png)
- [Evidence of pipelines that include automated hyperparameter tuning and conditional model registration](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/azure-pipeline-completed.png)
- [Screenshots of experiment tracking dashboards showing automated metric logging and visualisation in Azure ML](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/learning-visualizations.png)
- [Visualisations of learning curves or example outputs](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/learning-visualizations.png)
- [Examples of pipeline triggers or schedules based on criteria such as new data availability, time-based scheduling, or performance thresholds](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/schedule.png)