# ILO 4.4: Problem Analysis

**ILO 4.4**: The student is able to propose a design and plan for a ML application architecture and data pipeline and evaluate the financial costs of the application.

| Pre-requisites | A: 5 points | B: 5 points |
|--|--|--|
| The student submits a complete learning log, work log, and self-assessment rubric. The student exhibits professional behaviour during DataLab sessions. | The student demonstrates the ability to design and plan a cloud-based ML application architecture and data pipeline that addresses the creative brief requirements. | The student demonstrates the ability to evaluate the financial costs of the proposed cloud-based ML application, employing cost-effective resource management and monitoring strategies to balance performance and budget constraints. |

## Pre-requisites

Maintained daily learning log, all tasks in Azure DevOps Boards, self-assessment rubric updated weekly. Professional conduct in every DataLab session.

## Criterion A (5 points)

Designed a four stage ML pipeline (preprocess, train+sweep, evaluate, register, deploy) running on Azure ML with lambda-0 as the Kubernetes compute. Data flows: raw images in cv5_train/val/test data assets, preprocessed into versioned cv5_patches, consumed by training, output models registered and combined into a serving artifact, deployed to a Kubernetes online endpoint behind a FastAPI backend. Each pipeline is a separate Python submitter script using azure-ai-ml SDK, scheduled weekly with cron triggers.

**Possible evidence:**
- [Architecture diagrams (for each deployment environment)](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/architecture/data_pipeline.drawio)
- [Package plan with modules structure and dependencies](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/requirements)
- [Data pipeline diagram](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/architecture/data_pipeline.drawio)
- [Links to infrastructure, workflow, or pipeline configuration files (extra)](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/submit_pipeline.py)

## Criterion B (5 points)

Wrote a full cost analysis (docs/cost-analysis.md) covering compute, storage, ACR, bandwidth, and training versus inference. Chose CPU for inference (low traffic makes GPU wasteful) and GPU for training (continuous math justifies the reservation). Migrated all compute from Azure managed VMs to lambda-0 Kubernetes, reducing Azure spend from around 200 euro per month to around 8 euro per month. Disabled flip merge in score.py to halve inference latency after verifying no quality regression. Combine job is cached by model version to avoid repeat work. Monitoring via kubectl top, container logs, and Azure ML Studio metrics.

**Possible evidence:**
- [Cost report](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/cost-analysis.md)