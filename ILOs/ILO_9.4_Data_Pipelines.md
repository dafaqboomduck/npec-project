# ILO 9.4: Data Solutions & Pipelines (Must Pass)

**ILO 9.4 (Must Pass)**: The student demonstrates the ability to design and manage data solutions, including the secure storage and version control of data and implement data pipelines using industry-standard tools and practices.

| Pre-requisites | A: 5 points | B: 10 points |
|--|--|--|
| The student submits a complete learning log, work log, and self-assessment rubric. The student exhibits professional behaviour during DataLab sessions. | The student demonstrates the ability to store data in the cloud and manage data through code. The student applies version control to data assets. | The student is able to convert data preprocessing or processing code into a data pipeline. The group is able to perform all required data preprocessing and processing steps in data pipelines using industry-standard tools. The data pipelines are run on a schedule or triggered automatically based on appropriate criteria. |

## Pre-requisites

Daily learning log, all tasks tracked in Azure DevOps Boards, self-assessment rubric updated weekly. Professional conduct in every DataLab session.

## Criterion A (5 points)

Raw data lives in Azure ML workspace blob storage as three registered data assets (cv5_train, cv5_val, cv5_test) consumed by every pipeline via azureml:<name>:<version> references. Processed patches are stored as a separate versioned asset cv5_patches, with a new integer version registered automatically each pipeline run by preprocess_dataset.py:get_next_version(). All data references are pinned to specific versions in code, never to a moving "latest" pointer.

**Possible evidence:**
- [Links to Azure ML data stores](https://ml.azure.com/model/list?wsid=/subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourceGroups/buas-y2/providers/Microsoft.MachineLearningServices/workspaces/team-CV5-2026&tid=0a33589b-0036-4fe8-a829-3ed0926af886)
- [Links to registered data assets in Azure ML](https://ml.azure.com/data?wsid=/subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourceGroups/buas-y2/providers/Microsoft.MachineLearningServices/workspaces/team-CV5-2026&tid=0a33589b-0036-4fe8-a829-3ed0926af886)
- [Data management scripts](https://ml.azure.com/experiments/id/6329f316-f8e5-46c6-8ae8-f7e853f87335/runs/teal_sail_zqlpfn4tgd?wsid=/subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourceGroups/buas-y2/providers/Microsoft.MachineLearningServices/workspaces/team-CV5-2026&tid=0a33589b-0036-4fe8-a829-3ed0926af886#)


## Criterion B (10 points)

*Explain how you converted preprocessing or processing code into a repeatable data pipeline using industry-standard tools, including how the pipeline is scheduled or triggered automatically.*

**Possible evidence:**
- [Screenshots of successful pipeline runs.](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/ILO_1.8_Professional_Standards.md)
- [Link to preprocessing pipeline submitter with cron schedule and run-now triggers](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/submit_preprocessing_pipeline.py)
- [Link to the worker script the pipeline runs (patching plus data asset registration)](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/preprocess_dataset.py)
- [Link to full retrain pipeline showing the train, evaluate, register, deploy chain](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/submit_pipeline.py)
- [Link to training-only pipeline variant for fast iteration](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/submit_training_only_pipeline.py)
- [Link to conditional registration step that drives downstream deploy](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/register_model_if_better.py)
