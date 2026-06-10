# ILO 9.5: Deployment & Maintenance (Must Pass)

**ILO 9.5 (Must Pass)**: The student demonstrates the ability to deploy and maintain machine learning solutions using industry-standard tools, integrating version control, continuous integration, automated pipelines, and monitoring practices for both local and cloud environments.

| Pre-requisites | A: 5 points | B: 5 points | C: 5 points | D: 5 points |
|--|--|--|--|--|
| The student submits a complete learning log, work log, and self-assessment rubric. The student exhibits professional behaviour during DataLab sessions. | The student is able to deploy an inference pipeline using industry-standard tools. The student is able to interact with a deployed model locally and on on-premise infrastructure. | The student is able to implement a GitHub branching strategy and automate both linting and unit tests. The deployment of the application is automated. The student is able to merge their code and ensure the tests pass. | The student is able to deploy registered models as endpoints and integrate them into the application. The student applies version control to models and is able to select model versions for use in the inference pipeline. The deployment process is automated and well-documented. The student deploys new model versions using appropriate deployment strategies. | The student can implement an automated machine learning retraining pipeline including automated model evaluation and deployment. Implement an appropriate monitoring solution, taking into account technical and business metrics, on the training process and the deployed models. |

## Pre-requisites

Daily learning log, all tasks tracked in Azure DevOps Boards, self-assessment rubric updated weekly. Professional conduct in every DataLab session.

## Criterion A (5 points)

Inference is deployed as a Kubernetes online endpoint on the lab's lambda-0 cluster using Azure ML, an industry-standard MLOps platform. The deployment is fully scripted in deploy_endpoint.py and uses the same score.py contract Azure ML expects (init() + run()). For local interaction, test_endpoint.py --local loads score.py and runs it directly against a test image. For remote interaction, the same script in default mode hits the deployed endpoint via its scoring URI plus key. The FastAPI backend (backend/app.py) consumes the endpoint over HTTP.

**Possible evidence:**
- Application or inference service repository links
- Local deployment instructions (must be containerised)
- Screenshots or logs of successful inference (local)
- Link to working on-premise deployment
- Instructions on how to interact with the on-premise deployment and sample data for testing

## Criterion B (5 points)
 
### Branching strategy
 
The repository uses a two-tier branching model. `main` holds stable, tested code, and `dev` is the integration branch for ongoing development. All actual work happens on short-lived `feature/`, `fix/`, and `hotfix/` branches.
 
Once a feature or fix is finished, it is **squash merged into `dev`**, which keeps the history on `dev` clean and linear by collapsing each branch into a single commit. From there, batches of changes are promoted from `dev` into `main` using a **normal merge** rather than a squash, which preserves the history lineage between the two branches and avoids the false merge conflicts that squash-on-promotion would otherwise create.
 
The normal flow is therefore: `feature/` or `fix/` branch → `dev` → `main` (as a batch of changes).
 
Both `main` and `dev` are **protected branches**, so no one can push to them directly. All changes must go through a pull request and be merged via GitHub. Any PR into `main` additionally requires approval from at least one other team member, and we discuss it as a team before merging.
 
### CI: automated linting and tests
 
On pull requests to `main`, a set of checks runs automatically:
 
- [Frontend linting](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml) and [Python linting](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml)
- [Python unit tests](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml) with a coverage requirement (85%)
- A [Docker build check](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml) that verifies the images build correctly
 
By team agreement these checks are **advisory rather than blocking**, so they surface problems and inform the review without hard-failing the merge. The mandatory gate is instead the required human approval: every PR to `main` must be reviewed and approved by at least one teammate, and we discuss it before merging.
 
### CD: automated deployment
 
On pushes to `main`, two further workflows run. They **build and push the container images** needed for deployment to GHCR. There are two separate workflows because the on-prem and cloud deployments use different images: [`build-push-prod.yml`](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml) for the on-prem images and [`build-deploy-cloud.yml`](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml) for the cloud images. Each of these workflows then **triggers an automatic redeployment** on its respective platform, so a merge into `main` flows straight through to a redeployed application without manual intervention.
 
### Evidence
 
- Branching strategy diagram and explanation (above)
- Commit history showing the branching model in action: [`main`](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/commits/main/) and [`dev`](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/commits/dev/)
- CI workflow files: [frontend lint](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml), [Python lint](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml), [Python tests + coverage](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml), [Docker build check](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml)
- [Screenshots of the checks running on a PR to `main`](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/ILOs/ci_checks_main)
- CD workflow files: [on-prem build/push](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml), [cloud build/deploy](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/frontend-ci.yml)
- Screenshots of a protected-branch PR requiring approval before merge
 
Criterion C (5 points)  

Models are versioned in the Azure ML model registry as root-detection:N. The combine job bundles each root model with the anchor model into root-detection-serving:N. deploy_endpoint.py picks up the latest registered version (or a pinned --root-version) and rolls out a Kubernetes online deployment named current with a rolling update. A second deployment proof can be created via add_proof_deployment.py to demonstrate blue/green capability with 0% traffic split. The full process is documented in the deploy script docstring and in docs/.

Evidence:

- [Link to deployment automation with rolling update and conditional pipeline-mode short-circuit](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/deploy_endpoint.py)
- [Link to second-deployment helper proving multi-deployment capability for blue green strategy](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/add_proof_deployment.py)
- [Link to combine job that bundles root and anchor models into a single serving artifact](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/_combine/_combine_models.py)
- [Link to model selection logic by registered version with f1 gate](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/register_model_if_better.py)
- [Link to backend integration consuming the endpoint scoring URI and key](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/api)
- [Link to model inspection helper for the registry](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/inspect_models.py)
- [Link to rolling deployment behaviour and pod overlap documentation](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/ROLLING_DEPLOYMENT_ILLUSTRATION.md)

Criterion D (5 points)

Automated retraining is implemented as a weekly cron-scheduled Azure ML pipeline (submit_pipeline.py) that trains with a batch-size hyperparameter sweep, evaluates on the held-out test set, conditionally registers the model only if F1 beats the best registered version and meets a minimum threshold, and conditionally rolls the endpoint forward only if a new model was registered. Per-epoch metrics (loss, val_loss, accuracy, f1_score, val_f1_score) log to MLflow via the custom MlflowEpochMetricsCallback and render as Studio Metrics tab charts. Live endpoint monitoring is done via get_logs.py which pulls container logs including per-request diagnostics (latency, plant counts, per-zone activations) and via kubectl top for resource usage.

Evidence:

- [Link to automated retraining pipeline with sweep, conditional registration, and conditional deploy](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/submit_pipeline.py)
- [Link to conditional model registration gated on F1 threshold and best-registered comparison](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/register_model_if_better.py)
- [Link to model evaluation job logic](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/evaluate_root_detection.py)
- [Link to training script with the per-epoch MLflow metric logging callback](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/train_root_detection.py)
- [Link to endpoint monitoring helper for container log retrieval](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/get_logs.py)
- [Link to cost analysis report documenting monitoring strategy and technical metrics](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/cost-analysis.md)

