# ILO 9.3: High Quality Code (Must Pass)

**ILO 9.3 (Must Pass)**: The student demonstrates the ability to produce high quality code that adheres to industry standards. The code is packaged in such a way that it can be used by stakeholders outside of the development environment.

| Pre-requisites | A: 5 points | B: 5 points | C: 5 points |
|--|--|--|--|
| The student submits a complete learning log, work log, and self-assessment rubric. The student exhibits professional behaviour during DataLab sessions. Clear individual contribution to this ILO is documented. | The student demonstrates the ability to write production-ready code that adheres to industry technical standards for readability, logging, and error handling. Code must have function-specific docstrings that enhance clarity and usability. Code conforms to industry-standard style guidelines and passes applicable linting and/or formatting checks. | The student is able to document their code in a clear and professional manner. The documentation must include detailed usage instructions, installation instructions, and examples. | The student is able to create comprehensive unit tests designed to achieve coverage levels that meet client requirements and ensure code reliability. Tests are written for all key functions and modules. |

## Pre-requisites

Daily learning log, all tasks tracked in Azure DevOps Boards with my name on the cards I owned, self-assessment rubric updated weekly. Professional conduct in every DataLab session.

## Criterion A (5 points)

All scripts and modules I authored have function-level docstrings explaining purpose, args, and return values. Logging via the standard logging module (no print debugging in production paths). Error handling: explicit try/except around Azure SDK calls and model loading, with informative messages and fallback paths (for example get_client in deploy_endpoint.py, get_next_version in preprocess_dataset.py). Code passes ruff and black checks configured in pyproject.toml and runs pre-commit hooks. Virtual environment pinned in requirements/dev.txt and conda env files under environments/.

**Possible evidence:**
- [Link to key source file showing modular, well-structured code, docstrings, structured logging, and Azure SDK error handling](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/deploy_endpoint.py)
- [Link to preprocessing pipeline script with function-level docstrings, fallback paths, and clean argparse usage](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/preprocess_dataset.py)
- [Link to Azure ML pipeline submitter showing modular component composition](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/submit_pipeline.py)
- [Link to deployed endpoint scoring code with diagnostic logging and defensive error handling](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/scripts/score.py)
- [Link to linting and formatting configuration (ruff and black) enforcing PEP 8 compliance](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/pyproject.toml)
- [Link to pre-commit hook configuration enforcing format and lint on every commit](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.pre-commit-config.yaml)
- [Link to CI workflow running lint checks on every push and pull request](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/py-lint.yml)
- [Link to conda environment configurations for training and endpoint serving](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/environments)
- [Link to pinned development and runtime requirement files](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/requirements)
- [Commit history showing your individual contribution](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/ILOs/commits.png)

## Criterion B (5 points)

Top-level README explains the project, install steps, and common commands. Every script in scripts/ has a docstring at the top with usage examples. The docs/ folder is organised by audience (api specifications, deployment setup, lineage, cost analysis, architecture). Sphinx config builds a documentation site. Deployment instructions are in deploy_endpoint.py docstrings and docs/azure-setup.md.

**Possible evidence:**
- [Link to top-level README with project overview, installation, and usage instructions](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/README.md)
- [Link to the docs folder organised by audience (API, architecture, lineage, cost, setup)](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/docs)
- [Link to Sphinx configuration that builds the documentation site](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/conf.py)
- [Link to CI workflow that publishes the documentation site on every merge](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/docs.yml)
- [Link to Azure setup guide covering workspace, compute, and credentials configuration](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/azure-setup.md)
- [Link to cost analysis report with compute, storage, and inference cost estimates](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/cost-analysis.md)
- [Link to architecture explanation with deployment diagrams and component breakdown](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/docs/architecture)
- [Link to lineage tracking documentation showing how predictions trace back to training data](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/docs/lineage)
- [Link to rolling deployment behaviour illustration for the Kubernetes online endpoint](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/docs/ROLLING_DEPLOYMENT_ILLUSTRATION.md)

## Criterion C (5 points)

Test suite under tests/ is split by concern (preprocessing, inference, postprocessing, e2e, cli, api, scripts, lineage). CI workflow runs pytest with coverage and enforces 85 percent minimum via --cov-fail-under=85. Tests validate critical functions: image patching, model loading, endpoint response shape, pipeline submission helpers. CI passing on every merge confirms reliability.

**Possible evidence:**
- [Link to full test suite organised by concern (preprocessing, inference, postprocessing, e2e, cli, api, scripts, lineage)](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/tests)
- [Link to script-level tests covering pipeline submitters and deploy helpers](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/tests/scripts)
- [Link to API tests validating FastAPI endpoints and response shapes](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/tests/api)
- [Link to end-to-end tests validating the full inference flow](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/tests/e2e)
- [Link to inference unit tests covering model loading and prediction logic](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/tree/main/tests/inference)
- [Link to CI workflow running pytest with coverage and enforcing 85 percent minimum](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/.github/workflows/py-tests.yml)
- [Link to shared pytest fixtures used across the test suite](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/blob/main/tests/conftest.py)
- [Link to passing CI runs proving tests execute green on every merge](https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-computervision5/actions/workflows/py-tests.yml)