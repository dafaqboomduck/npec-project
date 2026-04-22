# NPEC Root Inoculation Pipeline

## Package Layout

The monolithic src layout has been split into two installable packages:

- npec_cv: plant root detection, segmentation, and length measurement.
- api: FastAPI.
- cli: CLI interface.

Distribution names are also npec_cv and npec_robotics.

npec_robotics imports and uses npec_cv; npec_cv does not depend on robotics.

## Source Tree

```
├── db/                               # Database setup and migrations
│   ├── alembic/                      # Alembic migrations
│   ├── seeds/                        # Database seeds
│   ├── base.py
│   ├── orm.py
│   ├── session.py
│   └── README.md
├── docs/                             # Project documentation
│   ├── database/
│   ├── api_endpoints.md
│   ├── api_spec.yaml
│   └── IO specifications.md
├── frontend/                         # Frontend Application (React/Vite)
│   ├── public/
│   ├── src/
│   │   ├── api/                      # Frontend API integration
│   │   ├── components/               # Reusable UI components
│   │   ├── pages/                    # Page components
│   │   ├── styles/                   # CSS and styling
│   │   └── types/                    # TypeScript type definitions
│   ├── wireframe/                    # UI wireframes and designs
│   ├── package.json
│   └── vite.config.ts
├── requirements/                     # Project dependencies
│   ├── api.txt
│   ├── cuda.txt
│   ├── cv.txt
│   └── dev.txt
├── scripts/                          # Utility scripts
│   └── db-migrate.sh
├── src/                              # Backend + ML system (monorepo style)
│   ├── README.md
│   │
│   ├── api/                          # FastAPI layer
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── main.py
│   │   │
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── memory.py
│   │   │
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   └── inference.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── common.py
│   │   │   ├── health.py
│   │   │   └── predictions.py
│   │   │
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── image_validation.py
│   │       └── mock_inference.py
│   │
│   ├── CLI/                          # CLI Interface
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   └── npec_cv/                      # Core CV/ML package
│       ├── __init__.py
│       ├── _tf_setup.py
│       ├── pipeline.py
│       ├── pyproject.toml
│       ├── README.md
│       │
│       ├── artifacts/
│       │   └── models/
│       │       ├── anchor_det_model_512px.keras
│       │       ├── distilled_student_model.keras
│       │       ├── root_det_model_512px.keras
│       │       └── root_det_model_512px_old.keras
│       │
│       ├── configs/
│       │   ├── __init__.py
│       │   ├── inference.py
│       │   ├── lengths.py
│       │   ├── segmentation.py
│       │   └── thresholds.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── plant_master.py
│       │   ├── plant_segmentation.py
│       │   └── root_lengths.py
│       │
│       ├── exceptions/
│       │   ├── __init__.py
│       │   └── exceptions.py
│       │
│       ├── helpers/
│       │   ├── __init__.py
│       │   ├── dataset_manager.py
│       │   ├── extract_petri_dish.py
│       │   ├── padder.py
│       │   ├── patchify.py
│       │   └── sort_keys.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── custom_functions.py
│       │   └── inference_manager.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── adaptive_threshold.py
│           ├── inference.py
│           ├── length.py
│           ├── pipeline.py
│           ├── plant_master.py
│           └── segmentation.py
│
├── tests/                            # Testing scripts
│   └── cli/
│       ├── conftest.py
│       ├── test_cli.py
│       └── __init__.py
│
├── .env.example
├── docker-compose.yml
├── project-roadmap.png
├── Project Diagram.drawio
└── pyproject.toml
```

## Quick Imports

```python
from npec_cv import CVPipeline, PlantMaster
```

## Dependency Files

- requirements/api.txt
- requirements/cv.txt
- requirements/dev.txt

## Package Installation

Install npec-cv from its package directory:

```bash
pip install -e src/npec_cv
```

Optional npec-cv extras:

```bash
pip install -e src/npec_cv[cuda]
```

Install api as a separate distribution:

```bash
pip install -e src/api
```
