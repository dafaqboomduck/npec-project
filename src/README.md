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
├── src/                              # Backend + ML system (monorepo style)
│   ├── README.md
│   │
│   ├── api/                          # FastAPI layer
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   │
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── inference.py
│   │   │   └── ...
│   │   │
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── common.py
│   │   │   ├── predictions.py
│   │   │   └── ...
│   │   │
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── memory.py
│   │   │   └── ...
│   │   │
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── mock_interface.py
│   │       ├── image_validation.py
│   │       └── ...
│   │
│   ├── cli/                        # CLI Interface
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py
│   │   │   ├── info.py
│   │   │   └── version.py
│   │   └── utils.py
│   │
│   └── npec_cv/                      # Core CV/ML package
│       ├── pipeline.py
│       ├── _tf_setup.py
│       ├── pyproject.toml
│       ├── README.md
│       ├── __init__.py
│       │
│       ├── artifacts/
│       │   └── models/
│       │       ├── anchor_det_model_512px.keras
│       │       ├── root_det_model_512px.keras
│       │       ├── root_det_model_512px_old.keras
│       │       └── ...
│       │
│       ├── configs/
│       │   ├── inference.py
│       │   ├── lengths.py
│       │   ├── segmentation.py
│       │   ├── thresholds.py
│       │   └── __init__.py
│       │
│       ├── core/
│       │   ├── plant_master.py
│       │   ├── plant_segmentation.py
│       │   ├── root_lengths.py
│       │   └── __init__.py
│       │
│       ├── models/
│       │   ├── pipeline.py
│       │   ├── segmentation.py
│       │   ├── plant_master.py
│       │   ├── inference.py
│       │   ├── adaptive_threshold.py
│       │   ├── length.py
│       │   └── __init__.py
│       │
│       ├── inference/
│       │   ├── inference_manager.py
│       │   ├── custom_functions.py
│       │   └── __init__.py
│       │
│       ├── helpers/
│       │   ├── dataset_manager.py
│       │   ├── extract_petri_dish.py
│       │   ├── padder.py
│       │   ├── patchify.py
│       │   ├── sort_keys.py
│       │   └── __init__.py
│       │
│       └── exceptions/
│           ├── exceptions.py
│           └── __init__.py
│        
├── tests/                      
│   ├── __init__.py
│   ├── conftest.py
│   ├── api/
│   │   ├── test_health.py
│   │   ├── test_inference.py
│   │   └── ...
│   │
│   ├── cli/
│   │   ├── test_predict.py
│   │   ├── test_info.py
│   │   └── ...
│   │
│   └── npec_cv/
│       ├── test_configs.py
│       ├── test_models.py
│       └── ...
│
└── pytest.ini            
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
