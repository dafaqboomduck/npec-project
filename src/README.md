# NPEC Root Inoculation Pipeline

## Package Layout

The monolithic src layout has been split into two installable packages:

- npec_cv: plant root detection, segmentation, and length measurement.
- npec_robotics: OT-2 simulation and robotic manipulation pipeline.

Distribution names are also npec_cv and npec_robotics.

npec_robotics imports and uses npec_cv; npec_cv does not depend on robotics.

## Source Tree

```
src/
├── npec_cv/
│   ├── pipeline.py
│   ├── core/
│   ├── inference/
│   ├── models/
│   ├── configs/
│   ├── helpers/
│   ├── exceptions/
│   └── artifacts/models/*.keras
└── npec_robotics/
    ├── pipeline.py
    ├── controllers/
    ├── models/
    ├── configs/
    ├── exceptions/
    ├── meshes/
    ├── textures/
    ├── *.urdf
    └── artifacts/models/*.zip
```

## Quick Imports

```python
from npec_cv import CVPipeline, PlantMaster
from npec_robotics import RoboticsPipeline
```

## Dependency Files

- requirements/cv.txt
- requirements/robotics.txt
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

Install npec-robotics as a separate distribution:

```bash
pip install -e src/npec_robotics
```
