# npec_cv

Computer-vision package for plant root analysis.

## Public API

```python
from src.npec_cv import CVPipeline, cv_pipeline, PlantMaster
from src.npec_cv.models import PlantMeasurement, SegmentationConfig, ModelConfig
```

## Includes

- Inference manager and custom Keras functions
- Root segmentation and anchor assignment
- Geodesic root length calculation
- Trained .keras artifacts in artifacts/models

---

## CLI

`npec_cv` ships with a command-line tool — **`npec-cv`** — that lets you run
plant image analysis without writing any Python.

### Installation

```bash
pip install ./npec_cv          # from the src/ directory
# or, with GPU support:
pip install "./npec_cv[cuda]"
```

After installation the `npec-cv` command is available in your terminal:

```bash
npec-cv --help
```

### Commands

| Command | Description |
|---------|-------------|
| `predict` | Run inference on a single image or a directory of images |
| `info` | Print model metadata and pipeline configuration |
| `version` | Print the installed package version |

---

### Usage examples

#### 1. Analyse a single image

```bash
npec-cv predict plant_image.jpg
```

Example output:

```json
{
  "image": "plant_image",
  "num_detected": 5,
  "total_plants": 6,
  "plants": {
    "0": {
      "detected": true,
      "length_px": 342,
      "tip_location": [420, 180],
      "confidence_score": 0.45,
      "pixel_count": 1024,
      "mask_available": true
    },
    "5": {
      "detected": false,
      "length_px": null,
      "tip_location": null,
      "confidence_score": null,
      "pixel_count": null,
      "mask_available": false
    }
  }
}
```

#### 2. Batch process a directory of images

```bash
npec-cv predict images/
```

Processes every `.jpg`, `.jpeg`, and `.png` in `images/` and outputs a JSON
array — one object per image — to stdout:

```json
[
  { "image": "plate_01", "num_detected": 6, "total_plants": 6, "plants": { ... } },
  { "image": "plate_02", "num_detected": 5, "total_plants": 6, "plants": { ... } }
]
```

#### 3. Custom confidence threshold, save results to a file

```bash
npec-cv predict images/ --confidence 0.05 --output results.json
```

Lowering `--confidence` (default `0.01`) causes the pipeline to detect roots
at lower probability thresholds — useful for faint or young roots.
Results are written to `results.json` instead of stdout.

#### 4. Print model metadata

```bash
npec-cv info
```

```json
{
  "package": "npec_cv",
  "version": "0.2.0",
  "models": {
    "root_detection": {
      "checkpoint": ".../root_det_model_512px.keras",
      "patch_size": 512,
      "batch_size": 4,
      "default_threshold": 0.01
    },
    "anchor_detection": {
      "checkpoint": ".../anchor_det_model_512px.keras",
      "patch_size": 512,
      "batch_size": 4,
      "default_threshold": 0.001
    }
  },
  "supported_formats": [".jpeg", ".jpg", ".png"]
}
```

#### 5. Print package version

```bash
npec-cv version
# 0.2.0
```

### Options reference

#### `predict`

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `PATH` | — | required | Image file or directory |
| `--confidence` | `-c` | `0.01` | Detection threshold (0–1) |
| `--output` | `-o` | stdout | Write JSON to this file |

#### `info` / `version`

No options — just run the command.
