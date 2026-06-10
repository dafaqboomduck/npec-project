# npec-cv CLI

Command-line tool for running NPEC plant root analysis from the terminal — no Python code required.

---

## Installation

From the repo root:

```bash
pip install -e src/npec_cv
```

Verify the install:

```bash
npec-cv --help
```

---

## Requirements

Before running inference you need the trained `.keras` model files.  
They live in the repo at:

```
scripts/
├── root_det_model_512px.keras
└── anchor_det_model_512px.keras
```

Pass this folder with `--artifact-dir` / `-a` on every `predict` call.

Full path example (Windows):

```
C:\Users\koenm\Documents\repositorys\year_2\2025-26d-fai2-adsai-group-computervision5\scripts
```

example prediction:

```
npec-cv predict "C:\Users\koenm\Downloads\train_Alican_230858_im2.png" --artifact-dir "C:\Users\koenm\Documents\repositorys\year_2\2025-26d-fai2-adsai-group-computervision5\scripts"
```
---

## Commands

| Command | Description |
|---------|-------------|
| `predict` | Run inference on a single image or a directory of images |
| `info` | Print model metadata and pipeline configuration |
| `version` | Print the installed package version |

---

## predict

Run plant-root inference on an image or a batch directory.

```
npec-cv predict PATH --artifact-dir DIR [OPTIONS]
```

### Arguments

| Name | Required | Description |
|------|----------|-------------|
| `PATH` | yes | Image file or directory of images |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--artifact-dir` | `-a` | — | **Required.** Directory containing the `.keras` model files |
| `--confidence` | `-c` | `0.01` | Detection threshold (0–1). Lower = detect fainter roots |
| `--output` | `-o` | stdout | Write JSON results to this file instead of stdout |

### Image constraints

| Constraint | Value |
|------------|-------|
| Accepted formats | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` |
| Max file size | 50 MB |
| Min resolution | 512 × 512 px |
| Max resolution | 5000 × 5000 px |

### Output format

**Single image** → JSON object  
**Directory** → JSON array (one object per image)

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
      "confidence_score": 0.5,
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

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `image` | Filename stem (no extension) |
| `num_detected` | Number of plants with a detected root |
| `total_plants` | Total expected plants in the image |
| `detected` | Whether a root was found for this plant |
| `length_px` | Geodesic root length in pixels |
| `tip_location` | `[row, col]` pixel coordinate of the root tip |
| `confidence_score` | Adaptive threshold at which the root was detected (0.5 = detected immediately, lower = faint root) |
| `pixel_count` | Total pixels in the root mask |
| `mask_available` | Whether a binary mask was produced |

### Examples

```bash
# Single image (Windows)
npec-cv predict "C:\Users\koenm\Downloads\train_Alican_230858_im2.png" --artifact-dir "C:\path\to\repo\scripts"

# Single image (short flag)
npec-cv predict "C:\Users\koenm\Downloads\plant.png" -a "C:\path\to\repo\scripts"

# Batch process a directory
npec-cv predict "C:\Users\koenm\Downloads\images" -a "C:\path\to\repo\scripts"

# Lower threshold for faint/young roots
npec-cv predict "C:\Users\koenm\Downloads\plant.png" -a "C:\path\to\repo\scripts" --confidence 0.005

# Save results to a file instead of stdout
npec-cv predict "C:\Users\koenm\Downloads\plant.png" -a "C:\path\to\repo\scripts" --output results.json

# Combine options
npec-cv predict "C:\Users\koenm\Downloads\images" -a "C:\path\to\repo\scripts" -c 0.05 -o results.json
```

> **Tip:** Replace `C:\path\to\repo` with your actual repo root, e.g.
> `C:\Users\koenm\Documents\repositorys\year_2\2025-26d-fai2-adsai-group-computervision5`

---

## info

Print model metadata and pipeline configuration as JSON.

```bash
npec-cv info
npec-cv info -a scripts/
```

Example output:

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
  "accepted_extensions": [".jpeg", ".jpg", ".png", ".tif", ".tiff"],
  "constraints": {
    "max_file_size_mb": 50,
    "min_resolution_px": 512,
    "max_resolution_px": 5000
  }
}
```

---

## version

Print the installed package version.

```bash
npec-cv version
# 0.2.0
```

---

## Running tests

From the repo root:

```bash
# CLI tests only
pytest tests/cli/ -v

# With coverage
pytest tests/cli/ -v --cov=src/npec_cv/cli --cov-report=term-missing

# All tests
pytest -v
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `npec-cv` not recognised | Run `pip install -e src/npec_cv` |
| `--artifact-dir does not exist` | Check the path to your `.keras` files |
| `ModuleNotFoundError: No module named 'src'` | Run pytest from the repo root, not from inside `src/` |
| TensorFlow logs on `--help` | Expected on first import — only appears once per session |
