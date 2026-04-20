# Model I/O Specification - Plant Root Segmentation Model

**Document version:** 1.0.0  
**Last updated:** 2026-04-17  
**Status:** Draft - Sprint 1  
**Owner:** NPEC Computer Vision Pipeline  
**Applies to:** Sprint 2 model refactoring · API schema definition  
**ILO:** 4.4A · 9.4A

---

## Table of Contents

1. [Overview](#overview)
2. [Input Specification](#input-specification)
3. [Output Specification](#output-specification)
4. [Normalisation](#normalisation)
5. [Edge Cases](#edge-cases)
6. [Validation Contract](#validation-contract)
7. [Version History](#version-history)

---

## Overview

This document defines the authoritative input/output contract for the NPEC plant root segmentation model (U-Net). It is the single source of truth for:

- Pre-processing logic in the inference pipeline
- API request/response schema (Sprint 2)
- Monitoring and data validation checks in production

Any change to this spec must be version-bumped and reviewed before being merged into `main`.

---

## Input Specification

### 2.1 Image Source

| Property | Value |
|---|---|
| Imaging system | HADES phenotyping platform |
| Capture modes accepted | **Greyscale (B&W)** · **Colour (RGB)** |
| Primary operational mode | Greyscale - preferred for segmentation accuracy |

### 2.2 Accepted File Types

| Format | Extension(s) | Notes |
|---|---|---|
| PNG | `.png` | Lossless, preferred |
| TIFF | `.tiff` · `.tif` | Lossless, multi-page not supported - single frame only |
| JPEG | `.jpg` · `.jpeg` | Lossy - accepted but **not recommended**; compression artefacts may degrade segmentation |

> **Note for Sprint 2 API:** Reject multi-page TIFFs at the ingestion layer with HTTP 422 and a descriptive error body.

### 2.3 Resolution

| Property | Value |
|---|---|
| Minimum resolution | 512 × 512 px |
| Maximum resolution | 4096 × 4096 px |
| Recommended / native resolution | 2775 px wide (HADES default) |
| Aspect ratio | Any - pipeline crops per-plant ROI before inference |
| Spatial scale (at native res) | `150 px / mm` → scale factor `150 / 2775 ≈ 0.054 mm/px` |

Images outside the resolution range are **rejected** before inference (see [Validation Contract](#validation-contract)).

### 2.4 Channel Count

| Mode | Channels | Accepted |
|---|---|---|
| Greyscale | 1 | Yes - primary mode |
| RGB | 3 | Yes - converted to greyscale internally before inference |
| RGBA | 4 | Alpha channel stripped; remainder treated as RGB |
| Other | > 4 | Rejected |

Conversion from RGB to greyscale uses the standard luminance formula:

```
Y = 0.299·R + 0.587·G + 0.114·B
```

### 2.5 Normalisation

> **Source:** Block B training dataset (NPEC, 2025). Update these values if the model is retrained on a new dataset.

| Parameter | Value |
|---|---|
| Method | Z-score normalisation (per-image, channel-wise) |
| Training dataset mean (greyscale) | `[ask razvan]` |
| Training dataset std (greyscale) | `[ask razvan]` |
| Applied before | Patch extraction |

**Formula:**

```
pixel_normalised = (pixel_raw / 255.0 - mean) / std (ask razvan to confirmm)
```

If the image arrives as RGB and is converted to greyscale, normalisation is applied **after** conversion.


### 2.7 File Size Limit

| Limit | Value |
|---|---|
| Maximum file size | 50 MB |
| Recommended range | < 20 MB |

Files exceeding 50 MB are rejected at ingestion with an appropriate error (see [Edge Cases](#edge-cases)).

---

## Output Specification

### 3.1 Segmentation Mask

| Property | Value |
|---|---|
| Type | Binary (root vs. background) |
| Format | PNG - single channel, 8-bit |
| Resolution | Same as the input image |
| Pixel encoding | `255` = root · `0` = background |
| Confidence threshold applied | `≥ ask razvan` (raw U-Net logit → sigmoid → threshold) |
| One mask per | Input image (all 5 plant ROIs composited into a single output mask) |

**File naming convention:**

```
{original_filename}_mask.png
```

### 3.2 Root Tip Locations

A JSON array of detected root/plant tips, one entry per detected tip.

```json
[
  {
    "plant_id": 1,
    "tip_x": 412,
    "tip_y": 1803,
    "confidence": 0.94
  },
  {
    "plant_id": 2,
    "tip_x": 967,
    "tip_y": 1765,
    "confidence": 0.87
  }
]
```

| Field | Type | Description |
|---|---|---|
| `plant_id` | `int` | 1-indexed, left-to-right in image |
| `tip_x` | `int` | X coordinate in input image pixel space |
| `tip_y` | `int` | Y coordinate in input image pixel space |
| `confidence` | `float [0–1]` | Softmax confidence at the tip location |

Tip detection uses the **bottommost skeleton pixel** of the Dijkstra primary root path. If no tip is detected for a plant slot, that plant is omitted from the array (see [Edge Cases](#edge-cases)).

### 3.3 Root Length

| Field | Type | Description |
|---|---|---|
| `root_length_px` | `float` | Dijkstra path length in pixels 'or other' |
| `root_length_mm` | `float` | `root_length_px × (1 / scale_factor)` |

Scale factor: `150 / 2775` mm/px (see §2.3). If camera setup changes, this value **must** be updated in `config.yaml` - it is not hardcoded in the model itself as of Sprint 2.

### 3.4 Overall Confidence Score

A single scalar summarising inference quality for the entire image.

```json
{
  "overall_confidence": 0.91
}
```

| Property | Value |
|---|---|
| Scale | 0.0 – 1.0 |
| Computation | Mean of all per-tip confidence scores for detected plants |
| If no plants detected | `0.0` |
| Interpretation | `< 0.5` → flag for manual review |

### 3.5 Inference Metadata

Included in every response payload.

```json
{
  "model_version": "unet-npec-v1.2.0",
  "inference_timestamp": "2026-04-17T10:23:45Z",
  "processing_time_ms": 843,
  "patch_count": 40,
  "input_resolution": [2775, 2048],
  "input_channels": 1,
  "normalisation_applied": true,
  "scale_factor_mm_per_px": 0.05405
}
```

| Field | Type | Description |
|---|---|---|
| `model_version` | `string` | Semantic version of the deployed U-Net weights |
| `inference_timestamp` | `string` | ISO 8601 UTC |
| `processing_time_ms` | `int` | Wall-clock time from image ingestion to output ready |
| `patch_count` | `int` | Total patches processed |
| `input_resolution` | `[int, int]` | `[width, height]` of the received image |
| `input_channels` | `int` | Channels after conversion (always `1` at inference time) |
| `normalisation_applied` | `bool` | Whether z-score normalisation was applied |
| `scale_factor_mm_per_px` | `float` | Scale factor used for length conversion |

### 3.6 Full Response Envelope (JSON)

```json
{
  "status": "ok",
  "metadata": { ... },
  "segmentation_mask_url": "s3://npec-outputs/{job_id}_mask.png",
  "plants": [
    {
      "plant_id": 1,
      "tip_x": 412,
      "tip_y": 1803,
      "confidence": 0.94,
      "root_length_px": 1643.2,
      "root_length_mm": 88.7
    }
  ],
  "overall_confidence": 0.91,
  "warnings": []
}
```

---

## Normalisation

Full normalisation pipeline in order:

```
Raw file
  └─► Format validation (type · size · resolution)
        └─► Decode to numpy array (uint8)
              └─► Channel conversion (RGB → greyscale if needed)
                    └─► Divide by 255.0  →  [0.0, 1.0]
                          └─► Z-score: (x - mean) / std
                                └─► ROI crop (per plant, 5 slots)
                                      └─► Patch extraction (256×256, stride 128)
                                            └─► U-Net inference
```

---

## Edge Cases

All edge cases return a structured JSON error or warning. The pipeline **never silently fails**.

### 5.1 Empty / Blank Image

**Condition:** Image is all-black, all-white, or has zero variance (std ≈ 0).  
**Detection:** Computed before normalisation; std of pixel values < 1e-4.  
**Behaviour:**

```json
{
  "status": "error",
  "error_code": "EMPTY_IMAGE",
  "message": "Input image has no content (pixel variance ≈ 0). Check imaging system.",
  "plants": [],
  "overall_confidence": 0.0
}
```

---

### 5.2 No Plants Detected

**Condition:** All 5 plant ROIs produce masks with zero root pixels above the confidence threshold.  
**Behaviour:**

```json
{
  "status": "ok",
  "warnings": ["NO_PLANTS_DETECTED"],
  "plants": [],
  "overall_confidence": 0.0
}
```

HTTP status: `200` with warning. Flag for manual review in the monitoring dashboard.

---

### 5.3 Partial Detection (Fewer Than 5 Plants)

**Condition:** At least one but fewer than 5 ROI slots produce valid detections.  
**Behaviour:** Return only detected plants in the `plants` array. Missing slots are logged as warnings.

```json
{
  "status": "ok",
  "warnings": ["PLANT_NOT_DETECTED: plant_id=3", "PLANT_NOT_DETECTED: plant_id=5"],
  "plants": [
    { "plant_id": 1, ... },
    { "plant_id": 2, ... },
    { "plant_id": 4, ... }
  ],
  "overall_confidence": 0.76
}
```

---

### 5.4 Multiple / Overlapping Root Structures

**Condition:** Within a single ROI, the largest connected component is ambiguous (two components within 5 % of each other in pixel area).  
**Behaviour:** The larger component is selected (existing logic). A warning is appended:

```json
{
  "warnings": ["AMBIGUOUS_ROOT_STRUCTURE: plant_id=2, top_2_areas=[12450, 12201]"]
}
```

This case should be reviewed manually; consider flagging in the LIMS system.

---

### 5.5 Poor Lighting / Low Contrast

**Condition:** Mean pixel intensity of the full image < 20 or > 235 (on 0–255 scale), indicating under- or over-exposure.  
**Behaviour:**

```json
{
  "status": "ok",
  "warnings": ["LOW_CONTRAST: mean_intensity=14.2 - segmentation quality may be reduced"],
  "overall_confidence": 0.43
}
```

Inference proceeds but `overall_confidence` is expected to be low. Any result with `overall_confidence < 0.5` is automatically flagged for manual review.

---

### 5.6 Image Condensation Artefacts

**Condition:** Circular high-intensity blobs detected in pre-processing (blob detection, area range 200–5000 px, circularity > 0.85).  
**Behaviour:** Log warning; condensation removal pre-processing step is applied if enabled in `config.yaml` (`preprocessing.remove_condensation: true`).

```json
{
  "warnings": ["CONDENSATION_DETECTED: 3 blobs found - pre-processing applied"]
}
```

---

### 5.7 Corrupted / Unreadable File

**Condition:** File cannot be decoded by the image reader (truncated, wrong magic bytes, corrupted JPEG, etc.).  
**Behaviour:**

```json
{
  "status": "error",
  "error_code": "CORRUPT_FILE",
  "message": "File could not be decoded. Ensure the file is a valid PNG, TIFF, or JPEG.",
  "plants": [],
  "overall_confidence": 0.0
}
```

HTTP status: `422 Unprocessable Entity`.

---

### 5.8 File Too Large

**Condition:** File size exceeds 50 MB.  
**Behaviour:**

```json
{
  "status": "error",
  "error_code": "FILE_TOO_LARGE",
  "message": "File size 63.4 MB exceeds maximum allowed 50 MB.",
  "plants": [],
  "overall_confidence": 0.0
}
```

HTTP status: `413 Content Too Large`. Rejected **before** decoding.

---

### 5.9 Unsupported Resolution

**Condition:** Image width or height outside [512, 4096] px.  
**Behaviour:**

```json
{
  "status": "error",
  "error_code": "INVALID_RESOLUTION",
  "message": "Image resolution 300x200 is below the minimum of 512x512.",
  "plants": [],
  "overall_confidence": 0.0
}
```

HTTP status: `422 Unprocessable Entity`.

---

## Validation Contract

All inputs **must** pass the following checks in order before inference begins. Any failure returns an error immediately - inference does not start.

| # | Check | Pass condition | Error code |
|---|---|---|---|
| 1 | File extension | `.png` · `.tif` · `.tiff` · `.jpg` · `.jpeg` | `UNSUPPORTED_FORMAT` |
| 2 | File size | ≤ 50 MB | `FILE_TOO_LARGE` |
| 3 | File integrity | Decodable by PIL/OpenCV | `CORRUPT_FILE` |
| 4 | Resolution | Both dimensions in [512, 4096] | `INVALID_RESOLUTION` |
| 5 | Pixel variance | std > 1e-4 | `EMPTY_IMAGE` |
| 6 | Intensity range | Mean intensity in [20, 235] | Warning only (inference continues) |

---

## Version History

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0.0 | 2026-04-17 | Koen | Initial draft - Sprint 1 |

---

> **Placeholders to resolve before Sprint 2:**
> - [ ] Fill in Block B training mean and std for normalisation (§2.5)
> - [ ] Confirm native HADES resolution for scale factor verification (§2.3)
> - [ ] Decide whether JPEG should be fully rejected or retained with a warning