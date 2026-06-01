# Root-Detection Endpoint — Integration Guide

Reference for any service (backend, frontend, CLI) that needs to call the
NPEC plant-root segmentation endpoint deployed on Azure ML.

The output schema follows [`docs/IO specifications.md`](../docs/IO%20specifications.md)
(IO spec v1.0.0); a couple of small deviations are listed at the end.

---

## 1. Endpoint coordinates

| Field | Value |
|---|---|
| Workspace | `team-CV5-2026` (resource group `buas-y2`, subscription `0a94de80-…`) |
| Endpoint name | `root-detection-endpoint` |
| Deployment name | `current` |
| Auth mode | `key` (static primary/secondary key) |
| HTTP method | `POST` |
| Content-Type | `application/json` |
| Request timeout | 180 s (deployment-side; image inference itself takes ~30–70 s) |
| Max payload | 50 MB (set in scoring code; Azure ML caps at ~100 MB) |
| Concurrency per replica | 1 (synchronous TF inference; queue up to 60 s) |

### Looking up scoring URI + auth key

You only need to do this **once** (store the result in your service config).
The URI changes if the endpoint is recreated; the key can be rotated.

**Python (recommended):**
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

client = MLClient(
    DefaultAzureCredential(),
    subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    resource_group_name="buas-y2",
    workspace_name="team-CV5-2026",
)
ep = client.online_endpoints.get("root-detection-endpoint")
keys = client.online_endpoints.get_keys("root-detection-endpoint")

scoring_uri = ep.scoring_uri          # e.g. https://...azureml.net/score
primary_key = keys.primary_key
```

**Azure CLI:**
```bash
az ml online-endpoint show \
   --name root-detection-endpoint --resource-group buas-y2 \
   --workspace-name team-CV5-2026 --query scoring_uri -o tsv

az ml online-endpoint get-credentials \
   --name root-detection-endpoint --resource-group buas-y2 \
   --workspace-name team-CV5-2026 --query primaryKey -o tsv
```

---

## 2. Request

### Headers

```
Authorization: Bearer <primary_key>
Content-Type:  application/json
```

> Azure ML's managed-identity-auth path uses an AAD token; this endpoint uses
> **key** auth, so a static Bearer token is fine.

### Body (JSON)

```json
{
  "image_b64": "<base64-encoded image file bytes>",
  "filename":  "img_001.png"
}
```

| Field | Required | Type | Notes |
|---|---|---|---|
| `image_b64` | yes | string | Base64 of the **whole file** (PNG / TIFF / JPEG). NOT raw pixels. |
| `filename` | no  | string | Used in response metadata + by the validator for extension check. Defaults to `"image.png"`. |

### Input validation (rejected before inference runs)

| # | Check | Pass condition | `error_code` on failure |
|---|---|---|---|
| 1 | File extension | `.png` `.tif` `.tiff` `.jpg` `.jpeg` | `UNSUPPORTED_FORMAT` |
| 2 | File size | ≤ 50 MB | `FILE_TOO_LARGE` |
| 3 | Decodable | OpenCV can decode the bytes | `CORRUPT_FILE` |
| 4 | Resolution | width ∈ [512, 5120], height ∈ [512, 4096] | `INVALID_RESOLUTION` |
| 5 | Pixel variance | std > 1e-4 | `EMPTY_IMAGE` |
| 6 | Intensity | mean ∈ [20, 235] | warning only (`LOW_CONTRAST`), inference still runs |

If `image_b64` is missing or not valid base64 → `INVALID_REQUEST`.
If the JSON body is malformed → `INVALID_REQUEST`.

---

## 3. Response

### 3.1 Success envelope

```json
{
  "status": "ok",
  "metadata": {
    "model_version":          "root-detection-v4",
    "inference_timestamp":    "2026-05-27T17:10:11.559638+00:00",
    "processing_time_ms":     49778,
    "patch_count":            54,
    "input_resolution":       [4202, 3006],
    "input_channels":         3,
    "normalisation_applied":  true,
    "scale_factor_mm_per_px": 0.00667
  },
  "segmentation_mask_b64": "<base64 PNG, single-channel 8-bit, 0=background 255=root>",
  "plants": [
    {
      "plant_id":       3,
      "tip_x":          2397,
      "tip_y":          2846,
      "confidence":     0.2581,
      "root_length_px": 2240.0,
      "root_length_mm": 14.933
    }
  ],
  "overall_confidence": 0.169,
  "warnings": [
    "PLANT_NOT_DETECTED: plant_id=1",
    "PLANT_NOT_DETECTED: plant_id=2",
    "LOW_OVERALL_CONFIDENCE: 0.169 — flag for manual review"
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `status` | `"ok"` \| `"error"` | Top-level outcome. |
| `metadata.model_version` | string | Identifies the deployed serving model (e.g. `root-detection-v4`). |
| `metadata.inference_timestamp` | string | ISO 8601 UTC, when the response was produced. |
| `metadata.processing_time_ms` | int | Wall-clock time from request decode to response build. |
| `metadata.patch_count` | int | How many 512×512 patches were processed. |
| `metadata.input_resolution` | `[w, h]` | Of the decoded input image. |
| `metadata.input_channels` | int | 1 (greyscale) or 3 (colour). |
| `metadata.normalisation_applied` | bool | Always `true` currently. |
| `metadata.scale_factor_mm_per_px` | float | mm-per-pixel used for the `root_length_mm` conversion (≈ `1/150`). |
| `segmentation_mask_b64` | string | **Base64 PNG** of the mask. Same H×W as input image, single-channel 8-bit, `255` = root, `0` = background. Spec §3.1 calls for an S3 URL — we inline the PNG instead so the response is self-contained. |
| `plants` | array | One entry per *detected* plant; missing slots show up in `warnings` instead. Sorted by `plant_id`. |
| `plants[].plant_id` | int | **1-indexed**, left-to-right (slot 1 is leftmost). |
| `plants[].tip_x`, `tip_y` | int | Root-tip pixel coords in the **input image** frame (not mask-relative). |
| `plants[].confidence` | float `[0,1]` | Mean root-probability in a 5-px window around the tip. |
| `plants[].root_length_px` | float | Dijkstra path length along the skeleton, in pixels. |
| `plants[].root_length_mm` | float | `root_length_px * scale_factor_mm_per_px`, rounded to 3 dp. |
| `overall_confidence` | float `[0,1]` | Mean of per-plant `confidence` values. `0.0` if no plants detected. |
| `warnings` | array of strings | Non-fatal notes — see table below. |

### 3.2 Warnings (success-path)

| Warning prefix | Meaning |
|---|---|
| `PLANT_NOT_DETECTED: plant_id=N` | Slot `N` (1-indexed) had no anchor or no valid root. |
| `NO_PLANTS_DETECTED` | All five slots empty. |
| `LOW_CONTRAST: mean_intensity=X` | Image fell outside [20, 235] mean intensity. |
| `LOW_OVERALL_CONFIDENCE: X` | `overall_confidence < 0.5`. Recommend manual review. |

### 3.3 Error envelope

```json
{
  "status":             "error",
  "error_code":         "INVALID_RESOLUTION",
  "message":            "Image resolution 300x200 is outside the allowed range ...",
  "plants":             [],
  "overall_confidence": 0.0
}
```

The endpoint **returns HTTP 200** in all cases (errors are encoded in the
`status` field). Inspect `status` and `error_code` rather than the HTTP code.

| `error_code` | When |
|---|---|
| `INVALID_REQUEST` | Malformed JSON, missing `image_b64`, or `image_b64` is not valid base64. |
| `UNSUPPORTED_FORMAT` | File extension not in {`.png`, `.tif`, `.tiff`, `.jpg`, `.jpeg`}. |
| `FILE_TOO_LARGE` | Decoded bytes > 50 MB. |
| `CORRUPT_FILE` | OpenCV could not decode the bytes. |
| `INVALID_RESOLUTION` | Width or height outside [512, 5120] × [512, 4096]. |
| `EMPTY_IMAGE` | Pixel std < 1e-4 (all-black, all-white, or single colour). |
| `INFERENCE_ERROR` | Unhandled exception inside `PlantMaster.process_image`. Message has the exception text — log it and retry; if persistent, check container logs. |
| `NON_JSON_RESPONSE` / `NON_OBJECT_RESPONSE` | *Test-side* errors raised by `scripts/test_endpoint.py` when the SDK double-wraps the body. Not produced by the endpoint itself. |

---

## 4. End-to-end example — Python

```python
import base64
import json
import requests

SCORING_URI = "https://root-detection-endpoint.westeurope.inference.ml.azure.com/score"
API_KEY     = "<primary_key>"

with open("plant.png", "rb") as fh:
    image_b64 = base64.b64encode(fh.read()).decode()

body = {"image_b64": image_b64, "filename": "plant.png"}

resp = requests.post(
    SCORING_URI,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    },
    data=json.dumps(body),
    timeout=200,
)
resp.raise_for_status()

result = resp.json()
# Azure ML SDK + raw HTTP both return a JSON string; the SDK additionally
# wraps it in a second JSON layer. With raw HTTP and Content-Type=application/json,
# resp.json() gives you the dict directly.

if result["status"] != "ok":
    raise RuntimeError(f"{result['error_code']}: {result['message']}")

print(f"Detected {len(result['plants'])} plants:")
for p in result["plants"]:
    print(f"  plant {p['plant_id']}: tip=({p['tip_x']},{p['tip_y']}) "
          f"length={p['root_length_mm']} mm  conf={p['confidence']}")

# Save the mask
with open("plant_mask.png", "wb") as fh:
    fh.write(base64.b64decode(result["segmentation_mask_b64"]))
```

## 5. End-to-end example — Node.js

```js
import fs from "fs/promises";

const SCORING_URI = "https://root-detection-endpoint.westeurope.inference.ml.azure.com/score";
const API_KEY     = process.env.NPEC_ENDPOINT_KEY;

const bytes = await fs.readFile("plant.png");
const body  = {
  image_b64: bytes.toString("base64"),
  filename:  "plant.png",
};

const r = await fetch(SCORING_URI, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${API_KEY}`,
    "Content-Type":  "application/json",
  },
  body: JSON.stringify(body),
});
if (!r.ok) throw new Error(`HTTP ${r.status}`);

const result = await r.json();
if (result.status !== "ok") {
  throw new Error(`${result.error_code}: ${result.message}`);
}

console.log(`Detected ${result.plants.length} plants`);
await fs.writeFile("plant_mask.png", Buffer.from(result.segmentation_mask_b64, "base64"));
```

## 6. End-to-end example — curl

```bash
# Encode an image to base64 + wrap as JSON. (jq makes the JSON properly escaped.)
jq -n --arg img "$(base64 -w0 plant.png)" \
      --arg name "plant.png" \
      '{image_b64: $img, filename: $name}' > body.json

curl -s -X POST "$SCORING_URI" \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     --data @body.json | jq .
```

---

## 7. Latency, retries, and operational notes

- **First request after a deployment update** takes longer (~90 s) because
  the container has to load the Keras models. Subsequent requests are
  ~30–70 s per image on the current CPU SKU (`Standard_DS3_v2`).
- The endpoint is **single-instance, single-concurrent** (1 worker). If you
  send N requests in parallel, they queue up to 60 s; beyond that you get
  HTTP 429. Plan for sequential calls or scale up the deployment.
- Idempotency: same input → same output (no randomness in inference).
- Safe to retry on HTTP 5xx / timeout. Do **not** retry on a successful
  response with `status: "error"` — fix the input instead.
- The endpoint always returns **HTTP 200**, errors are surfaced via the
  `status` and `error_code` fields. Watch your alerts accordingly.

---

## 8. Differences from the IO specification

The IO spec (`docs/IO specifications.md`) was drafted before the Azure
deployment, so a couple of details differ in implementation:

| Spec § | Spec says | We actually return |
|---|---|---|
| §3.1 | `segmentation_mask_url` pointing to S3 | `segmentation_mask_b64` (inline base64 PNG) — easier to consume, no extra GET, no S3 dependency. |
| §2.3 | Max resolution `4096 × 4096` | We allow up to **5120 × 4096** (width × height) to fit HADES uncropped images. |
| §3.5 | `model_version` like `"unet-npec-v1.2.0"` | We use the registry-version naming, e.g. `"root-detection-v4"`. |

Everything else (warnings, error codes, field names, edge-case behaviour)
matches the spec.

---

## 9. Things to monitor / FYI

- **`overall_confidence < 0.5`** in the response always triggers a
  `LOW_OVERALL_CONFIDENCE` warning. Backend should surface that to the user
  as "needs review".
- **`PLANT_NOT_DETECTED` warnings** are common — the model doesn't always
  fire on all 5 plant slots. Don't treat a missing slot as an error.
- The model version visible in `metadata.model_version` is the *root* model
  version. The shoot/anchor model is fixed at `shoot-detection:1`.
- For redeployment / new model versions, talk to the CV team. The endpoint
  contract (this document) is stable; only `metadata.model_version` should
  change between deployments.

---

_Maintainer: NPEC CV team (CV5-2026). Last updated 2026-05-27._
