## API Data Models & Schemas
**Version:** 0.1  
**Author:** Erik Hrina (230395)  
**Refer to:** `docs/api_spec.yaml`

---
### 1. Conventions
- All request and response bodies are `application/json` unless stated otherwise.
- File upload endpoints use `multipart/form-data`.
- All timestamps are ISO 8601 UTC strings: `"2026-04-17T10:28:42Z"`.
- Required fields are marked R. Optional fields are marked O.
- String enums list all accepted values explicitly.
- Constraints (min, max, pattern) are listed in the Notes column.

---
### 2. Auth
#### 2.1 `AuthRequest`
**Endpoint:** `POST /auth/token`  
**Content-Type:** `application/x-www-form-urlencoded` (OAuth2 Password flow standard)

| Field | Type | R/O | Constraints | Notes |
|-------|------|-----|-------------|-------|
| `username` | `string` | R | max 254 chars | User's email address |
| `password` | `string` | R | max 128 chars | Plaintext over HTTPS; hashed server-side |
| `grant_type` | `string` | R | enum: `"password"` | Fixed value for Password flow |
---

**Example request:**
```
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

username=dr.hrina%40npec.nl&password=secret&grant_type=password
```

---
#### 2.2 `AuthResponse`
**HTTP 200 - success**

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `access_token` | `string` | R | Signed JWT |
| `token_type` | `string` | R | Always `"bearer"` |
| `expires_in` | `integer` | R | Seconds until expiry. Default `3600` (1 hour) |
| `refresh_token` | `string` | R | Opaque token used to obtain a new access token without re-entering credentials. Long-live - deafault expiry: 1 day |
| `role` | `string` | R | enum: `"user"`, `"admin"` - used by the frontend to show/hide admin UI elements |
---

**Example response:**
```json
{
    "access_token": "randomtoken",
    "token_type": "bearer",
    "expires_in": 3600,
    "refresh_token": "refreshtoken",
    "role": "user"
}
```

**JWT payload (informational - not returned directly):**
| Claimn | Type | Notes |
|--------|------|-------|
| `sub` | `string` | User ID |
| `email` | `string` | User's email |
| `role` | `string` | "user" or "admin" |
| `exp` | `integer` | Unix timestamp of expiry |
| `iat` | `integer` | Unix timestamp of issue |
---

**HTTP 401 - invalid credentials:** See ErrorResponse.


---
#### 2.3 `LogoutRequest`
**Endpoint:** `POST /auth/logout`  
**Content-Type:** none (token is passed via `Authorization: Bearer <token>` header)

No request body. The server invalidates the token from the Authorization header.

**HTTP 200 - success:**
```json
{
    "message": "Logged out successfully."
}
```
---
#### 2.4 `TokenRefreshRequest`
**Endpoint:** `POST /auth/refresh`  
**Content-Type:** `application/json`
**Auth required:** No (the refresh token itself is the credential)

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `refresh_token` | `string` | R | The refresh token received from `POST /auth/token` or a previous refresh |
---

**HTTP 200 - success:** Returns a new `AuthResponse` object (see 2.2) with a fresh `access_token`, updated `expires_in`, and a rotated `refresh_token`. The old refresh token is invalidated immediately on use.

**HTTP 401 - expired or invalid token:** See ErrorResponse. The user must log in again via `POST /auth/token`.

---
### 3. Health
#### 3.1 `HealthResponse`
**Endpoint:** `GET /health`  
**Auth required:** No

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `status` | `string` | R | enum: `"ok"`, `"degraded"`, `"unavailable"` |
| `version` | `string` | R | API version string, e.g. `"1.0.0"` |
| `timestamp` | `string` | R | ISO 8601 UTC - time of the health check |
---

**Example response:**
```json
{
    "status": "ok",
    "version": "1.0.0",
    "timestamp": "2026-04-17T10:48:42Z"
}
```

---
### 4. Prediction
#### 4.1 `PredictionRequest`
**Endpoint:** `POST /predict`  
**Content-Type:** `multipart/form-data`

| Field | Type | R/O | Constraints | Notes |
|-------|------|-----|-------------|-------|
| `file` | `file` | R | See format and size rules below | The root scan image |
| `confidence_threshold` | `float` | O | 0.50-0.99, default: `0.50` | Minimum confidence for a prediction to be accepted |
---

**File constraints:**

| Rule | Value|
|------|------|
| Accepted formats | `.png`, `.tiff`, `.tif`, `.jpg`, `.jpeg`
| Maximum file size | 50 MB — rejected before decoding (HTTP  |413)
| Minimum resolution | 512 × 512 px (HTTP 422 if violated) |
| Maximum resolution | 4096 × 4096 px (HTTP 422 if violated) |
| Multi-page TIFF | Rejected with HTTP 422 |
| JPEG | Accepted with a warning — compression artefacts may degrade segmentation quality |
| Channel modes | Greyscale (1ch), RGB (3ch), RGBA (4ch, alpha stripped). More than 4 channels rejected. |
---

**Example request:**
```
POST /api/v1/predict
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <binary image data>
confidence_threshold: 0.75
```

---
#### 4.2 `PredictionResponse`
**Endpoint:** `GET /predictions/{prediction_id}`  
**HTTP 200 - success**


| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `prediction_id` | `string` | R | UUID of the prediction |
| `status` | `string` | R | enum: `"ok"`, `"error"` |
| `metadata` | `PredictionMetadata` | R | See 4.2.1 |
| `segmentation_mask_url` | `string` | O | Presigned URL to the output mask PNG. Null if inference failed |
| `plants` | `Plant[]` | R | Array of per-plant results. Empty array if no plants detected |
| `overall_confidence` | `float` | R | 0.0-1.0. Mean of per-plant confidence scores. `0.0` if no plants detected |
| `warnings` | `string[]` | R | List of warning codes. Empty array if none. See 4.3 |
---
#### 4.2.1 `PredictionMetadata`

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `model_version` | `string` | R | Semantic version of deployed model, e.g. `"unet-npec-v1.2.0"` |
| `inference_timestamp` | `string` | R | ISO 8601 UTC |
| `processing_time_ms` | `integer` | R | Wall-clock time from image receipt to result ready |
| `input_resolution` | `[integer, integer]` | R | `[width, height]` of the received image in pixels |
| `input_channels` | `integer` | R | Channel count after conversion - always `1` at inference time |
| `normalisation_applied` | `boolean` | R | Whether z-score normalisation was applied |
| `scale_factor_mm_per_px` | `float` | R | Scale factor used for root length conversion |
---
#### 4.2.2 `Plant`
One entry per detected plant. Plants with no detected root are omitted from the array.

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `plant_id` | `string` | R | 1-indexed, left-to-right across the image |
| `tip_x` | `integer` | R | X coordinate of root tip in input image pixel space |
| `tip_y` | `integer` | R | Y coordinate of root tip in input image pixel space |
| `confidence` | `float` | R | 0.0-1.0. Confidence at the tip location |
| `root_length_px` | `float` | R | Dijkstra path length in pixels |
| `root_length_mm` | `float` | R | `root_length_px * scale_factor_mm_per_px` |
---

**Example response:*
```json
{
  "prediction_id": "a3f1c2d4-...",
  "status": "ok",
  "metadata": {
    "model_version": "unet-npec-v1.2.0",
    "inference_timestamp": "2026-04-17T10:23:45Z",
    "processing_time_ms": 843,
    "input_resolution": [2775, 2048],
    "input_channels": 1,
    "normalisation_applied": true,
    "scale_factor_mm_per_px": 0.05405
  },
  "segmentation_mask_url": "https://storage.example.com/masks/a3f1c2d4_mask.png?token=...",
  "plants": [
    {
      "plant_id": 1,
      "tip_x": 412,
      "tip_y": 1803,
      "confidence": 0.94,
      "root_length_px": 1643.2,
      "root_length_mm": 88.7
    },
    {
      "plant_id": 2,
      "tip_x": 967,
      "tip_y": 1765,
      "confidence": 0.87,
      "root_length_px": 1501.4,
      "root_length_mm": 81.1
    }
  ],
  "overall_confidence": 0.91,
  "warnings": []
}
```

---
#### 4.3 Prediction Warning Codes
Warnings appear in the `warnings` array when inference proceeds but the result may be unreliable. They do not change the HTTP status (still `200`).

| Code | Meaning |
|------|---------|
| `NO_PLANTS_DETECTED` | All 5 ROI slots produced no root pixels above the confidence threshold |
| `PLANT_NOT_DETECTED: plant_id=N` | One or more specific plant slots produced no detection |
| `AMBIGUOUS_ROOT_STRUCTURE: plant_id=N` | Two root components within 5% of each other in pixel area - larger was selected |
| `LOW_CONTRAST` | Mean pixel intensity outside [20, 235] - segmentation quality may be reduced |
| `CONDENSATION_DETECTED` | Circular blobs detected; pre-processing applied if enabled in config |
| `JPEG_COMPRESSION` | Input was JPEG - compression artefacts may affect accuracy |
---
#### 4.4 `ConfirmPredictionRequest`
**Endpoint:** `POST /predictions/{prediction_id}/confirm`  
**Content-Type:** `application/json`

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `scan_id` | `string` | R | UUID of the scan this prediction belongs to. Links the result into the library |
---

**HTTP 200 - success:**
```json
{
    "message": "Prediction confirmed and saved.",
    "prediction_id": "predictionid"
}
```
---

### 5. Annotation
#### 5.1 `AnnotationResponse`
**Endpoint:** `GET /annotations/{annotation_id}`  
**HTTP 200 - success**

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `annotation_id` | `string` | R | UUID |
| `scan_id` | `string` | R | UUID of the scan this annotation belongs to |
| `status` | `string` | R | enum: `"pending"`, `"annotated"`, `"review_required"` |
| `mask_url` | `string` | O | Presigned URL to the `.tiff` annotation mask. Null if not yet annotated |
| `species` | `string` | R | e.g. `"Arabidopsis thaliana"` |
| `age_days` | `integer` | O | Age of the plant in days at time of scan |
| `treatment` | `string` | O | e.g. `"Control-H20"` |
| `created_at` | `string` | R | ISO 8601 UTC |
| `updated_at` | `string` | R | ISO 8601 UTC |
---
#### 5.2 `AnnotationConfirmRequest`
**Endpoint:** `POST /annotations/{annotation_id}/confirm`  
**Content-Type:** `multipart/form-data`

| Field | Type | R/O | Constraints | Notes |
|-------|------|-----|-------------|-------|
| `mask` | `file` | R | `.tiff` only, single-frame | The corrected segmentation mask. This becomes ground truth for retraining |

**HTTP 200 - success:** Returns the updated `AnnotationResponse` object (see 5.1).

---
### 6. Feedback
#### 6.1 `FeedbackRequest`
**Endpoint:** `POST /feedback`  
**Content-Type:** `application/json`

| Field | Type | R/O | Constraints | Notes |
|-------|------|-----|-------------|-------|
| `prediction_id` | `string` | R | | UUID |
| `signal` | `string` | R | enum: `"correct"`, `"incorrect"` | Thumbs up / thumbs down |
| `comment` | `string` | O | max 500 chars | Optional free-text note from the user |
---

**Example request:**
```json
{
    "prediction_id": "predictionid",
    "signal": "incorrect",
    "comment": "Plant 3 tip location is off - root appears shorter than measured."
}
```
**HTTP 200 - success:**
```json
{
    "message": "Feedback recorded. Thank you."
}
```

---
### 7. Error Responses
#### 7.1 `ErrorResponse`

| Field | Type | R/O | Notes |
|-------|------|-----|-------|
| `status` | `string` | R | Always `"error"` |
| `error_code` | `string` | R | Machine-readable code. See 7.2 |
| `message` | `string` | R | Human-readable description of the error |
| `detail` | `string` | O | Additional structured context (e.g. which field failed validation) |
---

**Example:**
```json
{
  "status": "error",
  "error_code": "FILE_TOO_LARGE",
  "message": "File size 63.4 MB exceeds the maximum allowed 50 MB.",
  "detail": {
    "received_bytes": 66498765,
    "max_bytes": 52428800
  }
}
```

---
#### 7.2 Error Code Reference

| HTTP Status | Error Code | Trigger |
|-------------|------------|---------|
| `400` | `INVALID_REQUEST` | Malformed JSON body or missing required field |
| `401` | `UNAUTHORIZED` | Missing or invalid JWT token |
| `403` | `FORBIDDEN` | Valid token but insufficient role |
| `404` | `NOT_FOUND` | Resource ID does not exist |
| `409` | `CONFLICT` | Action not allowed in current state (e.g. confirming an already-confirmed prediction) |
| `413` | `FILE_TOO_LARGE` | Uploaded file exceeds 50 MB |
| `415` | `UNSUPPORTED_FORMAT` | File extension not in accepted list |
| `422` | `INVALID_RESOLUTION` | Image dimensions outside [512, 4096] px |
| `422` | `CORRUPT_FILE` | File cannot be decoded by the image reader |
| `422` | `MULTI_PAGE_TIFF` | Uploaded TIFF contains more than one frame |
| `422` | `EMPTY_IMAGE` | Image has zero pixel variance (blank or all-white scan) |
| `422` | `INVALID_CHANNELS` | Image has more than 4 channels |
| `500` | `INTERNAL_ERROR` | Unexpected server error - should never reach the user in production |
---