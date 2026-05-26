# Backend Requirements: Mask Correction & Advanced Feedback

This document outlines the necessary backend changes to support the interactive mask correction feature implemented in the frontend.

## 1. Schema Changes (`api/src/schemas/feedback.py`)

Currently, `FeedbackRequest` only accepts a `signal` and `comment`. It needs to support binary mask uploads.

```python
# Change from JSON-only to Form-compatible schema
class FeedbackRequest(BaseModel):
    prediction_id: UUID
    signal: FeedbackSignal
    comment: Optional[str] = None
    # No file field here, FastAPI handles UploadFile separately in the router
```

## 2. API Endpoints (`api/src/routers/feedback.py`)

The `submit_feedback` endpoint should be updated to handle `multipart/form-data`.

### Changes needed:

- Change argument from `body: FeedbackRequest` to `prediction_id: UUID = Form(...)`, `signal: str = Form(...)`, etc.
- Add `mask: Optional[UploadFile] = File(None)`.
- **Logic**:
  1.  If `mask` is provided:
      - Validate it's a valid image (resolution matches prediction).
      - Save to `/app/data/corrected_masks/{prediction_id}_corrected.png`.
      - Store the path in the `Feedback` record.
  2.  Update/Create an **Annotation** record:
      - Set `status = 'review_required'`.
      - Set `mask_uri` to the new corrected mask path.

## 3. Database Updates (`api/src/db/orm.py`)

- Add `corrected_mask_uri: Mapped[Optional[str]]` to the `Feedback` model.
- Ensure the relationship between `Feedback` and `Annotation` is clear if we want to promote corrections to annotations automatically.

## 4. Artifact Storage (`api/src/services/artifact_storage.py`)

Add a helper function:

```python
def save_corrected_mask(prediction_id: UUID, mask_bytes: bytes) -> str:
    # Save to /app/data/corrected_masks/
    # Return relative path
```

## 5. Rationale

Allowing users to correct masks directly in the browser creates a high-quality feedback loop. These "corrected" masks are the most valuable data for retraining models to handle edge cases where the initial U-Net segmentation failed.
