# Backend Requirements

This document outlines the missing backend implementations required to fully support the frontend features in the Library and Inference modules.

## 1. Dataset Image Upload ("Add Scans")

The frontend now has an "Add Scans" button and a drag-and-drop modal on the Dataset Detail page, but the endpoint does not exist.

*   **Endpoint:** `POST /api/v1/datasets/{dataset_id}/images` (or `/scans`)
*   **Payload:** `multipart/form-data` with `files: list[UploadFile] = File(...)`
*   **Logic:**
    1. Validate images.
    2. Run inference on each image (`inference_service.run()`) to satisfy the schema requirement that Annotations must link to a Prediction.
    3. Save artifacts (input, mask, tip photo) to storage.
    4. Persist a `Prediction` row.
    5. Persist an `Annotation` row linked to both the `prediction_id` and `dataset_id`.
    6. Increment the Dataset `count`.
*   **Risks:** Running synchronous inference on multiple uploaded images will likely cause HTTP timeouts. Implement batch limits or a background worker queue (e.g., Celery) to handle this safely.

## 2. Dataset Real Stats and Metadata

The Dataset Detail view in the frontend currently mocks several statistics and metadata fields (e.g., "Annotated", "Pending", "Plates", "Stage", "Treatment") because the backend `DatasetDetail` schema does not provide them.

*   **Endpoint to Update:** `GET /api/v1/datasets/{dataset_id}`
*   **Logic:** 
    *   The `DatasetRepository.get` method needs to dynamically calculate the number of "annotated" vs "pending" annotations linked to this dataset.
    *   Update the `DatasetDetail` Pydantic schema to include these computed statistics so the frontend can replace the hardcoded values (e.g., `count * 0.74`).

## 3. Dataset Scans Retrieval

The Dataset Detail view shows a "Scans" grid that currently renders 15 hardcoded placeholders because there is no endpoint to fetch the actual images/annotations belonging to a dataset.

*   **New Endpoint:** `GET /api/v1/datasets/{dataset_id}/annotations`
*   **Response:** Paginated list of `AnnotationResponse` objects.
*   **Frontend Impact:** The frontend will use this to populate the Scans grid with actual `input_url` and `mask_url` thumbnails.

## 4. Save Inference to Library

On the Inference Results view, there is a "Save to library" button that currently just shows a "Coming Soon" toast. 

*   **New Endpoint:** `POST /api/v1/predictions/{prediction_id}/save-to-library`
*   **Payload:** `{"dataset_id": "uuid"}`
*   **Logic:** Creates an `Annotation` row linking the existing `prediction_id` to the specified `dataset_id`. 
*   **Frontend Impact:** Allows users to process a single image in the Inference tab and then logically group it into a Dataset.

## 5. Exports Module

The frontend has an "Exports" tab in the Library that is currently a placeholder. Users will need a way to download a dataset and its masks for external training.

*   **New Endpoint:** `GET /api/v1/datasets/{dataset_id}/export`
*   **Logic:** 
    *   Fetch all annotations and corresponding images/masks for the dataset.
    *   Package them into a `.zip` file or generate a standard format (e.g., COCO JSON).
    *   Return the file stream or a signed URL to the generated archive.

## 6. Artifact URL Resolution

As identified previously, the backend needs to ensure that when returning `input_url` and `tip_photo_url` in schemas, the paths are fully resolvable by the frontend browser (whether via MinIO presigned URLs or a local static file proxy), rather than absolute `file://` paths which the browser blocks.

## 7. Import Dataset Flow

The frontend has a new **"Import Dataset"** button on the main Library page that triggers a multi-step import wizard.

*   **Endpoint:** `POST /api/v1/datasets/import`
*   **Method:** `POST`
*   **Content-Type:** `multipart/form-data`
*   **Parameters:**
    *   `file: UploadFile` (Required, ZIP file containing scans and optionally masks)
    *   `target_mode: str` (Required, `"new"` or `"existing"`)
    *   `dataset_id: UUID` (Optional, required if `target_mode` is `"existing"`)
    *   `name: str` (Optional, required if `target_mode` is `"new"`)
    *   `description: str` (Optional)
    *   `species: str` (Optional)
    *   `growth_stage: str` (Optional, `"seedling"` or `"mature"`)
*   **Logic:**
    1. Validate ZIP format and ensure file size does not exceed the safe limit (e.g. 200MB).
    2. Extract contents safely to a temporary directory in `/app/data/temp/`.
    3. Retrieve the target dataset or insert a new one if `target_mode` is `"new"`.
    4. Iterate over raw images:
        * Match with pre-existing ground truth masks if provided (using naming convention matching like `filename_mask.png` or a JSON catalog).
        * If no mask exists, run inference using `inference_service.run()`.
        * Save artifacts to the active storage backend.
        * Save `Prediction` and `Annotation` database records linked to the dataset.
    5. Clean up temporary directory and return processing details (number of imported scans, warnings for skipped invalid formats, and the target dataset ID).
*   **Recommended Async Pattern:**
    Because processing multiple files sequentially will timeout the standard request loop, it is highly recommended to:
    1. Immediately return a `202 Accepted` response with a `job_id` after receiving and verifying the file.
    2. Run the extraction, validation, and batch inference in a background thread or worker (e.g. Celery or FastAPI `BackgroundTasks`).
    3. Create a polling endpoint: `GET /api/v1/datasets/import/status/{job_id}` that returns the current progress (e.g. `"status": "processing"`, `"completed": 25`, `"total": 100`).

