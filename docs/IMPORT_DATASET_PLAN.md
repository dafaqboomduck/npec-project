# Implementation Plan: Import Dataset Flow

This document details the architecture, user flow, API design, and risks for implementing a new **Import Dataset Flow**. This feature allows users to upload a single `.zip` archive containing root scan images (and optional pre-existing annotation masks), processing them to create a new dataset or populate an existing one.

---

## 1. User Experience & Frontend Flow (UI/UX)

We will integrate the import flow into the main **Library page** (under the "Datasets" tab).

### **UI Components**
1.  **Import Button:** 
    *   Add an **"Import Dataset"** button next to the existing **"+ New Dataset"** button.
2.  **Import Wizard Modal:** A 3-step modal/dialog to guide the user:
    *   **Step 1: Upload ZIP Archive**
        *   A dedicated file dropzone that accepts only `.zip` files.
        *   Shows basic file info (file name, size) upon drop/select.
    *   **Step 2: Configuration & Target**
        *   **Target Selector:** Radio buttons for:
            *   *Create New Dataset:* Displays text input fields for `Name` (required), `Description`, `Species`, and a dropdown for `Growth Stage`.
            *   *Add to Existing Dataset:* Displays a searchable dropdown listing all existing datasets.
    *   **Step 3: Upload, Progress & Validation Report**
        *   Displays an active upload progress bar (0-100%).
        *   After upload completes, changes status to **"Processing & Extracting..."** (shows loading spinner).
        *   **Real-time Logs / Steps:** Displays checked items (e.g., `✓ Extracting ZIP`, `✓ Validating image formats`, `Processing images...`).
        *   **Validation Errors / Warnings Panel:** If files are skipped or invalid, show a list of warnings (e.g., `⚠️ skipped non-image file 'resume.pdf'`, `⚠️ corrupt image 'root_04.png'`).
        *   **Success Screen:** Displays a summary (e.g., `Successfully imported 42 root scans to dataset 'Arabidopsis Test'`) and an "Open Dataset" button that takes the user directly to the new dataset detail page.

---

## 2. Backend Architecture & API Design

A new dedicated endpoint will be added to the Library router.

### **API Signature**
*   **Route:** `POST /api/v1/datasets/import`
*   **Content-Type:** `multipart/form-data`
*   **Parameters (Form Fields):**
    *   `file: UploadFile` (The zip file)
    *   `target_mode: str` (`"new"` or `"existing"`)
    *   `dataset_id: UUID?` (Required if `target_mode` is `"existing"`)
    *   `name: str?`, `description: str?`, `species: str?`, `growth_stage: str?` (Required if `target_mode` is `"new"`)
*   **Response Model:**
    ```json
    {
      "dataset_id": "uuid",
      "dataset_name": "Arabidopsis Test",
      "scans_imported": 42,
      "warnings": [
        "Skipped non-image file: folder/notes.txt",
        "Invalid format for: folder/photo.bmp"
      ],
      "errors": []
    }
    ```

### **Processing Sequence (Backend)**
1.  **Archive Extraction:** Save the zip to a temporary directory in `/app/data/temp/` and extract it safely.
2.  **Structure Parsing & Validation:**
    *   Scan extracted contents to separate **raw images** (PNG, JPG, TIFF) and potential **annotation masks** (e.g., images matching a suffix like `*_mask.png` or a standard metadata catalog).
    *   Identify non-image or unsupported formats and add them to the `warnings` list.
3.  **Database/Dataset Init:** Resolve the target dataset (retrieve the existing one or insert a new `Dataset` row).
4.  **Batch Processing Pipeline:**
    *   For each raw image:
        *   *Inference:* If the ZIP contains a matching human-made mask, use it directly. Otherwise, run the image through `inference_service.run()` to generate the baseline prediction mask.
        *   *Artifact Storage:* Save input images, masks, and tip photos to MinIO or LocalStorage.
        *   *DB Save:* Insert the `Prediction` record and link an `Annotation` record (status = `"pending"` if auto-inferred, status = `"annotated"` if human masks were supplied in the zip) to the `dataset_id`.
5.  **Cleanup:** Recursively delete temporary extraction directories.
6.  **Update Count:** Recalculate or increment the target dataset's total count.

---

## ⚠️ Potential Risks & Advanced Mitigations

### 1. The ZIP Bomb & Size Limits (High Risk)
*   **Risk:** A user uploads a 2GB zip or a nested ZIP bomb designed to exhaust disk space or memory.
*   **Mitigations:**
    *   Implement a strict size limit on the API endpoint (e.g., max 200MB zip).
    *   Validate the uncompressed size *before* fully extracting by checking the ZIP's central directory metadata headers. Reject if uncompressed size exceeds a safe threshold (e.g., 500MB).

### 2. Extreme Timeout on Inference (Critical Risk)
*   **Risk:** Running U-Net inference sequentially on 100 images inside a standard request loop will take 2-3 minutes. The HTTP request will definitely timeout.
*   **Mitigations:**
    *   **Asynchronous Job Pattern:**
        1. The `POST /api/v1/datasets/import` endpoint immediately extracts, registers a background job/task, and returns a `202 Accepted` along with a `job_id`.
        2. Create a status endpoint: `GET /api/v1/datasets/import/status/{job_id}`.
        3. The frontend polls this status endpoint to show progress (e.g., `"processing": 24/100 completed`).
        4. Alternatively, use FastAPI `BackgroundTasks` for quick, in-process async running if a full queue like Celery is not set up yet.

### 3. File Matching Logic Complexity
*   **Risk:** If the user uploads pre-existing masks, matching them to their raw images (e.g., pairing `img_01.png` with `img_01_mask.png` or matching via a JSON manifest) is error-prone.
*   **Mitigations:**
    *   Enforce a simple, standardized directory structure inside the ZIP (e.g., `/images` and `/masks` folders, where filenames must match exactly).
    *   Document the required archive format directly inside the frontend's Step 1 file picker to guide the user.
