# Dataset versioning and naming strategy

> **Status:** proposal — to be agreed before Sprint 3
> **Scope:** naming and versioning of Azure ML Data Assets produced from the
> HADES/NPEC Block B dataset.

## 1. Goals

- Any team member can find the right version of the data just from its name
  and tags.
- Every training run can be traced back to an exact, immutable data version.
- Adding more raw data over time is easy and doesn't churn the asset name.

## 2. Naming convention

### 2.1 Data asset name

The Azure ML asset is a 1:1 mirror of the raw dataset. There is **one
asset**, holding everything in scope, in its raw form:

```
cv5_hades_raw
```

| Component | Meaning |
|---|---|
| `cv5` | project (Block B plant segmentation) |
| `hades` | source (HADES/NPEC delivery) |
| `raw` | unmodified — files are uploaded as-is |

Rules:
- lowercase, underscores between words, no spaces, no dates
- the name is **permanent**; new data ⇒ new version, not a new name

There is no per-class asset (e.g. no `cv5_hades_root_*`) because filtering
is now done at training time. If a future deliverable produces transformed
data (cropped, patchified, embedded), that would be a separate asset with
its own name (e.g. `cv5_hades_cropped_512`).

### 2.2 File names inside the asset

The asset preserves the original folder structure exactly:

```
cv5_hades_raw/
├── Y2B_23/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── masks/
├── Y2B_24/
│   ├── images/
│   └── masks/<annotator>/
└── Y2B_25/
    ├── images/
    └── masks/<annotator>/
```

Filenames are **untouched** from the raw delivery. Spaces in filenames
(`Fish Eye Corrected`) are preserved; consumers must handle them.

The `Kaggle/` folder and the lone `.txt` file are **not** uploaded.

## 3. Versioning rules

Uses Azure ML's built-in asset versioning (`version="1.2.0"` etc.):

| Change type | Version bump | Example |
|---|---|---|
| Raw dataset modified upstream (re-export, schema change, field added) | **Major** (X.0.0) | 1.4.2 → 2.0.0 |
| Added images or masks (e.g. Y2B_26 cohort, more annotations completed) | **Minor** (x.Y.0) | 1.4.2 → 1.5.0 |
| Removed corrupted files, fixed misnamed files, metadata-only fix | **Patch** (x.y.Z) | 1.4.2 → 1.4.3 |
| Inclusion rules changed (e.g. Kaggle brought in scope) | **Major** | 1.x.x → 2.0.0 |

Versions are **never overwritten**. Bad upload ⇒ bump patch, document the
fix in the asset description.

### Why no `preprocessing_version` here

Earlier drafts had a `preprocessing_version` tag. That's gone — the asset
contains no preprocessing, so there's nothing to version. Preprocessing
decisions live in training-pipeline configs and are versioned there (git
SHA, MLflow run params).

## 4. Tags

Every registered version of the asset carries:

| Tag key | Values | Required? |
|---|---|---|
| `source` | `hades` | yes |
| `cohorts` | comma-separated, e.g. `Y2B_23,Y2B_24,Y2B_25` | yes |
| `is_raw` | `true` | yes |
| `num_images` | integer | yes |
| `num_masks` | integer | yes |
| `created_by` | GitHub / ADO username | yes |
| `created_date` | `YYYY-MM-DD` | yes |
| `git_commit` | short SHA of the upload-script code | yes |
| `excluded_paths` | comma-separated, e.g. `Kaggle/,*.txt` | yes |
| `notes` | free text | optional |

### Example: `cv5_hades_raw` v1.0.0

```yaml
source: hades
cohorts: Y2B_23,Y2B_24,Y2B_25
is_raw: true
num_images: 685
num_masks: 1592
created_by: <github-username>
created_date: YYYY-MM-DD
git_commit: <short-sha>
excluded_paths: Kaggle/,*.txt
notes: "1:1 mirror of the raw delivery. Training-time filtering and preprocessing applied by training code; see docs/data/preprocessing_spec.md."
```

## 5. Description field

The asset description must include:

1. One-line summary: "Raw HADES/NPEC plant segmentation dataset, mirrored
   from on-disk delivery."
2. Source pointer (raw dataset location + delivery date).
3. What was excluded and why.
4. Link to `docs/data/data_inventory.md` for content details.
5. Link to `docs/data/preprocessing_spec.md` for the consumer contract.

## 6. Review

Reviewed and agreed: YYYY-MM-DD
