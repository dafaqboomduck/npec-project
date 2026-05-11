# Preprocessing specification — HADES/NPEC root segmentation

> **Status:** draft
> **Scope:** data pipeline only. Model training, augmentation, loss
> functions, and architecture are out of scope.
> **Filename note:** kept as `preprocessing_spec.md` to match ADO ticket 1417
> and the rest of the repo. The doc now covers two things: (1) what the
> upload script does, and (2) what the training pipeline is expected to do
> with the asset. The latter is not enforced by the upload pipeline; it is
> documented so consumers know the contract.

## 1. Scope and design decision

- **The Azure ML data asset is a 1:1 mirror of the raw dataset.** No crop,
  no resize, no alpha-channel removal, no format conversion, no train/val/
  test split assignment. What's on disk goes to Azure unchanged.
- **All transformations happen at training time.** The training pipeline
  loads the raw asset, applies preprocessing, splits the data, patchifies,
  and feeds the model.
- **Why this design:** the upload runs once; preprocessing decisions evolve
  often. Putting transforms in training code keeps the asset stable and
  lets multiple training runs experiment with different preprocessing
  without re-uploading 24.5 GB.

## 2. Pipeline (end-to-end)

```
raw 4K RGBA PNG image            raw mask TIFFs (root/shoot/seed/occluded_root)
         │                                     │
         └──────────────┬──────────────────────┘
                        │   (uploaded as-is, no transformation)
                        ▼
              Azure ML data asset
                        │
                        ▼  (loaded by training pipeline)
        ┌───────────────┴───────────────┐
        ▼                               ▼
  filter to _root_mask.tif        drop alpha (RGBA → RGB)
        │                               │
        └──────────────┬────────────────┘
                       ▼
            petri-dish crop (OpenCV)
            via src/npec_cv/helpers/extract_petri_dish.py
                       │
                       ▼
              ~2K aligned image + mask
                       │
                       ▼
             train/val/test split
                       │
                       ▼
            patchify to 512×512
            via src/npec_cv/helpers/patchify.py
                       │
                       ▼
              mask patches → model (binary segmentation)
              image patches → overlay at validation
```

Everything **above** the dotted line ("uploaded as-is") is the data
pipeline's responsibility. Everything **below** is the training pipeline.

## 3. What the upload script does

1. Walk the raw dataset root.
2. Upload every file under `Y2B_23/`, `Y2B_24/`, and `Y2B_25/` to the
   workspace blob datastore, preserving relative paths.
3. Skip the `Kaggle/` folder (out of scope, no masks).
4. Skip the lone `.txt` file unless it turns out to contain something useful
   (decision deferred until its contents are known).
5. Register the uploaded data as an Azure ML data asset following
   `versioning_strategy.md`.

The script does **not**:
- filter masks by class
- assign train/val/test splits
- crop, resize, or change pixel data in any way
- write a manifest CSV (training code derives what it needs at runtime)

## 4. What the training pipeline is expected to do

This section is documentation, not an enforceable contract. It describes
how the asset is intended to be consumed so anyone writing training code
knows the assumptions.

### 4.1 Filter to root masks

Keep only files matching `_root_mask.tif`. Drop `_shoot_mask.tif`,
`_seed_mask.tif`, and `_occluded_root_mask.tif`. After filtering: **492
image/root-mask pairs** (115 Y2B_23 + 314 Y2B_24 + 63 Y2B_25). Two Y2B_24
images have no root mask and are silently excluded.

### 4.2 Drop alpha channel

Raw images are RGBA `uint8`. Drop the alpha channel before any further
processing; the model takes RGB. Alpha is assumed uninformative — verify
with a quick check on first run if not already known.

### 4.3 Petri-dish crop

Use `src/npec_cv/helpers/extract_petri_dish.py`. Run on the image, then
apply the same `(x, y, w, h)` crop to the mask using **nearest-neighbour
sampling** (not bilinear — interpolation invents class values). Output is
~2K-resolution image + mask, pixel-aligned.

### 4.4 Train/val/test split

- **70 / 15 / 15** by image (not by patch).
- Pool all three cohorts together. Y2B_23's pre-existing train/test split
  is discarded.
- **Seed `42`** so the split is reproducible across training runs.
- Split happens at training time. Two training runs with the same seed
  must produce the same split.

### 4.5 Patchify

Use `src/npec_cv/helpers/patchify.py`. Window 512 × 512.
- Stride is the training pipeline's call. Either `512` (non-overlapping)
  or `256` (50% overlap) are reasonable starting points.
- Whether to drop all-background patches is also a training decision (class
  balance, wall-clock cost).

## 5. Mask values

After the upload, masks on Azure are TIFF `uint8` with values `{0, 1}` —
binary, no further encoding. Training code that loads them should assert
this and fail loudly if a mask contains anything else (the inventory found
one anomalous 3-D mask; the assertion will catch any others).

## 6. Integrity checks (training-time, recommended)

The training code should, per image/mask pair:

- assert image and mask have the same spatial shape **after** the crop step
- assert mask values ⊆ `{0, 1}`
- assert no NaN, no all-zero images
- log and skip pairs that fail any of the above

Failure handling is the training pipeline's choice. The upload pipeline
does not validate.

## 7. What invalidates the asset and forces a new version

Per `versioning_strategy.md` §3, the asset gets a **major** version bump if:
- the raw dataset on disk gains, loses, or modifies files
- the inclusion rules in §3 change (e.g. Kaggle is brought in scope)

Changes to the **training-time** preprocessing in §4 do **not** affect the
data asset version, because none of those transforms are baked into the
asset. They are recorded by the training pipeline (run config, MLflow
tags, etc.).

## 8. Review

Reviewed: YYYY-MM-DD
