# Preprocessing specification — HADES/NPEC root segmentation

> **Status:** draft (revised)
> **Supersedes:** the 1:1-mirror upload approach previously documented here.
> **Scope:** what the data-upload pipeline does, and the contract that the
> training pipeline can rely on. Model architecture, augmentation, loss
> functions, and inference-time logic are out of scope.

## 1. Design decision

The Azure ML data assets are **pre-split, flattened, and filtered to the
root-segmentation working set**:

- The cohort/annotator folder structure of the raw delivery (`Y2B_23/`,
  `Y2B_24/`, `Y2B_25/`, annotator subfolders) is **collapsed**. The
  workspace contains flat `images/` and `masks/` folders only.
- The train/val/test split (80/10/10 by image stem, seed 42) is **baked
  into the assets** at upload time, not derived at training time.
- Only images with a `_root_mask.tif` are included. Shoot and seed masks
  travel alongside the image when they exist; they are opportunistic, not
  required.

**Why this design.** The model owners need data they can load by name and
train against, without re-implementing filtering, pooling, or splitting in
training code. Baking the split in trades flexibility (one fixed split per
asset version) for simplicity (a teammate downloads `cv5_train` and gets a
folder of training images). The flexibility cost is mitigated by Azure ML
versioning — a different split is a new version, not a workflow change.

What stays training-side: petri-dish cropping, alpha-channel removal,
patchification, normalisation, augmentation, integrity assertions. The
assets are the source of bytes; transforms remain in training code.

## 2. Data assets

Four assets are registered per upload, all at the same version, all as
`URI_FOLDER`:

| Asset | Contents | Manifest |
|---|---|---|
| `cv5_raw` | Pooled working set, unsplit. | No |
| `cv5_train` | 80% of working set. | Yes |
| `cv5_val` | 10% of working set. | Yes |
| `cv5_test` | 10% of working set. | Yes |

`cv5_train ∪ cv5_val ∪ cv5_test` at the same version reconstitutes
`cv5_raw` at that version exactly.

Every asset has the same internal structure:

```
<asset>/
├── images/        # all images, flat, original filenames preserved
├── masks/         # all masks, flat, original filenames preserved
└── manifest.csv   # (train/val/test only)
```

The manifest has four columns — `image_filename`, `root_mask_filename`,
`shoot_mask_filename`, `seed_mask_filename` — with empty cells where a
class of mask is absent for that image. Training code uses this to pair
files; it does not have to parse filenames.

## 3. What the upload pipeline does

1. Walks `raw/Y2B_23/`, `raw/Y2B_24/`, `raw/Y2B_25/`.
2. Pairs each image with any `_root_mask.tif`, `_shoot_mask.tif`,
   `_seed_mask.tif` sharing its stem. Drops `_occluded_root_mask.tif` and
   any other suffixes.
3. Drops images that lack a root mask.
4. Checks for filename collisions across cohorts (the flattening would
   otherwise overwrite silently). Fails loudly if any collision is found.
5. Sorts records by stem (stable across machines), then shuffles with
   seed 42.
6. Splits 80/10/10 by image stem. An image and its 1–3 masks travel as a
   unit.
7. Stages locally into flat `images/`/`masks/` folders; writes a
   `manifest.csv` for each split.
8. Uploads each folder to `workspaceblobstore` and registers it as a
   versioned `URI_FOLDER` asset with descriptive tags.

The upload script lives at `scripts/upload_data.py`. The pipeline does
**not** crop, resize, patchify, normalise, or alpha-strip.

## 4. What the training pipeline is expected to do

This section is contract, not enforcement. Training code can rely on:

1. **No filtering needed.** Every image in `cv5_train|val|test` has a
   `_root_mask.tif`. No need to filter by suffix.
2. **No splitting needed.** Use `cv5_train` for training,
   `cv5_val` for validation, `cv5_test` for test. Do not pool and re-split.
3. **Use the manifest to pair files.** The image-to-mask relationship is
   in `manifest.csv`. Do not derive it from filename string-matching at
   runtime if you can avoid it.
4. **Alpha-channel handling is yours.** Raw images are RGBA `uint8`; drop
   alpha before training. The asset preserves the four-channel file.
5. **Cropping and patchification are yours.** Use
   `src/npec_cv/helpers/extract_petri_dish.py` and
   `src/npec_cv/helpers/patchify.py`. Apply the same `(x, y, w, h)` crop
   to image and mask with nearest-neighbour sampling on the mask.
6. **Integrity checks are yours.** Per image/mask pair, training code
   should assert matching spatial shape, mask values ⊆ `{0, 1}`, no NaN,
   no all-zero images. The inventory found one anomalous 3D mask in the
   raw delivery; the upload preserves it as-is, and the training-time
   assertion is what catches it.

## 5. Reproducibility

Same git commit + same raw delivery + same seed → byte-identical staged
folders. The asset content is therefore deterministic; only the
`created_date` tag varies between runs. The shuffle uses Python's
`random.Random(seed)` with a sorted input list, which is stable across
Python 3 patch versions and OS.

## 6. What invalidates an asset

Per `versioning_strategy.md`, the asset set gets a **major** version bump
if any of the following change:

- The raw dataset on disk gains, loses, or modifies files.
- The inclusion rule changes (e.g. shoot-only images become acceptable).
- The split seed or ratio changes.
- The cohort scope changes (e.g. Y2B_26 added, or Kaggle brought in).

Changes to training-side preprocessing (§4) do not affect asset versions.

## 7. Review

Reviewed: YYYY-MM-DD