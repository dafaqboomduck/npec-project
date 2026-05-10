# Data inventory — HADES/NPEC Block B

> **Owner:** Alican  ·  **Last updated:** 2026-05-11  ·  **Status:** draft

This document describes the raw HADES/NPEC plant segmentation dataset as it
lives on disk, before any preprocessing. It exists so anyone on the team
understands what we have without opening the files themselves.

## 1. Source

| Field | Value |
|---|---|
| Project | Block B — plant segmentation (HADES/NPEC) |
| Origin | NPEC (Netherlands Plant Eco-phenotyping Centre) |
| Delivery method | _TODO: how was it delivered_ |
| On-disk location | `raw/` (structure below) |
| Access | _TODO: who has read access, who is the data steward_ |
| Licence / usage restrictions | _TODO_ |

## 2. Summary figures

_Numbers below come from `scripts/inventory_dataset.py` (sample 150) and
`scripts/dataset_deep_check.py` (full scan)._

| Metric | Value |
|---|---|
| Total size on disk | 24.54 GB |
| Total files | 2,278 |
| Image files (PNG) | 685 |
| Mask files (TIFF) | 1,592 |
| Other | 1 `.txt` (annotation-protocol note from NPEC; see §8) |
| Cohorts on disk | Y2B_23, Y2B_24, Y2B_25, Kaggle |
| In-scope cohorts for training | Y2B_23, Y2B_24, Y2B_25 (Kaggle excluded — no masks) |
| Image stems with at least one mask | 494 |
| **Root masks available (training ceiling)** | **492** (115 Y2B_23 + 314 Y2B_24 + 63 Y2B_25) |
| `occluded_root` masks (excluded) | 115 (all in Y2B_23) |

## 3. Folder layout

```
raw/
├── Y2B_23/
│   ├── images/
│   │   ├── train/              # 96 files (pre-made split, not preserved going forward)
│   │   └── test/               # 30 files
│   └── masks/                  # 460 files (flat, no annotator subfolders)
├── Y2B_24/
│   ├── images/                 # 451 files
│   └── masks/
│       ├── Alican/             # 165
│       ├── Karna/              # 165
│       ├── Elavendan/          # 157
│       ├── Shival/             # 156
│       ├── Dean/               # 150
│       ├── Myrthe/             # 90
│       └── Jason/              # 60
├── Y2B_25/
│   ├── images/                 # 89 files
│   └── masks/
│       ├── Alican/             # 42
│       ├── Dean/               # 39
│       ├── Karna/              # 30
│       ├── Myrthe/             # 30
│       ├── Jason/              # 27
│       └── Uther/              # 21
└── Kaggle/                     # 19 files (out of scope — no masks)
```

> **Annotator subfolders are a work partition, not an overlay.** Each annotator
> was assigned a disjoint slice of images. Masks from different annotators can
> be concatenated without reconciliation. To be confirmed by
> `scripts/dataset_deep_check.py` (see §6).

## 4. File formats

### 4.1 Images

| Property | Value |
|---|---|
| Extension | `.png` |
| Resolution | 4202×3006 or 4112×3006 (two sizes, ~53/47 split in the sample) |
| Channels | **4 (RGBA)** — alpha is dropped at preprocessing |
| Dtype | `uint8` |
| Colour space | sRGB assumed |

### 4.2 Masks

| Property | Value |
|---|---|
| Extension | `.tif` |
| Resolution | matches paired image (4202×3006 or 4112×3006) |
| Channels | single-channel (one anomaly found: 1 mask is 3006×4202×2, flagged in §6) |
| Dtype | `uint8` |
| Values | `{0, 1}` — binary foreground/background |
| Filename pattern | `<image_stem>_<class>_mask.tif` where `<class>` ∈ `root`, `shoot`, `seed`, `occluded_root` |

**Class breakdown per image:** each annotated image has up to 4 mask files
(one per class). Expected maximum: 685 × 3 = 2,055. Actual: 1,592. Gap of
~463 is a mix of (a) images not yet fully annotated and (b) `occluded_root`
being a subset rather than applying to every image. Exact per-class counts
come from `dataset_deep_check.py`.

> **Training scope:** only `_root_mask.tif` files are used by the current
> root-segmentation model. `shoot`, `seed`, and `occluded_root` masks are
> uploaded to Azure as part of the raw asset but **filtered out at training
> time**, not at upload. See `preprocessing_spec.md` §4.1.

## 5. Naming convention (as received)

Image filenames carry embedded metadata, e.g.:

```
032_43-6-ROOT1-2023-08-08_pvd_OD01_Col0_01-Fish Eye Corrected.png
train_Alican_230858_im1_seed.png
```

Observed patterns (non-exhaustive):
- `{idx}_{rig}-{plant}-ROOT1-{date}_{condition}_{genotype}_{plate}-Fish Eye Corrected`
- `{split}_{annotator}_{studentid}_im{n}`

**Note:** spaces in filenames (`Fish Eye Corrected`) must be handled by any
script that touches these files. Quote paths when passing to shell commands.

## 6. Known issues

- [x] **Multi-mask per image is expected** — 1,592 masks / 685 images is ~2.3,
      explained by the root/shoot/seed/occluded_root split. Not an error.
- [x] **TIFF readability** — an initial 150-image sample had 13 (~15%)
      unreadable with PIL. A full scan with `dataset_deep_check.py` found
      **all 1,592 masks readable**. The sample result was transient
      (likely a filesystem hiccup during the first run). Resolved, no dead
      files.
- [x] **Annotator partition confirmed clean** — 494 image stems, zero
      overlap across annotators. Folders can be concatenated safely.
- [ ] **Mask-to-image gap:** 685 images on disk but only **494 have any
      mask** → 191 unannotated images. Of those 494, only **492 have a root
      mask**. The 2 images without a root mask:
    - Y2B_24: 1 image with `{shoot}` only, 1 image with `{seed, shoot}` only.
    - These will be silently excluded from the training set by the root-mask
      filter. Not a blocker, but worth logging.
- [ ] **Y2B_23 occluded_root coincidence:** every single one of the 115
      Y2B_23 root masks has a matching `occluded_root` mask (1:1 pairing).
      Y2B_24 and Y2B_25 have zero occluded_root masks. Suggests Y2B_23 used
      a different annotation protocol. `occluded_root` is excluded from
      training regardless (see `preprocessing_spec.md` §1).
- [ ] **One mask has anomalous shape `(3006, 4202, 2)`** instead of 2D
      single-channel. Found in the original sample; needs inspection. Might
      be a multi-layer TIFF that silently loaded.
- [ ] **Images are RGBA.** Alpha channel needs to be dropped at training
      time (the asset preserves the raw 4-channel files; see
      `preprocessing_spec.md` §4.2).
- [ ] **Y2B_23 has a pre-made train/test split.** Not preserved — training
      code re-splits all three cohorts with a new seed (see
      `preprocessing_spec.md` §4.4).
- [ ] **Spaces in filenames** — all scripts must handle this.
- [ ] **Kaggle folder** (19 files) has no masks and is excluded from scope
      until that changes.

## 7. How to regenerate

```bash
pip install pillow numpy tifffile
python scripts/inventory_dataset.py raw/ --sample 150 \
    --out docs/data/_inventory_report.md \
    --json-out docs/data/_inventory_report.json

python scripts/dataset_deep_check.py raw/ \
    --out docs/data/_deep_check_report.md
```

Raw reports are committed alongside this doc so the tables can be regenerated
if the dataset changes.

## 8. Notes on dataset history

Resolved from prior open questions:

- **Why Y2B_23 has a flat masks folder.** 2023 was the first year the school
  ran this task. The annotation method wasn't mature and the image set was
  smaller. Y2B_24 and Y2B_25 adopted a better system: images divided as
  evenly as possible between student groups, each folder named after the
  mentor leading the group.
- **Why all 115 Y2B_23 root masks have a paired `occluded_root` mask.** A
  legacy of the 2023 annotation protocol. The lone `.txt` file in the raw
  delivery is the source instruction telling subsequent cohorts to ignore
  occluded-root annotation — which is why Y2B_24 and Y2B_25 have zero such
  masks. This matches the training-time decision to filter out
  `occluded_root` masks (see `preprocessing_spec.md` §4.1).
- **Y2B_24 outliers** (1 image with `{shoot}`, 1 with `{seed, shoot}`, 2 with
  `{root, seed}`). Attributed to incomplete annotations, failed quality
  checks, or students forgetting to upload a file. The root-mask filter
  silently drops the two images without a root mask; no further action.
- **Mask with shape `(3006, 4202, 2)`.** Bad annotation, to be ignored. The
  asset preserves the file as-is; the training-time integrity check
  recommended in `preprocessing_spec.md` §6 will catch and log it.
