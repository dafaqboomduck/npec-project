# Dataset versioning and naming strategy

> **Status:** draft (revised)
> **Supersedes:** the single-asset `cv5_hades_raw` strategy previously
> documented here.
> **Scope:** naming, versioning, and lifecycle of the Azure ML data assets
> produced from the HADES/NPEC Block B dataset.

## 1. Goals

- A teammate can find and load the right asset from name and version
  alone, with no further context.
- A training run can be traced to an exact, immutable asset version.
- The asset set survives new cohorts, corrections, and split changes
  without naming churn.

## 2. Assets

Four assets per upload, all registered at the same version:

| Name | Role |
|---|---|
| `cv5_raw` | Pooled working set, unsplit. Source of truth for the version. |
| `cv5_train` | 80% split, used for training. |
| `cv5_val` | 10% split, used for validation. |
| `cv5_test` | 10% split, used for the held-out test set. |

All four are `URI_FOLDER` assets. All four have the same internal layout:
flat `images/` and `masks/` folders. The three split assets additionally
contain a `manifest.csv`. See `preprocessing_spec.md §2` for full content
rules.

The names are **permanent**. New data, new splits, or new versions of any
kind always go to a new version under these same names, never to a new
name.

## 3. Versioning rules

Versions follow semantic-style `MAJOR.MINOR.PATCH`. All four assets are
**always bumped together** to the same version. There is no scenario where
`cv5_raw` v1.2.0 coexists with `cv5_train` v1.1.0 — that would mean the
split no longer matches the raw working set.

| Change | Bump | Example |
|---|---|---|
| Raw dataset modified upstream (re-export, new corrections) | Major | 1.4.2 → 2.0.0 |
| Inclusion rule changed (e.g. shoot-only images now allowed) | Major | 1.x.x → 2.0.0 |
| Split seed or ratio changed | Major | 1.x.x → 2.0.0 |
| Cohort scope changed (e.g. Y2B_26 added) | Major | 1.x.x → 2.0.0 |
| Additional annotations completed within existing scope | Minor | 1.4.2 → 1.5.0 |
| Corrupted files removed, metadata-only fix | Patch | 1.4.2 → 1.4.3 |

Versions are **never overwritten**. A bad upload is fixed by bumping the
patch number and documenting the fix in the asset description.

### Subset versions

`0.x.y` versions are reserved for **subset uploads** — small samples used
to validate the upload workflow before the full run. The first production
upload is `1.0.0`. Subset versions are tagged `is_subset=true` and remain
in the workspace as smoke-test fixtures.

## 4. Required tags

Every registered asset version carries the following tags:

| Tag | Values | Notes |
|---|---|---|
| `source` | `hades` | Origin of the raw delivery. |
| `cohorts` | comma-separated, e.g. `Y2B_23,Y2B_24,Y2B_25` | Which cohorts are pooled. |
| `role` | `raw`, `train`, `val`, `test` | Which asset this is. |
| `mask_classes` | e.g. `root,shoot,seed` | Which mask classes the asset contains. |
| `num_images` | integer | File count. |
| `num_masks` | integer | File count. |
| `is_subset` | `true` or `false` | Subset uploads vs full uploads. |
| `created_by` | username | Who ran the upload script. |
| `created_date` | `YYYY-MM-DD` | Upload date (UTC). |
| `git_commit` | short SHA | Commit of the upload script. |

Split assets additionally carry:

| Tag | Values |
|---|---|
| `split_ratio` | `80_10_10` |
| `split_seed` | `42` |
| `derived_from` | `cv5_raw:<version>` |

`cv5_raw` additionally carries `excluded_paths`, listing what was dropped
from the raw delivery (e.g. `Kaggle/,*.txt,*_occluded_root_mask.tif,images_without_root_mask`).

## 5. Description field

Every asset description includes:

1. One-line summary of what the asset is and its role.
2. Pooling and split policy (cohorts + seed + ratio for splits).
3. Inclusion rule (what was kept, what was dropped).
4. Git commit of the upload script.
5. Whether this is a subset upload.
6. Pointers to `docs/data/preprocessing_spec.md` and
   `docs/data/versioning_strategy.md`.

The upload script generates these descriptions automatically — they should
not be hand-edited after registration.

## 6. Legacy assets

The following assets were registered under the previous strategy and are
**archived**, not deleted (Azure ML does not support deleting assets):

| Name | Versions | Status |
|---|---|---|
| `cv5_hades_raw` | 1.0.0, 1.1.0 | Archived. Used the now-superseded 1:1-mirror strategy with cohort/annotator folder structure preserved. |
| `cv5_test_images` | 1.0.0, 1.1.0, 1.1.1 | Archived. Was a one-off test fixture under the old naming scheme. |

The underlying datastore files for these assets are left in place. They
take negligible storage and removing them risks breaking references from
any training runs that pinned to those versions.

New training runs should reference the new `cv5_*` asset set exclusively.

## 7. Loading an asset

```python
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

ml_client = MLClient(
    credential=InteractiveBrowserCredential(),
    subscription_id="...",
    resource_group_name="buas-y2",
    workspace_name="team-CV5-2026",
)

asset = ml_client.data.get(name="cv5_train", version="1.0.0")
# asset.path is the URI in workspaceblobstore
```

For reading file lists without downloading:

```python
from azureml.fsspec import AzureMachineLearningFileSystem
fs = AzureMachineLearningFileSystem(asset.path)
images = fs.ls("images/")
```

For training jobs, pass the asset as an input to the job and let Azure ML
mount or download it onto the compute target.

## 8. Review

Reviewed and agreed: YYYY-MM-DD