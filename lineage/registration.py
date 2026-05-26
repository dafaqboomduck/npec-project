"""Register Azure ML models with full lineage tagging.

This module replaces direct calls to ``client.models.create_or_update``.
Two registration entry points:

* ``register_model_with_lineage`` — the strict path. Requires a
  ``LineageContext`` with at least a git commit. Refuses to register
  a model that cannot be traced.

* ``register_baseline_model`` — the explicit escape hatch for
  pre-pipeline / externally-sourced models. Writes a clear
  ``baseline=true`` tag and a ``baseline_reason`` so tracing later
  reports the model honestly as a baseline rather than as broken.

Plus ``lineage_from_job`` for recovering lineage from a completed
Azure ML job's properties.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from lineage.context import (
    DatasetRef,
    LineageContext,
    LineageError,
    gather_lineage_context,
)

log = logging.getLogger(__name__)


REQUIRED_TAG_KEYS = ("git_commit",)


def register_model_with_lineage(
    client: MLClient,
    *,
    name: str,
    path: str | Path,
    description: str,
    datasets: tuple[DatasetRef, ...] = (),
    asset_type: str = AssetTypes.CUSTOM_MODEL,
    stage: Optional[str] = None,
    extra_tags: Optional[dict[str, str]] = None,
    context: Optional[LineageContext] = None,
) -> Model:
    """Register a model in Azure ML with lineage tags attached.

    Refuses to register if the lineage context lacks a git_commit.
    Use ``register_baseline_model`` for pre-pipeline models.
    """
    artifact_path = Path(str(path))
    if not _looks_like_azureml_uri(str(path)) and not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    ctx = context or gather_lineage_context(datasets=datasets)
    tags = ctx.to_tags()

    missing = [k for k in REQUIRED_TAG_KEYS if not tags.get(k)]
    if missing:
        raise LineageError(
            f"Refusing to register '{name}': missing required lineage "
            f"tags {missing}. Either run inside an Azure ML job, pass "
            f"git_commit_override to gather_lineage_context(), call "
            f"lineage_from_job() to recover lineage from the completed "
            f"job, or use register_baseline_model() if this is a "
            f"genuine pre-pipeline baseline."
        )

    if extra_tags:
        conflicts = set(extra_tags) & set(tags)
        if conflicts:
            raise ValueError(
                f"extra_tags overlaps with lineage tags: {sorted(conflicts)}. " "Lineage tags are reserved."
            )
        tags = {**tags, **extra_tags}

    # Defensive: never accidentally register a "real" lineage model as
    # baseline. The dedicated function adds this tag intentionally.
    tags["baseline"] = "false"

    log.info(
        "Registering '%s' (%d tags, commit=%s, job=%s)",
        name,
        len(tags),
        tags.get("git_commit", "?")[:8],
        tags.get("training_job_id", "local"),
    )

    model_kwargs = dict(
        path=str(path),
        name=name,
        description=description,
        type=asset_type,
        tags=tags,
    )
    if stage:
        model_kwargs["stage"] = stage
    model = Model(**model_kwargs)
    return client.models.create_or_update(model)


def register_baseline_model(
    client: MLClient,
    *,
    name: str,
    path: str | Path,
    description: str,
    reason: str,
    asset_type: str = AssetTypes.CUSTOM_MODEL,
    stage: Optional[str] = None,
    extra_tags: Optional[dict[str, str]] = None,
) -> Model:
    """Register a baseline / externally-trained model in Azure ML.

    Use this when the model genuinely has no upstream training run
    inside the project's pipeline — e.g. a pre-existing artifact bundled
    with the application, a model trained before the pipeline existed,
    or a third-party reference checkpoint.

    The model is tagged ``baseline=true`` plus ``baseline_reason``, so
    that trace output later reports it honestly as a baseline rather
    than treating the lineage gaps as broken records.

    Parameters
    ----------
    client:
        Authenticated MLClient for the workspace.
    name:
        Model registry name.
    path:
        Local filesystem path or AzureML URI to the model artifact.
    description:
        Human-readable description.
    reason:
        Short string explaining why this is a baseline, e.g.
        ``"shipped_with_app_v1"`` or ``"vendor_supplied_2024"``. Stored
        as the ``baseline_reason`` tag for forensic clarity.
    asset_type:
        AssetTypes constant; defaults to CUSTOM_MODEL.
    stage:
        Optional Model stage.
    extra_tags:
        Free-form additional tags. Cannot overwrite the baseline tags.

    Returns:
    -------
    Model
        The newly registered model version.
    """
    artifact_path = Path(str(path))
    if not _looks_like_azureml_uri(str(path)) and not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    if not reason or not reason.strip():
        raise ValueError(
            "register_baseline_model requires a non-empty `reason` so the "
            "audit trail explains WHY lineage is absent. Examples: "
            "'shipped_with_app_v1', 'vendor_supplied_2024', "
            "'trained_before_pipeline'."
        )

    tags: dict[str, str] = {
        "baseline": "true",
        "baseline_reason": reason.strip(),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra_tags:
        conflicts = set(extra_tags) & set(tags)
        if conflicts:
            raise ValueError(
                f"extra_tags overlaps with baseline tags: {sorted(conflicts)}. " "Baseline tags are reserved."
            )
        tags = {**tags, **extra_tags}

    log.info("Registering BASELINE model '%s' (reason=%s)", name, reason)

    model_kwargs = dict(
        path=str(path),
        name=name,
        description=description,
        type=asset_type,
        tags=tags,
    )
    if stage:
        model_kwargs["stage"] = stage
    model = Model(**model_kwargs)
    return client.models.create_or_update(model)


def lineage_from_job(
    client: MLClient,
    job_name: str,
    *,
    datasets: tuple[DatasetRef, ...] = (),
    environment: Optional[str] = None,
) -> LineageContext:
    """Build a LineageContext from a completed Azure ML job.

    Reads the job's MLflow-captured properties (git commit/branch/repo,
    MLflow run id) and the job's run id. The caller supplies the dataset
    versions — the job itself does not know them (data was mounted as a
    versioned input by the submitter).
    """
    job = client.jobs.get(job_name)
    props = dict(job.properties or {})

    git_commit = props.get("mlflow.source.git.commit") or props.get("azureml.git.commit")
    if not git_commit:
        raise LineageError(
            f"Job {job_name!r} has no git commit in its properties. "
            "It may have been submitted from outside a git checkout, "
            "or it predates Azure ML's git auto-capture."
        )

    return LineageContext(
        git_commit=git_commit,
        git_branch=props.get("mlflow.source.git.branch"),
        git_repo=props.get("mlflow.source.git.repoURL"),
        pipeline_run_id=props.get("azureml.pipelinerunid"),
        training_job_id=job_name,
        mlflow_run_id=props.get("mlflow.runId") or job_name,
        environment=environment,
        datasets=datasets,
    )


def _looks_like_azureml_uri(path: str) -> bool:
    return path.startswith("azureml:") or path.startswith("azureml://")
