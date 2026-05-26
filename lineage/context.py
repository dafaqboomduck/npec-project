"""Gather lineage context from the current execution environment.

In an Azure ML job, most facts are exposed via environment variables
Azure ML sets automatically. Locally (e.g. the registration script run
from a laptop), some facts come from ``git rev-parse``. Some facts
cannot be discovered automatically and must be supplied by the caller —
``gather_lineage_context`` provides explicit overrides.

This module has no Azure ML SDK dependency and no DB dependency, so it
can be unit-tested without either.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Azure ML sets these in every job.
ENV_AZUREML_RUN_ID = "AZUREML_RUN_ID"
ENV_AZUREML_PIPELINE_RUN_ID = "AZUREML_ROOT_RUN_ID"
ENV_MLFLOW_RUN_ID = "MLFLOW_RUN_ID"
ENV_GIT_COMMIT = "AZUREML_GIT_COMMIT"
ENV_GIT_BRANCH = "AZUREML_GIT_BRANCH"
ENV_GIT_REPO = "AZUREML_GIT_REPOSITORY_URI"


@dataclass(frozen=True)
class DatasetRef:
    """A versioned reference to an Azure ML data asset used in training."""

    role: str  # e.g. "train", "val", "test"
    name: str  # e.g. "cv5_hades_raw"
    version: str  # e.g. "1.1.0"

    def to_dict(self) -> dict[str, str]:
        """Render as {role: 'name:version'} suitable for a JSON column."""
        return {self.role: f"{self.name}:{self.version}"}


@dataclass(frozen=True)
class LineageContext:
    """Everything we need to write on a model at registration time.

    A LineageContext with no ``git_commit`` is invalid for the lineage-
    enforced registration path. Use ``register_baseline_model`` for
    pre-pipeline / baseline models instead.
    """

    git_commit: str
    git_branch: Optional[str]
    git_repo: Optional[str]
    pipeline_run_id: Optional[str]
    training_job_id: Optional[str]
    mlflow_run_id: Optional[str]
    environment: Optional[str] = None
    datasets: tuple[DatasetRef, ...] = field(default_factory=tuple)
    registered_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_tags(self) -> dict[str, str]:
        """Flatten the lineage context to a dict of model registry tags.

        Tag values are strings (Azure ML registry constraint). None values
        are omitted. Datasets are flattened to ``dataset_<role>`` keys.

        Returns:
            A flat dict of string keys to string values, ready to pass to
            ``Model(tags=...)`` at registration time.
        """
        tags: dict[str, str] = {
            "git_commit": self.git_commit,
            "registered_at_utc": self.registered_at_utc,
        }
        if self.git_branch:
            tags["git_branch"] = self.git_branch
        if self.git_repo:
            tags["git_repo"] = self.git_repo
        if self.pipeline_run_id:
            tags["pipeline_run_id"] = self.pipeline_run_id
        if self.training_job_id:
            tags["training_job_id"] = self.training_job_id
        if self.mlflow_run_id:
            tags["mlflow_run_id"] = self.mlflow_run_id
        if self.environment:
            tags["environment"] = self.environment
        for ds in self.datasets:
            tags[f"dataset_{ds.role}"] = f"{ds.name}:{ds.version}"
        return tags

    def dataset_versions_jsonb(self) -> dict[str, str]:
        """Flatten datasets into a single dict for the DB JSONB column."""
        out: dict[str, str] = {}
        for ds in self.datasets:
            out.update(ds.to_dict())
        return out

    def to_dict(self) -> dict:
        """Return the context as a plain dict for logging or debugging.

        Returns:
            A dict representation of all fields, suitable for JSON
            serialization or pretty-printing.
        """
        return asdict(self)


class LineageError(RuntimeError):
    """Raised when required lineage information cannot be determined."""


def gather_lineage_context(
    *,
    datasets: tuple[DatasetRef, ...] = (),
    git_commit_override: Optional[str] = None,
    training_job_id_override: Optional[str] = None,
    environment: Optional[str] = None,
) -> LineageContext:
    """Collect lineage facts from the current execution environment."""
    git_commit = git_commit_override or os.environ.get(ENV_GIT_COMMIT) or _git_commit_from_shell()
    if not git_commit:
        raise LineageError(
            "Could not determine git commit. Set git_commit_override, "
            "run from inside a git checkout, or run inside an Azure ML "
            "job where AZUREML_GIT_COMMIT is set."
        )

    return LineageContext(
        git_commit=git_commit,
        git_branch=os.environ.get(ENV_GIT_BRANCH) or _git_branch_from_shell(),
        git_repo=os.environ.get(ENV_GIT_REPO) or _git_repo_from_shell(),
        pipeline_run_id=os.environ.get(ENV_AZUREML_PIPELINE_RUN_ID),
        training_job_id=training_job_id_override or os.environ.get(ENV_AZUREML_RUN_ID),
        mlflow_run_id=os.environ.get(ENV_MLFLOW_RUN_ID),
        environment=environment,
        datasets=datasets,
    )


def _git_commit_from_shell() -> Optional[str]:
    return _run_git(["rev-parse", "HEAD"])


def _git_branch_from_shell() -> Optional[str]:
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"])


def _git_repo_from_shell() -> Optional[str]:
    return _run_git(["config", "--get", "remote.origin.url"])


def _run_git(args: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
