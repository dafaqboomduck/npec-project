"""Walk the lineage chain backward from a prediction or audit-log id.

The chain::

    inference_audit_log
        | model_version_id
        v
    model_versions
        { git_commit, pipeline_run_id, training_job_name,
          dataset_versions, mlflow_run_id, baseline, ... }

Provides:
  - ``trace(...)``           — async function for use from the API
                                 or scripts that already have a session.
  - CLI ``python -m lineage.traceability <id>`` — for admins, opens
                                 its own session from DATABASE_URL.

Baseline models (``baseline=True`` on model_versions) are reported as
such rather than as broken-lineage records. A trace of a prediction
served by a baseline model produces a clearly labeled output explaining
that the model has no upstream training pipeline by design.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

log = logging.getLogger(__name__)


@dataclass
class TraceResult:
    """Everything we recovered for one prediction or audit row."""

    lookup_key: str
    found: bool = False
    request_id: Optional[str] = None
    prediction_id: Optional[uuid.UUID] = None
    model_name: Optional[str] = None
    model_version_label: Optional[str] = None
    deployment: Optional[str] = None
    logged_at: Optional[str] = None
    input_ref: Optional[str] = None
    output_summary: Optional[dict] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    model_version: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return the trace result as a JSON-serializable dict.

        Returns:
            A nested dict with ``audit``, ``model_version``, and ``notes``
            sections, ready for ``json.dumps``. Used by the CLI's
            ``--json`` flag.
        """
        return {
            "lookup_key": self.lookup_key,
            "found": self.found,
            "is_baseline": self.is_baseline,
            "audit": {
                "request_id": self.request_id,
                "prediction_id": str(self.prediction_id) if self.prediction_id else None,
                "model_name": self.model_name,
                "model_version_label": self.model_version_label,
                "deployment": self.deployment,
                "logged_at": self.logged_at,
                "input_ref": self.input_ref,
                "output_summary": self.output_summary,
                "latency_ms": self.latency_ms,
                "error": self.error,
            },
            "model_version": self.model_version,
            "notes": self.notes,
        }

    def pretty(self) -> str:
        """Format the trace result as multi-line human-readable text.

        Distinguishes baseline models (no upstream pipeline) from
        pipeline-trained models with full lineage. Used by the CLI's
        default (non-JSON) output.

        Returns:
            A multi-line string suitable for printing to the terminal.
        """
        if not self.found:
            return f"Trace {self.lookup_key}: not found in audit log."
        mv = self.model_version
        lines = [
            f"Audit entry {self.request_id}",
            f"  prediction id   {self.prediction_id}",
            f"  served by       {self.model_name}:{self.model_version_label}",
            f"  deployment      {self.deployment}",
            f"  logged at       {self.logged_at}",
            f"  input ref       {self.input_ref}",
            f"  latency         {self.latency_ms} ms",
            f"  error           {self.error or '(none)'}",
            "",
        ]
        if self.is_baseline:
            lines.extend(
                [
                    "Model lineage",
                    "  BASELINE MODEL — no upstream training pipeline.",
                    f"  reason          {mv.get('baseline_reason') or '(none recorded)'}",
                    f"  artifact_uri    {mv.get('artifact_uri', '(missing)')}",
                    "  Lineage fields below are expected to be empty for baselines.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "Model lineage",
                    f"  git commit      {mv.get('git_commit', '(missing)')}",
                    f"  git branch      {mv.get('git_branch', '(missing)')}",
                    f"  training job    {mv.get('training_job_name', '(missing)')}",
                    f"  pipeline run    {mv.get('pipeline_run_id', '(missing)')}",
                    f"  mlflow run      {mv.get('mlflow_run_id', '(missing)')}",
                    f"  environment     {mv.get('environment', '(missing)')}",
                    f"  datasets        {mv.get('dataset_versions') or '(missing)'}",
                ]
            )
        if self.notes:
            lines.append("")
            lines.append("Notes")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


async def trace(db: AsyncSession, lookup_key: str) -> TraceResult:
    """Recover the lineage chain for one prediction or audit row.

    ``lookup_key`` can be an audit request_id, a prediction UUID, or an
    audit log UUID. The function tries each interpretation in order and
    returns the first hit.
    """
    from api.src.db.orm import InferenceAuditLog, ModelVersion

    result = TraceResult(lookup_key=lookup_key)
    audit_row = None

    # Try as audit log id (UUID)
    try:
        audit_id = uuid.UUID(lookup_key)
        audit_row = await db.get(InferenceAuditLog, audit_id)
    except (ValueError, TypeError):
        pass

    # Try as prediction_id (UUID)
    if audit_row is None:
        try:
            pred_id = uuid.UUID(lookup_key)
            q = (
                select(InferenceAuditLog)
                .where(InferenceAuditLog.prediction_id == pred_id)
                .order_by(InferenceAuditLog.logged_at.desc())
                .limit(1)
            )
            r = await db.execute(q)
            audit_row = r.scalar_one_or_none()
        except (ValueError, TypeError):
            pass

    # Try as request_id (free text)
    if audit_row is None:
        q = (
            select(InferenceAuditLog)
            .where(InferenceAuditLog.request_id == lookup_key)
            .order_by(InferenceAuditLog.logged_at.desc())
            .limit(1)
        )
        r = await db.execute(q)
        audit_row = r.scalar_one_or_none()

    if audit_row is None:
        result.notes.append(f"No audit row matched {lookup_key!r} by id, prediction_id, or request_id.")
        return result

    result.found = True
    result.request_id = audit_row.request_id
    result.prediction_id = audit_row.prediction_id
    result.model_name = audit_row.model_name
    result.model_version_label = audit_row.model_version_label
    result.deployment = audit_row.deployment
    result.logged_at = audit_row.logged_at.isoformat() if audit_row.logged_at else None
    result.input_ref = audit_row.input_ref
    result.output_summary = audit_row.output_summary
    result.latency_ms = audit_row.latency_ms
    result.error = audit_row.error

    if audit_row.prediction_id is None:
        result.notes.append(
            "Audit row's prediction_id is NULL — the source prediction "
            "has been deleted. Audit row preserved for forensic purposes."
        )

    if audit_row.model_version_id is None:
        result.notes.append(
            "Audit row's model_version_id is NULL — the source model "
            "version row has been deleted or was never linked."
        )
        return result

    mv = await db.get(ModelVersion, audit_row.model_version_id)
    if mv is None:
        result.notes.append(f"model_version_id {audit_row.model_version_id} resolves to nothing.")
        return result

    result.is_baseline = bool(getattr(mv, "baseline", False))

    result.model_version = {
        "id": str(mv.id),
        "name": mv.name,
        "version": mv.version,
        "status": mv.status,
        "baseline": result.is_baseline,
        "git_commit": mv.git_commit,
        "git_branch": mv.git_branch,
        "git_repo": mv.git_repo,
        "training_job_name": mv.training_job_name,
        "pipeline_run_id": mv.pipeline_run_id,
        "mlflow_run_id": mv.mlflow_run_id,
        "environment": mv.environment,
        "dataset_versions": mv.dataset_versions,
        "artifact_uri": mv.artifact_uri,
    }

    if not result.is_baseline and not mv.git_commit:
        result.notes.append(
            f"Model version {mv.name}:{mv.version} has no git_commit and "
            "is not marked as a baseline. It was registered before "
            "lineage tagging was adopted, or the registration path "
            "skipped the lineage helper. Either re-register with full "
            "lineage, or mark it as a baseline via the baseline column."
        )

    return result


# ─── CLI ─────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lineage.traceability",
        description="Trace an inference call back through model, data, and code.",
    )
    p.add_argument("lookup_key", help="audit request_id, prediction UUID, or audit log UUID")
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Async SQLAlchemy DB URL. Defaults to DATABASE_URL env var.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text.")
    return p


async def _run_cli(lookup_key: str, database_url: str, json_mode: bool) -> int:
    if not database_url.startswith("postgresql+asyncpg") and not database_url.startswith("sqlite+aiosqlite"):
        log.warning(
            "DATABASE_URL %r does not look async. Use postgresql+asyncpg://...",
            database_url,
        )
    engine = create_async_engine(database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        result = await trace(db, lookup_key)
    await engine.dispose()

    if json_mode:
        print(json.dumps(result.as_dict(), indent=2, default=str))
    else:
        print(result.pretty())
    return 0 if result.found else 1


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``python -m lineage.traceability``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
            Provided for testability.

    Returns:
        Exit code: 0 if the lookup found an audit row, 1 if not, 2 if
        no database URL was configured.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if not args.database_url:
        print("ERROR: --database-url not provided and DATABASE_URL not set.", file=sys.stderr)
        return 2
    return asyncio.run(_run_cli(args.lookup_key, args.database_url, args.json))


if __name__ == "__main__":
    sys.exit(main())
