"""Smoke tests for the lineage module.

These tests do NOT touch Azure ML or a real database. Fake MLClient +
in-memory SQLite via aiosqlite. Run with:

    python tests/test_lineage_smoke.py

Or via pytest:

    pytest tests/test_lineage_smoke.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from lineage.context import (  # noqa: E402
    DatasetRef,
    LineageContext,
    LineageError,
    gather_lineage_context,
)
from lineage.registration import (  # noqa: E402
    lineage_from_job,
    register_baseline_model,
    register_model_with_lineage,
)


@contextmanager
def env(**overrides):
    """Temporarily set / unset env vars."""
    original = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestDatasetRef(unittest.TestCase):
    def test_to_dict(self):
        ref = DatasetRef(role="train", name="cv5_hades_raw", version="1.1.0")
        self.assertEqual(ref.to_dict(), {"train": "cv5_hades_raw:1.1.0"})


class TestLineageContext(unittest.TestCase):
    def test_to_tags(self):
        ctx = LineageContext(
            git_commit="abc",
            git_branch="main",
            git_repo=None,
            pipeline_run_id="pipe",
            training_job_id="job",
            mlflow_run_id="mlf",
            environment="env:1",
            datasets=(DatasetRef(role="train", name="cv5_hades_raw", version="1.1.0"),),
        )
        tags = ctx.to_tags()
        self.assertEqual(tags["git_commit"], "abc")
        self.assertEqual(tags["dataset_train"], "cv5_hades_raw:1.1.0")
        self.assertEqual(tags["environment"], "env:1")

    def test_dataset_versions_jsonb(self):
        ctx = LineageContext(
            git_commit="abc",
            git_branch=None,
            git_repo=None,
            pipeline_run_id=None,
            training_job_id=None,
            mlflow_run_id=None,
            datasets=(
                DatasetRef(role="train", name="cv5_hades_raw", version="1.1.0"),
                DatasetRef(role="val", name="cv5_val", version="1.0.0"),
            ),
        )
        self.assertEqual(
            ctx.dataset_versions_jsonb(),
            {"train": "cv5_hades_raw:1.1.0", "val": "cv5_val:1.0.0"},
        )


class TestGatherLineageContext(unittest.TestCase):
    def test_reads_azureml_env(self):
        with env(
            AZUREML_GIT_COMMIT="d" * 40,
            AZUREML_RUN_ID="step-1",
            MLFLOW_RUN_ID="ml-1",
        ):
            ctx = gather_lineage_context()
        self.assertEqual(ctx.git_commit, "d" * 40)
        self.assertEqual(ctx.training_job_id, "step-1")

    def test_raises_when_no_git_anywhere(self):
        with env(AZUREML_GIT_COMMIT=None, PATH="/nonexistent"):
            with self.assertRaises(LineageError):
                gather_lineage_context()


class TestRegisterModelWithLineage(unittest.TestCase):
    def test_writes_tags(self):
        fake_client = MagicMock()
        fake_client.models.create_or_update.return_value = MagicMock(version="3")
        ctx = LineageContext(
            git_commit="abc",
            git_branch=None,
            git_repo=None,
            pipeline_run_id="pipe",
            training_job_id="job",
            mlflow_run_id=None,
        )
        register_model_with_lineage(
            fake_client,
            name="root-detection",
            path="azureml://x/",
            description="t",
            context=ctx,
        )
        model_arg = fake_client.models.create_or_update.call_args[0][0]
        self.assertEqual(model_arg.tags["git_commit"], "abc")
        self.assertEqual(model_arg.tags["baseline"], "false")

    def test_extra_tags_cannot_shadow(self):
        ctx = LineageContext(
            git_commit="abc",
            git_branch=None,
            git_repo=None,
            pipeline_run_id=None,
            training_job_id=None,
            mlflow_run_id=None,
        )
        with self.assertRaises(ValueError):
            register_model_with_lineage(
                MagicMock(),
                name="x",
                path="azureml://x/",
                description="t",
                extra_tags={"git_commit": "evil"},
                context=ctx,
            )


class TestRegisterBaselineModel(unittest.TestCase):
    def test_writes_baseline_tags(self):
        fake_client = MagicMock()
        fake_client.models.create_or_update.return_value = MagicMock(version="1")
        register_baseline_model(
            fake_client,
            name="root-detection",
            path="azureml://baseline/",
            description="Baseline shipped with v1",
            reason="shipped_with_app_v1",
        )
        model_arg = fake_client.models.create_or_update.call_args[0][0]
        self.assertEqual(model_arg.tags["baseline"], "true")
        self.assertEqual(model_arg.tags["baseline_reason"], "shipped_with_app_v1")
        # Baseline path does NOT write git_commit.
        self.assertNotIn("git_commit", model_arg.tags)

    def test_refuses_empty_reason(self):
        with self.assertRaises(ValueError):
            register_baseline_model(
                MagicMock(),
                name="x",
                path="azureml://x/",
                description="d",
                reason="",
            )

    def test_refuses_whitespace_reason(self):
        with self.assertRaises(ValueError):
            register_baseline_model(
                MagicMock(),
                name="x",
                path="azureml://x/",
                description="d",
                reason="   ",
            )


class TestLineageFromJob(unittest.TestCase):
    def test_builds_context_from_job_properties(self):
        fake_job = MagicMock()
        fake_job.properties = {
            "mlflow.source.git.commit": "5a10a63d" + "f" * 32,
            "mlflow.source.git.branch": "Feature/conditional-registration",
            "mlflow.source.git.repoURL": "https://github.com/X/Y.git",
            "mlflow.runId": "mlf-1",
        }
        fake_client = MagicMock()
        fake_client.jobs.get.return_value = fake_job
        ctx = lineage_from_job(
            fake_client,
            "helpful_lettuce_s3yfyhtlq3",
            datasets=(DatasetRef(role="train", name="cv5", version="1.0.0"),),
            environment="env:1",
        )
        self.assertEqual(ctx.git_commit, "5a10a63d" + "f" * 32)
        self.assertEqual(ctx.training_job_id, "helpful_lettuce_s3yfyhtlq3")

    def test_raises_if_no_git_in_properties(self):
        fake_job = MagicMock()
        fake_job.properties = {}
        fake_client = MagicMock()
        fake_client.jobs.get.return_value = fake_job
        with self.assertRaises(LineageError):
            lineage_from_job(fake_client, "some_job")


# ─── async tests for InferenceLogger + trace ─────────────────────────


class TestAsyncLineage(unittest.TestCase):
    """End-to-end: InferenceLogger writes a row, trace finds it.

    Tests both a pipeline-trained model AND a baseline model so the
    trace output is verified for both cases.
    """

    def test_pipeline_model_trace(self):
        asyncio.run(self._run(baseline=False))

    def test_baseline_model_trace(self):
        asyncio.run(self._run(baseline=True))

    async def _run(self, baseline: bool):
        try:
            from sqlalchemy import (
                JSON,
                Boolean,
                Column,
                DateTime,
                Float,
                ForeignKey,
                Integer,
                String,
                Text,
                func,
            )
            from sqlalchemy.ext.asyncio import (
                async_sessionmaker,
                create_async_engine,
            )
            from sqlalchemy.orm import DeclarativeBase
        except ImportError:
            self.skipTest("SQLAlchemy / aiosqlite not installed")
            return

        class Base(DeclarativeBase):
            pass

        class ModelVersion(Base):
            __tablename__ = "model_versions"
            id = Column(String, primary_key=True)
            name = Column(String, nullable=False)
            version = Column(String, nullable=False)
            status = Column(String, nullable=False, default="registered")
            git_commit = Column(String)
            git_branch = Column(String)
            git_repo = Column(Text)
            training_job_name = Column(String)
            pipeline_run_id = Column(String)
            mlflow_run_id = Column(String)
            environment = Column(String)
            dataset_versions = Column(JSON)
            artifact_uri = Column(Text)
            baseline = Column(Boolean, nullable=False, default=False)

        class InferenceAuditLog(Base):
            __tablename__ = "inference_audit_log"
            id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            prediction_id = Column(String, nullable=True)
            model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
            model_name = Column(String, nullable=False)
            model_version_label = Column(String, nullable=False)
            request_id = Column(String, nullable=False)
            input_ref = Column(Text, nullable=False)
            output_summary = Column(JSON)
            latency_ms = Column(Integer)
            error = Column(Text)
            endpoint = Column(String, default="api")
            deployment = Column(String, nullable=False, default="unknown")
            cpu_pct_system = Column(Float, nullable=True)
            cpu_pct_process = Column(Float, nullable=True)
            cpu_count = Column(Integer, nullable=True)
            ram_used_mb = Column(Integer, nullable=True)
            ram_available_mb = Column(Integer, nullable=True)
            gpu_pct = Column(Float, nullable=True)
            vram_used_mb = Column(Integer, nullable=True)
            disk_free_mb = Column(Integer, nullable=True)
            logged_at = Column(DateTime(timezone=True), server_default=func.now())

        # Inject our test ORM into the api.src.db.orm module path so
        # the lineage code's late imports resolve to it.
        import types

        api_pkg = types.ModuleType("api")
        api_src = types.ModuleType("api.src")
        api_src_db = types.ModuleType("api.src.db")
        api_src_db_orm = types.ModuleType("api.src.db.orm")
        api_src_db_orm.InferenceAuditLog = InferenceAuditLog
        api_src_db_orm.ModelVersion = ModelVersion
        sys.modules["api"] = api_pkg
        sys.modules["api.src"] = api_src
        sys.modules["api.src.db"] = api_src_db
        sys.modules["api.src.db.orm"] = api_src_db_orm

        import lineage.inference_logger as logger_mod
        import lineage.traceability as trace_mod

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        # Seed one model_versions row, baseline or not depending on test.
        mv_id = str(uuid.uuid4())
        async with Session() as db:
            if baseline:
                db.add(
                    ModelVersion(
                        id=mv_id,
                        name="NPEC U-Net Baseline",
                        version="unet-npec-v0.0.0-placeholder",
                        status="deployed",
                        artifact_uri="/app/artifacts/models/v1/",
                        baseline=True,
                    )
                )
            else:
                db.add(
                    ModelVersion(
                        id=mv_id,
                        name="root-detection",
                        version="unet-npec-v1.0.0",
                        status="deployed",
                        git_commit="aaaa" * 10,
                        training_job_name="helpful_lettuce_s3yfyhtlq3",
                        dataset_versions={"train": "cv5_hades_raw:1.1.0"},
                        baseline=False,
                    )
                )
            await db.commit()

        async with Session() as db:
            logger = logger_mod.InferenceLogger(
                db,
                model_version_id=mv_id,
                model_name="NPEC U-Net Baseline" if baseline else "root-detection",
                model_version_label="unet-npec-v0.0.0-placeholder" if baseline else "unet-npec-v1.0.0",
                deployment="local",
            )
            async with logger.measure(
                request_id="req-xyz",
                input_ref="sha256:beef",
            ) as rec:
                rec.set_output({"num_plants_detected": 3})
            await db.commit()

        async with Session() as db:
            result = await trace_mod.trace(db, "req-xyz")

        self.assertTrue(result.found)
        self.assertEqual(result.deployment, "local")
        self.assertEqual(result.is_baseline, baseline)

        pretty = result.pretty()
        if baseline:
            self.assertIn("BASELINE MODEL", pretty)
        else:
            self.assertIn("git commit", pretty)
            self.assertIn("aaaa", pretty)
            self.assertNotIn("BASELINE MODEL", pretty)

        await engine.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)
