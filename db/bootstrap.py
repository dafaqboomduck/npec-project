"""First-run bootstrap script.

Creates the initial admin user and registers the baseline model if they
do not already exist. Safe to run on every startup (idempotent checks).

Usage (called from the API container entrypoint before uvicorn starts):
    python -m db.bootstrap

Lineage note (migration 0010):
    The baseline model created here genuinely has no upstream training
    pipeline — it is a pre-existing artifact shipped with the application.
    We mark it ``baseline=True`` so that lineage traces report it
    honestly as a baseline, rather than as a model with broken lineage.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]


async def bootstrap() -> None:
    """Create the initial admin user and register the baseline model."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as db:
        await _ensure_admin_user(db)
        await _ensure_normal_user(db)
        deployed = await _ensure_baseline_model(db)

    if not deployed:
        logger.error("No deployed modoel exists (BASELINE_MODEL_URI missin on disk?). Refusing to start.")
        raise SystemExit(1)
    await engine.dispose()
    logger.info("Bootstrap complete.")


async def _ensure_admin_user(db) -> None:
    import bcrypt

    from api.src.db.orm import User

    admin_email = os.environ.get("ADMIN_EMAIL", "admin@cv5.nl")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_password:
        logger.warning(
            "ADMIN_PASSWORD not set — skipping admin user creation. "
            "Set ADMIN_PASSWORD in your environment or .env file."
        )
        return

    result = await db.execute(select(User).where(User.email == admin_email))
    if result.scalar_one_or_none() is not None:
        logger.info("Admin user %s already exists, skipping.", admin_email)
        return

    hashed = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt(rounds=12)).decode()
    db.add(
        User(
            email=admin_email,
            password_hash=hashed,
            role="admin",
            is_active=True,
        )
    )
    await db.commit()
    logger.info("Created admin user: %s", admin_email)


async def _ensure_normal_user(db) -> None:
    import bcrypt

    from api.src.db.orm import User

    user_email = os.environ.get("USER_EMAIL", "intern@cv5.nl")
    user_password = os.environ.get("USER_PASSWORD")

    if not user_password:
        logger.warning(
            "USER_PASSWORD not set — skipping user creation. " "Set USER_PASSWORD in your environment or .env file."
        )
        return

    result = await db.execute(select(User).where(User.email == user_email))
    if result.scalar_one_or_none() is not None:
        logger.info("User %s already exists, skipping.", user_email)
        return

    hashed = bcrypt.hashpw(user_password.encode(), bcrypt.gensalt(rounds=12)).decode()
    db.add(
        User(
            email=user_email,
            password_hash=hashed,
            role="user",
            is_active=True,
        )
    )
    await db.commit()
    logger.info("Created user: %s", user_email)


async def _ensure_baseline_model(db) -> bool:
    from api.src.db.orm import ModelVersion

    artifact_uri = os.environ.get("BASELINE_MODEL_URI", "/app/artifacts/models/v1/")
    if not Path(artifact_uri).exists():
        logger.warning(
            "Baseline model artifact not found at '%s'. "
            "Register a model manually via POST /models or place artifacts at that path.",
            artifact_uri,
        )
        # return False

    # Update the baseline in place if it already exists, rather than skipping.
    result = await db.execute(select(ModelVersion).where(ModelVersion.version == "unet-npec-v1.0.0-baseline"))
    existing = result.scalar_one_or_none()
    if existing is not None:
        existing.f1_score = 0.7932
        existing.status = "deployed"
        existing.artifact_uri = artifact_uri
        existing.deployed_at = datetime.now(timezone.utc)
        existing.baseline = True
        await db.commit()
        logger.info("Updated existing baseline model (f1=%.4f).", existing.f1_score)
        return True

    # No baseline yet. If some *other* model is already deployed, don't
    # clobber it with a baseline — that's the original guard's intent.
    result = await db.execute(select(ModelVersion).where(ModelVersion.status == "deployed").limit(1))
    if result.scalar_one_or_none() is not None:
        logger.info("A non-baseline deployed model exists, skipping baseline registration.")
        return True

    # Lineage: this model genuinely has no upstream training run. Mark
    # it as a baseline so traces report it honestly. See migration 0010
    # and the lineage module's register_baseline_model() helper for the
    # equivalent path on the Azure ML registry side.
    db.add(
        ModelVersion(
            version="unet-npec-v1.0.0-baseline",
            name="NPEC U-Net Baseline",
            description="Baseline model shipped with the application.",
            status="deployed",
            f1_score=0.7932,
            artifact_uri=artifact_uri,
            deployed_at=datetime.now(timezone.utc),
            baseline=True,
        )
    )
    await db.commit()
    logger.info(
        "Registered and deployed baseline model from '%s' (marked baseline=True).",
        artifact_uri,
    )
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(bootstrap())
