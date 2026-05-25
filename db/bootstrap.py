"""First-run bootstrap script.

Creates the initial admin user and registers the baseline model if they
do not already exist. Safe to run on every startup (idempotent checks).

Usage (called from the API container entrypoint before uvicorn starts):
    python -m db.bootstrap
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
    """Creates the initial admin user and registers the baseline model."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as db:
        await _ensure_admin_user(db)
        await _ensure_baseline_model(db)

    await engine.dispose()
    logger.info("Bootstrap complete.")


async def _ensure_admin_user(db) -> None:
    import bcrypt

    from api.src.db.orm import User

    admin_email = os.environ.get("ADMIN_EMAIL", "admin@cv5nl.")
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


async def _ensure_baseline_model(db) -> None:
    from api.src.db.orm import ModelVersion

    result = await db.execute(select(ModelVersion).where(ModelVersion.status == "deployed").limit(1))
    if result.scalar_one_or_none() is not None:
        logger.info("Deployed model already exists, skipping baseline registration.")
        return

    artifact_uri = os.environ.get("BASELINE_MODEL_URI", "/app/artifacts/models/v1/")
    if not Path(artifact_uri).exists():
        logger.warning(
            "Baseline model artifact not found at '%s'. "
            "Register a model manually via POST /models or place artifacts at that path.",
            artifact_uri,
        )
        return

    db.add(
        ModelVersion(
            version="unet-npec-v1.0.0-baseline",
            name="NPEC U-Net Baseline",
            description="Baseline model shipped with the application.",
            status="deployed",
            artifact_uri=artifact_uri,
            deployed_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()
    logger.info("Registered and deployed baseline model from '%s'.", artifact_uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(bootstrap())
