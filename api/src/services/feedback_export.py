"""Fold human-confirmed corrections into the Azure ML `cv5_corrections` data asset.

Cloud-direct: reads confirmed annotations from the app DB, downloads the corrected
image+mask from the APP blob, normalises to the cv5_raw layout (images/<stem>.png +
masks/<stem>_root_mask.tif, 0/255), appends to a persistent dir, and registers a new
`cv5_corrections` version (SDK uploads the dir to the workspace default datastore).

Accumulating + idempotent: the watermark (annotations.exported_to_raw_at) exports each
annotation once. NPEC_CORRECTIONS_DIR MUST be on a persistent, mounted volume.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
_ASSET = "cv5_corrections"


def _ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential, DefaultAzureCredential

    sub = os.getenv("AZUREML_SUBSCRIPTION_ID")
    rg = os.getenv("AZUREML_RESOURCE_GROUP")
    ws = os.getenv("AZUREML_WORKSPACE")
    if not all([sub, rg, ws]):
        raise RuntimeError("AZUREML_SUBSCRIPTION_ID / RESOURCE_GROUP / WORKSPACE required")
    try:
        cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        cred.get_token("https://management.azure.com/.default")
    except Exception:  # noqa: BLE001
        cred = AzureCliCredential()
    return MLClient(cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws)


def _next_version(ml_client) -> str:
    try:
        cur = ml_client.data.get(name=_ASSET, label="latest").version
        a, b, c = (int(p) for p in cur.split("."))
        return f"{a}.{b}.{c + 1}"
    except Exception:  # noqa: BLE001 — absent or non-semver
        return "1.0.0"


def _register(corrections_dir: Path, added: int) -> str:
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data

    ml = _ml_client()
    version = _next_version(ml)
    ml.data.create_or_update(
        Data(
            name=_ASSET,
            version=version,
            path=str(corrections_dir),
            type=AssetTypes.URI_FOLDER,
            description=f"Accumulated human-confirmed correction pairs (cv5_raw layout). +{added} this run.",
            tags={
                "role": "corrections",
                "mask_classes": "root",
                "added_this_run": str(added),
                "created_by": "feedback-export",
                "created_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            },
        )
    )
    return version


def _materialise(storage, ann_id, image_uri: str, mask_uri: str, img_dir: Path, msk_dir: Path) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    stem = f"fb_{uuid.UUID(str(ann_id)).hex}"
    img = np.array(Image.open(io.BytesIO(storage.download(image_uri))).convert("RGB"))
    msk = np.array(Image.open(io.BytesIO(storage.download(mask_uri))).convert("L"))
    if msk.shape[:2] != img.shape[:2]:  # alignment backstop
        msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    msk = np.where(msk > 0, 255, 0).astype("uint8")  # canonical 0/255 root mask
    Image.fromarray(img, "RGB").save(img_dir / f"{stem}.png", format="PNG")
    Image.fromarray(msk, "L").save(msk_dir / f"{stem}_root_mask.tif", format="TIFF")


async def export_confirmed_feedback(db: AsyncSession, *, limit: int | None = None, rebuild: bool = False) -> int:
    """Export confirmed corrections into cv5_corrections:N+1. Returns pairs added."""
    from api.src.db.orm import Annotation, Prediction
    from api.src.schemas.library import AnnotationStatus
    from api.src.services.storage import get_storage_backend

    corrections_dir = Path(os.getenv("NPEC_CORRECTIONS_DIR", "/app/artifacts/cv5_corrections"))
    img_dir, msk_dir = corrections_dir / "images", corrections_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    q = (
        select(Annotation.id, Annotation.image_uri, Annotation.mask_uri)
        .join(Prediction, Prediction.id == Annotation.prediction_id)
        .where(
            Annotation.status == AnnotationStatus.ANNOTATED.value,
            Annotation.image_uri.is_not(None),
            Annotation.mask_uri.is_not(None),
            Prediction.confirmed_at.is_not(None),
        )
        .order_by(Annotation.created_at)
    )
    if not rebuild:
        q = q.where(Annotation.exported_to_raw_at.is_(None))
    if limit:
        q = q.limit(limit)

    rows = (await db.execute(q)).all()
    if not rows:
        logger.info("Feedback export: no new confirmed pairs.")
        return 0

    storage = get_storage_backend()
    exported: list[uuid.UUID] = []
    for ann_id, image_uri, mask_uri in rows:
        try:
            await asyncio.to_thread(_materialise, storage, ann_id, image_uri, mask_uri, img_dir, msk_dir)
            exported.append(uuid.UUID(str(ann_id)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feedback export: skipping %s: %s", ann_id, exc)
    if not exported:
        logger.error("Feedback export: nothing materialised.")
        return 0

    version = await asyncio.to_thread(_register, corrections_dir, len(exported))
    logger.info("Feedback export: registered cv5_corrections:%s (+%d).", version, len(exported))

    if not rebuild:
        await db.execute(
            update(Annotation).where(Annotation.id.in_(exported)).values(exported_to_raw_at=datetime.now(timezone.utc))
        )
        await db.commit()
    return len(exported)
