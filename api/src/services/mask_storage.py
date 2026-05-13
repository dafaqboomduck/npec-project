"""Mask storage service.

Saves uploaded annotation masks to the configured file storage path and
return a URL pointing to the saved file.

In a later sprint this will upload to Azure Blob Storage and return a
presigned HTTPS URL. The interface will not change.
"""

from __future__ import annotations

import io
import logging
from uuid import UUID

from api.src.services.storage import AbstractStorageBackend

logger = logging.getLogger(__name__)


def validate_tiff_mask(mask_bytes: bytes, filename: str) -> None:
    """Validate the `mask_bytes` is a single-frame .tiff.

    Args:
        mask_bytes: Raw bytes of the uploaded file.
        filename: Original filename - used in error messages.

    Raises:
        ValueError: If the file is not a valid single-frame TIFF.
    """
    from PIL import Image, UnidentifiedImageError

    if not filename.lower().endswith((".tiff", ".tif")):
        raise ValueError(f"Annotation masks must be .tiff files, got '{filename}'.")

    try:
        img = Image.open(io.BytesIO(mask_bytes))
        img.verify()
        img = Image.open(io.BytesIO(mask_bytes))
    except UnidentifiedImageError:
        raise ValueError("The uploaded file could not be decoded as an image.")
    except Exception as exc:
        raise ValueError(f"Could not open mask file: {exc}")

    try:
        img.seek(1)
        raise ValueError("Only single-frame TIFF files are accepted.")
    except EOFError:
        pass  # Single frame - correct


def save_annotation_mask(
    annotation_id: UUID,
    mask_bytes: bytes,
    backend: AbstractStorageBackend,
) -> str:
    """Write a .tiff mask to disk and return a file URL.

    Args:
        annotation_id: UUID of the annotation - used as the filename.
        mask_bytes: Raw byts of the uplaoded .tiff mask.
        backend: Storage backend to upload to.

    Returns:
        URL of the uploaded mask.

    Raises:
        Exception: On any upload failure.
    """
    object_name = f"annotation-masks/{annotation_id}_mask.tiff"
    url = backend.upload(
        object_name=object_name,
        data=mask_bytes,
        content_type="image/tiff",
    )
    logger.debug("Saved annotation mask: %s", object_name)
    return url
