"""Mask storage service.

Saves uploaded annotation masks to the configured file storage path and
return a URL pointing to the saved file.

In a later sprint this will upload to Azure Blob Storage and return a
presigned HTTPS URL. The interface will not change.
"""

import logging
import os
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

DEFAULT_MASK_DIR = Path("/tmp/npec_annotation_mask")
MASK_DIR = Path(os.getenv("NPEC_ANNOTATION_MASK_DIR"), str(DEFAULT_MASK_DIR))


def validate_tiff_mask(mask_bytes: bytes, filename: str) -> None:
    """Validate the `mask_bytes` is a single-frame .tiff.

    Args:
        mask_bytes: Raw bytes of the uploaded file.
        filename: Original filename - used in error messages.

    Raises:
        ValueError: If the file is not a valid single-frame TIFF.
    """
    import io

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


def save_annotation_mask(annotation_id: UUID, mask_bytes: bytes) -> str:
    """Write a .tiff mask to disk and return a file URL.

    Args:
        annotation_id: UUID of the annotation - used as the filename.
        mask_bytes: Raw byts of the uplaoded .tiff mask.

    Returns:
        A `file://` URL pointing to the saved mask.

    Note:
        Replace with Azure Blob Storage upload in a later sprint.
        The return type will not change.
    """
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = MASK_DIR / f"{annotation_id}_mask.tiff"
    mask_path.write_bytes(mask_bytes)
    logger.debug("Saved annotation mask to %s", mask_path)
    return mask_path.as_uri()
