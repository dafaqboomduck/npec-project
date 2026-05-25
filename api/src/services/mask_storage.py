"""Mask storage service.

Saves uploaded annotation masks to the configured storage backend and
returns a key pointing to the saved file.

Masks are stored as PNG. PNG is browser-renderable, losslessly encodes
binary and grayscale data, and is supported by all downstream tooling.
"""

from __future__ import annotations

import io
import logging
from uuid import UUID

from api.src.services.storage import AbstractStorageBackend

logger = logging.getLogger(__name__)

_ACCEPTED_MASK_EXTENSIONS = frozenset({".png", ".tiff", ".tif"})


def validate_mask(mask_bytes: bytes, filename: str) -> None:
    """Validate that ``mask_bytes`` is a single-frame PNG or TIFF image.

    Args:
        mask_bytes: Raw bytes of the uploaded file.
        filename: Original filename — used in error messages.

    Raises:
        ValueError: If the extension is unsupported, the bytes cannot be
            decoded as an image, or the image has more than one frame.
    """
    from PIL import Image, UnidentifiedImageError

    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in _ACCEPTED_MASK_EXTENSIONS:
        raise ValueError(f"Annotation masks must be PNG or TIFF, got '{filename}'.")

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
        raise ValueError("Only single-frame images are accepted.")
    except EOFError:
        pass  # Single frame — correct


def normalise_to_png(image_bytes: bytes) -> bytes:
    """Convert image bytes to a grayscale PNG.

    If the input is already a single-channel PNG the bytes are returned
    unchanged to avoid a re-encode cycle.

    Args:
        image_bytes: Raw bytes of a PNG or TIFF image.

    Returns:
        Raw bytes of a grayscale PNG.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))

    if img.format == "PNG" and img.mode == "L":
        return image_bytes

    if img.mode != "L":
        img = img.convert("L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_annotation_mask(
    annotation_id: UUID,
    mask_bytes: bytes,
    backend: AbstractStorageBackend,
) -> str:
    """Normalise mask to grayscale PNG, write to storage, return the key.

    Args:
        annotation_id: UUID of the annotation — used as the filename.
        mask_bytes: Raw bytes of the mask (PNG or TIFF input accepted).
        backend: Storage backend to upload to.

    Returns:
        Key of the uploaded mask.

    Raises:
        Exception: On any upload failure.
    """
    png_bytes = normalise_to_png(mask_bytes)
    object_name = f"annotation-masks/{annotation_id}_mask.png"
    key = backend.upload(
        object_name=object_name,
        data=png_bytes,
        content_type="image/png",
    )
    logger.debug("Saved annotation mask: %s", object_name)
    return key
