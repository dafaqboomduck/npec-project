"""Artifact storage service.

Saves three artefacts per prediction to configurable local directories:
1. Input image          - the raw uploaded file before inference
2. Segmentation mask    - the combined binary mask output from the pipeline
3. Tip photo            - the visualisation image with root tips annotated


Object name conventions
-----------------------
inputs/{prediction_id}_input.{ext}     - raw uploaded image
masks/{prediction_id}_mask.png         - combined segmentation mask
tips/{prediction_id}_tips.png          - tip visualisation overlay
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from api.src.services.storage import AbstractStorageBackend

logger = logging.getLogger(__name__)


def _mime_from_ext(ext: str) -> str:
    """Return a MIME type for a file extension."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(ext.lower(), "application/octet-stream")


def save_input_image(
    prediction_id: UUID,
    image_bytes: bytes,
    original_filename: str,
    backend: AbstractStorageBackend,
) -> str | None:
    """Upload the raw input image and return its URL.

    Args:
        prediction_id: UUID of the prediction - used as the filename stem.
        image_bytes: Raw bytes of the uploaded file.
        original_filename: Original filename from the upload, used to
            extract the file extension.
        backend: Storage backend to upload to.

    Returns:
        An URL string, or `None` on failure.
    """
    try:
        ext = Path(original_filename).suffix.lower() or ".png"
        object_name = f"inputs/{prediction_id}_input{ext}"
        url = backend.upload(
            object_name=object_name,
            data=image_bytes,
            content_type=_mime_from_ext(ext),
        )
        logger.debug("Saved input image: %s", object_name)
        return url
    except Exception as exc:
        logger.warning("Failed to save input image for prediction %s: %s", prediction_id, exc)
        return None


def save_segmentation_mask(
    prediction_id: UUID,
    image_result,
    backend: AbstractStorageBackend,
) -> str | None:
    """Combine all plant masks and upload the result.

    Args:
        prediction_id: UUID of the prediction - used as the filename stem.
        image_result: `ImageResult` from the `npec_cv` pipeline containing
            per-plant masks in `image_result.plant_results`.
        backend: Storage backend to uploado to.

    Returns:
        A URL string, or `None` if no masks were available or upload failed.
    """
    try:
        import io

        import numpy as np
        from PIL import Image

        masks = [pr.mask for pr in image_result.plant_results.values() if pr.mask is not None]
        if not masks:
            logger.debug(
                "No masks available for prediction %s - skipping mask save.",
                prediction_id,
            )
            return None

        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)

        buf = io.BytesIO()
        Image.fromarray(combined).save(buf, format="PNG")
        mask_bytes = buf.getvalue()

        object_name = f"masks/{prediction_id}_mask.png"
        url = backend.upload(
            object_name=object_name,
            data=mask_bytes,
            content_type="image/png",
        )
        logger.debug("Saved segmentation mask %s", object_name)
        return url
    except Exception as exc:
        logger.warning(
            "Failed to save segmentation mask for prediction %s: %s",
            prediction_id,
            exc,
        )
        return None


def save_tip_photo(
    prediction_id: UUID,
    image_result,
    original_image_bytes: bytes,
    backend: AbstractStorageBackend,
) -> str | None:
    """Draw root tips on the original image and save to `TIP_PHOTO_STORAGE_PATH`.

    Overlays a coloured circle at each detected tip coordinate from
    `image_result.plant_results` onto the original input image.

    Args:
        prediction_id: UUID of the prediction - used as the filename stem.
        image_result: `ImageResult` from the pipeline with detected tips.
        original_image_bytes: Raw bytes of the input image used as background.
        backend: Storage backend to upload to.

    Returns:
        A URL string, or `None` on failure.
    """
    try:
        import io

        from PIL import Image, ImageDraw

        detected = [pr for pr in image_result.plant_results.values() if pr.detected and pr.end_point is not None]
        if not detected:
            logger.debug(
                "No detected plants for prediction %s - skipping tip photo save.",
                prediction_id,
            )
            return None

        img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        radius = max(8, min(img.width, img.height) // 100)
        for pr in detected:
            row, col = int(pr.end_point[0]), int(pr.end_point[1])
            bbox = [col - radius, row - radius, col + radius, row + radius]
            draw.ellipse(bbox, outline="red", fill="red", width=2)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tip_bytes = buf.getvalue()

        object_name = f"tips/{prediction_id}_tips.png"
        url = backend.upload(
            object_name=object_name,
            data=tip_bytes,
            content_type="image/png",
        )
        logger.debug("Saved tip photo: %s", object_name)
        return url
    except Exception as exc:
        logger.warning("Failed to save tip photo for prediction %s: %s", prediction_id, exc)
        return None
