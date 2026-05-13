"""Artifact storage service.

Saves three artefacts per prediction to configurable local directories:
1. Input image          - the raw uploaded file before inference
2. Segmentation mask    - the combined binary mask output from the pipeline
3. Tip photo            - the visualisation image with root tips annotated

All three save operations are independent. A failure in one is caught,
logged as WARNING, and does not block the others or the prediction response.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

# Configurable storage directories

INPUT_STORAGE_PATH = Path(os.getenv("INPUT_STORAGE_PATH", "/app/data/inputs"))
MASK_STORAGE_PATH = Path(os.getenv("MASK_STORAGE_PATH", "/app/data/masks"))
TIP_PHOTO_STORAGE_PATH = Path(os.getenv("TIP_PHOTO_STORAGE_PATH", "/app/data/tips"))


def save_input_image(
    prediction_id: UUID,
    image_bytes: bytes,
    original_filename: str,
) -> str | None:
    """Save the raw uploaded image to `INPUT_STORAGE_PATH`.

    The filename is `{prediction_id}_input.{ext}` so it is deterministic
    and directly traceable to the prediction record.

    Args:
        prediction_id: UUID of the prediction - used as the filename stem.
        image_bytes: Raw bytes of the uploaded file.
        original_filename: Original filename from the upload, used to
            extract the file extension.

    Returns:
        An absolute `file://` URI of the saved file, or `None` if saving
        failed.
    """
    try:
        INPUT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        ext = Path(original_filename).suffix.lower() or ".png"
        dest = INPUT_STORAGE_PATH / f"{prediction_id}_input{ext}"
        dest.write_bytes(image_bytes)
        logger.debug("Saved input image to %s", dest)
        return dest.resolve().as_uri()
    except Exception as exc:
        logger.warning("Failed to save input image for prediction %s: %s", prediction_id, exc)
        return None


def save_segmentation_mask(
    prediction_id: UUID,
    image_result,
) -> str | None:
    """Combine all plant masks and save to `MASK_STORAGE_PATH`.

    Args:
        prediction_id: UUID ofo the prediction - used as the filename stem.
        image_result: `ImageResult` from the `npec_cv` pipeline containing
            per-plant masks in `image_result.plant_results`.

    Returns:
        An absolute `file://` URI of the saved mask, or `None` if no
        masks were available or saving field.
    """
    try:
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

        MASK_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        dest = MASK_STORAGE_PATH / f"{prediction_id}_mask.png"
        Image.fromarray(combined).save(dest)
        logger.debug("Saved segmentation mask to %s", dest)
        return dest.resolve().as_uri()
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
) -> str | None:
    """Draw root tips on the original image and save to `TIP_PHOTO_STORAGE_PATH`.

    Overlays a coloured circle at each detected tip coordinate from
    `image_result.plant_results` onto the original input image.

    Args:
        prediction_id: UUID of the prediction - used as the filename stem.
        image_result: `ImageResult` from the pipeline with detected tips.
        original_image_bytes: Raw bytes of the input image used as background.

    Returns:
        An absolute `file://` URI of the saved tip photo, or `None` if
        saving failed.
    """
    try:
        import io

        from PIL import Image, ImageDraw

        img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        radius = max(8, min(img.width, img.height) // 100)
        for plant_id, pr in image_result.plant_results.items():
            if not pr.detected or pr.end_point is None:
                continue
            row, col = int(pr.end_point[0]), int(pr.end_point[1])
            bbox = [col - radius, row - radius, col + radius, row + radius]
            draw.ellipse(bbox, outline="red", fill="red", width=2)

        TIP_PHOTO_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        dest = TIP_PHOTO_STORAGE_PATH / f"{prediction_id}_tips.png"
        img.save(dest)
        logger.debug("Saved tip photo to %s", dest)
        return dest.resolve().as_uri()
    except Exception as exc:
        logger.warning("Failed to save tip photo for prediction %s: %s", prediction_id, exc)
        return None
