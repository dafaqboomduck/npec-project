"""Module for extracting the petri dish region from an image.

Plant phenotyping images from the HADES setup look like this:

    +-----------------------------------------+
    |              black border               |  pixels ≈ 0
    |     +-----------------------------+     |
    |     |        bright dish          |     |  pixels > ~30
    |     |        + dark plants        |     |
    |     +-----------------------------+     |
    +-----------------------------------------+

The dish content is bright relative to the surrounding black frame; plants
inside the dish are dark again but are small and isolated. The earlier
implementation used Otsu thresholding + the largest contour, which worked
poorly when:

  - The black border was very thin so Otsu picked a threshold inside the
    plant/agar range and contoured fragments of plants instead of the dish.
  - The image had glare or condensation droplets at the top that formed a
    larger contiguous bright region than the dish itself.

This rewrite uses a direct "anything brighter than the border is part of the
dish" approach with morphological closing to bridge the plants' dark pixels.
That's robust because the dish/border contrast is by far the dominant
intensity transition in these images.
"""

import logging
from typing import Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_petri_dish(
    img: np.ndarray,
    padding: int = 30,
    return_coords: bool = False,
    *,
    dark_thresh: int = 25,
    min_dish_ratio: float = 0.25,
) -> Union[np.ndarray, Tuple[int, int, int, int]]:
    """Return the petri dish region (or its bounding box) from an HADES image.

    Args:
        img: The input image (BGR, RGB, or grayscale — color channels are
            collapsed to luminance before processing).
        padding: Extra pixels added on each side of the detected dish before
            returning. Default 30.
        return_coords: If True, return ``(x_start, y_start, x_end, y_end)``
            instead of the cropped image. Default False.
        dark_thresh: Pixel intensity (0-255) below which is considered the
            "border" (i.e. not the dish). Default 25 — anything below ~10 %
            brightness is treated as outside the dish. Lower if the border in
            your images is paler than pure black; raise if the agar inside
            the dish is very dim.
        min_dish_ratio: The detected bright region must occupy at least this
            fraction of the image area to be accepted as the dish. If it's
            smaller (suggesting detection failed and we found a glare blob
            or stray bright object instead), the function falls back to the
            full image. Default 0.25 (25 %).

    Returns:
        - When ``return_coords=False`` (default): the cropped image array.
        - When ``return_coords=True``: a 4-tuple ``(x1, y1, x2, y2)`` giving
          the crop coordinates in the original image. ``x2`` and ``y2`` are
          exclusive (NumPy slice convention).

    Notes:
        Padding is applied *after* the bounding box of the largest bright
        connected component is computed, then clamped to image bounds. The
        returned region is not forced to be square — the U-Net works on
        arbitrary aspect ratios because patching pads to a multiple of
        ``patch_size`` regardless.
    """
    # 1. Collapse to grayscale luminance.
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    img_area = h * w

    # 2. Separate the bright dish from the dark border. Plants inside the
    #    dish are dark too, so they'd punch holes through the dish mask;
    #    the morphological close after this step seals those holes.
    _, dish_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY)

    # 3. Close holes punched by plants/dust inside the dish. The kernel size
    #    is set generously (relative to image size) because plants can be
    #    several hundred pixels wide; a too-small kernel would leave a
    #    fragmented mask that gives a tight bounding box around just one
    #    plant. The kernel is at most 51 px to keep it cheap on big images.
    kernel_size = max(15, min(51, min(h, w) // 80))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dish_mask = cv2.morphologyEx(dish_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Pick the largest bright connected component.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dish_mask, connectivity=8)
    if num_labels <= 1:
        logger.info(
            f"DIAGNOSTIC: extract_petri_dish — no bright components found "
            f"(num_labels={num_labels}); falling back to full image"
        )
        return _fallback_full_image(img, return_coords)

    # Skip label 0 (background) when ranking by area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])

    largest_pct = 100.0 * largest_area / img_area
    logger.info(
        f"DIAGNOSTIC: extract_petri_dish — image {w}x{h}, dark_thresh={dark_thresh}, "
        f"kernel={kernel_size}, num_components={num_labels-1}, "
        f"largest_area={largest_area} ({largest_pct:.1f}% of image), "
        f"min_dish_ratio={min_dish_ratio}"
    )

    # 5. Sanity check — if the biggest blob is too small, detection failed
    #    (e.g. condensation droplets formed the largest single bright region
    #    and the dish itself got fragmented). Fall back to the full image
    #    rather than crop wildly off-centre.
    if largest_area < min_dish_ratio * img_area:
        logger.info(
            f"DIAGNOSTIC: extract_petri_dish — largest blob {largest_pct:.1f}% < "
            f"min_dish_ratio {100*min_dish_ratio:.0f}%; falling back to full image"
        )
        return _fallback_full_image(img, return_coords)

    # 6. Bounding box of the dish blob, with padding clamped to image bounds.
    bx = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    by = int(stats[largest_idx, cv2.CC_STAT_TOP])
    bw = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])

    x_start = max(0, bx - padding)
    y_start = max(0, by - padding)
    x_end = min(w, bx + bw + padding)
    y_end = min(h, by + bh + padding)

    if return_coords:
        return x_start, y_start, x_end, y_end
    return img[y_start:y_end, x_start:x_end]


def _fallback_full_image(
    img: np.ndarray,
    return_coords: bool,
) -> Union[np.ndarray, Tuple[int, int, int, int]]:
    """Return the full image (or its full-bounds coordinates) when detection fails."""
    h, w = img.shape[:2]
    if return_coords:
        return 0, 0, w, h
    return img.copy()
