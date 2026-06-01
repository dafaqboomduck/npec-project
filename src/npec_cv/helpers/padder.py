"""Module for padding images and masks."""


def padder(image, patch_size):
    """Pads an image (or mask) so that its dimensions are multiples of patch_size.

    Padding is added on the BOTTOM and RIGHT only (top-left alignment). This
    matches how the training pipeline pads inputs (train_root_detection.py and
    evaluate_root_detection.py both pad bottom/right). A previous version of
    this function distributed padding evenly on all sides, which inserted zero
    columns on the LEFT of inference images — a pattern the model never saw
    during training. The result was a strong right-side prediction bias.

    Args:
        image (np.ndarray): The input image or mask.
        patch_size (int): The desired patch size.

    Returns:
        np.ndarray: The padded image/mask.
    """
    import cv2

    h, w = image.shape[:2]

    # Pad bottom and right only so existing pixel coordinates are preserved
    # and the model sees the same edge-zero pattern as during training.
    height_padding = (patch_size - h % patch_size) % patch_size
    width_padding = (patch_size - w % patch_size) % patch_size

    if height_padding == 0 and width_padding == 0:
        return image

    if image.ndim == 2:  # mask (grayscale)
        pad_value = 0
    else:  # image (color)
        pad_value = [0, 0, 0]

    return cv2.copyMakeBorder(image, 0, height_padding, 0, width_padding, cv2.BORDER_CONSTANT, value=pad_value)
