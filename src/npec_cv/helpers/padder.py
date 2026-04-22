"""Module for padding images and masks."""


def padder(image, patch_size):
    """Pads an image (or mask) so that its dimensions are multiples of patch_size.

    Padding is distributed evenly on all sides.

    Args:
        image (np.ndarray): The input image or mask.
        patch_size (int): The desired patch size.

    Returns:
        np.ndarray: The padded image/mask.
    """
    import cv2

    h, w = image.shape[:2]

    # Calculate the required padding to make dimensions a multiple of patch_size
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    # Distribute padding to top/bottom, left/right
    top = height_padding // 2
    bottom = height_padding - top
    left = width_padding // 2
    right = width_padding - left

    # Handle channels for pad value
    if image.ndim == 2:  # mask (grayscale)
        pad_value = 0
    else:  # image (color)
        pad_value = [0, 0, 0]

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
