"""Module for extracting the petri dish region from an image."""

def extract_petri_dish(img, padding=30, return_coords: bool = False):
    """Identifies the largest object in an image and extracts a padded square region.

    Uses Otsu's thresholding and contour finding to find the petri dish. The extracted
    region is guaranteed to be a square and is clamped to the image boundaries.

    Args:
        img (np.ndarray): The input image (BGR or grayscale).
        padding (int, optional): Extra pixels to add around the bounding box
                                    to form the final square side. Defaults to 30.
        return_coords (bool, optional): If True, returns the coordinates (x_start,
                                        y_start, x_end, y_end) instead of the
                                        cropped image. Defaults to False.

    Returns:
        np.ndarray or tuple: The cropped square image, or a tuple of
                                (x_start, y_start, x_end, y_end) if
                                return_coords is True.
    """
    import cv2

    def bgr2gray(img):
        """Converts a BGR image to grayscale.

        Returns a copy if the image is already 2-dimensional (grayscale).

        Args:
            img (np.ndarray): The input image.

        Returns:
            np.ndarray: The grayscale image.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        return gray

    def calculate_square(img, rect, padding):
        """Calculates the coordinates for a padded square crop based on a bounding box.

        The resulting square is centered on the bounding box and its side length
        is max(width, height) + 2*padding. The coordinates are clamped to the
        image boundaries (0 to img.shape[0/1]).

        Args:
            img (np.ndarray): The full image array.
            rect (tuple): The bounding box (x, y, w, h) from cv2.boundingRect.
            padding (int): The padding to add on all sides.

        Returns:
            tuple: A tuple (x_start, y_start, x_end, y_end) representing the
                    final square crop coordinates.
        """
        x, y, w, h = rect

        # 1. Determine the largest dimension (W or H) for the square's side
        side = max(w, h)

        # 2. Calculate the center point of the original bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # 3. Calculate the new square coordinates based on the center and the side length
        padded_side = side + 2 * padding
        half_side = padded_side // 2

        # New top-left corner without image bounds check yet
        x_new = center_x - half_side
        y_new = center_y - half_side

        # 4. Define the final crop coordinates (x_start, y_start, x_end, y_end)
        # Calculate final coordinates, clamping them to the image boundaries
        x_start = max(0, x_new)
        y_start = max(0, y_new)

        # Calculate the *desired* bottom-right corner
        x_desired_end = x_new + padded_side
        y_desired_end = y_new + padded_side

        # Clamp the end coordinates to the image width/height
        x_end = min(img.shape[1], x_desired_end)
        y_end = min(img.shape[0], y_desired_end)

        # 5. Handle cases where clamping truncated the square (e.g., if the dish is at an edge)
        # Calculate the actual dimensions after clamping
        actual_w = x_end - x_start
        actual_h = y_end - y_start

        # Use the *smaller* of the actual dimensions as the final side length
        final_side = min(actual_w, actual_h)

        # Recalculate the end points to form a perfect square using 'final_side'
        # This part adjusts the *start* point if the image boundary cut off the *end* point.

        # If the X dimension was cut short, adjust x_start to maintain the square size
        if actual_w < padded_side and x_end == img.shape[1]:
            x_start = max(0, x_end - final_side)

        # If the Y dimension was cut short, adjust y_start to maintain the square size
        if actual_h < padded_side and y_end == img.shape[0]:
            y_start = max(0, y_end - final_side)

        # Ensure the final square uses the calculated 'final_side'
        # (This is mostly a safety check after the boundary adjustments)
        x_end = x_start + final_side
        y_end = y_start + final_side

        return x_start, y_start, x_end, y_end

    # Convert to grayscale for easier processing
    gray = bgr2gray(img)

    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the Largest Contour
    # Safety: handle case where no contours are found (shouldn't happen with Otsu)
    if not contours:
        return None if not return_coords else (0, 0, img.shape[1], img.shape[0])

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the Bounding Box
    # x, y are the top-left coordinates, w is width, h is height
    rect = cv2.boundingRect(largest_contour)

    # Calculate padded coordinates, ensuring they don't go out of image bounds
    x_start, y_start, x_end, y_end = calculate_square(img, rect, padding)

    cropped_image = img[y_start:y_end, x_start:x_end]
    if not return_coords:
        return cropped_image
    else:
        return x_start, y_start, x_end, y_end
