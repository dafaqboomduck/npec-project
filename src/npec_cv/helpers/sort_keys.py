"""Module for extracting sorting keys from plant identifiers."""

import re


def extract_sort_keys(plant_id):
    """Extract image number and plant number for sorting.

    Expected format: something_XX_plant_Y where XX is image number and Y is plant number.

    Args:
        plant_id (str): The plant identifier string.

    Returns:
        tuple: (image_number, plant_number)
    """
    parts = plant_id.split("_plant_")
    plant_num = int(parts[1])
    # Try to extract image number from the image name
    img_name = parts[0]
    # Find digits in the image name for sorting
    numbers = re.findall(r"\d+", img_name)
    img_num = int(numbers[-1]) if numbers else 0
    return (img_num, plant_num)
