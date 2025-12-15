from src.models.length_models import LengthResult

EMPTY_RESULT = LengthResult(detected=False, length=0, pixel_count=0)

# Return ID constants for calculate_geodesic_length
RETURN_EMPTY_MASK = 0
RETURN_SUCCESS = 1
RETURN_INSUFFICIENT_PIXELS = 2
RETURN_NO_PATH = 3