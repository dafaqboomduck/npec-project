from src.models.length import LengthResult, LengthCalculatorConfig

EMPTY_RESULT = LengthResult(detected=False, length=0, pixel_count=0)

# Return ID constants for calculate_geodesic_length
RETURN_EMPTY_MASK = 0
RETURN_SUCCESS = 1
RETURN_INSUFFICIENT_PIXELS = 2
RETURN_NO_PATH = 3

NUM_PLANTS = 5
MAX_GAP = 50
MAX_PATH = 50
USE_FARTHEST_ENDPOINT = True

LENGTH_CALCULATOR_CONFIG = LengthCalculatorConfig(
    num_plants=NUM_PLANTS,
    max_gap=MAX_GAP,
    max_path=MAX_PATH,
    use_farthest_endpoint=USE_FARTHEST_ENDPOINT,
)
