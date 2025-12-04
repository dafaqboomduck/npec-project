from src.config import BASE_DIR
from src.models.inference_models import MaskCombineStrategy

PATCH_SIZE = 512
MODEL_PATH = BASE_DIR / "src" / "artifacts" / "models" / "unet_model_512px_v1.keras"
SECONDARY_MODEL_PATH = BASE_DIR / "src" / "artifacts" / "models" / "unet_model_256px_v1.keras"
ENSEMBLE_STRATEGY = MaskCombineStrategy.UNION