from src.config import BASE_DIR
from src.models.inference_models import MaskCombineStrategy

PATCH_SIZE = 512
MODEL_PATH = BASE_DIR / "src" / "artifacts" / "models" / "root_det_model_512px.keras"
ENSEMBLE_STRATEGY = MaskCombineStrategy.UNION