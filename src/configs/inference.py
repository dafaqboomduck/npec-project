from src.config import BASE_DIR
from src.models.inference_models import MaskCombineStrategy

PATCH_SIZE = 512
# ROOT_MODEL_PATH = (
#     BASE_DIR / "src" / "artifacts" / "models" / "distilled_student_model.keras"
# )
ROOT_MODEL_PATH = (
    BASE_DIR / "src" / "artifacts" / "models" / "root_det_model_512px.keras"
)
ANCHOR_MODEL_PATH = (
    BASE_DIR / "src" / "artifacts" / "models" / "anchor_det_model_512px.keras"
)
ENSEMBLE_STRATEGY = MaskCombineStrategy.UNION

