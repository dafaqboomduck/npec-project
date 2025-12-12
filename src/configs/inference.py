from src.config import BASE_DIR
from src.models.inference_models import ModelConfig, MaskCombineStrategy
from src.inference.custom_functions import dice_bce_loss, f1, iou

# Patch size for the model
PATCH_SIZE = 512

# Path to the saved root detection model
ROOT_MODEL_PATH = (
    BASE_DIR / "src" / "artifacts" / "models" / "root_det_model_512px.keras"
)

# Path to the saved anchor detection model
ANCHOR_MODEL_PATH = (
    BASE_DIR / "src" / "artifacts" / "models" / "anchor_det_model_512px.keras"
)

# Custom loss functions and metrics for model loading
CUSTOM_OBJECTS = {"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}

# Ensamble strategy used for combining masks
ENSEMBLE_STRATEGY = MaskCombineStrategy.UNION

# Model configurations
ROOT_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ROOT_MODEL_PATH),
    threshold=0.5,  # 50% confidence threshold
    patch_size=512,
    custom_objects=CUSTOM_OBJECTS,
)

ANCHOR_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ANCHOR_MODEL_PATH),
    threshold=0.01,  # Lower threshold for anchor detection
    patch_size=512,
    custom_objects=CUSTOM_OBJECTS,
)

INFERENCE_CONFIG = (ROOT_MODEL_CONFIG, ANCHOR_MODEL_CONFIG)
