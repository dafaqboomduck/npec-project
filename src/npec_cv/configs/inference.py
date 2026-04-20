from pathlib import Path
from npec_cv.models.inference import ModelConfig, MaskCombineStrategy
from npec_cv.inference.custom_functions import dice_bce_loss, f1, iou

# Patch size for the model
PATCH_SIZE = 512

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_ARTIFACTS_DIR = _PACKAGE_DIR / "artifacts" / "models"

# Path to the saved root detection model
ROOT_MODEL_PATH = _ARTIFACTS_DIR / "root_det_model_512px.keras"

# Path to the saved anchor detection model
ANCHOR_MODEL_PATH = _ARTIFACTS_DIR / "anchor_det_model_512px.keras"

# Custom loss functions and metrics for model loading
CUSTOM_OBJECTS = {"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}

# Ensamble strategy used for combining masks
ENSEMBLE_STRATEGY = MaskCombineStrategy.UNION

# Model configurations
ROOT_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ROOT_MODEL_PATH),
    threshold=0.01,
    patch_size=512,
    batch_size=4,
    custom_objects=CUSTOM_OBJECTS,
)

ANCHOR_MODEL_CONFIG = ModelConfig(
    checkpoint=str(ANCHOR_MODEL_PATH),
    threshold=0.001,
    patch_size=512,
    batch_size=4,
    custom_objects=CUSTOM_OBJECTS,
)

INFERENCE_CONFIG = (ROOT_MODEL_CONFIG, ANCHOR_MODEL_CONFIG)
