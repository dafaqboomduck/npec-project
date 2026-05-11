"""Configuration settings for inference models."""

from pathlib import Path
from typing import Iterable

from src.npec_cv.inference.custom_functions import dice_bce_loss, f1, iou
from src.npec_cv.models.inference import MaskCombineStrategy, ModelConfig

# Patch size for the model
PATCH_SIZE = 512

# Canonical and legacy model filenames accepted inside artifact_dir
ROOT_MODEL_FILENAME = "root_det_model_512px.keras"
ANCHOR_MODEL_FILENAME = "anchor_det_model_512px.keras"

ROOT_MODEL_LEGACY_FILENAMES = ("root_model.keras", "root_model")
ANCHOR_MODEL_LEGACY_FILENAMES = ("anchor_model.keras", "anchor_model")

# Defaults intentionally keep only filenames (not package-relative paths).
# Call build_inference_config(artifact_dir) to resolve real checkpoint paths.
ROOT_MODEL_PATH = Path(ROOT_MODEL_FILENAME)
ANCHOR_MODEL_PATH = Path(ANCHOR_MODEL_FILENAME)

# Custom loss functions and metrics for model loading
CUSTOM_OBJECTS = {"f1": f1, "iou": iou, "dice_bce_loss": dice_bce_loss}

# Ensamble strategy used for combining masks
ENSEMBLE_STRATEGY = MaskCombineStrategy.UNION


def _resolve_model_path(artifact_dir: Path, names: Iterable[str], model_label: str) -> Path:
    """Resolve a model file path from accepted filenames in artifact_dir."""
    for name in names:
        candidate = artifact_dir / name
        if candidate.is_file():
            return candidate

    accepted = ", ".join(names)
    raise FileNotFoundError(f"Missing {model_label} model in '{artifact_dir}'. " f"Expected one of: {accepted}")


def build_inference_config(artifact_dir: str | Path) -> tuple[ModelConfig, ModelConfig]:
    """Build model configs by resolving checkpoints from artifact_dir.

    The directory must contain either canonical names from this module or
    accepted legacy names:
    - root: root_det_model_512px.keras or root_model(.keras)
    - anchor: anchor_det_model_512px.keras or anchor_model(.keras)
    """
    artifact_path = Path(artifact_dir).expanduser().resolve()
    if not artifact_path.exists() or not artifact_path.is_dir():
        raise ValueError(f"artifact_dir must be an existing directory: {artifact_dir}")

    root_model_path = _resolve_model_path(
        artifact_path,
        (ROOT_MODEL_FILENAME, *ROOT_MODEL_LEGACY_FILENAMES),
        "root",
    )
    anchor_model_path = _resolve_model_path(
        artifact_path,
        (ANCHOR_MODEL_FILENAME, *ANCHOR_MODEL_LEGACY_FILENAMES),
        "anchor",
    )

    root_cfg = ModelConfig(
        checkpoint=str(root_model_path),
        threshold=0.01,
        patch_size=512,
        batch_size=4,
        custom_objects=CUSTOM_OBJECTS,
    )

    anchor_cfg = ModelConfig(
        checkpoint=str(anchor_model_path),
        threshold=0.001,
        patch_size=512,
        batch_size=4,
        custom_objects=CUSTOM_OBJECTS,
    )

    return root_cfg, anchor_cfg


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
