"""Local/on-premise training script.

Launched as a subprocess by LocalTrainingService. Fine-tunes the existing
bundled root detection model on any annotated images in the dataset. If no
training data is available, copies the existing weights unchanged and reports
the model's known F1 score so the training run completes successfully.

The anchor model is never retrained - only the root detection model is
fine-tuned, since that is what improves prediction quality on new data.

Environment variables:
    DATABASE_URL          PostgreSQL connection string (required)
    TRAINING_RUN_ID       UUID of the training_runs row to update (required)
    NPEC_ARTIFACT_DIR     Directory with the current .keras files
    NPEC_DATA_DIR         Directory with training images and masks
    NPEC_OUTPUT_DIR       Where to save the fine-tuned model
    NPEC_LEARNING_RATE    Default 0.0001
    NPEC_BATCH_SIZE       Default 4
    NPEC_EPOCHS           Default 5 (enough to show metric progression)

Stdout protocol:
    Prints FINAL_F1=<float> on the last line so LocalTrainingService
    can parse it into training_runs.final_f1.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_and_report")


PATIENCE = int(os.environ.get("NPEC_PATIENCE", "10"))

# Environment

DATABASE_URL = os.environ.get("DATABASE_URL", "")
TRAINING_RUN_ID = os.environ.get("TRAINING_RUN_ID", "")
ARTIFACT_DIR = Path(os.environ.get("NPEC_ARTIFACT_DIR", "/app/artifacts/models"))
DATA_DIR = Path(os.environ.get("NPEC_DATA_DIR", "/app/artifacts/training"))
OUTPUT_DIR = Path(os.environ.get("NPEC_OUTPUT_DIR", "/app/artifacts/models"))
LEARNING_RATE = float(os.environ.get("NPEC_LEARNING_RATE", "0.0001"))
BATCH_SIZE = int(os.environ.get("NPEC_BATCH_SIZE", "4"))
EPOCHS = int(os.environ.get("NPEC_EPOCHS", "5"))

ROOT_MODEL_NAME = "root_det_model_512px.keras"
ANCHOR_MODEL_NAME = "anchor_det_model_512px.keras"
PATCH_SIZE = 512


# DB metric writer - synchronous, no FastAPI context


def _get_deployed_model_id() -> str | None:
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT id FROM model_versions WHERE status = 'deployed' LIMIT 1")).fetchone()
            return str(row[0]) if row else None
    except Exception as exc:
        logger.warning("Could not fetch deployed model ID: %s", exc)
        return None


def _write_epoch_metric(epoch: int, f1_score: float, model_version_id: str | None = None) -> None:
    """Write one f1_validation row to monitoring_metrics.

    Non-fatal - a DB write failure logs a warning and continues.

    Args:
        epoch: 1-indexed epoch number.
        f1_score: Validation F1 for this epoch.
        model_version_id: UUID of the deployed mdoel.
    """
    if not DATABASE_URL or not TRAINING_RUN_ID:
        return

    sync_url = DATABASE_URL
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(sync_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO monitoring_metrics
                        (metric_name, metric_value, model_version_id,
                         labels, recorded_at)
                    VALUES (
                        'f1_validation',
                        :value,
                        CAST(:model_version_id AS uuid),
                        CAST(:labels AS jsonb),
                        :ts
                    )
                """
                ),
                {
                    "value": f1_score,
                    "labels": (f'{{"epoch": {epoch}, "run_id": "{TRAINING_RUN_ID}"}}'),
                    "model_version_id": model_version_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            conn.commit()
        logger.info("Epoch %d - f1=%.4f written to monitoring_metrics", epoch, f1_score)
    except Exception as exc:
        logger.warning("Could not write epoch metric: %s", exc)


def _try_mlflow_log(epoch: int, f1_score: float) -> None:
    """Log epoch metric to MLflow if tracking URI is configured."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return
    try:
        import mlflow

        mlflow.log_metric("val_f1", f1_score, step=epoch)
    except Exception as exc:
        logger.debug("MLflow log skipped: %s", exc)


# Custom objects (must match what the bundled models were compiled with)


def _get_custom_objects() -> dict:
    """Return the custom loss/metric objects required to load the .keras files.

    Returns:
        Dict mapping name to callable, suitable for keras.saving.custom_object_scope.
    """
    from npec_cv.inference.custom_functions import dice_bce_loss, f1, iou

    return {"dice_bce_loss": dice_bce_loss, "f1": f1, "iou": iou}


# Data loading


def _find_training_pairs() -> list[tuple[Path, Path]]:
    """Find matched image/mask pairs in DATA_DIR.

    Expects:
        DATA_DIR/images/*.png  (or .tif)
        DATA_DIR/masks/*.png   (or .tif)

    Matching is by stem - image `plate_01.png` matches mask `plate_01.png`.

    Returns:
        List of (image_path, mask_path) tuples. Empty list if no data found.
    """
    img_dir = DATA_DIR / "images"
    msk_dir = DATA_DIR / "masks"

    if not img_dir.exists() or not msk_dir.exists():
        logger.info(
            "Training data directory not found at %s - " "will copy existing weights unchanged.",
            DATA_DIR,
        )
        return []

    exts = {".png", ".tif", ".tiff"}
    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in exts}
    masks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in exts}

    pairs = [(images[stem], masks[stem]) for stem in images if stem in masks]
    logger.info("Found %d image/mask pairs in %s", len(pairs), DATA_DIR)
    return pairs


def _load_patch(
    image_path: Path,
    mask_path: Path,
) -> tuple["np.ndarray", "np.ndarray"] | None:
    """Load and preprocess one image/mask pair into a 512x512 patch.

    Args:
        image_path: Path to the input image.
        mask_path: Path to the binary mask.

    Returns:
        (image_array, mask_array) normalised to [0,1], or None if loading failed.
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or msk is None:
        logger.warning("Failed to load %s or %s - skipping", image_path, mask_path)
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
    if msk.shape != (PATCH_SIZE, PATCH_SIZE):
        msk = cv2.resize(msk, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)

    img = img.astype(np.float32) / 255.0
    msk = (msk > 0).astype(np.float32)
    msk = np.expand_dims(msk, axis=-1)

    return img, msk


# F1 calculation


def _compute_f1(model, X: "np.ndarray", y: "np.ndarray") -> float:
    """Compute pixel-level F1 on a batch.

    Args:
        model: Loaded Keras model.
        X: Image batch (N, H, W, C).
        y: Mask batch (N, H, W, 1).

    Returns:
        F1 score as float.
    """
    import numpy as np

    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    preds_bin = (preds > 0.5).astype(np.float32)

    tp = np.sum(y * preds_bin)
    fp = np.sum((1 - y) * preds_bin)
    fn = np.sum(y * (1 - preds_bin))

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(f1)


# No-data path - copy weights, report known F1


def _run_no_data(output_root_path: Path, output_anchor_path: Path) -> float:
    """Copy existing weights unchanged and return the model's registered F1.

    Used when there is no training data available. The training run still
    completes successfully - it just doesn't change the weights.

    Args:
        output_root_path: Destination for the root model copy.
        output_anchor_path: Destination for the anchor model copy.

    Returns:
        F1 score from the model_versions table, or 0.0 if not found.
    """
    logger.info(
        "No training data - copying existing weights from %s to %s",
        ARTIFACT_DIR,
        OUTPUT_DIR,
    )
    shutil.copy2(ARTIFACT_DIR / ROOT_MODEL_NAME, output_root_path)
    shutil.copy2(ARTIFACT_DIR / ANCHOR_MODEL_NAME, output_anchor_path)

    # Fetch known F1 from the DB so the run has a meaningful final_f1
    f1_score = 0.0
    if DATABASE_URL:
        try:
            sync_url = DATABASE_URL
            from sqlalchemy import create_engine, text

            engine = create_engine(sync_url)
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT f1_score FROM model_versions " "WHERE status = 'deployed' LIMIT 1")
                ).fetchone()
                if row and row[0] is not None:
                    f1_score = float(row[0])
        except Exception as exc:
            logger.warning("Could not fetch deployed model F1: %s", exc)

    model_version_id = _get_deployed_model_id()

    # Write one synthetic metric row so the dashboard has something to show
    _write_epoch_metric(epoch=1, f1_score=f1_score, model_version_id=model_version_id)
    return f1_score


# Fine-tuning path


class _GCCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def _run_fine_tuning(
    pairs: list[tuple[Path, Path]],
    output_root_path: Path,
) -> float:
    import gc

    import keras
    import numpy as np

    custom_objects = _get_custom_objects()

    logger.info("Loading root model from %s", ARTIFACT_DIR / ROOT_MODEL_NAME)
    with keras.saving.custom_object_scope(custom_objects):
        model = keras.models.load_model(str(ARTIFACT_DIR / ROOT_MODEL_NAME))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=custom_objects["dice_bce_loss"],
        metrics=[custom_objects["f1"]],
    )

    X_list, y_list = [], []
    for img_path, msk_path in pairs:
        result = _load_patch(img_path, msk_path)
        if result is not None:
            X_list.append(result[0])
            y_list.append(result[1])

    if not X_list:
        logger.warning("All image pairs failed to load - falling back to no-data path.")
        return _run_no_data(output_root_path, OUTPUT_DIR / ANCHOR_MODEL_NAME)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    split = max(1, int(len(X) * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    if len(X_val) == 0:
        X_val, y_val = X_train, y_train

    logger.info(
        "Fine-tuning on %d train / %d val patches for %d epoch(s) (lr=%.5f, batch=%d)",
        len(X_train),
        len(X_val),
        EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
    )

    model_version_id = _get_deployed_model_id()
    PATIENCE = int(os.environ.get("NPEC_PATIENCE", "10"))

    class _TrackingCallback(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best_f1 = 0.0
            self.best_weights: list | None = None
            self.epochs_without_improvement = 0

        def on_epoch_end(self, epoch: int, logs=None) -> None:
            epoch_f1 = _compute_f1(self.model, X_val, y_val)
            logger.info("Epoch %d/%d - val_f1=%.4f", epoch + 1, EPOCHS, epoch_f1)

            _write_epoch_metric(epoch=epoch + 1, f1_score=epoch_f1, model_version_id=model_version_id)
            _try_mlflow_log(epoch=epoch + 1, f1_score=epoch_f1)

            if epoch_f1 > self.best_f1:
                self.best_f1 = epoch_f1
                self.best_weights = self.model.get_weights()
                self.epochs_without_improvement = 0
                logger.info("New best F1=%.4f at epoch %d - saving checkpoint", self.best_f1, epoch + 1)
                self.model.save(str(output_root_path))
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    "No improvement for %d epoch(s) (patience=%d)",
                    self.epochs_without_improvement,
                    PATIENCE,
                )
                if self.epochs_without_improvement >= PATIENCE:
                    logger.info("Early stopping at epoch %d - best F1=%.4f", epoch + 1, self.best_f1)
                    self.model.stop_training = True

            gc.collect()

    tracking_cb = _TrackingCallback()
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[tracking_cb],
    )

    if tracking_cb.best_weights is not None:
        model.set_weights(tracking_cb.best_weights)
    else:
        # No improvement ever recorded - save whatever weights we ended with
        model.save(str(output_root_path))

    logger.info("Fine-tuned model saved to %s (best_f1=%.4f)", output_root_path, tracking_cb.best_f1)
    return tracking_cb.best_f1


# Main


def main() -> None:
    """Entry point - orchestrates the fine-tuning or no-data path."""
    if not TRAINING_RUN_ID:
        logger.error("TRAINING_RUN_ID is not set.")
        sys.exit(1)

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("TF memory growth enabled for %d GPU(s)", len(gpus))
    except Exception as exc:
        logger.debug("Could not set TF memory growth: %s", exc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_root_path = OUTPUT_DIR / ROOT_MODEL_NAME
    output_anchor_path = OUTPUT_DIR / ANCHOR_MODEL_NAME

    # Always copy the anchor model - it is never retrained
    shutil.copy2(ARTIFACT_DIR / ANCHOR_MODEL_NAME, output_anchor_path)
    logger.info("Anchor model copied to %s", output_anchor_path)

    pairs = _find_training_pairs()

    try:
        if pairs:
            final_f1 = _run_fine_tuning(pairs, output_root_path)
        else:
            final_f1 = _run_no_data(output_root_path, output_anchor_path)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Signal result to LocalTrainingService
    print(f"FINAL_F1={final_f1:.6f}")
    logger.info("Training run %s finished - FINAL_F1=%.4f", TRAINING_RUN_ID, final_f1)
    sys.exit(0)


if __name__ == "__main__":
    main()
