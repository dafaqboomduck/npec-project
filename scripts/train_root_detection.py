"""Azure ML training script — root detection U-Net trained from scratch.

Usage (submitted by Azure ML, not run manually):
    python scripts/train_root_detection.py
        --data_dir  <mounted path to cv5_train>
        --output_dir <path where trained model is saved>
        [--val_dir  <mounted path to cv5_val>]
        [--patch_size 512]
        [--epochs 30]
        [--batch_size 16]

If --val_dir is provided the pre-split validation set is used directly.
If omitted the training data is split 80/20 internally.
"""

import argparse
import os
import shutil
import sys
import types
from glob import glob
from pathlib import Path

import cv2
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Make repo packages importable (Azure ML sets CWD to the uploaded code root).
# cv2/tf/sklearn are in the conda env and don't need this; only local repo
# modules do. The npec_cv stub prevents its heavy __init__.py from running.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO_ROOT))

_npec_stub = types.ModuleType("npec_cv")
_npec_stub.__path__ = [str(_SRC / "npec_cv")]
_npec_stub.__package__ = "npec_cv"
sys.modules.setdefault("npec_cv", _npec_stub)

# ── Hyperparameters (overridable via args) ────────────────────────────────────
IMG_CHANNELS = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training job."""
    parser = argparse.ArgumentParser("train_root_detection")
    parser.add_argument("--data_dir", required=True, help="Mounted path to cv5_train")
    parser.add_argument("--val_dir", default=None, help="Mounted path to cv5_val (optional pre-split validation set)")
    parser.add_argument(
        "--corrections_dir",
        default=None,
        help="Mounted path to cv5_corrections (optional set of corrected annotations)",
    )
    parser.add_argument("--output_dir", default="outputs", help="Directory to write the trained model")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of source images used (None = use all). Set low for fast test runs.",
    )
    return parser.parse_args()


def setup_gpu() -> None:
    """Enable dynamic GPU memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs found: {len(gpus)}")
    else:
        print("No GPU found — training on CPU")


# ── Data generator ────────────────────────────────────────────────────────────


class DataGenerator(keras.utils.Sequence):
    """Loads image/mask patches from disk in batches — no full dataset in memory."""

    def __init__(self, image_dir, mask_dir, batch_size=16, img_size=(512, 512), shuffle=True):
        """Initialise the generator with image and mask directories."""
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

        self.image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))

        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        if len(self.mask_paths) == 0:
            self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")

        if len(self.image_paths) != len(self.mask_paths):
            print("WARNING: Number of images and masks don't match!")

        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Return one batch of (images, masks)."""
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        return self.__data_generation(indices)

    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch."""
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        """Load and return a batch of images and masks."""
        X = np.empty((self.batch_size, *self.img_size, IMG_CHANNELS), dtype=np.float32)
        y = np.empty((self.batch_size, *self.img_size, 1), dtype=np.float32)

        for i, idx in enumerate(indices):
            img = cv2.imread(self.image_paths[idx])
            if img is None:
                print(f"Failed to load: {self.image_paths[idx]}")
                img = np.zeros((*self.img_size, IMG_CHANNELS), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[:2] != self.img_size:
                img = cv2.resize(img, self.img_size)

            X[i] = img.astype(np.float32) / 255.0

            mask = cv2.imread(self.mask_paths[idx], 0)
            if mask is None:
                print(f"Failed to load: {self.mask_paths[idx]}")
                mask = np.zeros(self.img_size, dtype=np.uint8)

            if mask.shape != self.img_size:
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            y[i] = np.expand_dims((mask > 0).astype(np.float32), axis=-1)

        return X, y


# ── Model architecture ────────────────────────────────────────────────────────


def build_unet(patch_size: int) -> keras.Model:
    """Build and return a U-Net model trained from random initialisation."""
    inputs = layers.Input((patch_size, patch_size, IMG_CHANNELS))

    # Encoder
    c1 = layers.Conv2D(16, 3, activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = layers.Conv2D(128, 3, activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = layers.Conv2D(256, 3, activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

    return keras.Model(inputs=[inputs], outputs=[outputs])


# ── Custom loss & metric ──────────────────────────────────────────────────────


class F1Score(keras.metrics.Metric):
    """Pixel-level F1 score metric for binary segmentation."""

    def __init__(self, threshold=0.5, **kwargs):
        """Initialise F1Score metric with prediction threshold."""
        super().__init__(**kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate TP, FP, FN counts."""
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        """Return F1 score from accumulated counts."""
        eps = keras.backend.epsilon()
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        return 2 * precision * recall / (precision + recall + eps)

    def reset_state(self):
        """Reset all accumulators."""
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


def dice_loss(y_true, y_pred):
    """Dice loss for handling class imbalance in segmentation."""
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def combined_loss(y_true, y_pred):
    """Binary cross-entropy + Dice loss."""
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# ── Patchify + split helpers ──────────────────────────────────────────────────

_IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def _pad_to_multiple(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Zero-pad array so H and W are multiples of patch_size."""
    h, w = arr.shape[:2]
    ph = ((h + patch_size - 1) // patch_size) * patch_size
    pw = ((w + patch_size - 1) // patch_size) * patch_size
    out = np.zeros((ph, pw) + arr.shape[2:], dtype=arr.dtype)
    out[:h, :w] = arr
    return out


def _find_data_dirs(data_dir: str) -> tuple[str, str]:
    """Return (img_dir, mask_dir), checking Y2B_24 subfolder first."""
    for subdir in ["Y2B_24", ""]:
        base = os.path.join(data_dir, subdir) if subdir else data_dir
        img_d = os.path.join(base, "images")
        msk_d = os.path.join(base, "masks")
        if os.path.isdir(img_d) and os.path.isdir(msk_d):
            return img_d, msk_d
    raise FileNotFoundError(f"Cannot find images/ and masks/ inside {data_dir}.\n" f"Contents: {os.listdir(data_dir)}")


def patchify_dir(
    data_dir: str, out_img_dir: str, out_msk_dir: str, patch_size: int, max_images: int | None = None
) -> int:
    """Patchify all image/mask pairs from data_dir into out_img_dir / out_msk_dir.

    Returns total number of patches written.
    """
    img_dir, mask_dir = _find_data_dirs(data_dir)
    print(f"Images : {img_dir}")
    print(f"Masks  : {mask_dir}")

    img_files = {Path(f).stem.lower(): f for f in os.listdir(img_dir) if Path(f).suffix.lower() in _IMG_EXTS}

    msk_files: dict[str, str] = {}
    for root, _, files in os.walk(mask_dir):
        for f in files:
            if Path(f).suffix.lower() in _IMG_EXTS:
                msk_files[Path(f).stem.lower()] = os.path.join(root, f)

    print(f"Masks dir contents (top level): {os.listdir(mask_dir)[:10]}")
    print(f"Image files found : {len(img_files)}  e.g. {list(img_files.keys())[:3]}")
    print(f"Mask  files found : {len(msk_files)}  e.g. {list(msk_files.keys())[:3]}")

    valid_pairs: list[tuple[str, str]] = []
    for stem, img_fname in sorted(img_files.items()):
        if stem in msk_files:
            valid_pairs.append((img_fname, msk_files[stem]))
        elif stem + "_root_mask" in msk_files:
            valid_pairs.append((img_fname, msk_files[stem + "_root_mask"]))

    print(f"Valid pairs: {len(valid_pairs)}")
    if not valid_pairs:
        raise ValueError(
            "No image-mask pairs matched.\n"
            f"Image stems (sample): {list(img_files.keys())[:5]}\n"
            f"Mask stems  (sample): {list(msk_files.keys())[:5]}"
        )

    if max_images is not None:
        valid_pairs = valid_pairs[:max_images]
        print(f"max_images={max_images} — using {len(valid_pairs)} image(s) for this run.")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    total = 0
    for img_fname, msk_fname in valid_pairs:
        img = cv2.imread(os.path.join(img_dir, img_fname))
        msk = cv2.imread(msk_fname, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print(f"  Skipping {img_fname} — failed to load")
            continue
        img = _pad_to_multiple(img, patch_size)
        msk = _pad_to_multiple(msk, patch_size)
        h, w = img.shape[:2]
        stem = Path(img_fname).stem
        idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                name = f"{stem}_patch_{idx}.png"
                cv2.imwrite(os.path.join(out_img_dir, name), img[i : i + patch_size, j : j + patch_size])
                cv2.imwrite(os.path.join(out_msk_dir, name), msk[i : i + patch_size, j : j + patch_size])
                idx += 1
        total += idx

    print(f"Patches written: {total}")
    if total == 0:
        raise ValueError("Patchify produced 0 patches.")
    return total


def patchify_presplit(
    train_dir: str, val_dir: str, patch_dir: str, patch_size: int, max_images: int | None = None
) -> tuple[str, str, str, str]:
    """Patchify pre-split train and val directories separately.

    Returns (tr_img, tr_msk, vl_img, vl_msk).
    """
    tr_img = os.path.join(patch_dir, "train_images")
    tr_msk = os.path.join(patch_dir, "train_masks")
    vl_img = os.path.join(patch_dir, "val_images")
    vl_msk = os.path.join(patch_dir, "val_masks")

    print("Patchifying training set...")
    patchify_dir(train_dir, tr_img, tr_msk, patch_size, max_images=max_images)
    print("Patchifying validation set...")
    patchify_dir(val_dir, vl_img, vl_msk, patch_size, max_images=max_images)

    return tr_img, tr_msk, vl_img, vl_msk


def patchify_and_split(
    data_dir: str, patch_dir: str, patch_size: int, max_images: int | None = None
) -> tuple[str, str, str, str]:
    """Patchify a single data directory and split 80/20 into train/val.

    Used as fallback when no pre-split val set is provided.
    Returns (tr_img, tr_msk, vl_img, vl_msk).
    """
    raw_img = os.path.join(patch_dir, "all_images")
    raw_msk = os.path.join(patch_dir, "all_masks")
    patchify_dir(data_dir, raw_img, raw_msk, patch_size, max_images=max_images)

    all_files = sorted(os.listdir(raw_img))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(f"Internal split — Train: {len(train_files)}  Val: {len(val_files)}")

    tr_img = os.path.join(patch_dir, "train_images")
    tr_msk = os.path.join(patch_dir, "train_masks")
    vl_img = os.path.join(patch_dir, "val_images")
    vl_msk = os.path.join(patch_dir, "val_masks")

    for d in [tr_img, tr_msk, vl_img, vl_msk]:
        os.makedirs(d, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(raw_img, f), os.path.join(tr_img, f))
        shutil.copy(os.path.join(raw_msk, f), os.path.join(tr_msk, f))
    for f in val_files:
        shutil.copy(os.path.join(raw_img, f), os.path.join(vl_img, f))
        shutil.copy(os.path.join(raw_msk, f), os.path.join(vl_msk, f))

    return tr_img, tr_msk, vl_img, vl_msk


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full patchify → train → save pipeline."""
    args = parse_args()
    setup_gpu()

    patch_dir = "/tmp/patches_512px"

    # 1. Patchify + split
    if args.val_dir:
        print("Using pre-split validation set.")
        tr_img, tr_msk, vl_img, vl_msk = patchify_presplit(
            args.data_dir, args.val_dir, patch_dir, args.patch_size, max_images=args.max_images
        )
    else:
        print("No val_dir provided — splitting training data 80/20 internally.")
        tr_img, tr_msk, vl_img, vl_msk = patchify_and_split(
            args.data_dir, patch_dir, args.patch_size, max_images=args.max_images
        )

    if args.corrections_dir and os.path.isdir(args.corrections_dir):
        try:
            added = patchify_dir(args.corrections_dir, tr_img, tr_msk, args.patch_size)
            print(f"Merged {added} correction patch(es) into the training set.")
        except (FileNotFoundError, ValueError) as exc:
            print(f"No corrections merged ({exc}).")  # empty/seed asset -> train on base only

    # 2. Data generators
    print("Creating data generators...")
    train_gen = DataGenerator(tr_img, tr_msk, batch_size=args.batch_size, img_size=(args.patch_size, args.patch_size))
    val_gen = DataGenerator(
        vl_img, vl_msk, batch_size=args.batch_size, img_size=(args.patch_size, args.patch_size), shuffle=False
    )
    print(f"Training batches: {len(train_gen)}  Validation batches: {len(val_gen)}")

    # 3. Build model from scratch
    print("Building U-Net from scratch...")
    model = build_unet(args.patch_size)
    model.compile(
        optimizer="adam",
        loss=combined_loss,
        metrics=["accuracy", F1Score(name="f1_score")],
    )
    model.summary()

    # 4. Callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "root_det_model_512px.keras")

    class MlflowEpochMetricsCallback(keras.callbacks.Callback):
        """Logs every Keras metric to MLflow at the end of each epoch.

        Workaround for mlflow.tensorflow.autolog() not capturing custom
        Metric subclasses (e.g. our F1Score) on tf 2.16 / mlflow 2.x.
        Without this callback the Studio's Metrics tab only shows the two
        final scalars logged after fit() — no per-epoch curves.
        """

        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return
            for name, value in logs.items():
                try:
                    mlflow.log_metric(name, float(value), step=epoch)
                except Exception:
                    # Don't let a logging failure crash training.
                    pass

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_model_path, monitor="val_f1_score", mode="max", verbose=1, save_best_only=True
        ),
        keras.callbacks.EarlyStopping(patience=5, monitor="val_f1_score", mode="max", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        MlflowEpochMetricsCallback(),
    ]

    # 5. Train
    print("\nStarting training...")
    with mlflow.start_run():
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("patch_size", args.patch_size)
        mlflow.tensorflow.autolog()

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        best_val_f1 = float(max(history.history["val_f1_score"]))
        best_val_loss = float(min(history.history["val_loss"]))
        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.log_metric("best_val_loss", best_val_loss)

    print(f"\nBest val_f1  : {best_val_f1:.4f}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved  → {best_model_path}")

    # 6. Cleanup
    shutil.rmtree(patch_dir, ignore_errors=True)
    print("Temp files cleaned up.")


if __name__ == "__main__":
    main()
