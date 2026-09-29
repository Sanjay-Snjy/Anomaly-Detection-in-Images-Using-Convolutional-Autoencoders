"""
Anomaly Detection in Images Using Convolutional Autoencoders
=============================================================

Core reusable pipeline for the MVTec AD dataset (category: `screw`).

Pipeline:
    MVTec AD -> load normal training images -> resize -> normalize
    -> train/validation split -> CNN encoder -> latent representation
    -> CNN decoder -> reconstructed image -> reconstruction error
    -> anomaly threshold (from normal validation data only)
    -> NORMAL / ANOMALY -> anomaly heatmap -> evaluation.

Design rules honoured by this module:
    * The autoencoder is trained ONLY on `train/good` images.
    * Test images / ground-truth masks are NEVER used for training.
    * The anomaly threshold is derived from NORMAL VALIDATION data only
      (no test labels are used to tune it).
    * The exact same preprocessing is used for training and inference.

Author: (student project) — TensorFlow/Keras implementation.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnomalyDetectionConfig:
    """Central, configurable settings for the whole pipeline.

    Nothing is hard-coded downstream: every script reads its knobs from
    here (or from CLI overrides of these values).
    """

    category: str = "screw"                 # MVTec AD category to use
    dataset_root: Path = Path("mvtec_anomaly_detection")
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")

    image_size: int = 128                   # resize target (square)
    channels: int = 1                       # screw images are grayscale
    pixel_range: Tuple[float, float] = (0.0, 1.0)

    validation_split: float = 0.2           # of the normal TRAIN images
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3

    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    # Image-level threshold strategy, computed from NORMAL VALIDATION data:
    #   "mean_std"    -> mean + n_std * std           (spec baseline)
    #   "percentile"  -> given percentile of normal scores
    #   "max"         -> max normal validation score (no normal image flagged)
    threshold_strategy: str = "mean_std"
    threshold_n_std: float = 3.0
    threshold_percentile: float = 99.0
    # Pixel-level mask threshold uses the same strategy on normal val
    # per-pixel errors (mask_n_std / mask_percentile).
    mask_n_std: float = 3.0
    mask_percentile: float = 99.0

    seed: int = 42

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def category_root(self) -> Path:
        return self.dataset_root / self.category

    @property
    def train_good_dir(self) -> Path:
        return self.category_root / "train" / "good"

    @property
    def test_dir(self) -> Path:
        return self.category_root / "test"

    @property
    def ground_truth_dir(self) -> Path:
        return self.category_root / "ground_truth"

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.category}_autoencoder.keras"

    @property
    def threshold_path(self) -> Path:
        return self.models_dir / f"{self.category}_threshold.json"

    @property
    def heatmaps_dir(self) -> Path:
        return self.results_dir / "heatmaps"

    def to_dict(self) -> Dict:
        """JSON-serializable view of the config (for logging runs)."""
        return {
            "category": self.category,
            "dataset_root": str(self.dataset_root),
            "models_dir": str(self.models_dir),
            "results_dir": str(self.results_dir),
            "image_size": self.image_size,
            "channels": self.channels,
            "validation_split": self.validation_split,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "threshold_strategy": self.threshold_strategy,
            "threshold_n_std": self.threshold_n_std,
            "threshold_percentile": self.threshold_percentile,
            "mask_n_std": self.mask_n_std,
            "mask_percentile": self.mask_percentile,
            "seed": self.seed,
        }

    def ensure_output_dirs(self) -> None:
        """Create models/ and results/ directories if missing."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility & environment
# ---------------------------------------------------------------------------


def set_seeds(seed: int = 42) -> None:
    """Seed python, numpy and tensorflow RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def check_gpu() -> List:
    """Print and return the GPUs TensorFlow can see.

    In Colab: Runtime -> Change runtime type -> Hardware accelerator: GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow version:", tf.__version__)
    if gpus:
        for gpu in gpus:
            print("GPU detected:", gpu)
    else:
        print(
            "WARNING: No GPU detected. Training will be slow.\n"
            "In Google Colab enable: Runtime > Change runtime type > GPU."
        )
    return gpus


def validate_dataset(config: AnomalyDetectionConfig) -> None:
    """Raise a helpful error if the expected MVTec structure is missing."""
    if not config.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: '{config.dataset_root.resolve()}'. "
            "Extract mvtec_anomaly_detection.tar.xz or point "
            "AnomalyDetectionConfig.dataset_root at the extracted folder."
        )
    if not config.train_good_dir.exists():
        raise FileNotFoundError(
            f"Expected normal training images at "
            f"'{config.train_good_dir}' but the folder does not exist. "
            f"Expected structure: {config.category}/train/good/*.png"
        )
    if not config.test_dir.exists():
        raise FileNotFoundError(
            f"Expected test images at '{config.test_dir}'. "
            f"Expected structure: {config.category}/test/<defect_type>/*.png"
        )


# ---------------------------------------------------------------------------
# Image loading & preprocessing (single source of truth)
# ---------------------------------------------------------------------------


def load_image(image_path: Path | str, image_size: int) -> np.ndarray:
    """Load an image from disk and return a preprocessed float32 array.

    Steps (shared by training AND inference — do not duplicate):
        1. Read as grayscale (MVTec `screw` images are single-channel).
        2. Resize to (image_size, image_size) with bilinear interpolation.
        3. Scale pixels to the 0-1 float range.
        4. Add a channel dimension -> (H, W, 1).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be decoded as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    # IMREAD_GRAYSCALE: screw PNGs are 8-bit grayscale; this also makes the
    # loader robust if an RGB image is ever passed (it converts for us).
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not decode image (corrupt/unsupported): {path}")
    return preprocess_image(raw, image_size=image_size)


def preprocess_image(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize + normalize a raw image array to (image_size, image_size, 1).

    Works on uint8 arrays (0-255) or float arrays (0-1). This is THE one
    preprocessing function used everywhere, so train/inference never drift.
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:  # uint8-style input -> scale to 0-1
        img = img / 255.0
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = np.clip(img, 0.0, 1.0)
    return img[..., np.newaxis]  # (H, W, 1)


def load_mask(mask_path: Path | str | None, image_size: int) -> Optional[np.ndarray]:
    """Load a MVTec ground-truth mask as a boolean (H, W) array.

    Nearest-neighbour resize keeps the mask strictly binary (bilinear would
    blur mask edges and corrupt the 0/1 labels).
    """
    if mask_path is None:
        return None
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not decode mask image: {path}")
    mask = cv2.resize(raw, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def _list_images(directory: Path) -> List[Path]:
    """Sorted *.png files in a directory; raises if none are found."""
    files = sorted(p for p in directory.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png images found in: {directory}")
    return files


def load_normal_training_data(
    config: AnomalyDetectionConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Path], List[Path]]:
    """Load ONLY normal training images and split into train/validation.

    Returns:
        x_train: (N_train, H, W, 1) float32 in [0, 1]
        x_val:   (N_val,   H, W, 1) float32 in [0, 1]
        train_paths, val_paths: corresponding file paths (for visualisation)

    Note: the validation images are normal images held OUT of training.
    They are reused later to compute the anomaly threshold — a clean,
    leak-free protocol.
    """
    validate_dataset(config)
    paths = _list_images(config.train_good_dir)
    print(f"Found {len(paths)} normal training images in {config.train_good_dir}")

    images = np.stack(
        [load_image(p, config.image_size) for p in tqdm(paths, desc="Loading train")]
    ).astype(np.float32)

    x_train, x_val, train_paths, val_paths = train_test_split(
        images,
        paths,
        test_size=config.validation_split,
        random_state=config.seed,
        shuffle=True,
    )
    print(f"Train/Validation split: {len(x_train)} train, {len(x_val)} validation")
    return x_train, x_val, list(train_paths), list(val_paths)


def load_test_dataset(
    config: AnomalyDetectionConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Optional[Path]]]:
    """Load the full test set with image-level labels and mask paths.

    Returns:
        images:      (N, H, W, 1) float32 in [0, 1]
        labels:      (N,) int — 0 = NORMAL (good), 1 = ANOMALY (defect)
        defect_type: per-image folder name ("good", "scratch_head", ...)
        mask_paths:  ground-truth mask path or None for normal images
    """
    validate_dataset(config)
    images: List[np.ndarray] = []
    labels: List[int] = []
    defect_types: List[str] = []
    mask_paths: List[Optional[Path]] = []

    test_dirs = sorted(d for d in config.test_dir.iterdir() if d.is_dir())
    if not test_dirs:
        raise FileNotFoundError(f"No test sub-directories found in {config.test_dir}")

    for defect_dir in test_dirs:
        defect_name = defect_dir.name
        for img_path in tqdm(
            sorted(defect_dir.glob("*.png")), desc=f"Loading test/{defect_name}"
        ):
            images.append(load_image(img_path, config.image_size))
            labels.append(0 if defect_name == "good" else 1)
            defect_types.append(defect_name)
            if defect_name == "good":
                mask_paths.append(None)
            else:
                # MVTec convention: 000.png -> ground_truth/<defect>/000_mask.png
                mask_paths.append(
                    config.ground_truth_dir / defect_name / f"{img_path.stem}_mask.png"
                )

    return (
        np.stack(images).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        defect_types,
        mask_paths,
    )


# ---------------------------------------------------------------------------
# CNN Autoencoder architecture
# ---------------------------------------------------------------------------


def build_autoencoder(config: AnomalyDetectionConfig) -> keras.Model:
    """Build the convolutional autoencoder.

    Architecture (128x128 input, 8x spatial downsample):

        Encoder                          Latent              Decoder
        Input (128,128,1)
        Conv2D(32, 3x3, ReLU, same)  \
        MaxPool 2x2                    64x64
        Conv2D(64, 3x3, ReLU, same)  \
        MaxPool 2x2                    32x32
        Conv2D(128, 3x3, ReLU, same) \
        MaxPool 2x2                    16x16x128  ->  Conv2DTranspose(128, s2)
                                                      ->  Conv2DTranspose(64, s2)
                                                      ->  Conv2DTranspose(32, s2)
                                                      ->  Conv2D(1, 3x3, sigmoid)

    The "latent representation" is the (16, 16, 128) feature map produced by
    the encoder's last convolution + pooling. The decoder mirrors the
    encoder with transposed convolutions so the output has the input shape.

    BatchNorm after each conv stabilises training on the small MVTec sets;
    the final sigmoid keeps pixels in 0-1, matching the normalised inputs so
    MSE compares like with like.
    """
    inp = keras.Input(shape=(config.image_size, config.image_size, config.channels))

    # ----- Encoder -----
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="enc_conv1")(inp)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPooling2D(2, name="enc_pool1")(x)          # 128 -> 64

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.MaxPooling2D(2, name="enc_pool2")(x)          # 64 -> 32

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.MaxPooling2D(2, name="enc_pool3")(x)          # 32 -> 16

    # ----- Latent representation: (16, 16, 128) feature map -----
    latent = layers.Conv2D(
        128, 3, padding="same", activation="relu", name="latent_conv"
    )(x)

    # ----- Decoder -----
    x = layers.Conv2DTranspose(
        128, 3, strides=2, padding="same", activation="relu", name="dec_up1"
    )(latent)                                                # 16 -> 32
    x = layers.BatchNormalization(name="dec_bn1")(x)

    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding="same", activation="relu", name="dec_up2"
    )(x)                                                     # 32 -> 64
    x = layers.BatchNormalization(name="dec_bn2")(x)

    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding="same", activation="relu", name="dec_up3"
    )(x)                                                     # 64 -> 128
    x = layers.BatchNormalization(name="dec_bn3")(x)

    out = layers.Conv2D(
        config.channels, 3, padding="same", activation="sigmoid", name="output"
    )(x)

    model = keras.Model(inp, out, name=f"{config.category}_conv_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def describe_architecture(model: keras.Model) -> None:
    """Print a human-readable Encoder / Latent / Decoder map of the model."""
    print("=" * 62)
    print("CONVOLUTIONAL AUTOENCODER — ARCHITECTURE OVERVIEW")
    print("=" * 62)
    print("ENCODER   : enc_conv1 -> enc_pool1 -> enc_conv2 -> enc_pool2")
    print("            -> enc_conv3 -> enc_pool3")
    print("LATENT    : latent_conv  (16 x 16 x 128 feature map)")
    print("DECODER   : dec_up1 -> dec_up2 -> dec_up3 -> output (sigmoid)")
    print("-" * 62)
    model.summary()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_autoencoder(
    model: keras.Model,
    x_train: np.ndarray,
    x_val: np.ndarray,
    config: AnomalyDetectionConfig,
) -> keras.callbacks.History:
    """Train the autoencoder to reconstruct its own input (x -> x).

    Callbacks:
        * EarlyStopping   — stop when val_loss stops improving.
        * ModelCheckpoint — always keep the best (lowest val_loss) weights
                            saved to models/<category>_autoencoder.keras.
        * ReduceLROnPlateau — halve the LR when val_loss plateaus.
    """
    config.ensure_output_dirs()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        x_train,  # autoencoder: input == target
        validation_data=(x_val, x_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    return history


def plot_training_history(
    history: keras.callbacks.History,
    save_path: Path | str,
    show: bool = False,
) -> Path:
    """Plot training vs validation MSE loss per epoch and save the figure."""
    hist = history.history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["loss"], marker="o", label="Training loss (MSE)")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], marker="s", label="Validation loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training vs Validation Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved training history plot: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Reconstruction, error and anomaly score
# ---------------------------------------------------------------------------


def reconstruct_image(model: keras.Model, image: np.ndarray) -> np.ndarray:
    """Reconstruct one image with the trained autoencoder.

    Accepts a (H, W) / (H, W, 1) array or an already batched (1, H, W, 1)
    array. Returns an array of shape (H, W, 1) in [0, 1].
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    recon = model.predict(arr, verbose=0)
    return recon[0]


def calculate_reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> np.ndarray:
    """Pixel-wise squared-error map, averaged over channels -> (H, W).

        error_map = mean_c( (original - reconstructed)^2 )

    High values mark pixels the network could not reconstruct well, which
    is the core signal for localising anomalies.
    """
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    return np.mean(np.square(diff), axis=-1)


def calculate_anomaly_score(error_map: np.ndarray) -> float:
    """Image-level anomaly score = mean of the pixel error map."""
    return float(np.mean(error_map))


def score_images(
    model: keras.Model, images: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Score a stack of images in batches.

    Returns:
        scores:     (N,) mean reconstruction error per image
        error_maps: (N, H, W) per-pixel error maps
    """
    scores = np.zeros(len(images), dtype=np.float64)
    error_maps = np.zeros(images.shape[:3], dtype=np.float32)
    iterator = range(0, len(images), batch_size)
    for start in tqdm(iterator, desc="Scoring images", total=len(iterator)):
        batch = images[start : start + batch_size]
        recons = model.predict(batch, verbose=0)
        err = np.mean(np.square(batch - recons), axis=-1)  # (B, H, W)
        error_maps[start : start + len(batch)] = err
        scores[start : start + len(batch)] = err.mean(axis=(1, 2))
    return scores, error_maps


# ---------------------------------------------------------------------------
# Threshold (from NORMAL VALIDATION data only)
# ---------------------------------------------------------------------------


def calculate_threshold(
    normal_scores: np.ndarray,
    n_std: float = 3.0,
    strategy: str = "mean_std",
    percentile: float = 99.0,
) -> Dict[str, float]:
    """Statistically motivated image-level threshold from NORMAL scores.

    Strategies (all use normal validation data only — never test labels):
        "mean_std"   : mean + n_std * std. The spec's baseline; assumes a
                       roughly Gaussian error distribution. Conservative
                       when the distribution has a heavy right tail.
        "percentile" : given percentile of the normal validation scores
                       (e.g. 99.0 -> flags ~1% of normal images).
        "max"        : the largest normal validation score. Guarantees
                       (on validation data) that no normal image is flagged.

    This function is the single place to change the threshold policy.
    """
    scores = np.asarray(normal_scores, dtype=np.float64)
    mean_err = float(scores.mean())
    std_err = float(scores.std())

    if strategy == "mean_std":
        threshold = mean_err + n_std * std_err
    elif strategy == "percentile":
        threshold = float(np.percentile(scores, percentile))
    elif strategy == "max":
        threshold = float(scores.max())
    else:
        raise ValueError(
            f"Unknown threshold strategy '{strategy}'. "
            "Expected 'mean_std', 'percentile' or 'max'."
        )

    return {
        "threshold": threshold,
        "strategy": strategy,
        "mean_normal_error": mean_err,
        "std_normal_error": std_err,
        "n_std": n_std,
        "percentile": percentile,
    }


def calculate_pixel_threshold(
    normal_error_maps: Sequence[np.ndarray],
    n_std: float = 3.0,
    strategy: str = "mean_std",
    percentile: float = 99.0,
) -> float:
    """Pixel-level threshold used to binarise error maps into masks.

    Computed from normal validation error maps ONLY (same strategies as
    `calculate_threshold`):

        mean_std   : mean(all normal pixel errors) + n_std * std
        percentile : given percentile of all normal pixel errors
        max        : maximum normal pixel error

    Error map -> predicted anomaly mask: every pixel whose reconstruction
    error exceeds this value is flagged as anomalous (optionally cleaned
    with a 3x3 median filter to remove isolated speckles).
    """
    stacked = np.stack([m.ravel() for m in normal_error_maps]).ravel()
    if strategy == "mean_std":
        return float(stacked.mean() + n_std * stacked.std())
    if strategy == "percentile":
        return float(np.percentile(stacked, percentile))
    if strategy == "max":
        return float(stacked.max())
    raise ValueError(
        f"Unknown pixel threshold strategy '{strategy}'. "
        "Expected 'mean_std', 'percentile' or 'max'."
    )


def create_predicted_mask(
    error_map: np.ndarray, pixel_threshold: float, median_filter: bool = True
) -> np.ndarray:
    """error_map -> binary predicted anomaly mask (bool (H, W)).

    1. Flag pixels with reconstruction error above `pixel_threshold`.
    2. Apply a 3x3 median filter to suppress isolated false-positive pixels.
    """
    mask = (error_map > pixel_threshold).astype(np.uint8)
    if median_filter:
        mask = cv2.medianBlur(mask, 3)
    return mask.astype(bool)


def save_threshold(threshold_info: Dict, path: Path | str) -> Path:
    """Persist threshold metadata as JSON (consumed later by FastAPI)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(threshold_info, fh, indent=4)
    print(f"Saved threshold info: {path}")
    return path


def load_threshold(path: Path | str) -> Dict:
    """Load threshold metadata saved by `save_threshold`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {path}. Train the model first (train.py)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Save / load model
# ---------------------------------------------------------------------------


def save_model(model: keras.Model, path: Path | str) -> Path:
    """Save the full model (.keras format: architecture + weights + compile)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"Saved model: {path}")
    return path


def load_trained_model(path: Path | str) -> keras.Model:
    """Load a saved .keras model without retraining."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Train the model first (train.py)."
        )
    model = keras.models.load_model(path)
    print(f"Loaded model: {path}")
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_image(
    model: keras.Model,
    image_path: Path | str,
    threshold: float,
    image_size: int,
    pixel_threshold: Optional[float] = None,
) -> Dict:
    """Full inference for one image.

    Steps: load -> preprocess -> reconstruct -> error map -> anomaly score
    -> compare with threshold -> NORMAL / ANOMALY.

    Returns a dict with (at least) the keys required by the project spec:
        prediction, anomaly_score, threshold, reconstruction, error_map
    plus `original`, `image_path` and (optionally) `predicted_mask` for
    heatmap visualisation and FastAPI responses.
    """
    original = load_image(image_path, image_size)
    reconstruction = reconstruct_image(model, original)
    error_map = calculate_reconstruction_error(original, reconstruction)
    anomaly_score = calculate_anomaly_score(error_map)
    prediction = "ANOMALY" if anomaly_score > threshold else "NORMAL"

    result: Dict = {
        "image_path": str(image_path),
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "threshold": float(threshold),
        "original": original,
        "reconstruction": reconstruction,
        "error_map": error_map,
    }
    if pixel_threshold is not None:
        result["predicted_mask"] = create_predicted_mask(error_map, pixel_threshold)
    return result


# ---------------------------------------------------------------------------
# Visualisation: heatmaps & result grids
# ---------------------------------------------------------------------------


def _error_heatmap_overlay(original: np.ndarray, error_map: np.ndarray) -> np.ndarray:
    """Blend a JET-coloured error map over the (grayscale) original image."""
    err = error_map / max(float(error_map.max()), 1e-12)  # normalise 0-1
    heat = cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    base = original.squeeze()
    base = (base * 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) if base.ndim == 2 else base

    # 60% original + 40% heatmap keeps the object recognisable.
    overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
    return overlay


def generate_anomaly_heatmap(
    result: Dict,
    save_path: Optional[Path | str] = None,
    show: bool = False,
    title_prefix: str = "",
) -> plt.Figure:
    """2x2 result grid: original | reconstruction // error map | heatmap overlay.

    The heatmap makes high-error regions visually obvious: hot (red)
    colours mark pixels the autoencoder reconstructed poorly.
    """
    original = result["original"]
    reconstruction = result["reconstruction"]
    error_map = result["error_map"]
    overlay = _error_heatmap_overlay(original, error_map)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(original.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(reconstruction.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Reconstructed Image")
    im = axes[1, 0].imshow(error_map, cmap="hot")
    axes[1, 0].set_title("Error Map (squared error)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Heatmap Overlay")

    for ax in axes.ravel():
        ax.axis("off")

    caption = (
        f"{title_prefix}Prediction: {result['prediction']}   "
        f"Anomaly Score: {result['anomaly_score']:.4f}   "
        f"Threshold: {result['threshold']:.4f}"
    )
    fig.suptitle(caption, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved heatmap figure: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def visualize_localization_comparison(
    result: Dict,
    gt_mask: np.ndarray,
    save_path: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Figure:
    """Compare predicted anomaly mask vs the MVTec ground-truth mask.

    Panels: original | GT mask // predicted mask | error-map overlay.
    Used to sanity-check the localization metrics in evaluate.py.
    """
    overlay = _error_heatmap_overlay(result["original"], result["error_map"])
    pred_mask = result.get("predicted_mask", result["error_map"] > 0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(result["original"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original Image (defect)")
    axes[0, 1].imshow(gt_mask, cmap="gray")
    axes[0, 1].set_title("Ground-Truth Mask (MVTec)")
    axes[1, 0].imshow(pred_mask, cmap="gray")
    axes[1, 0].set_title("Predicted Anomaly Mask")
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Error Heatmap Overlay")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Localization: predicted mask vs ground truth", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved localization comparison: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_image_level(
    model: keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    batch_size: int = 32,
) -> Dict:
    """Image-level NORMAL vs ANOMALY metrics on the test set.

    Uses the CONTINUOUS anomaly scores for ROC-AUC (proper practice) and
    the thresholded scores for accuracy / precision / recall / F1.
    """
    scores, error_maps = score_images(model, test_images, batch_size)
    y_pred = (scores > threshold).astype(np.int64)
    y_true = np.asarray(test_labels, dtype=np.int64)

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    if len(np.unique(y_true)) == 2:  # ROC-AUC needs both classes present
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
        metrics["fpr"], metrics["tpr"] = fpr, tpr
    else:
        metrics["roc_auc"] = float("nan")
        metrics["fpr"] = metrics["tpr"] = None

    metrics.update(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "scores": scores,
            "error_maps": error_maps,
        }
    )
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: Path | str) -> Path:
    """Save a labelled confusion-matrix figure (rows = true, cols = predicted)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["NORMAL (good)", "ANOMALY (defect)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, color="black",
            )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (image level)")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {save_path}")
    return save_path


def plot_roc_curve(
    fpr: Optional[np.ndarray],
    tpr: Optional[np.ndarray],
    roc_auc: float,
    save_path: Path | str,
) -> Path:
    """Save the ROC curve computed from continuous anomaly scores."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if fpr is not None and tpr is not None:
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (image level)")
    ax.grid(alpha=0.3)
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve: {save_path}")
    return save_path


def evaluate_localization(
    model: keras.Model,
    test_images: np.ndarray,
    mask_paths: Sequence[Optional[Path]],
    pixel_threshold: float,
    batch_size: int = 32,
    image_size: int = 128,
) -> Dict[str, float]:
    """Pixel-level localization metrics against MVTec ground-truth masks.

    Predicted mask generation (exactly what is evaluated):
        1. Reconstruct every defective test image.
        2. error_map = mean_c((original - reconstruction)^2).
        3. predicted_mask = error_map > pixel_threshold   (from normal val data)
        4. 3x3 median filter to remove isolated speckles.

    Aggregated over all defective test images (micro-averaged):
        IoU = TP / (TP + FP + FN), plus pixel precision / recall / F1.
    """
    tp = fp = fn = tn = 0
    n_defect = 0
    for start in range(0, len(test_images), batch_size):
        batch = test_images[start : start + batch_size]
        recons = model.predict(batch, verbose=0)
        err_maps = np.mean(np.square(batch - recons), axis=-1)
        for idx, err_map in enumerate(err_maps):
            global_idx = start + idx
            if mask_paths[global_idx] is None:
                continue
            gt_mask = load_mask(mask_paths[global_idx], image_size)
            pred_mask = create_predicted_mask(err_map, pixel_threshold)

            tp += int(np.sum(pred_mask & gt_mask))
            fp += int(np.sum(pred_mask & ~gt_mask))
            fn += int(np.sum(~pred_mask & gt_mask))
            tn += int(np.sum(~pred_mask & ~gt_mask))
            n_defect += 1

    iou_denom = tp + fp + fn
    precision_denom = tp + fp
    recall_denom = tp + fn
    return {
        "n_defect_images": n_defect,
        "iou": tp / iou_denom if iou_denom else 0.0,
        "pixel_precision": tp / precision_denom if precision_denom else 0.0,
        "pixel_recall": tp / recall_denom if recall_denom else 0.0,
        "pixel_f1": (
            2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        ),
    }


def print_metrics_report(
    image_metrics: Dict, localization_metrics: Dict[str, float], threshold: float
) -> None:
    """Pretty console report used by evaluate.py / the notebook."""
    cm = image_metrics["confusion_matrix"]
    print("=" * 62)
    print("IMAGE-LEVEL EVALUATION (test set)")
    print("=" * 62)
    print(f"Threshold used            : {threshold:.6f}")
    print(f"Accuracy                  : {image_metrics['accuracy']:.4f}")
    print(f"Precision                 : {image_metrics['precision']:.4f}")
    print(f"Recall                    : {image_metrics['recall']:.4f}")
    print(f"F1-score                  : {image_metrics['f1']:.4f}")
    print(f"ROC-AUC (continuous score): {image_metrics['roc_auc']:.4f}")
    print("Confusion matrix [true=rows, pred=cols] [NORMAL, ANOMALY]:")
    print(cm)
    print("-" * 62)
    print("LOCALIZATION EVALUATION (defective images, vs GT masks)")
    print("=" * 62)
    print(f"Defective images evaluated: {localization_metrics['n_defect_images']}")
    print(f"IoU                       : {localization_metrics['iou']:.4f}")
    print(f"Pixel precision           : {localization_metrics['pixel_precision']:.4f}")
    print(f"Pixel recall              : {localization_metrics['pixel_recall']:.4f}")
    print(f"Pixel F1                  : {localization_metrics['pixel_f1']:.4f}")
    print("=" * 62)


__all__ = [
    "AnomalyDetectionConfig",
    "score_images",
    "set_seeds",
    "check_gpu",
    "validate_dataset",
    "load_image",
    "preprocess_image",
    "load_mask",
    "load_normal_training_data",
    "load_test_dataset",
    "build_autoencoder",
    "describe_architecture",
    "train_autoencoder",
    "plot_training_history",
    "reconstruct_image",
    "calculate_reconstruction_error",
    "calculate_anomaly_score",
    "calculate_threshold",
    "calculate_pixel_threshold",
    "create_predicted_mask",
    "save_threshold",
    "load_threshold",
    "save_model",
    "load_trained_model",
    "predict_image",
    "generate_anomaly_heatmap",
    "visualize_localization_comparison",
    "evaluate_image_level",
    "evaluate_localization",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "print_metrics_report",
]
