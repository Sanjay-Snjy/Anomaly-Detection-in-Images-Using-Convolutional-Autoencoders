"""
Anomaly Detection in Images Using Convolutional Autoencoders
=============================================================
MVTec AD dataset. The default category is "screw"; the scripts also accept
`--category wood` and `--category bottle`.

The autoencoder is trained only on normal (good) images of one category. At test time
the image is reconstructed and the reconstruction error is compared with a
threshold learnt from normal images:

    error_map       = mean over channels of (original - reconstructed)^2
    anomaly_score   = mean of the error map
    prediction      = ANOMALY if anomaly_score > threshold else NORMAL

The threshold is computed as  mean + 3 * std  of the errors on normal
validation images (the same formula taught in class). No test image and no
defect label is used while training or while fixing the threshold.
"""

import json
import random
from pathlib import Path

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
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Settings (change these here; every script imports them)
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("mvtec_anomaly_detection")
CATEGORY = "screw"  # default; train/evaluate/predict override it with --category

IMAGE_SIZE = 128
CHANNELS = 1

BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
EARLY_STOP_PATIENCE = 10

# BatchNormalization keeps running estimates of mean/variance that are only
# used at prediction time. On the first few epochs those estimates are still
# near their defaults, so the validation loss looks terrible even while the
# training loss falls, and EarlyStopping would freeze an unusable model
# (this happened on the bottle category). Ignore validation loss until the
# estimates have had a few epochs to settle.
EARLY_STOP_START = 5

# How fast BatchNormalization forgets its initial mean=0 / variance=1 guess.
# The Keras default (0.99) needs ~500 batches to converge; with 200 training
# images that is 30 epochs, so predictions would use badly wrong statistics
# and the reconstruction collapses to a constant image. 0.9 converges in
# about 30 batches (a few epochs).
BATCHNORM_MOMENTUM = 0.9

# Light augmentation of the NORMAL training images only (flips, brightness,
# contrast, a little noise). Validation and test images are never augmented.
# This stops the autoencoder from memorising the training pictures; it has to
# learn the general structure of a normal part instead.
AUGMENT = True
AUGMENT_BRIGHTNESS = 0.08
AUGMENT_NOISE_STD = 0.01

# Width of the latent (bottleneck) feature map. At 128x128 the encoder ends
# at 16x16, so 16 filters give 16*16*16 = 4096 latent values for 16384 input
# pixels. The network is forced to compress, which is what makes it fail on
# defects instead of copying them. A wider bottleneck (e.g. 128 filters)
# gives more latent values than the image has pixels, and the model simply
# memorises the input.
LATENT_FILTERS = 16

SEED = 42

MODELS_DIR = Path("models")
RESULTS_ROOT = Path("results")


# Every category gets its own model files and its own results folder, so
# training wood does not overwrite the screw model or its figures.

def category_paths(category=CATEGORY):
    """All input and output paths that belong to one MVTec category."""
    results = RESULTS_ROOT / category
    return {
        "model": MODELS_DIR / f"{category}_autoencoder.keras",
        "threshold": MODELS_DIR / f"{category}_threshold.json",
        "calibrated_threshold":
            MODELS_DIR / f"{category}_threshold_calibrated.json",
        "results": results,
        "heatmaps": results / "heatmaps",
        "train_good": DATASET_ROOT / category / "train" / "good",
        "test": DATASET_ROOT / category / "test",
        "ground_truth": DATASET_ROOT / category / "ground_truth",
    }


# Defaults = the screw category, used by the web app and by predict.py.
PATHS = category_paths()

MODEL_PATH = PATHS["model"]
THRESHOLD_PATH = PATHS["threshold"]
CALIBRATED_THRESHOLD_PATH = PATHS["calibrated_threshold"]

RESULTS_DIR = PATHS["results"]
HEATMAPS_DIR = PATHS["heatmaps"]

TRAIN_GOOD_DIR = PATHS["train_good"]
TEST_DIR = PATHS["test"]
GROUND_TRUTH_DIR = PATHS["ground_truth"]

# mean_std -> mean + n_std * std, percentile -> given percentile, max -> largest
THRESHOLD_STRATEGY = "mean_std"
THRESHOLD_N_STD = 3.0
THRESHOLD_PERCENTILE = 99.0
MASK_N_STD = 3.0
MASK_PERCENTILE = 99.0


# ---------------------------------------------------------------------------
# Reproducibility and dataset checks
# ---------------------------------------------------------------------------


def set_seeds(seed=SEED):
    """Fix the random seeds so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def check_gpu():
    """Print the GPUs TensorFlow can see (useful in Colab)."""
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow version:", tf.__version__)
    if gpus:
        for gpu in gpus:
            print("GPU:", gpu)
    else:
        print("No GPU found, training will run on CPU.")
    return gpus


def check_dataset(category=CATEGORY):
    """Stop early with a clear message if the MVTec folder is missing."""
    paths = category_paths(category)
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT.resolve()}"
        )
    if not paths["train_good"].exists():
        raise FileNotFoundError(
            f"[{category}] Missing normal training images: {paths['train_good']}"
        )
    if not paths["test"].exists():
        raise FileNotFoundError(
            f"[{category}] Missing test folder: {paths['test']}"
        )



# Image loading and preprocessing


def preprocess_image(image):
 
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img = np.clip(img, 0.0, 1.0)
    return img[..., np.newaxis]  # (IMAGE_SIZE, IMAGE_SIZE, 1)


def load_image(image_path):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return preprocess_image(image)


def load_mask(mask_path):

    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def _image_files(folder):

    files = sorted(Path(folder).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png images in {folder}")
    return files


def load_normal_images(category=CATEGORY):

    check_dataset(category)

    paths = _image_files(category_paths(category)["train_good"])
    print(f"[{category}] Normal training images: {len(paths)}")

    images = np.stack([load_image(p) for p in paths]).astype(np.float32)

    x_train, x_val = train_test_split(
        images, test_size=VALIDATION_SPLIT, random_state=SEED, shuffle=True
    )
    print(f"Split: {len(x_train)} train, {len(x_val)} validation")
    return x_train, x_val


def load_test_images(category=CATEGORY):

    check_dataset(category)
    paths = category_paths(category)

    images = []
    labels = []
    defect_types = []
    mask_paths = []

    for defect_dir in sorted(d for d in paths["test"].iterdir() if d.is_dir()):
        name = defect_dir.name
        files = sorted(defect_dir.glob("*.png"))
        print(f"test/{name}: {len(files)} images")

        for path in files:
            images.append(load_image(path))
            labels.append(0 if name == "good" else 1)
            defect_types.append(name)

            if name == "good":
                mask_paths.append(None)
            else:
                mask_paths.append(
                    paths["ground_truth"] / name / f"{path.stem}_mask.png"
                )

    return (
        np.stack(images).astype(np.float32),
        np.array(labels),
        defect_types,
        mask_paths,
    )

# Model

def build_autoencoder():
    
    inp = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    # Encoder: 128 -> 64 -> 32 -> 16
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization(momentum=BATCHNORM_MOMENTUM)(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization(momentum=BATCHNORM_MOMENTUM)(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization(momentum=BATCHNORM_MOMENTUM)(x)
    x = layers.MaxPool2D()(x)

    # bottleneck
    x = layers.Conv2D(LATENT_FILTERS, 3, activation="relu", padding="same")(x)

    # Decoder: 16 -> 32 -> 64 -> 128
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=BATCHNORM_MOMENTUM)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=BATCHNORM_MOMENTUM)(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=BATCHNORM_MOMENTUM)(x)

    out = layers.Conv2D(CHANNELS, 3, activation="sigmoid", padding="same")(x)

    model = models.Model(inp, out, name="screw_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def describe_architecture(model):
    """Print the layer table and the size of the bottleneck."""
    latent_size = IMAGE_SIZE // 8
    values = latent_size * latent_size * LATENT_FILTERS
    print("=" * 60)
    print("CONVOLUTIONAL AUTOENCODER")
    print(f"Latent shape: {latent_size} x {latent_size} x {LATENT_FILTERS} "
          f"= {values} values")
    print("=" * 60)
    model.summary()


# Training

def _augment_dataset(images):

    dataset = tf.data.Dataset.from_tensor_slices(images.astype(np.float32))
    dataset = dataset.shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)

    def augment(batch):
        x = tf.image.random_flip_left_right(batch)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_brightness(x, max_delta=AUGMENT_BRIGHTNESS)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        x = x + tf.random.normal(tf.shape(x), stddev=AUGMENT_NOISE_STD)
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x, x

    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def train_autoencoder(model, x_train, x_val):
    """Train the autoencoder to reproduce its input (x -> x)."""
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        start_from_epoch=EARLY_STOP_START,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    callbacks = [early_stop, reduce_lr]

    if AUGMENT:
        print("Augmentation ON (normal training images only)")
        history = model.fit(
            _augment_dataset(x_train),
            validation_data=(x_val, x_val),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2,
        )
    else:
        history = model.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=True,
            verbose=2,
        )
    return history


def plot_training_history(history, save_path):
    """Save the training and validation loss curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], marker="o", label="Training loss")
    plt.plot(history.history["val_loss"], marker="s", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training vs Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved training curve:", save_path)


# ---------------------------------------------------------------------------
# Reconstruction error and anomaly score
# ---------------------------------------------------------------------------


def reconstruct(model, image):
    """Reconstruct one (H, W, 1) image."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if image.ndim == 3:
        image = image[np.newaxis, ...]

    reconstruction = model.predict(image, verbose=0)
    return reconstruction[0]


def reconstruction_error(original, reconstructed):
    """Pixel-wise squared error, averaged over channels -> (H, W) map."""
    difference = original.astype(np.float32) - reconstructed.astype(np.float32)
    return np.mean(np.square(difference), axis=-1)


def anomaly_score(error_map):
    """One number per image: the mean of its error map."""
    return float(np.mean(error_map))


def score_images(model, images, batch_size=BATCH_SIZE):
    """Score many images in batches.

    Returns scores (N,) and error_maps (N, H, W).
    """
    scores = np.zeros(len(images), dtype=np.float64)
    error_maps = np.zeros(images.shape[:3], dtype=np.float32)

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        reconstructions = model.predict(batch, verbose=0)
        errors = np.mean(np.square(batch - reconstructions), axis=-1)

        error_maps[start:start + len(batch)] = errors
        scores[start:start + len(batch)] = errors.mean(axis=(1, 2))

    return scores, error_maps


# ---------------------------------------------------------------------------
# Thresholds (computed from normal images only)
# ---------------------------------------------------------------------------


def compute_threshold(normal_scores, strategy=THRESHOLD_STRATEGY,
                      n_std=THRESHOLD_N_STD, percentile=THRESHOLD_PERCENTILE):
    """Image-level threshold from the scores of normal images.

    mean_std   : mean + n_std * std        (the formula used in class)
    percentile : e.g. the 99th percentile
    max        : the largest normal score
    """
    scores = np.asarray(normal_scores, dtype=np.float64)
    mean_error = float(scores.mean())
    std_error = float(scores.std())

    if strategy == "mean_std":
        threshold = mean_error + n_std * std_error
    elif strategy == "percentile":
        threshold = float(np.percentile(scores, percentile))
    elif strategy == "max":
        threshold = float(scores.max())
    else:
        raise ValueError("strategy must be 'mean_std', 'percentile' or 'max'")

    return {
        "threshold": threshold,
        "strategy": strategy,
        "mean_normal_error": mean_error,
        "std_normal_error": std_error,
        "n_std": n_std,
        "percentile": percentile,
    }


def compute_pixel_threshold(normal_error_maps, strategy=THRESHOLD_STRATEGY,
                            n_std=MASK_N_STD, percentile=MASK_PERCENTILE):
    """Pixel-level threshold from the error maps of normal images."""
    pixels = np.concatenate([m.ravel() for m in normal_error_maps])

    if strategy == "mean_std":
        return float(pixels.mean() + n_std * pixels.std())
    if strategy == "percentile":
        return float(np.percentile(pixels, percentile))
    if strategy == "max":
        return float(pixels.max())
    raise ValueError("strategy must be 'mean_std', 'percentile' or 'max'")


def calibrate_threshold(model, normal_images, category=CATEGORY):
    """Fit the threshold on normal images from the deployment batch.

    train/good and test/good come from different captures, so a threshold
    fitted on train/good can be too high for test/good (it then flags
    nothing). Here the reference images come from the same batch that will
    be scored, which is what a real line-side calibration would do.
    """
    scores, error_maps = score_images(model, normal_images)

    info = compute_threshold(scores)
    info["category"] = category
    info["calibration"] = "in_domain_normal_images"
    info["pixel_threshold"] = compute_pixel_threshold(error_maps)
    info["n_calibration_images"] = int(len(normal_images))
    return info


def make_mask(error_map, pixel_threshold):
    """Turn an error map into a binary mask and remove single speckles."""
    mask = (error_map > pixel_threshold).astype(np.uint8)
    mask = cv2.medianBlur(mask, 3)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------


def save_model(model, path=MODEL_PATH):
    """Save architecture, weights and compile settings in one .keras file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print("Saved model:", path)
    return path


def load_model(path=MODEL_PATH):
    """Load a saved model without retraining."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    print("Loaded model:", path)
    return models.load_model(path)


def save_threshold(info, path=THRESHOLD_PATH):
    """Save the threshold dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    print("Saved threshold:", path)
    return path


def load_threshold(path=THRESHOLD_PATH):
    """Load a threshold JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Threshold not found: {path}. Run train.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(model, image_path, threshold, pixel_threshold=None):
    """Full prediction for one image.

    Returns a dictionary with the prediction, the score, the reconstruction
    and the error map (plus a mask when a pixel threshold is given).
    """
    original = load_image(image_path)
    reconstructed = reconstruct(model, original)
    error_map = reconstruction_error(original, reconstructed)
    score = anomaly_score(error_map)

    result = {
        "image_path": str(image_path),
        "prediction": "ANOMALY" if score > threshold else "NORMAL",
        "anomaly_score": score,
        "threshold": float(threshold),
        "original": original,
        "reconstruction": reconstructed,
        "error_map": error_map,
    }

    if pixel_threshold is not None:
        result["predicted_mask"] = make_mask(error_map, pixel_threshold)

    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def heatmap_overlay(original, error_map):
    """Blend a JET colour map of the error over the grayscale image."""
    error = error_map / max(float(error_map.max()), 1e-12)
    heat = cv2.applyColorMap((error * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    base = (original.squeeze() * 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    return cv2.addWeighted(base, 0.6, heat, 0.4, 0)


def show_heatmap(result, save_path=None, show=False, title_prefix=""):
    """2x2 figure: original | reconstruction // error map | overlay."""
    overlay = heatmap_overlay(result["original"], result["error_map"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(result["original"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(result["reconstruction"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Reconstruction")
    image = axes[1, 0].imshow(result["error_map"], cmap="hot")
    axes[1, 0].set_title("Error map")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Heatmap overlay")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"{title_prefix}{result['prediction']}   "
        f"score {result['anomaly_score']:.4f}   "
        f"threshold {result['threshold']:.4f}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print("Saved heatmap:", save_path)
    if show:
        plt.show()
    plt.close(fig)


def show_localization(result, gt_mask, save_path=None):
    """2x2 figure: original | ground truth // predicted mask | overlay."""
    overlay = heatmap_overlay(result["original"], result["error_map"])
    predicted_mask = result.get("predicted_mask")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(result["original"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(gt_mask, cmap="gray")
    axes[0, 1].set_title("Ground-truth mask")
    axes[1, 0].imshow(predicted_mask, cmap="gray")
    axes[1, 0].set_title("Predicted mask")
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Heatmap overlay")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("Localization: predicted mask vs ground truth")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print("Saved localization figure:", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_image_level(model, images, labels, threshold, batch_size=BATCH_SIZE):
    """Accuracy, precision, recall, F1 and ROC-AUC on the test set.

    ROC-AUC uses the continuous scores; the other metrics use the
    thresholded predictions.
    """
    scores, error_maps = score_images(model, images, batch_size)
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = (scores > threshold).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": scores,
        "error_maps": error_maps,
    }

    fpr, tpr, _ = roc_curve(y_true, scores)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    return metrics


def evaluate_localization(model, images, mask_paths, pixel_threshold,
                          batch_size=BATCH_SIZE):
    """Pixel-level IoU / precision / recall / F1 against the MVTec masks."""
    tp = fp = fn = 0
    defect_count = 0

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        reconstructions = model.predict(batch, verbose=0)
        errors = np.mean(np.square(batch - reconstructions), axis=-1)

        for i, error_map in enumerate(errors):
            index = start + i
            if mask_paths[index] is None:
                continue

            gt_mask = load_mask(mask_paths[index])
            predicted_mask = make_mask(error_map, pixel_threshold)

            tp += int(np.sum(predicted_mask & gt_mask))
            fp += int(np.sum(predicted_mask & ~gt_mask))
            fn += int(np.sum(~predicted_mask & gt_mask))
            defect_count += 1

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "n_defect_images": defect_count,
        "iou": iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
    }


def plot_confusion_matrix(cm, save_path):
    """Save the confusion matrix as a figure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    labels = ["NORMAL", "ANOMALY"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("Saved confusion matrix:", save_path)


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Save the ROC curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved ROC curve:", save_path)


def print_report(image_metrics, localization_metrics, threshold):
    """Print the evaluation results in a readable form."""
    print("=" * 60)
    print("IMAGE-LEVEL EVALUATION")
    print("=" * 60)
    print(f"Threshold        : {threshold:.6f}")
    print(f"Accuracy         : {image_metrics['accuracy']:.4f}")
    print(f"Precision        : {image_metrics['precision']:.4f}")
    print(f"Recall           : {image_metrics['recall']:.4f}")
    print(f"F1-score         : {image_metrics['f1']:.4f}")
    print(f"ROC-AUC          : {image_metrics['roc_auc']:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(image_metrics["confusion_matrix"])
    print("-" * 60)

    if localization_metrics:
        print("LOCALIZATION EVALUATION")
        print("=" * 60)
        print(f"Defective images : {localization_metrics['n_defect_images']}")
        print(f"IoU              : {localization_metrics['iou']:.4f}")
        print(f"Pixel precision  : {localization_metrics['pixel_precision']:.4f}")
        print(f"Pixel recall     : {localization_metrics['pixel_recall']:.4f}")
        print(f"Pixel F1         : {localization_metrics['pixel_f1']:.4f}")
        print("=" * 60)
