"""
train.py — train the convolutional autoencoder on normal screw images.

Flow:
    load training data (normal only)
        -> build model
        -> train model (EarlyStopping / ModelCheckpoint / ReduceLROnPlateau)
        -> save model
        -> compute image-level threshold from NORMAL VALIDATION errors
        -> save threshold JSON
        -> plot training history

Run:
    python train.py [--epochs 50] [--batch-size 32] [--dataset-root PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomaly_detection import (
    AnomalyDetectionConfig,
    build_autoencoder,
    calculate_anomaly_score,
    calculate_pixel_threshold,
    calculate_threshold,
    describe_architecture,
    load_normal_training_data,
    plot_training_history,
    save_model,
    save_threshold,
    score_images,
    set_seeds,
    train_autoencoder,
)


def parse_args() -> argparse.Namespace:
    """CLI overrides for the config defaults."""
    parser = argparse.ArgumentParser(description="Train MVTec screw autoencoder")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Path to mvtec_anomaly_detection/")
    parser.add_argument("--category", type=str, default=None,
                        help="MVTec category (default: screw)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Resize target, e.g. 128")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config: AnomalyDetectionConfig, args: argparse.Namespace) -> None:
    """Copy any non-None CLI argument onto the config object."""
    if args.dataset_root is not None:
        config.dataset_root = Path(args.dataset_root)
    if args.category is not None:
        config.category = args.category
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.validation_split is not None:
        config.validation_split = args.validation_split
    if args.seed is not None:
        config.seed = args.seed


def main() -> None:
    args = parse_args()
    config = AnomalyDetectionConfig()
    apply_overrides(config, args)

    print("Configuration:")
    print(json.dumps(config.to_dict(), indent=2, default=str))

    set_seeds(config.seed)
    config.ensure_output_dirs()

    # 1. Load ONLY normal training images + held-out normal validation split.
    x_train, x_val, _train_paths, _val_paths = load_normal_training_data(config)

    # 2. Build the CNN autoencoder and show its structure.
    model = build_autoencoder(config)
    describe_architecture(model)

    # 3. Train (input == target because this is an autoencoder).
    history = train_autoencoder(model, x_train, x_val, config)

    # 4. Save the best model (also checkpointed during training).
    save_model(model, config.model_path)

    # 5. Threshold from NORMAL VALIDATION errors only.
    val_scores, val_error_maps = score_images(model, x_val, config.batch_size)
    threshold_info = calculate_threshold(
        val_scores,
        n_std=config.threshold_n_std,
        strategy=config.threshold_strategy,
        percentile=config.threshold_percentile,
    )
    pixel_threshold = calculate_pixel_threshold(
        val_error_maps,
        n_std=config.mask_n_std,
        strategy=config.threshold_strategy,
        percentile=config.mask_percentile,
    )
    threshold_info.update(
        {
            "category": config.category,
            "pixel_threshold": pixel_threshold,
            "n_validation_images": int(len(x_val)),
        }
    )
    save_threshold(threshold_info, config.threshold_path)

    # 6. Training curves.
    plot_training_history(history, config.results_dir / "training_history.png")

    print("\nTraining complete.")
    print(f"  Image-level threshold : {threshold_info['threshold']:.6f}"
          f"  (mean {threshold_info['mean_normal_error']:.6f}"
          f" + {config.threshold_n_std:.0f}*std"
          f" {threshold_info['std_normal_error']:.6f})")
    print(f"  Pixel-level threshold : {pixel_threshold:.6f}")
    print(f"  Model                 : {config.model_path}")
    print(f"  Threshold JSON        : {config.threshold_path}")
    print("\nNext: python evaluate.py")


if __name__ == "__main__":
    main()
