
import argparse

from anomaly_detection import (
    build_autoencoder,
    category_paths,
    check_gpu,
    compute_pixel_threshold,
    compute_threshold,
    describe_architecture,
    load_normal_images,
    plot_training_history,
    save_model,
    save_threshold,
    score_images,
    set_seeds,
    train_autoencoder,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the autoencoder")
    parser.add_argument("--category", default="screw",
                        help="MVTec category folder name (screw, wood, bottle, ...)")
    return parser.parse_args()


def main():
    args = parse_args()
    category = args.category
    paths = category_paths(category)

    set_seeds()
    check_gpu()

    # 1. Only normal images: x_train and x_val both come from train/good.
    x_train, x_val = load_normal_images(category)

    # 2. Model.
    model = build_autoencoder()
    describe_architecture(model)

    # 3. Training target is the input image itself.
    history = train_autoencoder(model, x_train, x_val)

    # 4. Save model and thresholds for this category.
    save_model(model, paths["model"])

    val_scores, val_error_maps = score_images(model, x_val)

    threshold_info = compute_threshold(val_scores)
    threshold_info["category"] = category
    threshold_info["pixel_threshold"] = compute_pixel_threshold(val_error_maps)
    threshold_info["n_validation_images"] = int(len(x_val))

    save_threshold(threshold_info, paths["threshold"])

    # 5. Training curves.
    paths["results"].mkdir(parents=True, exist_ok=True)
    plot_training_history(history, paths["results"] / "training_history.png")

    print()
    print(f"Training complete [{category}]")
    print(f"  Model              : {paths['model']}")
    print(f"  Threshold          : {threshold_info['threshold']:.6f}")
    print(f"  Pixel threshold    : {threshold_info['pixel_threshold']:.6f}")
    print(f"  Next               : python evaluate.py --category {category}")


if __name__ == "__main__":
    main()
