"""
predict.py — predict NORMAL / ANOMALY for a single image using the trained
screw autoencoder + saved threshold (no retraining).

Run:
    python predict.py --image mvtec_anomaly_detection/screw/test/scratch_head/000.png
    python predict.py --image some_bottle.png --show-heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_detection import (
    AnomalyDetectionConfig,
    generate_anomaly_heatmap,
    load_threshold,
    load_trained_model,
    predict_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image to check")
    parser.add_argument("--category", type=str, default=None,
                        help="MVTec category (default: screw)")
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--show-heatmap", action="store_true",
                        help="Display the anomaly heatmap after prediction")
    parser.add_argument("--save-heatmap", type=str, default=None,
                        help="Optional path to save the heatmap PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnomalyDetectionConfig()
    if args.category is not None:
        config.category = args.category
    if args.models_dir is not None:
        config.models_dir = Path(args.models_dir)

    # Load the trained model and threshold — no retraining happens here.
    model = load_trained_model(config.model_path)
    threshold_info = load_threshold(config.threshold_path)
    threshold = float(threshold_info["threshold"])
    pixel_threshold = threshold_info.get("pixel_threshold")

    result = predict_image(
        model,
        args.image,
        threshold=threshold,
        image_size=config.image_size,
        pixel_threshold=float(pixel_threshold) if pixel_threshold is not None else None,
    )

    # Console output exactly as requested in the project spec.
    print("Prediction   :", result["prediction"])
    print("Anomaly Score:", f"{result['anomaly_score']:.4f}")
    print("Threshold    :", f"{result['threshold']:.4f}")
    print("Image        :", result["image_path"])

    if args.show_heatmap or args.save_heatmap:
        save_path = (
            Path(args.save_heatmap)
            if args.save_heatmap
            else config.heatmaps_dir / f"predict_{Path(args.image).stem}.png"
        )
        generate_anomaly_heatmap(result, save_path=save_path, show=args.show_heatmap)


if __name__ == "__main__":
    main()
