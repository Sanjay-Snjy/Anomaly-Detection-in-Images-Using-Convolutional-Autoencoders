"""
predict.py — predict NORMAL / ANOMALY for one image.

Run:
    python predict.py --image mvtec_anomaly_detection/screw/test/scratch_head/000.png
    python predict.py --image tree.png --category wood --show-heatmap
    python predict.py --image bottle.png --category bottle
    python predict.py --image some_screw.png --threshold-file models/screw_threshold_calibrated.json
"""

import argparse
from pathlib import Path

from anomaly_detection import (
    category_paths,
    load_model,
    load_threshold,
    predict,
    show_heatmap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one image")
    parser.add_argument("--image", required=True, help="Image to check")
    parser.add_argument("--category", default="screw",
                        help="Which model to use (screw, wood, bottle, ...)")
    parser.add_argument("--threshold-file", default=None,
                        help="Threshold JSON (default: models/<category>_threshold.json)")
    parser.add_argument("--show-heatmap", action="store_true",
                        help="Display the heatmap after prediction")
    parser.add_argument("--save-heatmap", default=None,
                        help="Optional path to save the heatmap PNG")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = category_paths(args.category)

    threshold_path = (
        Path(args.threshold_file) if args.threshold_file else paths["threshold"]
    )

    model = load_model(paths["model"])
    threshold_info = load_threshold(threshold_path)
    threshold = float(threshold_info["threshold"])

    pixel_threshold = threshold_info.get("pixel_threshold")
    if pixel_threshold is not None:
        pixel_threshold = float(pixel_threshold)

    result = predict(model, args.image, threshold, pixel_threshold)

    print("Prediction   :", result["prediction"])
    print("Anomaly Score:", f"{result['anomaly_score']:.4f}")
    print("Threshold    :", f"{result['threshold']:.4f}")
    print("Image        :", result["image_path"])

    if args.show_heatmap or args.save_heatmap:
        save_path = (
            Path(args.save_heatmap)
            if args.save_heatmap
            else paths["heatmaps"] / f"predict_{Path(args.image).stem}.png"
        )
        show_heatmap(result, save_path=save_path, show=args.show_heatmap)


if __name__ == "__main__":
    main()
