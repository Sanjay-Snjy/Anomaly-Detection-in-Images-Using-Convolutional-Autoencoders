"""
evaluate.py — evaluate a trained autoencoder on one MVTec test set.

Steps:
    load model + threshold for the category
    load test images (good + all defect types, with ground-truth masks)
    predict every image and compute image-level metrics
    compute pixel-level IoU / precision / recall / F1
    save confusion matrix, ROC curve and example heatmaps

Two threshold protocols:
    python evaluate.py
        threshold from train.py, fitted on the validation split of train/good
    python evaluate.py --domain-calibration 0.5
        threshold fitted on half of the normal TEST images (same capture batch);
        those images are then removed from the evaluation set

Run:
    python evaluate.py --category screw
    python evaluate.py --category wood --domain-calibration 0.5
    python evaluate.py --category bottle --domain-calibration 0.5
"""

import argparse
import json

import numpy as np

from anomaly_detection import (
    anomaly_score,
    calibrate_threshold,
    category_paths,
    evaluate_image_level,
    evaluate_localization,
    load_mask,
    load_model,
    load_test_images,
    load_threshold,
    make_mask,
    plot_confusion_matrix,
    plot_roc_curve,
    print_report,
    reconstruction_error,
    save_threshold,
    set_seeds,
    show_heatmap,
    show_localization,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the autoencoder")
    parser.add_argument("--category", default="screw",
                        help="MVTec category folder name (screw, wood, bottle, ...)")
    parser.add_argument("--num-heatmap-examples", type=int, default=4,
                        help="How many normal/defect heatmaps to save")
    parser.add_argument("--skip-localization", action="store_true",
                        help="Skip the pixel-level evaluation")
    parser.add_argument("--domain-calibration", type=float, default=0.0,
                        help="Fraction of normal test images used to fit the "
                             "threshold (e.g. 0.5). Those images are excluded "
                             "from the metrics. 0 keeps the train.py threshold.")
    return parser.parse_args()


def apply_domain_calibration(model, test_images, test_labels, defect_types,
                             mask_paths, fraction, category, threshold_path):
    """Fit the threshold on held-out normal test images."""
    good_index = np.flatnonzero(test_labels == 0)
    n_calibration = int(round(fraction * len(good_index)))

    if n_calibration < 1 or n_calibration >= len(good_index):
        raise ValueError(
            f"--domain-calibration {fraction} would use {n_calibration} of "
            f"{len(good_index)} normal images; choose a fraction that leaves "
            f"at least one normal image for evaluation."
        )

    calibration_index = good_index[:n_calibration]
    keep = np.setdiff1d(np.arange(len(test_labels)), calibration_index)

    print(f"Calibrating on {n_calibration} normal test images; "
          f"evaluating on the remaining {len(keep)} images.")

    info = calibrate_threshold(model, test_images[calibration_index], category)
    save_threshold(info, threshold_path)

    return (
        float(info["threshold"]),
        float(info["pixel_threshold"]),
        test_images[keep],
        test_labels[keep],
        [defect_types[i] for i in keep],
        [mask_paths[i] for i in keep],
    )


def main():
    args = parse_args()
    category = args.category
    paths = category_paths(category)

    set_seeds()

    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["heatmaps"].mkdir(parents=True, exist_ok=True)

    # 1. Load the trained model and the saved threshold.
    model = load_model(paths["model"])
    threshold_info = load_threshold(paths["threshold"])
    threshold = float(threshold_info["threshold"])
    pixel_threshold = float(threshold_info.get("pixel_threshold", threshold))

    # 2. Test data.
    test_images, test_labels, defect_types, mask_paths = load_test_images(category)
    print(f"[{category}] Test set: {len(test_images)} images "
          f"({int(test_labels.sum())} defective, "
          f"{int((test_labels == 0).sum())} normal)")

    # 2b. Optional in-domain calibration.
    if args.domain_calibration > 0:
        (threshold, pixel_threshold, test_images, test_labels,
         defect_types, mask_paths) = apply_domain_calibration(
            model, test_images, test_labels, defect_types, mask_paths,
            args.domain_calibration, category, paths["calibrated_threshold"],
        )
        protocol = "in-domain calibration"
        threshold_source = str(paths["calibrated_threshold"])
    else:
        protocol = "validation split"
        threshold_source = str(paths["threshold"])

    print("Threshold protocol:", protocol)

    # 3. Image-level metrics.
    metrics = evaluate_image_level(model, test_images, test_labels, threshold)

    # 4. Pixel-level metrics.
    localization_metrics = {}
    if not args.skip_localization:
        localization_metrics = evaluate_localization(
            model, test_images, mask_paths, pixel_threshold
        )

    # 5. Report and figures.
    print_report(metrics, localization_metrics, threshold)

    plot_confusion_matrix(metrics["confusion_matrix"],
                          paths["results"] / "confusion_matrix.png")
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"],
                   paths["results"] / "roc_curve.png")

    # 6. Machine-readable summary.
    summary = {
        "category": category,
        "threshold_protocol": protocol,
        "threshold_source": threshold_source,
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        **localization_metrics,
    }
    summary_path = paths["results"] / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    protocol_name = (
        "metrics_summary_domain_calibration.json"
        if args.domain_calibration > 0
        else "metrics_summary_val_threshold.json"
    )
    protocol_path = paths["results"] / protocol_name
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Saved metrics summary:", summary_path)
    print("Saved protocol summary:", protocol_path)

    # 7. Example heatmaps.
    n_examples = args.num_heatmap_examples
    normal_index = [i for i, y in enumerate(test_labels) if y == 0][:n_examples]
    defect_index = [i for i, y in enumerate(test_labels) if y == 1][:n_examples]

    for i in normal_index + defect_index:
        reconstruction = model.predict(test_images[i:i + 1], verbose=0)[0]
        error_map = reconstruction_error(test_images[i], reconstruction)
        score = anomaly_score(error_map)

        result = {
            "original": test_images[i],
            "reconstruction": reconstruction,
            "error_map": error_map,
            "anomaly_score": score,
            "prediction": "ANOMALY" if score > threshold else "NORMAL",
            "threshold": threshold,
            "predicted_mask": make_mask(error_map, pixel_threshold),
        }

        is_defect = test_labels[i] == 1
        prefix = "defect" if is_defect else "normal"
        name = f"{prefix}_{defect_types[i]}_{i:03d}"

        show_heatmap(
            result,
            save_path=paths["heatmaps"] / f"example_{name}.png",
            title_prefix=f"[{defect_types[i]}] ",
        )

        if is_defect and mask_paths[i] is not None:
            gt_mask = load_mask(mask_paths[i])
            show_localization(
                result,
                gt_mask,
                save_path=paths["heatmaps"] / f"localization_{name}.png",
            )

    print("\nEvaluation complete. Figures are in", paths["results"])


if __name__ == "__main__":
    main()
