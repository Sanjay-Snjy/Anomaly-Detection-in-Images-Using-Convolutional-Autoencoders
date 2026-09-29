"""
evaluate.py — evaluate the trained autoencoder on the MVTec screw test set.

Flow:
    load trained model + threshold
        -> load test dataset (good + all defect types, with GT masks)
        -> predict all images (continuous anomaly scores)
        -> image-level metrics: accuracy / precision / recall / F1 / ROC-AUC
        -> localization metrics vs GT masks: IoU / pixel P / R / F1
        -> plots: confusion_matrix.png, roc_curve.png, example heatmaps

Run:
    python evaluate.py [--num-heatmap-examples 4] [--dataset-root PATH]
"""

from __future__ import annotations

import argparse
import json

from anomaly_detection import (
    AnomalyDetectionConfig,
    calculate_anomaly_score,
    calculate_reconstruction_error,
    create_predicted_mask,
    evaluate_image_level,
    evaluate_localization,
    generate_anomaly_heatmap,
    load_mask,
    load_test_dataset,
    load_threshold,
    load_trained_model,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics_report,
    set_seeds,
    visualize_localization_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MVTec screw autoencoder")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-heatmap-examples", type=int, default=4,
                        help="How many example heatmaps to save per class")
    parser.add_argument("--skip-localization", action="store_true",
                        help="Skip pixel-level evaluation against GT masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnomalyDetectionConfig()
    if args.dataset_root is not None:
        config.dataset_root = Path(args.dataset_root)
    if args.category is not None:
        config.category = args.category
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    set_seeds(config.seed)
    config.ensure_output_dirs()

    # 1. Load trained artefacts (no retraining).
    model = load_trained_model(config.model_path)
    threshold_info = load_threshold(config.threshold_path)
    threshold = float(threshold_info["threshold"])
    pixel_threshold = float(threshold_info.get("pixel_threshold", threshold))

    # 2. Load test data (normal + defective, with ground-truth mask paths).
    test_images, test_labels, defect_types, mask_paths = load_test_dataset(config)
    print(f"Test set: {len(test_images)} images "
          f"({int(test_labels.sum())} anomalous, {int((test_labels == 0).sum())} normal)")

    # 3. Predict every test image and compute image-level metrics.
    metrics = evaluate_image_level(
        model, test_images, test_labels, threshold, batch_size=config.batch_size
    )

    # 4. Pixel-level localization metrics against the MVTec masks.
    localization_metrics: dict = {}
    if not args.skip_localization:
        localization_metrics = evaluate_localization(
            model,
            test_images,
            mask_paths,
            pixel_threshold,
            batch_size=config.batch_size,
            image_size=config.image_size,
        )

    # 5. Console report + saved figures.
    print_metrics_report(metrics, localization_metrics, threshold)

    plot_confusion_matrix(metrics["confusion_matrix"],
                          config.results_dir / "confusion_matrix.png")
    plot_roc_curve(metrics.get("fpr"), metrics.get("tpr"),
                   metrics["roc_auc"], config.results_dir / "roc_curve.png")

    # 6. Save a machine-readable metrics summary (nice for the report/viva).
    summary = {
        "category": config.category,
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        **localization_metrics,
    }
    summary_path = config.results_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4)
    print(f"Saved metrics summary: {summary_path}")

    # 7. Example heatmaps: a few normal and a few defective test images.
    n_examples = args.num_heatmap_examples
    normal_idx = [i for i, y in enumerate(test_labels) if y == 0][:n_examples]
    defect_idx = [i for i, y in enumerate(test_labels) if y == 1][:n_examples]
    for i in normal_idx + defect_idx:
        recon = model.predict(test_images[i : i + 1], verbose=0)[0]
        error_map = calculate_reconstruction_error(test_images[i], recon)
        anomaly_score = calculate_anomaly_score(error_map)
        result = {
            "original": test_images[i],
            "reconstruction": recon,
            "error_map": error_map,
            "anomaly_score": anomaly_score,
            "prediction": "ANOMALY" if anomaly_score > threshold else "NORMAL",
            "threshold": threshold,
            "predicted_mask": create_predicted_mask(error_map, pixel_threshold),
        }
        is_defect = test_labels[i] == 1
        prefix = "defect" if is_defect else "normal"
        stem = f"{prefix}_{defect_types[i]}_{i:03d}"
        generate_anomaly_heatmap(
            result,
            save_path=config.heatmaps_dir / f"example_{stem}.png",
            title_prefix=f"[{defect_types[i]}] ",
        )
        # Ground-truth localization comparison for defective images.
        if is_defect and mask_paths[i] is not None:
            gt_mask = load_mask(mask_paths[i], config.image_size)
            visualize_localization_comparison(
                result, gt_mask,
                save_path=config.heatmaps_dir / f"localization_{stem}.png",
            )

    print("\nEvaluation complete. Figures saved under", config.results_dir)


if __name__ == "__main__":
    main()
