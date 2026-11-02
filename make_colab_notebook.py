"""
make_colab_notebook.py — generate anomaly_detection_colab.ipynb.

The generator embeds the CURRENT contents of anomaly_detection.py, train.py,
evaluate.py and predict.py into %%writefile cells, so the notebook always
contains exactly the same code as the repository.

The notebook is category-driven: pick `screw`, `wood` or `bottle` once and
every cell (training, evaluation, figures, prediction, artefact download)
follows that category's model and results folder. The last cell compares the
three trained categories on the same protocol.

Run locally:
    python make_colab_notebook.py

Then upload anomaly_detection_colab.ipynb to Google Colab and run it top to
bottom (Runtime -> Change runtime type -> GPU first).
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def read_source(name):
    return (NB_DIR / name).read_text(encoding="utf-8")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def cells():
    anomaly_py = read_source("anomaly_detection.py")
    train_py = read_source("train.py")
    evaluate_py = read_source("evaluate.py")
    predict_py = read_source("predict.py")

    return [
        md(
            "# Anomaly Detection in Images Using Convolutional Autoencoders\n"
            "\n"
            "**Dataset:** MVTec AD — categories `screw`, `wood` and `bottle`  |  "
            "**Model:** CNN autoencoder (TensorFlow/Keras)\n"
            "\n"
            "Each category has its own autoencoder, trained only on that category's\n"
            "normal images, and its own threshold files, results folder and\n"
            "heatmaps folder:\n"
            "\n"
            "    models/<category>_autoencoder.keras\n"
            "    models/<category>_threshold.json              # validation split\n"
            "    models/<category>_threshold_calibrated.json   # in-domain calibration\n"
            "    results/<category>/                           # metrics + figures\n"
            "\n"
            "At test time the reconstruction error decides NORMAL vs ANOMALY:\n"
            "\n"
            "    error_map = mean over channels of (original - reconstruction)^2\n"
            "    score     = mean of the error map\n"
            "    threshold = mean + 3 * std of the scores of normal images\n"
            "\n"
            "The encoder ends at a 16x16x16 bottleneck. With 4,096 latent values\n"
            "for 16,384 input pixels the network is forced to compress, so it\n"
            "cannot simply copy a defect into the output.\n"
            "\n"
            "Two threshold protocols are shown:\n"
            "1. the validation-split threshold from `train.py`\n"
            "2. an in-domain threshold calibrated on held-out normal test images\n"
            "\n"
            "> Before running: **Runtime -> Change runtime type -> GPU**."
        ),
        code(
            "# Cell 1 - Install dependencies (Colab already has most of these)\n"
            "!pip -q install tensorflow opencv-python scikit-learn matplotlib pillow\n"
            "print('Dependencies ready.')\n"
        ),
        code(
            "# Cell 2 - Check GPU\n"
            "import tensorflow as tf\n"
            "gpus = tf.config.list_physical_devices('GPU')\n"
            "print('GPUs detected:', gpus)\n"
            "if not gpus:\n"
            "    print('WARNING: no GPU. Runtime > Change runtime type > GPU.')\n"
        ),
        code(
            "# Cell 3 - Dataset path and categories\n"
            "from pathlib import Path\n"
            "\n"
            "CATEGORIES = ['screw', 'wood', 'bottle']\n"
            "DATASET_ROOT = Path('mvtec_anomaly_detection')\n"
            "\n"
            "missing = [c for c in CATEGORIES\n"
            "           if not (DATASET_ROOT / c / 'train' / 'good').is_dir()]\n"
            "if not missing:\n"
            "    print('Dataset found:', DATASET_ROOT.resolve())\n"
            "else:\n"
            "    print('Missing:', missing, '- run Cell 4 to upload and extract it.')\n"
        ),
        code(
            "# Cell 4 - Upload the MVTec archive and extract it\n"
            "from google.colab import files\n"
            "\n"
            "if not all((DATASET_ROOT / c).is_dir() for c in CATEGORIES):\n"
            "    uploaded = files.upload()  # choose mvtec_anomaly_detection.tar.xz\n"
            "    archive = next(iter(uploaded))\n"
            "    print('Extracting (this takes a few minutes)...')\n"
            "    !tar -xJf {archive}\n"
            "for c in CATEGORIES:\n"
            "    assert (DATASET_ROOT / c / 'train' / 'good').is_dir(), c\n"
            "print('Dataset ready for:', ', '.join(CATEGORIES))\n"
        ),
        code(
            "# Cell 5 - Create anomaly_detection.py (core pipeline)\n"
            "%%writefile anomaly_detection.py\n"
            f"{anomaly_py}"
        ),
        code(
            "# Cell 6 - Create train.py\n"
            "%%writefile train.py\n"
            f"{train_py}"
        ),
        code(
            "# Cell 7 - Create evaluate.py\n"
            "%%writefile evaluate.py\n"
            f"{evaluate_py}"
        ),
        code(
            "# Cell 8 - Create predict.py\n"
            "%%writefile predict.py\n"
            f"{predict_py}"
        ),
        code(
            "# Cell 9 - Choose the category and bind its paths\n"
            "# 'screw', 'wood' and 'bottle' are all trained; switch CATEGORY and\n"
            "# re-run this cell to work on another one. Nothing is overwritten,\n"
            "# because every category has its own model and results folder.\n"
            "CATEGORIES = ['screw', 'wood', 'bottle']\n"
            "CATEGORY = 'screw'          # 'screw' | 'wood' | 'bottle'\n"
            "\n"
            "import numpy as np\n"
            "from anomaly_detection import (\n"
            "    IMAGE_SIZE, RESULTS_ROOT, category_paths,\n"
            "    set_seeds, check_gpu, check_dataset,\n"
            "    load_normal_images, build_autoencoder, describe_architecture,\n"
            "    train_autoencoder, save_model, score_images, compute_threshold,\n"
            "    compute_pixel_threshold, save_threshold, load_test_images,\n"
            "    evaluate_image_level, evaluate_localization, print_report,\n"
            "    plot_confusion_matrix, plot_roc_curve, calibrate_threshold,\n"
            "    plot_training_history,\n"
            "    reconstruct, reconstruction_error, anomaly_score, make_mask,\n"
            "    show_heatmap, predict, load_model, load_threshold,\n"
            ")\n"
            "\n"
            "PATHS = category_paths(CATEGORY)\n"
            "MODEL_PATH = PATHS['model']\n"
            "THRESHOLD_PATH = PATHS['threshold']\n"
            "CALIBRATED_THRESHOLD_PATH = PATHS['calibrated_threshold']\n"
            "RESULTS_DIR = PATHS['results']\n"
            "HEATMAPS_DIR = PATHS['heatmaps']\n"
            "\n"
            "set_seeds()\n"
            "check_gpu()\n"
            "check_dataset(CATEGORY)\n"
            "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "print('Selected category:', CATEGORY, '| image size:', IMAGE_SIZE)\n"
            "print('Model:', MODEL_PATH)\n"
            "for name in CATEGORIES:\n"
            "    saved = category_paths(name)['model'].exists()\n"
            "    print(f'  {name:7s} saved model on disk: {saved}')\n"
        ),
        code(
            "# Cell 10 - Train the autoencoder for the selected category\n"
            "# Equivalent to: !python train.py --category {CATEGORY}\n"
            "print('Training category:', CATEGORY)\n"
            "\n"
            "x_train, x_val = load_normal_images(CATEGORY)\n"
            "print('x_train:', x_train.shape, 'x_val:', x_val.shape)\n"
            "\n"
            "model = build_autoencoder()\n"
            "describe_architecture(model)\n"
            "\n"
            "history = train_autoencoder(model, x_train, x_val)\n"
            "save_model(model, MODEL_PATH)\n"
            "\n"
            "# Threshold from the normal validation images only\n"
            "val_scores, val_error_maps = score_images(model, x_val)\n"
            "\n"
            "threshold_info = compute_threshold(val_scores)\n"
            "threshold_info['category'] = CATEGORY\n"
            "threshold_info['pixel_threshold'] = compute_pixel_threshold(val_error_maps)\n"
            "threshold_info['n_validation_images'] = int(len(x_val))\n"
            "save_threshold(threshold_info, THRESHOLD_PATH)\n"
            "\n"
            "print('Image-level threshold:', threshold_info['threshold'])\n"
            "print('Pixel-level threshold:', threshold_info['pixel_threshold'])\n"
        ),
        code(
            "# Cell 11 - Training curves\n"
            "from IPython.display import Image, display\n"
            "\n"
            "plot_training_history(history, RESULTS_DIR / 'training_history.png')\n"
            "display(Image(filename=str(RESULTS_DIR / 'training_history.png')))\n"
        ),
        code(
            "# Cell 12 - Reconstruct one validation image\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "sample = x_val[0]\n"
            "recon = reconstruct(model, sample)\n"
            "err = reconstruction_error(sample, recon)\n"
            "score = anomaly_score(err)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "axes[0].imshow(sample.squeeze(), cmap='gray', vmin=0, vmax=1)\n"
            "axes[0].set_title('Original')\n"
            "axes[1].imshow(recon.squeeze(), cmap='gray', vmin=0, vmax=1)\n"
            "axes[1].set_title('Reconstructed')\n"
            "for ax in axes:\n"
            "    ax.axis('off')\n"
            "plt.suptitle(f'{CATEGORY} - normal validation image, score {score:.4f}')\n"
            "plt.show()\n"
            "\n"
            "fig.savefig(RESULTS_DIR / 'reconstruction_example.png', dpi=150,\n"
            "            bbox_inches='tight')\n"
        ),
        code(
            "# Cell 13 - Evaluate the full test set (validation threshold)\n"
            "test_images, test_labels, defect_types, mask_paths = load_test_images(CATEGORY)\n"
            "print(CATEGORY, 'test images:', len(test_images),\n"
            "      '| defective:', int(test_labels.sum()))\n"
            "\n"
            "threshold = float(threshold_info['threshold'])\n"
            "pixel_threshold = float(threshold_info['pixel_threshold'])\n"
            "\n"
            "metrics = evaluate_image_level(model, test_images, test_labels, threshold)\n"
            "loc_metrics = evaluate_localization(model, test_images, mask_paths,\n"
            "                                    pixel_threshold)\n"
            "print_report(metrics, loc_metrics, threshold)\n"
            "\n"
            "plot_confusion_matrix(metrics['confusion_matrix'],\n"
            "                      RESULTS_DIR / 'confusion_matrix.png')\n"
            "plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'],\n"
            "               RESULTS_DIR / 'roc_curve.png')\n"
        ),
        code(
            "# Cell 14 - Second protocol: calibrate on in-domain normal images\n"
            "# train/good and test/good come from different captures, so a threshold\n"
            "# fitted on train/good can be too high for test/good and flag nothing.\n"
            "# Half of the normal TEST images are used only for calibration and are\n"
            "# then removed from the evaluation set. Same protocol as\n"
            "# `python evaluate.py --category {CATEGORY} --domain-calibration 0.5`.\n"
            "good_index = np.flatnonzero(test_labels == 0)\n"
            "n_calibration = int(round(0.5 * len(good_index)))   # as evaluate.py\n"
            "cal_index = good_index[:n_calibration]\n"
            "keep = np.setdiff1d(np.arange(len(test_labels)), cal_index)\n"
            "print('Calibrating on', len(cal_index), 'normal images; evaluating on', len(keep))\n"
            "\n"
            "cal_info = calibrate_threshold(model, test_images[cal_index], CATEGORY)\n"
            "save_threshold(cal_info, CALIBRATED_THRESHOLD_PATH)\n"
            "\n"
            "threshold = float(cal_info['threshold'])\n"
            "pixel_threshold = float(cal_info['pixel_threshold'])\n"
            "\n"
            "cal_metrics = evaluate_image_level(model, test_images[keep],\n"
            "                                   test_labels[keep], threshold)\n"
            "print_report(cal_metrics, {}, threshold)\n"
        ),
        code(
            "# Cell 15 - Confusion matrix and ROC curve (calibrated threshold)\n"
            "import matplotlib.image as mpimg\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "for ax, name in zip(axes, ['confusion_matrix.png', 'roc_curve.png']):\n"
            "    ax.imshow(mpimg.imread(RESULTS_DIR / name))\n"
            "    ax.axis('off')\n"
            "plt.show()\n"
        ),
        code(
            "# Cell 16 - Anomaly heatmaps for example test images\n"
            "n_examples = 3\n"
            "normal_index = [i for i, y in enumerate(test_labels) if y == 0][:n_examples]\n"
            "defect_index = [i for i, y in enumerate(test_labels) if y == 1][:n_examples]\n"
            "\n"
            "for i in normal_index + defect_index:\n"
            "    recon = model.predict(test_images[i:i + 1], verbose=0)[0]\n"
            "    err = reconstruction_error(test_images[i], recon)\n"
            "    score = anomaly_score(err)\n"
            "    result = {\n"
            "        'original': test_images[i],\n"
            "        'reconstruction': recon,\n"
            "        'error_map': err,\n"
            "        'anomaly_score': score,\n"
            "        'prediction': 'ANOMALY' if score > threshold else 'NORMAL',\n"
            "        'threshold': threshold,\n"
            "        'predicted_mask': make_mask(err, pixel_threshold),\n"
            "    }\n"
            "    prefix = 'defect' if test_labels[i] == 1 else 'normal'\n"
            "    name = f'{prefix}_{defect_types[i]}_{i:03d}'\n"
            "    show_heatmap(result,\n"
            "                 save_path=HEATMAPS_DIR / f'example_{name}.png',\n"
            "                 show=True,\n"
            "                 title_prefix=f'[{CATEGORY} {defect_types[i]}] ')\n"
        ),
        code(
            "# Cell 17 - Single-image prediction for the selected category\n"
            "defect_dirs = sorted(p for p in PATHS['test'].iterdir()\n"
            "                     if p.is_dir() and p.name != 'good')\n"
            "sample_defect = str(sorted(defect_dirs[0].glob('*.png'))[0])\n"
            "print('Image:', sample_defect)\n"
            "\n"
            "result = predict(model, sample_defect, threshold, pixel_threshold)\n"
            "print('Prediction   :', result['prediction'])\n"
            "print('Anomaly Score:', f\"{result['anomaly_score']:.4f}\")\n"
            "print('Threshold    :', f\"{result['threshold']:.4f}\")\n"
        ),
        code(
            "# Cell 18 - Save/download this category's model + thresholds\n"
            "from google.colab import files\n"
            "import shutil\n"
            "\n"
            "for p in [MODEL_PATH, THRESHOLD_PATH, CALIBRATED_THRESHOLD_PATH,\n"
            "          RESULTS_DIR / 'training_history.png',\n"
            "          RESULTS_DIR / 'confusion_matrix.png',\n"
            "          RESULTS_DIR / 'roc_curve.png']:\n"
            "    print(' ', p)\n"
            "\n"
            "archive = f'{CATEGORY}_anomaly_artifacts'\n"
            "shutil.make_archive(archive, 'zip', '.', base_dir='models')\n"
            "files.download(archive + '.zip')  # comment out to skip\n"
        ),
        code(
            "# Cell 19 - Reload this category's artefacts without retraining\n"
            "reloaded_model = load_model(MODEL_PATH)\n"
            "reloaded_threshold = load_threshold(CALIBRATED_THRESHOLD_PATH)\n"
            "print('Category:', reloaded_threshold.get('category', CATEGORY))\n"
            "print('Reloaded threshold:', reloaded_threshold)\n"
            "\n"
            "r = predict(reloaded_model, sample_defect,\n"
            "            float(reloaded_threshold['threshold']),\n"
            "            float(reloaded_threshold['pixel_threshold']))\n"
            "print('Reloaded model:', r['prediction'], '| score', f\"{r['anomaly_score']:.4f}\")\n"
        ),
        md(
            "## All three categories\n"
            "\n"
            "`screw`, `wood` and `bottle` are all trained and each keeps its own\n"
            "model in `models/`. The cell below scores every category on its own\n"
            "test set with the in-domain calibrated threshold, so the numbers are\n"
            "directly comparable to\n"
            "`python evaluate.py --category <name> --domain-calibration 0.5`.\n"
            "\n"
            "It uses the models already saved in `models/`; run cells 9-14 once per\n"
            "category first if any of them is missing. The table is also written to\n"
            "`results/three_category_summary.json`."
        ),
        code(
            "# Cell 20 - Compare the three trained categories\n"
            "import json\n"
            "\n"
            "summary = {}\n"
            "print(f\"{'category':9s} {'thr':>8s} {'acc':>6s} {'prec':>6s} \"\n"
            "      f\"{'rec':>6s} {'f1':>6s} {'FP':>3s} {'FN':>3s}\")\n"
            "print('-' * 56)\n"
            "\n"
            "for name in CATEGORIES:\n"
            "    p = category_paths(name)\n"
            "    if not p['model'].exists():\n"
            "        print(f'{name:9s} no saved model - run cells 9-14 for it')\n"
            "        continue\n"
            "\n"
            "    cat_model = load_model(p['model'])\n"
            "    images, labels, _, _ = load_test_images(name)\n"
            "\n"
            "    # Same protocol as evaluate.py --domain-calibration 0.5\n"
            "    good = np.flatnonzero(labels == 0)\n"
            "    cal = good[: int(round(0.5 * len(good)))]\n"
            "    keep = np.setdiff1d(np.arange(len(labels)), cal)\n"
            "    info = calibrate_threshold(cat_model, images[cal], name)\n"
            "    save_threshold(info, p['calibrated_threshold'])\n"
            "\n"
            "    met = evaluate_image_level(cat_model, images[keep], labels[keep],\n"
            "                               float(info['threshold']))\n"
            "    tn, fp, fn, tp = met['confusion_matrix'].ravel()\n"
            "    summary[name] = {\n"
            "        'threshold': float(info['threshold']),\n"
            "        'accuracy': met['accuracy'],\n"
            "        'precision': met['precision'],\n"
            "        'recall': met['recall'],\n"
            "        'f1': met['f1'],\n"
            "        'roc_auc': met['roc_auc'],\n"
            "        'true_negatives': int(tn), 'false_positives': int(fp),\n"
            "        'false_negatives': int(fn), 'true_positives': int(tp),\n"
            "    }\n"
            "    print(f\"{name:9s} {info['threshold']:8.5f} {met['accuracy']:6.3f} \"\n"
            "          f\"{met['precision']:6.3f} {met['recall']:6.3f} \"\n"
            "          f\"{met['f1']:6.3f} {int(fp):3d} {int(fn):3d}\")\n"
            "\n"
            "RESULTS_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "out = RESULTS_ROOT / 'three_category_summary.json'\n"
            "with open(out, 'w', encoding='utf-8') as f:\n"
            "    json.dump(summary, f, indent=4)\n"
            "print('\\nSaved:', out)\n"
        ),
    ]


def main():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells(),
    }

    out_path = NB_DIR / "anomaly_detection_colab.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print("Created:", out_path)


if __name__ == "__main__":
    main()
