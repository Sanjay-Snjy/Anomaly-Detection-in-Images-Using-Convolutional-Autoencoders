"""
make_colab_notebook.py — generate anomaly_detection_colab.ipynb.

The generator embeds the CURRENT contents of anomaly_detection.py, train.py,
evaluate.py and predict.py into %%writefile cells, so the notebook always
ships exactly the same code as the repository files.

Run locally:
    python make_colab_notebook.py
Then upload anomaly_detection_colab.ipynb to Google Colab and run it top
to bottom (Runtime -> Change runtime type -> GPU first).
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def read_source(name: str) -> str:
    return (NB_DIR / name).read_text(encoding="utf-8")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def cells() -> list:
    anomaly_py = read_source("anomaly_detection.py")
    train_py = read_source("train.py")
    evaluate_py = read_source("evaluate.py")
    predict_py = read_source("predict.py")

    return [
        md(
            "# Anomaly Detection in Images Using Convolutional Autoencoders\n"
            "\n"
            "**Dataset:** MVTec AD — category: `screw`  |  "
            "**Model:** CNN Convolutional Autoencoder (TensorFlow/Keras)\n"
            "\n"
            "Pipeline: normal images only → resize → normalize → train/val split → "
            "CNN encoder → latent → CNN decoder → reconstruction error → threshold "
            "(from normal validation data) → NORMAL/ANOMALY → heatmap → evaluation.\n"
            "\n"
            "This notebook **executes and demonstrates** the `.py` files; the ML "
            "logic lives in `anomaly_detection.py` (created below via `%%writefile`).\n"
            "\n"
            "> Before running: **Runtime → Change runtime type → GPU**."
        ),
        code(
            "# Cell 1 — Install dependencies (Colab already ships TF, OpenCV, sklearn)\n"
            "!pip -q install tensorflow opencv-python scikit-learn matplotlib tqdm pillow\n"
            "print('Dependencies ready.')"
        ),
        code(
            "# Cell 2 — Check GPU\n"
            "import tensorflow as tf\n"
            "gpus = tf.config.list_physical_devices('GPU')\n"
            "print('GPUs detected:', gpus)\n"
            "if not gpus:\n"
            "    print('WARNING: no GPU. Runtime > Change runtime type > GPU.')"
        ),
        code(
            "# Cell 3 — Dataset path\n"
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "# The notebook expects the dataset at ./mvtec_anomaly_detection/\n"
            "# (created in Cell 4 from your upload of the 5 GB archive, or adapt\n"
            "#  DATASET_ROOT below if you keep it in Google Drive).\n"
            "DATASET_ROOT = Path('mvtec_anomaly_detection')\n"
            "if (DATASET_ROOT / 'screw' / 'train' / 'good').is_dir():\n"
            "    print('Using existing dataset root:', DATASET_ROOT.resolve())\n"
            "else:\n"
            "    print('Dataset not found yet — run Cell 4 to upload and extract it.')"
        ),
        code(
            "# Cell 4 — Upload the MVTec archive (5 GB) and extract it\n"
            "from google.colab import files\n"
            "from pathlib import Path\n"
            "\n"
            "if not Path('mvtec_anomaly_detection/screw').is_dir():\n"
            "    uploaded = files.upload()  # choose mvtec_anomaly_detection.tar.xz\n"
            "    archive = next(iter(uploaded))\n"
            "    print('Extracting (takes a few minutes for 5 GB)...')\n"
            "    !tar -xJf {archive}\n"
            "assert (DATASET_ROOT / 'screw' / 'train' / 'good').is_dir(), (\n"
            "    'Extraction failed — check the archive name and re-run Cell 4.'\n"
            ")\n"
            "print('Dataset ready.')" 
        ),
        code(
            "# Cell 5 — Create anomaly_detection.py (core reusable pipeline)\n"
            "%%writefile anomaly_detection.py\n"
            f"{anomaly_py}"
        ),
        code(
            "# Cell 6 — Create train.py\n"
            "%%writefile train.py\n"
            f"{train_py}"
        ),
        code(
            "# Cell 7 — Create evaluate.py\n"
            "%%writefile evaluate.py\n"
            f"{evaluate_py}"
        ),
        code(
            "# Cell 8 — Create predict.py\n"
            "%%writefile predict.py\n"
            f"{predict_py}"
        ),
        code(
            "# Cell 9 — Imports, seeds, config\n"
            "import random\n"
            "import numpy as np\n"
            "import tensorflow as tf\n"
            "from pathlib import Path\n"
            "\n"
            "from anomaly_detection import (\n"
            "    AnomalyDetectionConfig, set_seeds, check_gpu, validate_dataset,\n"
            "    load_normal_training_data, build_autoencoder, describe_architecture,\n"
            "    train_autoencoder, plot_training_history, score_images,\n"
            "    calculate_threshold, calculate_pixel_threshold, save_threshold,\n"
            "    save_model, load_test_dataset, evaluate_image_level,\n"
            "    evaluate_localization, plot_confusion_matrix, plot_roc_curve,\n"
            "    print_metrics_report, calculate_reconstruction_error,\n"
            "    calculate_anomaly_score, generate_anomaly_heatmap,\n"
            "    predict_image, load_threshold, load_trained_model,\n"
            ")\n"
            "\n"
            "SEED = 42\n"
            "set_seeds(SEED)\n"
            "\n"
            "config = AnomalyDetectionConfig(\n"
            "    category='screw',\n"
            "    dataset_root=DATASET_ROOT,\n"
            "    image_size=128,      # try 256 later for finer defects\n"
            "    channels=1,          # screw images are grayscale\n"
            "    batch_size=32,\n"
            "    epochs=50,\n"
            "    validation_split=0.2,\n"
            ")\n"
            "check_gpu()\n"
            "validate_dataset(config)\n"
            "config.ensure_output_dirs()\n"
            "print('Config:', config.to_dict())"
        ),
        code(
            "# Cell 10 — Train the model end-to-end (uses train.py's flow)\n"
            "# You can also simply run: !python train.py --epochs 50\n"
            "\n"
            "# 10a. Load ONLY normal training images (train/good) + validation split\n"
            "x_train, x_val, train_paths, val_paths = load_normal_training_data(config)\n"
            "print('x_train:', x_train.shape, 'x_val:', x_val.shape)\n"
            "\n"
            "# 10b. Build the CNN autoencoder\n"
            "model = build_autoencoder(config)\n"
            "describe_architecture(model)\n"
            "\n"
            "# 10c. Train (input == target), callbacks save the best model\n"
            "history = train_autoencoder(model, x_train, x_val, config)\n"
            "save_model(model, config.model_path)\n"
            "\n"
            "# 10d. Threshold from NORMAL VALIDATION errors only\n"
            "val_scores, val_error_maps = score_images(model, x_val, config.batch_size)\n"
            "threshold_info = calculate_threshold(val_scores, config.threshold_n_std,\n"
            "                                     strategy=config.threshold_strategy,\n"
            "                                     percentile=config.threshold_percentile)\n"
            "pixel_threshold = calculate_pixel_threshold(val_error_maps, config.mask_n_std,\n"
            "                                            strategy=config.threshold_strategy,\n"
            "                                            percentile=config.mask_percentile)\n"
            "threshold_info.update({'category': config.category,\n"
            "                       'pixel_threshold': float(pixel_threshold),\n"
            "                       'n_validation_images': int(len(x_val))})\n"
            "save_threshold(threshold_info, config.threshold_path)\n"
            "print('Image-level threshold:', threshold_info['threshold'])\n"
            "print('Pixel-level threshold:', pixel_threshold)"
        ),
        code(
            "# Cell 11 — Display training curves (also saved to results/)\n"
            "plot_training_history(history, config.results_dir / 'training_history.png', show=True)"
        ),
        code(
            "# Cell 12 — Test reconstruction on one validation image\n"
            "import matplotlib.pyplot as plt\n"
            "from anomaly_detection import reconstruct_image\n"
            "\n"
            "sample = x_val[0]\n"
            "recon = reconstruct_image(model, sample)\n"
            "err = calculate_reconstruction_error(sample, recon)\n"
            "score = calculate_anomaly_score(err)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "axes[0].imshow(sample.squeeze(), cmap='gray', vmin=0, vmax=1)\n"
            "axes[0].set_title('Original')\n"
            "axes[1].imshow(recon.squeeze(), cmap='gray', vmin=0, vmax=1)\n"
            "axes[1].set_title('Reconstructed')\n"
            "for ax in axes:\n"
            "    ax.axis('off')\n"
            "plt.suptitle(f'Normal validation image — anomaly score {score:.4f}')\n"
            "plt.show()\n"
            "\n"
            "config.results_dir.mkdir(parents=True, exist_ok=True)\n"
            "fig.savefig(config.results_dir / 'reconstruction_example.png', dpi=150,\n"
            "            bbox_inches='tight')"
        ),
        code(
            "# Cell 13 — Evaluate the full test set (image-level + localization)\n"
            "# Equivalent to: !python evaluate.py\n"
            "test_images, test_labels, defect_types, mask_paths = load_test_dataset(config)\n"
            "print('Test images:', len(test_images), '| anomalous:', int(test_labels.sum()))\n"
            "\n"
            "threshold = float(threshold_info['threshold'])\n"
            "metrics = evaluate_image_level(model, test_images, test_labels, threshold,\n"
            "                               batch_size=config.batch_size)\n"
            "\n"
            "pixel_threshold = float(threshold_info['pixel_threshold'])\n"
            "loc_metrics = evaluate_localization(model, test_images, mask_paths,\n"
            "                                    pixel_threshold,\n"
            "                                    batch_size=config.batch_size,\n"
            "                                    image_size=config.image_size)\n"
            "print_metrics_report(metrics, loc_metrics, threshold)"
        ),
        code(
            "# Cell 14 — Display confusion matrix and ROC curve\n"
            "plot_confusion_matrix(metrics['confusion_matrix'],\n"
            "                      config.results_dir / 'confusion_matrix.png')\n"
            "plot_roc_curve(metrics.get('fpr'), metrics.get('tpr'), metrics['roc_auc'],\n"
            "               config.results_dir / 'roc_curve.png')\n"
            "\n"
            "import matplotlib.image as mpimg\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "for ax, name in zip(axes, ['confusion_matrix.png', 'roc_curve.png']):\n"
            "    ax.imshow(mpimg.imread(config.results_dir / name))\n"
            "    ax.axis('off')\n"
            "plt.show()"
        ),
        code(
            "# Cell 15 — Anomaly heatmaps for example test images\n"
            "from anomaly_detection import create_predicted_mask\n"
            "\n"
            "n_examples = 3\n"
            "normal_idx = [i for i, y in enumerate(test_labels) if y == 0][:n_examples]\n"
            "defect_idx = [i for i, y in enumerate(test_labels) if y == 1][:n_examples]\n"
            "\n"
            "for i in normal_idx + defect_idx:\n"
            "    recon = model.predict(test_images[i:i+1], verbose=0)[0]\n"
            "    err = calculate_reconstruction_error(test_images[i], recon)\n"
            "    score = calculate_anomaly_score(err)\n"
            "    result = {\n"
            "        'original': test_images[i],\n"
            "        'reconstruction': recon,\n"
            "        'error_map': err,\n"
            "        'anomaly_score': score,\n"
            "        'prediction': 'ANOMALY' if score > threshold else 'NORMAL',\n"
            "        'threshold': threshold,\n"
            "    }\n"
            "    prefix = 'defect' if test_labels[i] == 1 else 'normal'\n"
            "    generate_anomaly_heatmap(\n"
            "        result,\n"
            "        save_path=config.heatmaps_dir / f'{prefix}_{defect_types[i]}_{i:03d}.png',\n"
            "        title_prefix=f'[{defect_types[i]}] ', show=True,\n"
            "    )"
        ),
        code(
            "# Cell 16 — Single-image prediction (uses predict_image)\n"
            "sample_defect = 'mvtec_anomaly_detection/screw/test/scratch_head/000.png'\n"
            "sample_defect = str(DATASET_ROOT / 'screw' / 'test' / 'scratch_head' / '000.png')\n"
            "result = predict_image(model, sample_defect, threshold=threshold,\n"
            "                       image_size=config.image_size)\n"
            "print('Prediction   :', result['prediction'])\n"
            "print('Anomaly Score:', f\"{result['anomaly_score']:.4f}\")\n"
            "print('Threshold    :', f\"{result['threshold']:.4f}\")"
        ),
        code(
            "# Cell 17 — Save/download model + threshold for the FastAPI backend\n"
            "from google.colab import files\n"
            "import shutil\n"
            "\n"
            "print('Artefacts:')\n"
            "for p in [config.model_path, config.threshold_path,\n"
            "          config.results_dir / 'training_history.png',\n"
            "          config.results_dir / 'confusion_matrix.png',\n"
            "          config.results_dir / 'roc_curve.png']:\n"
            "    print('  ', p)\n"
            "\n"
            "# Zip results/ + models/ for download\n"
            "shutil.make_archive('screw_anomaly_artifacts', 'zip', '.',\n"
            "                    base_dir='models')\n"
            "files.download('screw_anomaly_artifacts.zip')  # comment out to skip"
        ),
        code(
            "# Cell 18 — Sanity check: reload artefacts without retraining\n"
            "# This is exactly what the FastAPI backend will do at startup.\n"
            "reloaded_model = load_trained_model(config.model_path)\n"
            "reloaded_threshold = load_threshold(config.threshold_path)\n"
            "print('Reloaded threshold:', reloaded_threshold)\n"
            "r = predict_image(reloaded_model, str(DATASET_ROOT / 'screw' / 'test' /\n"
            "                                      'scratch_head' / '000.png'),\n"
            "                  threshold=float(reloaded_threshold['threshold']),\n"
            "                  image_size=config.image_size)\n"
            "print('Reloaded model prediction:', r['prediction'],\n"
            "      '| score', f\"{r['anomaly_score']:.4f}\")"
        ),
    ]


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells(),
    }
    out = NB_DIR / "anomaly_detection_colab.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out} with {len(notebook['cells'])} cells.")


if __name__ == "__main__":
    main()
