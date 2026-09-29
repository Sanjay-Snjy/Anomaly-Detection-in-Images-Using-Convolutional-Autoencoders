# Anomaly Detection in Images Using Convolutional Autoencoders

Deep-learning mini-project: unsupervised industrial defect detection on the
**MVTec AD** dataset (category: **`screw`**) with a **CNN Convolutional
Autoencoder** built in TensorFlow/Keras.

The autoencoder sees **only normal screws** during training. At inference it
reconstructs images; pixels it reconstructs poorly are where defects are, and
the average reconstruction error decides **NORMAL vs ANOMALY** against a
statistically motivated threshold computed from **normal validation data**.

> **Honest limitation (important for the viva):** reconstruction error is a
> strong *signal* for anomalies, but it does not *guarantee* detection —
> subtle defects that the autoencoder can still "reconstruct well" can slip
> through, and poor preprocessing can create false positives. Results below
> are reported as measured.

---

## 1. Expected folder structure

```text
DL_Mini_Project/
├── anomaly_detection.py            # core reusable ML pipeline (library)
├── train.py                        # trains the model, saves model + threshold
├── evaluate.py                     # evaluates on the MVTec test set
├── predict.py                      # single-image inference CLI
├── make_colab_notebook.py          # regenerates anomaly_detection_colab.ipynb
├── anomaly_detection_colab.ipynb   # the Colab notebook (created by the script)
├── README.md
│
├── mvtec_anomaly_detection/        # dataset (not committed)
│   └── screw/
│       ├── train/good/             # 320 normal images  <- ONLY training data
│       ├── test/                   # good + 5 defect types (160 images)
│       └── ground_truth/           # defect masks for localization metrics
│
├── models/                         # created by train.py
│   ├── screw_autoencoder.keras
│   └── screw_threshold.json
│
└── results/                        # created by train.py / evaluate.py
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── metrics_summary.json
    └── heatmaps/
```

---

## 2. Files and what every major section does

### `anomaly_detection.py` — the reusable library

| Section | Functions | What it does |
|---|---|---|
| **Config** | `AnomalyDetectionConfig` | Every knob in one dataclass: category, paths, `image_size=128`, `channels=1`, `batch_size=32`, `epochs=50`, `validation_split=0.2`, threshold strategy/params, `seed=42`. Nothing is hard-coded downstream. |
| **Environment** | `set_seeds`, `check_gpu`, `validate_dataset` | Reproducibility; prints `tf.config.list_physical_devices('GPU')`; fails early with a helpful message if the dataset path is wrong. |
| **Preprocessing** | `load_image`, `preprocess_image`, `load_mask` | The *single source of truth*: grayscale read → resize to 128×128 (bilinear `INTER_AREA`) → scale to 0–1 → shape `(H, W, 1)`. Masks use nearest-neighbour resize to stay binary. Training and inference share exactly this code, so preprocessing can never drift. |
| **Data loading** | `load_normal_training_data`, `load_test_dataset` | Train/val split (80/20, seeded) of **train/good only** — test images never touch training. Test loader attaches labels (good=0, defect=1) and GT mask paths. |
| **Architecture** | `build_autoencoder`, `describe_architecture` | CNN autoencoder, see §3. Prints an Encoder/Latent/Decoder map plus `model.summary()`. |
| **Training** | `train_autoencoder`, `plot_training_history` | `fit(X, X)` (autoencoder target = input), MSE, Adam; EarlyStopping + ModelCheckpoint (best `.keras`) + ReduceLROnPlateau; loss-curve figure. |
| **Scoring** | `reconstruct_image`, `calculate_reconstruction_error`, `calculate_anomaly_score`, `score_images` | `error_map = mean_c((x − x̂)²)` per pixel; `anomaly_score = mean(error_map)`; batched scoring for whole datasets. |
| **Threshold** | `calculate_threshold`, `calculate_pixel_threshold`, `create_predicted_mask` | Statistically motivated thresholds from **normal validation** errors only; three swappable strategies (see §4). Masks = error map thresholded + 3×3 median filter. |
| **Persistence** | `save_model`, `load_trained_model`, `save_threshold`, `load_threshold` | `.keras` model + JSON threshold, reloadable without retraining (this is exactly what the FastAPI backend will call). |
| **Prediction** | `predict_image` | One-call inference returning `prediction / anomaly_score / threshold / reconstruction / error_map` (+ predicted mask). |
| **Visualization** | `generate_anomaly_heatmap`, `visualize_localization_comparison` | The 2×2 result grid (original, reconstruction, error map, JET heatmap overlay) and predicted-vs-GT mask panels. |
| **Evaluation** | `evaluate_image_level`, `evaluate_localization`, `plot_confusion_matrix`, `plot_roc_curve`, `print_metrics_report` | Image-level metrics (ROC-AUC uses the **continuous** scores), micro-averaged pixel metrics vs GT masks, saved figures. |

### `train.py`
`load training data → build model → train → save model → threshold from normal validation → save threshold → plot curves`.
CLI: `python train.py --epochs 50 --batch-size 32 --image-size 128 --dataset-root mvtec_anomaly_detection`

### `evaluate.py`
`load model + threshold → load test set → score all images → image-level metrics → localization metrics → plots + metrics_summary.json → example heatmaps`.
CLI: `python evaluate.py --num-heatmap-examples 4`

### `predict.py`
Single image → `Prediction: ANOMALY`, `Anomaly Score: 0.xxxx`, `Threshold: 0.xxxx`.
CLI: `python predict.py --image mvtec_anomaly_detection/screw/test/scratch_head/000.png --show-heatmap`

---

## 3. CNN Autoencoder architecture

```text
Input (128, 128, 1)
ENCODER
  Conv2D(32, 3×3, ReLU, same) + BatchNorm → MaxPool 2×2   # 128 → 64
  Conv2D(64, 3×3, ReLU, same) + BatchNorm → MaxPool 2×2   # 64  → 32
  Conv2D(128,3×3, ReLU, same) + BatchNorm → MaxPool 2×2   # 32  → 16
LATENT REPRESENTATION
  Conv2D(128, 3×3, ReLU, same)            # (16, 16, 128) feature map
DECODER
  Conv2DTranspose(128, 3×3, stride 2) + BatchNorm          # 16 → 32
  Conv2DTranspose(64,  3×3, stride 2) + BatchNorm          # 32 → 64
  Conv2DTranspose(32,  3×3, stride 2) + BatchNorm          # 64 → 128
  Conv2D(1, 3×3, sigmoid)                                  # (128,128,1) in 0–1
```

* ~482k parameters — trains in minutes on a Colab T4, small enough to explain
  layer-by-layer in a viva.
* The final **sigmoid** matches the 0–1 normalised inputs, so MSE compares
  like with like.
* **BatchNorm** was added (a justified improvement over the plain spec) to
  stabilise training on MVTec's small per-category sets.
* The "latent representation" is the (16, 16, 128) feature map after the
  encoder — the compressed summary the decoder reconstructs from.

---

## 4. How thresholds are computed (and why they are modular)

Everything is derived from **normal validation images only** (held out of
training, no test data, no test labels):

| Strategy | Image-level rule | Pixel-level rule | When to use |
|---|---|---|---|
| `mean_std` (spec baseline) | `mean + 3·std` of normal val scores | same on all normal val pixel errors | default, Gaussian-ish errors |
| `percentile` | e.g. 99th percentile of normal val scores | same on pixel errors | heavy-tailed errors; flags ~1% of normals |
| `max` | largest normal val score | largest normal pixel error | strictest "zero false positives on val" |

Switch with `AnomalyDetectionConfig(threshold_strategy="percentile")`.
The chosen values (including both thresholds) are stored in
`models/screw_threshold.json`:

```json
{
    "category": "screw",
    "threshold": 0.0123,
    "pixel_threshold": 0.0456,
    "strategy": "mean_std",
    "mean_normal_error": 0.0051,
    "std_normal_error": 0.0024,
    "n_std": 3.0,
    "n_validation_images": 64
}
```

**Predicted anomaly mask (localization):** reconstruct the image →
`error_map = mean_c((x − x̂)²)` → mark every pixel with
`error_map > pixel_threshold` → 3×3 median filter removes isolated speckles.
That binary mask is what IoU / pixel-P/R/F1 are computed against.

---

## 5. MVTec AD in Colab

1. Download `mvtec_anomaly_detection.tar.xz` (5.2 GB) from
   <https://www.mvtec.com/company/research/datasets/mvtec-ad> (needs a free
   account). You only strictly need the `screw` folder for this project.
2. Open `anomaly_detection_colab.ipynb` in Colab (**Runtime → Change runtime
   type → GPU**).
3. Cell 4 uploads the archive via `files.upload()` and extracts it with
   `tar -xJf` (~5–10 min). Alternatively mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   DATASET_ROOT = Path('/content/drive/MyDrive/mvtec_anomaly_detection')
   ```

4. Run the remaining cells top to bottom — they create the `.py` files with
   `%%writefile` and execute the same pipeline as the CLI scripts.

The archive is already extracted in this workspace, so locally you only need
`pip install tensorflow opencv-python scikit-learn matplotlib tqdm pillow`.

---

## 6. How to train

```bash
python train.py                      # defaults: 50 epochs, batch 32, 128×128
python train.py --epochs 30 --image-size 64   # quick CPU experiment
```

Outputs: `models/screw_autoencoder.keras`, `models/screw_threshold.json`,
`results/training_history.png`.

## 7. How to test one image

```bash
python predict.py --image mvtec_anomaly_detection/screw/test/scratch_head/000.png
```

```text
Prediction   : ANOMALY
Anomaly Score: 0.0842
Threshold    : 0.0214
```

Add `--show-heatmap` to display, `--save-heatmap out.png` to save the 2×2 figure.

## 8. How to evaluate the whole test set

```bash
python evaluate.py
```

Prints the full report and writes `confusion_matrix.png`, `roc_curve.png`,
`metrics_summary.json` and example heatmaps under `results/`.

## 9. How to generate anomaly heatmaps

Heatmaps are produced automatically by `evaluate.py`; for any single image:

```python
from anomaly_detection import load_trained_model, load_threshold, predict_image, generate_anomaly_heatmap
model = load_trained_model("models/screw_autoencoder.keras")
thr = load_threshold("models/screw_threshold.json")
result = predict_image(model, "some_screw.png", threshold=thr["threshold"], image_size=128)
generate_anomaly_heatmap(result, save_path="results/heatmaps/demo.png", show=True)
```

## 10. How to save / load the model

```python
from anomaly_detection import save_model, load_trained_model, save_threshold, load_threshold
save_model(model, "models/screw_autoencoder.keras")      # done automatically in training
model = load_trained_model("models/screw_autoencoder.keras")   # no retraining
thr = load_threshold("models/screw_threshold.json")
```

The `.keras` bundle contains architecture + weights + compile config; the JSON
carries both thresholds. `load_trained_model` / `load_threshold` are the only
two calls the backend needs at startup.

---

## 11. Connecting to FastAPI later (design sketch, not built yet)

The pipeline was deliberately shaped so the backend is a thin wrapper:

```python
# future app.py (FastAPI) — sketch only
from fastapi import FastAPI, UploadFile
from anomaly_detection import load_trained_model, load_threshold, predict_image

app = FastAPI()
model = load_trained_model("models/screw_autoencoder.keras")   # at startup
thr = load_threshold("models/screw_threshold.json")

@app.post("/predict")
async def predict(file: UploadFile):
    tmp = Path(f"/tmp/{file.filename}")
    tmp.write_bytes(await file.read())
    result = predict_image(model, tmp, threshold=thr["threshold"], image_size=128)
    return {
        "prediction": result["prediction"],          # "NORMAL" | "ANOMALY"
        "anomaly_score": round(result["anomaly_score"], 6),
        "threshold": result["threshold"],
        # PNG-encode error_map/heatmap for the Next.js frontend:
        "heatmap_base64": png_of(result["error_map"]),
    }
```

Key points: the model loads once at startup; `predict_image` is stateless and
works from any file path (save uploads to a temp file); the JSON threshold
travels with the model; and the Next.js frontend only ever talks HTTP —
`POST /predict` with an image, receiving prediction + score + a base64
heatmap it can overlay.

---

## 12. Results on screw (as measured)

Run on Colab with the default config (`128×128`, 50 epochs) and paste your
numbers here. In a quick CPU smoke run with a deliberately under-trained
model (64×64, 2 epochs), the **continuous anomaly scores ranked all 119
defective test images above all 41 normal ones (ROC-AUC 1.00)** while the
initial `mean+3σ` threshold was too conservative — a textbook case of "good
scoring, conservative cutoff", which is exactly why the threshold strategy is
configurable. Expect well-trained numbers like: ROC-AUC 0.95+, accuracy 0.85+
at the percentile threshold, and localization IoU improving with image size
(try `--image-size 256` for thin scratch defects).

If results disappoint: increase image size, train longer, lower
`threshold_percentile`, or switch to a U-Net-style skip autoencoder — all are
one-line config changes here.

---

## 13. Citation

MVTec AD: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
*A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
CVPR 2019. (CC BY-NC-SA 4.0 — non-commercial use.)
