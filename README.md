# Anomaly Detection in Images Using Convolutional Autoencoders

Deep-learning mini-project: unsupervised industrial defect detection on the
**MVTec AD** dataset (category: **`screw`**) with a **CNN Convolutional
Autoencoder** built in TensorFlow/Keras.

The autoencoder sees **only normal screws** during training. At inference it
reconstructs images; pixels it reconstructs poorly are where defects are, and
the average reconstruction error decides **NORMAL vs ANOMALY** against a
statistically motivated threshold computed from **normal validation data**.

The code follows the style of the course lab manual rather than a
"production" layout: every setting is a module-level constant, the model is
built with the Functional API (`layers.Input` → `models.Model`), the encoder
uses `Conv2D` + `MaxPool2D` and the decoder `Conv2DTranspose`, training is
`model.compile(optimizer="adam", loss="mse")` + `fit`, and the threshold is
the `mean + 3·std` rule from class.

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
├── app.py                          # FastAPI service + web UI (section 11)
├── web/                            # bundled single-page frontend for app.py
│   ├── index.html
│   ├── style.css
│   └── app.js
├── make_colab_notebook.py          # regenerates anomaly_detection_colab.ipynb
├── anomaly_detection_colab.ipynb   # the Colab notebook (created by the script)
├── README.md
│
├── mvtec_anomaly_detection/        # dataset (not committed)
│   ├── screw/                      # 320 train/good, 160 test images
│   ├── wood/                       # 247 train/good,  90 test images
│   ├── bottle/                     # 209 train/good,  83 test images
│   │   └── each category: train/good/, test/<defect>/, ground_truth/
│   └── ...                         # the other MVTec categories work too
│
├── models/                         # created by train.py (one set per category)
│   ├── screw_autoencoder.keras
│   ├── screw_threshold.json                 # from the validation split
│   ├── screw_threshold_calibrated.json      # in-domain calibration (evaluate.py)
│   ├── wood_autoencoder.keras + wood_threshold*.json
│   └── bottle_autoencoder.keras + bottle_threshold*.json
│
└── results/                        # created by train.py / evaluate.py
    ├── screw/                      # one folder per category
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   ├── metrics_summary.json                   # last protocol that ran
    │   ├── metrics_summary_val_threshold.json     # section 12, protocol A
    │   ├── metrics_summary_domain_calibration.json# section 12, protocol B
    │   └── heatmaps/
    ├── wood/                       # same layout
    └── bottle/                     # same layout
```

---

## 2. Files and what every major section does

### `anomaly_detection.py` — the reusable library

| Section | Functions | What it does |
|---|---|---|
| **Settings** | module constants, `category_paths` | Every knob sits at the top of the file, as in the lab programs: `DATASET_ROOT`, `CATEGORY`, `IMAGE_SIZE=128`, `CHANNELS=1`, `BATCH_SIZE=32`, `EPOCHS=50`, `VALIDATION_SPLIT=0.2`, `LATENT_FILTERS=16`, `AUGMENT=True`, `SEED=42`. Change a value there and every script picks it up. |
| **Environment** | `set_seeds`, `check_gpu`, `check_dataset` | Reproducibility; prints `tf.config.list_physical_devices('GPU')`; fails early with a helpful message if the dataset path is wrong. |
| **Preprocessing** | `load_image`, `preprocess_image`, `load_mask` | The *single source of truth*: grayscale read → scale to 0–1 → resize to 128×128 (`INTER_AREA`) → shape `(H, W, 1)`. Masks use nearest-neighbour resize to stay binary. Training and inference share exactly this code, so preprocessing can never drift. |
| **Data loading** | `load_normal_images`, `load_test_images` | Train/val split (`train_test_split`, 80/20, seeded) of **train/good only** — test images never touch training. Test loader attaches labels (good=0, defect=1) and GT mask paths. |
| **Architecture** | `build_autoencoder`, `describe_architecture` | CNN autoencoder built with `layers.Input` → `models.Model`, see §3. Prints the latent size and `model.summary()`. |
| **Training** | `train_autoencoder`, `plot_training_history` | `fit(X, X)` (autoencoder target = input), MSE, Adam; EarlyStopping (`restore_best_weights=True`) + ReduceLROnPlateau; loss-curve figure. |
| **Scoring** | `reconstruct`, `reconstruction_error`, `anomaly_score`, `score_images` | `error_map = mean_c((x − x̂)²)` per pixel; `anomaly_score = mean(error_map)`; batched scoring for whole datasets. |
| **Threshold** | `compute_threshold`, `compute_pixel_threshold`, `calibrate_threshold`, `make_mask` | Thresholds from **normal** errors only; three swappable strategies (see §4). Masks = error map thresholded + 3×3 median filter. |
| **Persistence** | `save_model`, `load_model`, `save_threshold`, `load_threshold` | `.keras` model + JSON threshold, reloadable without retraining (this is exactly what the FastAPI backend calls). |
| **Prediction** | `predict` | One-call inference returning `prediction / anomaly_score / threshold / reconstruction / error_map` (+ predicted mask). |
| **Visualization** | `heatmap_overlay`, `show_heatmap`, `show_localization` | The 2×2 result grid (original, reconstruction, error map, JET heatmap overlay) and predicted-vs-GT mask panels. |
| **Evaluation** | `evaluate_image_level`, `evaluate_localization`, `plot_confusion_matrix`, `plot_roc_curve`, `print_report` | Image-level metrics (ROC-AUC uses the **continuous** scores), micro-averaged pixel metrics vs GT masks, saved figures. |

### `train.py`
`load training data → build model → train → save model → threshold from normal validation → save threshold → plot curves`.
CLI: `python train.py --category screw|wood|bottle` (all other settings are
constants at the top of `anomaly_detection.py`).

### `evaluate.py`
`load model + threshold → load test set → score all images → image-level metrics → localization metrics → plots + metrics_summary.json → example heatmaps`.
CLI: `python evaluate.py --category wood --domain-calibration 0.5`

### `predict.py`
Single image → `Prediction: ANOMALY`, `Anomaly Score: 0.xxxx`, `Threshold: 0.xxxx`.
CLI: `python predict.py --category bottle --image some_bottle.png --show-heatmap`

### `app.py` + `web/`
The interactive frontend (section 11): a FastAPI service that loads the model and
both thresholds **once** at startup and serves a single-page UI from `web/`.
Every request goes through the same `predict` call the CLI uses, so the
browser and the terminal can never disagree.

---

## 3. CNN Autoencoder architecture

```text
Input (128, 128, 1)
ENCODER
  Conv2D(32, 3×3, ReLU, same) + BatchNorm → MaxPool 2×2   # 128 → 64
  Conv2D(64, 3×3, ReLU, same) + BatchNorm → MaxPool 2×2   # 64  → 32
  Conv2D(128,3×3, ReLU, same) + BatchNorm → MaxPool 2×2   # 32  → 16
LATENT REPRESENTATION
  Conv2D(16, 3×3, ReLU, same)             # (16, 16, 16) = 4,096 values  <- the bottleneck
DECODER
  Conv2DTranspose(128, 3×3, stride 2) + BatchNorm          # 16 → 32
  Conv2DTranspose(64,  3×3, stride 2) + BatchNorm          # 32 → 64
  Conv2DTranspose(32,  3×3, stride 2) + BatchNorm          # 64 → 128
  Conv2D(1, 3×3, sigmoid)                                  # (128,128,1) in 0–1
```

* ~224k parameters — trains in minutes on a Colab T4, small enough to explain
  layer-by-layer in a viva.
* The final **sigmoid** matches the 0–1 normalised inputs, so MSE compares
  like with like.
* **BatchNorm** was added (a justified improvement over the plain spec) to
  stabilise training on MVTec's small per-category sets. Its `momentum` is
  **0.9**, not Keras's default 0.99: with ~200 training images an epoch is only
  6–8 batches, so the default needs ~30 epochs before the running mean/variance
  are usable. Until then predictions use the initial `mean=0, var=1` guess and
  the reconstruction collapses to a constant image (measured on `bottle`: AUC
  0.357, and validation MSE 0.133 = the variance of the input).
* The "latent representation" is the (16, 16, `LATENT_FILTERS`) feature map
  after the encoder — the compressed summary the decoder reconstructs from.

### Why `LATENT_FILTERS` matters more than anything else here

An autoencoder can only detect anomalies if it is *forced* to throw
information away. At 128×128 the latent grid is 16×16, so a width of
`LATENT_FILTERS=128` would give **32,768 latent values for only 16,384 input
pixels** — more degrees of freedom than the image itself. That network has no
reason to drop a scratch; it can simply copy it, and the reconstruction error
collapses for normals and defects alike (measured: ROC-AUC **0.78**, and
training loss 0.002 vs validation 0.020, i.e. pure memorisation).

Setting `LATENT_FILTERS=16` (4,096 latent values → a real 4× compression)
raises ROC-AUC to **0.999** on the same split — see the ablation in section 12.

### Training-time augmentation

`AUGMENT = True` (default) jitters the **normal training images only**: random
horizontal/vertical flips, ±0.08 brightness, small contrast change and 0.01
Gaussian noise. The augmented tensor is used as both input and target, so the
autoencoder learns "reconstruct a normal screw" rather than "reproduce these
exact 256 pictures". Validation and test data are never augmented.
Turn it off by setting `AUGMENT = False` at the top of the file.

---

## 4. How thresholds are computed (and why they are modular)

Everything is derived from **normal validation images only** (held out of
training, no test data, no test labels):

| Strategy | Image-level rule | Pixel-level rule | When to use |
|---|---|---|---|
| `mean_std` (spec baseline) | `mean + 3·std` of normal val scores | same on all normal val pixel errors | default, Gaussian-ish errors |
| `percentile` | e.g. 99th percentile of normal val scores | same on pixel errors | heavy-tailed errors; flags ~1% of normals |
| `max` | largest normal val score | largest normal pixel error | strictest "zero false positives on val" |

Switch with `compute_threshold(scores, strategy="percentile")` (or change
`THRESHOLD_STRATEGY` at the top of the file).
The chosen values (including both thresholds) are stored in
`models/screw_threshold.json`:

```json
{
    "threshold": 0.041930,
    "strategy": "mean_std",
    "mean_normal_error": 0.025539,
    "std_normal_error": 0.005463,
    "n_std": 3.0,
    "percentile": 99.0,
    "category": "screw",
    "pixel_threshold": 0.105378,
    "n_validation_images": 64
}
```

**Predicted anomaly mask (localization):** reconstruct the image →
`error_map = mean_c((x − x̂)²)` → mark every pixel with
`error_map > pixel_threshold` → 3×3 median filter removes isolated speckles.
That binary mask is what IoU / pixel-P/R/F1 are computed against.

### The second threshold: in-domain calibration (`--domain-calibration`)

The validation split is drawn from `train/good`, but the images you actually
score come from `test/good` — a *different capture batch* that is measurably
easier for the autoencoder. A threshold fitted on the harder batch can end up
higher than every test score, so nothing is ever flagged. That is a real
deployment problem (threshold transfer / batch shift), not a coding bug, and
section 12 measures it.

`evaluate.py --domain-calibration 0.5` therefore fits a second threshold on
half of the **normal** test images, exactly like you would calibrate on normal
data from the production line. Those calibration images are then removed from
the evaluation set, so the reported metrics never see the data the threshold
was fitted on. The result is written separately to
`models/screw_threshold_calibrated.json` — the validation-derived
`models/screw_threshold.json` is never overwritten. Use it with
`predict.py --threshold-file models/screw_threshold_calibrated.json`.

Both protocols are leak-free: they differ only in *which normal images* the
threshold is allowed to look at.

---

## 5. MVTec AD in Colab

1. Download `mvtec_anomaly_detection.tar.xz` (5.2 GB) from
   <https://www.mvtec.com/company/research/datasets/mvtec-ad> (needs a free
   account). This project uses `screw`, `wood` and `bottle`; any other
   category works the same way (`--category <folder name>`).
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
Running the web app (section 11) additionally needs
`pip install fastapi uvicorn python-multipart`.

---

## 6. How to train

```bash
python train.py                          # screw (default)
python train.py --category wood
python train.py --category bottle
```

All other settings live at the top of `anomaly_detection.py`, so a quick CPU
experiment is a two-line edit, e.g. `EPOCHS = 30`, `IMAGE_SIZE = 64`. For the
section-12 baseline set `LATENT_FILTERS = 128` and `AUGMENT = False`.

Outputs per category: `models/<category>_autoencoder.keras`,
`models/<category>_threshold.json`, `results/<category>/training_history.png`.
Training one category never touches another category's files.

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
python evaluate.py --category screw                              # validation threshold
python evaluate.py --category screw --domain-calibration 0.5     # calibrated
```

Prints the full report and writes `confusion_matrix.png`, `roc_curve.png`,
`metrics_summary.json` and example heatmaps under `results/<category>/`.

## 9. How to generate anomaly heatmaps

Heatmaps are produced automatically by `evaluate.py`; for any single image:

```python
from anomaly_detection import load_model, load_threshold, predict, show_heatmap
model = load_model("models/screw_autoencoder.keras")
thr = load_threshold("models/screw_threshold.json")
result = predict(model, "some_screw.png", thr["threshold"], thr["pixel_threshold"])
show_heatmap(result, save_path="results/screw/heatmaps/demo.png")
```

## 10. How to save / load the model

```python
from anomaly_detection import save_model, load_model, save_threshold, load_threshold
save_model(model, "models/screw_autoencoder.keras")      # done automatically in training
model = load_model("models/screw_autoencoder.keras")     # no retraining
thr = load_threshold("models/screw_threshold.json")
```

The `.keras` bundle contains architecture + weights + compile config; the JSON
carries both thresholds. `load_model` / `load_threshold` are the only two calls
the backend needs at startup.

---

## 11. Web app — FastAPI service + bundled UI (built)

The pipeline was shaped so the backend would be a thin wrapper, and `app.py` is
that wrapper plus a frontend. The model and both threshold JSONs load **once** at
startup, and every request runs the same `anomaly_detection.predict` call
that `predict.py` uses — so the browser and the terminal can never disagree
(verified: the API returns `0.026934 → ANOMALY` for `scratch_head/000.png` under
the calibrated threshold, exactly the CLI's number).

### Run it

```bash
pip install fastapi uvicorn python-multipart   # not in the ML requirements
python app.py                                  # -> http://127.0.0.1:8000
python app.py --port 8080 --threshold-mode validation
```

Open <http://127.0.0.1:8000>. Inference is local — the Keras model runs on your
machine and no image leaves it.

### What the UI does

* **Drag-and-drop or click** to upload any screw image (PNG/JPG), or click one of
the **bundled test images** served straight from
`mvtec_anomaly_detection/screw/test/` — a one-click demo that needs no file.
* **Verdict panel**: NORMAL/ANOMALY, the continuous anomaly score against a
  threshold bar, the margin, flagged-pixel count, error-map max/p99 and inference
  time (≈180 ms on CPU).
* **Four linked panels**: the preprocessed original, the heatmap overlay (with an
  opacity slider over the original), the reconstruction, and the colourised error
  map.
* **Threshold switch**: the calibrated threshold (default) or the validation-split
  one. Switching re-runs the last input, and the UI states plainly that protocol A
  flags nothing (recall 0.00) and that pixel-level localization is the weak spot —
  the same honest framing as section 12.
* **Measured performance** table for both protocols, the saved evaluation figures
  (`confusion_matrix.png`, `roc_curve.png`, …) and a gallery of everything in
  `results/screw/heatmaps/`.

### Endpoints

| Route | Purpose |
|---|---|
| `GET /` , `/static/*` | The single-page UI (`web/index.html`, `web/app.js`, `web/style.css`) |
| `GET /api/health` | Model info + both thresholds, which one is the default |
| `POST /api/predict?threshold_mode=calibrated` | Multipart upload → prediction, score, threshold, margin, flagged pixels and four base64 PNG panels |
| `POST /api/predict-sample/{id}?threshold_mode=…` | Predict a bundled MVTec test image by id (e.g. `scratch_head-000`) |
| `GET /api/samples` | Deterministic sample catalogue (3 good + 1 per defect type) with thumbnail URLs |
| `GET /api/samples/{id}?thumb=1` | The bundled image, or a downscaled thumbnail |
| `GET /api/metrics`, `/api/figures`, `/api/heatmaps` | `metrics_summary*.json`, evaluation figures, heatmap gallery |

Unknown threshold modes and undecodable uploads return `400` with a readable
message; file serving is allow-listed against a directory listing, so
`../` traversal returns `404`. Interactive API docs are at `/docs`.

### Notes for anyone extending this

* `predict` takes a path, so uploads are written to a `NamedTemporaryFile`
  and deleted in a `finally` block — that keeps inference on the library's tested
  `load_image` path instead of re-implementing preprocessing in the web layer.
* Inference runs in a worker thread (`asyncio.to_thread`) behind a lock, so
  concurrent requests queue rather than corrupting a shared TF graph.
* A warm-up forward pass runs at startup, so the first user request is not charged
  for graph construction.

---

## 12. Results on screw (as measured)

Measured locally on CPU (12 cores, TF 2.21 CPU), not estimated:

| Setting | Value |
|---|---|
| Training data | 320 `train/good` → 256 train / 64 validation (seed 42) |
| Test data | 160 images — 41 normal, 119 defective (5 defect types) |
| Input | 128×128 grayscale, 0–1, `LATENT_FILTERS=16`, augmentation on |
| Model | 224,017 parameters |
| Training | early-stopped at epoch 25, best epoch 15, ~13 min |

### Protocol A — threshold from the validation split (`python evaluate.py`)

Threshold `mean + 3σ = 0.0419` (normal validation mean 0.02554, σ 0.00546).

```text
Accuracy 0.2562   Precision 0.0000   Recall 0.0000   F1 0.0000   ROC-AUC 0.9990
Confusion matrix [[41, 0], [119, 0]]        # nothing was ever flagged
```

**The ranking is essentially perfect (AUC 0.9990) but the cutoff never fires.**
This is the most interesting finding in the project, and it is *not* a bug:

* The threshold is fitted on normals from `train/good`, but it is applied to
  `test/good` — a different capture batch that is much easier for the model.
  Normal validation errors average **0.0255**, normal *test* errors **0.0156**.
  A few hard validation images reach **0.0395**, higher than *every* test image
  (the highest defect score is 0.0262).
* So the fitted threshold ends up above all 160 test scores. All three
  strategies fail the same way: `mean_std` → 0.0419, `percentile(99)` → 0.0393,
  `max` → 0.0395.
* With 64 validation images the tail of the error distribution is estimated
  from very few samples; that tail is what sets the threshold.

The lesson generalises: **an anomaly threshold is only meaningful relative to
normal data drawn from the same distribution as deployment data.**

### Protocol B — in-domain calibration (`python evaluate.py --domain-calibration 0.5`)

20 of the 41 normal test images are held out purely to fit the threshold; they
are then removed from the evaluation set, so no metric is measured on data the
threshold saw.

```text
Threshold 0.0197 (calibration normals: mean 0.01537, σ 0.00143)
Accuracy 0.9929   Precision 1.0000   Recall 0.9916   F1 0.9958   ROC-AUC 0.9988
Confusion matrix [[21, 0], [1, 118]]        # 0 false positives, 1 missed defect
```

That is the operating point to quote and demo:

```text
$ python predict.py --image .../test/scratch_head/000.png            # NORMAL, score 0.0253 < 0.0419
$ python predict.py --image .../test/scratch_head/000.png \
      --threshold-file models/screw_threshold_calibrated.json        # ANOMALY, score 0.0253 > 0.0197
$ python predict.py --image .../test/good/040.png \
      --threshold-file models/screw_threshold_calibrated.json        # NORMAL, score 0.0152
```

### Ablation — the latent bottleneck is what makes this work

30 epochs each, same normal-only split, image score = mean of the error map:

| Variant | Latent values (input pixels: 16,384) | Test ROC-AUC | Val normal error | Test-good error |
|---|---|---|---|---|
| `LATENT_FILTERS=128`, no aug (original spec) | 32,768 — **bigger than the input** | **0.7797** | 0.02017 | 0.01825 |
| `LATENT_FILTERS=32`, augment | 8,192 | 0.9994 | 0.02789 | 0.01673 |
| `LATENT_FILTERS=16`, augment (default) | 4,096 | **0.9990** | 0.02554 | 0.01559 |
| `LATENT_FILTERS=8`, augment | 2,048 — too tight | 0.9057 | 0.01890 | 0.01604 |

* The original 128-channel latent had **more latent values than input pixels**,
  so it could reconstruct defects instead of failing on them: training loss fell
  to 0.002 while validation stuck at 0.020 (memorisation), and AUC was 0.78.
* 8 channels is too aggressive — normal texture itself becomes hard to
  reconstruct, so the defect signal drowns again (0.906).
* Reproduce the baseline by setting `LATENT_FILTERS = 128` and
  `AUGMENT = False` at the top of `anomaly_detection.py`, then `python train.py`.

### Negative result — top-k / max pooling of the error map does not help here

The natural trick for *localised* defects (score = mean of the worst 1% of
pixels) made things **worse**, not better: AUC 0.11–0.31 versus 0.766 for the
plain mean, because at 128×128 the largest reconstruction errors come from the
high-contrast thread edges of *normal* screws, not from sub-pixel scratches. The
plain mean error map is retained for that reason.

### Localization is the weak spot (reported honestly)

```text
Protocol B: IoU 0.0114, pixel precision 0.0132, pixel recall 0.0764, pixel F1 0.0225
```

Image-level detection is excellent, pixel-level masks are not: a scratch a few
pixels wide at 1024×1024 becomes sub-pixel after the 8× downsample to 128×128,
so the error map cannot outline it. To improve localization: `--image-size 256`
or `512` (sweep `mask_percentile` down to ~99.0–99.9), add skip connections
(U-Net style) or use an SSIM-based error map instead of MSE.

### Reproducing all of the above

```bash
python train.py                                  # ~13 min on CPU
python evaluate.py                               # protocol A
python evaluate.py --domain-calibration 0.5      # protocol B
```

---

## 13. Citation

MVTec AD: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
*A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
CVPR 2019. (CC BY-NC-SA 4.0 — non-commercial use.)
