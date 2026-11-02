from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from tensorflow import keras

from anomaly_detection import (
    CALIBRATED_THRESHOLD_PATH,
    CATEGORY,
    DATASET_ROOT,
    HEATMAPS_DIR,
    IMAGE_SIZE,
    MODEL_PATH,
    RESULTS_DIR,
    RESULTS_ROOT,
    TEST_DIR,
    THRESHOLD_PATH,
    heatmap_overlay,
    load_model,
    load_threshold,
    predict,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
THRESHOLD_MODES = ("calibrated", "validation")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------


@dataclass
class ServiceState:
    """Everything the handlers need, built once at startup."""

    model: keras.Model
    thresholds: Dict[str, Dict] = field(default_factory=dict)
    default_mode: str = "calibrated"
    load_seconds: float = 0.0
    warmup_seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _threshold_for(state: ServiceState, mode: str) -> Dict:
    """Resolve a threshold mode to its JSON payload, with a helpful 400."""
    mode = (mode or state.default_mode).lower()
    if mode == "auto":
        mode = state.default_mode
    if mode not in THRESHOLD_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown threshold_mode '{mode}'; use one of {THRESHOLD_MODES}.",
        )
    if mode not in state.thresholds:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The '{mode}' threshold file is missing. Run `python train.py` "
                "and/or `python evaluate.py --domain-calibration 0.5` first."
            ),
        )
    return state.thresholds[mode]


def load_service_state(default_mode: str) -> ServiceState:
    """Load model + thresholds and warm the model up so the first call is fast."""
    started = time.perf_counter()
    model = load_model(MODEL_PATH)
    load_seconds = time.perf_counter() - started

    thresholds: Dict[str, Dict] = {}
    for mode, path in (
        ("validation", THRESHOLD_PATH),
        ("calibrated", CALIBRATED_THRESHOLD_PATH),
    ):
        if Path(path).exists():
            thresholds[mode] = load_threshold(path)

    if not thresholds:
        raise RuntimeError(
            "No threshold JSON found in models/. Run `python train.py` first."
        )

    resolved_default = default_mode if default_mode in thresholds else (
        "calibrated" if "calibrated" in thresholds else "validation"
    )

    state = ServiceState(
        model=model,
        thresholds=thresholds,
        default_mode=resolved_default,
        load_seconds=load_seconds,
    )

    # Warm-up: one throwaway forward pass so TF's graph is built and the
    # first real request is not charged for it.
    warm = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    started = time.perf_counter()
    with state.lock:
        model.predict(warm, verbose=0)
    state.warmup_seconds = time.perf_counter() - started
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = load_service_state(default_mode=DEFAULT_MODE)
    app.state.service = state
    print(
        f"[app] model loaded in {state.load_seconds:.1f}s "
        f"(warm-up {state.warmup_seconds:.2f}s) | "
        f"thresholds: {', '.join(sorted(state.thresholds))} "
        f"| default: {state.default_mode}"
    )
    yield


DEFAULT_MODE = "calibrated"
app = FastAPI(
    title="Screw anomaly detection",
    description="Upload a screw image; a CNN autoencoder reconstructs it and "
                "the reconstruction error decides NORMAL vs ANOMALY.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _to_u8(array: np.ndarray) -> np.ndarray:
    """float 0-1 array -> uint8, tolerating (H, W, 1) input."""
    return (np.clip(array.squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8)


def _data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _encode_gray(array: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", _to_u8(array))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return _data_uri(buf.tobytes())


def _encode_rgb(array: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return _data_uri(buf.tobytes())


def _encode_colormap(array: np.ndarray, colormap: int) -> str:
    """Normalise a float map to 0-1 and colourise it for display."""
    scaled = array / max(float(array.max()), 1e-12)
    coloured = cv2.applyColorMap((scaled * 255).astype(np.uint8), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return _encode_rgb(coloured)


def _listed(directory: Path, suffix: str = ".png") -> List[Path]:
    """Deterministic listing used as a serving allow-list (no path traversal)."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == suffix)


def _subdirs(directory: Path) -> List[Path]:
    """Sorted subdirectories (used to enumerate MVTec defect types)."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_dir())


def _resolve(directory: Path, name: str) -> Path:
    """Serve only files that are actually in `directory`."""
    if not SAFE_NAME.match(name):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    for path in _listed(directory):
        if path.name == name:
            return path
    raise HTTPException(status_code=404, detail=f"No such file: {name}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


async def _predict_path(state: ServiceState, path: Path, mode: str) -> Dict:
    """Run the shared pipeline on a path and shape the JSON for the UI."""
    info = _threshold_for(state, mode)
    threshold = float(info["threshold"])
    pixel_threshold = info.get("pixel_threshold")
    pixel_threshold = float(pixel_threshold) if pixel_threshold is not None else None

    def run() -> Dict:
        # `predict` is the same call predict.py uses; a temp file keeps
        # this on the library's tested load_image() path. The lock is taken in
        # the worker thread so the event loop is never blocked.
        with state.lock:
            return predict(
                state.model,
                path,
                threshold,
                pixel_threshold,
            )

    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(run)
    except (ValueError, FileNotFoundError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    error_map = result["error_map"]
    predicted_mask = result.get("predicted_mask")
    payload = {
        "prediction": result["prediction"],
        "anomaly_score": round(float(result["anomaly_score"]), 6),
        "threshold": round(threshold, 6),
        "threshold_mode": mode,
        "threshold_source": str(
            CALIBRATED_THRESHOLD_PATH
            if mode == "calibrated"
            else THRESHOLD_PATH
        ),
        "threshold_strategy": info.get("strategy"),
        "n_reference_images": int(
            info.get("n_calibration_images")
            or info.get("n_validation_images")
            or 0
        ),
        "pixel_threshold": round(pixel_threshold, 6) if pixel_threshold else None,
        "margin": round(float(result["anomaly_score"]) - threshold, 6),
        "error_map_max": round(float(error_map.max()), 6),
        "error_map_p99": round(float(np.percentile(error_map, 99)), 6),
        "flagged_pixels": int(predicted_mask.sum()) if predicted_mask is not None else None,
        "flagged_fraction": (
            round(float(predicted_mask.mean()), 6) if predicted_mask is not None else None
        ),
        "image_size": IMAGE_SIZE,
        "elapsed_ms": round(elapsed_ms, 1),
        "images": {
            "original": _encode_gray(result["original"]),
            "reconstruction": _encode_gray(result["reconstruction"]),
            "error_map": _encode_colormap(error_map, cv2.COLORMAP_HOT),
            "overlay": _encode_rgb(
                heatmap_overlay(result["original"], error_map)
            ),
        },
    }
    return payload


# ---------------------------------------------------------------------------
# Sample + results browsing
# ---------------------------------------------------------------------------


def _evenly_spaced(total: int, wanted: int) -> List[int]:
    """Indices spread across a list, always including first and last."""
    if total <= 0:
        return []
    wanted = max(1, min(wanted, total))
    if wanted == 1:
        return [0]
    step = (total - 1) / (wanted - 1)
    return sorted({int(round(i * step)) for i in range(wanted)})


def _sample_catalogue(per_defect: int = 1, n_good: int = 3) -> List[Dict]:
    """A deterministic handful of bundled test images for one-click demos."""
    test_dir = TEST_DIR
    if not test_dir.is_dir():
        return []

    samples: List[Dict] = []

    def add(defect_type: str, indices: List[int]) -> None:
        files = _listed(test_dir / defect_type)
        for idx in indices:
            if 0 <= idx < len(files):
                path = files[idx]
                samples.append(
                    {
                        "id": f"{defect_type}-{path.stem}",
                        "defect_type": defect_type,
                        "label": "GOOD" if defect_type == "good" else "DEFECT",
                        "name": path.name,
                        "url": f"/api/samples/{defect_type}-{path.stem}",
                        "thumb_url": f"/api/samples/{defect_type}-{path.stem}?thumb=1",
                    }
                )

    add("good", _evenly_spaced(len(_listed(test_dir / "good")), n_good))
    for directory in _subdirs(test_dir):
        if directory.name == "good":
            continue
        add(directory.name, _evenly_spaced(len(_listed(directory)), per_defect))
    return samples


def _sample_path(sample_id: str) -> Path:
    defect_type, separator, stem = sample_id.partition("-")
    if not separator or not SAFE_NAME.match(stem):
        raise HTTPException(status_code=400, detail="Invalid sample id.")
    return _resolve(TEST_DIR / defect_type, f"{stem}.png")


def _thumbnail(path: Path, width: int = 240) -> bytes:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Could not read {path.name}.")
    height = max(1, int(round(raw.shape[0] * width / raw.shape[1])))
    small = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", small)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode thumbnail.")
    return buf.tobytes()


def _figure_listing() -> List[Dict]:
    """Saved evaluation figures (confusion matrix, ROC curve, training curves)."""
    return [
        {"name": path.name, "url": f"/api/figures/{path.name}"}
        for path in _listed(RESULTS_DIR)
        if path.name.startswith(("confusion_matrix", "roc_curve", "training_history"))
    ]


def _metrics_payload() -> Dict:
    """Collect every metrics_summary*.json that exists under results/."""
    mapping = {
        "validation": ("metrics_summary_val_threshold.json",
                       "Protocol A — threshold from the validation split"),
        "calibrated": ("metrics_summary_domain_calibration.json",
                       "Protocol B — in-domain calibrated threshold"),
        "last_run": ("metrics_summary.json", "Last evaluate.py run"),
    }
    protocols = []
    for mode, (filename, label) in mapping.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        protocols.append({"id": mode, "label": label, "summary": summary})

    return {"protocols": protocols, "figures": _figure_listing()}


# ---------------------------------------------------------------------------
# Cross-category results (results/<category>/ written by evaluate.py)
# ---------------------------------------------------------------------------

PRIMARY_RESULT_FILES = (
    ("metrics_summary_domain_calibration.json",
     "In-domain calibrated threshold"),
    ("metrics_summary.json", "Latest evaluate.py run"),
)
SECONDARY_RESULT_FILES = (
    ("metrics_summary_val_threshold.json", "Validation-split threshold"),
)
FIGURE_LABELS = {
    "confusion_matrix.png": "Confusion matrix",
    "roc_curve.png": "ROC curve",
    "training_history.png": "Training history",
}


def _summary_file(path: Path) -> Optional[Dict]:
    """Read one metrics JSON, tolerating a missing or corrupt file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _result_categories() -> List[str]:
    """Category folders under results/, used as a serving allow-list."""
    if not RESULTS_ROOT.is_dir():
        return []
    return sorted(p.name for p in RESULTS_ROOT.iterdir()
                  if p.is_dir() and SAFE_NAME.match(p.name))


def _category_result(category: str) -> Dict:
    """Everything the results page shows for one category."""
    directory = RESULTS_ROOT / category

    primary = None
    for filename, label in PRIMARY_RESULT_FILES:
        summary = _summary_file(directory / filename)
        if summary:
            primary = {"file": filename, "label": label, "summary": summary}
            break

    secondary = None
    for filename, label in SECONDARY_RESULT_FILES:
        summary = _summary_file(directory / filename)
        if summary:
            secondary = {"file": filename, "label": label, "summary": summary}
            break

    test_dir = DATASET_ROOT / category / "test"
    normal_images = len(_listed(test_dir / "good"))
    test_images = sum(len(_listed(folder)) for folder in _subdirs(test_dir))

    figures = [
        {"name": name, "label": label,
         "url": f"/api/results/{category}/figures/{name}"}
        for name, label in FIGURE_LABELS.items()
        if (directory / name).exists()
    ]

    heatmaps = []
    for path in _listed(directory / "heatmaps"):
        name = path.name
        if name.startswith("example_normal"):
            kind = "normal"
        elif name.startswith("example_"):
            kind = "defect"
        elif name.startswith("localization_"):
            kind = "localization"
        else:
            kind = "other"
        heatmaps.append({"name": name, "kind": kind,
                         "url": f"/api/results/{category}/heatmaps/{name}"})

    return {
        "category": category,
        "primary": primary,
        "secondary": secondary,
        "dataset": {
            "test_images": test_images,
            "normal_images": normal_images,
            "defect_images": max(test_images - normal_images, 0),
        },
        "figures": figures,
        "heatmaps": heatmaps,
    }


# ---------------------------------------------------------------------------
# Routes: API
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> Dict:
    state: ServiceState = app.state.service
    return {
        "status": "ok",
        "category": CATEGORY,
        "image_size": IMAGE_SIZE,
        "model_path": str(MODEL_PATH),
        "model_parameters": int(state.model.count_params()),
        "load_seconds": round(state.load_seconds, 2),
        "warmup_seconds": round(state.warmup_seconds, 3),
        "default_threshold_mode": state.default_mode,
        "thresholds": {
            mode: {
                "path": str(
                    CALIBRATED_THRESHOLD_PATH
                    if mode == "calibrated"
                    else THRESHOLD_PATH
                ),
                "threshold": round(float(info["threshold"]), 6),
                "pixel_threshold": (
                    round(float(info["pixel_threshold"]), 6)
                    if info.get("pixel_threshold") is not None
                    else None
                ),
                "strategy": info.get("strategy"),
                "n_reference_images": int(
                    info.get("n_calibration_images")
                    or info.get("n_validation_images")
                    or 0
                ),
            }
            for mode, info in state.thresholds.items()
        },
        "available_threshold_modes": sorted(state.thresholds),
    }


@app.post("/api/predict")
async def predict_upload(
    file: UploadFile = File(...),
    threshold_mode: str = Query(default="calibrated"),
) -> Dict:
    state: ServiceState = app.state.service
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload.")

    suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
    if not SAFE_NAME.match(suffix.lstrip(".")) or len(suffix) > 6:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)

    try:
        result = await _predict_path(state, temp_path, threshold_mode)
    finally:
        temp_path.unlink(missing_ok=True)

    result["filename"] = file.filename
    result["bytes"] = len(payload)
    return result


@app.post("/api/predict-sample/{sample_id}")
async def predict_sample(
    sample_id: str,
    threshold_mode: str = Query(default="calibrated"),
) -> Dict:
    state: ServiceState = app.state.service
    path = _sample_path(sample_id)
    result = await _predict_path(state, path, threshold_mode)
    result["filename"] = path.name
    result["sample_id"] = sample_id
    return result


@app.get("/api/samples")
async def samples(
    per_defect: int = Query(default=1, ge=1, le=5),
    n_good: int = Query(default=3, ge=0, le=10),
) -> Dict:
    state: ServiceState = app.state.service
    catalogue = _sample_catalogue(per_defect=per_defect, n_good=n_good)
    return {"count": len(catalogue), "samples": catalogue}


@app.get("/api/samples/{sample_id}")
async def sample_image(sample_id: str, thumb: int = Query(default=0)) -> Response:
    state: ServiceState = app.state.service
    path = _sample_path(sample_id)
    if thumb:
        return Response(
            content=_thumbnail(path),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/metrics")
async def metrics() -> Dict:
    state: ServiceState = app.state.service
    return _metrics_payload()


@app.get("/api/figures")
async def figures() -> Dict:
    state: ServiceState = app.state.service
    return {"figures": _figure_listing()}


@app.get("/api/figures/{name}")
async def figure(name: str) -> FileResponse:
    state: ServiceState = app.state.service
    return FileResponse(_resolve(RESULTS_DIR, name),
                        media_type="image/png")


@app.get("/api/heatmaps")
async def heatmaps() -> Dict:
    """The heatmaps evaluate.py already wrote, grouped for a gallery."""
    state: ServiceState = app.state.service
    items = []
    for path in _listed(HEATMAPS_DIR):
        if path.name.startswith("example_normal"):
            kind = "normal"
        elif path.name.startswith("example_"):
            kind = "defect"
        elif path.name.startswith("localization_"):
            kind = "localization"
        else:
            kind = "other"
        items.append(
            {
                "name": path.name,
                "kind": kind,
                "url": f"/api/heatmaps/{path.name}",
            }
        )
    return {"count": len(items), "heatmaps": items}


@app.get("/api/heatmaps/{name}")
async def heatmap(name: str) -> FileResponse:
    state: ServiceState = app.state.service
    return FileResponse(_resolve(HEATMAPS_DIR, name),
                        media_type="image/png")


@app.get("/api/results")
async def results() -> Dict:
    """Every evaluated category, straight from results/<category>/."""
    return {"categories": [_category_result(name)
                           for name in _result_categories()]}


@app.get("/api/results/{category}/figures/{name}")
async def result_figure(category: str, name: str) -> FileResponse:
    if category not in _result_categories():
        raise HTTPException(status_code=404, detail=f"No results for {category}.")
    return FileResponse(_resolve(RESULTS_ROOT / category, name),
                        media_type="image/png")


@app.get("/api/results/{category}/heatmaps/{name}")
async def result_heatmap(category: str, name: str) -> FileResponse:
    if category not in _result_categories():
        raise HTTPException(status_code=404, detail=f"No results for {category}.")
    return FileResponse(_resolve(RESULTS_ROOT / category / "heatmaps", name),
                        media_type="image/png")


# ---------------------------------------------------------------------------
# Routes: bundled UI
# ---------------------------------------------------------------------------


NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}


class NoCacheStaticFiles(StaticFiles):
    """Serve the UI so an edited page is never hidden by a stale browser cache."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


@app.get("/")
async def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail=f"Missing {page}.")
    return FileResponse(page, media_type="text/html", headers=NO_CACHE)


@app.get("/results")
async def results_page() -> FileResponse:
    page = WEB_DIR / "results.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail=f"Missing {page}.")
    return FileResponse(page, media_type="text/html", headers=NO_CACHE)


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


if WEB_DIR.is_dir():
    app.mount("/static", NoCacheStaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the screw anomaly detector")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threshold-mode", type=str, default=DEFAULT_MODE,
                        choices=list(THRESHOLD_MODES) + ["auto"],
                        help="Which saved threshold new requests use by default.")
    return parser.parse_args()


def main() -> None:
    global DEFAULT_MODE
    args = parse_args()
    DEFAULT_MODE = args.threshold_mode
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
