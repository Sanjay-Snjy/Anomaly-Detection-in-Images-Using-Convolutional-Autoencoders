

const $ = (id) => document.getElementById(id);

const state = {
  lastInput: null,        // { kind: "sample" | "file", id?, file? }
  busy: false,
  health: null,
};

/* ------------------------------------------------------------------ utils */

const fmt = (value, digits = 4) =>
  value === null || value === undefined ? "—" : Number(value).toFixed(digits);

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ----------------------------------------------------------- health panel */

async function loadHealth() {
  const pill = $("status-pill");
  try {
    const health = await api("/api/health");
    state.health = health;
    pill.textContent = "model ready";
    pill.className = "pill ok";
    $("status-detail").textContent =
      `${health.category} · ${health.image_size}×${health.image_size} · ` +
      `${health.model_parameters.toLocaleString()} params · loaded in ` +
      `${health.load_seconds}s`;
  } catch (error) {
    pill.textContent = "model unavailable";
    pill.className = "pill error";
    $("status-detail").textContent = error.message;
  }
}

/* ------------------------------------------------------------- file input */

function setPreview(file) {
  const image = $("preview-img");
  const objectUrl = URL.createObjectURL(file);
  if (image.dataset.objectUrl) URL.revokeObjectURL(image.dataset.objectUrl);
  image.src = objectUrl;
  image.dataset.objectUrl = objectUrl;
  image.hidden = false;
  $("preview-name").textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  $("predict-btn").disabled = false;
}

function clearInput() {
  const image = $("preview-img");
  if (image.dataset.objectUrl) URL.revokeObjectURL(image.dataset.objectUrl);
  image.removeAttribute("src");
  image.dataset.objectUrl = "";
  image.hidden = true;
  $("preview-name").textContent = "nothing selected yet";
  $("file-input").value = "";
  $("predict-btn").disabled = true;
  state.lastInput = null;
}

function wireFileInput() {
  const dropzone = $("dropzone");
  const input = $("file-input");

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (file) setPreview(file);
  });

  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    }));

  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    }));

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setPreview(file);
  });

  const clear = $("clear-btn");
  const run = $("predict-btn");
  if (clear) clear.addEventListener("click", clearInput);
  if (run) run.addEventListener("click", runFilePrediction);
}

/* --------------------------------------------------------------- samples */

async function loadSamples() {
  const container = $("samples");
  if (!container) return;  // the samples panel is not part of this page
  try {
    const data = await api("/api/samples");
    if (!data.samples.length) {
      container.innerHTML = '<span class="muted">No test images found on disk.</span>';
      return;
    }
    container.innerHTML = "";
    for (const sample of data.samples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample";
      button.dataset.sampleId = sample.id;
      button.title = `Predict ${sample.defect_type}/${sample.name}`;
      button.innerHTML =
        `<img src="${sample.thumb_url}" alt="${escapeHtml(sample.defect_type)} sample" />` +
        `<span class="${sample.label === "GOOD" ? "lbl-good" : "lbl-defect"}">${sample.label}</span>` +
        `<span>${escapeHtml(sample.defect_type)}</span>`;
      button.addEventListener("click", () => runSamplePrediction(sample));
      container.appendChild(button);
    }
  } catch (error) {
    container.innerHTML = `<span class="muted">Could not load samples: ${escapeHtml(error.message)}</span>`;
  }
}

/* ------------------------------------------------------------- prediction */

function setBusy(busy, label = "Run detection") {
  state.busy = busy;
  const button = $("predict-btn");
  button.disabled = busy || !state.lastInput;
  button.textContent = busy ? "Running…" : label;
  document.querySelectorAll(".sample").forEach((node) => {
    node.disabled = busy;
  });
}

async function runFilePrediction() {
  const file = $("file-input").files && $("file-input").files[0];
  if (!file) return;
  state.lastInput = { kind: "file", file };
  const body = new FormData();
  body.append("file", file);
  await predict("/api/predict?threshold_mode=calibrated", { method: "POST", body });
}

async function runSamplePrediction(sample) {
  if (!sample || !sample.id) return;
  state.lastInput = { kind: "sample", sample };
  $("preview-img").hidden = true;
  const name = sample.defect_type && sample.name
    ? `${sample.defect_type}/${sample.name}`
    : sample.id;
  const truth = sample.label ? `, ground truth: ${sample.label}` : "";
  $("preview-name").textContent = `${name} (bundled test image${truth})`;
  await predict(`/api/predict-sample/${sample.id}?threshold_mode=calibrated`, { method: "POST" });
}

async function predict(url, options) {
  if (state.busy) return;
  setBusy(true);
  $("result-empty").hidden = true;
  try {
    const result = await api(url, options);
    renderResult(result);
  } catch (error) {
    $("result-body").hidden = true;
    $("result-empty").hidden = false;
    $("result-empty").innerHTML =
      `<strong style="color:var(--bad)">Detection failed:</strong> ${escapeHtml(error.message)}`;
    console.error("predict failed:", error);
  } finally {
    setBusy(false);
  }
}

function renderResult(result) {
  const isAnomaly = result.prediction === "ANOMALY";
  const verdict = $("verdict");
  verdict.textContent = result.prediction;
  verdict.className = `verdict ${isAnomaly ? "anomaly" : "normal"}`;

  const scale = Math.max(result.anomaly_score, result.threshold) * 1.3 || 1;
  $("bar-fill").style.width = `${Math.min(100, (result.anomaly_score / scale) * 100)}%`;
  $("bar-marker").style.left = `${Math.min(100, (result.threshold / scale) * 100)}%`;
  $("bar-scale").textContent = `scale ≈ ${fmt(scale)}`;
  $("score-value").textContent = fmt(result.anomaly_score, 5);
  $("threshold-value").textContent = `threshold ${fmt(result.threshold, 5)}`;

  $("img-original").src = result.images.original;
  $("img-overlay").src = result.images.overlay;
  $("img-error").src = result.images.error_map;

  const rows = [["Threshold", fmt(result.threshold, 5)]];
  $("stats").innerHTML = rows
    .map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`)
    .join("");

  $("explain").innerHTML = explain(result, isAnomaly);
  $("result-body").hidden = false;
}

function explain(result, isAnomaly) {
  const relative = result.threshold ? result.margin / result.threshold : 0;
  const parts = [];

  if (isAnomaly) {
    parts.push(
      `Reconstruction error <strong>${fmt(result.anomaly_score, 5)}</strong> is ` +
      `${(relative * 100).toFixed(0)}% above the threshold ` +
      `${fmt(result.threshold, 5)} — the autoencoder reproduced this image worse than it ` +
      `reproduced normal screws.`);
  } else if (Math.abs(relative) < 0.15) {
    parts.push(
      `Borderline: score <strong>${fmt(result.anomaly_score, 5)}</strong> sits only ` +
      `${Math.abs(relative * 100).toFixed(0)}% below the threshold ${fmt(result.threshold, 5)}.`);
  } else {
    parts.push(
      `Reconstruction error <strong>${fmt(result.anomaly_score, 5)}</strong> is ` +
      `${Math.abs(relative * 100).toFixed(0)}% below the threshold ${fmt(result.threshold, 5)} — ` +
      `this image looks like the normal screws the model was trained on.`);
  }

  parts.push(
    `<br />Pixel-level masks are this project's weak spot (IoU ≈ 0.012): a scratch a few ` +
    `pixels wide at 1024×1024 is sub-pixel after the 8× downsample to 128×128, so the heatmap ` +
    `shows <em>where</em> error concentrates but does not outline the defect cleanly.`);
  return parts.join(" ");
}

/* ---------------------------------------------------------------- metrics */

const METRIC_ROWS = [
  ["Accuracy", (s) => fmt(s.accuracy)],
  ["Precision", (s) => fmt(s.precision)],
  ["Recall", (s) => fmt(s.recall)],
  ["F1-score", (s) => fmt(s.f1)],
];

async function loadMetrics() {
  const container = $("metrics");
  try {
    const data = await api("/api/metrics");
    const chosen =
      data.protocols.find((item) => item.id === "calibrated") ||
      data.protocols.find((item) => item.id === "last_run") ||
      data.protocols[0];
    if (!chosen) {
      container.innerHTML = '<span class="muted">No metrics_summary*.json found — run evaluate.py.</span>';
      return;
    }

    const rows = METRIC_ROWS.map(
      ([label, render]) => `<tr><td>${label}</td><td>${render(chosen.summary)}</td></tr>`
    ).join("");
    container.innerHTML =
      "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>" +
      rows + "</tbody></table>";
  } catch (error) {
    container.innerHTML = `<span class="muted">Could not load metrics: ${escapeHtml(error.message)}</span>`;
  }
}

async function loadFigures() {
  const container = $("figures");
  if (!container) return;  // the figures panel is not part of this page
  try {
    const figures = (await api("/api/figures")).figures;
    container.innerHTML = figures.map((figure) =>
      `<a href="${figure.url}" target="_blank" rel="noopener">
         <img src="${figure.url}" alt="${escapeHtml(figure.name)}" loading="lazy" />
         <span>${escapeHtml(figure.name)}</span>
       </a>`).join("") || '<span class="muted">No figures found — run evaluate.py.</span>';
  } catch (error) {
    container.innerHTML = `<span class="muted">Could not load figures: ${escapeHtml(error.message)}</span>`;
  }
}

async function loadGallery() {
  const container = $("gallery");
  if (!container) return;  // the gallery panel is not part of this page
  try {
    const items = (await api("/api/heatmaps")).heatmaps;
    const rank = { normal: 0, defect: 1, localization: 2, other: 3 };
    items.sort((a, b) => rank[a.kind] - rank[b.kind] || a.name.localeCompare(b.name));
    container.innerHTML = items.map((item) =>
      `<a href="${item.url}" target="_blank" rel="noopener">
         <img src="${item.url}" alt="${escapeHtml(item.name)}" loading="lazy" />
         <span class="kind ${item.kind}">${item.kind}</span>
         <br /><span>${escapeHtml(item.name.replace(/\.png$/, ""))}</span>
       </a>`).join("") || '<span class="muted">results/heatmaps/ is empty — run evaluate.py.</span>';
  } catch (error) {
    container.innerHTML = `<span class="muted">Could not load gallery: ${escapeHtml(error.message)}</span>`;
  }
}

/* ------------------------------------------------------------------ init */

function wireOverlaySlider() {
  const slider = $("overlay-opacity");
  if (!slider) return;
  slider.addEventListener("input", () => {
    $("img-overlay").style.opacity = String(slider.value / 100);
    $("opacity-value").textContent = `${slider.value}%`;
  });
}

wireFileInput();
wireOverlaySlider();

loadHealth();
loadSamples();
loadMetrics();

