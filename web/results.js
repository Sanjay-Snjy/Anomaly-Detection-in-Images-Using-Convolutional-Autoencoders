/* Results page — every evaluated category, rendered from /api/results. */

const $ = (id) => document.getElementById(id);

const fmt = (value, digits = 4) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(digits);

const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function basename(path) {
  if (!path) return "—";
  return String(path).split(/[\\/]/).pop();
}

async function api(path) {
  const response = await fetch(path);
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

/* --------------------------------------------------------------- summary */

const SUMMARY_COLUMNS = [
  ["Accuracy", (s) => fmt(s.accuracy)],
  ["Precision", (s) => fmt(s.precision)],
  ["Recall", (s) => fmt(s.recall)],
  ["F1", (s) => fmt(s.f1)],
  ["ROC-AUC", (s) => fmt(s.roc_auc)],
  ["Pixel F1", (s) => fmt(s.pixel_f1)],
  ["Threshold", (s) => fmt(s.threshold, 5)],
];

function renderSummary(categories) {
  const container = $("summary");
  const withMetrics = categories.filter((item) => item.primary);

  if (!withMetrics.length) {
    container.innerHTML =
      '<span class="muted">No metrics_summary*.json found — run ' +
      "<code>python evaluate.py --category screw</code>.</span>";
    return;
  }

  const head =
    "<tr><th>Category</th><th>Test images</th><th>Defects</th>" +
    SUMMARY_COLUMNS.map(([label]) => `<th>${label}</th>`).join("") +
    "</tr>";

  const rows = withMetrics.map((item) => {
    const summary = item.primary.summary;
    const cells = SUMMARY_COLUMNS
      .map(([, render]) => `<td>${render(summary)}</td>`)
      .join("");
    return (
      `<tr>
         <td><a class="cat-link" href="#cat-${escapeHtml(item.category)}">` +
      `${escapeHtml(item.category)}</a></td>
         <td>${item.dataset.test_images}</td>
         <td>${item.dataset.defect_images}</td>
         ${cells}
       </tr>`
    );
  }).join("");

  container.innerHTML = `<div class="table-scroll"><table><thead>${head}</thead>` +
    `<tbody>${rows}</tbody></table></div>`;
}

/* ------------------------------------------------------- category card */

function verdict(summary, defectImages) {
  const recall = Number(summary.recall);
  const f1 = Number(summary.f1);
  const missed = Math.round((1 - recall) * defectImages);

  if (f1 >= 0.95) {
    return { tone: "good",
             text: `Strong operating point — ${pct(summary.precision)} precision, ` +
                   `${pct(recall)} recall (${missed} of ${defectImages} defects missed).` };
  }
  if (recall < 0.7) {
    return { tone: "weak",
             text: `Recall is the limit here — ${missed} of ${defectImages} defect images ` +
                   `are missed even though precision is ${pct(summary.precision)}. ` +
                   `ROC-AUC ${fmt(summary.roc_auc)} shows the score still ranks well, so the ` +
                   `threshold alone is not the whole story.` };
  }
  return { tone: "warn",
           text: `Workable — ${pct(summary.precision)} precision, ${pct(recall)} recall ` +
                 `(${missed} of ${defectImages} defects missed).` };
}

function metricsTable(summary) {
  const rows = SUMMARY_COLUMNS
    .map(([label, render]) => `<tr><td>${label}</td><td>${render(summary)}</td></tr>`)
    .join("");
  return `<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>` +
         `<tbody>${rows}</tbody></table>`;
}

function provenance(item) {
  const summary = item.primary.summary;
  const rows = [
    ["Threshold file", basename(summary.threshold_source)],
    ["Model", `${item.category}_autoencoder.keras`],
    ["Test set", `${item.dataset.test_images} images`],
    ["Defect images", `${item.dataset.defect_images}`],
    ["Normal test images", `${item.dataset.normal_images}`],
    ["Source file", item.primary.file],
  ];

  if (item.secondary) {
    const other = item.secondary.summary;
    rows.push([
      "Validation threshold",
      `${fmt(other.threshold, 5)} → F1 ${fmt(other.f1)}`,
    ]);
  }

  return `<dl class="stats">` + rows
    .map(([key, value]) => `<div><dt>${key}</dt><dd>${escapeHtml(value)}</dd></div>`)
    .join("") + `</dl>`;
}

function figuresFor(item) {
  if (!item.figures.length) {
    return '<span class="muted">No figures saved for this category.</span>';
  }
  return `<div class="figure-grid">` + item.figures.map((figure) =>
    `<a href="${figure.url}" target="_blank" rel="noopener">
       <img src="${figure.url}" alt="${escapeHtml(figure.label)}" loading="lazy" />
       <span>${escapeHtml(figure.label)}</span>
     </a>`).join("") + `</div>`;
}

function galleryFor(item) {
  if (!item.heatmaps.length) {
    return '<p class="muted">No heatmaps saved for this category.</p>';
  }
  const rank = { normal: 0, defect: 1, localization: 2, other: 3 };
  const items = [...item.heatmaps].sort(
    (a, b) => rank[a.kind] - rank[b.kind] || a.name.localeCompare(b.name));

  return `<h3>Example heatmaps <span class="badge">${items.length}</span></h3>
    <div class="gallery">` + items.map((item_) =>
    `<a href="${item_.url}" target="_blank" rel="noopener">
       <img src="${item_.url}" alt="${escapeHtml(item_.name)}" loading="lazy" />
       <span class="kind ${item_.kind}">${item_.kind}</span>
       <span>${escapeHtml(item_.name.replace(/\.png$/, ""))}</span>
     </a>`).join("") + `</div>`;
}

function renderCard(item) {
  const category = escapeHtml(item.category);

  if (!item.primary) {
    return `<section class="card wide" id="cat-${category}">
      <div class="cat-head"><h2>${category}</h2>
        <span class="badge">no results yet</span></div>
      <p class="muted">Not evaluated yet — run
        <code>python evaluate.py --category ${category}</code>.</p>
    </section>`;
  }

  const summary = item.primary.summary;
  const tone = verdict(summary, item.dataset.defect_images);

  return `<section class="card wide" id="cat-${category}">
    <div class="cat-head">
      <h2>${category}</h2>
      <span class="badge">${escapeHtml(item.primary.label)}</span>
      <span class="badge">${item.dataset.test_images} test images</span>
      <span class="badge">${item.dataset.defect_images} defects</span>
    </div>
    <p class="verdict-note ${tone.tone}">${tone.text}</p>

    <div class="cat-body">
      <div class="cat-metrics">
        ${metricsTable(summary)}
        ${provenance(item)}
      </div>
      <div class="cat-figures">${figuresFor(item)}</div>
    </div>

    ${galleryFor(item)}
  </section>`;
}

/* ------------------------------------------------------------------ init */

async function load() {
  const stamp = $("stamp");
  try {
    const data = await api("/api/results");
    const categories = data.categories || [];
    renderSummary(categories);
    $("sections").innerHTML = categories.length
      ? categories.map(renderCard).join("")
      : '<section class="card wide"><span class="muted">results/ is empty — ' +
        "run evaluate.py for a category.</span></section>";
    stamp.textContent = `${categories.length} categor${categories.length === 1 ? "y" : "ies"}`;
  } catch (error) {
    $("summary").innerHTML =
      `<span class="muted">Could not load results: ${escapeHtml(error.message)}</span>`;
    stamp.textContent = "";
  }
}

load();
