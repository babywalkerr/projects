const comment = document.querySelector("#comment");
const checkBtn = document.querySelector("#checkBtn");
const modelStatus = document.querySelector("#modelStatus");
const llmStatus = document.querySelector("#llmStatus");
const scoreEl = document.querySelector("#score");
const gauge = document.querySelector("#gauge");
const verdictLabel = document.querySelector("#verdictLabel");
const violationEl = document.querySelector("#violation");
const reasonEl = document.querySelector("#reason");
const modelLine = document.querySelector("#modelLine");
const highlightedText = document.querySelector("#highlightedText");
const highlightedContent = document.querySelector("#highlightedContent");
const metricEls = {
  datasetRows: document.querySelector("#datasetRows"),
  modelF1: document.querySelector("#modelF1"),
  chainF1: document.querySelector("#chainF1"),
  hfF1: document.querySelector("#hfF1"),
  vectorizerName: document.querySelector("#vectorizerName"),
  modelName: document.querySelector("#modelName"),
  thresholdValue: document.querySelector("#thresholdValue"),
  rocValue: document.querySelector("#rocValue"),
};

let loadedSamples = null;

function fmt(value, digits = 3) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function compactNumber(value) {
  if (value === undefined || value === null) return "-";
  return new Intl.NumberFormat("ru-RU").format(value);
}

function setPill(el, text, state) {
  el.textContent = text;
  el.className = `status-pill ${state}`;
}

function restartAnimation(el, className) {
  if (!el) return;
  el.classList.remove(className);
  void el.offsetWidth;
  el.classList.add(className);
}

async function loadSystem() {
  const [healthRes, configRes, metricsRes, samplesRes] = await Promise.all([
    fetch("/api/health"),
    fetch("/api/config"),
    fetch("/api/metrics"),
    fetch("/static/samples.json"),
  ]);
  const health = await healthRes.json();
  const config = await configRes.json();
  const metrics = await metricsRes.json();
  loadedSamples = await samplesRes.json();

  setPill(modelStatus, health.model_loaded ? "Layer 2 ready" : "Layer 2 missing", health.model_loaded ? "ok" : "bad");
  setPill(llmStatus, health.llm_ready ? `${health.llm_provider} ready` : `${health.llm_provider}: key missing`, health.llm_ready ? "ok" : "warn");

  const meta = config.model_metadata || {};
  const modelMetrics = meta.metrics || {};
  metricEls.datasetRows.textContent = compactNumber(meta.dataset_rows || metrics.production?.dataset_rows);
  metricEls.modelF1.textContent = fmt(modelMetrics.f1_toxic);
  metricEls.chainF1.textContent = fmt(metrics.chain_no_llm?.metrics?.f1_toxic);
  metricEls.hfF1.textContent = fmt(metrics.chain_hf_sample?.metrics?.f1_toxic);
  metricEls.vectorizerName.textContent = meta.vectorizer_name || "-";
  metricEls.modelName.textContent = meta.model_name || "-";
  metricEls.thresholdValue.textContent = fmt(config.model_threshold, 2);
  metricEls.rocValue.textContent = fmt(modelMetrics.roc_auc, 4);
  modelLine.textContent = meta.model_name ? `${meta.vectorizer_name} + ${meta.model_name}, threshold ${metricEls.thresholdValue.textContent}` : "Модель не загружена";
  renderBars(metrics.chain_no_llm?.stage_counts || {});
}

function renderBars(stageCounts) {
  const root = document.querySelector("#stageBars");
  root.innerHTML = "";
  const labels = {
    layer_1_lexicon: "Слой 1: словарь",
    layer_2_low_risk: "Слой 2: низкий риск",
    layer_2_regression: "Слой 2: блок",
    allow_without_llm: "Спорные без LLM",
    layer_3_llm: "Слой 3: HF",
  };
  const entries = Object.entries(stageCounts);
  const total = entries.reduce((sum, [, value]) => sum + Number(value), 0) || 1;
  for (const [key, value] of entries.sort((a, b) => b[1] - a[1])) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const pct = (Number(value) / total) * 100;
    row.innerHTML = `
      <header><span>${labels[key] || key}</span><strong>${compactNumber(value)}</strong></header>
      <div class="bar-track"><div class="bar-fill" style="--bar-target:${pct.toFixed(2)}%"></div></div>
    `;
    root.appendChild(row);
  }
}

function resetStages() {
  for (const id of ["stage-1", "stage-2", "stage-3"]) {
    const stage = document.querySelector(`#${id}`);
    stage.className = "stage skipped is-idle";
    stage.querySelector("p").textContent = "Не вызывался";
    stage.querySelector("strong").textContent = "-";
  }
}

function actionLabel(action) {
  return {
    pass: "Пройден",
    block: "Блок",
    skipped: "Пропущен",
    allow: "Разрешен",
  }[action] || action;
}

function renderStages(layers) {
  resetStages();
  for (const layer of layers || []) {
    const stage = document.querySelector(`#stage-${layer.layer}`);
    if (!stage) continue;
    stage.className = `stage ${layer.action} is-updated`;
    const confidence = fmt(layer.confidence || 0, 2);
    const details = [];
    if (layer.model) details.push(layer.model);
    if (layer.vectorizer) details.push(layer.vectorizer);
    if (layer.matched?.length) details.push(`совпадений: ${layer.matched.join(", ")}`);
    if (layer.error) details.push(layer.error);
    stage.querySelector("p").textContent = details.join(" · ") || actionLabel(layer.action);
    stage.querySelector("strong").textContent = `${actionLabel(layer.action)} · ${confidence}`;
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderHighlightedText(text, layers) {
  const layer1 = (layers || []).find((l) => l.layer === 1);
  const spans = layer1?.matched_spans || [];

  if (!spans.length) {
    highlightedText.hidden = true;
    highlightedContent.innerHTML = "";
    return;
  }

  // Build HTML with highlighted spans
  let html = "";
  let cursor = 0;
  for (const span of spans) {
    if (span.start > cursor) {
      html += escapeHtml(text.slice(cursor, span.start));
    }
    html += `<span class="profanity-word" title="Слой 1: словарь">${escapeHtml(text.slice(span.start, span.end))}</span>`;
    cursor = span.end;
  }
  if (cursor < text.length) {
    html += escapeHtml(text.slice(cursor));
  }

  highlightedContent.innerHTML = html;
  highlightedText.hidden = false;
  restartAnimation(highlightedText, "is-pulsing");
}

function renderResult(data) {
  const risk = data.blocked ? data.confidence : 1 - data.confidence;
  const percent = Math.round(Math.max(0, Math.min(1, risk)) * 100);
  const state = data.blocked ? "blocked" : risk > 0.35 ? "review" : "allow";
  document.body.dataset.result = state;
  gauge.style.setProperty("--risk", percent);
  gauge.style.setProperty("--gauge-color", data.blocked ? "var(--red)" : risk > 0.35 ? "var(--amber)" : "var(--teal)");
  scoreEl.textContent = risk.toFixed(2);
  verdictLabel.textContent = data.blocked ? "Заблокировать" : "Пропустить";
  violationEl.textContent = data.violation_type || "none";
  reasonEl.textContent = data.reason || "-";
  renderStages(data.layers);
  renderHighlightedText(comment.value, data.layers);
  restartAnimation(gauge, "is-pulsing");
  restartAnimation(document.querySelector(".verdict-copy"), "is-pulsing");
}

async function moderate() {
  document.body.classList.add("is-checking");
  highlightedText.hidden = true;
  checkBtn.disabled = true;
  checkBtn.textContent = "Проверка";
  try {
    const response = await fetch("/api/moderate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: comment.value }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderResult(await response.json());
  } catch (error) {
    verdictLabel.textContent = "Ошибка";
    violationEl.textContent = "api_error";
    reasonEl.textContent = String(error);
    resetStages();
  } finally {
    document.body.classList.remove("is-checking");
    checkBtn.disabled = false;
    checkBtn.textContent = "Проверить";
  }
}

function setSampleText(category, buttonEl) {
  if (!loadedSamples || !loadedSamples[category]) return;
  const list = loadedSamples[category];
  if (!list.length) return;
  
  // Pick random
  let text = list[Math.floor(Math.random() * list.length)];
  
  // Try to pick a different one if possible
  if (list.length > 1 && text === comment.value) {
    let attempts = 0;
    while (text === comment.value && attempts < 10) {
      text = list[Math.floor(Math.random() * list.length)];
      attempts++;
    }
  }

  // Deactivate all buttons in sample-row
  document.querySelectorAll(".sample-row button").forEach((item) => {
    item.classList.remove("is-active");
  });
  
  // Activate current
  buttonEl.classList.add("is-active");
  
  comment.value = text;
  moderate();
}

const btnL1 = document.querySelector("#btn-layer1");
const btnL2 = document.querySelector("#btn-layer2");
const btnL3 = document.querySelector("#btn-layer3");
const btnLong = document.querySelector("#btn-long");

if (btnL1) btnL1.addEventListener("click", (e) => setSampleText("layer1", e.currentTarget));
if (btnL2) btnL2.addEventListener("click", (e) => setSampleText("layer2", e.currentTarget));
if (btnL3) btnL3.addEventListener("click", (e) => setSampleText("layer3", e.currentTarget));
if (btnLong) btnLong.addEventListener("click", (e) => setSampleText("long_text", e.currentTarget));

checkBtn.addEventListener("click", () => {
  // Clear button highlights on manual click
  document.querySelectorAll(".sample-row button").forEach((item) => {
    item.classList.remove("is-active");
  });
  moderate();
});
resetStages();
loadSystem().then(() => {
  // Select a default random Layer 3 example on startup to show it off
  if (btnL3) {
    btnL3.click();
  } else {
    moderate();
  }
}).catch(() => {
  setPill(modelStatus, "API offline", "bad");
  setPill(llmStatus, "API offline", "bad");
});
