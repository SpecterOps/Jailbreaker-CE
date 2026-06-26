import { domElement } from "./components/dom.js";
import { mountSidebarNav } from "./components/sidebar-nav.js";
import { mountVaultControl } from "./components/vault-control.js";

const API_BASE_URL = "";

const state = {
  items: [],
  facets: {},
  total: 0,
  offset: 0,
  limit: 25,
};

const HISTORY_NAV_ITEMS = [
  { label: "Dashboard", href: "/" },
  { label: "Experiments", href: "/experiments/" },
  { label: "History", href: "/history/", active: true, ariaCurrent: "page" },
  { label: "Analysis", href: "/analysis/" },
];

const el = {
  sidebarNavRoot: document.querySelector("#sidebar-nav-root"),
  sidebarVaultRoot: document.querySelector("#sidebar-vault-root"),
  status: document.querySelector("#history-status"),
  form: document.querySelector("#history-filter-form"),
  advancedToggle: document.querySelector("#history-advanced-toggle"),
  advancedFilters: document.querySelector("#history-advanced-filters"),
  search: document.querySelector("#history-search"),
  technique: document.querySelector("#history-technique-filter"),
  family: document.querySelector("#history-family-filter"),
  surface: document.querySelector("#history-surface-filter"),
  statusFilter: document.querySelector("#history-status-filter"),
  severity: document.querySelector("#history-severity-filter"),
  success: document.querySelector("#history-success-filter"),
  mode: document.querySelector("#history-mode-filter"),
  targetModel: document.querySelector("#history-target-model-filter"),
  completedAfter: document.querySelector("#history-completed-after"),
  completedBefore: document.querySelector("#history-completed-before"),
  minScore: document.querySelector("#history-min-score"),
  maxScore: document.querySelector("#history-max-score"),
  minAsr: document.querySelector("#history-min-asr"),
  maxAsr: document.querySelector("#history-max-asr"),
  sort: document.querySelector("#history-sort"),
  limit: document.querySelector("#history-limit"),
  applyBtn: document.querySelector("#history-apply-btn"),
  resetBtn: document.querySelector("#history-reset-btn"),
  summaryGrid: document.querySelector("#history-summary-grid"),
  countLabel: document.querySelector("#history-count-label"),
  tableBody: document.querySelector("#history-table-body"),
  prevBtn: document.querySelector("#history-prev-btn"),
  nextBtn: document.querySelector("#history-next-btn"),
  pageLabel: document.querySelector("#history-page-label"),
  detailModal: document.querySelector("#history-detail-modal"),
  detailTitle: document.querySelector("#history-detail-title"),
  detailBody: document.querySelector("#history-detail-body"),
  detailClose: document.querySelector("#history-detail-close"),
};

const advancedFilterControls = [
  el.technique,
  el.family,
  el.surface,
  el.statusFilter,
  el.mode,
  el.targetModel,
  el.minScore,
  el.maxScore,
  el.minAsr,
  el.maxAsr,
];

async function fetchJson(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ error: "Request failed" }));
    throw new Error(payload.error || "Request failed");
  }
  return response.json();
}

mountSidebarNav(el.sidebarNavRoot, {
  items: HISTORY_NAV_ITEMS,
  activePath: "/history/",
});

mountVaultControl(el.sidebarVaultRoot, {
  fetchJson,
  onChange: () => {},
});

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  state.offset = 0;
  updateAdvancedToggleState();
  loadHistory().catch((err) => setStatus(err.message));
});

el.advancedToggle.addEventListener("click", toggleAdvancedFilters);

el.sort.addEventListener("change", () => {
  state.offset = 0;
  loadHistory().catch((err) => setStatus(err.message));
});

el.limit.addEventListener("change", () => {
  state.offset = 0;
  loadHistory().catch((err) => setStatus(err.message));
});

el.resetBtn.addEventListener("click", () => {
  el.form.reset();
  el.sort.value = "completed_desc";
  el.limit.value = "25";
  state.offset = 0;
  el.advancedFilters.hidden = true;
  el.advancedToggle.setAttribute("aria-expanded", "false");
  updateAdvancedToggleState();
  loadHistory().catch((err) => setStatus(err.message));
});

for (const control of advancedFilterControls) {
  control.addEventListener("input", updateAdvancedToggleState);
  control.addEventListener("change", updateAdvancedToggleState);
}

el.prevBtn.addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadHistory().catch((err) => setStatus(err.message));
});

el.nextBtn.addEventListener("click", () => {
  if (state.offset + state.limit >= state.total) return;
  state.offset += state.limit;
  loadHistory().catch((err) => setStatus(err.message));
});

el.detailClose.addEventListener("click", closeRunDetail);
el.detailModal.addEventListener("click", (event) => {
  if (event.target === el.detailModal) closeRunDetail();
});

async function loadHistory() {
  setStatus("Loading history...");
  const params = buildHistoryQuery();
  const payload = await fetchJson(`/api/run-history?${params.toString()}`);
  state.items = payload.items || [];
  state.facets = payload.facets || {};
  state.total = Number(payload.total) || 0;
  state.limit = Number(payload.limit) || state.limit;
  state.offset = Number(payload.offset) || 0;
  renderHistoryFacets();
  updateAdvancedToggleState();
  renderHistorySummary();
  renderHistoryTable();
  renderPagination();
  setStatus(`${state.total} one-off runs`);
}

function toggleAdvancedFilters() {
  const expanded = el.advancedToggle.getAttribute("aria-expanded") === "true";
  el.advancedFilters.hidden = expanded;
  el.advancedToggle.setAttribute("aria-expanded", expanded ? "false" : "true");
  updateAdvancedToggleState();
}

function updateAdvancedToggleState() {
  const activeCount = advancedFilterControls.filter((control) => String(control.value || "").trim()).length;
  el.advancedToggle.textContent = activeCount > 0 ? `Advanced (${activeCount})` : "Advanced";
  el.advancedToggle.classList.toggle("active", activeCount > 0);
}

function buildHistoryQuery() {
  const params = new URLSearchParams();
  appendParam(params, "q", el.search.value);
  appendParam(params, "technique_key", el.technique.value);
  appendParam(params, "technique_family", el.family.value);
  appendParam(params, "technique_surface", el.surface.value);
  appendParam(params, "status", el.statusFilter.value);
  appendParam(params, "highest_severity", el.severity.value);
  appendParam(params, "success", el.success.value);
  appendParam(params, "mode", el.mode.value);
  appendParam(params, "target_model", el.targetModel.value);
  appendParam(params, "completed_after", el.completedAfter.value);
  appendParam(params, "completed_before", el.completedBefore.value);
  appendParam(params, "min_score", el.minScore.value);
  appendParam(params, "max_score", el.maxScore.value);
  appendParam(params, "min_asr", el.minAsr.value);
  appendParam(params, "max_asr", el.maxAsr.value);
  appendParam(params, "sort", el.sort.value);
  appendParam(params, "limit", el.limit.value);
  appendParam(params, "offset", String(state.offset));
  return params;
}

function appendParam(params, key, value) {
  const normalized = String(value || "").trim();
  if (normalized) params.set(key, normalized);
}

function renderHistoryFacets() {
  fillSelect(el.technique, "All techniques", state.facets.techniques || []);
  fillSelect(el.family, "All families", state.facets.families || []);
  fillSelect(el.surface, "All surfaces", state.facets.surfaces || []);
  fillSelect(el.statusFilter, "Any status", state.facets.statuses || []);
  fillSelect(el.severity, "Any severity", state.facets.severities || []);
  fillSelect(el.mode, "Any mode", state.facets.modes || []);
  fillSelect(el.targetModel, "All models", state.facets.target_models || []);
}

function fillSelect(select, emptyLabel, options) {
  const previous = select.value;
  const nodes = [optionElement("", emptyLabel)];
  for (const option of options) {
    nodes.push(optionElement(option.key, `${option.label || option.key} (${option.count})`));
  }
  select.replaceChildren(...nodes);
  if ([...select.options].some((option) => option.value === previous)) {
    select.value = previous;
  }
}

function optionElement(value, label) {
  const option = document.createElement("option");
  option.value = String(value || "");
  option.textContent = String(label || value || "");
  return option;
}

function renderHistorySummary() {
  const successes = state.items.filter((item) => item.success).length;
  const avgScore = average(state.items.map((item) => Number(item.score) || 0));
  const highSeverity = state.items.filter((item) => severityRank(item.highest_severity) >= severityRank("high")).length;
  const models = new Set(state.items.map((item) => item.target_model).filter(Boolean));
  const cards = [
    { label: "Matching Runs", value: state.total, meta: `${state.items.length} visible` },
    { label: "Visible Successes", value: successes, meta: `${state.items.length - successes} not successful` },
    { label: "Average Score", value: avgScore.toFixed(1), meta: "visible page" },
    { label: "High Severity", value: highSeverity, meta: `${models.size} target models` },
  ];
  el.summaryGrid.replaceChildren(...cards.map(renderSummaryCard));
}

function renderSummaryCard(card) {
  const article = domElement("article", "history-summary-card");
  article.append(
    domElement("span", "history-summary-label", card.label),
    domElement("strong", "history-summary-value", card.value),
    domElement("span", "history-summary-meta", card.meta),
  );
  return article;
}

function renderHistoryTable() {
  if (!state.items.length) {
    const row = document.createElement("tr");
    const cell = domElement("td", "analysis-empty-cell", "No one-off runs match the current filters.");
    cell.colSpan = 8;
    row.append(cell);
    el.tableBody.replaceChildren(row);
    el.countLabel.textContent = "0 runs";
    return;
  }

  el.tableBody.replaceChildren(...state.items.map(renderHistoryRow));
  el.countLabel.textContent = `${state.total} runs`;
}

function renderHistoryRow(item) {
  const row = document.createElement("tr");
  row.append(
    tableCell(formatDateTime(item.completed_at)),
    tableCell(techniqueLabel(item)),
    tableCell(modelLabel(item.target_model, item.target_kind)),
    tableCell(modelLabel(item.attacker_model)),
    tableCell(modelLabel(item.judge_model)),
    tableCell(outcomeLabel(item)),
    tableCell(trim(item.unsafe_intent || "", 72)),
    actionCell(item.run_id),
  );
  return row;
}

function tableCell(value) {
  const cell = document.createElement("td");
  cell.textContent = String(value ?? "");
  return cell;
}

function severityCell(severity) {
  const cell = document.createElement("td");
  const pill = domElement("span", `tag tag-${severityClass(severity)}`, severity || "info");
  cell.append(pill);
  return cell;
}

function actionCell(runId) {
  const cell = document.createElement("td");
  cell.className = "history-action-cell";
  const button = document.createElement("button");
  button.className = "btn btn-secondary btn-sm";
  button.type = "button";
  button.textContent = "Open";
  button.addEventListener("click", () => openRunDetail(runId));
  cell.append(button);
  return cell;
}

async function openRunDetail(runId) {
  el.detailTitle.textContent = runId;
  el.detailBody.replaceChildren(domElement("p", "text-muted", "Loading run detail..."));
  el.detailModal.style.display = "flex";
  try {
    const { item } = await fetchJson(`/api/run-history/${encodeURIComponent(runId)}`);
    renderRunDetail(item);
  } catch (err) {
    el.detailBody.replaceChildren(domElement("p", "text-muted", err.message));
  }
}

function closeRunDetail() {
  el.detailModal.style.display = "none";
  el.detailBody.replaceChildren();
}

function renderRunDetail(item) {
  el.detailTitle.textContent = `${item.run_id} · ${item.technique_key}`;
  const meta = domElement("section", "history-detail-grid");
  for (const [label, value] of [
    ["Completed", formatDateTime(item.completed_at)],
    ["Technique", techniqueLabel(item)],
    ["Family", item.technique_family],
    ["Surface", item.technique_surface],
    ["Mode", item.mode],
    ["Target", item.target_model || item.target_kind],
    ["Attacker", item.attacker_model],
    ["Judge", item.judge_model],
    ["Score", `${item.score} / ASR ${item.asr}%`],
    ["Severity", item.highest_severity],
    ["Duration", `${Number(item.duration_seconds || 0).toFixed(1)}s`],
    ["Outcome", outcomeLabel(item)],
  ]) {
    meta.append(detailMeta(label, value));
  }

  el.detailBody.replaceChildren(
    meta,
    detailTextBlock("Intent", item.unsafe_intent),
    detailTextBlock("Final Prompt", item.final_prompt),
    detailTextBlock("Final Output", item.final_output),
    detailListBlock("Detections", renderDetections(item.detections || [])),
    detailListBlock("Conversation", renderConversation(item.conversation || [])),
    detailListBlock("Traces", renderTraces(item.traces || [])),
  );
}

function detailMeta(label, value) {
  const node = domElement("div", "history-detail-meta");
  node.append(domElement("span", "", label), domElement("strong", "", value || ""));
  return node;
}

function detailTextBlock(title, value) {
  const section = domElement("section", "history-detail-section");
  section.append(domElement("h3", "", title), domElement("pre", "history-detail-pre", value || ""));
  return section;
}

function detailListBlock(title, children) {
  const section = domElement("section", "history-detail-section");
  section.append(domElement("h3", "", title));
  if (!children.length) {
    section.append(domElement("p", "text-muted", "None recorded."));
  } else {
    section.append(...children);
  }
  return section;
}

function renderDetections(detections) {
  return detections.map((det) => {
    const node = domElement("div", "history-evidence-row");
    node.append(
      domElement("strong", "", `${det.detection_type || "detection"} · ${det.severity || "info"}`),
      domElement("span", "", det.matched ? "matched" : "not matched"),
      domElement("p", "", det.rationale || ""),
      domElement("pre", "history-detail-pre", JSON.stringify(det.evidence || [], null, 2)),
    );
    return node;
  });
}

function renderConversation(conversation) {
  return conversation.map((entry, index) => {
    const node = domElement("div", "history-evidence-row");
    node.append(
      domElement("strong", "", `${index + 1}. ${entry.kind || "entry"}`),
      domElement("pre", "history-detail-pre", JSON.stringify(entry, null, 2)),
    );
    return node;
  });
}

function renderTraces(traces) {
  return traces.map((trace) => {
    const node = domElement("div", "history-evidence-row");
    node.append(
      domElement("strong", "", `${trace.event_type || "trace"} · ${trace.actor || "system"}`),
      domElement("span", "", formatDateTime(trace.event_timestamp)),
      domElement("p", "", trace.message || ""),
    );
    return node;
  });
}

function renderPagination() {
  const page = Math.floor(state.offset / state.limit) + 1;
  const pages = Math.max(1, Math.ceil(state.total / state.limit));
  el.pageLabel.textContent = `Page ${page} of ${pages}`;
  el.prevBtn.disabled = state.offset <= 0;
  el.nextBtn.disabled = state.offset + state.limit >= state.total;
}

function techniqueLabel(item) {
  return item.technique_name || item.technique_key || "";
}

function outcomeLabel(item) {
  return item.success ? "Successful" : "Not successful";
}

function modelLabel(...values) {
  const value = values.find((entry) => String(entry || "").trim());
  return value || "-";
}

function severityClass(severity) {
  const normalized = String(severity || "info").toLowerCase();
  if (normalized === "critical" || normalized === "high") return "danger";
  if (normalized === "medium") return "warning";
  if (normalized === "low") return "success";
  return "info";
}

function severityRank(severity) {
  return { critical: 5, high: 4, medium: 3, low: 2, info: 1 }[String(severity || "info").toLowerCase()] || 1;
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function trim(value, max) {
  const text = String(value || "");
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function setStatus(message) {
  el.status.textContent = message;
}

loadHistory().catch((err) => setStatus(err.message));
