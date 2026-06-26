import { mountSidebarNav } from "./components/sidebar-nav.js";
import { mountVaultControl } from "./components/vault-control.js";

const API_BASE_URL = "";

const state = {
  views: [],
  activeView: "run_overview",
  activePanel: "views",
  rows: [],
  columns: [],
};

const ANALYSIS_NAV_ITEMS = [
  { label: "Dashboard", href: "/" },
  { label: "Experiments", href: "/experiments/" },
  { label: "History", href: "/history/" },
  {
    label: "Analysis",
    href: "/analysis/",
    active: true,
    ariaCurrent: "page",
    children: [
      { id: "views", label: "Views", href: "/analysis/#views" },
      { id: "query", label: "Query", href: "/analysis/#query" },
    ],
  },
];

const el = {
  sidebarNavRoot: document.querySelector("#sidebar-nav-root"),
  sidebarVaultRoot: document.querySelector("#sidebar-vault-root"),
  status: document.querySelector("#analysis-status"),
  viewTabs: document.querySelector("#analysis-view-tabs"),
  summaryGrid: document.querySelector("#analysis-summary-grid"),
  tableTitle: document.querySelector("#analysis-table-title"),
  tableHead: document.querySelector("#analysis-table-head"),
  tableBody: document.querySelector("#analysis-table-body"),
  panels: Array.from(document.querySelectorAll("[data-panel]")),
  summaryTitle: document.querySelector("#analysis-summary-title"),
  queryInput: document.querySelector("#analysis-query-input"),
  queryRun: document.querySelector("#analysis-query-run"),
  querySample: document.querySelector("#analysis-query-sample"),
  queryStatus: document.querySelector("#analysis-query-status"),
  queryMessage: document.querySelector("#analysis-query-message"),
  queryHead: document.querySelector("#analysis-query-head"),
  queryBody: document.querySelector("#analysis-query-body"),
};

async function fetchJson(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ error: "Request failed" }));
    throw new Error(payload.error || "Request failed");
  }
  return response.json();
}

mountVaultControl(el.sidebarVaultRoot, {
  fetchJson,
  onChange: () => {},
});

function setPanel(panel) {
  state.activePanel = panel;
  el.panels.forEach((item) => {
    const active = item.dataset.panel === panel;
    item.hidden = !active;
    item.classList.toggle("active", active);
  });
  renderAnalysisSidebarNav();
}

function renderAnalysisSidebarNav() {
  mountSidebarNav(el.sidebarNavRoot, {
    items: ANALYSIS_NAV_ITEMS,
    activePath: "/analysis/",
    activeChild: state.activePanel,
    onNavigate(item, event) {
      if (!item.id) return;
      event.preventDefault();
      setPanel(item.id);
    },
  });
}

el.querySample.addEventListener("click", () => {
  el.queryInput.value = [
    "SELECT technique_key, run_count, success_count, avg_score, avg_asr",
    "FROM analysis_technique_summary",
    "ORDER BY run_count DESC, technique_key ASC",
    "LIMIT 25",
  ].join("\n");
});

el.queryRun.addEventListener("click", () => {
  runQuery().catch((err) => {
    el.queryStatus.textContent = "Query failed";
    el.queryMessage.textContent = err.message;
  });
});

async function loadViews() {
  const { items } = await fetchJson("/api/analysis");
  state.views = items;
  renderViewTabs();
}

async function loadView(viewKey) {
  state.activeView = viewKey;
  setStatus(`Loading ${viewKey}...`);
  renderViewTabs();

  const payload = await fetchJson(`/api/analysis/${viewKey}`);
  state.columns = payload.columns || [];
  state.rows = payload.items || [];

  renderSummary();
  renderTable();
  setStatus(`${payload.table} · ${state.rows.length} rows`);
}

async function runQuery() {
  const sql = el.queryInput.value.trim();
  el.queryStatus.textContent = "Running...";
  el.queryMessage.textContent = "";

  const payload = await fetchJson("/api/analysis/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql }),
  });

  el.queryStatus.textContent = `${payload.row_count || 0} rows`;
  renderQueryTable(payload.columns || [], payload.items || []);
}

function renderViewTabs() {
  const buttons = state.views.map((view) => {
    const button = document.createElement("button");
    button.className = `analysis-tab ${view.key === state.activeView ? "active" : ""}`;
    button.append(
      textElement("span", "analysis-tab-title", prettify(view.key)),
      textElement("span", "analysis-tab-meta", view.table),
    );
    button.addEventListener("click", () => loadView(view.key).catch((err) => setStatus(err.message)));
    return button;
  });
  el.viewTabs.replaceChildren(...buttons);
}

function renderSummary() {
  el.summaryTitle.textContent = `${prettify(state.activeView)} Snapshot`;
  const cards = buildSummaryCards(state.activeView, state.rows);
  el.summaryGrid.replaceChildren(...cards.map(renderCard));
}

function buildSummaryCards(view, rows) {
  if (!rows.length) {
    return [{ title: "No rows", severity: "info", lines: ["This view is empty."] }];
  }

  if (view === "run_overview") {
    const successful = rows.filter((row) => row.success === true).length;
    const avgScore = average(rows.map((row) => Number(row.score) || 0));
    const avgAsr = average(rows.map((row) => Number(row.asr) || 0));
    return [
      { title: "Runs", severity: "info", lines: [`${rows.length} recorded runs`] },
      { title: "Successes", severity: "low", lines: [`${successful} successful`, `${rows.length - successful} failed`] },
      { title: "Average Score", severity: "medium", lines: [`${avgScore.toFixed(1)} points`] },
      { title: "Average ASR", severity: "high", lines: [`${avgAsr.toFixed(1)}%`] },
    ];
  }

  if (view === "model_summary" || view === "technique_summary") {
    const top = [...rows].sort((a, b) => (Number(b.run_count) || 0) - (Number(a.run_count) || 0)).slice(0, 4);
    return top.map((row) => ({
      title: row.target_model || row.technique_key || "Summary",
      severity: "info",
      lines: [
        `${row.run_count} runs`,
        `avg score ${Number(row.avg_score || 0).toFixed(1)}`,
        `avg ASR ${Number(row.avg_asr || 0).toFixed(1)}%`,
      ],
    }));
  }

  if (view === "detection_summary") {
    return rows.slice(0, 4).map((row) => ({
      title: row.detection_type || "Detection",
      severity: row.severity || "info",
      lines: [
        `${row.matched_count} matched`,
        `${row.detection_rows} rows`,
        `rate ${Number(row.match_rate || 0).toFixed(2)}`,
      ],
    }));
  }

  return rows.slice(0, 4).map((row) => ({
    title: row.technique_key || row.run_id || "Detail",
    severity: "info",
    lines: [
      `round ${row.round_number || row.entry_index || "n/a"}`,
      trim(row.target_model || "", 40),
      trim(row.judge_score || "", 24),
    ].filter(Boolean),
  }));
}

function renderCard(card) {
  const article = document.createElement("article");
  article.className = `result-card analysis-card severity-${classToken(card.severity)}`;

  const header = document.createElement("div");
  header.className = "card-header";
  header.append(textElement("span", "technique-name", card.title));
  article.append(header);

  card.lines.forEach((line) => {
    article.append(textElement("div", "analysis-card-line", line));
  });

  return article;
}

function renderTable() {
  el.tableTitle.textContent = `${prettify(state.activeView)} Data`;
  renderTableHead(el.tableHead, state.columns);
  renderTableBody(el.tableBody, state.columns, state.rows, "No rows found for this view.", Math.max(state.columns.length, 1));
}

function renderQueryTable(columns, rows) {
  if (columns.length) {
    renderTableHead(el.queryHead, columns);
  } else {
    el.queryHead.replaceChildren();
  }
  renderTableBody(el.queryBody, columns, rows, "Query returned no rows.", columns.length);
}

function renderTableHead(head, columns) {
  if (!columns.length) {
    head.replaceChildren();
    return;
  }
  head.replaceChildren(tableRow(columns.map(prettify), "th"));
}

function renderTableBody(body, columns, rows, emptyMessage, emptyColspan) {
  if (!rows.length) {
    if (emptyColspan > 0) {
      body.replaceChildren(emptyTableRow(emptyMessage, emptyColspan));
    } else {
      body.replaceChildren();
    }
    return;
  }

  const tableRows = rows.map((row) => tableRow(columns.map((column) => formatValue(row[column])), "td"));
  body.replaceChildren(...tableRows);
}

function tableRow(values, cellTag) {
  const row = document.createElement("tr");
  values.forEach((value) => {
    row.append(textElement(cellTag, "", value));
  });
  return row;
}

function emptyTableRow(message, colspan) {
  const row = document.createElement("tr");
  const cell = textElement("td", "analysis-empty-cell", message);
  cell.colSpan = colspan;
  row.append(cell);
  return row;
}

function textElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = String(text ?? "");
  return node;
}

function classToken(value) {
  return String(value || "info").toLowerCase().replace(/[^a-z0-9_-]/g, "") || "info";
}

function setStatus(message) {
  el.status.textContent = message;
}

function formatValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function prettify(value) {
  return String(value).replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function trim(value, max) {
  const text = String(value || "");
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

setPanel("views");
loadViews()
  .then(() => loadView(state.activeView))
  .catch((err) => setStatus(err.message));
