import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const htmlUrl = new URL("../public/history.html", import.meta.url);
const scriptUrl = new URL("../public/history.js", import.meta.url);
const html = existsSync(htmlUrl) ? readFileSync(htmlUrl, "utf8") : "";
const script = existsSync(scriptUrl) ? readFileSync(scriptUrl, "utf8") : "";
const css = readFileSync(new URL("../public/styles.css", import.meta.url), "utf8");
const server = readFileSync(new URL("../server.mjs", import.meta.url), "utf8");

function ruleBody(selector, source = css) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = source.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`, "m"));
  assert.ok(match, `Expected CSS rule for ${selector}`);
  return match[1];
}

test("frontend server routes history paths to the history page", () => {
  assert.match(server, /req\.url === "\/history"/, "server should route /history");
  assert.match(server, /serveStatic\("\/history\.html"/, "server should serve history.html");
});

test("history page exposes navigation, filters, table, and detail modal", () => {
  assert.ok(existsSync(htmlUrl), "history.html should exist");
  assert.match(html, /id="sidebar-nav-root"/, "history should mount the shared sidebar nav");
  assert.match(html, /id="history-filter-form"/, "history should expose a filter form");
  assert.match(html, /id="history-search"/, "history should expose free text search");
  for (const id of [
    "history-technique-filter",
    "history-family-filter",
    "history-surface-filter",
    "history-status-filter",
    "history-severity-filter",
    "history-success-filter",
    "history-mode-filter",
    "history-target-model-filter",
    "history-completed-after",
    "history-completed-before",
    "history-min-score",
    "history-max-score",
    "history-min-asr",
    "history-max-asr",
    "history-sort",
    "history-limit",
    "history-table-body",
    "history-detail-modal",
    "history-detail-body",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`), `history should include ${id}`);
  }
});

test("history filters keep primary controls visible and advanced filters collapsed", () => {
  assert.match(html, /id="history-primary-filters"/, "history should group primary filters");
  assert.match(html, /id="history-advanced-filters"[^>]*hidden/, "advanced filters should start collapsed");
  assert.match(html, /id="history-advanced-toggle"[^>]*aria-expanded="false"/, "advanced toggle should expose collapsed state");

  const primaryStart = html.indexOf('id="history-primary-filters"');
  const advancedStart = html.indexOf('id="history-advanced-filters"');
  assert.notEqual(primaryStart, -1, "primary filters should exist");
  assert.notEqual(advancedStart, -1, "advanced filters should exist");
  const primary = html.slice(primaryStart, advancedStart);
  const advanced = html.slice(advancedStart);

  for (const id of [
    "history-search",
    "history-completed-after",
    "history-completed-before",
    "history-success-filter",
    "history-severity-filter",
  ]) {
    assert.match(primary, new RegExp(`id="${id}"`), `${id} should stay visible while collapsed`);
  }
  assert.doesNotMatch(primary, /id="history-sort"/, "sort should live with pagination controls");
  assert.doesNotMatch(primary, /id="history-limit"/, "row count should live with pagination controls");

  for (const id of [
    "history-technique-filter",
    "history-family-filter",
    "history-surface-filter",
    "history-status-filter",
    "history-mode-filter",
    "history-target-model-filter",
    "history-min-score",
    "history-max-score",
    "history-min-asr",
    "history-max-asr",
  ]) {
    assert.match(advanced, new RegExp(`id="${id}"`), `${id} should move into advanced filters`);
  }
});

test("history pagination owns sort and row controls near page numbers", () => {
  const paginationStart = html.indexOf('class="history-pagination"');
  assert.notEqual(paginationStart, -1, "pagination controls should exist");
  const pagination = html.slice(paginationStart, html.indexOf("</section>", paginationStart));

  for (const id of ["history-sort", "history-limit", "history-prev-btn", "history-page-label", "history-next-btn"]) {
    assert.match(pagination, new RegExp(`id="${id}"`), `${id} should be in the pagination footer`);
  }
});

test("history table shows model columns instead of score and severity columns", () => {
  const headerStart = html.indexOf("<thead>");
  assert.notEqual(headerStart, -1, "history table header should exist");
  const header = html.slice(headerStart, html.indexOf("</thead>", headerStart));

  for (const label of ["Target", "Attacker", "Judge"]) {
    assert.match(header, new RegExp(`<th>${label}</th>`), `${label} should be visible in the table`);
  }
  assert.doesNotMatch(header, /<th>Score<\/th>/, "score should not be a table column");
  assert.doesNotMatch(header, /<th>Severity<\/th>/, "severity should not be a table column");

  assert.match(script, /item\.attacker_model/, "history rows should render attacker model values");
  assert.match(script, /item\.judge_model/, "history rows should render judge model values");
});

test("history table pins column widths so long content truncates instead of stretching", () => {
  const tableStart = html.indexOf('<table class="history-table">');
  assert.notEqual(tableStart, -1, "history table should exist");
  const tableMarkup = html.slice(tableStart, html.indexOf("</table>", tableStart));

  assert.match(tableMarkup, /<colgroup>/, "history table should define explicit columns");
  for (const className of [
    "history-col-completed",
    "history-col-technique",
    "history-col-model",
    "history-col-outcome",
    "history-col-intent",
    "history-col-action",
  ]) {
    assert.match(tableMarkup, new RegExp(`class="${className}"`), `${className} should be assigned in colgroup`);
  }

  const table = ruleBody(".history-table");
  const tableCells = ruleBody(".history-table th,\n.history-table td");
  const modelColumns = ruleBody(".history-table col.history-col-model");
  const actionColumn = ruleBody(".history-table col.history-col-action");
  assert.match(table, /table-layout:\s*fixed/, "history table should use fixed layout");
  assert.match(table, /min-width:\s*1150px/, "history table should keep a stable scroll width");
  assert.match(tableCells, /overflow:\s*hidden/, "history cells should hide overflow");
  assert.match(tableCells, /text-overflow:\s*ellipsis/, "history cells should ellipsize overflow");
  assert.match(tableCells, /white-space:\s*nowrap/, "history cells should stay single-line");
  assert.match(modelColumns, /width:\s*150px/, "model columns should have a fixed width");
  assert.match(actionColumn, /width:\s*96px/, "action column should fit the Open button");
});

test("history action cells do not render table ellipsis beside the Open button", () => {
  const actionCell = ruleBody(".history-table td.history-action-cell");
  assert.match(script, /cell\.className = "history-action-cell"/, "action cells should be identifiable");
  assert.match(actionCell, /overflow:\s*visible/, "action cells should not clip the Open button");
  assert.match(actionCell, /text-overflow:\s*clip/, "action cells should not draw an ellipsis");
  assert.match(actionCell, /text-align:\s*right/, "action button should align to the edge");
});

test("history script fetches run history with URLSearchParams and renders without innerHTML", () => {
  assert.ok(existsSync(scriptUrl), "history.js should exist");
  assert.match(script, /HISTORY_NAV_ITEMS/, "history script should define nav items");
  assert.match(script, /label:\s*"History"/, "history nav should include History");
  assert.match(script, /new URLSearchParams\(\)/, "history script should build query strings safely");
  assert.match(script, /fetchJson\(`\/api\/run-history\?\$\{params\.toString\(\)\}`\)/, "history should call list API");
  assert.match(script, /fetchJson\(`\/api\/run-history\/\$\{encodeURIComponent\(runId\)\}`\)/, "history should call detail API");
  assert.match(script, /function renderHistoryTable\(\)/, "history should render the results table");
  assert.match(script, /function renderHistoryFacets\(\)/, "history should render facet controls");
  assert.match(script, /function toggleAdvancedFilters\(\)/, "history should toggle advanced filters");
  assert.match(script, /function updateAdvancedToggleState\(\)/, "history should summarize advanced state");
  assert.match(script, /function openRunDetail\(runId\)/, "history should open run details");
  assert.doesNotMatch(script, /\.innerHTML\b/, "history rendering should avoid direct HTML injection");
});

test("history page styles dense filter and table layouts", () => {
  const filterGrid = ruleBody(".history-filter-grid");
  const primaryFilters = ruleBody(".history-primary-filters");
  const effectivePrimaryFilters = ruleBody(".history-filter-grid.history-primary-filters");
  const table = ruleBody(".history-table");
  const detailGrid = ruleBody(".history-detail-grid");
  assert.match(filterGrid, /grid-template-columns:\s*repeat\(auto-fit,\s*minmax\(180px,\s*1fr\)\)/);
  assert.match(primaryFilters, /grid-template-columns:\s*minmax\(420px,\s*3fr\)\s+repeat\(4,\s*minmax\(112px,\s*0\.75fr\)\)/);
  assert.match(effectivePrimaryFilters, /grid-template-columns:\s*minmax\(420px,\s*3fr\)\s+repeat\(4,\s*minmax\(112px,\s*0\.75fr\)\)/);
  assert.match(table, /border-collapse:\s*collapse/);
  assert.match(detailGrid, /grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(css, /@media\s+\(max-width:\s*900px\)[\s\S]*\.history-detail-grid/);
});
