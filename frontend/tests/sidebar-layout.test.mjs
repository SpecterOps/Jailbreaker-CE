import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const css = readFileSync(new URL("../public/styles.css", import.meta.url), "utf8");
const dashboardHtml = readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
const analysisHtml = readFileSync(new URL("../public/analysis.html", import.meta.url), "utf8");
const experimentsHtml = readFileSync(new URL("../public/experiments.html", import.meta.url), "utf8");
const analysisScript = readFileSync(new URL("../public/analysis.js", import.meta.url), "utf8");
const experimentsScript = readFileSync(new URL("../public/experiments.js", import.meta.url), "utf8");

function ruleBody(selector, source = css) {
  const pattern = new RegExp(`${selector.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}\\s*\\{([^}]*)\\}`, "m");
  const match = source.match(pattern);
  assert.ok(match, `Expected to find CSS rule for ${selector}`);
  return match[1];
}

function hasDeclaration(body, property, valuePattern) {
  return new RegExp(`${property}\\s*:\\s*${valuePattern}\\s*;`).test(body);
}

function assertTokenOrder(source, tokens, name) {
  let previousIndex = -1;
  for (const token of tokens) {
    const index = source.indexOf(token);
    assert.notEqual(index, -1, `${name} should include ${token}`);
    assert.ok(index > previousIndex, `${name} should order Dashboard > Experiments > Analysis`);
    previousIndex = index;
  }
}

test("desktop sidebar stays pinned while page content scrolls", () => {
  const sidebar = ruleBody(".sidebar");

  assert.ok(
    hasDeclaration(sidebar, "position", "(sticky|fixed)"),
    "sidebar needs sticky or fixed positioning",
  );
  assert.ok(hasDeclaration(sidebar, "top", "0"), "sidebar needs top: 0");
  assert.ok(
    hasDeclaration(sidebar, "height", "100vh"),
    "sidebar needs viewport height",
  );
  assert.ok(
    hasDeclaration(sidebar, "overflow-y", "auto"),
    "sidebar needs its own vertical overflow",
  );
});

test("sidebar brand divider aligns with topbar divider", () => {
  const root = ruleBody(":root");
  const sidebar = ruleBody(".sidebar");
  const sidebarBrand = ruleBody(".sidebar-brand");
  const topbar = ruleBody(".topbar");

  assert.ok(
    hasDeclaration(root, "--shell-header-height", "56px"),
    "shell header height should be a shared token",
  );
  assert.ok(
    hasDeclaration(topbar, "height", "var\\(--shell-header-height\\)"),
    "topbar should use the shared shell header height",
  );
  assert.ok(
    hasDeclaration(sidebarBrand, "height", "var\\(--shell-header-height\\)"),
    "sidebar brand should use the shared shell header height",
  );
  assert.ok(
    hasDeclaration(sidebar, "padding", "0\\s+0\\s+var\\(--sp-5\\)"),
    "sidebar top padding should not offset the brand divider",
  );
});

test("primary sidebar orders experiments between dashboard and analysis", () => {
  assertTokenOrder(dashboardHtml, [">Dashboard</a>", ">Experiments</a>", ">Analysis</a>"], "dashboard sidebar");
  assertTokenOrder(analysisScript, ['label: "Dashboard"', 'label: "Experiments"', 'label: "Analysis"'], "analysis sidebar model");
  assertTokenOrder(experimentsScript, ['label: "Dashboard"', 'label: "Experiments"', 'label: "Analysis"'], "experiments sidebar model");
});

test("global topbars keep page identity compact and move status into page content", () => {
  assert.match(dashboardHtml, /<header class="topbar">\s*<h1 class="page-title">Dashboard<\/h1>\s*<\/header>/, "dashboard topbar should contain only page identity");
  assert.match(analysisHtml, /<h1 class="page-title">Analysis<\/h1>/, "analysis topbar should contain page identity");
  assert.match(analysisHtml, /<span class="config-status" id="analysis-status">Ready<\/span>/, "analysis status should stay in the topbar cluster");
  assert.match(experimentsHtml, /<header class="topbar">\s*<h1 class="page-title">Experiments<\/h1>\s*<\/header>/, "experiments topbar should contain only page identity");
  assert.match(dashboardHtml, /id="config-status"[^>]*role="status"/, "dashboard run status should live in the run panel");
  assert.match(css, /\.topbar\s*\{[\s\S]*height:\s*var\(--shell-header-height\)/, "topbar should use the shared shell height");
  assert.match(css, /\.page-title\s*\{/, "topbar page title should use the shared title style");
});
