import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const html = readFileSync(new URL("../public/analysis.html", import.meta.url), "utf8");
const script = readFileSync(new URL("../public/analysis.js", import.meta.url), "utf8");
const css = readFileSync(new URL("../public/styles.css", import.meta.url), "utf8");
const componentUrl = new URL("../public/components/sidebar-nav.js", import.meta.url);
const vaultControl = readFileSync(new URL("../public/components/vault-control.js", import.meta.url), "utf8");

test("analysis sidebar is mounted from the reusable nested nav component", () => {
  assert.ok(existsSync(componentUrl), "shared sidebar nav component should exist");
  assert.match(html, /id="sidebar-nav-root"/, "analysis page should mount the shared sidebar nav");
  assert.doesNotMatch(
    html,
    /<ul class="sidebar-nav">\s*<li/s,
    "analysis page should not hand-write the sidebar item list",
  );
  assert.match(script, /import\s+\{\s*mountSidebarNav\s*\}/, "analysis script should import the shared nav component");
  assert.match(script, /ANALYSIS_NAV_ITEMS/, "analysis script should define its nav model");
  assert.match(script, /onNavigate/, "analysis nav children should switch panels without page reloads");

  for (const tab of ["Views", "Query"]) {
    assert.match(script, new RegExp(`label:\\s*"${tab}"`), `nav model should include ${tab}`);
  }
});

test("analysis page exposes direct views and query panels", () => {
  assert.match(html, /<div class="analysis-panel active" data-panel="views">/, "views panel should open directly");
  assert.match(html, /<div class="analysis-panel" data-panel="query" hidden>/, "query panel should be mounted and hidden until selected");
  assert.match(html, /id="analysis-view-tabs"/, "views panel should mount the view selector");
  assert.match(html, /id="analysis-query-input"/, "query panel should expose the SQL editor");
  assert.match(html, /id="analysis-status"/, "analysis status should remain available");
});

test("analysis page keeps current panel and view selector styles", () => {
  assert.match(css, /\.analysis-panel\[hidden\]\s*\{/, "hidden analysis panels should be explicitly styled");
  assert.match(css, /\.analysis-panel\.active\s*\{/, "active analysis panels should be explicitly styled");
  assert.match(css, /\.analysis-view-selector\s*\{/, "view selector layout should remain");
  assert.match(css, /\.analysis-tab\s*\{/, "analysis view buttons should keep their existing tab styling");
});

test("analysis script renders dynamic content without direct innerHTML", () => {
  assert.doesNotMatch(script, /\.innerHTML\b/, "analysis rendering should use DOM construction and textContent");
});

test("analysis shared vault helper avoids innerHTML", () => {
  assert.doesNotMatch(vaultControl, /\.innerHTML\b/, "vault helper should avoid direct HTML injection");
});
