import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const html = readFileSync(new URL("../public/experiments.html", import.meta.url), "utf8");
const script = readFileSync(new URL("../public/experiments.js", import.meta.url), "utf8");
const css = readFileSync(new URL("../public/styles.css", import.meta.url), "utf8");
const componentUrl = new URL("../public/components/sidebar-nav.js", import.meta.url);
const profileConfig = readFileSync(new URL("../public/components/profile-config.js", import.meta.url), "utf8");
const judgePolicyControl = readFileSync(new URL("../public/components/judge-policy-control.js", import.meta.url), "utf8");
const vaultControl = readFileSync(new URL("../public/components/vault-control.js", import.meta.url), "utf8");
const htmlSink = "inner" + "HTML";

function ruleBody(selector, source = css) {
  const pattern = new RegExp(`${selector.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}\\s*\\{([^}]*)\\}`, "m");
  const match = source.match(pattern);
  assert.ok(match, `Expected to find CSS rule for ${selector}`);
  return match[1];
}

function hasDeclaration(body, property, valuePattern) {
  return new RegExp(`${property}\\s*:\\s*${valuePattern}\\s*;`).test(body);
}

function experimentPanel(panelName) {
  const startToken = `data-panel="${panelName}"`;
  const start = html.indexOf(startToken);
  assert.notEqual(start, -1, `Expected ${panelName} panel to exist`);
  const nextPanel = html.indexOf('<section class="panel experiment-panel"', start + startToken.length);
  return nextPanel === -1 ? html.slice(start) : html.slice(start, nextPanel);
}

test("experiments sidebar is mounted from the reusable nested nav component", () => {
  assert.ok(existsSync(componentUrl), "shared sidebar nav component should exist");
  const component = readFileSync(componentUrl, "utf8");

  assert.match(component, /export function createSidebarNav/, "component should expose a DOM factory");
  assert.match(component, /export function mountSidebarNav/, "component should expose a mount helper");
  assert.match(component, /children/, "component should support nested child links");
  assert.match(html, /id="sidebar-nav-root"/, "experiments page should mount the shared sidebar nav");
  assert.doesNotMatch(
    html,
    /<ul class="sidebar-nav">\s*<li/s,
    "experiments page should not hand-write the sidebar item list",
  );
  assert.match(script, /mountSidebarNav/, "experiments script should mount the shared nav component");
  assert.match(script, /onNavigate/, "experiments nav children should switch panels without page reloads");

  for (const tab of ["Design", "Results", "History"]) {
    assert.match(script, new RegExp(`label:\\s*"${tab}"`), `nav model should include ${tab}`);
  }
  assert.match(script, /const EXPERIMENT_PANEL_IDS = new Set\(\["design", "results", "history"\]\)/, "panel routing should match the visible nav children");
});

test("nested experiment sidebar links use stable child navigation styles", () => {
  const childList = ruleBody(".sidebar-child-list");
  const childLink = ruleBody(".sidebar-child-link");
  const activeChild = ruleBody(".sidebar-child-link.active");

  assert.ok(hasDeclaration(childList, "padding-left", "var\\(--sp-5\\)"), "child list should be indented");
  assert.ok(hasDeclaration(childList, "border-left", "1px\\s+solid\\s+var\\(--sidebar-border\\)"), "child list should show a vertical guide");
  assert.ok(hasDeclaration(childLink, "font-size", "var\\(--text-sm\\)"), "child links should be visually subordinate");
  assert.ok(hasDeclaration(activeChild, "border-left-color", "var\\(--accent\\)"), "active child should carry the accent marker");
});

test("experiments page exposes direct design, results, and history panels", () => {
  assert.match(html, /<section class="panel experiment-panel active" data-panel="design">/, "design panel should open directly");
  assert.match(html, /<section class="panel experiment-panel" data-panel="results" hidden>/, "results panel should be mounted and hidden until selected");
  assert.match(html, /<section class="panel experiment-panel" data-panel="history" hidden>/, "history panel should be mounted and hidden until selected");
  assert.match(html, /id="experiment-form"/, "design panel should include the experiment form");
  assert.match(html, /id="experiment-history"/, "history panel should include the experiment list");
});

test("results panel consolidates experiment output metrics and data explorer", () => {
  const resultsPanel = experimentPanel("results");

  assert.match(resultsPanel, /id="exp-results"/, "results panel should keep the interpreted result summary");
  assert.match(resultsPanel, /id="metrics-content"/, "results panel should include paper metrics");
  assert.match(resultsPanel, /class="data-explorer"/, "results panel should include the data explorer");
  assert.match(script, /tab === "data" \|\| tab === "metrics"\) return "results"/, "legacy data and metrics routes should normalize to results");
  assert.match(script, /const restored = saved \|\| \{\}/, "hash route normalization should work without saved state");
});

test("results panel is organized as an analyst workbench", () => {
  const resultsPanel = experimentPanel("results");
  const expectedOrder = [
    'href="#results-overview"',
    'href="#results-compare"',
    'href="#results-matrix"',
    'href="#results-evidence"',
    'href="#results-raw-data"',
  ];

  assert.match(resultsPanel, /class="[^"]*results-workbench-header/, "results should start with a contextual workbench header");
  assert.match(resultsPanel, /class="[^"]*results-section-nav/, "results should include in-page section navigation");
  assert.match(resultsPanel, /id="results-risk-list"/, "overview should include highest-risk rankings");
  assert.match(resultsPanel, /id="results-evidence-list"/, "results should include curated evidence rows");
  assert.match(resultsPanel, /<details class="[^"]*results-raw-data[^"]*" id="results-raw-data">/, "raw data should be a collapsed advanced section");
  assert.match(script, /function renderEvidence/, "experiments script should render curated evidence, not just raw tables");

  let previousIndex = -1;
  for (const token of expectedOrder) {
    const index = resultsPanel.indexOf(token);
    assert.notEqual(index, -1, `results section nav should include ${token}`);
    assert.ok(index > previousIndex, "results sections should follow the analyst workflow order");
    previousIndex = index;
  }
});

test("results experiment switcher reloads the full workbench", () => {
  const resultsPanel = experimentPanel("results");

  assert.match(resultsPanel, /id="load-metrics-btn"[^>]*>Load Results<\/button>/, "switch control should describe full results loading");
  assert.match(script, /async function loadSelectedExperimentResults/, "script should expose a full workbench reload handler");
  assert.match(script, /loadSelectedExperimentResults[\s\S]*loadExperiment\(eid\)/, "switch handler should fetch the selected experiment");
  assert.match(
    script,
    /el\.loadMetricsBtn\.addEventListener\("click", loadSelectedExperimentResults\)/,
    "switch button should reload the full selected experiment, not metrics only",
  );
});

test("results empty state exposes the previous experiment switcher", () => {
  const resultsPanel = experimentPanel("results");
  const emptyIndex = resultsPanel.indexOf('id="exp-results-empty"');
  const selectorIndex = resultsPanel.indexOf('id="metrics-selector"');
  const workbenchIndex = resultsPanel.indexOf('id="exp-results" class="results-workbench"');

  assert.notEqual(emptyIndex, -1, "results panel should keep an empty state");
  assert.notEqual(selectorIndex, -1, "results panel should expose the experiment selector");
  assert.notEqual(workbenchIndex, -1, "results panel should keep the results workbench");
  assert.ok(selectorIndex < workbenchIndex, "experiment selector should be available before the hidden workbench loads");
  assert.match(
    resultsPanel,
    /id="exp-results-empty"[\s\S]*Choose a previous experiment/,
    "empty state should invite direct selection from previous experiments",
  );
});

test("results workbench keeps one compare, matrix, and evidence surface", () => {
  const resultsPanel = experimentPanel("results");

  assert.match(resultsPanel, /id="results-compare"/, "compare section should be the metric comparison surface");
  assert.match(resultsPanel, /id="metrics-body"/, "compare section should mount the metrics renderer");
  assert.match(resultsPanel, /id="matrix-wrap"/, "matrix section should mount the heatmap renderer");
  assert.match(resultsPanel, /id="results-evidence-list"/, "evidence section should mount curated evidence rows");
  assert.match(script, /function renderMetrics/, "script should render compare metrics");
  assert.match(script, /function renderMatrix/, "script should render the results matrix");
  assert.match(script, /function renderEvidence/, "script should render evidence rows");
});

test("experiments save profile ids instead of raw API keys", () => {
  assert.match(html, /id="exp-target-profile"/, "experiments should expose a target profile selector");
  assert.match(html, /id="exp-attacker-profile"/, "experiments should expose an attacker profile selector");
  assert.match(html, /id="exp-judge-profile"/, "experiments should expose a judge profile selector");
  assert.match(script, /function getStoredTargetConfig/, "experiments should have a non-secret persistence helper");
  assert.match(script, /target_profile_id/, "experiments persisted config should include target profile id");
  assert.match(script, /attacker_profile_id/, "experiments persisted config should include attacker profile id");
  assert.match(script, /judge_profile_id/, "experiments persisted config should include judge profile id");
  assert.match(profileConfig, /delete config\.api_key/, "shared profile helper should drop target API keys");
  assert.match(profileConfig, /delete config\.attacker_api_key/, "shared profile helper should drop attacker API keys");
  assert.match(profileConfig, /delete config\.judge_api_key/, "shared profile helper should drop judge API keys");
  assert.match(script, /function sanitizeExperimentState/, "experiments should sanitize restored state");
  assert.match(
    script,
    /localStorage\.setItem\(STATE_KEY,\s*JSON\.stringify\(sanitized\)\)/,
    "experiments should remove legacy API keys from saved experiment state",
  );
  assert.match(
    script,
    /localStorage\.setItem\(CONFIG_KEY,\s*JSON\.stringify\(saved\)\)/,
    "experiments should remove legacy API keys from shared target config",
  );
});

test("experiments explain session-key precedence over saved profiles", () => {
  assert.equal(
    (html.match(/session key overrides the selected profile's saved key/g) || []).length,
    3,
    "target, attacker, and custom Judge fields should explain session-key precedence",
  );
});

test("experiments judge profile selector can use target or attacker as judge", () => {
  assert.match(script, /EXPERIMENT_JUDGE_TARGET_VALUE\s*=\s*"__judge_target__"/, "experiments should define a target-as-judge sentinel");
  assert.match(script, /EXPERIMENT_JUDGE_ATTACKER_VALUE\s*=\s*"__judge_attacker__"/, "experiments should define an attacker-as-judge sentinel");
  assert.match(script, /Use Target LLM/, "judge profile selector should include target-as-judge");
  assert.match(script, /Use Attacker LLM/, "judge profile selector should include attacker-as-judge");
  assert.match(script, /function profileIdFromSelect/, "experiments should normalize profile selector values");
  assert.doesNotMatch(
    script,
    /judge_profile_id:\s*el\.judgeProfile\.value/,
    "experiments should not persist judge source sentinels as profile ids",
  );
});

test("experiments use the shared configurable judge policy", () => {
  assert.match(html, /id="exp-judge-policy-summary-root"/, "design should mount the shared Judge policy summary");
  assert.match(html, /id="exp-judge-policy-editor-root"/, "design should mount the shared Judge policy editor");
  assert.match(script, /mountJudgePolicyControl/, "experiments should mount the shared Judge policy component");
  assert.match(script, /configuredJudgePolicy/, "experiments should serialize non-standard Judge policies");
  assert.match(script, /judge_policy: judgePolicy/, "experiment target config should include configured Judge criteria");
  assert.match(script, /judgePolicyControl\.validationMessage/, "experiments should validate Judge criteria before starting");
  assert.match(script, /retryJudgePolicyControl\.setPolicy\(targetConfig\.judge_policy \|\| \{\}\)/, "retry should display the experiment's Judge policy");
  assert.match(script, /configuredJudgePolicy\(retryJudgePolicyControl\.getPolicy\(\)\)/, "retry should serialize the reviewed Judge policy");
  assert.match(script, /retryJudgePolicyControl\.validationMessage\(\)/, "retry should validate edited Judge criteria");
  assert.match(script, /id = "retry-judge-policy-summary-root"/, "retry dialog should mount a policy summary");
  assert.match(script, /id = "retry-judge-policy-editor-root"/, "retry dialog should mount a policy editor");
  assert.match(judgePolicyControl, /Success looks like/, "shared control should expose outcome success criteria");
  assert.match(judgePolicyControl, /Failure looks like/, "shared control should expose outcome failure criteria");
  assert.match(judgePolicyControl, /Complete rubric/, "shared control should expose the advanced rubric");
});

test("experiments profile selectors render options without innerHTML", () => {
  assert.doesNotMatch(script, /select\.innerHTML\b/, "experiments profile selectors should build option nodes instead of injecting HTML");
});

test("experiments metrics dropdown renders options without innerHTML", () => {
  assert.doesNotMatch(script, /metricsExpSelect\.innerHTML\b/, "metrics experiment selector should build option nodes instead of injecting HTML");
});

test("experiments clear dynamic containers without innerHTML", () => {
  assert.doesNotMatch(
    script,
    new RegExp(String.raw`(?:container|dataEl\.resultsWrap)\.${htmlSink}\s*=\s*""`),
    "experiments should clear dynamic containers with replaceChildren",
  );
});

test("experiments dataset and history renderers avoid innerHTML", () => {
  assert.doesNotMatch(script, /datasetList\.innerHTML\b/, "dataset list should render with DOM nodes and textContent");
  assert.doesNotMatch(script, /historyList\.innerHTML\b/, "history list should render with DOM nodes and textContent");
});

test("experiments results overview renderers avoid innerHTML", () => {
  assert.doesNotMatch(script, /resultsMeta\.innerHTML\b/, "results metadata should render with DOM nodes and textContent");
  assert.doesNotMatch(script, /resultsRiskList\.innerHTML\b/, "risk list should render with DOM nodes and textContent");
});

test("experiments posture and severity renderers avoid innerHTML", () => {
  assert.doesNotMatch(script, /postureSummary\.innerHTML\b/, "posture summary should render with DOM nodes and textContent");
  assert.doesNotMatch(script, /severityBar\.innerHTML\b/, "severity bar should render with DOM nodes");
  assert.doesNotMatch(script, /severityLegend\.innerHTML\b/, "severity legend should render with DOM nodes and textContent");
});

test("experiments matrix and evidence renderers avoid innerHTML", () => {
  assert.doesNotMatch(script, /matrixWrap\.innerHTML\b/, "matrix should render with DOM nodes and textContent");
  assert.doesNotMatch(script, /resultsEvidenceList\.innerHTML\b/, "evidence list should render with DOM nodes and textContent");
});

test("experiments data explorer renders query results without innerHTML", () => {
  assert.doesNotMatch(script, /dataEl\.resultsWrap\.innerHTML\b/, "data explorer results should render with DOM nodes and textContent");
});

test("experiments technique selector avoids innerHTML", () => {
  assert.doesNotMatch(script, /(?:blGroup|header|row)\.innerHTML\b/, "technique selector should render checkbox rows with DOM nodes and textContent");
});

test("experiments metrics renderer avoids innerHTML", () => {
  assert.doesNotMatch(script, /metricsBody\.innerHTML\b/, "metrics body should render with DOM nodes and textContent");
});

test("experiments retry and rejudge overlays avoid innerHTML", () => {
  assert.doesNotMatch(script, /overlay\.innerHTML\b/, "experiment action overlays should render forms with DOM nodes and textContent");
});

test("experiment conversations render Judge failures explicitly", () => {
  assert.match(script, /if \(judge\.error\)/, "single-shot conversations should detect Judge errors");
  assert.match(script, /"Judge Error"/, "conversation modal should label Judge failures");
  assert.match(css, /\.judge-error\s*\{/, "Judge errors should have a distinct visual treatment");
});

test("experiments conversation modal avoids innerHTML", () => {
  assert.doesNotMatch(script, /modalBody\.innerHTML\b/, "experiment conversation modal should render with DOM nodes and textContent");
});

test("experiments script has no direct innerHTML usage", () => {
  assert.doesNotMatch(script, /\.innerHTML\b/, "experiments script should avoid direct HTML injection entirely");
  assert.doesNotMatch(judgePolicyControl, /\.innerHTML\b/, "shared Judge policy control should avoid HTML injection");
  assert.doesNotMatch(vaultControl, /\.innerHTML\b/, "shared vault helper should avoid direct HTML injection");
});

test("experiments show public technique keys without aliases", () => {
  assert.match(
    script,
    /function displayTechniqueKey\(key\) \{\s*return String\(key \|\| ""\);\s*\}/,
    "technique labels should use the public registry key directly",
  );
  assert.match(
    html,
    /id="exp-group" placeholder="e\.g\. baseline-comparison"/,
    "experiment groups should use a neutral public placeholder",
  );
});
