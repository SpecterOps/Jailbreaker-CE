import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const html = readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
const css = readFileSync(new URL("../public/styles.css", import.meta.url), "utf8");
const app = readFileSync(new URL("../public/app.js", import.meta.url), "utf8");
const domHelpers = readFileSync(new URL("../public/components/dom.js", import.meta.url), "utf8");
const profileConfig = readFileSync(new URL("../public/components/profile-config.js", import.meta.url), "utf8");
const vaultControl = readFileSync(new URL("../public/components/vault-control.js", import.meta.url), "utf8");
const htmlSink = "inner" + "HTML";

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function ruleBody(selector, source = css) {
  const pattern = new RegExp(`${escapeRegExp(selector)}\\s*\\{([^}]*)\\}`, "m");
  const match = source.match(pattern);
  assert.ok(match, `Expected to find CSS rule for ${selector}`);
  return match[1];
}

function mediaBlock(condition, source = css) {
  const pattern = new RegExp(`@media\\s+\\(${escapeRegExp(condition)}\\)\\s*\\{`, "m");
  const match = pattern.exec(source);
  assert.ok(match, `Expected to find @media (${condition}) block`);

  const blockStart = match.index + match[0].length;
  let depth = 1;
  for (let index = blockStart; index < source.length; index += 1) {
    if (source[index] === "{") {
      depth += 1;
    } else if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) {
        return source.slice(blockStart, index);
      }
    }
  }

  assert.fail(`Expected to find closing brace for @media (${condition})`);
}

function selectorListRuleBody(selector, source = css) {
  const pattern = /([^{}]+)\{([^{}]*)\}/g;
  let match;
  while ((match = pattern.exec(source)) !== null) {
    const selectors = match[1].split(",").map((item) => item.trim());
    if (selectors.includes(selector)) {
      return match[2];
    }
  }

  assert.fail(`Expected to find CSS rule including ${selector}`);
}

function hasDeclaration(body, property, valuePattern) {
  return new RegExp(`${property}\\s*:\\s*${valuePattern}\\s*;`).test(body);
}

test("dashboard uses operator console structure", () => {
  assert.match(html, /class="operator-run-strip"/, "dashboard needs a compact run control strip");
  assert.match(html, /class="operator-workspace"/, "dashboard needs a split operator workspace");
  assert.match(html, /class="operator-sidebar"/, "dashboard needs a persistent monitor sidebar");
  assert.match(html, /id="operator-target-summary"/, "dashboard needs a target summary");
  assert.match(html, /id="operator-judge-summary"/, "dashboard needs a judge summary");
  assert.match(html, /id="config-status"[^>]*role="status"/, "config status should be announced as a live status");
});

test("operator run control uses profile selector blocks instead of credential details", () => {
  assert.match(html, /class="operator-profile-row"/, "dashboard should render the LLM profile row");
  assert.match(html, /data-profile-role="target"/, "dashboard should render a target profile block");
  assert.match(html, /data-profile-role="judge"/, "dashboard should render a judge profile block");
  assert.match(html, /data-profile-role="attacker"/, "dashboard should render an attacker profile block");
  assert.match(html, /id="operator-profile-editor"/, "dashboard should render one shared profile editor");
  assert.match(html, /id="profile-editor-save-btn"/, "shared editor should save profiles");
  assert.match(html, /id="profile-editor-cancel-btn"/, "shared editor should cancel edits");
  assert.doesNotMatch(html, /<details class="operator-config-details"/, "target credential details should be removed");
  assert.doesNotMatch(html, /id="attacker-config-section"/, "attacker credential details should be removed");
  assert.doesNotMatch(html, /id="judge-config-section"/, "judge credential details should be removed");
});

test("operator profile cards expose configure actions and selector shells", () => {
  const configureCopies = {
    target: "Configure Target LLM",
    judge: "Configure Judge LLM",
    attacker: "Configure Attacker LLM",
  };

  for (const [role, configureCopy] of Object.entries(configureCopies)) {
    const blockStart = html.indexOf(`data-profile-role="${role}"`);
    assert.notEqual(blockStart, -1, `${role} profile block should exist`);
    const blockEnd = html.indexOf("</div>", html.indexOf(`operator-${role}-summary`, blockStart));
    const block = html.slice(blockStart, blockEnd);

    assert.match(
      block,
      new RegExp(`id="${role}-profile-configure-btn"`),
      `${role} card should expose a configure button`,
    );
    assert.match(
      block,
      /class="[^"]*\bbtn-primary\b[^"]*\boperator-profile-configure-btn\b[^"]*"/,
      `${role} configure button should use the primary run-button treatment`,
    );
    assert.doesNotMatch(
      block.match(new RegExp(`<button[\\s\\S]*?id="${role}-profile-configure-btn"[\\s\\S]*?</button>`))?.[0] || "",
      /\bbtn-secondary\b/,
      `${role} configure button should not use secondary styling`,
    );
    assert.match(
      block,
      new RegExp(`id="${role}-profile-select-shell"`),
      `${role} card should wrap the select in a styled shell`,
    );
    assert.match(
      block,
      new RegExp(`id="${role}-profile-select-row"`),
      `${role} card should keep the select and add action in one row`,
    );
    assert.match(
      block,
      new RegExp(`id="${role}-profile-add-btn"`),
      `${role} card should expose an adjacent add profile action`,
    );
    assert.match(block, new RegExp(`id="${role}-profile"`), `${role} select id should remain unchanged`);
    assert.match(
      block,
      new RegExp(`>${escapeRegExp(configureCopy)}<`),
      `${role} card should use role-specific configure copy`,
    );
  }
});

test("operator top strip keeps only intent and run actions", () => {
  const stripStart = html.indexOf('<div class="operator-run-strip">');
  assert.notEqual(stripStart, -1, "operator run strip should exist");
  const stripEnd = html.indexOf('<div class="operator-profile-row"', stripStart);
  assert.notEqual(stripEnd, -1, "operator profile row should follow the top strip");
  const strip = html.slice(stripStart, stripEnd);

  assert.match(strip, /id="unsafe-intent"/, "intent input should remain in the top strip");
  assert.match(strip, /id="save-config-btn"/, "save button should remain in the top strip");
  assert.match(strip, /id="start-eval-btn"/, "run all button should remain in the top strip");
  assert.doesNotMatch(strip, /id="model"/, "target model should move out of the top strip");
  assert.doesNotMatch(strip, /id="judge-mode"/, "judge source should move out of the top strip");
});

test("operator workspace keeps techniques primary and monitor visible", () => {
  const workspace = ruleBody(".operator-workspace");
  const runStrip = ruleBody(".operator-run-strip");
  const sidebar = ruleBody(".operator-sidebar");

  assert.ok(hasDeclaration(workspace, "display", "grid"), "workspace should use grid layout");
  assert.ok(
    hasDeclaration(workspace, "grid-template-columns", "minmax\\(0,\\s*1fr\\)\\s+minmax\\(300px,\\s*320px\\)"),
    "workspace should reserve a stable right monitor column",
  );
  assert.ok(hasDeclaration(runStrip, "display", "grid"), "run strip should use grid layout");
  assert.ok(
    hasDeclaration(runStrip, "grid-template-columns", "minmax\\(280px,\\s*1fr\\)\\s+auto"),
    "run strip should reserve intent and action columns only",
  );
  assert.ok(hasDeclaration(sidebar, "position", "sticky"), "monitor sidebar should stay visible");
});

test("run monitor exposes live run state fields", () => {
  const monitorStart = html.indexOf('<section class="panel operator-monitor-panel">');
  assert.notEqual(monitorStart, -1, "run monitor panel should exist");
  const monitorEnd = html.indexOf("</section>", monitorStart);
  const monitor = html.slice(monitorStart, monitorEnd);

  for (const id of [
    "operator-monitor-state",
    "operator-monitor-current",
    "operator-monitor-elapsed",
    "operator-monitor-baseline",
    "operator-monitor-error",
    "operator-monitor-results",
    "eval-progress",
    "progress-current",
  ]) {
    assert.match(monitor, new RegExp(`id="${id}"`), `run monitor should expose ${id}`);
  }
  assert.doesNotMatch(monitor, /Control row in library/, "baseline monitor copy should be dynamic");
  assert.doesNotMatch(monitor, /Shown after batch run/, "results monitor copy should be dynamic");
});

test("dashboard script drives run monitor from all run paths", () => {
  assert.match(app, /monitor:\s*\{/, "dashboard state should include monitor state");
  assert.match(app, /function renderRunMonitor\(\)/, "dashboard should centralize monitor rendering");
  assert.match(app, /function startMonitorRun\(/, "dashboard should start monitor state for active runs");
  assert.match(app, /function finishMonitorRun\(/, "dashboard should finish monitor state after runs");
  assert.match(app, /function failMonitorRun\(/, "dashboard should record monitor errors");
  assert.match(app, /function setMonitorProgress\(/, "dashboard should update monitor progress");
  assert.match(
    app,
    /async function runTechnique[\s\S]*startMonitorRun\(\{[\s\S]*type:\s*"single"/,
    "single technique runs should activate the monitor",
  );
  assert.match(
    app,
    /async function pollRunUntilDone[\s\S]*setMonitorProgress\(\{[\s\S]*currentLabel:/,
    "single run polling should update monitor current activity",
  );
  assert.match(
    app,
    /async function runBaseline[\s\S]*setMonitorBaseline\("running"/,
    "baseline runs should update monitor baseline state",
  );
  assert.match(
    app,
    /function startPolling\(evalId\)[\s\S]*startMonitorRun\(\{[\s\S]*type:\s*"batch"/,
    "batch evaluations should activate the monitor",
  );
  assert.match(
    app,
    /function renderProgress\(\)[\s\S]*setMonitorProgress\(\{/,
    "batch progress rendering should also update monitor progress",
  );
  assert.match(
    app,
    /catch \(err\) \{[\s\S]*failMonitorRun\(`Error: \$\{err\.message\}`\)/,
    "run errors should be reflected in the monitor",
  );
});

test("operator profile row and shared editor are styled", () => {
  const runPanel = ruleBody(".operator-run-panel");
  const profileRow = ruleBody(".operator-profile-row");
  const profileBlock = ruleBody(".operator-profile-block");
  const profileEditor = ruleBody(".operator-profile-editor");
  const profileEditorHidden = ruleBody(".operator-profile-editor[hidden]");
  const backingFields = ruleBody(".operator-backing-fields");
  const mobileProfileRow = selectorListRuleBody(".operator-profile-row", mediaBlock("max-width: 900px"));

  assert.ok(
    hasDeclaration(runPanel, "border-top", "1px solid var\\(--border-light\\)"),
    "run control should not keep the bright accent top border",
  );
  assert.ok(hasDeclaration(profileRow, "display", "grid"), "profile row should use grid layout");
  assert.ok(
    hasDeclaration(profileRow, "grid-template-columns", "repeat\\(3,\\s*minmax\\(0,\\s*1fr\\)\\)"),
    "profile row should reserve three equal selector blocks",
  );
  assert.ok(hasDeclaration(profileBlock, "border", "1px solid var\\(--border-light\\)"), "selector blocks should use panel borders");
  assert.ok(hasDeclaration(profileBlock, "border-radius", "var\\(--radius\\)"), "selector blocks should keep compact radius");
  assert.ok(hasDeclaration(profileEditor, "margin-top", "var\\(--sp-4\\)"), "shared editor should sit below the selector row");
  assert.ok(hasDeclaration(profileEditorHidden, "display", "none !important"), "hidden editor should not occupy layout space");
  assert.ok(hasDeclaration(backingFields, "display", "none !important"), "backing fields should stay hidden");
  assert.ok(hasDeclaration(mobileProfileRow, "grid-template-columns", "1fr"), "profile row should stack on mobile");
});

test("operator profile configure buttons and selector shells are styled", () => {
  const configureButton = ruleBody(".btn.operator-profile-configure-btn");
  const configureButtonHover = ruleBody(".btn.operator-profile-configure-btn:hover");
  const configureButtonFocus = ruleBody(".btn.operator-profile-configure-btn:focus-visible");
  const configureButtonHidden = ruleBody(".btn.operator-profile-configure-btn[hidden]");
  const selectRow = ruleBody(".operator-profile-select-row");
  const selectRowHidden = ruleBody(".operator-profile-select-row[hidden]");
  const selectShell = ruleBody(".operator-profile-select-shell");
  const selectShellHidden = ruleBody(".operator-profile-select-shell[hidden]");
  const shellSelect = ruleBody(".operator-profile-select-shell select");
  const shellChevron = ruleBody(".operator-profile-select-shell::after");
  const addButton = ruleBody(".operator-profile-add-btn");
  const addButtonHidden = ruleBody(".operator-profile-add-btn[hidden]");
  const editButton = ruleBody(".operator-profile-edit-btn");
  const editButtonHidden = ruleBody(".operator-profile-edit-btn[hidden]");

  assert.ok(hasDeclaration(configureButton, "display", "flex"), "configure buttons should center their contents");
  assert.ok(hasDeclaration(configureButton, "width", "100%"), "configure buttons should span the card");
  assert.ok(hasDeclaration(configureButton, "min-height", "44px"), "configure buttons should keep stable height");
  assert.ok(hasDeclaration(configureButtonHidden, "display", "none !important"), "hidden configure buttons should leave layout");
  assert.ok(hasDeclaration(configureButton, "background", "#536176"), "configure buttons should use the mature slate tone");
  assert.ok(hasDeclaration(configureButton, "color", "#fff"), "configure buttons should keep high-contrast text");
  assert.ok(
    hasDeclaration(configureButton, "box-shadow", "0 0 16px rgba\\(83, 97, 118, 0\\.22\\)"),
    "configure buttons should use a restrained slate glow",
  );
  assert.ok(hasDeclaration(configureButtonHover, "background", "#62728a"), "configure hover should subtly brighten slate");
  assert.ok(hasDeclaration(configureButtonHover, "transform", "translateY\\(-1px\\)"), "configure hover should keep primary lift");
  assert.ok(
    hasDeclaration(configureButtonHover, "box-shadow", "0 0 24px rgba\\(83, 97, 118, 0\\.3\\)"),
    "configure hover should keep a restrained slate glow",
  );
  assert.doesNotMatch(css, /data-profile-role="target"\]\s+\.btn\.operator-profile-configure-btn\s*\{/);
  assert.doesNotMatch(css, /data-profile-role="judge"\]\s+\.btn\.operator-profile-configure-btn\s*\{/);
  assert.doesNotMatch(css, /data-profile-role="attacker"\]\s+\.btn\.operator-profile-configure-btn\s*\{/);
  assert.ok(hasDeclaration(configureButtonFocus, "outline", "2px solid var\\(--accent\\)"), "configure focus should keep visible accent outline");
  assert.ok(hasDeclaration(selectRow, "display", "flex"), "selector row should place select and add action side by side");
  assert.ok(hasDeclaration(selectRow, "align-items", "stretch"), "selector row should keep adjacent controls the same height");
  assert.ok(hasDeclaration(selectRow, "gap", "var\\(--sp-2\\)"), "selector row should use compact spacing");
  assert.ok(hasDeclaration(selectRowHidden, "display", "none !important"), "hidden selector row should leave layout");
  assert.ok(hasDeclaration(selectShell, "position", "relative"), "selector shell should own the chevron");
  assert.ok(hasDeclaration(selectShell, "flex", "1 1 auto"), "selector shell should fill remaining row width");
  assert.ok(hasDeclaration(selectShell, "min-height", "44px"), "selector shell should keep stable height");
  assert.ok(hasDeclaration(selectShell, "border", "1px solid var\\(--border\\)"), "selector shell should use the form border");
  assert.ok(hasDeclaration(selectShellHidden, "display", "none !important"), "hidden selector shell should leave layout");
  assert.ok(hasDeclaration(addButton, "flex", "0 0 44px"), "add action should keep a stable square footprint");
  assert.ok(hasDeclaration(addButton, "width", "44px"), "add action should use a compact fixed width");
  assert.ok(hasDeclaration(addButton, "min-height", "44px"), "add action should align with the selector height");
  assert.ok(hasDeclaration(addButtonHidden, "display", "none !important"), "hidden add action should leave layout");
  assert.ok(hasDeclaration(shellSelect, "appearance", "none"), "native select should hide browser chrome inside shell");
  assert.ok(hasDeclaration(shellSelect, "width", "100%"), "native select should fill shell");
  assert.ok(hasDeclaration(shellSelect, "border", "0"), "native select should rely on the shell border");
  assert.ok(
    hasDeclaration(shellSelect, "padding", "0 calc\\(var\\(--sp-5\\) \\+ 10px\\) 0 var\\(--sp-3\\)"),
    "native select should leave room for the chevron",
  );
  assert.match(shellChevron, /content:\s*"";/, "selector shell should draw a chevron");
  assert.ok(hasDeclaration(shellChevron, "pointer-events", "none"), "selector chevron should not intercept select clicks");
  assert.ok(
    hasDeclaration(shellChevron, "transform", "rotate\\(45deg\\) translateY\\(-2px\\)"),
    "selector chevron should be angled downward",
  );
  assert.ok(hasDeclaration(editButton, "width", "100%"), "edit action should align with selector width");
  assert.ok(hasDeclaration(editButtonHidden, "display", "none !important"), "hidden edit action should leave layout");
});

test("technique table preserves readable columns in the operator workspace", () => {
  const primary = ruleBody(".operator-primary");
  const tableHeader = ruleBody(".table-header");
  const techniqueRow = ruleBody(".technique-row");

  assert.ok(
    hasDeclaration(primary, "overflow-x", "auto"),
    "technique panel should scroll horizontally instead of clipping columns",
  );
  assert.ok(
    hasDeclaration(tableHeader, "min-width", "760px"),
    "table header should keep enough width for all desktop columns",
  );
  assert.ok(
    hasDeclaration(techniqueRow, "min-width", "760px"),
    "technique rows should align with the readable desktop header",
  );
  assert.match(
    css,
    /@media\s+\(max-width:\s*1280px\)[\s\S]*?\.operator-workspace[\s\S]*?grid-template-columns:\s*1fr;/,
    "operator workspace should collapse before the table gets squeezed",
  );
});

test("dashboard script refreshes operator summaries with render lifecycle", () => {
  assert.match(app, /operatorTargetSummary/, "app should cache operator target summary element");
  assert.match(app, /function renderOperatorSummary/, "app should render operator summary text");
  assert.match(app, /renderOperatorSummary\(\);/, "render lifecycle should refresh operator summary");
});

test("dashboard saves profile ids instead of raw API keys", () => {
  assert.match(html, /id="target-profile"/, "dashboard should expose a target profile selector");
  assert.match(html, /id="attacker-profile"/, "dashboard should expose an attacker profile selector");
  assert.match(html, /id="judge-profile"/, "dashboard should expose a judge profile selector");
  assert.match(app, /function getStoredTargetConfig/, "dashboard should have a non-secret persistence helper");
  assert.match(app, /target_profile_id/, "dashboard persisted config should include target profile id");
  assert.match(app, /attacker_profile_id/, "dashboard persisted config should include attacker profile id");
  assert.match(app, /judge_profile_id/, "dashboard persisted config should include judge profile id");
  assert.match(app, /stripSecretConfigKeys/, "dashboard should use centralized persisted secret scrubbing");
  assert.match(profileConfig, /function stripSecretConfigKeys/, "profile helper should centralize persisted secret scrubbing");
  assert.match(
    app,
    /hasSecretConfigKeys\(parsed\)[\s\S]*localStorage\.setItem\(STORAGE_KEY,\s*JSON\.stringify\(c\)\)/,
    "dashboard should remove legacy API keys from localStorage when loading old config",
  );
  assert.doesNotMatch(
    app,
    /localStorage\.setItem\(STORAGE_KEY,\s*JSON\.stringify\(getTargetConfig\(\)\)\)/,
    "dashboard should not persist the full target config with API keys",
  );
});

test("dashboard profile selectors render options without innerHTML", () => {
  assert.doesNotMatch(app, /select\.innerHTML\b/, "dashboard profile selectors should build option nodes instead of injecting HTML");
});

test("profile select helper supports configure-new sentinel options", () => {
  assert.match(
    profileConfig,
    /export const CONFIGURE_NEW_VALUE\s*=\s*"__configure_new__"/,
    "profile helper should export the shared configure-new sentinel",
  );
  assert.match(
    profileConfig,
    /configureLabel/,
    "profile helper should accept a configure label option",
  );
  assert.match(
    profileConfig,
    /createSelectOption\(CONFIGURE_NEW_VALUE,\s*configureLabel\)/,
    "profile helper should append the configure-new option with DOM APIs",
  );
  assert.match(
    profileConfig,
    /beforeOptions/,
    "profile helper should support built-in options before saved profiles",
  );
  assert.doesNotMatch(
    profileConfig,
    /\.innerHTML\b/,
    "profile helper should keep option construction DOM-safe",
  );
});

test("dashboard app handles configure-new without persisting sentinels", () => {
  assert.match(app, /profileEditorRole:\s*null/, "dashboard state should track the active shared editor role");
  assert.match(app, /profileEditorPreviousValue/, "dashboard should remember the previous non-editor selection");
  assert.match(app, /profileEditorProfileId:\s*null/, "dashboard should track which saved profile is being edited");
  assert.match(app, /function handleProfileSelection\(role\)/, "profile change handling should be role-aware");
  assert.match(app, /function profileIdForRole\(role\)/, "config serialization should normalize role profile ids");
  assert.match(app, /function editProfileRole\(role\)/, "dashboard should edit existing saved profiles through the shared editor");
  assert.match(app, /CONFIGURE_NEW_VALUE/, "dashboard should share the configure-new sentinel");
  assert.match(
    app,
    /if \(value === CONFIGURE_NEW_VALUE\)[\s\S]*openProfileEditor\(role\)/,
    "selecting configure-new should open the shared editor",
  );
  assert.doesNotMatch(
    app,
    /target_profile_id:\s*el\.targetProfile\.value/,
    "target_profile_id should not serialize the raw select value",
  );
  assert.doesNotMatch(
    app,
    /attacker_profile_id:\s*el\.attackerProfile\.value/,
    "attacker_profile_id should not serialize the raw select value",
  );
  assert.doesNotMatch(
    app,
    /judge_profile_id:\s*el\.judgeProfile\.value/,
    "judge_profile_id should not serialize judge sentinels",
  );
});

test("dashboard renders role-specific selector options", () => {
  assert.match(app, /targetProfileAddBtn/, "target selector should expose adjacent add action wiring");
  assert.match(app, /judgeProfileAddBtn/, "judge selector should expose adjacent add action wiring");
  assert.match(app, /attackerProfileAddBtn/, "attacker selector should expose adjacent add action wiring");
  assert.match(app, /Use Target LLM/, "judge selector should include target source mode");
  assert.match(app, /Use Attacker LLM/, "judge selector should include attacker source mode");
  assert.match(app, /No attacker selected/, "attacker selector should keep the no-attacker option");
  assert.match(app, /firstVaultProfileId/, "selectors should default to a saved vault profile when one exists");
  assert.match(app, /setProfileSelectionForAllRoles/, "saved profiles should be selected for target, judge, and attacker together");
});

test("dashboard clears dynamic lists without innerHTML", () => {
  assert.doesNotMatch(
    app,
    new RegExp(String.raw`el\.(surfacePills|techniqueList|resultsGrid)\.${htmlSink}\s*=\s*""`),
    "dashboard should clear dynamic list containers with replaceChildren",
  );
});

test("dashboard technique list avoids innerHTML", () => {
  assert.doesNotMatch(app, /(?:header|preview|row)\.innerHTML\b/, "dashboard technique list should render group and row content with DOM nodes and textContent");
});

test("dashboard baseline row avoids innerHTML", () => {
  assert.doesNotMatch(app, /baselineRow\.innerHTML\b/, "dashboard baseline row should render with DOM nodes and textContent");
});

test("dashboard result summaries avoid innerHTML", () => {
  assert.doesNotMatch(
    app,
    /(?:postureSummary|severityBar|severityLegend|card)\.innerHTML\b/,
    "dashboard result summaries and cards should render with DOM nodes and textContent",
  );
});

test("dashboard conversation modals avoid innerHTML", () => {
  assert.doesNotMatch(app, /modalBody\.innerHTML\b/, "dashboard conversation modals should render with DOM nodes and textContent");
});


test("shared frontend helper modules avoid innerHTML", () => {
  assert.doesNotMatch(`${domHelpers}\n${profileConfig}\n${vaultControl}`, /\.innerHTML\b/, "shared helpers should avoid direct HTML injection");
});
