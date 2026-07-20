import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { mountVaultControl } from "../public/components/vault-control.js";
import { fillProfileSelect } from "../public/components/profile-config.js";

const componentUrl = new URL("../public/components/vault-control.js", import.meta.url);
const css = readFileSync(new URL("../public/styles.css", import.meta.url), "utf8");
const dashboardHtml = readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
const experimentsHtml = readFileSync(new URL("../public/experiments.html", import.meta.url), "utf8");
const analysisHtml = readFileSync(new URL("../public/analysis.html", import.meta.url), "utf8");
const experimentsScript = readFileSync(new URL("../public/experiments.js", import.meta.url), "utf8");
const analysisScript = readFileSync(new URL("../public/analysis.js", import.meta.url), "utf8");

function componentSource() {
  assert.ok(existsSync(componentUrl), "shared vault control component should exist");
  return readFileSync(componentUrl, "utf8");
}

test("vault control component exists and exports mountVaultControl", () => {
  const source = componentSource();

  assert.match(source, /export function mountVaultControl\b/);
});

test("vault control source includes required render and action helpers", () => {
  const source = componentSource();

  for (const helper of ["renderVaultButton", "renderVaultDialog", "createVault", "unlockVault", "lockVault"]) {
    assert.match(source, new RegExp(`\\b${helper}\\b`), `expected ${helper} helper`);
  }
});

test("vault control avoids direct innerHTML rendering", () => {
  assert.doesNotMatch(componentSource(), /\.innerHTML\b/);
});

test("vault control includes create, unlock, and unlocked modal states", () => {
  const source = componentSource();

  for (const marker of ["vault-mode-create", "vault-mode-unlock", "vault-mode-unlocked"]) {
    assert.match(source, new RegExp(marker), `expected ${marker} state marker`);
  }

  for (const label of ["Passphrase", "Confirm Passphrase", "Create Vault", "Unlock Vault", "Lock Vault"]) {
    assert.match(source, new RegExp(label), `expected ${label} label`);
  }
});

test("vault control styles define sidebar and dialog selectors", () => {
  for (const selector of [".sidebar-vault-root", ".vault-sidebar-button", ".vault-modal-overlay", ".vault-dialog"]) {
    assert.match(css, new RegExp(`${selector.replace(".", "\\.")}\\s*\\{`), `expected ${selector} rule`);
  }
});

test("vault control avoids browser storage", () => {
  const source = componentSource();

  assert.doesNotMatch(source, /\blocalStorage\b/, "vault passphrases should not touch localStorage");
  assert.doesNotMatch(source, /\bsessionStorage\b/, "vault passphrases should not touch sessionStorage");
});

test("profile selector offers saved profiles across all LLM roles", () => {
  const previousDocument = globalThis.document;
  const documentRef = new FakeDocument();
  globalThis.document = documentRef;
  try {
    const select = documentRef.createElement("select");
    fillProfileSelect(
      select,
      [
        { profile_id: "target-profile", name: "Target Cloud", model: "gpt-4o-mini", scope: "target" },
        { profile_id: "judge-profile", name: "Judge Local", model: "llama-3.1", scope: "judge" },
        { profile_id: "attacker-profile", name: "Attacker", model: "mixtral", scope: "attacker" },
      ],
      "attacker",
    );

    assert.deepEqual(
      select.children.map((option) => option.value),
      ["", "target-profile", "judge-profile", "attacker-profile"],
      "attacker selector should include target, judge, and attacker profiles",
    );
  } finally {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});

test("profile selector falls back from stale current value to saved profile", () => {
  const previousDocument = globalThis.document;
  const documentRef = new FakeDocument();
  globalThis.document = documentRef;
  try {
    const select = documentRef.createElement("select");
    select.value = "deleted-profile";
    fillProfileSelect(
      select,
      [{ profile_id: "shared-profile", name: "Shared Cloud", model: "gpt-4o-mini", scope: "all" }],
      "target",
      "shared-profile",
    );

    assert.equal(select.value, "shared-profile");
  } finally {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});

test("vault sidebar styles keep desktop and mobile footer controls stable", () => {
  const desktopFooter = ruleBody(".sidebar-footer");
  const desktopRoot = ruleBody(".sidebar-vault-root");
  const desktopButton = ruleBody(".vault-sidebar-button");
  const mobile = mediaBlock("@media (max-width: 900px)");
  const mobileFooter = ruleBody(".sidebar-footer", mobile);
  const mobileButton = ruleBody(".vault-sidebar-button", mobile);

  assert.ok(hasDeclaration(desktopFooter, "display", "grid"), "desktop sidebar footer should use grid layout");
  assert.ok(hasDeclaration(desktopFooter, "justify-items", "center"), "desktop sidebar footer should center its vault root");
  assert.ok(hasDeclaration(desktopFooter, "gap", "var\\(--sp-3\\)"), "desktop sidebar footer should use stable spacing");
  assert.ok(hasDeclaration(desktopRoot, "display", "grid"), "desktop vault root should center the icon button");
  assert.ok(hasDeclaration(desktopRoot, "place-items", "center"), "desktop vault root should center the icon button");
  assert.ok(hasDeclaration(desktopRoot, "width", "100%"), "desktop vault root should span the sidebar column for centering");
  assert.ok(hasDeclaration(desktopButton, "width", "42px"), "desktop vault button should keep stable width");
  assert.ok(hasDeclaration(desktopButton, "height", "42px"), "desktop vault button should keep stable height");
  assert.ok(hasDeclaration(desktopButton, "min-height", "42px"), "desktop vault button should keep stable height");
  assert.ok(hasDeclaration(desktopButton, "place-items", "center"), "vault icon should be centered in the button");
  assert.ok(hasDeclaration(desktopButton, "padding", "0"), "desktop vault button should not offset icon with padding");
  assert.ok(hasDeclaration(mobileFooter, "display", "flex"), "mobile sidebar footer should use flex layout");
  assert.ok(hasDeclaration(mobileFooter, "gap", "var\\(--sp-2\\)"), "mobile footer should keep compact spacing");
  assert.ok(hasDeclaration(mobileFooter, "align-items", "center"), "mobile footer should align controls");
  assert.ok(hasDeclaration(mobileButton, "width", "36px"), "mobile vault button should use compact stable width");
  assert.ok(hasDeclaration(mobileButton, "height", "36px"), "mobile vault button should use compact stable height");
  assert.ok(hasDeclaration(mobileButton, "min-height", "36px"), "mobile vault button should use compact stable height");
  assert.ok(hasDeclaration(mobileButton, "padding", "0"), "mobile vault button should not offset icon with padding");
});

test("vault sidebar button is icon-only with hover and accessible text", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  const fetchJson = async () => ({ exists: true, unlocked: false, profiles: [] });

  mountVaultControl(root, { fetchJson, documentRef });
  await flushPromises();

  const button = root.querySelector(".vault-sidebar-button");
  assert.ok(root.querySelector(".vault-icon"), "vault button should render a vault icon");
  assert.equal(root.querySelector(".vault-sidebar-label"), null, "vault button should not show text label");
  assert.equal(root.querySelector(".vault-sidebar-status"), null, "vault button should not show status text");
  assert.equal(button.getAttribute("title"), "LLM Vault: Locked");
  assert.equal(button.getAttribute("aria-label"), "LLM Vault: Locked");
});

test("vault control returns focus to the visible sidebar button after close", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  const fetchJson = async () => ({ exists: true, unlocked: false, profiles: [] });

  const control = mountVaultControl(root, { fetchJson, documentRef });
  await flushPromises();

  const trigger = root.querySelector(".vault-sidebar-button");
  trigger.focus();
  control.open();
  documentRef.querySelector(".vault-dialog-close").dispatchEvent({ type: "click" });

  assert.equal(documentRef.activeElement, root.querySelector(".vault-sidebar-button"));
});

test("vault dialog mounts outside the sticky sidebar root", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  documentRef.body.append(root);
  const fetchJson = async () => ({ exists: true, unlocked: false, profiles: [] });

  const control = mountVaultControl(root, { fetchJson, documentRef });
  await flushPromises();
  control.open();

  const overlay = documentRef.querySelector(".vault-modal-overlay");
  assert.ok(overlay, "vault dialog overlay should render");
  assert.equal(overlay.parent, documentRef.body, "overlay should be portaled to the document body");
  assert.equal(root.querySelector(".vault-modal-overlay"), null, "sidebar root should only contain the trigger button");
});

test("vault control waits for initial vault state before showing passphrase fields", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  const pending = deferred();
  const fetchJson = () => pending.promise;

  const control = mountVaultControl(root, { fetchJson, documentRef });
  control.open();

  assert.equal(documentRef.querySelector(".vault-modal-overlay").querySelector("#vault-create-passphrase"), null);

  pending.resolve({ exists: false, unlocked: false, profiles: [] });
  await flushPromises();

  assert.ok(documentRef.querySelector(".vault-modal-overlay").querySelector("#vault-create-passphrase"));
});

test("vault control ignores stale refresh responses", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  const initial = deferred();
  const latest = deferred();
  const pending = [initial, latest];
  const changes = [];
  const fetchJson = async () => pending.shift().promise;

  const control = mountVaultControl(root, {
    fetchJson,
    onChange: (vault) => changes.push(vault),
    documentRef,
  });
  const latestRefresh = control.refresh();

  latest.resolve({
    exists: true,
    unlocked: true,
    profiles: [{ profile_id: "fresh-profile", scope: "target" }],
  });
  await latestRefresh;

  initial.resolve({ exists: false, unlocked: false, profiles: [] });
  await flushPromises();

  assert.equal(changes.length, 1);
  assert.equal(changes[0].profiles[0].profile_id, "fresh-profile");
  assert.equal(control.getState().profiles[0].profile_id, "fresh-profile");
  assert.equal(root.querySelector(".vault-sidebar-button").getAttribute("title"), "LLM Vault: Unlocked");
  assert.ok(root.querySelector(".vault-icon").className.includes("unlocked"));
});

test("vault control submits create payload and clears passphrase fields", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  const calls = [];
  const fetchJson = async (path, options = {}) => {
    calls.push({ path, options });
    if (path === "/api/llm-vault") return { exists: false, unlocked: false, profiles: [] };
    return { exists: true, unlocked: true, profiles: [] };
  };

  const control = mountVaultControl(root, { fetchJson, documentRef });
  await flushPromises();
  control.open();

  const overlay = documentRef.querySelector(".vault-modal-overlay");
  const passphrase = overlay.querySelector("#vault-create-passphrase");
  const confirm = overlay.querySelector("#vault-create-confirm");
  assert.equal(passphrase.getAttribute("autocomplete"), "new-password");
  assert.equal(confirm.getAttribute("autocomplete"), "new-password");
  passphrase.value = "vault secret";
  confirm.value = "vault secret";
  overlay.querySelector("form").dispatchEvent({ type: "submit", preventDefault() {} });
  await flushPromises();

  assert.equal(calls.at(-1).path, "/api/llm-vault/create");
  assert.equal(calls.at(-1).options.body, JSON.stringify({ passphrase: "vault secret" }));
  assert.equal(passphrase.value, "");
  assert.equal(confirm.value, "");
});

test("vault control submits unlock and lock actions", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  const calls = [];
  const fetchJson = async (path, options = {}) => {
    calls.push({ path, options });
    if (path === "/api/llm-vault") return { exists: true, unlocked: false, profiles: [] };
    if (path === "/api/llm-vault/unlock") {
      return { exists: true, unlocked: true, profiles: [{ profile_id: "target-profile", scope: "target" }] };
    }
    return { exists: true, unlocked: false, profiles: [] };
  };

  const control = mountVaultControl(root, { fetchJson, documentRef });
  await flushPromises();
  control.open();

  const overlay = documentRef.querySelector(".vault-modal-overlay");
  const passphrase = overlay.querySelector("#vault-unlock-passphrase");
  passphrase.value = "vault secret";
  overlay.querySelector("form").dispatchEvent({ type: "submit", preventDefault() {} });
  await flushPromises();

  assert.equal(calls.at(-1).path, "/api/llm-vault/unlock");
  assert.equal(calls.at(-1).options.body, JSON.stringify({ passphrase: "vault secret" }));
  assert.equal(passphrase.value, "");
  assert.equal(control.getState().unlocked, true);

  control.open();
  const lockButton = documentRef.querySelectorAll("button").find((button) => button.textContent === "Lock Vault");
  lockButton.dispatchEvent({ type: "click" });
  await flushPromises();

  assert.equal(calls.at(-1).path, "/api/llm-vault/lock");
  assert.equal(calls.at(-1).options.body, JSON.stringify({}));
  assert.equal(control.getState().unlocked, false);
});

test("vault control closes from overlay click and escape key", async () => {
  const documentRef = new FakeDocument();
  const root = documentRef.createElement("div");
  const fetchJson = async () => ({ exists: true, unlocked: false, profiles: [] });
  const control = mountVaultControl(root, { fetchJson, documentRef });
  await flushPromises();

  control.open();
  const overlay = documentRef.querySelector(".vault-modal-overlay");
  overlay.dispatchEvent({ type: "click", target: overlay });
  assert.equal(documentRef.querySelector(".vault-modal-overlay"), null);

  control.open();
  documentRef.querySelector(".vault-modal-overlay").dispatchEvent({ type: "keydown", key: "Escape" });
  assert.equal(documentRef.querySelector(".vault-modal-overlay"), null);
});

test("dashboard exposes vault control from sidebar footer", () => {
  assert.match(
    dashboardHtml,
    /<div id="sidebar-vault-root" class="sidebar-vault-root"/,
    "dashboard sidebar should mount the global vault control",
  );
});

test("dashboard run control no longer owns credential details", () => {
  const profileRowStart = dashboardHtml.indexOf('<div class="operator-profile-row"');
  assert.notEqual(profileRowStart, -1, "profile selector row should exist");
  const profileRowEnd = dashboardHtml.indexOf('<section\n                class="operator-profile-editor"', profileRowStart);
  assert.notEqual(profileRowEnd, -1, "shared profile editor should follow the profile row");
  const profileRow = dashboardHtml.slice(profileRowStart, profileRowEnd);
  const backingFieldsStart = dashboardHtml.indexOf('<div class="operator-backing-fields" hidden>');
  assert.notEqual(backingFieldsStart, -1, "hidden backing fields should exist");
  const backingFieldsEnd = dashboardHtml.indexOf("</form>", backingFieldsStart);
  assert.notEqual(backingFieldsEnd, -1, "hidden backing fields should remain inside the run form");
  const backingFields = dashboardHtml.slice(backingFieldsStart, backingFieldsEnd);

  assert.doesNotMatch(dashboardHtml, /<summary>Target LLM Config<\/summary>/, "target credential details should be removed");
  assert.doesNotMatch(dashboardHtml, /<summary>Attacker LLM \(for iterative techniques\)<\/summary>/, "attacker credential details should be removed");
  assert.doesNotMatch(dashboardHtml, /<summary>Custom Judge LLM credentials<\/summary>/, "judge credential details should be removed");
  assert.match(profileRow, /id="target-profile"/, "target saved profile selector should remain visible");
  assert.match(profileRow, /id="judge-profile"/, "judge saved profile selector should remain visible");
  assert.match(profileRow, /id="attacker-profile"/, "attacker saved profile selector should remain visible");
  assert.doesNotMatch(backingFields, /id="target-profile"/, "target selector should not move into hidden backing fields");
  assert.doesNotMatch(backingFields, /id="judge-profile"/, "judge selector should not move into hidden backing fields");
  assert.doesNotMatch(backingFields, /id="attacker-profile"/, "attacker selector should not move into hidden backing fields");
  assert.match(dashboardHtml, /id="operator-profile-editor"/, "dashboard should use one shared profile editor");
  for (const role of ["target", "judge", "attacker"]) {
    assert.match(dashboardHtml, new RegExp(`id="${role}-profile-edit-btn"`), `${role} profile should expose an edit action`);
  }
  assert.doesNotMatch(dashboardHtml, /id="save-target-profile-btn"/, "old target save button should be removed");
  assert.doesNotMatch(dashboardHtml, /id="save-attacker-profile-btn"/, "old attacker save button should be removed");
  assert.doesNotMatch(dashboardHtml, /id="save-judge-profile-btn"/, "old judge save button should be removed");
});

test("dashboard locked profile save opens vault from shared editor and skips profile POST", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls);
  try {
    await import(`../public/app.js?locked-profile-editor-save=${Date.now()}`);
    await flushPromises();

    const targetProfile = documentRef.querySelector("#target-profile");
    targetProfile.value = "__configure_new__";
    targetProfile.dispatchEvent({ type: "change" });
    documentRef.querySelector("#profile-editor-save-btn").dispatchEvent({ type: "click" });
    await flushPromises();

    assert.equal(documentRef.querySelector("#config-status").textContent, "Unlock vault before saving profiles");
    assert.ok(documentRef.querySelector(".vault-modal-overlay"), "locked save should open the vault dialog");
    assert.match(
      documentRef.querySelector(".vault-message").textContent,
      /Unlock the vault to save encrypted profile API keys\./,
    );
    assert.equal(fetchCalls.some(({ path }) => path === "/api/llm-profiles"), false);
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("dashboard empty profile cards expose configure buttons and open shared editor", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls);
  try {
    await import(`../public/app.js?empty-profile-card-actions=${Date.now()}`);
    await flushPromises();

    for (const [role, summary] of [
      ["target", "No target profile configured"],
      ["attacker", "No attacker profile configured"],
    ]) {
      assert.equal(documentRef.querySelector(`#${role}-profile-configure-btn`).hidden, false);
      assert.equal(documentRef.querySelector(`#${role}-profile-select-row`).hidden, true);
      assert.equal(documentRef.querySelector(`#${role}-profile-select-shell`).hidden, true);
      assert.equal(documentRef.querySelector(`#${role}-profile-add-btn`).hidden, true);
      assert.equal(documentRef.querySelector(`#${role}-profile-edit-btn`).hidden, true);
      assert.equal(documentRef.querySelector(`#operator-${role}-summary`).textContent, summary);
    }
    assert.equal(documentRef.querySelector("#judge-profile-configure-btn").hidden, true);
    assert.equal(documentRef.querySelector("#judge-profile-select-row").hidden, false);
    assert.equal(documentRef.querySelector("#judge-profile-select-shell").hidden, false);
    assert.equal(documentRef.querySelector("#judge-profile-add-btn").hidden, false);
    assert.equal(documentRef.querySelector("#judge-profile-edit-btn").hidden, true);
    assert.equal(documentRef.querySelector("#operator-judge-summary").textContent, "Using target LLM as judge");

    let previousRole = null;
    for (const [role, title] of [
      ["target", "New Target LLM Profile"],
      ["judge", "New Judge LLM Profile"],
      ["attacker", "New Attacker LLM Profile"],
    ]) {
      const actionId = role === "judge" ? "judge-profile-add-btn" : `${role}-profile-configure-btn`;
      documentRef.querySelector(`#${actionId}`).dispatchEvent({ type: "click" });

      assert.equal(documentRef.querySelector("#operator-profile-editor").hidden, false);
      assert.equal(documentRef.querySelector("#operator-profile-editor").dataset.role, role);
      assert.equal(documentRef.querySelector("#profile-editor-title").textContent, title);
      assert.equal(documentRef.querySelector(`#${role}-profile-configure-btn`).getAttribute("aria-expanded"), "true");
      if (previousRole) {
        assert.equal(
          documentRef.querySelector(`#${previousRole}-profile-configure-btn`).getAttribute("aria-expanded"),
          "false",
        );
      }
      previousRole = role;
    }
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("dashboard shows dropdown and add action for every role when profiles exist", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  const profiles = [{
    profile_id: "shared-profile",
    name: "Shared Cloud",
    provider: "openai",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    scope: "all",
  }];
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls, (path) => {
    if (path === "/api/llm-vault") return fakeResponse({ exists: true, unlocked: true, profiles });
    return fakeDashboardResponse(path);
  });
  try {
    await import(`../public/app.js?profile-card-dropdowns-for-all=${Date.now()}`);
    await flushPromises();
    await flushPromises();

    for (const role of ["target", "judge", "attacker"]) {
      assert.equal(documentRef.querySelector(`#${role}-profile`).value, "shared-profile");
      assert.equal(documentRef.querySelector(`#${role}-profile-configure-btn`).hidden, true);
      assert.equal(documentRef.querySelector(`#${role}-profile-select-row`).hidden, false);
      assert.equal(documentRef.querySelector(`#${role}-profile-select-shell`).hidden, false);
      assert.equal(documentRef.querySelector(`#${role}-profile-add-btn`).hidden, false);
      assert.equal(documentRef.querySelector(`#${role}-profile-edit-btn`).hidden, false);
      assert.match(documentRef.querySelector(`#operator-${role}-summary`).textContent, /Shared Cloud/);
    }
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("dashboard ignores stale saved profile ids when a vault profile exists", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  const profiles = [{
    profile_id: "fresh-profile",
    name: "Fresh Cloud",
    provider: "openai",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    scope: "all",
  }];
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls, (path) => {
    if (path === "/api/llm-vault") return fakeResponse({ exists: true, unlocked: true, profiles });
    return fakeDashboardResponse(path);
  });
  try {
    globalThis.localStorage.setItem("jailbreaker_target_config", JSON.stringify({
      target_profile_id: "deleted-target-profile",
      attacker_profile_id: "deleted-attacker-profile",
      judge_profile_id: "deleted-judge-profile",
      judge_mode: "custom",
    }));

    await import(`../public/app.js?stale-profile-fallback=${Date.now()}`);
    await flushPromises();
    await flushPromises();

    for (const role of ["target", "judge", "attacker"]) {
      assert.equal(documentRef.querySelector(`#${role}-profile`).value, "fresh-profile");
      assert.equal(documentRef.querySelector(`#${role}-profile-configure-btn`).hidden, true);
      assert.equal(documentRef.querySelector(`#${role}-profile-select-row`).hidden, false);
      assert.match(documentRef.querySelector(`#operator-${role}-summary`).textContent, /Fresh Cloud/);
    }
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("dashboard profile cards switch between empty and configured states", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  const profiles = [
    {
      profile_id: "target-profile",
      name: "Target Cloud",
      provider: "openai",
      base_url: "https://api.openai.com/v1",
      model: "gpt-4o-mini",
      scope: "target",
    },
    {
      profile_id: "judge-profile",
      name: "Judge Local",
      provider: "local",
      base_url: "http://localhost:1234/v1",
      model: "llama-3.1-8b",
      scope: "judge",
    },
  ];
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls, (path) => {
    if (path === "/api/llm-vault") return fakeResponse({ exists: true, unlocked: true, profiles });
    return fakeDashboardResponse(path);
  });
  try {
    globalThis.localStorage.setItem("jailbreaker_target_config", JSON.stringify({
      target_profile_id: "target-profile",
      judge_mode: "target",
      judge_profile_id: "",
    }));

    await import(`../public/app.js?configured-profile-card-states=${Date.now()}`);
    await flushPromises();
    await flushPromises();

    assert.equal(documentRef.querySelector("#target-profile-configure-btn").hidden, true);
    assert.equal(documentRef.querySelector("#target-profile-select-shell").hidden, false);
    assert.match(documentRef.querySelector("#operator-target-summary").textContent, /Target Cloud/);

    assert.equal(documentRef.querySelector("#judge-profile-configure-btn").hidden, true);
    assert.equal(documentRef.querySelector("#judge-profile-select-shell").hidden, false);
    assert.equal(documentRef.querySelector("#judge-profile-edit-btn").hidden, false);
    assert.equal(documentRef.querySelector("#judge-profile").value, "target-profile");
    assert.match(documentRef.querySelector("#operator-judge-summary").textContent, /Target Cloud/);

    const judgeProfile = documentRef.querySelector("#judge-profile");
    judgeProfile.value = "judge-profile";
    judgeProfile.dispatchEvent({ type: "change" });

    assert.equal(documentRef.querySelector("#judge-profile-configure-btn").hidden, true);
    assert.equal(documentRef.querySelector("#judge-profile-select-shell").hidden, false);
    assert.equal(documentRef.querySelector("#judge-profile-edit-btn").hidden, false);
    assert.match(documentRef.querySelector("#operator-judge-summary").textContent, /Judge Local/);

    judgeProfile.value = "__judge_target__";
    judgeProfile.dispatchEvent({ type: "change" });

    assert.equal(documentRef.querySelector("#judge-profile-configure-btn").hidden, true);
    assert.equal(documentRef.querySelector("#judge-profile-select-shell").hidden, false);
    assert.equal(documentRef.querySelector("#judge-profile-edit-btn").hidden, true);
    assert.equal(documentRef.querySelector("#operator-judge-summary").textContent, "Using target LLM as judge");
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("dashboard fans out an existing saved profile id to every role on load", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  const profiles = [{
    profile_id: "judge-only-profile",
    name: "Shared Judge",
    provider: "openai",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    scope: "judge",
  }];
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls, (path) => {
    if (path === "/api/llm-vault") return fakeResponse({ exists: true, unlocked: true, profiles });
    return fakeDashboardResponse(path);
  });
  try {
    globalThis.localStorage.setItem("jailbreaker_target_config", JSON.stringify({
      judge_profile_id: "judge-only-profile",
      judge_mode: "custom",
    }));

    await import(`../public/app.js?fan-out-existing-profile=${Date.now()}`);
    await flushPromises();
    await flushPromises();

    assert.equal(documentRef.querySelector("#target-profile").value, "judge-only-profile");
    assert.equal(documentRef.querySelector("#attacker-profile").value, "judge-only-profile");
    assert.equal(documentRef.querySelector("#judge-profile").value, "judge-only-profile");
    assert.equal(documentRef.querySelector("#target-profile-configure-btn").hidden, true);
    assert.equal(documentRef.querySelector("#attacker-profile-configure-btn").hidden, true);
    assert.equal(documentRef.querySelector("#judge-profile-configure-btn").hidden, true);
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("dashboard local shared editor saves profile once and selects it for every role", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  let savedProfile = null;
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls, (path, options = {}) => {
    if (path === "/api/llm-vault") {
      return fakeResponse({ exists: true, unlocked: true, profiles: savedProfile ? [savedProfile] : [] });
    }
    if (path === "/api/llm-profiles") {
      const payload = JSON.parse(options.body);
      savedProfile = {
        profile_id: "local-target-profile",
        name: payload.name,
        provider: payload.provider,
        base_url: payload.base_url,
        model: payload.model,
        scope: payload.scope,
      };
      return fakeResponse({ item: savedProfile });
    }
    return fakeDashboardResponse(path);
  });
  try {
    await import(`../public/app.js?local-shared-editor-save=${Date.now()}`);
    await flushPromises();

    const targetProfile = documentRef.querySelector("#target-profile");
    targetProfile.value = "__configure_new__";
    targetProfile.dispatchEvent({ type: "change" });

    const baseUrl = documentRef.querySelector("#profile-editor-base-url");
    const provider = documentRef.querySelector("#profile-editor-provider");
    assert.equal(baseUrl.value, "https://api.openai.com/v1", "target editor starts from the hidden target default");
    provider.value = "openai";
    documentRef.querySelector("#profile-editor-system-prompt").value = "Run-specific target prompt.";

    const localToggle = documentRef.querySelector("#profile-editor-local");
    localToggle.checked = true;
    localToggle.dispatchEvent({ type: "change" });

    assert.equal(baseUrl.value, "http://localhost:1234/v1");
    assert.equal(provider.value, "local");

    documentRef.querySelector("#profile-editor-save-btn").dispatchEvent({ type: "click" });
    await flushPromises();
    await flushPromises();

    const profilePost = fetchCalls.find(({ path }) => path === "/api/llm-profiles");
    assert.ok(profilePost, "unlocked local editor save should post a profile");
    const body = JSON.parse(profilePost.options.body);
    assert.equal(body.provider, "local");
    assert.equal(body.base_url, "http://localhost:1234/v1");
    assert.equal(body.api_key, "lm-studio");
    assert.equal(Object.hasOwn(body, "system_prompt"), false, "profiles should never receive run prompts");
    assert.equal(body.scope, "all");
    assert.equal(documentRef.querySelector("#target-profile").value, "local-target-profile");
    assert.equal(documentRef.querySelector("#attacker-profile").value, "local-target-profile");
    assert.equal(documentRef.querySelector("#judge-profile").value, "local-target-profile");
    assert.equal(documentRef.querySelector("#judge-mode").value, "custom");
    assert.deepEqual(
      JSON.parse(globalThis.localStorage.getItem("jailbreaker_target_config")),
      {
        target_local: true,
        target_profile_id: "local-target-profile",
        api_base_url: "http://localhost:1234/v1",
        model: "gpt-4o-mini",
        system_prompt: "Run-specific target prompt.",
        unsafe_intent: "test intent",
        attacker_local: true,
        attacker_profile_id: "local-target-profile",
        attacker_api_base_url: "http://localhost:1234/v1",
        attacker_model: "gpt-4o-mini",
        attacker_system_prompt: "You are a helpful assistant.",
        judge_mode: "custom",
        judge_local: true,
        judge_profile_id: "local-target-profile",
        judge_api_base_url: "http://localhost:1234/v1",
        judge_model: "gpt-4o-mini",
        judge_system_prompt: "",
        cs_system_prompt: false,
      },
      "saving one profile should persist all three role profile ids",
    );
    assert.equal(documentRef.querySelector("#operator-profile-editor").hidden, true);
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("dashboard can edit a selected saved profile", async () => {
  const documentRef = new FakeDocument();
  const fetchCalls = [];
  const profiles = [{
    profile_id: "shared-profile",
    name: "Shared Cloud",
    provider: "openai",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    scope: "all",
  }];
  const previousGlobals = installDashboardGlobals(documentRef, fetchCalls, (path, options = {}) => {
    if (path === "/api/llm-vault") return fakeResponse({ exists: true, unlocked: true, profiles });
    if (path === "/api/llm-profiles") {
      const payload = JSON.parse(options.body);
      profiles[0] = { ...profiles[0], ...payload };
      return fakeResponse({ item: { ...profiles[0], profile_id: "shared-profile" } });
    }
    return fakeDashboardResponse(path);
  });
  try {
    globalThis.localStorage.setItem("jailbreaker_target_config", JSON.stringify({
      target_profile_id: "shared-profile",
    }));
    await import(`../public/app.js?edit-selected-profile=${Date.now()}`);
    await flushPromises();
    await flushPromises();

    documentRef.querySelector("#target-profile-edit-btn").dispatchEvent({ type: "click" });

    assert.equal(documentRef.querySelector("#operator-profile-editor").hidden, false);
    assert.equal(documentRef.querySelector("#profile-editor-title").textContent, "Edit Target LLM Profile");
    assert.equal(documentRef.querySelector("#profile-editor-name").value, "Shared Cloud");
    assert.equal(documentRef.querySelector("#profile-editor-model").value, "gpt-4o-mini");
    assert.equal(documentRef.querySelector("#profile-editor-system-prompt").value, "You are a helpful assistant.");
    assert.equal(documentRef.querySelector("#profile-editor-api-key").value, "");

    documentRef.querySelector("#profile-editor-system-prompt").value = "Run-specific target prompt.";
    documentRef.querySelector("#profile-editor-name").value = "Shared Edited";
    documentRef.querySelector("#profile-editor-save-btn").dispatchEvent({ type: "click" });
    await flushPromises();
    await flushPromises();

    const profilePost = fetchCalls.find(({ path }) => path === "/api/llm-profiles");
    const body = JSON.parse(profilePost.options.body);
    assert.equal(body.profile_id, "shared-profile");
    assert.equal(body.name, "Shared Edited");
    assert.equal(Object.hasOwn(body, "system_prompt"), false);
    assert.equal(body.api_key, "");
    assert.equal(documentRef.querySelector("#system-prompt").value, "Run-specific target prompt.");
  } finally {
    restoreDashboardGlobals(previousGlobals);
  }
});

test("experiments and analysis expose sidebar vault mount points", () => {
  assert.match(
    experimentsHtml,
    /<div id="sidebar-vault-root" class="sidebar-vault-root"/,
    "experiments sidebar should mount the vault control",
  );
  assert.match(
    analysisHtml,
    /<div id="sidebar-vault-root" class="sidebar-vault-root"/,
    "analysis sidebar should mount the vault control",
  );
});

test("experiments and analysis mount shared vault control", () => {
  assert.match(experimentsScript, /import\s+\{\s*mountVaultControl\s*\}/, "experiments should import mountVaultControl");
  assert.match(experimentsScript, /mountVaultControl\(el\.sidebarVaultRoot/, "experiments should mount the vault control");
  assert.match(analysisScript, /import\s+\{\s*mountVaultControl\s*\}/, "analysis should import mountVaultControl");
  assert.match(analysisScript, /mountVaultControl\(el\.sidebarVaultRoot/, "analysis should mount the vault control");
});

test("sidebar footers omit environment text", () => {
  for (const [name, html] of [
    ["dashboard", dashboardHtml],
    ["experiments", experimentsHtml],
    ["analysis", analysisHtml],
  ]) {
    assert.doesNotMatch(html, /class="sidebar-env"/, `${name} sidebar should not show environment text`);
    assert.doesNotMatch(html, />development<\/span>/, `${name} sidebar should not show development label`);
  }
});

function ruleBody(selector, source = css) {
  const pattern = new RegExp(`${escapeRegExp(selector)}\\s*\\{([^}]*)\\}`, "m");
  const match = source.match(pattern);
  assert.ok(match, `Expected to find CSS rule for ${selector}`);
  return match[1];
}

function mediaBlock(query) {
  const start = css.indexOf(query);
  assert.notEqual(start, -1, `Expected to find ${query}`);
  const open = css.indexOf("{", start);
  let depth = 0;
  for (let index = open; index < css.length; index += 1) {
    if (css[index] === "{") depth += 1;
    if (css[index] === "}") depth -= 1;
    if (depth === 0) return css.slice(open + 1, index);
  }
  assert.fail(`Expected ${query} block to close`);
}

function hasDeclaration(body, property, valuePattern) {
  return new RegExp(`${property}\\s*:\\s*${valuePattern}\\s*;`).test(body);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
    this.body = new FakeElement("body", this);
    this.elements = new Map();
    this.listeners = new Map();
  }

  createElement(tagName) {
    return new FakeElement(tagName, this);
  }

  createTextNode(text) {
    const node = new FakeElement("#text", this);
    node.textContent = String(text ?? "");
    return node;
  }

  querySelector(selector) {
    if (selector.startsWith("#")) {
      const id = selector.slice(1);
      if (!this.elements.has(id)) {
        const element = this.createElement(defaultTagForId(id));
        element.setAttribute("id", id);
        seedDashboardElement(id, element);
        this.elements.set(id, element);
        this.body.append(element);
      }
      return this.elements.get(id);
    }
    return this.body.querySelector(selector);
  }

  querySelectorAll(selector) {
    return this.body.querySelectorAll(selector);
  }

  addEventListener(type, handler) {
    const handlers = this.listeners.get(type) || [];
    handlers.push(handler);
    this.listeners.set(type, handlers);
  }
}

class FakeElement {
  constructor(tagName, documentRef) {
    this.tagName = tagName.toLowerCase();
    this.documentRef = documentRef;
    this.children = [];
    this.parent = null;
    this.attributes = new Map();
    this.listeners = new Map();
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.id = "";
    this.type = "";
    this.value = "";
    this.textContent = "";
    this.tabIndex = 0;
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this.required = false;
  }

  get classList() {
    return {
      add: (...tokens) => {
        const classes = new Set(this.className.split(/\s+/).filter(Boolean));
        tokens.forEach((token) => classes.add(token));
        this.className = [...classes].join(" ");
      },
      remove: (...tokens) => {
        const removals = new Set(tokens);
        this.className = this.className.split(/\s+/).filter((token) => token && !removals.has(token)).join(" ");
      },
      toggle: (token, force) => {
        const classes = new Set(this.className.split(/\s+/).filter(Boolean));
        const shouldAdd = force === undefined ? !classes.has(token) : Boolean(force);
        if (shouldAdd) classes.add(token);
        else classes.delete(token);
        this.className = [...classes].join(" ");
        return shouldAdd;
      },
      contains: (token) => this.className.split(/\s+/).includes(token),
    };
  }

  append(...children) {
    for (const child of children.flat()) {
      if (!child) continue;
      child.parent = this;
      this.children.push(child);
    }
  }

  replaceChildren(...children) {
    for (const child of this.children) {
      child.parent = null;
    }
    this.children = [];
    this.append(...children);
  }

  remove() {
    if (!this.parent) return;
    this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
    if (name === "id") this.id = String(value);
    if (name === "class") this.className = String(value);
    if (name.startsWith("data-")) {
      this.dataset[dataKey(name)] = String(value);
    }
  }

  getAttribute(name) {
    return this.attributes.get(name) || null;
  }

  addEventListener(type, handler) {
    const handlers = this.listeners.get(type) || [];
    handlers.push(handler);
    this.listeners.set(type, handlers);
  }

  dispatchEvent(event) {
    event.target = event.target || this;
    for (const handler of this.listeners.get(event.type) || []) {
      handler(event);
    }
  }

  focus() {
    this.documentRef.activeElement = this;
  }

  querySelector(selector) {
    return querySelector(this, selector);
  }

  querySelectorAll(selector) {
    return descendants(this).filter((item) => matches(item, selector));
  }
}

function installDashboardGlobals(documentRef, fetchCalls, responder = fakeDashboardResponse) {
  const previous = {
    document: globalThis.document,
    window: globalThis.window,
    fetch: globalThis.fetch,
    localStorage: globalThis.localStorage,
    sessionStorage: globalThis.sessionStorage,
    Event: globalThis.Event,
  };
  const storage = new FakeStorage();
  globalThis.document = documentRef;
  globalThis.window = { __JAILBREAKER_API_BASE_URL__: "" };
  globalThis.localStorage = storage;
  globalThis.sessionStorage = storage;
  globalThis.Event = class {
    constructor(type) {
      this.type = type;
    }
  };
  globalThis.fetch = async (path, options = {}) => {
    fetchCalls.push({ path, options });
    return responder(path, options);
  };
  return previous;
}

function restoreDashboardGlobals(previous) {
  for (const [key, value] of Object.entries(previous)) {
    if (value === undefined) delete globalThis[key];
    else globalThis[key] = value;
  }
}

function fakeDashboardResponse(path) {
  const payloads = {
    "/api/dashboard": { surfaces: {} },
    "/api/techniques": { items: [] },
    "/api/custom-techniques": { items: [] },
    "/api/llm-vault": { exists: true, unlocked: false, profiles: [] },
  };
  return fakeResponse(payloads[path] || {});
}

function fakeResponse(payload, { ok = true } = {}) {
  return {
    ok,
    json: async () => payload,
  };
}

class FakeStorage {
  constructor() {
    this.values = new Map();
  }

  getItem(key) {
    return this.values.get(key) || null;
  }

  setItem(key, value) {
    this.values.set(key, String(value));
  }
}

function querySelector(root, selector) {
  const parts = selector.trim().split(/\s+/);
  let candidates = [root];
  for (const part of parts) {
    const next = [];
    for (const candidate of candidates) {
      next.push(...descendants(candidate).filter((item) => matches(item, part)));
    }
    if (next.length === 0) return null;
    candidates = next;
  }
  return candidates[0] || null;
}

function descendants(root) {
  const items = [];
  for (const child of root.children) {
    items.push(child, ...descendants(child));
  }
  return items;
}

function matches(element, selector) {
  if (selector.includes("[")) {
    const tag = selector.split("[")[0];
    return tag ? element.tagName === tag.toLowerCase() : true;
  }
  if (selector.startsWith(".")) {
    return element.className.split(/\s+/).includes(selector.slice(1));
  }
  if (selector.startsWith("#")) {
    return element.id === selector.slice(1);
  }
  return element.tagName === selector.toLowerCase();
}

function defaultTagForId(id) {
  if (id === "operator-profile-editor") return "section";
  if (id.includes("btn") || id.includes("button") || id.includes("close") || id.includes("reset")) return "button";
  if (id.includes("form")) return "form";
  if (id.includes("local") || id.includes("toggle")) return "input";
  if (
    id.includes("api-key") ||
    id.includes("base-url") ||
    id.includes("model") ||
    id.includes("profile-editor-name") ||
    id.includes("profile-editor-provider")
  ) return "input";
  if (id.endsWith("-select-row")) return "div";
  if (id.endsWith("-select-shell")) return "div";
  if (id.includes("profile") || id.includes("mode")) return "select";
  if (id.includes("prompt") || id.includes("template") || id.includes("summary")) return "textarea";
  return "div";
}

function seedDashboardElement(id, element) {
  const values = {
    "api-base-url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api-key": "sk-test",
    "attacker-api-base-url": "",
    "attacker-model": "",
    "attacker-api-key": "",
    "judge-mode": "target",
    "judge-api-base-url": "",
    "judge-model": "",
    "judge-api-key": "",
    "system-prompt": "You are a helpful assistant.",
    "attacker-system-prompt": "You are a helpful assistant.",
    "judge-system-prompt": "",
    "unsafe-intent": "test intent",
    "custom-technique-mode": "single",
    "custom-max-rounds": "5",
  };
  if (Object.prototype.hasOwnProperty.call(values, id)) {
    element.value = values[id];
  }
}

function dataKey(attributeName) {
  return attributeName
    .slice(5)
    .replace(/-([a-z])/g, (_, char) => char.toUpperCase());
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function flushPromises() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}
