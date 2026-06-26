// ---------------------------------------------------------------------------
// Jailbreaker — Experiments page
// ---------------------------------------------------------------------------

import { appendPresent, domElement as conversationElement, textNodeElement } from "./components/dom.js";
import { fillProfileSelect, hasSecretConfigKeys, selectedProfileFromList, stripSecretConfigKeys } from "./components/profile-config.js";
import { mountSidebarNav } from "./components/sidebar-nav.js";
import { mountVaultControl } from "./components/vault-control.js";

const API = (() => {
  const loc = window.location;
  return `${loc.protocol}//${loc.hostname}:${loc.port}`;
})();

const EXPERIMENT_JUDGE_TARGET_VALUE = "__judge_target__";
const EXPERIMENT_JUDGE_ATTACKER_VALUE = "__judge_attacker__";

// ---------------------------------------------------------------------------
// State + Persistence
// ---------------------------------------------------------------------------

const STATE_KEY = "jailbreaker_experiment_state";

const state = {
  techniques: [],
  datasets: [],
  experiments: [],
  currentExperiment: null,
  selectedDatasetId: null,
  pollTimer: null,
  activeTab: "design",
  // Persisted form fields
	  formName: "",
	  formDescription: "",
	  formHypothesis: "",
	  formGroup: "",
	  deselectedTechniques: [],  // store deselected (smaller than selected)
  currentExperimentId: null, // to restore results view
  vault: { exists: false, unlocked: false, profiles: [] },
};

const EXPERIMENT_NAV_ITEMS = [
  { label: "Dashboard", href: "/" },
  {
    label: "Experiments",
    href: "/experiments/",
    active: true,
    ariaCurrent: "page",
    children: [
      { id: "design", label: "Design", href: "/experiments/#design" },
      { id: "results", label: "Results", href: "/experiments/#results" },
      { id: "history", label: "History", href: "/experiments/#history" },
    ],
  },
  { label: "History", href: "/history/" },
  { label: "Analysis", href: "/analysis/" },
];

const EXPERIMENT_PANEL_IDS = new Set(["design", "results", "history"]);

function normalizeExperimentTab(tab) {
  if (tab === "data" || tab === "metrics") return "results";
  if (EXPERIMENT_PANEL_IDS.has(tab)) return tab;
  return "design";
}

function saveState() {
  const toSave = {
    activeTab: state.activeTab,
    selectedDatasetId: state.selectedDatasetId,
	    formName: el.expName?.value || "",
	    formDescription: el.expDescription?.value || "",
	    formHypothesis: el.expHypothesis?.value || "",
	    formGroup: el.expGroup?.value || "",
    repeatCount: el.repeatCount?.value || "1",
	    maxRounds: el.maxRounds?.value || "5",
    targetConfig: getStoredTargetConfig(),
    deselectedTechniques: getDeselectedTechniques(),
    expandedGroups: [...expandedGroups],
    currentExperimentId: state.currentExperiment?.experiment_id || state.currentExperimentId || null,
  };
  try {
    localStorage.setItem(STATE_KEY, JSON.stringify(toSave));
  } catch { /* quota */ }
}

function loadSavedState() {
  try {
    const raw = localStorage.getItem(STATE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const sanitized = sanitizeExperimentState(parsed);
    if (hasSecretConfigKeys(parsed?.targetConfig)) {
      localStorage.setItem(STATE_KEY, JSON.stringify(sanitized));
    }
    return sanitized;
  } catch {
    return null;
  }
}

function sanitizeExperimentState(saved) {
  const sanitized = { ...(saved || {}) };
  if (sanitized.targetConfig) {
    sanitized.targetConfig = stripSecretConfigKeys({ ...sanitized.targetConfig });
  }
  return sanitized;
}

function getDeselectedTechniques() {
  const all = el.techniqueSelector.querySelectorAll(".exp-technique-cb");
  const deselected = [];
  all.forEach((cb) => { if (!cb.checked) deselected.push(cb.value); });
  return deselected;
}

function restoreFormState(saved) {
  const restored = saved || {};
	  if (restored.formName) el.expName.value = restored.formName;
	  if (restored.formDescription) el.expDescription.value = restored.formDescription;
	  if (restored.formHypothesis) el.expHypothesis.value = restored.formHypothesis;
	  if (restored.formGroup) el.expGroup.value = restored.formGroup;
	  if (restored.repeatCount) el.repeatCount.value = restored.repeatCount;
	  if (restored.maxRounds) el.maxRounds.value = restored.maxRounds;
  if (restored.selectedDatasetId) {
    state.selectedDatasetId = restored.selectedDatasetId;
  }
  if (restored.expandedGroups) {
    expandedGroups.clear();
    for (const g of restored.expandedGroups) expandedGroups.add(g);
  }
  state.currentExperimentId = restored.currentExperimentId || null;
  const hashTab = window.location.hash ? window.location.hash.slice(1) : "";
  const requestedTab = hashTab || restored.activeTab;
  if (requestedTab) {
    switchTab(requestedTab);
  }
}

function restoreTechniqueSelections(saved) {
  if (!saved || !saved.deselectedTechniques || saved.deselectedTechniques.length === 0) return;
  const deselected = new Set(saved.deselectedTechniques);
  el.techniqueSelector.querySelectorAll(".exp-technique-cb").forEach((cb) => {
    if (deselected.has(cb.value)) cb.checked = false;
  });
  // Sync group checkboxes
  el.techniqueSelector.querySelectorAll(".technique-group[data-group-key]").forEach(syncGroupCheckbox);
  updateTechniqueCount();
}

// Auto-save on form changes
function wireAutoSave() {
	  [el.expName, el.expDescription, el.expHypothesis, el.expGroup, el.repeatCount, el.maxRounds].forEach((input) => {
	    if (input) input.addEventListener("input", saveState);
	  });
}

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const el = {
  sidebarNavRoot: document.getElementById("sidebar-nav-root"),
  sidebarVaultRoot: document.getElementById("sidebar-vault-root"),
  panels: document.querySelectorAll(".experiment-panel"),
  // Design form
  form: document.getElementById("experiment-form"),
	  expName: document.getElementById("exp-name"),
	  expDescription: document.getElementById("exp-description"),
	  expHypothesis: document.getElementById("exp-hypothesis"),
	  expGroup: document.getElementById("exp-group"),
	  repeatCount: document.getElementById("exp-repeat-count"),
	  maxRounds: document.getElementById("exp-max-rounds"),
  // Dataset
  dsName: document.getElementById("ds-name"),
  dsPrompts: document.getElementById("ds-prompts"),
  uploadBtn: document.getElementById("upload-dataset-btn"),
  dsUploadStatus: document.getElementById("ds-upload-status"),
  datasetList: document.getElementById("dataset-list"),
  datasetSelected: document.getElementById("dataset-selected"),
  selectedDatasetTag: document.getElementById("selected-dataset-tag"),
  dsTabs: document.querySelectorAll(".dataset-tab"),
  dsPanels: document.querySelectorAll(".dataset-panel"),
  // Techniques
  techniqueSelector: document.getElementById("technique-selector"),
  selectAllBtn: document.getElementById("select-all-btn"),
  deselectAllBtn: document.getElementById("deselect-all-btn"),
  techniqueCount: document.getElementById("technique-count"),
  // Target config
  apiKey: document.getElementById("exp-api-key"),
  targetProfile: document.getElementById("exp-target-profile"),
  apiBaseUrl: document.getElementById("exp-api-base-url"),
  model: document.getElementById("exp-model"),
  systemPrompt: document.getElementById("exp-system-prompt"),
  attackerProfile: document.getElementById("exp-attacker-profile"),
  attackerApiKey: document.getElementById("exp-attacker-api-key"),
  attackerBaseUrl: document.getElementById("exp-attacker-base-url"),
  attackerModel: document.getElementById("exp-attacker-model"),
  judgeMode: document.getElementById("exp-judge-mode"),
  judgeCustomFields: document.getElementById("exp-judge-custom-fields"),
  judgeLocal: document.getElementById("exp-judge-local"),
  judgeProfile: document.getElementById("exp-judge-profile"),
  judgeKeyField: document.getElementById("exp-judge-key-field"),
  judgeApiKey: document.getElementById("exp-judge-api-key"),
  judgeBaseUrl: document.getElementById("exp-judge-base-url"),
  judgeModel: document.getElementById("exp-judge-model"),
  runBtn: document.getElementById("run-experiment-btn"),
  // (progress bars are now inline in History tab)
  // Results
  results: document.getElementById("exp-results"),
  resultsTitle: document.getElementById("results-title"),
  resultsMeta: document.getElementById("results-meta"),
  postureSummary: document.getElementById("exp-posture-summary"),
  severityBar: document.getElementById("exp-severity-bar"),
  severityLegend: document.getElementById("exp-severity-legend"),
  resultsRiskList: document.getElementById("results-risk-list"),
  resultsEvidenceList: document.getElementById("results-evidence-list"),
  exportBar: document.getElementById("exp-export-bar"),
  matrixWrap: document.getElementById("matrix-wrap"),
  exportCsvBtn: document.getElementById("export-csv-btn"),
  exportJsonBtn: document.getElementById("export-json-btn"),
  // History
  historyList: document.getElementById("experiment-history"),
  // Results empty message
  resultsEmpty: document.getElementById("exp-results-empty"),
  // Metrics
  metricsExpSelect: document.getElementById("metrics-experiment-select"),
  loadMetricsBtn: document.getElementById("load-metrics-btn"),
  downloadFullCsvBtn: document.getElementById("download-full-csv-btn"),
  downloadMetricsCsvBtn: document.getElementById("download-metrics-csv-btn"),
  metricsBody: document.getElementById("metrics-body"),
  // Modal
  modalOverlay: document.getElementById("modal-overlay"),
  modalTitle: document.getElementById("modal-title"),
  modalBody: document.getElementById("modal-body"),
  modalClose: document.getElementById("modal-close"),
  toggleAttacker: document.getElementById("toggle-attacker"),
  toggleRawInput: document.getElementById("toggle-raw-input"),
};

let vaultControl = null;

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function fetchJson(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

async function loadVaultState() {
  if (vaultControl) {
    await vaultControl.refresh();
    return;
  }
  try {
    handleVaultChange(await fetchJson("/api/llm-vault"));
  } catch {
    handleVaultChange({ exists: false, unlocked: false, profiles: [] });
  }
}

function handleVaultChange(vault) {
  state.vault = vault || { exists: false, unlocked: false, profiles: [] };
  renderProfileSelectors();
}

function renderProfileSelectors() {
  const profiles = state.vault.profiles || [];
  fillProfileSelect(el.targetProfile, profiles, "target", el.targetProfile?.value || "");
  fillProfileSelect(el.attackerProfile, profiles, "attacker", el.attackerProfile?.value || "");
  fillProfileSelect(el.judgeProfile, profiles, "judge", currentJudgeProfileSelectValue(), {
    beforeOptions: [
      { value: EXPERIMENT_JUDGE_TARGET_VALUE, label: "Use Target LLM" },
      { value: EXPERIMENT_JUDGE_ATTACKER_VALUE, label: "Use Attacker LLM" },
    ],
  });
}

function selectedProfile(select) {
  return selectedProfileFromList(state.vault.profiles || [], select);
}

function applySelectedProfile(purpose) {
  if (purpose === "target") {
    const profile = selectedProfile(el.targetProfile);
    if (profile) {
      el.apiBaseUrl.value = profile.base_url || el.apiBaseUrl.value;
      el.model.value = profile.model || el.model.value;
    }
  }
  if (purpose === "attacker") {
    const profile = selectedProfile(el.attackerProfile);
    if (profile) {
      el.attackerBaseUrl.value = profile.base_url || el.attackerBaseUrl.value;
      el.attackerModel.value = profile.model || el.attackerModel.value;
    }
  }
  if (purpose === "judge") {
    const value = el.judgeProfile?.value || "";
    if (value === EXPERIMENT_JUDGE_TARGET_VALUE || value === EXPERIMENT_JUDGE_ATTACKER_VALUE) {
      el.judgeMode.value = value === EXPERIMENT_JUDGE_ATTACKER_VALUE ? "attacker" : "target";
      updateJudgeConfigVisibility();
      saveState();
      return;
    }
    const profile = selectedProfile(el.judgeProfile);
    if (profile) {
      el.judgeBaseUrl.value = profile.base_url || el.judgeBaseUrl.value;
      el.judgeModel.value = profile.model || el.judgeModel.value;
      el.judgeMode.value = "custom";
      updateJudgeConfigVisibility();
    }
  }
  saveState();
}

function currentJudgeProfileSelectValue() {
  const value = el.judgeProfile?.value || "";
  if (value && !isExperimentJudgeSourceValue(value)) return value;
  if (el.judgeMode?.value === "attacker") return EXPERIMENT_JUDGE_ATTACKER_VALUE;
  if (el.judgeMode?.value === "target") return EXPERIMENT_JUDGE_TARGET_VALUE;
  return value;
}

function profileIdFromSelect(select) {
  const value = select?.value || "";
  return isExperimentJudgeSourceValue(value) ? "" : value;
}

function isExperimentJudgeSourceValue(value) {
  return value === EXPERIMENT_JUDGE_TARGET_VALUE || value === EXPERIMENT_JUDGE_ATTACKER_VALUE;
}

function syncJudgeProfileSelectionFromMode() {
  if (!el.judgeProfile) return;
  if (el.judgeMode.value === "target" && selectHasValue(el.judgeProfile, EXPERIMENT_JUDGE_TARGET_VALUE)) {
    el.judgeProfile.value = EXPERIMENT_JUDGE_TARGET_VALUE;
  } else if (el.judgeMode.value === "attacker" && selectHasValue(el.judgeProfile, EXPERIMENT_JUDGE_ATTACKER_VALUE)) {
    el.judgeProfile.value = EXPERIMENT_JUDGE_ATTACKER_VALUE;
  } else if (el.judgeMode.value === "custom" && isExperimentJudgeSourceValue(el.judgeProfile.value)) {
    el.judgeProfile.value = "";
  }
}

function selectHasValue(select, value) {
  return [...(select?.children || [])].some((option) => option.value === value);
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function switchTab(tab) {
  const nextTab = normalizeExperimentTab(tab);
  state.activeTab = nextTab;
  el.panels.forEach((p) => {
    const match = p.dataset.panel === nextTab;
    p.classList.toggle("active", match);
    p.hidden = !match;
  });
  renderExperimentSidebarNav();
  if (nextTab === "results" || nextTab === "history") { loadExperiments(); _ensureRunningPoll(); }
  saveState();
}

function renderExperimentSidebarNav() {
  mountSidebarNav(el.sidebarNavRoot, {
    items: EXPERIMENT_NAV_ITEMS,
    activePath: "/experiments/",
    activeChild: state.activeTab,
    onNavigate(item, event) {
      if (!item.id) return;
      event.preventDefault();
      switchTab(item.id);
    },
  });
}

// Dataset sub-tabs
el.dsTabs.forEach((t) =>
  t.addEventListener("click", () => {
    el.dsTabs.forEach((b) => b.classList.toggle("active", b === t));
    el.dsPanels.forEach((p) => {
      const match = p.dataset.dsPanel === t.dataset.dsTab;
      p.classList.toggle("active", match);
      p.hidden = !match;
    });
  })
);

// ---------------------------------------------------------------------------
// Technique display groups (mirrors app.js mapping)
// ---------------------------------------------------------------------------

const DISPLAY_GROUP_LABELS = {
  custom: "Custom Techniques",
  jailbreak: "Jailbreak Attacks",
  injection: "Prompt Injection & Leakage",
  agent: "Agent & Multi-Agent Attacks",
  "customer-service": "Customer Service Testing",
};

const DISPLAY_GROUP_ORDER = [
  "custom",
  "customer-service",
  "jailbreak",
  "injection",
  "agent",
];

const FAMILY_TO_GROUP = {
  "custom": "custom",
  "human-based": "jailbreak",
  "obfuscation-based": "jailbreak",
  "heuristic-based": "jailbreak",
  "feedback-based": "jailbreak",
  "generation-parameter-based": "jailbreak",
  "scenario-nesting": "jailbreak",
  "instruction-override": "injection",
  "prompt-leakage": "injection",
  "context-exploitation": "injection",
  "agent-regime": "agent",
  "agent-capability-abuse": "agent",
  "stateful-agent-attack": "agent",
  "multi-agent-propagation": "agent",
  "multi-agent-reconnaissance": "agent",
  "multi-agent-multimodal": "agent",
  "manual-black-box": "jailbreak",
  "code-agent": "agent",
  "customer-service": "customer-service",
};

// ---------------------------------------------------------------------------
// Load techniques
// ---------------------------------------------------------------------------

async function loadTechniques() {
  try {
    const data = await fetchJson("/api/techniques");
    state.techniques = data.items || [];
    renderTechniqueSelector();
  } catch (err) {
    console.error("Failed to load techniques:", err);
  }
}

// Track which groups are expanded (all start collapsed)
const expandedGroups = new Set();

function renderTechniqueSelector() {
  const container = el.techniqueSelector;
  container.replaceChildren();

  // Group techniques by display group
  const grouped = {};
  for (const technique of state.techniques) {
    const group = FAMILY_TO_GROUP[technique.family] || "jailbreak";
    (grouped[group] = grouped[group] || []).push(technique);
  }

  container.append(renderBaselineTechniqueGroup());

  // Technique groups - reuse dashboard's .technique-group structure
  for (const groupKey of DISPLAY_GROUP_ORDER) {
    const techniques = grouped[groupKey];
    if (!techniques || techniques.length === 0) continue;
    const label = DISPLAY_GROUP_LABELS[groupKey] || groupKey;
    const isExpanded = expandedGroups.has(groupKey);

    const groupEl = document.createElement("div");
    groupEl.className = "technique-group";
    if (!isExpanded) groupEl.classList.add("collapsed");
    groupEl.dataset.groupKey = groupKey;

    const body = document.createElement("div");
    body.className = "technique-group-body";
    if (!isExpanded) body.style.display = "none";

    const groupHeader = renderTechniqueGroupHeader(groupKey, label, techniques.length);
    groupHeader.toggle.addEventListener("click", (e) => {
      if (e.target.tagName === "INPUT") return;
      if (expandedGroups.has(groupKey)) {
        expandedGroups.delete(groupKey);
        groupEl.classList.add("collapsed");
        body.style.display = "none";
      } else {
        expandedGroups.add(groupKey);
        groupEl.classList.remove("collapsed");
        body.style.display = "";
      }
    });

    groupHeader.checkbox.addEventListener("change", (e) => {
      e.stopPropagation();
      const checked = e.target.checked;
      body.querySelectorAll(".exp-technique-cb").forEach((cb) => (cb.checked = checked));
      updateTechniqueCount();
    });

    groupEl.append(groupHeader.element);

    for (const technique of techniques) {
      body.append(renderExperimentTechniqueRow(technique, groupEl));
    }

    groupEl.append(body);
    container.append(groupEl);
  }

  updateTechniqueCount();
}

function renderBaselineTechniqueGroup() {
  const group = document.createElement("div");
  group.className = "technique-group exp-baseline-group";

  const header = document.createElement("div");
  header.className = "technique-group-header exp-baseline-header";

  const toggle = document.createElement("div");
  toggle.className = "group-toggle";

  const checkbox = experimentCheckbox("exp-technique-cb", "__baseline__", true);
  checkbox.addEventListener("change", updateTechniqueCount);

  toggle.append(
    checkbox,
    textNodeElement("span", "group-label", "Baseline (Control)"),
    textNodeElement("span", "group-stats", "No jailbreak — raw prompt sent to target"),
  );
  header.append(toggle);
  group.append(header);
  return group;
}

function renderTechniqueGroupHeader(groupKey, label, techniqueCount) {
  const header = document.createElement("div");
  header.className = "technique-group-header";

  const toggle = document.createElement("div");
  toggle.className = "group-toggle";
  const checkbox = experimentCheckbox("exp-group-cb", "", true);
  checkbox.dataset.group = groupKey;

  toggle.append(
    textNodeElement("span", "group-chevron", ""),
    checkbox,
    textNodeElement("span", "group-label", label),
    textNodeElement("span", "group-stats", `${techniqueCount} technique${techniqueCount !== 1 ? "s" : ""}`),
  );
  header.append(toggle);
  return { element: header, toggle, checkbox };
}

function renderExperimentTechniqueRow(technique, groupEl) {
  const row = document.createElement("div");
  row.className = "exp-technique-row";

  const label = document.createElement("label");
  label.className = "exp-technique-label";

  const checkbox = experimentCheckbox("exp-technique-cb", technique.key, true);
  checkbox.addEventListener("change", () => {
    updateTechniqueCount();
    syncGroupCheckbox(groupEl);
  });

  const badges = textNodeElement("span", "exp-technique-badges", "");
  if (technique.requires_feedback_loop) badges.append(techniqueBadge("tag-iterative", "iterative"));
  if (technique.requires_attacker_llm) badges.append(techniqueBadge("tag-attacker", "attacker LLM"));

  label.append(
    checkbox,
    textNodeElement("span", "exp-technique-name", technique.name),
    badges,
  );
  row.append(label);
  return row;
}

function experimentCheckbox(className, value, checked) {
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.className = className;
  checkbox.value = value;
  checkbox.checked = checked;
  return checkbox;
}

function techniqueBadge(className, label) {
  return textNodeElement("span", `tag ${className}`, label);
}

function syncGroupCheckbox(groupEl) {
  const all = groupEl.querySelectorAll(".exp-technique-cb");
  const checked = groupEl.querySelectorAll(".exp-technique-cb:checked");
  const groupCb = groupEl.querySelector(".exp-group-cb");
  if (groupCb) {
    groupCb.checked = checked.length === all.length;
    groupCb.indeterminate = checked.length > 0 && checked.length < all.length;
  }
}

function getSelectedTechniques() {
  return Array.from(
    el.techniqueSelector.querySelectorAll(".exp-technique-cb:checked")
  ).map((cb) => cb.value);
}

function updateTechniqueCount() {
  const count = getSelectedTechniques().length;
  el.techniqueCount.textContent = `${count} selected`;
}

el.selectAllBtn.addEventListener("click", () => {
  el.techniqueSelector
    .querySelectorAll(".exp-technique-cb, .exp-group-cb")
    .forEach((cb) => { cb.checked = true; cb.indeterminate = false; });
  updateTechniqueCount();
});

el.deselectAllBtn.addEventListener("click", () => {
  el.techniqueSelector
    .querySelectorAll(".exp-technique-cb, .exp-group-cb")
    .forEach((cb) => { cb.checked = false; cb.indeterminate = false; });
  updateTechniqueCount();
});

// ---------------------------------------------------------------------------
// Dataset management
// ---------------------------------------------------------------------------

async function loadDatasets() {
  try {
    const data = await fetchJson("/api/datasets");
    state.datasets = data.items || [];
    renderDatasetList();
  } catch (err) {
    console.error("Failed to load datasets:", err);
  }
}

function renderDatasetList() {
  if (state.datasets.length === 0) {
    renderMutedMessage(el.datasetList, "No datasets uploaded yet.");
    return;
  }

  const cards = state.datasets.map((dataset) => {
    const selected = state.selectedDatasetId === dataset.dataset_id;
    const card = document.createElement("div");
    card.className = `dataset-card ${selected ? "selected" : ""}`.trim();
    card.dataset.id = dataset.dataset_id;
    card.append(
      textNodeElement("div", "dataset-card-name", dataset.name),
      textNodeElement("div", "dataset-card-meta", `${dataset.prompt_count} prompts`),
    );
    card.addEventListener("click", () => selectDataset(dataset.dataset_id));
    return card;
  });

  el.datasetList.replaceChildren(...cards);
}

function selectDataset(datasetId) {
  state.selectedDatasetId = datasetId;
  const ds = state.datasets.find((d) => d.dataset_id === datasetId);
  el.datasetSelected.hidden = false;
  el.selectedDatasetTag.textContent = ds
    ? `${ds.name} (${ds.prompt_count} prompts)`
    : datasetId;
  renderDatasetList();
  saveState();
}

el.uploadBtn.addEventListener("click", async () => {
  const name = el.dsName.value.trim();
  const raw = el.dsPrompts.value.trim();
  if (!name) {
    el.dsUploadStatus.textContent = "Name required";
    return;
  }
  let prompts;
  try {
    prompts = JSON.parse(raw);
    if (!Array.isArray(prompts) || prompts.length === 0) throw new Error();
  } catch {
    el.dsUploadStatus.textContent = "Invalid JSON — must be a non-empty array";
    return;
  }
  el.dsUploadStatus.textContent = "Uploading...";
  try {
    const result = await fetchJson("/api/datasets", {
      method: "POST",
      body: JSON.stringify({ name, description: "", prompts }),
    });
    const ds = result.item;
    state.datasets.push({
      dataset_id: ds.dataset_id,
      name: ds.name,
      prompt_count: ds.prompt_count,
      created_at: ds.created_at,
    });
    selectDataset(ds.dataset_id);
    el.dsUploadStatus.textContent = `Uploaded — ${ds.prompt_count} prompts`;
    el.dsName.value = "";
    el.dsPrompts.value = "";
    renderDatasetList();
  } catch (err) {
    el.dsUploadStatus.textContent = err.message;
  }
});

// ---------------------------------------------------------------------------
// Target config (load from localStorage)
// ---------------------------------------------------------------------------

const CONFIG_KEY = "jailbreaker_target_config";

function loadTargetConfig(savedTargetConfig = null) {
  try {
    const loadedFromStorage = savedTargetConfig === null;
    const parsed = savedTargetConfig || JSON.parse(localStorage.getItem(CONFIG_KEY) || "{}");
    const saved = stripSecretConfigKeys({ ...parsed });
    if (loadedFromStorage && hasSecretConfigKeys(parsed)) {
      localStorage.setItem(CONFIG_KEY, JSON.stringify(saved));
    }
    if (saved.target_profile_id) el.targetProfile.value = saved.target_profile_id;
    if (saved.api_base_url) el.apiBaseUrl.value = saved.api_base_url;
    if (saved.model) el.model.value = saved.model;
    if (saved.system_prompt) el.systemPrompt.value = saved.system_prompt;
    if (saved.attacker_profile_id) el.attackerProfile.value = saved.attacker_profile_id;
    if (saved.attacker_api_base_url)
      el.attackerBaseUrl.value = saved.attacker_api_base_url;
    if (saved.attacker_model) el.attackerModel.value = saved.attacker_model;
    const legacyJudgeMode = (
      saved.attacker_profile_id || saved.attacker_api_base_url || saved.attacker_model
    ) ? "attacker" : "target";
    el.judgeMode.value = saved.judge_mode || legacyJudgeMode;
    if (saved.judge_profile_id) {
      el.judgeProfile.value = saved.judge_profile_id;
    } else if (el.judgeMode.value === "attacker") {
      el.judgeProfile.value = EXPERIMENT_JUDGE_ATTACKER_VALUE;
    } else if (el.judgeMode.value === "target") {
      el.judgeProfile.value = EXPERIMENT_JUDGE_TARGET_VALUE;
    }
    if (saved.judge_api_base_url) el.judgeBaseUrl.value = saved.judge_api_base_url;
    if (saved.judge_model) el.judgeModel.value = saved.judge_model;
    if (saved.judge_local) {
      el.judgeLocal.checked = true;
      el.judgeLocal.dispatchEvent(new Event("change"));
    }
    updateJudgeConfigVisibility();
  } catch {
    /* ignore */
  }
}

function getTargetConfig() {
  return {
    target_profile_id: profileIdFromSelect(el.targetProfile),
    api_key: el.apiKey.value.trim(),
    api_base_url: el.apiBaseUrl.value.trim(),
    model: el.model.value.trim(),
    system_prompt: el.systemPrompt.value.trim(),
    attacker_profile_id: profileIdFromSelect(el.attackerProfile),
    attacker_api_key: el.attackerApiKey.value.trim(),
    attacker_api_base_url: el.attackerBaseUrl.value.trim(),
    attacker_model: el.attackerModel.value.trim(),
    judge_mode: el.judgeMode.value,
    judge_local: el.judgeLocal.checked,
    judge_profile_id: profileIdFromSelect(el.judgeProfile),
    judge_api_key: el.judgeLocal.checked ? "lm-studio" : el.judgeApiKey.value.trim(),
    judge_api_base_url: el.judgeBaseUrl.value.trim(),
    judge_model: el.judgeModel.value.trim(),
  };
}

function getStoredTargetConfig() {
  const config = { ...getTargetConfig() };
  return stripSecretConfigKeys(config);
}

function boundedInt(value, fallback, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function updateJudgeConfigVisibility() {
  const isCustom = el.judgeMode.value === "custom";
  el.judgeCustomFields.style.display = isCustom ? "" : "none";
}

el.judgeMode.addEventListener("change", () => {
  syncJudgeProfileSelectionFromMode();
  updateJudgeConfigVisibility();
  saveState();
});
el.targetProfile.addEventListener("change", () => applySelectedProfile("target"));
el.attackerProfile.addEventListener("change", () => applySelectedProfile("attacker"));
el.judgeProfile.addEventListener("change", () => applySelectedProfile("judge"));

el.judgeLocal.addEventListener("change", () => {
  const isLocal = el.judgeLocal.checked;
  el.judgeKeyField.style.display = isLocal ? "none" : "";
  if (isLocal) {
    el.judgeApiKey.value = "lm-studio";
    if (!el.judgeBaseUrl.value) el.judgeBaseUrl.value = "http://localhost:1234/v1";
  } else {
    if (el.judgeApiKey.value === "lm-studio") el.judgeApiKey.value = "";
  }
});

// ---------------------------------------------------------------------------
// Create experiment
// ---------------------------------------------------------------------------

el.form.addEventListener("submit", async (e) => {
  e.preventDefault();

	  const name = el.expName.value.trim();
	  const description = el.expDescription.value.trim();
	  const hypothesis = el.expHypothesis.value.trim();
	  const experimentGroup = el.expGroup.value.trim();
	  const repeatCount = boundedInt(el.repeatCount.value, 1, 1, 100);
	  const maxRounds = boundedInt(el.maxRounds.value, 5, 1, 50);
	  const datasetId = state.selectedDatasetId;
	  const techniqueKeys = getSelectedTechniques();
	  const targetConfig = getTargetConfig();

  if (!name) return alert("Experiment name is required.");
  if (!datasetId) return alert("Please upload or select a dataset first.");
  if (techniqueKeys.length === 0) return alert("Select at least one technique.");
  if (!targetConfig.api_key && !targetConfig.target_profile_id) return alert("API key or saved profile is required.");

  el.runBtn.disabled = true;
  el.runBtn.textContent = "Starting...";

  try {
    const result = await fetchJson("/api/experiments", {
      method: "POST",
      body: JSON.stringify({
        name,
        description,
        hypothesis,
        dataset_id: datasetId,
	        technique_keys: techniqueKeys,
	        target_config: targetConfig,
	        experiment_group: experimentGroup,
	        repeat_count: repeatCount,
	        max_rounds: maxRounds,
	        mode: "live",
	      }),
    });
    state.currentExperiment = result.item;
    state.currentExperimentId = result.item.experiment_id;
    trackExperiment(result.item.experiment_id);
    loadExperiments();
    switchTab("history");
    saveState();
  } catch (err) {
    alert("Failed to create experiment: " + err.message);
  } finally {
    el.runBtn.disabled = false;
    el.runBtn.textContent = "Run Experiment";
  }
});

// ---------------------------------------------------------------------------
// Polling — tracks all running experiments
// ---------------------------------------------------------------------------

const _trackedExperiments = new Set();
let _runningPollTimer = null;

function trackExperiment(experimentId) {
  _trackedExperiments.add(experimentId);
  _ensureRunningPoll();
}

function _ensureRunningPoll() {
  if (_runningPollTimer) return;
  _runningPollTimer = setInterval(_pollAllRunning, 2500);
  _pollAllRunning();
}

async function _pollAllRunning() {
  try {
    const data = await fetchJson("/api/experiments");
    const all = data.items || [];
	    const running = all.filter((e) => e.status === "running" || e.status === "pending" || e.status === "rejudging");

    // Add any running experiments we didn't know about
    for (const e of running) _trackedExperiments.add(e.experiment_id);

    // Check if any tracked experiment just finished
    const finished = all.filter(
	      (e) => _trackedExperiments.has(e.experiment_id) && (e.status === "completed" || e.status === "failed")
    );
    for (const e of finished) {
      _trackedExperiments.delete(e.experiment_id);
    }

    // Refresh history cards (updates progress bars inline)
    state.experiments = all;
    renderHistory();
    populateMetricsDropdown();

    // Stop polling if nothing is running or tracked
    if (running.length === 0 && _trackedExperiments.size === 0) {
      clearInterval(_runningPollTimer);
      _runningPollTimer = null;
    }
  } catch (err) {
    console.error("Poll error:", err);
  }
}

// Legacy compat — old code calls these
function startPolling(experimentId) {
  trackExperiment(experimentId);
}
function stopPolling() {
  // no-op — the global poller manages itself
}

// (progress display moved to inline bars in History tab)

// ---------------------------------------------------------------------------
// Render results
// ---------------------------------------------------------------------------

function renderResults(exp) {
  if (el.resultsEmpty) el.resultsEmpty.style.display = "none";
  el.results.style.display = "";
  const report = exp.report || {};
  renderResultsHeader(exp, report);

  const posture = report.security_posture || "unknown";
  const worstTechnique = topRiskStat(report.technique_stats);
  const worstCategory = topRiskStat(report.category_stats);
  renderPostureSummary(report, posture, worstTechnique, worstCategory);
  renderSeverityBreakdown(report);
  renderRiskList(report);

  // Matrix heatmap
  renderMatrix(exp);
  renderEvidence(exp);

  // Wire export buttons
  const eid = exp.experiment_id;
  el.exportCsvBtn.onclick = () => downloadExport(eid, "csv");
  el.exportJsonBtn.onclick = () => downloadExport(eid, "json");
}

function renderPostureSummary(report, posture, worstTechnique, worstCategory) {
  el.postureSummary.replaceChildren(
    insightCard("Overall ASR", `${report.overall_asr ?? 0}%`, formatCi(report.asr_ci_low, report.asr_ci_high)),
    insightCard("Avg Score", report.avg_score ?? 0, `SD ${report.score_stddev ?? 0}`),
    insightCard("Successes / Tested", `${report.successes ?? 0}/${report.tested_cells ?? report.total_cells ?? 0}`),
    insightCard("Worst Technique", worstTechnique.label, `${worstTechnique.asr}% ASR`, "posture-value posture-value--compact"),
    insightCard("Worst Category", worstCategory.label, `${worstCategory.asr}% ASR`, "posture-value posture-value--compact"),
    insightCard("Posture", posture, "", `posture-value posture-badge posture-badge--${classToken(posture)}`),
  );
}

function insightCard(label, value, subvalue = "", valueClass = "posture-value") {
  const card = document.createElement("div");
  card.className = "insight-card";
  card.append(
    textNodeElement("div", "posture-label", label),
    textNodeElement("div", valueClass, value),
  );
  if (subvalue !== "") {
    card.append(textNodeElement("div", "posture-subvalue", subvalue));
  }
  return card;
}

function renderSeverityBreakdown(report) {
  const breakdown = report.severity_breakdown || {};
  const total = Object.values(breakdown).reduce((sum, count) => sum + count, 0) || 1;
  const severities = ["critical", "high", "medium", "low", "info"];
  el.severityBar.replaceChildren(
    ...severities
      .filter((severity) => breakdown[severity])
      .map((severity) => renderSeveritySegment(severity, (breakdown[severity] / total) * 100)),
  );
  el.severityLegend.replaceChildren(
    ...severities.map((severity) => renderSeverityLegendItem(severity, breakdown[severity] || 0)),
  );
}

function renderSeveritySegment(severity, width) {
  const segment = document.createElement("div");
  segment.className = `severity-segment severity-segment--${severity}`;
  segment.style.width = `${width}%`;
  return segment;
}

function renderSeverityLegendItem(severity, count) {
  const item = document.createElement("span");
  item.className = "severity-legend-item";
  const dot = document.createElement("span");
  dot.className = `severity-dot severity-dot--${severity}`;
  item.append(dot, document.createTextNode(`${severity}: ${count}`));
  return item;
}

function classToken(value) {
  return String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "") || "unknown";
}

function renderResultsHeader(exp, report) {
  if (el.resultsTitle) el.resultsTitle.textContent = exp.name || exp.experiment_id || "Experiment results";
  if (!el.resultsMeta) return;
  const cells = `${exp.completed_cells ?? report.tested_cells ?? 0}/${exp.total_cells ?? report.total_cells ?? 0} cells`;
  const parts = [
    exp.status,
    exp.target_model ? `target: ${exp.target_model}` : "",
    exp.attacker_model ? `attacker: ${exp.attacker_model}` : "",
    exp.dataset_name || exp.dataset_id ? `dataset: ${exp.dataset_name || exp.dataset_id}` : "",
    cells,
  ].filter(Boolean);
  el.resultsMeta.replaceChildren(...parts.map((part) => textNodeElement("span", "", part)));
}

function topRiskStat(stats = {}) {
  const [label, value] = Object.entries(stats).sort(([, a], [, b]) => (b.asr || 0) - (a.asr || 0))[0] || ["n/a", { asr: 0 }];
  return { label: displayTechniqueKey(label), asr: value.asr ?? 0, tested: value.tested ?? 0 };
}

function renderRiskList(report) {
  if (!el.resultsRiskList) return;
  const techniqueRows = Object.entries(report.technique_stats || {})
    .sort(([, a], [, b]) => (b.asr || 0) - (a.asr || 0))
    .slice(0, 3)
    .map(([label, stat]) => ({ type: "Technique", label: displayTechniqueKey(label), stat }));
  const categoryRows = Object.entries(report.category_stats || {})
    .sort(([, a], [, b]) => (b.asr || 0) - (a.asr || 0))
    .slice(0, 3)
    .map(([label, stat]) => ({ type: "Category", label, stat }));
  const rows = [...techniqueRows, ...categoryRows].sort((a, b) => (b.stat.asr || 0) - (a.stat.asr || 0)).slice(0, 5);
  if (!rows.length) {
    renderMutedMessage(el.resultsRiskList, "No completed slices yet.");
    return;
  }
  el.resultsRiskList.replaceChildren(...rows.map(renderRiskRow));
}

function renderRiskRow(row) {
  const item = document.createElement("div");
  item.className = "risk-row";

  const details = document.createElement("div");
  details.append(
    textNodeElement("div", "risk-row-label", row.label),
    textNodeElement(
      "div",
      "risk-row-meta",
      `${row.type} / ${row.stat.successes ?? 0}/${row.stat.tested ?? 0} successes`,
    ),
  );

  const asr = row.stat.asr ?? 0;
  item.append(
    details,
    textNodeElement("div", `risk-row-score ${asrClass(asr)}`, `${asr}%`),
  );
  return item;
}

function asrClass(asr) {
  if (asr >= 80) return "asr-critical";
  if (asr >= 60) return "asr-high";
  if (asr >= 40) return "asr-medium";
  if (asr >= 20) return "asr-low";
  return "asr-info";
}

function formatCi(low, high) {
  if (low == null || high == null) return "n/a";
  return `${low}-${high}%`;
}

function displayTechniqueKey(key) {
  return String(key || "");
}

// ---------------------------------------------------------------------------
// Matrix heatmap
// ---------------------------------------------------------------------------

function renderMatrix(exp) {
  const cells = exp.cell_results || [];
  if (cells.length === 0) {
    renderMutedMessage(el.matrixWrap, "No results.");
    return;
  }

  const techniqueKeys = [...new Set(cells.map((cell) => cell.technique_key))];
  const promptIds = [...new Set(cells.map((cell) => cell.prompt_id))];
  const cellMap = buildMatrixCellMap(cells);
  const grid = document.createElement("div");
  grid.className = "matrix-grid";
  grid.style.gridTemplateColumns = `180px ${promptIds.map(() => "1fr").join(" ")}`;

  grid.append(matrixHeader("", "matrix-header matrix-corner"));
  promptIds.forEach((promptId) => {
    grid.append(matrixHeader(promptId, "matrix-header matrix-col-header", promptId));
  });

  techniqueKeys.forEach((techniqueKey) => {
    const displayKey = displayTechniqueKey(techniqueKey);
    grid.append(matrixHeader(truncate(displayKey, 22), "matrix-header matrix-row-header", displayKey));
    promptIds.forEach((promptId) => {
      const group = cellMap.get(matrixGroupKey(techniqueKey, promptId)) || [];
      grid.append(renderMatrixCell(techniqueKey, promptId, group));
    });
  });

  el.matrixWrap.replaceChildren(grid);
}

function buildMatrixCellMap(cells) {
  const cellMap = new Map();
  cells.forEach((cell) => {
    const key = matrixGroupKey(cell.technique_key, cell.prompt_id);
    const group = cellMap.get(key) || [];
    group.push(cell);
    cellMap.set(key, group);
  });
  return cellMap;
}

function matrixGroupKey(techniqueKey, promptId) {
  return JSON.stringify([techniqueKey, promptId]);
}

function matrixHeader(text, className, title = "") {
  const header = textNodeElement("div", className, text);
  if (title) header.title = title;
  return header;
}

function renderMatrixCell(techniqueKey, promptId, group) {
  if (group.length === 0) {
    return textNodeElement("div", "matrix-cell matrix-cell--empty", "");
  }

  const stats = matrixCellStats(group);
  const cell = textNodeElement("div", "matrix-cell", stats.status);
  cell.style.background = stats.color;
  cell.title = `${techniqueKey} / ${promptId}: avg score=${stats.score}, asr=${stats.asr}, runs=${stats.testedCount}/${group.length}`;
  cell.dataset.tk = techniqueKey;
  cell.dataset.pid = promptId;
  cell.addEventListener("click", () => {
    const selected = [...group].reverse().find((item) => item.status !== "error") || group[group.length - 1];
    if (selected && selected.conversation) showConversation(selected);
  });
  return cell;
}

function matrixCellStats(group) {
  const tested = group.filter((cell) => cell.status !== "error");
  const score = tested.length
    ? Math.round(tested.reduce((sum, cell) => sum + (cell.score || 0), 0) / tested.length)
    : 0;
  const successes = tested.filter((cell) => cell.asr === 100).length;
  const asr = tested.length ? Math.round((successes / tested.length) * 100) : 0;
  return {
    asr,
    color: scoreColor(score),
    score,
    status: tested.length ? score : "ERR",
    testedCount: tested.length,
  };
}

function renderEvidence(exp) {
  if (!el.resultsEvidenceList) return;
  const rows = selectEvidenceRows(exp.cell_results || []);
  if (!rows.length) {
    renderMutedMessage(el.resultsEvidenceList, "No evidence rows yet. Completed cells will appear here.");
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "breakdown-table-wrap evidence-table-wrap";
  const table = document.createElement("table");
  table.className = "breakdown-table evidence-table";
  table.append(renderEvidenceHead(), renderEvidenceBody(rows));
  wrapper.append(table);
  el.resultsEvidenceList.replaceChildren(wrapper);
}

function renderEvidenceHead() {
  const thead = document.createElement("thead");
  const row = document.createElement("tr");
  ["Why", "Technique", "Prompt", "Run", "Score", "ASR", "Severity", "Action"].forEach((label) => {
    row.append(textNodeElement("th", "", label));
  });
  thead.append(row);
  return thead;
}

function renderEvidenceBody(rows) {
  const tbody = document.createElement("tbody");
  rows.forEach((cell) => {
    tbody.append(renderEvidenceRow(cell));
  });
  return tbody;
}

function renderEvidenceRow(cell) {
  const row = document.createElement("tr");
  row.append(
    textNodeElement("td", "", evidenceReason(cell)),
    textNodeElement("td", "metric-technique-name", displayTechniqueKey(cell.technique_key)),
    textNodeElement("td", "", cell.prompt_id),
    textNodeElement("td", "", cell.repeat_index ?? 1),
    textNodeElement("td", "", cell.score ?? "n/a"),
    textNodeElement("td", asrClass(cell.asr || 0), `${cell.asr ?? 0}%`),
    textNodeElement("td", "", cell.highest_severity || "n/a"),
    renderEvidenceActionCell(cell),
  );
  return row;
}

function renderEvidenceActionCell(cell) {
  const action = document.createElement("td");
  if (!cell.conversation) {
    action.textContent = "n/a";
    return action;
  }
  const button = document.createElement("button");
  button.className = "btn btn-secondary btn-xs evidence-view-btn";
  button.textContent = "View";
  button.addEventListener("click", () => showConversation(cell));
  action.append(button);
  return action;
}

function selectEvidenceRows(cells) {
  const interesting = cells.filter((cell) =>
    cell.status === "error" ||
    (cell.score || 0) >= 80 ||
    cell.asr === 100 ||
    (cell.rejudge_results || []).length > 0
  );
  const source = interesting.length ? interesting : cells.filter((cell) => cell.status !== "pending");
  return source
    .slice()
    .sort((a, b) => evidencePriority(b) - evidencePriority(a))
    .slice(0, 12);
}

function evidencePriority(cell) {
  let priority = cell.score || 0;
  if (cell.asr === 100) priority += 120;
  if ((cell.score || 0) >= 80) priority += 80;
  if ((cell.rejudge_results || []).length) priority += 50;
  if (cell.status === "error") priority += 40;
  return priority;
}

function evidenceReason(cell) {
  if (cell.status === "error") return "Error";
  if ((cell.rejudge_results || []).length) return "Rejudge";
  if (cell.asr === 100) return "Success";
  if ((cell.score || 0) >= 80) return "High score";
  return "Top score";
}

function scoreColor(score) {
  // Green (safe) → Red (jailbroken) gradient
  if (score >= 80) return "rgba(248, 81, 73, 0.7)";
  if (score >= 60) return "rgba(240, 136, 62, 0.6)";
  if (score >= 40) return "rgba(210, 153, 34, 0.5)";
  if (score >= 20) return "rgba(63, 185, 80, 0.4)";
  return "rgba(63, 185, 80, 0.25)";
}

// ---------------------------------------------------------------------------
// Conversation modal
// ---------------------------------------------------------------------------

function showConversation(cell) {
  const repeat = cell.repeat_index ? ` / run ${cell.repeat_index}` : "";
  el.modalTitle.textContent = `${displayTechniqueKey(cell.technique_key)} — ${cell.prompt_id}${repeat}`;
  el.modalBody.replaceChildren(
    ...renderRejudgeResults(cell),
    ...renderConversation(cell.conversation || []),
  );
  el.modalOverlay.style.display = "";
}

el.modalClose.addEventListener("click", () => {
  el.modalOverlay.style.display = "none";
});

el.modalOverlay.addEventListener("click", (e) => {
  if (e.target === el.modalOverlay) el.modalOverlay.style.display = "none";
});

el.toggleAttacker.addEventListener("change", (e) => {
  document.body.classList.toggle("show-attacker", e.target.checked);
});

el.toggleRawInput.addEventListener("change", (e) => {
  document.body.classList.toggle("show-raw-input", e.target.checked);
});

function renderConversation(conv) {
  if (!conv || conv.length === 0) return [textNodeElement("p", "text-muted", "No conversation data.")];
  return conv.map((entry) => {
    if (entry.kind === "round") return renderRound(entry);
    if (entry.kind === "generation") return renderGeneration(entry);
    if (entry.kind === "artifact") return renderArtifact(entry);
    return renderSingleShot(entry);
  });
}

function renderRejudgeResults(cell) {
  const results = cell.rejudge_results || [];
  if (!results.length) return [];

  const entry = conversationElement("div", "conversation-entry conversation-entry--judge");
  entry.append(textNodeElement("div", "entry-header", "Rejudge Results"));
  for (const result of results) {
    const success = (result.target_score || 0) >= (result.threshold || 7);
    const box = conversationElement("div", `judge-box ${success ? "judge-success" : "judge-fail"}`);
    box.append(
      textNodeElement("strong", "", result.judge_model || result.judge_mode || "judge"),
      document.createTextNode(
        `: target ${result.target_score ?? "n/a"}/10, fabricator ${result.fabricator_score ?? "n/a"}/10, pipeline ASR ${result.pipeline_unsafe_asr ?? 0}%`,
      ),
    );
    entry.append(box);
  }
  return [entry];
}

function renderRound(entry) {
  const judge = entry.judge || {};
  const target = entry.target || {};
  const attacker = entry.attacker || {};
  const scoreNum = parseInt(judge.score, 10);
  const scoreClass = scoreNum >= 7 ? "judge-success" : "judge-fail";

  const round = conversationElement("div", "conv-round");
  const header = conversationElement("div", "conv-round-header");
  header.append(document.createTextNode(`Round ${entry.round_number ?? ""}`));
  if (entry.is_backtrack) {
    header.append(document.createTextNode(" "), textNodeElement("span", "tag tag-xs tag-warning", "BACKTRACK"));
  }
  round.append(header);

  if (attacker.input || attacker.output) {
    const flow = conversationElement("div", "conv-attacker-flow");
    appendPresent(
      flow,
      chatBubble("attacker-in", "Attacker Input", attacker.input),
      chatBubble("attacker-out", "Attacker Output", attacker.output),
    );
    round.append(flow);
  }

  appendPresent(
    round,
    target.input ? rawInputBubble("Raw Target Input", target.input) : null,
    chatBubble("user", "User", target.prompt || ""),
    chatBubble("target", "Target (Real Response)", target.response || ""),
  );

  if (judge.score && judge.score !== "?") {
    round.append(judgeBox(scoreClass, `Judge: ${judge.score}/10`, judge.rationale || ""));
  }

  if (target.poisoned) {
    round.append(chatBubble("poisoned", 'Fabricated Response (what target "sees" in history)', target.poisoned));
  }

  return round;
}

function renderGeneration(entry) {
  const candidates = entry.candidates || [];
  const best = candidates.find((candidate) => candidate.is_best);
  const round = conversationElement("div", "conv-round");
  round.append(textNodeElement(
    "div",
    "conv-round-header",
    `Generation ${entry.generation_number ?? ""} — ${entry.population_size} candidates, best score: ${entry.best_score ?? ""}`,
  ));

  if (best) {
    appendPresent(
      round,
      chatBubble("user", "Best Prompt", best.prompt),
      chatBubble("target", "Response", best.response),
    );
    if (best.judge_score) {
      const score = parseInt(best.judge_score, 10);
      round.append(judgeBox(score >= 7 ? "judge-success" : "judge-fail", `Judge: ${best.judge_score}/10`, best.judge_rationale || ""));
    }
  }
  return round;
}

function renderSingleShot(entry) {
  const judge = entry.judge || {};
  const round = conversationElement("div", "conv-round");

  appendPresent(
    round,
    entry.full_input ? rawInputBubble("Full Input", entry.full_input) : null,
    chatBubble("user", "Prompt", entry.prompt || ""),
    chatBubble("target", "Response", entry.response || ""),
  );

  if (judge.score && judge.score !== "?") {
    const score = parseInt(judge.score, 10);
    round.append(judgeBox(score >= 7 ? "judge-success" : "judge-fail", `Judge: ${judge.score}/10`, judge.rationale || ""));
  }

  return round;
}

function renderArtifact(entry) {
  const round = conversationElement("div", "conv-round");
  round.append(
    textNodeElement("div", "conv-round-header", entry.label || "Final artifact"),
    chatBubble("raw", "Final Output", entry.response || ""),
  );
  return round;
}

function chatBubble(kind, label, value) {
  if (value === null || value === undefined) return null;
  const bubble = conversationElement("div", `chat-bubble chat-bubble--${kind}`);
  bubble.append(textNodeElement("div", "chat-label", label), textNodeElement("pre", "", value));
  return bubble;
}

function rawInputBubble(label, value) {
  const wrapper = conversationElement("div", "conv-raw-input");
  wrapper.append(chatBubble("raw", label, value));
  return wrapper;
}

function judgeBox(className, label, detail) {
  const box = conversationElement("div", `judge-box ${className}`);
  box.append(textNodeElement("strong", "", label), document.createTextNode(` — ${detail}`));
  return box;
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

async function downloadExport(experimentId, format) {
  try {
    const url = `${API}/api/experiments/${experimentId}/export?format=${format}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Export failed");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${experimentId}.${format}`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    alert("Export failed: " + err.message);
  }
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

async function loadExperiments() {
  try {
    const data = await fetchJson("/api/experiments");
    state.experiments = data.items || [];
    renderHistory();
    populateMetricsDropdown();
  } catch (err) {
    console.error("Failed to load experiments:", err);
  }
}

function renderHistory() {
  if (state.experiments.length === 0) {
    renderMutedMessage(el.historyList, "No experiments yet.");
    return;
  }

  el.historyList.replaceChildren(...state.experiments.map(renderHistoryCard));
}

function renderHistoryCard(exp) {
  const isRunning = isRunningExperiment(exp);
  const hasGap = exp.completed_cells < exp.total_cells;
  const card = document.createElement("div");
  card.className = `history-card ${isRunning ? "history-card--running" : ""}`.trim();
  card.dataset.id = exp.experiment_id;

  const header = document.createElement("div");
  header.className = "history-card-header";
  header.append(
    textNodeElement("span", "history-card-name", exp.name),
    textNodeElement("span", `tag tag-xs ${historyStatusClass(exp.status)}`, exp.status),
  );
  card.append(header);

  const modelInfo = historyModelInfo(exp);
  if (modelInfo) {
    card.append(textNodeElement("div", "history-card-models", modelInfo));
  }

  card.append(renderHistoryMeta(exp));
  if (isRunning) {
    card.append(renderHistoryProgress(exp));
  }
  card.append(renderHistoryActions(exp, isRunning, hasGap));
  return card;
}

function isRunningExperiment(exp) {
  return exp.status === "running" || exp.status === "pending" || exp.status === "rejudging";
}

function historyStatusClass(status) {
  if (status === "completed") return "tag-success";
  if (status === "failed") return "tag-danger";
  return "tag-info";
}

function historyModelInfo(exp) {
  const targetModel = exp.target_model || "";
  const attackerModel = exp.attacker_model || "";
  const judgeModel = exp.judge_model || "";
  const modelParts = [];
  if (targetModel) modelParts.push(`target: ${targetModel}`);
  if (attackerModel && attackerModel !== targetModel) modelParts.push(`attacker: ${attackerModel}`);
  if (judgeModel && judgeModel !== targetModel && judgeModel !== attackerModel) modelParts.push(`judge: ${judgeModel}`);
  return modelParts.join(" / ");
}

function renderHistoryMeta(exp) {
  const meta = document.createElement("div");
  meta.className = "history-card-meta";
  meta.append(
    document.createTextNode(
      `${exp.technique_count} techniques \u00d7 ${exp.dataset_name || "dataset"} \u2014 `
      + `${exp.completed_cells}/${exp.total_cells} cells`,
    ),
  );
  if (exp.repeat_count && exp.repeat_count > 1) {
    meta.append(document.createTextNode(` \u00d7 ${exp.repeat_count} repeats`));
  }
  if (exp.experiment_group) {
    meta.append(document.createTextNode(` \u2014 ${exp.experiment_group}`));
  }
  if (exp.max_rounds) {
    meta.append(document.createTextNode(` \u2014 ${exp.max_rounds} rounds`));
  }
  if (exp.error_cells) {
    meta.append(document.createTextNode(" \u2014 "));
    meta.append(textNodeElement("span", "history-error-count", `${exp.error_cells} errors`));
  }
  return meta;
}

function renderHistoryProgress(exp) {
  const pct = exp.total_cells ? Math.round((exp.completed_cells / exp.total_cells) * 100) : 0;
  const progress = document.createElement("div");
  progress.className = "history-progress";
  const track = document.createElement("div");
  track.className = "progress-track";
  const fill = document.createElement("div");
  fill.className = "progress-fill";
  fill.style.width = `${pct}%`;
  track.append(fill);
  progress.append(track);
  return progress;
}

function renderHistoryActions(exp, isRunning, hasGap) {
  const actions = document.createElement("div");
  actions.className = "history-card-actions";
  if (isRunning) {
    actions.append(historyButton("hist-stop-btn", exp.experiment_id, "Stop"));
    return actions;
  }

  actions.append(
    historyButton("hist-view-btn", exp.experiment_id, "View"),
    historyButton("hist-rejudge-btn", exp.experiment_id, "Rejudge"),
  );
  if (hasGap || exp.status === "completed" || exp.status === "failed") {
    actions.append(historyButton("hist-retry-btn", exp.experiment_id, "Retry Errors"));
  }
  return actions;
}

function historyButton(actionClass, experimentId, label) {
  const button = document.createElement("button");
  button.className = `btn btn-secondary btn-xs ${actionClass}`;
  button.dataset.id = experimentId;
  button.textContent = label;
  return button;
}

function renderMutedMessage(container, message) {
  container.replaceChildren(textNodeElement("p", "text-muted", message));
}

// Use event delegation so buttons survive poll re-renders
el.historyList.addEventListener("click", (e) => {
  const viewBtn = e.target.closest(".hist-view-btn");
  if (viewBtn) { e.stopPropagation(); loadExperiment(viewBtn.dataset.id); return; }
  const stopBtn = e.target.closest(".hist-stop-btn");
  if (stopBtn) { e.stopPropagation(); stopExperiment(stopBtn.dataset.id); return; }
	  const retryBtn = e.target.closest(".hist-retry-btn");
	  if (retryBtn) { e.stopPropagation(); retryExperiment(retryBtn.dataset.id); return; }
	  const rejudgeBtn = e.target.closest(".hist-rejudge-btn");
	  if (rejudgeBtn) { e.stopPropagation(); rejudgeExperiment(rejudgeBtn.dataset.id); return; }
	});

async function stopExperiment(experimentId) {
  try {
    const data = await fetchJson(`/api/experiments/${experimentId}/stop`, { method: "POST" });
    const item = data.item || {};
    alert(`Stopped "${item.name || experimentId}" at ${item.completed_cells}/${item.total_cells} cells.`);
    loadExperiments();
  } catch (err) {
    alert("Failed to stop: " + err.message);
  }
}


function buildRejudgeOverlay(exp, completedCellCount, rejudgeMode, localChecked, baseUrl, model) {
  const modal = buildActionModal("Rejudge Experiment", "rejudge-cancel", "520px");
  const form = overlayForm("rejudge-form");
  const customFields = document.createElement("div");
  customFields.id = "rejudge-judge-custom-fields";
  customFields.style.display = rejudgeMode === "custom" ? "" : "none";

  customFields.append(
    overlayToggle("rejudge-judge-local", localChecked, "Local (LM Studio / Ollama - no API key)"),
    formRow(formField("API Key", overlayInput("rejudge-judge-key", "password", "", "sk-..."), { id: "rejudge-judge-key-field", className: "flex-1", hidden: localChecked })),
    formRow(
      formField("Base URL", overlayInput("rejudge-judge-base-url", "text", baseUrl), { className: "flex-1" }),
      formField("Model", overlayInput("rejudge-judge-model", "text", model)),
    ),
  );

  const group = configGroup("Judge LLM");
  group.append(
    formRow(formField("Judge Source", judgeModeSelect("rejudge-judge-mode", rejudgeMode))),
    customFields,
  );
  form.append(
    group,
    formActions(
      overlayButton("rejudge-cancel-btn", "Cancel", "button", "btn btn-secondary"),
      overlayButton("rejudge-confirm-btn", "Run Rejudge", "submit", "btn btn-primary"),
    ),
  );

  modal.body.append(
    overlaySummary(exp.name, `${completedCellCount} completed cells will be rescored.`),
    form,
  );
  return modal.overlay;
}

function buildRetryOverlay(exp, targetConfig, retryCount, errorCount, missingCount, cells, retryJudgeMode, retryJudgeLocalChecked) {
  const modal = buildActionModal(`Retry ${retryCount} Error Cells`, "retry-cancel", "560px");
  const form = overlayForm("retry-form");
  form.append(
    retryTargetGroup(targetConfig),
    retryAttackerGroup(targetConfig),
    retryJudgeGroup(targetConfig, retryJudgeMode, retryJudgeLocalChecked),
    formActions(
      overlayButton("retry-cancel-btn", "Cancel", "button", "btn btn-secondary"),
      overlayButton("retry-confirm-btn", `Retry ${retryCount} Cells`, "submit", "btn btn-primary"),
    ),
  );

  modal.body.append(
    overlaySummary(
      exp.name,
      `${errorCount} errors, ${missingCount} missing out of ${exp.total_cells} total. ${cells.length - errorCount} successful cells will be kept.`,
    ),
    form,
  );
  return modal.overlay;
}

function retryTargetGroup(targetConfig) {
  const group = configGroup("Target LLM");
  group.append(
    formRow(formField("API Key", overlayInput("retry-api-key", "password", "", "sk-..."), { className: "flex-1" })),
    formRow(
      formField("Base URL", overlayInput("retry-base-url", "text", targetConfig.api_base_url || ""), { className: "flex-1" }),
      formField("Model", overlayInput("retry-model", "text", targetConfig.model || "")),
    ),
    formRow(formField("System Prompt", overlayInput("retry-system-prompt", "text", targetConfig.system_prompt || "You are a helpful assistant."), { className: "flex-1" })),
  );
  return group;
}

function retryAttackerGroup(targetConfig) {
  const group = configGroup("Attacker LLM");
  group.append(
    formRow(formField("API Key (leave blank to use target key)", overlayInput("retry-attacker-key", "password", "", "same as target"), { className: "flex-1" })),
    formRow(
      formField("Base URL", overlayInput("retry-attacker-base-url", "text", targetConfig.attacker_api_base_url || targetConfig.api_base_url || ""), { className: "flex-1" }),
      formField("Model", overlayInput("retry-attacker-model", "text", targetConfig.attacker_model || targetConfig.model || "")),
    ),
  );
  return group;
}

function retryJudgeGroup(targetConfig, retryJudgeMode, retryJudgeLocalChecked) {
  const group = configGroup("Judge LLM");
  const customFields = document.createElement("div");
  customFields.id = "retry-judge-custom-fields";
  customFields.style.display = retryJudgeMode === "custom" ? "" : "none";
  customFields.append(
    overlayToggle("retry-judge-local", retryJudgeLocalChecked, "Local (LM Studio / Ollama — no API key)"),
    formRow(formField("API Key", overlayInput("retry-judge-key", "password", "", "sk-..."), { id: "retry-judge-key-field", className: "flex-1", hidden: retryJudgeLocalChecked })),
    formRow(
      formField("Base URL", overlayInput("retry-judge-base-url", "text", targetConfig.judge_api_base_url || targetConfig.api_base_url || ""), { className: "flex-1" }),
      formField("Model", overlayInput("retry-judge-model", "text", targetConfig.judge_model || "")),
    ),
  );
  group.append(
    formRow(formField("Judge Source", judgeModeSelect("retry-judge-mode", retryJudgeMode))),
    customFields,
  );
  return group;
}

function buildActionModal(title, closeId, maxWidth) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.style.display = "flex";

  const modal = document.createElement("div");
  modal.className = "modal";
  modal.style.maxWidth = maxWidth;

  const header = document.createElement("div");
  header.className = "modal-header";
  const closeButton = overlayButton(closeId, "×", "button", "btn btn-secondary btn-sm modal-close");
  header.append(textNodeElement("h2", "modal-title", title), closeButton);

  const body = document.createElement("div");
  body.className = "modal-body";
  body.style.padding = "1rem";

  modal.append(header, body);
  overlay.append(modal);
  return { overlay, body, closeButton };
}

function overlaySummary(name, summary) {
  const paragraph = document.createElement("p");
  paragraph.style.marginBottom = "1rem";
  paragraph.style.color = "var(--text-secondary)";
  paragraph.style.fontSize = "var(--text-sm)";
  paragraph.append(textNodeElement("strong", "", name), document.createTextNode(` — ${summary}`));
  return paragraph;
}

function overlayForm(id) {
  const form = document.createElement("form");
  form.id = id;
  form.className = "config-form";
  return form;
}

function configGroup(label) {
  const group = document.createElement("div");
  group.className = "config-group";
  group.append(textNodeElement("div", "config-group-label", label));
  return group;
}

function formRow(...fields) {
  const row = document.createElement("div");
  row.className = "form-row";
  row.append(...fields);
  return row;
}

function formField(labelText, control, options = {}) {
  const field = document.createElement("div");
  field.className = `form-field ${options.className || ""}`.trim();
  if (options.id) field.id = options.id;
  if (options.hidden) field.style.display = "none";
  field.append(textNodeElement("label", "", labelText), control);
  return field;
}

function overlayInput(id, type, value = "", placeholder = "") {
  const input = document.createElement("input");
  input.type = type;
  input.id = id;
  input.value = value;
  if (placeholder) input.placeholder = placeholder;
  return input;
}

function judgeModeSelect(id, selectedMode) {
  return overlaySelect(id, selectedMode, [
    ["target", "Target LLM"],
    ["attacker", "Attacker LLM"],
    ["custom", "Custom Judge LLM"],
  ]);
}

function overlaySelect(id, selectedValue, options) {
  const select = document.createElement("select");
  select.id = id;
  for (const [value, label] of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    option.selected = value === selectedValue;
    select.append(option);
  }
  return select;
}

function overlayToggle(inputId, checked, labelText) {
  const wrapper = document.createElement("div");
  wrapper.className = "attacker-toggle";
  const label = document.createElement("label");
  label.className = "toggle-label";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.id = inputId;
  checkbox.checked = checked;
  label.append(checkbox, textNodeElement("span", "", labelText));
  wrapper.append(label);
  return wrapper;
}

function formActions(...buttons) {
  const actions = document.createElement("div");
  actions.className = "form-actions";
  actions.style.marginTop = "1rem";
  actions.append(...buttons);
  return actions;
}

function overlayButton(id, label, type, className) {
  const button = document.createElement("button");
  button.id = id;
  button.type = type;
  button.className = className;
  button.textContent = label;
  return button;
}

async function rejudgeExperiment(experimentId) {
  let exp;
  try {
    const data = await fetchJson(`/api/experiments/${experimentId}`);
    exp = data.item;
  } catch (err) {
    alert("Failed to load experiment: " + err.message);
    return;
  }

  const cells = (exp.cell_results || []).filter((c) => c.status !== "error");
  if (cells.length === 0) {
    alert("No completed cells to rejudge.");
    return;
  }

  const tc = exp.target_config || {};
  const formTc = getTargetConfig();
  const rejudgeMode = tc.judge_mode || formTc.judge_mode || "custom";
  const localChecked = tc.judge_local === true || tc.judge_local === "true" || tc.judge_api_key === "lm-studio";
  const baseUrl = tc.judge_api_base_url || formTc.judge_api_base_url || tc.api_base_url || formTc.api_base_url || "";
  const model = tc.judge_model || formTc.judge_model || "";

  const overlay = buildRejudgeOverlay(exp, cells.length, rejudgeMode, localChecked, baseUrl, model);
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  overlay.querySelector("#rejudge-cancel").addEventListener("click", close);
  overlay.querySelector("#rejudge-cancel-btn").addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  const modeSelect = overlay.querySelector("#rejudge-judge-mode");
  const customFields = overlay.querySelector("#rejudge-judge-custom-fields");
  const localToggle = overlay.querySelector("#rejudge-judge-local");
  const keyField = overlay.querySelector("#rejudge-judge-key-field");
  const baseInput = overlay.querySelector("#rejudge-judge-base-url");
  const updateVisibility = () => {
    customFields.style.display = modeSelect.value === "custom" ? "" : "none";
  };
  modeSelect.addEventListener("change", updateVisibility);
  localToggle.addEventListener("change", () => {
    const isLocal = localToggle.checked;
    keyField.style.display = isLocal ? "none" : "";
    if (isLocal && !baseInput.value) baseInput.value = "http://localhost:1234/v1";
  });

  overlay.querySelector("#rejudge-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const judgeMode = modeSelect.value;
    const judgeIsLocal = localToggle.checked;
    const judgeKey = judgeIsLocal ? "lm-studio" : overlay.querySelector("#rejudge-judge-key").value.trim();
    const judgeModel = overlay.querySelector("#rejudge-judge-model").value.trim();
    if (judgeMode === "custom" && !judgeIsLocal && !judgeKey) {
      alert("Judge API key is required for a custom remote judge.");
      return;
    }
    if (judgeMode === "custom" && !judgeModel) {
      alert("Judge model is required.");
      return;
    }

    const judgeConfig = {
      judge_mode: judgeMode,
      judge_local: judgeIsLocal,
      judge_api_key: judgeKey,
      judge_api_base_url: overlay.querySelector("#rejudge-judge-base-url").value.trim(),
      judge_model: judgeModel,
    };

    const confirmBtn = overlay.querySelector("#rejudge-confirm-btn");
    confirmBtn.disabled = true;
    confirmBtn.textContent = "Starting...";
    try {
      await fetchJson(`/api/experiments/${experimentId}/rejudge`, {
        method: "POST",
        body: JSON.stringify({ judge_config: judgeConfig }),
      });
      close();
      loadExperiments();
      trackExperiment(experimentId);
      switchTab("history");
    } catch (err) {
      alert("Failed to rejudge: " + err.message);
      confirmBtn.disabled = false;
      confirmBtn.textContent = "Run Rejudge";
    }
  });
}

async function retryExperiment(experimentId) {
  // Get the experiment to read its original config
  let exp;
  try {
    const data = await fetchJson(`/api/experiments/${experimentId}`);
    exp = data.item;
  } catch (err) {
    alert("Failed to load experiment: " + err.message);
    return;
  }

  // Count errors
  const cells = exp.cell_results || [];
  const errorCount = cells.filter((c) => c.status === "error").length;
  const missingCount = exp.total_cells - cells.length;
  const retryCount = errorCount + missingCount;

  if (retryCount === 0) {
    alert("No error or missing cells to retry.");
    return;
  }

  // Build editable config from stored values
  const tc = exp.target_config || {};
  const retryJudgeMode = tc.judge_mode || (
    tc.attacker_api_base_url || tc.attacker_model ? "attacker" : "target"
  );
  const retryJudgeLocalChecked = tc.judge_local === true || tc.judge_local === "true" || tc.judge_api_key === "lm-studio";

  // Show review dialog
  const overlay = buildRetryOverlay(exp, tc, retryCount, errorCount, missingCount, cells, retryJudgeMode, retryJudgeLocalChecked);
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  overlay.querySelector("#retry-cancel").addEventListener("click", close);
  overlay.querySelector("#retry-cancel-btn").addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  const retryJudgeModeSelect = overlay.querySelector("#retry-judge-mode");
  const retryJudgeFields = overlay.querySelector("#retry-judge-custom-fields");
  const retryJudgeLocal = overlay.querySelector("#retry-judge-local");
  const retryJudgeKeyField = overlay.querySelector("#retry-judge-key-field");
  const retryJudgeBaseUrl = overlay.querySelector("#retry-judge-base-url");
  const updateRetryJudgeVisibility = () => {
    retryJudgeFields.style.display = retryJudgeModeSelect.value === "custom" ? "" : "none";
  };
  retryJudgeModeSelect.addEventListener("change", updateRetryJudgeVisibility);
  retryJudgeLocal.addEventListener("change", () => {
    const isLocal = retryJudgeLocal.checked;
    retryJudgeKeyField.style.display = isLocal ? "none" : "";
    if (isLocal && !retryJudgeBaseUrl.value) retryJudgeBaseUrl.value = "http://localhost:1234/v1";
  });

  overlay.querySelector("#retry-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const apiKey = overlay.querySelector("#retry-api-key").value.trim();
    if (!apiKey) { alert("API key is required."); return; }

    const attackerKey = overlay.querySelector("#retry-attacker-key").value.trim() || apiKey;
    const retryJudgeMode = overlay.querySelector("#retry-judge-mode").value;
    const retryJudgeIsLocal = overlay.querySelector("#retry-judge-local").checked;
    const retryJudgeKey = retryJudgeIsLocal
      ? "lm-studio"
      : overlay.querySelector("#retry-judge-key").value.trim();

    const targetConfig = {
      api_key: apiKey,
      api_base_url: overlay.querySelector("#retry-base-url").value.trim(),
      model: overlay.querySelector("#retry-model").value.trim(),
      system_prompt: overlay.querySelector("#retry-system-prompt").value.trim(),
      attacker_api_key: attackerKey,
      attacker_api_base_url: overlay.querySelector("#retry-attacker-base-url").value.trim(),
      attacker_model: overlay.querySelector("#retry-attacker-model").value.trim(),
      judge_mode: retryJudgeMode,
      judge_local: retryJudgeIsLocal,
      judge_api_key: retryJudgeKey,
      judge_api_base_url: overlay.querySelector("#retry-judge-base-url").value.trim(),
      judge_model: overlay.querySelector("#retry-judge-model").value.trim(),
    };

    overlay.querySelector("#retry-confirm-btn").disabled = true;
    overlay.querySelector("#retry-confirm-btn").textContent = "Starting...";

    try {
      await fetchJson(`/api/experiments/${experimentId}/retry`, {
        method: "POST",
        body: JSON.stringify({ target_config: targetConfig }),
      });
      close();
      loadExperiments();
      trackExperiment(experimentId);
      switchTab("history");
    } catch (err) {
      alert("Failed to retry: " + err.message);
      overlay.querySelector("#retry-confirm-btn").disabled = false;
      overlay.querySelector("#retry-confirm-btn").textContent = `Retry ${retryCount} Cells`;
    }
  });
}

async function loadExperiment(experimentId) {
  try {
    const data = await fetchJson(`/api/experiments/${experimentId}`);
    state.currentExperiment = data.item;
    state.currentExperimentId = experimentId;
    switchTab("results");
    const emptyMsg = document.getElementById("exp-results-empty");
    if (emptyMsg) emptyMsg.style.display = "none";
    el.results.style.display = "";
    renderResults(data.item);
    selectMetricsExperiment(data.item);
    await loadMetrics();
  } catch (err) {
    alert("Failed to load experiment: " + err.message);
  }
}

function selectMetricsExperiment(exp) {
  if (!el.metricsExpSelect || !exp?.experiment_id) return;
  populateMetricsDropdown();
  const eid = exp.experiment_id;
  const hasOption = [...el.metricsExpSelect.options].some((option) => option.value === eid);
  if (!hasOption) {
    const completed = exp.completed_cells ?? exp.report?.tested_cells ?? 0;
    const total = exp.total_cells ?? exp.report?.total_cells ?? completed;
    const option = document.createElement("option");
    option.value = eid;
    option.textContent = `${exp.name || eid} (${completed}/${total} cells)`;
    el.metricsExpSelect.appendChild(option);
  }
  el.metricsExpSelect.value = eid;
}

function truncate(str, max) {
  return str.length > max ? str.slice(0, max - 1) + "\u2026" : str;
}

// ---------------------------------------------------------------------------
// Data Explorer
// ---------------------------------------------------------------------------

const dataEl = {
  viewsList: document.getElementById("data-views-list"),
  queryInput: document.getElementById("data-query-input"),
  queryRun: document.getElementById("data-query-run"),
  resultsMeta: document.getElementById("data-results-meta"),
  resultsWrap: document.getElementById("data-results-wrap"),
};

async function runDataQuery(sql) {
  dataEl.resultsMeta.textContent = "Running...";
  dataEl.resultsWrap.replaceChildren();
  try {
    const data = await fetchJson("/api/analysis/query", {
      method: "POST",
      body: JSON.stringify({ sql }),
    });
    const cols = data.columns || [];
    const items = data.items || [];
    dataEl.resultsMeta.textContent = `${items.length} row${items.length !== 1 ? "s" : ""}`;
    if (items.length === 0) {
      renderMutedMessage(dataEl.resultsWrap, "No rows returned.");
      return;
    }
    dataEl.resultsWrap.replaceChildren(renderDataQueryTable(cols, items));
  } catch (err) {
    dataEl.resultsMeta.textContent = "Error";
    dataEl.resultsWrap.replaceChildren(renderDataError(err.message));
  }
}

function renderDataQueryTable(columns, items) {
  const table = document.createElement("table");
  table.className = "breakdown-table data-table";
  table.append(renderDataQueryHead(columns), renderDataQueryBody(columns, items));
  return table;
}

function renderDataQueryHead(columns) {
  const thead = document.createElement("thead");
  const row = document.createElement("tr");
  columns.forEach((column) => {
    row.append(textNodeElement("th", "", column));
  });
  thead.append(row);
  return thead;
}

function renderDataQueryBody(columns, items) {
  const tbody = document.createElement("tbody");
  items.forEach((item) => {
    const row = document.createElement("tr");
    columns.forEach((column) => {
      row.append(renderDataQueryCell(item[column]));
    });
    tbody.append(row);
  });
  return tbody;
}

function renderDataQueryCell(value) {
  const display = formatDataValue(value);
  const cell = textNodeElement("td", "", truncate(display, 120));
  cell.title = display;
  return cell;
}

function formatDataValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function renderDataError(message) {
  const error = textNodeElement("p", "text-muted", message);
  error.style.color = "var(--accent)";
  return error;
}

// View buttons
dataEl.viewsList.querySelectorAll(".data-view-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    dataEl.viewsList.querySelectorAll(".data-view-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const sql = btn.dataset.query;
    dataEl.queryInput.value = sql;
    runDataQuery(sql);
  });
});

// Custom query
dataEl.queryRun.addEventListener("click", () => {
  const sql = dataEl.queryInput.value.trim();
  if (!sql) return;
  dataEl.viewsList.querySelectorAll(".data-view-btn").forEach((b) => b.classList.remove("active"));
  runDataQuery(sql);
});

dataEl.queryInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    dataEl.queryRun.click();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

const saved = loadSavedState();

vaultControl = mountVaultControl(el.sidebarVaultRoot, {
  fetchJson,
  onChange: handleVaultChange,
});

renderExperimentSidebarNav();
loadTargetConfig(saved?.targetConfig);
restoreFormState(saved);
wireAutoSave();

// ---------------------------------------------------------------------------
// Metrics tab
// ---------------------------------------------------------------------------

function populateMetricsDropdown() {
  if (!el.metricsExpSelect) return;
  const prev = el.metricsExpSelect.value;
  const options = [
    createMetricsOption("", "-- Choose an experiment --"),
    ...state.experiments
      .filter((experiment) => experiment.completed_cells > 0)
      .map(createMetricsExperimentOption),
  ];
  el.metricsExpSelect.replaceChildren(...options);
  if (prev) el.metricsExpSelect.value = prev;
}

function createMetricsExperimentOption(experiment) {
  const status = experiment.status === "completed" ? "" : ` [${experiment.status}]`;
  return createMetricsOption(
    experiment.experiment_id,
    `${experiment.name} (${experiment.completed_cells}/${experiment.total_cells} cells)${status}`,
  );
}

function createMetricsOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

let _metricsCache = null;

async function loadMetrics() {
  const eid = el.metricsExpSelect?.value;
  if (!eid) return;
  el.metricsBody.replaceChildren(textNodeElement("p", "text-muted", "Loading metrics..."));
  try {
    const m = await fetchJson(`/api/experiments/${eid}/metrics`);
    _metricsCache = { eid, data: m };
    renderMetrics(m);
    el.downloadFullCsvBtn.style.display = "";
    el.downloadMetricsCsvBtn.style.display = "";
  } catch (err) {
    el.metricsBody.replaceChildren(textNodeElement("p", "text-error", `Error: ${err.message}`));
  }
}

async function loadSelectedExperimentResults() {
  const eid = el.metricsExpSelect?.value;
  if (!eid) return;
  await loadExperiment(eid);
}

function renderMetrics(m) {
  const techniqueMetrics = m.technique_metrics || {};
  const categoryMetrics = m.category_metrics || {};
  const categoryTechniqueMetrics = m.category_technique_metrics || {};
  const nodes = [renderMetricsMeta(m)];

  nodes.push(
    ...renderMetricsSection(
      "1. Per-Technique Metrics",
      renderMetricTable(
        [
          "Technique", "Tested", "Success", "Binary ASR", "95% CI", "Avg Score", "Score SD",
          "Avg Rounds", "Success Rounds", "Backtrack %",
          "Real Judge Avg",
        ],
        Object.entries(techniqueMetrics).map(([technique, values]) => metricTechniqueRow(technique, values)),
      ),
    ),
  );

  nodes.push(
    ...renderMetricsSection(
      "2. Per-Category Metrics",
      renderMetricTable(
        ["Category", "Tested", "Successes", "Binary ASR", "95% CI", "Avg Score", "Score SD"],
        Object.entries(categoryMetrics).map(([category, values]) => metricCategoryRow(category, values)),
      ),
    ),
  );

  const techniques = Object.keys(techniqueMetrics);
  const categories = Object.keys(categoryTechniqueMetrics);
  if (categories.length > 0 && techniques.length > 0) {
    nodes.push(
      ...renderMetricsSection(
        "3. Category x Technique Heatmap (Binary ASR %)",
        renderMetricTable(
          [
            "Category",
            ...techniques.map((technique) => ({
              className: "metric-technique-name",
              text: displayTechniqueKey(technique).replace("crescendo_", "c_"),
            })),
          ],
          categories.map((category) => metricCategoryTechniqueRow(category, techniques, categoryTechniqueMetrics)),
        ),
      ),
    );
  }

  el.metricsBody.replaceChildren(...nodes);
}

function renderMetricsMeta(metrics) {
  const meta = document.createElement("div");
  meta.className = "metrics-meta";
  const items = [
    ["Binary ASR:", `${metrics.overall_binary_asr}% (${formatCi(metrics.overall_asr_ci_low, metrics.overall_asr_ci_high)})`],
    ["Avg Score:", `${metrics.overall_avg_score} (SD ${metrics.overall_score_stddev ?? 0})`],
    ["Successes:", `${metrics.total_successes}/${metrics.tested_cells ?? metrics.total_cells}`],
    ["Target:", metrics.target_model],
    ["Attacker:", metrics.attacker_model],
    ["Dataset:", metrics.dataset_name],
    ["Repeats:", metrics.repeat_count ?? 1],
    ["Rounds:", metrics.max_rounds ?? 5],
  ];
  if (metrics.experiment_group) items.push(["Group:", metrics.experiment_group]);
  meta.replaceChildren(...items.map(([label, value]) => metricMetaItem(label, value)));
  return meta;
}

function metricMetaItem(label, value) {
  const item = document.createElement("span");
  const strong = textNodeElement("strong", "", label);
  item.append(strong, document.createTextNode(` ${value ?? ""}`));
  return item;
}

function renderMetricsSection(title, table) {
  const header = document.createElement("div");
  header.className = "panel-header";
  header.append(textNodeElement("h2", "", title));

  const wrapper = document.createElement("div");
  wrapper.className = "breakdown-table-wrap";
  wrapper.append(table);
  return [header, wrapper];
}

function renderMetricTable(headers, rows) {
  const table = document.createElement("table");
  table.className = "breakdown-table";
  table.append(renderMetricHead(headers), renderMetricBody(rows));
  return table;
}

function renderMetricHead(headers) {
  const thead = document.createElement("thead");
  const row = document.createElement("tr");
  headers.forEach((header) => row.append(metricCell("th", header)));
  thead.append(row);
  return thead;
}

function renderMetricBody(rows) {
  const tbody = document.createElement("tbody");
  rows.forEach((cells) => {
    const row = document.createElement("tr");
    cells.forEach((cell) => row.append(metricCell("td", cell)));
    tbody.append(row);
  });
  return tbody;
}

function metricCell(tag, spec) {
  const cellSpec = metricCellSpec(spec);
  return textNodeElement(tag, cellSpec.className || "", cellSpec.text);
}

function metricCellSpec(spec) {
  if (spec && typeof spec === "object" && Object.prototype.hasOwnProperty.call(spec, "text")) {
    return spec;
  }
  return { text: spec ?? "" };
}

function metricTechniqueRow(technique, values) {
  const avgSuccessRounds = values.avg_success_rounds != null ? values.avg_success_rounds : "N/A";
  const realJudgeAvg = values.real_judge_avg != null ? values.real_judge_avg : "N/A";
  return [
    { className: "metric-technique-name", text: displayTechniqueKey(technique) },
    values.tested,
    values.successes,
    { className: metricRateClass(values.binary_asr), text: `${values.binary_asr}%` },
    formatCi(values.asr_ci_low, values.asr_ci_high),
    values.avg_score,
    values.score_stddev ?? 0,
    values.avg_rounds,
    avgSuccessRounds,
    `${values.backtrack_rate}%`,
    realJudgeAvg,
  ];
}

function metricCategoryRow(category, values) {
  return [
    category,
    values.tested,
    values.successes,
    { className: metricRateClass(values.binary_asr), text: `${values.binary_asr}%` },
    formatCi(values.asr_ci_low, values.asr_ci_high),
    values.avg_score,
    values.score_stddev ?? 0,
  ];
}

function metricCategoryTechniqueRow(category, techniques, categoryTechniqueMetrics) {
  return [
    category,
    ...techniques.map((technique) => {
      const cell = categoryTechniqueMetrics[category]?.[technique];
      if (!cell) return "-";
      return { className: metricRateClass(cell.binary_asr), text: `${cell.binary_asr}%` };
    }),
  ];
}

function metricRateClass(value) {
  if (value >= 70) return "metric-high";
  if (value > 0) return "metric-mid";
  return "metric-low";
}

function downloadFullCsv() {
  const eid = el.metricsExpSelect?.value;
  if (!eid) return;
  window.open(`${API}/api/experiments/${eid}/export-full`, "_blank");
}

function generateMetricsCsv() {
  if (!_metricsCache) return;
  const m = _metricsCache.data;
  const tm = m.technique_metrics || {};
  const lines = [];

  // Header
	  lines.push([
	    "technique_key", "tested", "successes", "binary_asr_pct",
    "asr_ci_low_pct", "asr_ci_high_pct", "avg_score", "score_stddev",
    "min_score", "max_score", "avg_rounds", "avg_success_rounds",
    "min_rounds", "max_rounds", "backtrack_count", "backtrack_rate_pct",
    "real_judge_avg",
  ].join(","));

	  for (const [t, v] of Object.entries(tm)) {
	    lines.push([
	      `"${t}"`, v.tested, v.successes, v.binary_asr,
      v.asr_ci_low ?? "", v.asr_ci_high ?? "", v.avg_score, v.score_stddev ?? "",
      v.min_score, v.max_score, v.avg_rounds, v.avg_success_rounds ?? "",
      v.min_rounds, v.max_rounds, v.backtrack_count, v.backtrack_rate,
      v.real_judge_avg ?? "",
    ].join(","));
  }

  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${m.experiment_name || m.experiment_id}-metrics.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

if (el.loadMetricsBtn) el.loadMetricsBtn.addEventListener("click", loadSelectedExperimentResults);
if (el.downloadFullCsvBtn) el.downloadFullCsvBtn.addEventListener("click", downloadFullCsv);
if (el.downloadMetricsCsvBtn) el.downloadMetricsCsvBtn.addEventListener("click", generateMetricsCsv);

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async () => {
  await Promise.all([loadVaultState(), loadTechniques(), loadDatasets(), loadExperiments()]);
  loadTargetConfig(saved?.targetConfig);

  // Restore technique selections after techniques are loaded
  restoreTechniqueSelections(saved);

  // Restore dataset selection after datasets are loaded
  if (state.selectedDatasetId) {
    const ds = state.datasets.find((d) => d.dataset_id === state.selectedDatasetId);
    if (ds) {
      el.datasetSelected.hidden = false;
      el.selectedDatasetTag.textContent = `${ds.name} (${ds.prompt_count} prompts)`;
      renderDatasetList();
    }
  }

  // Restore current experiment results if we were viewing one
  if (saved?.currentExperimentId && state.activeTab === "results") {
    loadExperiment(saved.currentExperimentId);
  }

  // Populate metrics dropdown
  populateMetricsDropdown();
})();
