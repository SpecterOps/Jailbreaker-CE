const DEFAULT_VAULT = { exists: false, unlocked: false, profiles: [] };

export function mountVaultControl(root, { fetchJson, onChange = () => {}, documentRef = document } = {}) {
  if (!root) {
    return { open() {}, refresh() {}, getState: () => ({ ...DEFAULT_VAULT }) };
  }
  if (typeof fetchJson !== "function") {
    throw new TypeError("mountVaultControl requires fetchJson");
  }

  const state = {
    vault: { ...DEFAULT_VAULT },
    loaded: false,
    open: false,
    message: "",
    error: "",
    requestVersion: 0,
  };
  const dialogHost = documentRef.body || root;
  let dialogOverlay = null;

  async function refresh() {
    const wasLoaded = state.loaded;
    const requestVersion = ++state.requestVersion;
    let vault = { ...DEFAULT_VAULT };
    try {
      vault = await fetchJson("/api/llm-vault");
    } catch {
      vault = { ...DEFAULT_VAULT };
    }
    if (requestVersion !== state.requestVersion) {
      return state.vault;
    }
    state.vault = vault;
    state.error = "";
    state.loaded = true;
    onChange(state.vault);
    if (!state.open || !wasLoaded) {
      render(state.open ? "first-input" : "");
    }
    return state.vault;
  }

  function open(message = "") {
    state.open = true;
    state.message = message;
    state.error = "";
    render(state.loaded ? "first-input" : "dialog");
  }

  function close() {
    state.open = false;
    state.message = "";
    state.error = "";
    render("button");
  }

  async function createVault(passphrase, confirmPassphrase) {
    if (!passphrase.trim()) {
      state.error = "Vault passphrase is required.";
      render();
      return;
    }
    if (passphrase !== confirmPassphrase) {
      state.error = "Passphrases do not match.";
      render();
      return;
    }
    await submitVaultAction("/api/llm-vault/create", { passphrase });
  }

  async function unlockVault(passphrase) {
    if (!passphrase.trim()) {
      state.error = "Vault passphrase is required.";
      render();
      return;
    }
    await submitVaultAction("/api/llm-vault/unlock", { passphrase });
  }

  async function lockVault() {
    await submitVaultAction("/api/llm-vault/lock", {});
  }

  async function submitVaultAction(path, payload) {
    const requestVersion = ++state.requestVersion;
    try {
      const vault = await fetchJson(path, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (requestVersion !== state.requestVersion) {
        return;
      }
      state.vault = vault;
      state.open = false;
      state.message = "";
      state.error = "";
      onChange(state.vault);
    } catch (err) {
      if (requestVersion !== state.requestVersion) {
        return;
      }
      state.error = err.message || "Vault action failed.";
    }
    render(state.open ? "first-input" : "button");
  }

  function render(focusTarget = "") {
    removeDialogOverlay();
    root.replaceChildren(renderVaultButton(documentRef, state, open));
    if (state.open) {
      dialogOverlay = renderVaultDialog(documentRef, state, { close, createVault, unlockVault, lockVault });
      dialogHost.append(dialogOverlay);
    }
    focusAfterRender(focusTarget);
  }

  function removeDialogOverlay() {
    if (!dialogOverlay) return;
    dialogOverlay.remove();
    dialogOverlay = null;
  }

  function focusAfterRender(target) {
    let next = null;
    if (target === "button") {
      next = root.querySelector(".vault-sidebar-button");
    }
    if (target === "first-input") {
      next = dialogOverlay?.querySelector(".vault-dialog input") || dialogOverlay?.querySelector(".vault-dialog button");
    }
    if (target === "dialog") {
      next = dialogOverlay?.querySelector(".vault-dialog");
    }
    if (next && typeof next.focus === "function") {
      next.focus();
    }
  }

  render();
  refresh();

  return {
    open,
    refresh,
    getState: () => ({ ...state.vault }),
  };
}

function renderVaultButton(documentRef, state, open) {
  const label = vaultButtonLabel(state.vault);
  const button = documentRef.createElement("button");
  button.type = "button";
  button.className = "vault-sidebar-button";
  button.setAttribute("aria-haspopup", "dialog");
  button.setAttribute("aria-label", label);
  button.setAttribute("title", label);
  button.addEventListener("click", () => open());

  button.append(renderVaultIcon(documentRef, state.vault));
  return button;
}

function renderVaultIcon(documentRef, vault) {
  const icon = documentRef.createElement("span");
  icon.className = `vault-icon ${vaultStatusKey(vault)}`;
  icon.setAttribute("aria-hidden", "true");

  const shackle = documentRef.createElement("span");
  shackle.className = "vault-icon-shackle";

  const door = documentRef.createElement("span");
  door.className = "vault-icon-door";

  const dial = documentRef.createElement("span");
  dial.className = "vault-icon-dial";

  icon.append(shackle, door, dial);
  return icon;
}

function renderVaultDialog(documentRef, state, actions) {
  const overlay = documentRef.createElement("div");
  overlay.className = "vault-modal-overlay";
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) actions.close();
  });
  overlay.addEventListener("keydown", (event) => {
    if (event.key === "Escape") actions.close();
  });

  const dialog = documentRef.createElement("section");
  dialog.className = "vault-dialog";
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  dialog.setAttribute("aria-labelledby", "vault-dialog-title");
  dialog.tabIndex = -1;

  const title = documentRef.createElement("h2");
  title.id = "vault-dialog-title";
  title.textContent = "LLM Vault";

  const closeButton = documentRef.createElement("button");
  closeButton.type = "button";
  closeButton.className = "vault-dialog-close";
  closeButton.setAttribute("aria-label", "Close vault dialog");
  closeButton.textContent = "x";
  closeButton.addEventListener("click", actions.close);

  const header = documentRef.createElement("div");
  header.className = "vault-dialog-header";
  header.append(title, closeButton);

  const body = documentRef.createElement("div");
  body.className = "vault-dialog-body";
  body.append(renderModeContent(documentRef, state, actions));

  dialog.append(header, body);
  overlay.append(dialog);
  return overlay;
}

function renderModeContent(documentRef, state, actions) {
  if (!state.loaded) return renderLoadingMode(documentRef, state);
  if (!state.vault.exists) return renderCreateMode(documentRef, state, actions);
  if (!state.vault.unlocked) return renderUnlockMode(documentRef, state, actions);
  return renderUnlockedMode(documentRef, state, actions);
}

function renderLoadingMode(documentRef, state) {
  const wrap = documentRef.createElement("div");
  wrap.className = "vault-mode vault-mode-loading";
  wrap.setAttribute("data-vault-mode", "loading");
  const message = documentRef.createElement("p");
  message.className = "vault-message";
  message.textContent = state.message || "Loading vault state...";
  wrap.append(message);
  return wrap;
}

function renderCreateMode(documentRef, state, actions) {
  const form = vaultForm(documentRef, state, "create");
  const passphrase = passwordField(documentRef, "vault-create-passphrase", "Passphrase", "new-password");
  const confirm = passwordField(documentRef, "vault-create-confirm", "Confirm Passphrase", "new-password");
  const submit = actionButton(documentRef, "Create Vault");
  form.append(passphrase.wrap, confirm.wrap, submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    actions.createVault(passphrase.input.value, confirm.input.value);
    passphrase.input.value = "";
    confirm.input.value = "";
  });
  return form;
}

function renderUnlockMode(documentRef, state, actions) {
  const form = vaultForm(documentRef, state, "unlock");
  const passphrase = passwordField(documentRef, "vault-unlock-passphrase", "Passphrase");
  const submit = actionButton(documentRef, "Unlock Vault");
  form.append(passphrase.wrap, submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    actions.unlockVault(passphrase.input.value);
    passphrase.input.value = "";
  });
  return form;
}

function renderUnlockedMode(documentRef, state, actions) {
  const wrap = documentRef.createElement("div");
  wrap.className = "vault-mode vault-mode-unlocked";
  wrap.setAttribute("data-vault-mode", "unlocked");

  const summary = documentRef.createElement("p");
  const count = state.vault.profiles?.length || 0;
  summary.textContent = `Vault unlocked. ${count} saved profile${count === 1 ? "" : "s"} available.`;

  const button = actionButton(documentRef, "Lock Vault");
  button.addEventListener("click", () => actions.lockVault());

  wrap.append(statusMessage(documentRef, state), summary, button);
  return wrap;
}

function vaultForm(documentRef, state, mode) {
  const form = documentRef.createElement("form");
  form.className = mode === "create" ? "vault-mode vault-mode-create" : "vault-mode vault-mode-unlock";
  form.setAttribute("data-vault-mode", mode);
  form.append(statusMessage(documentRef, state));
  return form;
}

function passwordField(documentRef, id, labelText, autocomplete = "current-password") {
  const wrap = documentRef.createElement("label");
  wrap.className = "vault-field";
  wrap.setAttribute("for", id);

  const label = documentRef.createElement("span");
  label.textContent = labelText;

  const input = documentRef.createElement("input");
  input.id = id;
  input.type = "password";
  input.autocomplete = autocomplete;
  input.setAttribute("autocomplete", autocomplete);

  wrap.append(label, input);
  return { wrap, input };
}

function actionButton(documentRef, text) {
  const button = documentRef.createElement("button");
  button.type = "submit";
  button.className = "btn btn-primary btn-sm";
  button.textContent = text;
  return button;
}

function statusMessage(documentRef, state) {
  const message = documentRef.createElement("p");
  message.className = state.error ? "vault-message error" : "vault-message";
  message.textContent = state.error || state.message || vaultStatusSentence(state.vault);
  return message;
}

function vaultStatusKey(vault) {
  if (!vault.exists) return "not-set";
  return vault.unlocked ? "unlocked" : "locked";
}

function vaultStatusLabel(vault) {
  if (!vault.exists) return "Not set";
  return vault.unlocked ? "Unlocked" : "Locked";
}

function vaultButtonLabel(vault) {
  return `LLM Vault: ${vaultStatusLabel(vault)}`;
}

function vaultStatusSentence(vault) {
  if (!vault.exists) return "Create a vault to save encrypted LLM profile API keys.";
  if (!vault.unlocked) return "Unlock the vault to save or use encrypted profile API keys.";
  return "Vault is unlocked for saved LLM profiles.";
}
