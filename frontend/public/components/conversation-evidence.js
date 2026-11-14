export function hasJudgeScore(value) {
  return value !== null && value !== undefined && value !== "" && value !== "?";
}

export function renderJsonArtifact(response, {
  createElement,
  mediaType = "application/json",
  maxNodes = 1000,
} = {}) {
  if (mediaType !== "application/json" || typeof response !== "string") return null;
  let value;
  try { value = JSON.parse(response); } catch { return null; }
  const state = { remaining: maxNodes };
  return renderJsonValue(value, { createElement, state, label: null, open: true });
}

function renderJsonValue(value, { createElement, state, label, open }) {
  if (state.remaining <= 0) {
    return renderOmissionMarker(createElement, state, label === null);
  }
  state.remaining -= 1;

  if (value !== null && typeof value === "object") {
    const entries = Array.isArray(value)
      ? value.map((item, index) => [String(index), item])
      : Object.entries(value);
    const typeLabel = Array.isArray(value)
      ? `Array (${entries.length} items)`
      : `Object (${entries.length} keys)`;
    const node = createElement(
      "details",
      label === null ? "json-artifact-tree json-artifact-node" : "json-artifact-node",
    );
    node.open = open;
    node.title = typeLabel;

    const summary = createElement("summary");
    if (label !== null) summary.append(createElement("span", "json-artifact-key", label));
    summary.append(createElement("span", "json-artifact-scalar", typeLabel));
    node.append(summary);

    for (const [key, item] of entries) {
      if (state.remaining <= 0) {
        if (!state.omissionRendered) {
          node.append(renderOmissionMarker(createElement, state));
        }
        break;
      }
      node.append(renderJsonValue(item, {
        createElement,
        state,
        label: key,
        open: false,
      }));
    }
    return node;
  }

  const node = createElement(
    "div",
    label === null ? "json-artifact-tree json-artifact-node" : "json-artifact-node",
  );
  if (label !== null) {
    node.append(createElement(
      "span",
      "json-artifact-key",
      value === null ? `${label} ` : label,
    ));
  }
  node.append(createElement("span", "json-artifact-scalar", formatJsonScalar(value)));
  return node;
}

function formatJsonScalar(value) {
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  return String(value);
}

function renderOmissionMarker(createElement, state, root = false) {
  state.omissionRendered = true;
  return createElement(
    "div",
    root ? "json-artifact-tree json-artifact-scalar" : "json-artifact-scalar",
    "… remaining values omitted",
  );
}

export function renderCanonicalTreeBranch(branch, options) {
  const {
    createElement,
    itemClass = "",
    headerClass = "",
    metadataClass = "",
    renderPrompt = () => null,
    renderResponse = () => null,
    renderJudge = () => null,
  } = options;
  if (typeof createElement !== "function") {
    throw new TypeError("canonical tree renderer requires createElement");
  }

  const item = createElement("div", itemClass);
  const identity = branch.branch_id || branch.candidate_id || "branch";
  const status = branch.status || "completed";
  item.append(createElement("div", headerClass, `${identity} · ${status}`));

  const provenance = [];
  if (branch.parent_id !== null && branch.parent_id !== undefined && branch.parent_id !== "") {
    provenance.push(`parent: ${branch.parent_id}`);
  }
  if (branch.origin !== null && branch.origin !== undefined && branch.origin !== "") {
    provenance.push(`origin: ${branch.origin}`);
  }
  for (const [field, label] of [
    ["is_retained", "retained"],
    ["is_pruned", "pruned"],
    ["is_selected", "selected"],
  ]) {
    if (typeof branch[field] === "boolean") {
      provenance.push(`${label}: ${branch[field] ? "yes" : "no"}`);
    }
  }
  if (provenance.length) {
    item.append(createElement("div", metadataClass, provenance.join(" · ")));
  }

  appendRendered(item, renderPrompt(branch.prompt || ""));
  appendRendered(item, renderResponse(branch.response || ""));
  const judge = branch.judge || {};
  if (
    hasJudgeScore(judge.score)
    || Boolean(judge.error || judge.rationale || judge.output)
  ) {
    appendRendered(item, renderJudge(judge));
  }
  return item;
}

function appendRendered(parent, value) {
  const values = Array.isArray(value) ? value.flat() : [value];
  for (const item of values) {
    if (item) parent.append(item);
  }
}
