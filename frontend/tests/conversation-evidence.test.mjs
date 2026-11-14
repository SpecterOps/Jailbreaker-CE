import assert from "node:assert/strict";
import { test } from "node:test";

import {
  hasJudgeScore,
  renderCanonicalTreeBranch,
  renderJsonArtifact,
} from "../public/components/conversation-evidence.js";

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.className = "";
    this.children = [];
    this._textContent = "";
    this.open = false;
    this.title = "";
  }

  append(...children) {
    this.children.push(...children.filter(Boolean));
  }

  set textContent(value) {
    this._textContent = String(value ?? "");
  }

  get textContent() {
    return [
      this._textContent,
      ...this.children.map((child) => child.textContent || String(child)),
    ].join("");
  }
}

function element(tag, className = "", text = null) {
  const node = new FakeElement(tag);
  node.className = className;
  if (text !== null && text !== undefined) node.textContent = text;
  return node;
}

test("JSON artifact explorer renders nested keys and presence-sensitive scalars", () => {
  const node = renderJsonArtifact(
    JSON.stringify({
      tools: [{ name: "search", enabled: true }],
      score: 0,
      disabled: false,
      empty: "",
      note: null,
    }),
    { createElement: element, maxNodes: 50 },
  );

  assert.match(node.textContent, /tools/);
  assert.match(node.textContent, /search/);
  assert.match(node.textContent, /score0/);
  assert.match(node.textContent, /disabledfalse/);
  assert.match(node.textContent, /empty""/);
  assert.match(node.textContent, /note null/);
  assert.equal(renderJsonArtifact("not-json", { createElement: element }), null);
  assert.equal(
    renderJsonArtifact("[]", { createElement: element, mediaType: "text/plain" }),
    null,
  );
});

test("JSON artifact explorer caps rendered values with one omission marker", () => {
  const node = renderJsonArtifact(
    JSON.stringify({ first: { nested: { value: 1, extra: 2 } }, second: 2 }),
    { createElement: element, maxNodes: 3 },
  );

  assert.equal(
    node.textContent,
    "Object (2 keys)firstObject (1 keys)nestedObject (2 keys)… remaining values omitted",
  );
});

test("canonical tree branch DOM preserves zero-score and provenance evidence", () => {
  const seenJudges = [];
  const node = renderCanonicalTreeBranch(
    {
      branch_id: "branch-child",
      parent_id: "branch-root",
      origin: "mutation",
      status: "pruned",
      is_retained: false,
      is_pruned: true,
      is_selected: false,
      prompt: "candidate prompt",
      response: "candidate response",
      judge: { score: 0, rationale: "", output: "" },
    },
    {
      createElement: element,
      itemClass: "tree-item",
      headerClass: "tree-header",
      metadataClass: "tree-metadata",
      renderPrompt: (value) => element("p", "tree-prompt", value),
      renderResponse: (value) => element("pre", "tree-response", value),
      renderJudge: (judge) => {
        seenJudges.push(judge);
        return element("span", "tree-judge", `Judge: ${judge.score}/10`);
      },
    },
  );

  assert.equal(hasJudgeScore(0), true);
  assert.equal(hasJudgeScore(null), false);
  assert.deepEqual(seenJudges, [{ score: 0, rationale: "", output: "" }]);
  assert.match(node.textContent, /branch-child · pruned/);
  assert.match(node.textContent, /parent: branch-root/);
  assert.match(node.textContent, /origin: mutation/);
  assert.match(node.textContent, /retained: no/);
  assert.match(node.textContent, /pruned: yes/);
  assert.match(node.textContent, /selected: no/);
  assert.match(node.textContent, /candidate prompt/);
  assert.match(node.textContent, /candidate response/);
  assert.match(node.textContent, /Judge: 0\/10/);
});
