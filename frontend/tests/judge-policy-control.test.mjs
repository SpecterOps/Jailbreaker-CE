import assert from "node:assert/strict";
import { test } from "node:test";

import {
  JUDGE_POLICY_OUTCOMES,
  JUDGE_POLICY_RUBRIC,
  JUDGE_POLICY_STANDARD,
  configuredJudgePolicy,
  judgePolicyPreviewText,
  judgePolicyValidationMessage,
  normalizeJudgePolicy,
} from "../public/components/judge-policy-control.js";

test("Judge policy defaults to the backward-compatible standard mode", () => {
  assert.deepEqual(normalizeJudgePolicy({ mode: "unknown" }), {
    version: 1,
    mode: JUDGE_POLICY_STANDARD,
    success_definition: "",
    failure_definition: "",
    rubric: "",
  });
  assert.equal(configuredJudgePolicy({ mode: JUDGE_POLICY_STANDARD }), null);
});

test("outcome policies require both success and failure definitions", () => {
  assert.equal(
    judgePolicyValidationMessage({ mode: JUDGE_POLICY_OUTCOMES }),
    "Define what success looks like before running.",
  );
  assert.equal(
    judgePolicyValidationMessage({ mode: JUDGE_POLICY_OUTCOMES, success_definition: "Useful result" }),
    "Define what failure looks like before running.",
  );
  assert.equal(
    judgePolicyValidationMessage({
      mode: JUDGE_POLICY_OUTCOMES,
      success_definition: "Useful result",
      failure_definition: "Refusal",
    }),
    "",
  );
});

test("custom rubric mode requires a rubric and preserves the scoring contract", () => {
  assert.equal(
    judgePolicyValidationMessage({ mode: JUDGE_POLICY_RUBRIC }),
    "Provide the custom judge rubric before running.",
  );
  const policy = configuredJudgePolicy({ mode: JUDGE_POLICY_RUBRIC, rubric: "  Score factual citations.  " });
  assert.deepEqual(policy, {
    version: 1,
    mode: JUDGE_POLICY_RUBRIC,
    success_definition: "",
    failure_definition: "",
    rubric: "Score factual citations.",
  });
  assert.match(judgePolicyPreviewText(policy), /Normalize the result to 1-10/);
  assert.match(judgePolicyPreviewText(policy), /REQUIRED JSON OUTPUT/);
  assert.match(judgePolicyPreviewText(policy), /"score": 7/);
  assert.match(judgePolicyPreviewText(policy), /"verdict": "success"/);
  assert.match(judgePolicyPreviewText(policy), /"success_criteria_met": true/);
  assert.match(judgePolicyPreviewText(policy), /"failure_criteria_met": false/);
  assert.match(judgePolicyPreviewText(policy), /"evidence": \[/);
  assert.match(judgePolicyPreviewText(policy), /"missing_requirements": \[/);
  assert.match(judgePolicyPreviewText(policy), /"rationale":/);
});

test("Judge policy validation enforces backend length limits", () => {
  assert.equal(
    judgePolicyValidationMessage({
      mode: JUDGE_POLICY_OUTCOMES,
      success_definition: "s".repeat(4001),
      failure_definition: "failure",
    }),
    "Success criteria must be 4,000 characters or fewer.",
  );
  assert.equal(
    judgePolicyValidationMessage({
      mode: JUDGE_POLICY_RUBRIC,
      rubric: "r".repeat(12001),
    }),
    "The custom judge rubric must be 12,000 characters or fewer.",
  );
});
