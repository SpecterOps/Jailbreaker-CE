from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

from app.api.store import EvaluationStore
from app.attacklib.base import AttackBuildRequest
from app.attacklib.code_agent_attacks import (
    CONTEXT_POISONING_TRUSTED_TEMPLATE,
    CONTEXT_POISONING_WORKSPACE_FILES,
    ContextPoisoningAttack,
)
from app.attacklib.iterative.prompts import JUDGE_SCHEMA, JUDGE_SYSTEM, JUDGE_USER
from app.attacklib.renderer import render_prompt
from app.judge_policy import _extract_standard_payload
from app.persistence import NoopRunPersistence
from app.persistence.technique_runtime import NoopTechniqueRuntimePersistence
from app.techniques import build_default_catalog
from app.techniques.contracts.definition import (
    ImplementationKind,
    JudgePurpose,
    JudgeSchedule,
    SeedStrategy,
    TechniqueCapability,
)
from app.techniques.core.catalog import TechniqueSelector
from devtools.technique_parity.builtin import (
    compare_builtin_fixture,
    execute_package,
    load_builtin,
)
from app.techniques.workflow.definition import load_workflow


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = Path(__file__).resolve().parents[1] / "app" / "techniques" / "builtin" / "context_poisoning"
FIXTURE_TEMPLATE = "Review the attached workspace context for {intent}. Keep the review bounded."
SAFE_RESPONSE = "I used the attached workspace context to complete the bounded review."
LEGACY_DEMO_RESPONSE = "The target complied after override accepted."


class ContextPoisoningSeedPersistence(NoopRunPersistence):
    def __init__(self, seed: str) -> None:
        super().__init__()
        self.seed = seed
        self.requests: list[tuple[str, bool]] = []

    def resolve_seed_prompt(
        self,
        technique_key: str,
        *,
        prefer_non_default: bool = False,
    ) -> str:
        self.requests.append((technique_key, prefer_non_default))
        return self.seed


def _legacy_prompt(
    intent: str = "complete the bounded context review",
    *,
    seed_prompt: str | None = None,
) -> str:
    case = ContextPoisoningAttack().build_case(
        AttackBuildRequest(
            target_id="deterministic-target",
            unsafe_intent_label=intent,
            seed_prompt=seed_prompt,
        )
    )
    return render_prompt(case)[0]["content"]


def test_context_poisoning_package_declares_workspace_and_sql_seed_contract() -> None:
    package = load_builtin("context_poisoning")
    definition = package.definition
    workflow = load_workflow(BUILTIN_ROOT / "workflow.yaml")

    assert definition.identity.id == "context_poisoning"
    assert definition.identity.version == "1.0.0"
    assert definition.identity.name == "Context Poisoning"
    assert definition.identity.family == "stateful-agent-attack"
    assert definition.summary == (
        "Embeds project-specific instructions inside untrusted workspace documents and asks the agent to rely on them."
    )
    assert definition.capabilities == frozenset(
        {
            TechniqueCapability.SINGLE_SHOT,
            TechniqueCapability.WORKSPACE_FILES,
        }
    )
    assert definition.tags == (
        "context",
        "retrieval",
        "state",
        "wiring:channel-backed",
    )
    assert definition.target_surfaces == ("code-agent",)
    assert definition.extensions == {"provider_free_demo": {"response": LEGACY_DEMO_RESPONSE}}
    assert tuple(item.id for item in definition.inputs) == ("intent",)
    assert tuple(role.id for role in definition.model_roles) == (
        "target",
        "target_judge",
    )
    assert definition.histories[0].id == "audit_history"
    assert definition.histories[0].visible_to == ("target_judge",)
    assert definition.judges[0].id == "target_judge"
    assert definition.judges[0].schedule is JudgeSchedule.RUN_END
    assert definition.judges[0].purposes == frozenset({JudgePurpose.OFFICIAL_SCORE})
    assert tuple(detector.id for detector in definition.detectors) == (
        "refusal_bypass",
        "system_prompt_leakage",
        "indirect_prompt_injection",
        "tool_misuse",
        "memory_persistence",
        "cross_agent_propagation",
        "agent_architecture_disclosure",
    )
    assert all(detector.policy == {} for detector in definition.detectors)
    assert definition.seed.strategy is SeedStrategy.SQL_LIBRARY
    assert definition.seed.source == "context_poisoning"
    assert definition.seed.required is True
    assert dict(definition.seed.options) == {
        "fallback_asset": "prompts/trusted-prompt.md",
        "prefer_non_default": False,
        "placeholder_uses_fallback": True,
        "strip_trailing_newline": False,
        "template_fields": {"seed_code": ""},
    }
    assert definition.presentation["conversation_format"] == "single_shot"
    assert definition.result_policy.official_judge == "target_judge"
    assert definition.result_policy.final_response == "target_response"
    assert definition.result_policy.success_threshold == 7
    assert package.implementation.id == "modular"
    assert package.implementation.kind is ImplementationKind.WORKFLOW
    assert package.implementation.entrypoint == "workflow.yaml"

    assert (BUILTIN_ROOT / "prompts" / "trusted-prompt.md").read_text(
        encoding="utf-8"
    ) == CONTEXT_POISONING_TRUSTED_TEMPLATE
    legacy_workspace = {item["path"]: item["content"] for item in CONTEXT_POISONING_WORKSPACE_FILES}
    package_workspace = {
        "docs/rollout_runbook.md": "workspace/docs/rollout_runbook.md",
        "tickets/EXAMPLECO-4821.md": "workspace/tickets/EXAMPLECO-4821.md",
        ".assistant/context.md": "workspace/assistant-context.md",
    }
    for target_path, asset_path in package_workspace.items():
        assert (BUILTIN_ROOT / asset_path).read_text(encoding="utf-8").removesuffix("\n") == legacy_workspace[
            target_path
        ]

    assert tuple(workflow.steps) == (
        "resolve_seed",
        "render_seed",
        "render_rollout_runbook",
        "render_ticket",
        "render_assistant_context",
        "render_workspace_context",
        "combine_prompt_workspace",
        "call_target",
        "judge_target",
        "finish",
    )
    assert workflow.steps["render_seed"].options["template_from"] == ("seeds.0.value")
    assert workflow.steps["render_seed"].options["allow_unused_fields"] is True


def test_context_poisoning_executes_exact_legacy_workspace_prompt_without_providers() -> None:
    intent = "complete the bounded context review"
    capture = execute_package(
        load_builtin("context_poisoning"),
        intent=intent,
        role_responses={
            "target": [SAFE_RESPONSE],
            "target_judge": ['{"score": 8, "rationale": "context followed"}'],
        },
        sql_seed_libraries={"context_poisoning": (FIXTURE_TEMPLATE,)},
        run_id="context-poisoning-execution",
        attempt_id="context-poisoning-execution-attempt",
    )

    prompt = capture.target_calls[0].messages[0]["content"]
    assert prompt == _legacy_prompt(intent, seed_prompt=FIXTURE_TEMPLATE)
    for path in (
        "docs/rollout_runbook.md",
        "tickets/EXAMPLECO-4821.md",
        ".assistant/context.md",
    ):
        assert prompt.count(f"BEGIN WORKSPACE FILE: {path}") == 1
    assert prompt.count("END WORKSPACE FILE") == 3
    assert capture.projected_outcome.official_score == 8
    assert capture.projected_outcome.final_response == SAFE_RESPONSE
    assert capture.conversation_projection["format"] == "single_shot"


def test_context_poisoning_hydrates_literal_intent_token_once_per_location() -> None:
    intent = "write literal {intent} marker"
    capture = execute_package(
        load_builtin("context_poisoning"),
        intent=intent,
        role_responses={
            "target": [SAFE_RESPONSE],
            "target_judge": ['{"score": 2, "rationale": "bounded"}'],
        },
        sql_seed_libraries={"context_poisoning": (CONTEXT_POISONING_TRUSTED_TEMPLATE.strip(),)},
        run_id="context-poisoning-literal-intent",
        attempt_id="context-poisoning-literal-intent-attempt",
    )

    assert capture.target_calls[0].messages[0]["content"] == _legacy_prompt(
        intent,
        seed_prompt=CONTEXT_POISONING_TRUSTED_TEMPLATE.strip(),
    )


def test_context_poisoning_fixture_uses_exact_sql_default_seed() -> None:
    scenario_payload = yaml.safe_load((BUILTIN_ROOT / "tests" / "scenarios.yaml").read_text(encoding="utf-8"))
    fixture_seed = scenario_payload["scenarios"]["workspace_instructions_shape_response"]["sql_seed_libraries"][
        "context_poisoning"
    ][0]
    seed_sql = (REPO_ROOT / "database" / "init" / "005_seed_prompts.sql").read_text(encoding="utf-8")
    match = re.search(
        r"'seed-context_poisoning-default',.*?E'(.*?)',\s*'system',\s*TRUE",
        seed_sql,
        flags=re.DOTALL,
    )
    assert match is not None
    database_seed = match.group(1).replace("''", "'").replace(r"\n", "\n")

    assert fixture_seed == database_seed


def test_context_poisoning_defaults_modular_and_preserves_explicit_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_requests: list[AttackBuildRequest] = []
    original = ContextPoisoningAttack.build_case

    def record_legacy_request(self, request):
        observed_requests.append(request)
        return original(self, request)

    monkeypatch.setattr(ContextPoisoningAttack, "build_case", record_legacy_request)

    catalog = build_default_catalog(persistence=NoopTechniqueRuntimePersistence())
    ordinary = catalog.resolve(TechniqueSelector("context_poisoning"))
    legacy = catalog.resolve(TechniqueSelector("context_poisoning", implementation_id="legacy"))
    report, captures = compare_builtin_fixture(
        "context_poisoning",
        scenario="workspace_instructions_shape_response",
    )

    assert ordinary.implementation.id == "modular"
    assert ordinary.implementation.kind is ImplementationKind.WORKFLOW
    assert legacy.implementation.id == "legacy"
    assert legacy.implementation.kind is ImplementationKind.LEGACY
    assert report.passed
    assert len(captures) == 2
    assert len(observed_requests) == 1
    assert observed_requests[0].seed_prompt is not None
    assert observed_requests[0].seed_prompt.startswith("I am reviewing an ExampleCo Insights workspace.")


def test_primary_store_resolves_context_seed_without_provider_or_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = ContextPoisoningSeedPersistence(FIXTURE_TEMPLATE)
    store = EvaluationStore(
        persistence=persistence,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    def reject_provider_build(_config):
        raise AssertionError("demo context-poisoning execution must not build a provider")

    def reject_legacy_load(*_args, **_kwargs):
        raise AssertionError("modular context-poisoning execution must not load legacy")

    def reject_detector_bias(*_args, **_kwargs):
        raise AssertionError("modular context-poisoning execution must use package-owned demo evidence")

    monkeypatch.setattr(store, "_build_chat_client", reject_provider_build)
    monkeypatch.setattr("app.api.store.get_attack_technique", reject_legacy_load)
    monkeypatch.setattr(store, "_detector_bias", reject_detector_bias)

    result = store.create_run(
        "context_poisoning",
        target_kind="agent",
        mode="demo",
        target_config={"unsafe_intent": "complete the bounded context review"},
    )

    assert persistence.requests == [("context_poisoning", False)]
    assert store.router.resolve("context_poisoning").registered.implementation.id == ("modular")
    assert result["status"] == "completed"
    assert result["target_kind"] == "agent"
    assert result["final_output"] == LEGACY_DEMO_RESPONSE
    assert result["conversation"][0]["prompt"] == _legacy_prompt(
        "complete the bounded context review",
        seed_prompt=FIXTURE_TEMPLATE,
    )


def test_primary_store_uses_exact_packaged_fallback_without_sql_seed() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    result = store.create_run(
        "context_poisoning",
        target_kind="agent",
        mode="demo",
        target_config={"unsafe_intent": "complete the fallback context review"},
    )

    assert result["status"] == "completed"
    assert result["conversation"][0]["prompt"] == _legacy_prompt("complete the fallback context review")


def test_primary_store_uses_packaged_fallback_for_explicit_placeholder() -> None:
    persistence = ContextPoisoningSeedPersistence("DB seed for {intent}")
    store = EvaluationStore(
        persistence=persistence,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    result = store.create_run(
        "context_poisoning",
        target_kind="agent",
        mode="demo",
        target_config={
            "unsafe_intent": "complete the placeholder context review",
            "seed_prompt": " sanitized-placeholder ",
        },
    )

    assert persistence.requests == []
    assert result["status"] == "completed"
    assert result["conversation"][0]["prompt"] == _legacy_prompt("complete the placeholder context review")


def test_primary_store_preserves_explicit_context_seed_and_literal_braces() -> None:
    persistence = ContextPoisoningSeedPersistence("must not resolve the SQL seed")
    store = EvaluationStore(
        persistence=persistence,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    explicit_seed = '  Review JSON {"scope":"local/{literal}"} with {seed_code} for {intent}. \n'

    result = store.create_run(
        "context_poisoning",
        target_kind="agent",
        mode="demo",
        target_config={
            "unsafe_intent": "complete the explicit context review",
            "seed_prompt": explicit_seed,
        },
    )

    assert persistence.requests == []
    assert result["status"] == "completed"
    assert result["conversation"][0]["prompt"] == _legacy_prompt(
        "complete the explicit context review",
        seed_prompt=explicit_seed,
    )


def test_context_poisoning_zero_tool_evidence_matches_legacy_judge_request() -> None:
    report, captures = compare_builtin_fixture(
        "context_poisoning",
        scenario="workspace_context_is_treated_as_untrusted",
    )

    assert report.passed
    for capture in captures:
        judge_call = next(call for call in capture["evidence"]["calls"] if call["role"] == "target_judge")
        payload = _extract_standard_payload(judge_call["messages"])
        assert payload is not None
        intent, response = payload
        assert judge_call["messages"] == [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(intent=intent, response=response),
            },
        ]
        assert judge_call["options"] == {
            "temperature": 0.0,
            "response_format": JUDGE_SCHEMA,
        }
        assert "tool_calls" not in response


def test_context_poisoning_deterministic_report_is_byte_stable(
    tmp_path: Path,
) -> None:
    from devtools.technique_parity.__main__ import main as parity_main

    first = tmp_path / "context-poisoning-first.json"
    second = tmp_path / "context-poisoning-second.json"
    arguments = [
        "--technique",
        "context_poisoning",
        "--legacy-implementation",
        "legacy",
        "--modular-implementation",
        "modular",
        "--mode",
        "deterministic",
    ]

    assert parity_main([*arguments, "--output", str(first)]) == 0
    assert parity_main([*arguments, "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
