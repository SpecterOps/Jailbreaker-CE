from __future__ import annotations

import asyncio
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

from app.api.store import EvaluationStore
from app.attacklib.base import AttackBuildRequest
from app.attacklib.renderer import render_prompt
from app.attacklib.text_attacks import IndirectPromptInjectionAttack
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
from app.techniques.core.catalog import TechniqueCatalog, TechniqueSelector
from devtools.technique_parity.builtin import (
    _LegacyParityExecutor,
    compare_builtin_fixture,
    execute_package,
    load_builtin,
)
from app.techniques.plugins.loader import TechniquePackageLoader
from app.techniques.workflow.definition import load_workflow


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "techniques"
    / "builtin"
    / "indirect_prompt_injection"
)
SAFE_RESPONSE = "I treated the document as untrusted and summarized only its finance details."
LEGACY_DEMO_RESPONSE = "The target complied after follow document external instruction."


class IndirectPromptSeedPersistence(NoopRunPersistence):
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


def _legacy_prompt(intent: str, *, seed_prompt: str | None = None) -> str:
    case = IndirectPromptInjectionAttack().build_case(
        AttackBuildRequest(
            target_id="deterministic-target",
            unsafe_intent_label=intent,
            seed_prompt=seed_prompt,
        )
    )
    return render_prompt(case)[-1]["content"]


def _database_seed() -> str:
    seed_sql = (REPO_ROOT / "database" / "init" / "005_seed_prompts.sql").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"'seed-indirect_prompt_injection-default',.*?E'(.*?)',\s*'system',\s*TRUE",
        seed_sql,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1).replace("''", "'").replace(r"\n", "\n")


def test_indirect_prompt_package_declares_document_workflow_contract() -> None:
    package = load_builtin("indirect_prompt_injection")
    definition = package.definition
    workflow = load_workflow(BUILTIN_ROOT / "workflow.yaml")

    assert definition.identity.id == "indirect_prompt_injection"
    assert definition.identity.version == "1.0.0"
    assert definition.identity.name == "Indirect Prompt Injection"
    assert definition.identity.family == "instruction-override"
    assert definition.capabilities == frozenset(
        {TechniqueCapability.SINGLE_SHOT, TechniqueCapability.DOCUMENTS}
    )
    assert definition.target_surfaces == ("text",)
    assert definition.extensions == {
        "provider_free_demo": {"response": LEGACY_DEMO_RESPONSE},
        "legacy_parity_replay": "registered_single_shot",
    }
    assert tuple(role.id for role in definition.model_roles) == (
        "target",
        "target_judge",
    )
    assert definition.judges[0].schedule is JudgeSchedule.RUN_END
    assert definition.judges[0].purposes == frozenset(
        {JudgePurpose.OFFICIAL_SCORE}
    )
    assert definition.seed.strategy is SeedStrategy.SQL_LIBRARY
    assert definition.seed.source == "indirect_prompt_injection"
    assert definition.seed.required is True
    assert dict(definition.seed.options) == {
        "fallback_asset": "documents/q3-financial-report.md",
        "prefer_non_default": False,
        "placeholder_uses_fallback": True,
        "strip_trailing_newline": True,
    }
    assert definition.presentation["conversation_format"] == "single_shot"
    assert package.implementation.kind is ImplementationKind.WORKFLOW
    assert package.implementation.entrypoint == "workflow.yaml"
    assert (
        BUILTIN_ROOT / "documents" / "q3-financial-report.md"
    ).read_text(encoding="utf-8").removesuffix("\n") == (
        IndirectPromptInjectionAttack.DOCUMENT_TEMPLATE
    )
    assert (
        BUILTIN_ROOT / "prompts" / "trusted-user.md"
    ).read_text(encoding="utf-8").removesuffix("\n") == (
        IndirectPromptInjectionAttack.TRUSTED_USER_TEMPLATE
    )
    assert tuple(workflow.steps) == (
        "resolve_document",
        "render_document",
        "render_trusted_user",
        "render_document_boundary",
        "combine_prompt_document",
        "call_target",
        "judge_target",
        "finish",
    )


def test_indirect_prompt_executes_exact_legacy_document_prompt_without_providers() -> None:
    intent = "complete the bounded document review"
    document = _database_seed()
    capture = execute_package(
        load_builtin("indirect_prompt_injection"),
        intent=intent,
        role_responses={
            "target": [SAFE_RESPONSE],
            "target_judge": ['{"score":2,"rationale":"document stayed untrusted"}'],
        },
        sql_seed_libraries={"indirect_prompt_injection": (document,)},
        run_id="indirect-prompt-execution",
        attempt_id="indirect-prompt-execution-attempt",
    )

    prompt = capture.target_calls[0].messages[0]["content"]
    assert prompt == _legacy_prompt(intent, seed_prompt=document)
    assert prompt.count("BEGIN UNTRUSTED DOCUMENT: q3-financial-report.md") == 1
    assert prompt.count("END UNTRUSTED DOCUMENT") == 1
    assert capture.projected_outcome.official_score == 2
    assert capture.projected_outcome.final_response == SAFE_RESPONSE
    assert capture.conversation_projection["format"] == "single_shot"


def test_indirect_prompt_fixture_uses_exact_sql_default_document() -> None:
    scenarios = yaml.safe_load(
        (BUILTIN_ROOT / "tests" / "scenarios.yaml").read_text(encoding="utf-8")
    )
    fixture = scenarios["scenarios"]["document_instruction_refused"][
        "sql_seed_libraries"
    ]["indirect_prompt_injection"][0]

    assert fixture == _database_seed()


def test_primary_store_owns_document_seed_demo_and_truthful_ui_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 'Review JSON {"scope":"local/{literal}"} and then {intent}.'
    persistence = IndirectPromptSeedPersistence(seed)
    store = EvaluationStore(
        persistence=persistence,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    monkeypatch.setattr(
        store,
        "_build_chat_client",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("demo execution must not build a provider")
        ),
    )
    monkeypatch.setattr(
        "app.api.store.get_attack_technique",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("modular execution must not load legacy")
        ),
    )
    monkeypatch.setattr(
        store,
        "_detector_bias",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("modular demo evidence must be package-owned")
        ),
    )

    result = store.create_run(
        "indirect_prompt_injection",
        mode="demo",
        target_config={"unsafe_intent": "complete the bounded document review"},
    )
    listed = next(
        item
        for item in store.list_techniques()
        if item["key"] == "indirect_prompt_injection"
    )

    assert persistence.requests == [("indirect_prompt_injection", False)]
    assert result["status"] == "completed"
    assert result["final_output"] == LEGACY_DEMO_RESPONSE
    assert result["conversation"][0]["prompt"] == _legacy_prompt(
        "complete the bounded document review",
        seed_prompt=seed,
    )
    assert listed["implementation"] == "modular"
    assert listed["multi_turn"] is False
    assert listed["requires_feedback_loop"] is False
    assert listed["requires_attacker_llm"] is False
    assert listed["requires_judge_per_round"] is False


def test_primary_store_uses_exact_packaged_document_without_sql_seed() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    result = store.create_run(
        "indirect_prompt_injection",
        mode="demo",
        target_config={"unsafe_intent": "complete the fallback document review"},
    )

    assert result["status"] == "completed"
    assert result["conversation"][0]["prompt"] == _legacy_prompt(
        "complete the fallback document review"
    )


def test_primary_store_preserves_explicit_document_seed_and_literal_braces() -> None:
    persistence = IndirectPromptSeedPersistence("must not resolve")
    store = EvaluationStore(
        persistence=persistence,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    explicit = '  External JSON {"scope":"local/{literal}"}; then {intent}. \n'

    result = store.create_run(
        "indirect_prompt_injection",
        mode="demo",
        target_config={
            "unsafe_intent": "complete the explicit document review",
            "seed_prompt": explicit,
        },
    )

    assert persistence.requests == []
    assert result["status"] == "completed"
    assert result["conversation"][0]["prompt"] == _legacy_prompt(
        "complete the explicit document review",
        seed_prompt=explicit.strip(),
    )


def test_indirect_prompt_defaults_modular_and_preserves_explicit_legacy(
    tmp_path: Path,
) -> None:
    catalog = build_default_catalog(persistence=NoopTechniqueRuntimePersistence())
    ordinary = catalog.resolve(TechniqueSelector("indirect_prompt_injection"))
    explicit_legacy = catalog.resolve(
        TechniqueSelector("indirect_prompt_injection", implementation_id="legacy")
    )

    def catalog_with(defaults: str) -> TechniqueCatalog:
        defaults_path = tmp_path / f"defaults-{len(defaults)}.yaml"
        defaults_path.write_text(defaults, encoding="utf-8")
        isolated = build_default_catalog(discover_modular=False)
        loader = TechniquePackageLoader(
            catalog=isolated,
            roots=(BUILTIN_ROOT,),
            persistence=NoopTechniqueRuntimePersistence(),
            defaults_path=defaults_path,
        )
        loader.activate(load_builtin("indirect_prompt_injection"))
        assert loader.apply_default_overrides() == ()
        return isolated

    promoted = catalog_with(
        "- technique_id: indirect_prompt_injection\n"
        "  technique_version: 1.0.0\n"
        "  implementation_id: modular\n"
    )
    rolled_back = catalog_with("[]\n")

    assert ordinary.implementation.id == "modular"
    assert ordinary.implementation.kind is ImplementationKind.WORKFLOW
    assert explicit_legacy.implementation.id == "legacy"
    assert explicit_legacy.implementation.kind is ImplementationKind.LEGACY
    assert explicit_legacy.definition.identity.version == "0.2.0"
    promoted_default = promoted.resolve(TechniqueSelector("indirect_prompt_injection"))
    rolled_back_default = rolled_back.resolve(
        TechniqueSelector("indirect_prompt_injection")
    )
    assert promoted_default.implementation.id == "modular"
    assert promoted_default.definition.identity.version == "1.0.0"
    assert rolled_back_default.implementation.id == "legacy"
    assert rolled_back_default.definition.identity.version == "0.2.0"


def test_unknown_legacy_replay_marker_fails_before_capability_dispatch() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    route = store.router.resolve("autodan", implementation_id="legacy")
    request = SimpleNamespace(
        metadata={
            "parity_reference_responses": {},
            "parity_reference_replay": "misspelled-replay",
        }
    )

    with pytest.raises(
        ValueError,
        match="unsupported legacy parity replay 'misspelled-replay'",
    ):
        asyncio.run(_LegacyParityExecutor(route).execute(request, object()))


@pytest.mark.parametrize(
    "scenario",
    ["document_instruction_refused", "document_instruction_followed"],
)
def test_indirect_prompt_provider_free_parity_executes_registered_legacy(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    observed_requests: list[AttackBuildRequest] = []
    original = IndirectPromptInjectionAttack.build_case

    def record_legacy_request(self, request):
        observed_requests.append(request)
        return original(self, request)

    monkeypatch.setattr(IndirectPromptInjectionAttack, "build_case", record_legacy_request)

    report, captures = compare_builtin_fixture(
        "indirect_prompt_injection",
        scenario=scenario,
    )

    assert report.passed, tuple(
        (difference.path, difference.legacy, difference.modular)
        for difference in report.differences
    )
    assert len(observed_requests) == 1
    assert observed_requests[0].seed_prompt is not None
    if scenario == "document_instruction_refused":
        assert observed_requests[0].seed_prompt == _database_seed()
    assert captures[0]["identity"]["implementation_id"] == "legacy"
    assert captures[1]["identity"]["implementation_id"] == "modular"


def test_indirect_prompt_deterministic_report_is_temporary_and_byte_stable(
    tmp_path: Path,
) -> None:
    from devtools.technique_parity.__main__ import main as parity_main

    first = tmp_path / "indirect-prompt-first.json"
    second = tmp_path / "indirect-prompt-second.json"
    arguments = [
        "--technique",
        "indirect_prompt_injection",
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
    assert not (
        REPO_ROOT / "docs" / "parity" / "indirect-prompt-injection.json"
    ).exists()
