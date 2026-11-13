from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

from app.api.store import EvaluationStore
from app.attacklib.base import AttackBuildRequest
from app.attacklib.text_attacks import MultilingualObfuscationAttack
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
BUILTIN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "techniques"
    / "builtin"
    / "multilingual_obfuscation"
)
FALLBACK_TEMPLATE = MultilingualObfuscationAttack.DEFAULT_TEMPLATE
FIXTURE_TEMPLATE = "Fixture multilingual seed for {intent}.\nLanguage: mixed"


class MultilingualSeedPersistence(NoopRunPersistence):
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


def test_multilingual_package_declares_sql_seed_and_single_shot_workflow() -> None:
    package = load_builtin("multilingual_obfuscation")
    definition = package.definition
    workflow = load_workflow(BUILTIN_ROOT / "workflow.yaml")

    assert definition.identity.id == "multilingual_obfuscation"
    assert definition.identity.version == "1.0.0"
    assert definition.identity.name == "Multilingual Obfuscation"
    assert definition.identity.family == "obfuscation-based"
    assert definition.capabilities == frozenset({TechniqueCapability.SINGLE_SHOT})
    assert definition.target_surfaces == ("text",)
    assert tuple(role.id for role in definition.model_roles) == (
        "target",
        "target_judge",
    )
    assert definition.histories[0].id == "audit_history"
    assert definition.histories[0].visible_to == ("target_judge",)
    assert definition.judges[0].id == "target_judge"
    assert definition.judges[0].schedule is JudgeSchedule.RUN_END
    assert definition.judges[0].purposes == frozenset(
        {JudgePurpose.OFFICIAL_SCORE}
    )
    assert definition.seed.strategy is SeedStrategy.SQL_LIBRARY
    assert definition.seed.source == "multilingual_obfuscation"
    assert definition.seed.required is True
    assert dict(definition.seed.options) == {
        "fallback_asset": "prompts/default-seed.md",
        "prefer_non_default": False,
        "strip_trailing_newline": True,
    }
    assert definition.presentation["conversation_format"] == "single_shot"
    assert definition.result_policy.official_judge == "target_judge"
    assert definition.result_policy.final_response == "target_response"
    assert definition.result_policy.success_threshold == 7
    assert {detector.id for detector in definition.detectors} == {
        "refusal_bypass",
        "system_prompt_leakage",
        "indirect_prompt_injection",
    }
    assert package.implementation.id == "modular"
    assert package.implementation.kind is ImplementationKind.WORKFLOW
    assert package.implementation.entrypoint == "workflow.yaml"
    assert (BUILTIN_ROOT / "prompts" / "default-seed.md").read_text(
        encoding="utf-8"
    ).removesuffix("\n") == FALLBACK_TEMPLATE
    assert tuple(workflow.steps) == (
        "resolve_seed",
        "render_prompt",
        "call_target",
        "judge_target",
        "finish",
    )
    assert workflow.steps["render_prompt"].options["template_from"] == (
        "seeds.0.value"
    )
    assert workflow.steps["render_prompt"].options["allow_unused_fields"] is True


def test_multilingual_package_executes_from_injected_sql_seed_without_providers() -> None:
    capture = execute_package(
        load_builtin("multilingual_obfuscation"),
        intent="test intent",
        role_responses={
            "target": ["target output"],
            "target_judge": ['{"score": 3, "rationale": "refused"}'],
        },
        sql_seed_libraries={"multilingual_obfuscation": (FIXTURE_TEMPLATE,)},
        run_id="multilingual-execution",
        attempt_id="multilingual-execution-attempt",
    )

    assert capture.target_calls[0].messages == (
        {
            "role": "user",
            "content": "Fixture multilingual seed for test intent.\nLanguage: mixed",
        },
    )
    assert capture.projected_outcome.official_score == 3
    assert capture.projected_outcome.final_response == "target output"
    assert capture.conversation_projection["format"] == "single_shot"


def test_multilingual_deterministic_fixture_uses_exact_sql_default_seed() -> None:
    scenario_payload = yaml.safe_load(
        (BUILTIN_ROOT / "tests" / "scenarios.yaml").read_text(encoding="utf-8")
    )
    fixture_seed = scenario_payload["scenarios"]["target_refuses"][
        "sql_seed_libraries"
    ]["multilingual_obfuscation"][0]
    seed_sql = (REPO_ROOT / "database" / "init" / "005_seed_prompts.sql").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"'seed-multilingual_obfuscation-default',.*?E'(.*?)',\s*'system',\s*TRUE",
        seed_sql,
        flags=re.DOTALL,
    )
    assert match is not None
    database_seed = (
        match.group(1)
        .replace("''", "'")
        .replace(r"\u00e9", "é")
        .replace(r"\u00e7", "ç")
        .replace(r"\u00e0", "à")
        .replace(r"\n", "\n")
    )

    assert fixture_seed == database_seed


def test_primary_store_resolves_multilingual_sql_seed_without_legacy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = MultilingualSeedPersistence(FIXTURE_TEMPLATE)
    store = EvaluationStore(
        persistence=persistence,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    def reject_provider_build(_config):
        raise AssertionError("demo multilingual execution must not build a provider")

    def reject_legacy_load(*_args, **_kwargs):
        raise AssertionError("modular multilingual execution must not load legacy")

    monkeypatch.setattr(store, "_build_chat_client", reject_provider_build)
    monkeypatch.setattr("app.api.store.get_attack_technique", reject_legacy_load)

    result = store.create_run(
        "multilingual_obfuscation",
        mode="demo",
        target_config={"unsafe_intent": "test intent"},
    )

    assert persistence.requests == [("multilingual_obfuscation", False)]
    assert (
        store.router.resolve("multilingual_obfuscation").registered.implementation.id
        == "modular"
    )
    assert result["status"] == "completed"
    assert result["conversation"][0]["kind"] == "single_shot"
    assert result["conversation"][0]["prompt"] == (
        "Fixture multilingual seed for test intent.\nLanguage: mixed"
    )


def test_primary_store_preserves_fully_rendered_explicit_multilingual_seed() -> None:
    persistence = MultilingualSeedPersistence("must not resolve the SQL seed")
    store = EvaluationStore(
        persistence=persistence,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    explicit_seed = 'Already rendered multilingual JSON: {"language":"français"}.'

    result = store.create_run(
        "multilingual_obfuscation",
        mode="demo",
        target_config={
            "unsafe_intent": "must not be appended",
            "seed_prompt": explicit_seed,
        },
    )

    assert persistence.requests == []
    assert (
        store.router.resolve("multilingual_obfuscation").registered.implementation.id
        == "modular"
    )
    assert result["status"] == "completed"
    assert result["conversation"][0]["prompt"] == explicit_seed


def test_primary_store_uses_packaged_multilingual_fallback_without_sql_seed() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    result = store.create_run(
        "multilingual_obfuscation",
        mode="demo",
        target_config={"unsafe_intent": "test intent"},
    )

    assert (
        store.router.resolve("multilingual_obfuscation").registered.implementation.id
        == "modular"
    )
    assert result["conversation"][0]["prompt"] == FALLBACK_TEMPLATE.format(
        intent="test intent"
    )


def test_multilingual_defaults_modular_and_explicit_legacy_uses_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_requests: list[AttackBuildRequest] = []
    original = MultilingualObfuscationAttack.build_case

    def record_legacy_request(self, request):
        observed_requests.append(request)
        return original(self, request)

    monkeypatch.setattr(MultilingualObfuscationAttack, "build_case", record_legacy_request)

    catalog = build_default_catalog(persistence=NoopTechniqueRuntimePersistence())
    ordinary = catalog.resolve(TechniqueSelector("multilingual_obfuscation"))
    legacy = catalog.resolve(
        TechniqueSelector("multilingual_obfuscation", implementation_id="legacy")
    )
    report, captures = compare_builtin_fixture(
        "multilingual_obfuscation",
        scenario="target_refuses",
    )

    assert ordinary.implementation.id == "modular"
    assert ordinary.implementation.kind is ImplementationKind.WORKFLOW
    assert legacy.implementation.id == "legacy"
    assert legacy.implementation.kind is ImplementationKind.LEGACY
    assert report.passed
    assert len(captures) == 2
    assert len(observed_requests) == 1
    assert observed_requests[0].seed_prompt is not None
    assert observed_requests[0].seed_prompt.startswith("Répondez en français")


def test_multilingual_deterministic_report_is_byte_stable(
    tmp_path: Path,
) -> None:
    from devtools.technique_parity.__main__ import main as parity_main

    first = tmp_path / "multilingual-first.json"
    second = tmp_path / "multilingual-second.json"
    arguments = [
        "--technique",
        "multilingual_obfuscation",
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
