from __future__ import annotations

import importlib
import json
from pathlib import Path
import re

import pytest

from app.api.store import EvaluationStore
from app.attacklib.base import AttackBuildRequest
from app.attacklib.pyrit_attacks import ManyShotJailbreakAttack
from app.attacklib.renderer import render_prompt
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


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "techniques"
    / "builtin"
    / "many_shot_jailbreak"
)
PAIR_ASSET = BUILTIN_ROOT / "seeds" / "fake-qa-pairs.json"


class RejectSeedPersistence(NoopRunPersistence):
    def resolve_seed_prompt(
        self,
        technique_key: str,
        *,
        prefer_non_default: bool = False,
    ) -> str:
        raise AssertionError(
            "Many-Shot must not resolve SQL seed "
            f"{technique_key}:{prefer_non_default}"
        )


def _legacy_template(intent: str = "test intent") -> str:
    case = ManyShotJailbreakAttack().build_case(
        AttackBuildRequest(
            target_id="deterministic-target",
            unsafe_intent_label=intent,
            seed_prompt="ignored legacy seed",
        )
    )
    return render_prompt(case)[0]["content"]


def test_many_shot_package_declares_bounded_python_construction() -> None:
    package = load_builtin("many_shot_jailbreak")
    definition = package.definition
    pairs = json.loads(PAIR_ASSET.read_text(encoding="utf-8"))

    assert definition.identity.id == "many_shot_jailbreak"
    assert definition.identity.version == "1.0.0"
    assert definition.identity.name == "Many-Shot Jailbreak"
    assert definition.identity.family == "context-exploitation"
    assert definition.summary == (
        "Primes the model with many fake compliant Q&A pairs, then appends "
        "the real harmful request to exploit in-context learning."
    )
    assert definition.capabilities == frozenset({TechniqueCapability.SINGLE_SHOT})
    assert definition.tags == (
        "context-exploitation",
        "in-context-learning",
        "long-context",
    )
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
    assert definition.seed.strategy is SeedStrategy.NONE
    assert definition.seed.required is False
    assert definition.presentation["conversation_format"] == "single_shot"
    assert definition.recommended_limits == {
        "max_rounds": 1,
        "max_fake_pairs": 10,
        "max_pair_field_characters": 4096,
        "max_pair_characters": 12000,
        "max_intent_characters": 4096,
        "max_conservative_input_tokens": 16384,
    }
    assert definition.result_policy.official_judge == "target_judge"
    assert definition.result_policy.final_response == "target_response"
    assert definition.result_policy.success_threshold == 7
    assert {detector.id for detector in definition.detectors} == {
        "refusal_bypass",
        "system_prompt_leakage",
        "indirect_prompt_injection",
    }
    assert package.implementation.id == "modular"
    assert package.implementation.kind is ImplementationKind.PYTHON
    assert package.implementation.entrypoint == "executor:ManyShotJailbreakExecutor"
    assert len(pairs) == 10
    assert tuple(
        (pair["question"], pair["answer"])
        for pair in pairs
    ) == ManyShotJailbreakAttack._FAKE_QA_PAIRS


def test_many_shot_package_executes_exact_legacy_prompt_without_providers() -> None:
    intent = "test intent"
    expected_prompt = _legacy_template(intent)
    capture = execute_package(
        load_builtin("many_shot_jailbreak"),
        intent=intent,
        role_responses={
            "target": ["target output"],
            "target_judge": ['{"score": 3, "rationale": "refused"}'],
        },
        sql_seed_libraries={
            "many_shot_jailbreak": ("SQL seed must be ignored: {intent}",),
        },
        run_id="many-shot-execution",
        attempt_id="many-shot-execution-attempt",
    )

    prompt = capture.target_calls[0].messages[0]["content"]
    assert prompt == expected_prompt
    assert prompt.encode("utf-8") == expected_prompt.encode("utf-8")
    assert prompt.count("Q: ") == 11
    assert prompt.count("A:") == 11
    assert prompt.endswith(f"Q: {intent}\nA:")
    assert capture.target_calls[0].options == {}
    target_snapshot_id = capture.target_calls[0].history_snapshot_id
    assert target_snapshot_id != "single-shot-input"
    target_snapshot = capture.context.histories.snapshots()[target_snapshot_id]
    assert tuple(
        {"role": message.role, "content": message.content}
        for message in target_snapshot.messages
    ) == capture.target_calls[0].messages
    assert capture.projected_outcome.official_score == 3
    assert capture.projected_outcome.final_response == "target output"
    assert capture.conversation_projection["format"] == "single_shot"


def test_many_shot_package_matches_unused_sql_seed_and_enforces_pair_bound() -> None:
    seed_sql = (REPO_ROOT / "database" / "init" / "005_seed_prompts.sql").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"'seed-many_shot_jailbreak-default',.*?E'(.*?)',\s*'system',\s*FALSE",
        seed_sql,
        flags=re.DOTALL,
    )
    assert match is not None
    database_template = match.group(1).replace("''", "'").replace(r"\n", "\n")
    legacy_template = _legacy_template("{intent}")

    assert database_template == legacy_template

    executor = importlib.import_module(
        "app.techniques.builtin.many_shot_jailbreak.executor"
    )
    pairs = json.loads(PAIR_ASSET.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="exactly 10"):
        executor._parse_fake_pairs(json.dumps(pairs[:-1]))
    with pytest.raises(ValueError, match="exactly 10"):
        executor._parse_fake_pairs(json.dumps([*pairs, pairs[0]]))
    whitespace = [dict(pair) for pair in pairs]
    whitespace[0]["question"] = "   "
    with pytest.raises(ValueError, match="non-empty"):
        executor._parse_fake_pairs(json.dumps(whitespace))
    oversized_field = [dict(pair) for pair in pairs]
    oversized_field[0]["answer"] = "x" * 4097
    with pytest.raises(ValueError, match="at most 4096"):
        executor._parse_fake_pairs(json.dumps(oversized_field))
    oversized_aggregate = [dict(pair) for pair in pairs]
    for pair in oversized_aggregate:
        pair["answer"] = "x" * 1300
    with pytest.raises(ValueError, match="at most 12000 total"):
        executor._parse_fake_pairs(json.dumps(oversized_aggregate))

    parsed = executor._parse_fake_pairs(PAIR_ASSET.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="intent must be at most 4096"):
        executor._render_many_shot_prompt(parsed, "x" * 4097)
    with pytest.raises(ValueError, match="conservative 16384-token"):
        executor._render_many_shot_prompt(parsed, "🔒" * 4096)


def test_primary_store_ignores_sql_and_explicit_seed_for_modular_many_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvaluationStore(
        persistence=RejectSeedPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    def reject_provider_build(_config):
        raise AssertionError("demo Many-Shot execution must not build a provider")

    def reject_legacy_load(*_args, **_kwargs):
        raise AssertionError("modular Many-Shot execution must not load legacy")

    monkeypatch.setattr(store, "_build_chat_client", reject_provider_build)
    monkeypatch.setattr("app.api.store.get_attack_technique", reject_legacy_load)

    result = store.create_run(
        "many_shot_jailbreak",
        mode="demo",
        target_config={
            "unsafe_intent": "test intent",
            "seed_prompt": 'Explicit seed must be ignored: {"value":"wrong"}',
        },
    )

    assert (
        store.router.resolve("many_shot_jailbreak").registered.implementation.id
        == "modular"
    )
    assert result["status"] == "completed"
    assert result["conversation"][0]["kind"] == "single_shot"
    assert result["conversation"][0]["prompt"] == _legacy_template()


def test_many_shot_default_is_modular_and_explicit_legacy_preserves_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_requests: list[AttackBuildRequest] = []
    original = ManyShotJailbreakAttack.build_case

    def record_legacy_request(self, request):
        observed_requests.append(request)
        return original(self, request)

    monkeypatch.setattr(ManyShotJailbreakAttack, "build_case", record_legacy_request)

    catalog = build_default_catalog(persistence=NoopTechniqueRuntimePersistence())
    ordinary = catalog.resolve(TechniqueSelector("many_shot_jailbreak"))
    legacy = catalog.resolve(
        TechniqueSelector("many_shot_jailbreak", implementation_id="legacy")
    )
    report, captures = compare_builtin_fixture(
        "many_shot_jailbreak",
        scenario="target_refuses",
    )

    assert ordinary.implementation.id == "modular"
    assert ordinary.implementation.kind is ImplementationKind.PYTHON
    assert legacy.implementation.id == "legacy"
    assert legacy.implementation.kind is ImplementationKind.LEGACY
    assert report.passed
    assert len(captures) == 2
    assert len(observed_requests) == 1
    assert observed_requests[0].seed_prompt is None


def test_many_shot_deterministic_report_is_byte_stable(
    tmp_path: Path,
) -> None:
    from devtools.technique_parity.__main__ import main as parity_main

    first = tmp_path / "many-shot-first.json"
    second = tmp_path / "many-shot-second.json"
    arguments = [
        "--technique",
        "many_shot_jailbreak",
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
