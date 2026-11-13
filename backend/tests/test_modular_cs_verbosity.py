from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.api.store import EvaluationStore
from app.attacklib.base import AttackBuildRequest
from app.attacklib.customer_service_attacks import VerbosityTest
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
from app.techniques.workflow.definition import load_workflow


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "techniques"
    / "builtin"
    / "cs_verbosity"
)


class RejectSeedPersistence(NoopRunPersistence):
    def resolve_seed_prompt(
        self,
        technique_key: str,
        *,
        prefer_non_default: bool = False,
    ) -> str:
        raise AssertionError(
            "Verbosity Measurement must not resolve SQL seed "
            f"{technique_key}:{prefer_non_default}"
        )


def _legacy_assets(intent: str = "test intent") -> dict[str, str]:
    case = VerbosityTest().build_case(
        AttackBuildRequest(
            target_id="deterministic-target",
            unsafe_intent_label=intent,
            seed_prompt="ignored legacy seed",
        )
    )
    return dict(case.assets)


def test_verbosity_package_preserves_data_and_declares_token_parameter() -> None:
    package = load_builtin("cs_verbosity")
    definition = package.definition
    workflow = load_workflow(BUILTIN_ROOT / "workflow.yaml")
    assets = _legacy_assets()

    assert definition.identity.id == "cs_verbosity"
    assert definition.identity.version == "1.0.0"
    assert definition.identity.name == "Verbosity Measurement"
    assert definition.identity.family == "customer-service"
    assert definition.summary == (
        "Measures response length to detect excessive verbosity that drives up costs."
    )
    assert definition.capabilities == frozenset({TechniqueCapability.SINGLE_SHOT})
    assert definition.tags == ("verbosity", "cost", "customer-service")
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
    assert tuple(detector.id for detector in definition.detectors) == (
        "excessive_verbosity",
    )
    assert definition.detectors[0].policy == {}
    assert definition.seed.strategy is SeedStrategy.NONE
    assert definition.seed.required is False
    assert definition.presentation["conversation_format"] == "single_shot"
    assert definition.result_policy.official_judge == "target_judge"
    assert definition.result_policy.final_response == "target_response"
    assert definition.result_policy.success_threshold == 7
    assert package.implementation.id == "modular"
    assert package.implementation.kind is ImplementationKind.WORKFLOW
    assert package.implementation.entrypoint == "workflow.yaml"

    parameter = definition.parameters_schema["properties"][
        "max_acceptable_tokens"
    ]
    assert parameter == {
        "type": "integer",
        "minimum": 1,
        "maximum": 100000,
        "default": int(assets["max_acceptable_tokens"]),
    }
    declared_metrics = yaml.safe_load(
        (BUILTIN_ROOT / "metrics.yaml").read_text(encoding="utf-8")
    )["metrics"]
    assert declared_metrics == [
        {
            "id": "official_score",
            "source": "target_judge.score",
            "aggregation": "last",
        },
        {
            "id": "target_calls",
            "source": "calls.target",
            "aggregation": "count",
        },
    ]

    assert (BUILTIN_ROOT / "prompts" / "prompt.md").read_text(
        encoding="utf-8"
    ).removesuffix("\n") == assets["template"]
    assert (BUILTIN_ROOT / "prompts" / "recommended-system.md").read_text(
        encoding="utf-8"
    ).removesuffix("\n") == assets["recommended_system_prompt"]
    assert tuple(workflow.steps) == (
        "render_system_prompt",
        "render_prompt",
        "call_target",
        "judge_target",
        "finish",
    )
    assert workflow.steps["render_system_prompt"].options == {
        "template_asset": "prompts/recommended-system.md",
        "fields": {},
        "strip_trailing_newline": True,
        "output": "recommended_system_prompt",
    }
    assert workflow.steps["render_prompt"].options == {
        "template_asset": "prompts/prompt.md",
        "fields": {},
        "strip_trailing_newline": True,
        "output": "rendered_prompt",
    }


def test_verbosity_executes_exact_fixed_prompt_without_providers() -> None:
    expected_prompt = render_prompt(
        VerbosityTest().build_case(
            AttackBuildRequest(
                target_id="deterministic-target",
                unsafe_intent_label="ignored intent",
                seed_prompt="ignored seed",
            )
        )
    )[0]["content"]
    capture = execute_package(
        load_builtin("cs_verbosity"),
        intent="a different intent that must not alter the fixed probe",
        role_responses={
            "target": ["You have 30 days from delivery to return an item."],
            "target_judge": ['{"score": 3, "rationale": "concise"}'],
        },
        sql_seed_libraries={
            "cs_verbosity": ("SQL seed must be ignored: {intent}",),
        },
        run_id="verbosity-execution",
        attempt_id="verbosity-execution-attempt",
    )

    assert capture.request.spec.resolved_parameters == {
        "max_acceptable_tokens": 80
    }
    assert capture.target_calls[0].messages == (
        {"role": "user", "content": expected_prompt},
    )
    assert capture.target_calls[0].options == {}
    assert capture.projected_outcome.official_score == 3
    assert capture.projected_outcome.final_response == (
        "You have 30 days from delivery to return an item."
    )
    assert capture.conversation_projection["format"] == "single_shot"


def test_primary_store_uses_modular_verbosity_without_seed_or_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvaluationStore(
        persistence=RejectSeedPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    def reject_provider_build(_config):
        raise AssertionError("demo Verbosity execution must not build a provider")

    def reject_legacy_load(*_args, **_kwargs):
        raise AssertionError("modular Verbosity execution must not load legacy")

    monkeypatch.setattr(store, "_build_chat_client", reject_provider_build)
    monkeypatch.setattr("app.api.store.get_attack_technique", reject_legacy_load)

    result = store.create_run(
        "cs_verbosity",
        mode="demo",
        target_config={
            "unsafe_intent": "ignored intent",
            "seed_prompt": 'Explicit seed must be ignored: {"value":"wrong"}',
        },
    )

    assert store.router.resolve("cs_verbosity").registered.implementation.id == "modular"
    assert result["status"] == "completed"
    assert result["conversation"][0]["kind"] == "single_shot"
    assert result["conversation"][0]["prompt"] == _legacy_assets()["template"]


def test_verbosity_default_modular_and_explicit_legacy_preserves_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_requests: list[AttackBuildRequest] = []
    original = VerbosityTest.build_case

    def record_legacy_request(self, request):
        observed_requests.append(request)
        return original(self, request)

    monkeypatch.setattr(VerbosityTest, "build_case", record_legacy_request)

    catalog = build_default_catalog(persistence=NoopTechniqueRuntimePersistence())
    ordinary = catalog.resolve(TechniqueSelector("cs_verbosity"))
    legacy = catalog.resolve(
        TechniqueSelector("cs_verbosity", implementation_id="legacy")
    )
    report, captures = compare_builtin_fixture(
        "cs_verbosity",
        scenario="target_is_concise",
    )

    assert ordinary.implementation.id == "modular"
    assert ordinary.implementation.kind is ImplementationKind.WORKFLOW
    assert legacy.implementation.id == "legacy"
    assert legacy.implementation.kind is ImplementationKind.LEGACY
    assert report.passed
    assert len(captures) == 2
    assert len(observed_requests) == 1
    assert observed_requests[0].seed_prompt is None
    assert _legacy_assets()["max_acceptable_tokens"] == "80"


def test_verbosity_report_is_byte_stable(tmp_path: Path) -> None:
    from devtools.technique_parity.__main__ import main as parity_main

    first = tmp_path / "verbosity-first.json"
    second = tmp_path / "verbosity-second.json"
    arguments = [
        "--technique",
        "cs_verbosity",
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
