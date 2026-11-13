from __future__ import annotations

from pathlib import Path

import pytest

from app.api.store import EvaluationStore
from app.attacklib.base import AttackBuildRequest
from app.attacklib.customer_service_attacks import DomainBoundaryProbeTest
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
    / "cs_domain_boundary_probe"
)


class RejectSeedPersistence(NoopRunPersistence):
    def resolve_seed_prompt(
        self,
        technique_key: str,
        *,
        prefer_non_default: bool = False,
    ) -> str:
        raise AssertionError(
            "Domain Boundary Probe must not resolve SQL seed "
            f"{technique_key}:{prefer_non_default}"
        )


def _legacy_assets(intent: str = "test intent") -> dict[str, str]:
    case = DomainBoundaryProbeTest().build_case(
        AttackBuildRequest(
            target_id="deterministic-target",
            unsafe_intent_label=intent,
            seed_prompt="ignored legacy seed",
        )
    )
    return dict(case.assets)


def test_domain_boundary_package_preserves_distinct_customer_service_data() -> None:
    package = load_builtin("cs_domain_boundary_probe")
    definition = package.definition
    workflow = load_workflow(BUILTIN_ROOT / "workflow.yaml")
    assets = _legacy_assets()

    assert definition.identity.id == "cs_domain_boundary_probe"
    assert definition.identity.version == "1.0.0"
    assert definition.identity.name == "Domain Boundary Probe"
    assert definition.identity.family == "customer-service"
    assert definition.summary == (
        "Tests gray-area requests that are adjacent to but outside the chatbot's "
        "domain."
    )
    assert definition.capabilities == frozenset({TechniqueCapability.SINGLE_SHOT})
    assert definition.tags == ("scope", "boundary", "customer-service")
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
        "scope_violation",
    )
    assert definition.seed.strategy is SeedStrategy.NONE
    assert definition.seed.required is False
    assert definition.presentation["conversation_format"] == "single_shot"
    assert definition.result_policy.official_judge == "target_judge"
    assert definition.result_policy.final_response == "target_response"
    assert definition.result_policy.success_threshold == 7
    assert package.implementation.id == "modular"
    assert package.implementation.kind is ImplementationKind.WORKFLOW
    assert package.implementation.entrypoint == "workflow.yaml"
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


def test_domain_boundary_executes_exact_fixed_prompt_without_providers() -> None:
    expected_prompt = render_prompt(
        DomainBoundaryProbeTest().build_case(
            AttackBuildRequest(
                target_id="deterministic-target",
                unsafe_intent_label="ignored intent",
                seed_prompt="ignored seed",
            )
        )
    )[0]["content"]
    capture = execute_package(
        load_builtin("cs_domain_boundary_probe"),
        intent="a different intent that must not alter the fixed probe",
        role_responses={
            "target": ["target output"],
            "target_judge": ['{"score": 3, "rationale": "boundary held"}'],
        },
        sql_seed_libraries={
            "cs_domain_boundary_probe": ("SQL seed must be ignored: {intent}",),
        },
        run_id="domain-boundary-execution",
        attempt_id="domain-boundary-execution-attempt",
    )

    assert capture.target_calls[0].messages == (
        {"role": "user", "content": expected_prompt},
    )
    assert capture.target_calls[0].options == {}
    assert capture.projected_outcome.official_score == 3
    assert capture.projected_outcome.final_response == "target output"
    assert capture.conversation_projection["format"] == "single_shot"


def test_primary_store_uses_modular_domain_boundary_without_seed_or_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvaluationStore(
        persistence=RejectSeedPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    def reject_provider_build(_config):
        raise AssertionError("demo Domain Boundary execution must not build a provider")

    def reject_legacy_load(*_args, **_kwargs):
        raise AssertionError("modular Domain Boundary execution must not load legacy")

    monkeypatch.setattr(store, "_build_chat_client", reject_provider_build)
    monkeypatch.setattr("app.api.store.get_attack_technique", reject_legacy_load)

    result = store.create_run(
        "cs_domain_boundary_probe",
        mode="demo",
        target_config={
            "unsafe_intent": "ignored intent",
            "seed_prompt": 'Explicit seed must be ignored: {"value":"wrong"}',
        },
    )

    assert (
        store.router.resolve("cs_domain_boundary_probe").registered.implementation.id
        == "modular"
    )
    assert result["status"] == "completed"
    assert result["conversation"][0]["kind"] == "single_shot"
    assert result["conversation"][0]["prompt"] == _legacy_assets()["template"]


def test_domain_boundary_default_modular_and_explicit_legacy_preserves_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_requests: list[AttackBuildRequest] = []
    original = DomainBoundaryProbeTest.build_case

    def record_legacy_request(self, request):
        observed_requests.append(request)
        return original(self, request)

    monkeypatch.setattr(DomainBoundaryProbeTest, "build_case", record_legacy_request)

    catalog = build_default_catalog(persistence=NoopTechniqueRuntimePersistence())
    ordinary = catalog.resolve(TechniqueSelector("cs_domain_boundary_probe"))
    legacy = catalog.resolve(
        TechniqueSelector("cs_domain_boundary_probe", implementation_id="legacy")
    )
    report, captures = compare_builtin_fixture(
        "cs_domain_boundary_probe",
        scenario="target_refuses",
    )

    assert ordinary.implementation.id == "modular"
    assert ordinary.implementation.kind is ImplementationKind.WORKFLOW
    assert legacy.implementation.id == "legacy"
    assert legacy.implementation.kind is ImplementationKind.LEGACY
    assert report.passed
    assert len(captures) == 2
    assert len(observed_requests) == 1
    assert observed_requests[0].seed_prompt is None


def test_domain_boundary_report_is_byte_stable(tmp_path: Path) -> None:
    from devtools.technique_parity.__main__ import main as parity_main

    first = tmp_path / "domain-boundary-first.json"
    second = tmp_path / "domain-boundary-second.json"
    arguments = [
        "--technique",
        "cs_domain_boundary_probe",
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
