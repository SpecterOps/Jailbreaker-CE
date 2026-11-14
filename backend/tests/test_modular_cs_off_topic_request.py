from __future__ import annotations

from pathlib import Path

import pytest

from app.api.store import EvaluationStore
from app.attacklib.base import AttackBuildRequest
from app.attacklib.customer_service_attacks import OffTopicRequestTest
from app.attacklib.renderer import render_prompt
from app.llm_client import ProviderCallResult
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
    / "cs_off_topic_request"
)


class RejectSeedPersistence(NoopRunPersistence):
    def resolve_seed_prompt(
        self,
        technique_key: str,
        *,
        prefer_non_default: bool = False,
    ) -> str:
        raise AssertionError(
            "Off-Topic Request must not resolve SQL seed "
            f"{technique_key}:{prefer_non_default}"
        )


def _legacy_assets(intent: str = "test intent") -> dict[str, str]:
    case = OffTopicRequestTest().build_case(
        AttackBuildRequest(
            target_id="deterministic-target",
            unsafe_intent_label=intent,
            seed_prompt="ignored legacy seed",
        )
    )
    return dict(case.assets)


def test_off_topic_package_declares_shared_customer_service_contract() -> None:
    package = load_builtin("cs_off_topic_request")
    definition = package.definition
    workflow = load_workflow(BUILTIN_ROOT / "workflow.yaml")
    assets = _legacy_assets()

    assert definition.identity.id == "cs_off_topic_request"
    assert definition.identity.version == "1.0.0"
    assert definition.identity.name == "Off-Topic Request"
    assert definition.identity.family == "customer-service"
    assert definition.summary == (
        "Sends questions outside the chatbot's intended domain to test scope "
        "enforcement."
    )
    assert definition.capabilities == frozenset({TechniqueCapability.SINGLE_SHOT})
    assert definition.tags == ("scope", "off-topic", "customer-service")
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


def test_off_topic_package_executes_exact_fixed_prompt_without_providers() -> None:
    expected_prompt = render_prompt(
        OffTopicRequestTest().build_case(
            AttackBuildRequest(
                target_id="deterministic-target",
                unsafe_intent_label="ignored intent",
                seed_prompt="ignored seed",
            )
        )
    )[0]["content"]
    capture = execute_package(
        load_builtin("cs_off_topic_request"),
        intent="a different intent that must not alter the fixed probe",
        role_responses={
            "target": ["target output"],
            "target_judge": ['{"score": 3, "rationale": "scope held"}'],
        },
        sql_seed_libraries={
            "cs_off_topic_request": ("SQL seed must be ignored: {intent}",),
        },
        run_id="off-topic-execution",
        attempt_id="off-topic-execution-attempt",
    )

    assert capture.target_calls[0].messages == (
        {"role": "user", "content": expected_prompt},
    )
    assert capture.target_calls[0].options == {}
    assert capture.projected_outcome.official_score == 3
    assert capture.projected_outcome.final_response == "target output"
    assert capture.conversation_projection["format"] == "single_shot"


def test_primary_store_uses_modular_off_topic_without_seed_or_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvaluationStore(
        persistence=RejectSeedPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )

    def reject_provider_build(_config):
        raise AssertionError("demo Off-Topic execution must not build a provider")

    def reject_legacy_load(*_args, **_kwargs):
        raise AssertionError("modular Off-Topic execution must not load legacy")

    monkeypatch.setattr(store, "_build_chat_client", reject_provider_build)
    monkeypatch.setattr("app.api.store.get_attack_technique", reject_legacy_load)

    result = store.create_run(
        "cs_off_topic_request",
        mode="demo",
        target_config={
            "unsafe_intent": "ignored intent",
            "seed_prompt": 'Explicit seed must be ignored: {"value":"wrong"}',
        },
    )

    assert (
        store.router.resolve("cs_off_topic_request").registered.implementation.id
        == "modular"
    )
    assert result["status"] == "completed"
    assert result["conversation"][0]["kind"] == "single_shot"
    assert result["conversation"][0]["prompt"] == _legacy_assets()["template"]


def test_modular_off_topic_live_bridge_uses_packaged_cs_system_prompt_provider_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LocalClient:
        network_capable = False
        default_max_tokens = 1024
        max_tokens = 1024

        def __init__(self, role: str) -> None:
            self.role = role
            self.model = f"local-{role}"
            self.calls: list[list[dict[str, str]]] = []

        def chat_correlated_once(
            self,
            messages,
            *,
            request_context,
            **_options,
        ) -> ProviderCallResult:
            assert request_context.role == self.role
            self.calls.append([dict(message) for message in messages])
            if self.role == "target_judge":
                return ProviderCallResult(
                    text='{"score": 3, "rationale": "scope held"}'
                )
            return ProviderCallResult(text="provider-free local target output")

    target = LocalClient("target")
    judge = LocalClient("target_judge")
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    monkeypatch.setattr(store, "resolve_profile_config", lambda config: dict(config))
    monkeypatch.setattr(store, "_build_chat_client", lambda _config: target)
    monkeypatch.setattr(store, "_build_attacker_client", lambda _config: None)
    monkeypatch.setattr(
        store,
        "_build_judge_client",
        lambda _config, *, target_client, attacker_client: judge,
    )

    summary, _result = store._execute_single(
        technique_key="cs_off_topic_request",
        target_kind="chatbot",
        mode="live",
        target_config={
            "unsafe_intent": "ignored intent",
            "provider": "local",
            "model": target.model,
            "judge_mode": "custom",
            "judge_provider": "local",
            "judge_model": judge.model,
            "cs_system_prompt": "true",
            "system_prompt": "must be replaced",
        },
    )

    assert target.network_capable is False and judge.network_capable is False
    assert summary.status == "completed"
    assert target.calls == [
        [
            {
                "role": "system",
                "content": _legacy_assets()["recommended_system_prompt"],
            },
            {"role": "user", "content": _legacy_assets()["template"]},
        ]
    ]


def test_off_topic_default_is_modular_and_explicit_legacy_preserves_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_requests: list[AttackBuildRequest] = []
    original = OffTopicRequestTest.build_case

    def record_legacy_request(self, request):
        observed_requests.append(request)
        return original(self, request)

    monkeypatch.setattr(OffTopicRequestTest, "build_case", record_legacy_request)

    catalog = build_default_catalog(persistence=NoopTechniqueRuntimePersistence())
    ordinary = catalog.resolve(TechniqueSelector("cs_off_topic_request"))
    legacy = catalog.resolve(
        TechniqueSelector("cs_off_topic_request", implementation_id="legacy")
    )
    report, captures = compare_builtin_fixture(
        "cs_off_topic_request",
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


def test_off_topic_deterministic_report_is_byte_stable(
    tmp_path: Path,
) -> None:
    from devtools.technique_parity.__main__ import main as parity_main

    first = tmp_path / "off-topic-first.json"
    second = tmp_path / "off-topic-second.json"
    arguments = [
        "--technique",
        "cs_off_topic_request",
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
