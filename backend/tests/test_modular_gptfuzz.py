from __future__ import annotations

from pathlib import Path

import pytest

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
from app.techniques.plugins import load_technique_package


BUILTIN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "techniques"
    / "builtin"
    / "gptfuzz"
)


def _load_builtin():
    return load_technique_package(
        BUILTIN_ROOT,
        supported_capabilities=frozenset(TechniqueCapability),
    )


def _execute(
    *,
    target: list[str],
    attacker: list[str],
    target_judge: list[str],
    parameters: dict[str, object] | None = None,
    effective_limits: dict[str, object] | None = None,
    seed: str = "Seed request for {intent}",
):
    from devtools.technique_parity.builtin import execute_package

    return execute_package(
        _load_builtin(),
        intent="test intent",
        role_responses={
            "target": target,
            "attacker": attacker,
            "target_judge": target_judge,
        },
        parameters=parameters or {},
        effective_limits=effective_limits,
        sql_seed_libraries={"gptfuzz": (seed,)},
        run_id="gptfuzz-run",
        attempt_id="gptfuzz-attempt",
    )


def _observe_failed_execute(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    from devtools.technique_parity import builtin
    from devtools.technique_parity.fixtures import ScriptedRoleClient

    captures: list[object] = []
    original_execute_route = builtin._execute_route

    def observed_execute_route(*args, **kwargs):
        capture = original_execute_route(*args, **kwargs)
        captures.append(capture)
        return capture

    monkeypatch.setattr(builtin, "_execute_route", observed_execute_route)
    monkeypatch.setattr(
        builtin._ScriptedTargetAdapter,
        "assert_exhausted",
        lambda _self: None,
    )
    monkeypatch.setattr(ScriptedRoleClient, "assert_exhausted", lambda _self: None)
    return captures


def test_gptfuzz_package_declares_feedback_driven_contract() -> None:
    package = _load_builtin()
    definition = package.definition

    assert definition.identity.id == "gptfuzz"
    assert definition.identity.version == "1.0.0"
    assert package.implementation.id == "modular"
    assert package.implementation.kind is ImplementationKind.PYTHON
    assert definition.capabilities == frozenset(
        {
            TechniqueCapability.ITERATIVE,
            TechniqueCapability.POPULATION_SEARCH,
            TechniqueCapability.MULTIPLE_HISTORIES,
        }
    )
    assert [role.id for role in definition.model_roles] == [
        "target",
        "attacker",
        "target_judge",
    ]
    [judge] = definition.judges
    assert judge.id == "target_judge"
    assert judge.schedule is JudgeSchedule.EVERY_ROUND
    assert judge.purposes == frozenset(
        {JudgePurpose.OFFICIAL_SCORE, JudgePurpose.SUCCESS_CONTROL}
    )
    assert definition.seed.strategy is SeedStrategy.SQL_LIBRARY
    assert definition.seed.source == "gptfuzz"
    assert definition.seed.required is True
    assert definition.parameters_schema["properties"] == {
        "max_rounds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 5,
        },
        "judge_threshold": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "default": 7,
        },
        "judge_per_round": {"type": "boolean", "default": True},
    }
    assert definition.recommended_limits == {"max_rounds": 5}
    assert definition.extensions["legacy_parity_replay"] == "feedback_mutation"
    assert (
        definition.extensions["parity_effective_judge_schedule"]
        == "judge_per_round"
    )


def test_gptfuzz_is_the_default_route_with_explicit_legacy_rollback() -> None:
    catalog = build_default_catalog(
        persistence=NoopTechniqueRuntimePersistence()
    )

    ordinary = catalog.resolve(TechniqueSelector("gptfuzz"))
    explicit_legacy = catalog.resolve(
        TechniqueSelector("gptfuzz", implementation_id="legacy")
    )

    assert ordinary.definition.identity.version == "1.0.0"
    assert ordinary.implementation.id == "modular"
    assert ordinary.implementation.kind is ImplementationKind.PYTHON
    assert explicit_legacy.definition.identity.version == "0.2.0"
    assert explicit_legacy.implementation.id == "legacy"
    assert explicit_legacy.implementation.kind is ImplementationKind.LEGACY


def test_gptfuzz_catalog_listing_derives_ui_flags_from_modular_contract() -> None:
    from app.api.store import EvaluationStore
    from app.persistence import NoopRunPersistence

    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=build_default_catalog(
            persistence=NoopTechniqueRuntimePersistence()
        ),
    )

    gptfuzz = next(
        item for item in store.list_techniques() if item["key"] == "gptfuzz"
    )

    assert gptfuzz["implementation"] == "modular"
    assert gptfuzz["multi_turn"] is True
    assert gptfuzz["requires_feedback_loop"] is True
    assert gptfuzz["requires_attacker_llm"] is True
    assert gptfuzz["requires_judge_per_round"] is True


def test_gptfuzz_feedback_selects_scored_mutant_as_next_parent() -> None:
    capture = _execute(
        attacker=[
            "<PROMPT>first mutant</PROMPT>",
            "<PROMPT>second mutant</PROMPT>",
        ],
        target=["first response", "second response"],
        target_judge=[
            '{"score":4,"rationale":"first score"}',
            '{"score":6,"rationale":"second score"}',
        ],
        parameters={"max_rounds": 2},
    )

    assert len(capture.role_calls["attacker"]) == 2
    assert len(capture.target_calls) == 2
    assert len(capture.role_calls["target_judge"]) == 2
    assert "Seed request for test intent" in str(
        capture.role_calls["attacker"][0].messages[-1]["content"]
    )
    second_mutation_request = str(
        capture.role_calls["attacker"][1].messages[-1]["content"]
    )
    assert "first mutant" in second_mutation_request
    assert "Seed request for test intent" not in second_mutation_request
    assert [
        [dict(message) for message in call.messages]
        for call in capture.target_calls
    ] == [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "first mutant"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "second mutant"},
        ],
    ]
    assert capture.projected_outcome.final_response == "second response"
    assert capture.projected_outcome.official_score == 6
    assert capture.outcome.outputs["final_prompt"] == "second mutant"
    assert capture.outcome.outputs["rounds_completed"] == 2


def test_gptfuzz_first_round_preserves_legacy_prompts_and_call_order() -> None:
    from app.attacklib.iterative.prompts import MUTATION_SYSTEM, MUTATION_USER

    capture = _execute(
        attacker=["<PROMPT>candidate mutant</PROMPT>"],
        target=["target response"],
        target_judge=['{"score":8,"rationale":"done"}'],
        parameters={"max_rounds": 1},
    )

    assert [dict(message) for message in capture.role_calls["attacker"][0].messages] == [
        {"role": "system", "content": MUTATION_SYSTEM},
        {
            "role": "user",
            "content": MUTATION_USER.format(
                prompt="Seed request for test intent",
                intent="test intent",
            ),
        },
    ]
    assert [dict(message) for message in capture.target_calls[0].messages] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "candidate mutant"},
    ]
    event_types = [event.event_type for event in capture.events]
    assert event_types.index("model.completed") < event_types.index("target.completed")
    assert event_types.index("target.completed") < event_types.index("judge.completed")


def test_gptfuzz_can_preserve_legacy_final_only_judge_flow() -> None:
    capture = _execute(
        attacker=[
            "<PROMPT>first mutant</PROMPT>",
            "<PROMPT>second mutant</PROMPT>",
        ],
        target=["first response", "second response"],
        target_judge=['{"score":6,"rationale":"final score"}'],
        parameters={"max_rounds": 2, "judge_per_round": False},
    )

    assert len(capture.role_calls["target_judge"]) == 1
    second_mutation_request = str(
        capture.role_calls["attacker"][1].messages[-1]["content"]
    )
    assert "Seed request for test intent" in second_mutation_request
    assert "first mutant" not in second_mutation_request
    assert capture.projected_outcome.final_response == "second response"
    assert capture.projected_outcome.official_score == 6
    assert capture.outcome.outputs["final_prompt"] == "second mutant"
    assert [
        event.payload["score"]
        for event in capture.events
        if event.event_type == "round.completed"
    ] == [None, None]


def test_gptfuzz_stops_when_feedback_reaches_the_judge_threshold() -> None:
    capture = _execute(
        attacker=["<PROMPT>successful mutant</PROMPT>"],
        target=["successful response"],
        target_judge=['{"score":8,"rationale":"threshold reached"}'],
        parameters={"max_rounds": 3, "judge_threshold": 7},
    )

    assert len(capture.role_calls["attacker"]) == 1
    assert len(capture.target_calls) == 1
    assert len(capture.role_calls["target_judge"]) == 1
    assert capture.outcome.outputs["rounds_completed"] == 1
    assert [
        event.payload["reason"]
        for event in capture.events
        if event.event_type == "stop.triggered"
    ] == ["judge_threshold_reached"]


def test_gptfuzz_retains_the_stronger_parent_when_a_mutant_scores_worse() -> None:
    capture = _execute(
        attacker=[
            "<PROMPT>strong mutant</PROMPT>",
            "<PROMPT>weak mutant</PROMPT>",
            "<PROMPT>third mutant</PROMPT>",
        ],
        target=["strong response", "weak response", "third response"],
        target_judge=[
            '{"score":5,"rationale":"strongest"}',
            '{"score":3,"rationale":"weaker"}',
            '{"score":4,"rationale":"still weaker"}',
        ],
        parameters={"max_rounds": 3},
    )

    third_mutation_request = str(
        capture.role_calls["attacker"][2].messages[-1]["content"]
    )
    assert "strong mutant" in third_mutation_request
    assert "weak mutant" not in third_mutation_request
    assert capture.projected_outcome.final_response == "strong response"
    assert capture.projected_outcome.official_score == 5
    assert capture.outcome.outputs["final_prompt"] == "strong mutant"


def test_gptfuzz_preserves_minimum_score_and_selection_control() -> None:
    capture = _execute(
        attacker=[
            "<PROMPT>zero mutant</PROMPT>",
            "<PROMPT>second mutant</PROMPT>",
        ],
        target=["zero response", "second response"],
        target_judge=[
            '{"score":1,"rationale":"minimum score"}',
            '{"score":1,"rationale":"same control score"}',
        ],
        parameters={"max_rounds": 2},
    )

    second_mutation_request = str(
        capture.role_calls["attacker"][1].messages[-1]["content"]
    )
    assert "zero mutant" in second_mutation_request
    assert capture.projected_outcome.final_response == "zero response"
    assert capture.projected_outcome.official_score == 1
    first_round = next(
        event
        for event in capture.events
        if event.event_type == "round.completed" and event.round == 1
    )
    assert first_round.payload["score"] == 1
    assert first_round.payload["control_score"] == 1


def test_gptfuzz_minimum_score_reaches_minimum_threshold() -> None:
    capture = _execute(
        attacker=["<PROMPT>zero mutant</PROMPT>"],
        target=["zero response"],
        target_judge=['{"score":1,"rationale":"minimum score"}'],
        parameters={"max_rounds": 2, "judge_threshold": 1},
    )

    assert capture.projected_outcome.official_score == 1
    assert capture.outcome.outputs["rounds_completed"] == 1
    assert [
        event.payload["reason"]
        for event in capture.events
        if event.event_type == "stop.triggered"
    ] == ["judge_threshold_reached"]


def test_gptfuzz_honors_a_pinned_effective_round_limit() -> None:
    capture = _execute(
        attacker=[
            "<PROMPT>first mutant</PROMPT>",
            "<PROMPT>second mutant</PROMPT>",
        ],
        target=["first response", "second response"],
        target_judge=[
            '{"score":2,"rationale":"first"}',
            '{"score":3,"rationale":"second"}',
        ],
        parameters={"max_rounds": 3},
        effective_limits={"max_rounds": 2},
    )

    assert capture.outcome.outputs["rounds_completed"] == 2
    started = next(
        event for event in capture.events if event.event_type == "run.started"
    )
    assert started.payload["max_rounds"] == 2


def test_gptfuzz_attacker_failure_emits_a_redacted_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from devtools.technique_parity.fixtures import ScriptedRoleClient

    captures = _observe_failed_execute(monkeypatch)

    def fail_attacker(self, messages, *, request_context, **options):
        del messages, request_context, options
        if self.role == "attacker":
            raise RuntimeError("attacker failed with sk-secret123")
        raise AssertionError(f"unexpected role {self.role}")

    monkeypatch.setattr(ScriptedRoleClient, "chat_correlated_once", fail_attacker)

    with pytest.raises(RuntimeError, match="did not complete"):
        _execute(
            attacker=["<PROMPT>unused</PROMPT>"],
            target=["unused response"],
            target_judge=['{"score":1,"rationale":"unused"}'],
            parameters={"max_rounds": 1},
        )

    failed = [
        event for event in captures[0].events if event.event_type == "run.failed"
    ]
    assert len(failed) == 1
    assert failed[0].payload == {
        "category": "RuntimeError",
        "error": "attacker failed with [REDACTED]",
    }


def test_gptfuzz_budget_exhaustion_is_terminal_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = _observe_failed_execute(monkeypatch)

    with pytest.raises(RuntimeError, match="did not complete"):
        _execute(
            attacker=["<PROMPT>unused</PROMPT>"],
            target=["unused response"],
            target_judge=['{"score":1,"rationale":"unused"}'],
            parameters={"max_rounds": 1},
            effective_limits={"max_model_calls": 0},
        )

    exhausted = [
        event
        for event in captures[0].events
        if event.event_type == "budget.exhausted"
    ]
    assert len(exhausted) == 1
    assert exhausted[0].payload == {
        "resource": "model_calls",
        "limit": 0,
        "requested": 1,
    }
    assert captures[0].role_calls["attacker"] == ()


def test_gptfuzz_cancellation_is_terminal_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from devtools.technique_parity import builtin

    captures = _observe_failed_execute(monkeypatch)
    original_execute = builtin._execute_builtin_parity_route

    async def cancelled_execute(router, route, request, context):
        context.cancellation.cancel("forced GPTFuzz cancellation")
        return await original_execute(router, route, request, context)

    monkeypatch.setattr(builtin, "_execute_builtin_parity_route", cancelled_execute)

    with pytest.raises(RuntimeError, match="did not complete"):
        _execute(
            attacker=["<PROMPT>unused</PROMPT>"],
            target=["unused response"],
            target_judge=['{"score":1,"rationale":"unused"}'],
            parameters={"max_rounds": 1},
        )

    capture = captures[0]
    assert [event.event_type for event in capture.events].count("run.cancelled") == 1
    assert capture.role_calls["attacker"] == ()
    assert capture.target_calls == ()


def test_gptfuzz_cancellation_after_judging_prevents_false_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from devtools.technique_parity import builtin
    from devtools.technique_parity.fixtures import ScriptedRoleClient

    captures = _observe_failed_execute(monkeypatch)
    original_execute = builtin._execute_builtin_parity_route
    original_call = ScriptedRoleClient.chat_correlated_once
    active_context: dict[str, object] = {}

    async def observed_execute(router, route, request, context):
        active_context["context"] = context
        return await original_execute(router, route, request, context)

    def cancel_after_judge(self, messages, *, request_context, **options):
        result = original_call(
            self,
            messages,
            request_context=request_context,
            **options,
        )
        if self.role == "target_judge":
            active_context["context"].cancellation.cancel(
                "forced late GPTFuzz cancellation"
            )
        return result

    monkeypatch.setattr(builtin, "_execute_builtin_parity_route", observed_execute)
    monkeypatch.setattr(ScriptedRoleClient, "chat_correlated_once", cancel_after_judge)

    with pytest.raises(RuntimeError, match="did not complete"):
        _execute(
            attacker=["<PROMPT>candidate</PROMPT>"],
            target=["response"],
            target_judge=['{"score":8,"rationale":"success"}'],
            parameters={"max_rounds": 1},
        )

    event_types = [event.event_type for event in captures[0].events]
    assert event_types.count("run.cancelled") == 1
    assert "run.completed" not in event_types


def test_gptfuzz_judge_failure_is_inconclusive_and_keeps_running() -> None:
    capture = _execute(
        attacker=[
            "<PROMPT>first mutant</PROMPT>",
            "<PROMPT>second mutant</PROMPT>",
        ],
        target=["first response", "second response"],
        target_judge=[
            "not valid judge JSON",
            '{"score":4,"rationale":"second scored"}',
        ],
        parameters={"max_rounds": 2},
    )

    first_round = next(
        event
        for event in capture.events
        if event.event_type == "round.completed" and event.round == 1
    )
    assert first_round.payload["judge_status"] == "failed"
    assert first_round.payload["score"] is None
    assert first_round.payload["control_score"] == 1
    assert capture.projected_outcome.final_response == "second response"
    assert capture.projected_outcome.official_score == 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("<PROMPT> tagged mutant </PROMPT>", "tagged mutant"),
        ("<PROMPT>\npartial mutant", "partial mutant"),
        ("Sure!\nHere is the prompt:\nPAYLOAD", "PAYLOAD"),
        ("Here's a candidate prompt:\nPAYLOAD", "PAYLOAD"),
        ('"quoted mutant"', "quoted mutant"),
        ("   ", ""),
    ),
)
def test_gptfuzz_extracts_mutants_with_exact_legacy_rules(
    raw: str,
    expected: str,
) -> None:
    from app.techniques.builtin.gptfuzz.executor import _extract_prompt

    assert _extract_prompt(raw) == expected


def test_gptfuzz_projects_each_round_and_saves_each_checkpoint() -> None:
    from app.api.store import TargetKind
    from app.techniques.application import (
        ApplicationExecutionCapture,
        project_application_capture,
    )

    capture = _execute(
        attacker=[
            "<PROMPT>first mutant</PROMPT>",
            "<PROMPT>second mutant</PROMPT>",
        ],
        target=["first response", "second response"],
        target_judge=[
            '{"score":2,"rationale":"first"}',
            '{"score":6,"rationale":"second"}',
        ],
        parameters={"max_rounds": 2},
    )
    application_capture = ApplicationExecutionCapture(
        request=capture.request,
        outcome=capture.outcome,
        projected_outcome=capture.projected_outcome,
        events=capture.events,
        history_snapshots=capture.history_snapshots,
        conversation_projection=capture.conversation_projection,
        failure_category=capture.failure_category,
    )

    projected = project_application_capture(
        application_capture,
        definition=_load_builtin().definition,
        target_kind=TargetKind.CHATBOT,
        target_id="deterministic-target",
        requested_by="test",
    )

    assert [entry["round_number"] for entry in projected.conversation] == ["1", "2"]
    assert [entry["attacker"]["output"] for entry in projected.conversation] == [
        "first mutant",
        "second mutant",
    ]
    assert [entry["target"]["response"] for entry in projected.conversation] == [
        "first response",
        "second response",
    ]
    assert [entry["judge"]["score"] for entry in projected.conversation] == [2, 6]
    checkpoints = [
        event for event in capture.events if event.event_type == "checkpoint.saved"
    ]
    assert len(checkpoints) == 2
    assert len({event.payload["checkpoint_id"] for event in checkpoints}) == 2


@pytest.mark.parametrize(
    "scenario",
    (
        "feedback_selects_parent",
        "best_parent_retained",
        "judge_threshold",
        "target_judge_zero",
        "target_judge_failure",
        "legacy_final_only",
    ),
)
def test_gptfuzz_feedback_driven_deterministic_parity(scenario: str) -> None:
    from devtools.technique_parity.builtin import compare_builtin_fixture

    report, captures = compare_builtin_fixture("gptfuzz", scenario=scenario)

    assert report.passed, report.render_text()
    assert len(captures) == 2


@pytest.mark.parametrize(
    ("scenario", "expected_judge_per_round"),
    (
        ("feedback_selects_parent", True),
        ("legacy_final_only", False),
    ),
)
def test_gptfuzz_legacy_parity_arm_executes_registered_loop(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_judge_per_round: bool,
) -> None:
    from app.attacklib.text_attacks import GPTFuzzAttack
    from devtools.technique_parity.builtin import compare_builtin_fixture

    observed: list[tuple[str, bool]] = []
    original = GPTFuzzAttack.run_iterative

    def observed_run(self, **kwargs):
        observed.append(
            (
                kwargs["request"].extra["run_id"],
                kwargs["judge_per_round"],
            )
        )
        return original(self, **kwargs)

    monkeypatch.setattr(GPTFuzzAttack, "run_iterative", observed_run)

    report, captures = compare_builtin_fixture(
        "gptfuzz",
        scenario=scenario,
    )

    assert report.passed, report.render_text()
    assert len(observed) == 1
    assert observed[0][1] is expected_judge_per_round
    assert "legacy" in observed[0][0]
    assert captures[0]["identity"]["implementation_id"] == "legacy"
    assert captures[1]["identity"]["implementation_id"] == "modular"


def test_primary_store_executes_feedback_driven_modular_gptfuzz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from app.api import store as store_module
    from app.api.store import (
        EvaluationStore,
        _definition_judge_policies,
        _schema_defaults,
    )
    from app.evaluation import TargetKind
    from app.persistence import NoopRunPersistence
    from app.techniques.core.models import ProviderCallResult

    scripted_roles = {
        "attacker": [
            "<PROMPT>first mutant</PROMPT>",
            "<PROMPT>second mutant</PROMPT>",
        ],
        "target_judge": [
            '{"score":6,"rationale":"stronger first"}',
            '{"score":4,"rationale":"weaker second"}',
        ],
    }
    target_requests: list[object] = []

    class RecordedRoleClient:
        default_max_tokens = 1024

        def __init__(self, role: str) -> None:
            self.role = role

        def chat_correlated_once(self, messages, *, request_context, **options):
            del messages, options
            assert request_context.role == self.role
            return ProviderCallResult(text=scripted_roles[self.role].pop(0))

    class RecordedTargetAdapter:
        def __init__(self, *, response: str) -> None:
            del response
            self.responses = ["first response", "second response"]

        def execute(self, request):
            target_requests.append(request)
            return SimpleNamespace(
                final_output=self.responses.pop(0),
                tool_calls=(),
                metadata={"adapter": "provider-free-recorded"},
            )

    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    route = store.router.resolve("gptfuzz")
    definition = route.registered.definition
    parameters = {
        **_schema_defaults(definition.parameters_schema),
        "max_rounds": 2,
    }
    technique_config = SimpleNamespace(
        parameters=parameters,
        judge_policies=_definition_judge_policies(definition, parameters),
        effective_limits=definition.recommended_limits,
        model_roles={},
    )
    monkeypatch.setattr(store_module, "ProviderFreeDemoRoleClient", RecordedRoleClient)
    monkeypatch.setattr(
        store_module,
        "ProviderFreeDemoTargetAdapter",
        RecordedTargetAdapter,
    )

    summary, _result = store._execute_modular_application_route(
        route=route,
        technique_config=technique_config,
        target_kind=TargetKind.CHATBOT,
        mode="demo",
        target_config={"unsafe_intent": "provider-free GPTFuzz fixture"},
        evaluation_id=None,
        run_id="gptfuzz-store-run",
        experiment_id=None,
        cell_id=None,
        attempt_id="gptfuzz-store-attempt",
    )

    assert route.registered.definition.identity.version == "1.0.0"
    assert route.registered.implementation.id == "modular"
    assert summary.status == "completed"
    assert summary.final_output == "first response"
    assert summary.score == 60
    assert len(summary.conversation) == 2
    assert [item["attacker"]["output"] for item in summary.conversation] == [
        "first mutant",
        "second mutant",
    ]
    assert [
        [dict(message) for message in request.messages]
        for request in target_requests
    ] == [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "first mutant"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "second mutant"},
        ],
    ]
