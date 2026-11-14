from __future__ import annotations

import ast
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Lock

import pytest

from app.api.store import EvaluationStore, _content_hash
from app.evaluation import TargetKind
from app.experiment_plan_integrity import cell_id as canonical_cell_id
from app.persistence import InMemoryExperimentAttemptPersistence, NoopRunPersistence
from app.persistence.technique_runtime import NoopTechniqueRuntimePersistence
from app.techniques.application import (
    ApplicationExecutionCapture,
    project_application_capture,
)
from app.techniques.contracts.definition import (
    ImplementationKind,
    JudgePurpose,
    JudgeSchedule,
    ResultPolicy,
    SeedStrategy,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
)
from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.execution import (
    CorrelationIds,
    FailureCategory,
    ModelRoleBinding,
    ResolvedRunSpec,
    TechniqueRequest,
)
from app.techniques.core.histories import HistoryMessage, HistorySnapshot
from app.techniques.core.models import ProviderCallResult
from app.techniques.projection.conversation import ConversationProjector
from app.techniques.projection.outcome import OutcomeProjector
from app.techniques.plugins.manifest import MANIFEST_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_TECHNIQUES = {
    "direct_prompt_injection",
    "crescendo_liar",
    "autodan",
    "tap",
}
CORRELATION = CorrelationIds(
    experiment_id="projection-experiment",
    cell_id="projection-cell",
    run_id="projection-run",
    attempt_id="projection-attempt",
)


class NetworkIncapableRecordedClient:
    """In-memory application-boundary client with no network implementation."""

    network_capable = False
    max_tokens = 1024

    def __init__(self, *, role: str, model: str) -> None:
        self.role = role
        self.model = model
        self.calls: list[dict[str, object]] = []

    def prepare_messages(self, messages):
        return [dict(message) for message in messages]

    def _response(self, options: dict[str, object]) -> str:
        if self.role == "target":
            return "Recorded target response with immutable ownership."
        if self.role == "target_judge":
            return json.dumps({"score": 8, "rationale": "recorded judge evidence"})
        response_format = options.get("response_format")
        schema_name = ""
        if isinstance(response_format, dict):
            raw_schema = response_format.get("json_schema")
            if isinstance(raw_schema, dict):
                schema_name = str(raw_schema.get("name") or "")
        if "crescendo" in schema_name:
            return json.dumps({"turn": "Recorded adjacent-topic candidate."})
        if "fabric" in schema_name:
            return json.dumps(
                {
                    "poisoned": (
                        "Recorded fabricated partial-compliance response long enough to satisfy the legacy validator."
                    )
                }
            )
        if "autodan" in schema_name:
            return json.dumps(
                {
                    "prompt": (
                        "Recorded long-form candidate prompt with enough deterministic "
                        "content to satisfy the population validator."
                    )
                }
            )
        return "<PROMPT>Recorded tree candidate prompt.</PROMPT>"

    def chat_with_retry(self, messages, **options):
        self.calls.append(
            {
                "transport": "legacy-recorded",
                "messages": tuple(dict(message) for message in messages),
                "options": dict(options),
            }
        )
        return self._response(dict(options)), {"total_tokens": 1}

    def chat_correlated_once(self, messages, *, request_context, **options):
        self.calls.append(
            {
                "transport": "modular-recorded",
                "messages": tuple(dict(message) for message in messages),
                "options": dict(options),
                "request_context": request_context,
            }
        )
        return ProviderCallResult(text=self._response(dict(options)))


class DelayedOperationOwnedRecordedClient:
    """Concurrent fixture whose response is derived from its request owner."""

    network_capable = False
    max_tokens = 1024

    def __init__(self, *, role: str, barrier: Barrier | None = None) -> None:
        self.role = role
        self.model = f"recorded-{role}"
        self.barrier = barrier
        self.calls: list[tuple[object, str]] = []
        self._lock = Lock()

    def prepare_messages(self, messages):
        return [dict(message) for message in messages]

    def chat_correlated_once(self, messages, *, request_context, **options):
        del messages, options
        correlation = request_context.correlation
        owner = str(correlation.experiment_id)
        with self._lock:
            self.calls.append((request_context, owner))
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
            with self._lock:
                delayed_owner = sorted(item[1] for item in self.calls)[-1]
            if owner == delayed_owner:
                time.sleep(0.02)
        if self.role == "target_judge":
            return ProviderCallResult(
                text=json.dumps({"score": 8, "rationale": f"owned judge for {owner}"}),
                provider_response_id=f"judge-response:{owner}",
            )
        return ProviderCallResult(
            text=f"owned target response for {owner}",
            provider_response_id=f"target-response:{owner}",
        )


class RecordingTechniqueRuntimePersistence(NoopTechniqueRuntimePersistence):
    """In-memory durable boundary that retains queryable normalized events."""

    def __init__(self) -> None:
        super().__init__()
        self.persisted_events: list[ExecutionEvent] = []
        self._capture_lock = Lock()

    def append_events(self, events: tuple[ExecutionEvent, ...]) -> None:
        super().append_events(events)
        with self._capture_lock:
            self.persisted_events.extend(events)


class SequencedOperationOwnedRecordedClient:
    """Provider-free client that keeps each scripted outcome with its call."""

    network_capable = False
    max_tokens = 1024

    def __init__(self, *, role: str, outcomes: list[object]) -> None:
        self.role = role
        self.model = f"recorded-{role}"
        self._outcomes = list(outcomes)
        self.calls: list[object] = []
        self._lock = Lock()

    def prepare_messages(self, messages):
        return [dict(message) for message in messages]

    def chat_correlated_once(self, messages, *, request_context, **options):
        del messages, options
        with self._lock:
            self.calls.append(request_context)
            if not self._outcomes:
                raise AssertionError("recorded outcome queue exhausted")
            outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return ProviderCallResult(text=str(outcome))


def _event(
    event_id: str,
    sequence: int,
    event_type: str,
    *,
    operation_id: str | None = None,
    branch_id: str | None = None,
    round_number: int | None = None,
    payload: dict[str, object] | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=event_id,
        correlation=CORRELATION,
        sequence=sequence,
        event_type=event_type,
        schema_version=1,
        timestamp=datetime(2026, 9, 1, tzinfo=UTC),
        technique_id="future_generation_probe",
        technique_version="1.0.0",
        implementation_id="modular",
        operation_id=operation_id,
        branch_id=branch_id,
        round=round_number,
        payload=payload or {},
    )


def _future_generation_capture() -> tuple[ApplicationExecutionCapture, TechniqueDefinition]:
    events = (
        _event("run-started", 1, "run.started"),
        _event(
            "candidate-winner-created",
            2,
            "candidate.created",
            operation_id="future:generation:1:candidate:winner",
            branch_id="winner",
            round_number=1,
            payload={
                "generation": 1,
                "candidate_id": "winner",
                "position": 1,
                "origin": "seed",
                "parent_ids": (),
                "prompt": "winning prompt",
            },
        ),
        _event(
            "target-winner",
            3,
            "target.completed",
            operation_id="future:generation:1:candidate:winner:target",
            branch_id="winner",
            round_number=1,
            payload={
                "target_call_id": "target-call-winner",
                "history_snapshot_id": "snapshot-winner",
                "response": "winning response",
                "response_ref": "selected-response",
                "tool_calls": ("winner-tool",),
                "metadata": {"candidate": "winner", "provider_response_id": "winner-id"},
            },
        ),
        _event(
            "judge-winner",
            4,
            "judge.completed",
            operation_id="future:generation:1:candidate:winner:target-judge",
            branch_id="winner",
            round_number=1,
            payload={
                "definition_id": "target_judge",
                "result_id": "judge-result-winner",
                "status": "succeeded",
                "score": 8,
                "rationale": "winner rationale",
                "target_response_event_id": "target-winner",
                "candidate_id": "winner",
            },
        ),
        _event(
            "candidate-winner-completed",
            5,
            "candidate.completed",
            operation_id="future:generation:1:candidate:winner",
            branch_id="winner",
            round_number=1,
            payload={
                "generation": 1,
                "candidate_id": "winner",
                "position": 1,
                "origin": "seed",
                "parent_ids": (),
                "prompt": "winning prompt",
                "target_response_event_id": "target-winner",
                "judge_status": "succeeded",
                "score": 8,
                "succeeded": True,
            },
        ),
        _event(
            "candidate-last-created",
            6,
            "candidate.created",
            operation_id="future:generation:1:candidate:last",
            branch_id="last",
            round_number=1,
            payload={
                "generation": 1,
                "candidate_id": "last",
                "position": 2,
                "origin": "mutation",
                "parent_ids": ("winner",),
                "prompt": "last prompt",
            },
        ),
        _event(
            "target-last",
            7,
            "target.completed",
            operation_id="future:generation:1:candidate:last:target",
            branch_id="last",
            round_number=1,
            payload={
                "target_call_id": "target-call-last",
                "history_snapshot_id": "snapshot-last",
                "response": "last response",
                "response_ref": "last-response",
                "tool_calls": ("last-tool",),
                "metadata": {"candidate": "last", "provider_response_id": "last-id"},
            },
        ),
        _event(
            "judge-last",
            8,
            "judge.completed",
            operation_id="future:generation:1:candidate:last:target-judge",
            branch_id="last",
            round_number=1,
            payload={
                "definition_id": "target_judge",
                "result_id": "judge-result-last",
                "status": "succeeded",
                "score": 4,
                "rationale": "last rationale",
                "target_response_event_id": "target-last",
                "candidate_id": "last",
            },
        ),
        _event(
            "candidate-last-completed",
            9,
            "candidate.completed",
            operation_id="future:generation:1:candidate:last",
            branch_id="last",
            round_number=1,
            payload={
                "generation": 1,
                "candidate_id": "last",
                "position": 2,
                "origin": "mutation",
                "parent_ids": ("winner",),
                "prompt": "last prompt",
                "target_response_event_id": "target-last",
                "judge_status": "succeeded",
                "score": 4,
                "succeeded": False,
            },
        ),
        _event(
            "candidate-selected",
            10,
            "candidate.selected",
            operation_id="future:generation:1:selection",
            branch_id="winner",
            round_number=1,
            payload={
                "generation": 1,
                "candidate_id": "winner",
                "target_response_event_id": "target-winner",
                "judge_event_id": "judge-winner",
                "score": 8,
            },
        ),
        _event("run-completed", 11, "run.completed"),
    )
    result_policy = ResultPolicy(
        official_judge="target_judge",
        final_response="selected-response",
        success_threshold=7,
    )
    definition = TechniqueDefinition(
        identity=TechniqueIdentity(
            "future_generation_probe",
            "1.0.0",
            "Future Generation Probe",
            "future",
        ),
        summary="Synthetic package proving format-driven projection.",
        capabilities=frozenset({TechniqueCapability.POPULATION_SEARCH, TechniqueCapability.PARALLEL_BRANCHES}),
        result_policy=result_policy,
        presentation={"conversation_format": "generation"},
    )
    spec = ResolvedRunSpec(
        technique_id=definition.identity.id,
        technique_version=definition.identity.version,
        implementation_id="modular",
        implementation_kind=ImplementationKind.PYTHON,
        contract_version="1",
        manifest_hash="a" * 64,
        model_role_bindings={
            "target": ModelRoleBinding.from_profile(
                logical_role="target",
                profile_id="recorded-target",
                provider="recorded",
                model="recorded-target",
            )
        },
    )
    request = TechniqueRequest(correlation=CORRELATION, spec=spec)
    projected = OutcomeProjector().project(events, result_policy=result_policy)
    prompt = HistoryMessage(
        message_id="prompt-message",
        history_id="audit_history",
        branch_id="winner",
        role="user",
        content="winning prompt",
        source_kind="candidate",
    )
    capture = ApplicationExecutionCapture(
        request=request,
        outcome=None,
        projected_outcome=projected,
        events=events,
        history_snapshots={
            "audit_history": HistorySnapshot(
                snapshot_id="snapshot-winner",
                history_id="audit_history",
                message_ids=(prompt.message_id,),
                messages=(prompt,),
            )
        },
        conversation_projection=ConversationProjector().project(
            events,
            presentation=definition.presentation,
        ),
        failure_category=None,
    )
    return capture, definition


def test_application_projection_uses_manifest_format_and_exact_selected_events() -> None:
    capture, definition = _future_generation_capture()

    projection = project_application_capture(
        capture,
        definition=definition,
        target_kind=TargetKind.CHATBOT,
        target_id="recorded-target",
        requested_by="task-29-test",
    )

    assert projection.result.response.final_output == "winning response"
    assert projection.result.response.tool_calls == ("winner-tool",)
    assert projection.result.response.metadata == {
        "candidate": "winner",
        "provider_response_id": "winner-id",
    }
    assert [entry["kind"] for entry in projection.conversation] == ["generation"]
    generation = projection.conversation[0]
    candidates = {candidate["candidate_id"]: candidate for candidate in generation["candidates"]}
    assert candidates["winner"] == {
        "candidate_id": "winner",
        "index": 1,
        "position": "1",
        "origin": "seed",
        "parent_ids": [],
        "prompt": "winning prompt",
        "response": "winning response",
        "judge_status": "succeeded",
        "judge_score": 8,
        "judge_rationale": "winner rationale",
        "is_best": True,
        "is_selected": True,
    }
    assert candidates["last"]["response"] == "last response"
    assert candidates["last"]["judge_rationale"] == "last rationale"
    judge_trace = next(trace for trace in projection.result.traces if trace.event_type == "judge_evaluation")
    assert judge_trace.attributes["judge_score"] == "8"
    assert judge_trace.attributes["judge_rationale"] == "winner rationale"


def test_application_projection_uses_exact_selected_failed_judge_error() -> None:
    capture, definition = _future_generation_capture()
    events = tuple(
        replace(
            event,
            event_type="judge.failed",
            payload={
                **dict(event.payload),
                "status": "failed",
                "score": None,
                "parse_status": (
                    "selected-error" if event.event_id == "judge-winner" else "last-error"
                ),
            },
        )
        if event.event_id in {"judge-winner", "judge-last"}
        else event
        for event in capture.events
    )
    failed_capture = replace(
        capture,
        events=events,
        projected_outcome=OutcomeProjector().project(
            events,
            result_policy=definition.result_policy,
        ),
        conversation_projection=ConversationProjector().project(
            events,
            presentation=definition.presentation,
        ),
        failure_category=FailureCategory.JUDGE_PARSE,
    )

    projection = project_application_capture(
        failed_capture,
        definition=definition,
        target_kind=TargetKind.CHATBOT,
        target_id="recorded-target",
        requested_by="task-30-test",
    )

    assert failed_capture.projected_outcome.official_judge_event_id == "judge-winner"
    assert projection.error == "target_judge failed: selected-error"


def test_future_tree_package_projects_and_serializes_without_id_registration() -> None:
    capture, definition = _future_generation_capture()
    tree_id = "future_tree_probe"
    tree_events = tuple(
        replace(
            event,
            technique_id=tree_id,
            payload={
                **dict(event.payload),
                **(
                    {
                        "depth": event.round or 1,
                        "parent_id": ("winner" if event.payload.get("candidate_id") == "last" else "root"),
                    }
                    if event.event_type.startswith("candidate.")
                    else {}
                ),
            },
        )
        for event in capture.events
    )
    tree_definition = replace(
        definition,
        identity=replace(
            definition.identity,
            id=tree_id,
            name="Future Tree Probe",
        ),
        presentation={"conversation_format": "tree"},
    )
    tree_request = replace(
        capture.request,
        spec=replace(capture.request.spec, technique_id=tree_id),
    )
    tree_capture = replace(
        capture,
        request=tree_request,
        events=tree_events,
        projected_outcome=OutcomeProjector().project(
            tree_events,
            result_policy=tree_definition.result_policy,
        ),
        conversation_projection=ConversationProjector().project(
            tree_events,
            presentation=tree_definition.presentation,
        ),
    )

    projection = project_application_capture(
        tree_capture,
        definition=tree_definition,
        target_kind=TargetKind.CHATBOT,
        target_id="recorded-target",
        requested_by="task-29-test",
    )

    assert [entry["kind"] for entry in projection.conversation] == ["tree"]
    branches = projection.conversation[0]["branches"]
    assert [branch["branch_id"] for branch in branches] == ["winner", "last"]
    assert branches[0]["response"] == "winning response"
    assert branches[0]["judge"]["rationale"] == "winner rationale"
    assert branches[0]["is_selected"] is True
    assert branches[1]["parent_id"] == "winner"
    assert branches[1]["is_selected"] is False
    assert json.loads(json.dumps(projection.conversation))[0]["kind"] == "tree"


def test_one_experiment_executes_eight_modular_and_real_legacy_cells_provider_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = NoopTechniqueRuntimePersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=runtime,
        experiment_attempt_persistence=attempts,
    )
    target = NetworkIncapableRecordedClient(role="target", model="recorded-target")
    attacker = NetworkIncapableRecordedClient(role="attacker", model="recorded-attacker")
    judge = NetworkIncapableRecordedClient(role="target_judge", model="recorded-judge")
    clients = (target, attacker, judge)
    assert all(client.network_capable is False for client in clients)
    monkeypatch.setattr(store, "resolve_profile_config", lambda config: dict(config))
    monkeypatch.setattr(store, "_build_chat_client", lambda _config: target)
    monkeypatch.setattr(store, "_build_attacker_client", lambda _config: attacker)
    monkeypatch.setattr(
        store,
        "_build_judge_client",
        lambda _config, *, target_client, attacker_client: judge,
    )

    registered_legacy: list[str] = []
    iterative_legacy: list[str] = []
    tap_loop_kwargs: list[dict[str, object]] = []
    original_registered = store.runner.execute_registered_technique
    original_iterative = store.runner.execute_iterative_technique
    original_tap_runner = store.runner.execute_legacy_async_tap

    def tracked_registered(**kwargs):
        registered_legacy.append(str(kwargs["technique_key"]))
        return original_registered(**kwargs)

    def tracked_iterative(**kwargs):
        iterative_legacy.append(str(kwargs["technique_key"]))
        return original_iterative(**kwargs)

    def tracked_tap_runner(**kwargs):
        return original_tap_runner(**kwargs)

    monkeypatch.setattr(store.runner, "execute_registered_technique", tracked_registered)
    monkeypatch.setattr(store.runner, "execute_iterative_technique", tracked_iterative)
    monkeypatch.setattr(store.runner, "execute_legacy_async_tap", tracked_tap_runner)

    from app.attacklib.iterative import async_tap

    original_tap_loop = async_tap.run_async_tap_loop

    async def tracked_tap_loop(**kwargs):
        tap_loop_kwargs.append(dict(kwargs))
        return await original_tap_loop(**kwargs)

    monkeypatch.setattr(async_tap, "run_async_tap_loop", tracked_tap_loop)

    dataset = store.create_dataset(
        name="Task 29 provider-free mixed cohort",
        prompts=[{"text": "recorded unsafe-intent fixture"}],
    )
    target_config = {
        "api_key": "recorded-provider-disabled",
        "model": target.model,
        "attacker_api_key": "recorded-provider-disabled",
        "attacker_model": attacker.model,
        "judge_mode": "custom",
        "judge_api_key": "recorded-provider-disabled",
        "judge_model": judge.model,
        "max_rounds": "1",
        "branch_factor": "1",
        "keep_top_k": "1",
        "max_concurrency": "1",
        "prune_threshold": "3",
    }
    with monkeypatch.context() as planning:
        planning.setattr("app.api.store.threading.Thread.start", lambda self: None)
        created = store.create_experiment(
            name="Task 29 provider-free mixed cohort",
            dataset_id=dataset["dataset_id"],
            technique_keys=[],
            technique_configs=[
                {"technique_key": "direct_prompt_injection"},
                {
                    "technique_key": "direct_prompt_injection",
                    "implementation_id": "legacy",
                },
                {
                    "technique_key": "crescendo_liar",
                    "parameters": {"max_rounds": 1},
                },
                {"technique_key": "crescendo_liar", "implementation_id": "legacy"},
                {
                    "technique_key": "autodan",
                    "parameters": {
                        "population_size": 1,
                        "num_generations": 1,
                        "random_seed": 29,
                    },
                },
                {"technique_key": "autodan", "implementation_id": "legacy"},
                {
                    "technique_key": "tap",
                    "parameters": {
                        "max_rounds": 1,
                        "branch_factor": 1,
                        "keep_top_k": 1,
                        "max_concurrency": 1,
                    },
                },
                {"technique_key": "tap", "implementation_id": "legacy"},
            ],
            target_config=target_config,
            mode="live",
            repeat_count=1,
            max_rounds=1,
        )

    state = store._experiments[created["experiment_id"]]
    assert state.plan is not None
    assert len(state.plan.technique_configs) == 8
    assert len(state.plan.cells) == 8
    assert len({item.technique_config_id for item in state.plan.technique_configs}) == 8
    assert len({item.spec_hash for item in state.plan.technique_configs}) == 8
    assert all(item.spec_hash == _content_hash(item.resolved_spec) for item in state.plan.technique_configs)
    assert all(
        cell.cell_id
        == canonical_cell_id(
            state.experiment_id,
            cell.technique_config_id,
            cell.prompt_id,
            cell.repeat_index,
        )
        for cell in state.plan.cells
    )
    for item in state.plan.technique_configs:
        route = store.router.resolve(
            item.technique_id,
            technique_version=item.technique_version,
            implementation_id=item.implementation_id,
        )
        assert route.registered.manifest_hash == item.manifest_hash
        assert route.registered.asset_hashes == item.asset_hashes
    assert [item.implementation_id for item in state.plan.technique_configs] == [
        "modular",
        "legacy",
    ] * 4

    store._run_experiment(
        state.experiment_id,
        target_config,
        target_config["api_key"],
    )

    assert state.status == "completed"
    assert len(state.cell_results) == 8
    assert len({cell["cell_id"] for cell in state.cell_results}) == 8
    assert len({cell["run_id"] for cell in state.cell_results}) == 8
    assert len({cell["attempt_id"] for cell in state.cell_results}) == 8
    assert len({id(cell["conversation"]) for cell in state.cell_results}) == 8
    assert all(cell["selected_attempt_id"] == cell["attempt_id"] for cell in state.cell_results)
    assert registered_legacy == ["direct_prompt_injection"]
    assert iterative_legacy == ["crescendo_liar", "autodan"]
    assert len(tap_loop_kwargs) == 1
    assert {
        key: tap_loop_kwargs[0][key]
        for key in (
            "max_rounds",
            "judge_threshold",
            "branch_factor",
            "prune_threshold",
            "keep_top_k",
        )
    } == {
        "max_rounds": 1,
        "judge_threshold": 7.0,
        "branch_factor": 1,
        "prune_threshold": 3.0,
        "keep_top_k": 1,
    }
    assert [entry["kind"] for entry in state.cell_results[0]["conversation"]] == ["generation"]
    assert [entry["kind"] for entry in state.cell_results[2]["conversation"]] == ["round"]
    assert [entry["kind"] for entry in state.cell_results[4]["conversation"]] == ["generation"]
    assert [entry["kind"] for entry in state.cell_results[6]["conversation"]] == ["tree"]


def test_mixed_experiment_response_ownership_is_complete_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RecordingTechniqueRuntimePersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=runtime,
        experiment_attempt_persistence=attempts,
    )
    target = DelayedOperationOwnedRecordedClient(
        role="target",
        barrier=Barrier(2),
    )
    judge = DelayedOperationOwnedRecordedClient(role="target_judge")
    assert target.network_capable is False and judge.network_capable is False
    monkeypatch.setattr(store, "resolve_profile_config", lambda config: dict(config))
    monkeypatch.setattr(store, "_build_chat_client", lambda _config: target)
    monkeypatch.setattr(store, "_build_attacker_client", lambda _config: None)
    monkeypatch.setattr(
        store,
        "_build_judge_client",
        lambda _config, *, target_client, attacker_client: judge,
    )
    dataset = store.create_dataset(
        name="Concurrent operation-owned cohort",
        prompts=[{"text": "operation-owned fixture"}],
    )
    config = {
        "api_key": "recorded-provider-disabled",
        "model": target.model,
        "judge_mode": "custom",
        "judge_api_key": "recorded-provider-disabled",
        "judge_model": judge.model,
    }
    states = []
    with monkeypatch.context() as planning:
        planning.setattr("app.api.store.threading.Thread.start", lambda self: None)
        for suffix in ("alpha", "beta"):
            created = store.create_experiment(
                name=f"Concurrent owner {suffix}",
                dataset_id=dataset["dataset_id"],
                technique_keys=[],
                technique_configs=[{"technique_key": "direct_prompt_injection"}],
                target_config=config,
                mode="live",
            )
            states.append(store._experiments[created["experiment_id"]])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                store._run_experiment,
                state.experiment_id,
                config,
                config["api_key"],
            )
            for state in states
        ]
        for future in futures:
            future.result(timeout=10)

    expected_experiments = {state.experiment_id for state in states}
    assert len(target.calls) == 2
    assert len(judge.calls) == 2
    all_contexts = [context for context, _owner in (*target.calls, *judge.calls)]
    assert {context.correlation.experiment_id for context in all_contexts} == expected_experiments
    assert len({context.model_call_id for context in all_contexts}) == 4
    for context in all_contexts:
        correlation = context.correlation
        assert correlation.cell_id
        assert correlation.run_id
        assert correlation.attempt_id
        assert context.operation_id
        assert context.model_call_id

    persisted_calls = [item["call"] for item in runtime._model_calls.values()]
    assert len(persisted_calls) == 2
    assert {call.correlation.experiment_id for call in persisted_calls} == expected_experiments
    assert (
        len(
            {
                (
                    call.correlation.experiment_id,
                    call.correlation.cell_id,
                    call.correlation.run_id,
                    call.correlation.attempt_id,
                    call.operation_id,
                    call.model_call_id,
                )
                for call in persisted_calls
            }
        )
        == 2
    )
    persisted_events = runtime.persisted_events
    assert persisted_events
    assert len({event.event_id for event in persisted_events}) == len(persisted_events)
    assert {event.correlation.experiment_id for event in persisted_events} == expected_experiments
    persisted_service_events = [
        event
        for event in persisted_events
        if event.event_type.startswith(("model.", "target.", "judge."))
    ]
    assert persisted_service_events
    for event in persisted_service_events:
        correlation = event.correlation
        assert correlation.experiment_id
        assert correlation.cell_id
        assert correlation.run_id
        assert correlation.attempt_id
        assert event.operation_id
        if event.event_type.startswith("model."):
            assert event.payload.get("model_call_id")
        if event.event_type.startswith("target."):
            assert event.payload.get("target_call_id")

    persisted_events_by_call = {
        str(event.payload["model_call_id"]): event
        for event in persisted_service_events
        if event.event_type == "model.completed"
    }
    for item in runtime._model_calls.values():
        call = item["call"]
        event = persisted_events_by_call[call.model_call_id]
        assert event.correlation == call.correlation
        assert event.operation_id == call.operation_id
        assert item["status"] == "completed"
        assert call.correlation.experiment_id in str(item["response"])

    for state in states:
        assert state.status == "completed"
        assert len(state.cell_results) == 1
        cell = state.cell_results[0]
        expected_response = f"owned target response for {state.experiment_id}"
        assert cell["conversation"][0]["candidates"][0]["response"] == expected_response
        assert cell["selected_attempt_id"] == cell["attempt_id"]
        attempt_history = attempts.list_attempts(cell["cell_id"])
        assert len(attempt_history) == 1
        assert attempt_history[0].attempt_id == cell["selected_attempt_id"]
        assert attempt_history[0].execution_status == "completed"
        events = store.normalized_events_by_run[cell["run_id"]]
        assert {
            (
                event.correlation.experiment_id,
                event.correlation.cell_id,
                event.correlation.run_id,
                event.correlation.attempt_id,
            )
            for event in events
        } == {
            (
                state.experiment_id,
                cell["cell_id"],
                cell["run_id"],
                cell["attempt_id"],
            )
        }
        owned_service_events = [
            event for event in events if event.event_type.startswith(("model.", "target.", "judge."))
        ]
        assert owned_service_events
        assert all(event.operation_id for event in owned_service_events)


def test_failed_target_attempts_remain_immutable_before_selected_successful_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = NoopTechniqueRuntimePersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=runtime,
        experiment_attempt_persistence=attempts,
    )
    target = SequencedOperationOwnedRecordedClient(
        role="target",
        outcomes=[
            RuntimeError("recorded target failure one"),
            TimeoutError("recorded target failure two"),
            "recorded target retry success",
        ],
    )
    judge = SequencedOperationOwnedRecordedClient(
        role="target_judge",
        outcomes=['{"score":8,"rationale":"owned retry judge"}'],
    )
    monkeypatch.setattr(store, "resolve_profile_config", lambda config: dict(config))
    monkeypatch.setattr(store, "_build_chat_client", lambda _config: target)
    monkeypatch.setattr(store, "_build_attacker_client", lambda _config: None)
    monkeypatch.setattr(
        store,
        "_build_judge_client",
        lambda _config, *, target_client, attacker_client: judge,
    )
    dataset = store.create_dataset(
        name="Immutable failed target retries",
        prompts=[{"text": "retry-owned fixture"}],
    )
    config = {
        "api_key": "recorded-provider-disabled",
        "model": target.model,
        "judge_mode": "custom",
        "judge_api_key": "recorded-provider-disabled",
        "judge_model": judge.model,
    }
    with monkeypatch.context() as planning:
        planning.setattr("app.api.store.threading.Thread.start", lambda self: None)
        created = store.create_experiment(
            name="Immutable failed target retries",
            dataset_id=dataset["dataset_id"],
            technique_keys=[],
            technique_configs=[{"technique_key": "direct_prompt_injection"}],
            target_config=config,
            mode="live",
        )
    state = store._experiments[created["experiment_id"]]
    store._run_experiment(state.experiment_id, config, config["api_key"])
    cell_id = state.plan.cells[0].cell_id

    for _ in range(2):
        with monkeypatch.context() as queuing:
            queuing.setattr("app.api.store.threading.Thread.start", lambda self: None)
            batch = store.create_retry_batch(
                state.experiment_id,
                action="restart",
                cell_ids=(cell_id,),
                completed_policy="replace_on_valid_completion",
            )
        store._run_retry_batch_once(
            state.experiment_id,
            batch.retry_batch_id,
            {
                "api_key": config["api_key"],
                "judge_api_key": config["judge_api_key"],
            },
        )

    history = attempts.list_attempts(cell_id)
    assert [attempt.attempt_number for attempt in history] == [1, 2, 3]
    assert [attempt.execution_status for attempt in history] == [
        "failed",
        "timed_out",
        "completed",
    ]
    assert [attempt.failure_category for attempt in history] == [
        FailureCategory.PROVIDER,
        FailureCategory.TIMEOUT,
        None,
    ]
    assert history[1].parent_attempt_id == history[0].attempt_id
    assert history[2].parent_attempt_id == history[1].attempt_id
    assert len({attempt.run_id for attempt in history}) == 3
    selected = attempts.get_cell(cell_id).selected_attempt_id
    assert selected == history[2].attempt_id
    assert state.cell_results[0]["selected_attempt_id"] == history[0].attempt_id
    assert state.cell_results[-1]["selected_attempt_id"] == history[2].attempt_id
    assert state.cell_results[-1]["conversation"][0]["candidates"][0]["response"] == "recorded target retry success"
    canonical = store._canonical_experiment_rows(state, state.cell_results)
    assert len(canonical) == 1
    assert canonical[0]["selected_attempt_id"] == history[2].attempt_id
    assert canonical[0]["attempt_count"] == 3
    assert canonical[0]["conversation"] == state.cell_results[-1]["conversation"]
    assert len(target.calls) == 3
    assert len({call.correlation.attempt_id for call in target.calls}) == 3
    assert len({call.model_call_id for call in target.calls}) == 3
    assert [call.correlation.attempt_id for call in target.calls] == [attempt.attempt_id for attempt in history]
    assert all(call.operation_id == "workflow:call_target:visit:1" for call in target.calls)
    assert len(judge.calls) == 1
    assert judge.calls[0].correlation.attempt_id == history[2].attempt_id


def _literal_strings(node: ast.AST) -> set[str]:
    return {value.value for value in ast.walk(node) if isinstance(value, ast.Constant) and isinstance(value.value, str)}


def test_modular_and_application_core_have_no_unlabeled_representative_id_branches() -> None:
    roots = (
        REPO_ROOT / "backend/app/techniques/contracts",
        REPO_ROOT / "backend/app/techniques/core",
        REPO_ROOT / "backend/app/techniques/projection",
        REPO_ROOT / "backend/app/techniques/workflow",
    )
    paths = [
        *(path for root in roots for path in root.rglob("*.py")),
        REPO_ROOT / "backend/app/techniques/application.py",
        REPO_ROOT / "backend/app/persistence/experiment_attempts.py",
        REPO_ROOT / "backend/app/api/conversation.py",
        REPO_ROOT / "backend/app/api/store.py",
    ]
    violations: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scope: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                scope.append(node.name)
                self.generic_visit(node)
                scope.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                scope.append(node.name)
                self.generic_visit(node)
                scope.pop()

            def _allowed(self) -> bool:
                return any("legacy_compatibility" in name.lower() for name in scope)

            def visit_Compare(self, node: ast.Compare) -> None:
                found = sorted(_literal_strings(node) & PROMOTED_TECHNIQUES)
                if found and not self._allowed():
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: compare {found}")
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                found = sorted(_literal_strings(node.value) & PROMOTED_TECHNIQUES)
                labels = {target.id for target in node.targets if isinstance(target, ast.Name)}
                labeled = any("LEGACY_COMPATIBILITY" in label for label in labels)
                if found and not (self._allowed() or labeled):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: assignment {found}")
                self.generic_visit(node)

        Visitor().visit(tree)
    assert violations == []


def _marked_table_values(path: Path, marker: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    assert start in text and end in text, f"{path.name} must contain {marker} markers"
    section = text.split(start, 1)[1].split(end, 1)[0]
    values: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        values.add(line.split("`", 2)[1])
    return values


def test_authoring_guide_field_and_enum_tables_match_runtime_contracts() -> None:
    guide = REPO_ROOT / "docs/technique-authoring.md"
    assert guide.exists(), "Task 29 must add the source-controlled authoring guide"
    properties = MANIFEST_SCHEMA["properties"]
    assert isinstance(properties, dict)
    assert _marked_table_values(guide, "manifest-top-level-fields") == set(properties)
    expected_enums = {
        *(f"ImplementationKind.{item.value}" for item in ImplementationKind),
        *(f"TechniqueCapability.{item.value}" for item in TechniqueCapability),
        *(f"JudgeSchedule.{item.value}" for item in JudgeSchedule),
        *(f"JudgePurpose.{item.value}" for item in JudgePurpose),
        *(f"SeedStrategy.{item.value}" for item in SeedStrategy),
    }
    assert _marked_table_values(guide, "manifest-enum-values") == expected_enums


def test_authoring_required_fields_layout_and_entrypoint_links_match_runtime() -> None:
    guide = REPO_ROOT / "docs/technique-authoring.md"
    assert _marked_table_values(guide, "manifest-required-fields") == set(MANIFEST_SCHEMA["required"])
    assert _marked_table_values(guide, "package-layout-entries") == {
        "technique.yaml",
        "workflow.yaml",
        "executor.py",
        "schemas/parameters.json",
        "presentation.yaml",
        "metrics.yaml",
        "tests/scenarios.yaml",
    }
    for entrypoint in (REPO_ROOT / "backend/README.md", REPO_ROOT / "docs/architecture.md"):
        text = entrypoint.read_text(encoding="utf-8")
        assert "docs/technique-authoring.md" in text


def test_authoring_scenario_contract_matches_executable_fixture_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from devtools.technique_parity import builtin
    from devtools.technique_parity.builtin import (
        _attempt_fixtures,
        _load_scenario,
        load_builtin,
    )

    response_shapes: set[str] = set()
    for technique_id in sorted(PROMOTED_TECHNIQUES):
        _scenario_name, fixture = _load_scenario(
            load_builtin(technique_id),
            None,
        )
        for attempt in _attempt_fixtures(fixture):
            responses = attempt["responses"]
            assert isinstance(responses, dict)
            response_shapes.update(
                "operation mapping" if isinstance(value, dict) else "ordered sequence"
                for value in responses.values()
            )

    assert response_shapes == {"operation mapping", "ordered sequence"}
    guide = (REPO_ROOT / "docs/technique-authoring.md").read_text(
        encoding="utf-8"
    )
    normalized_guide = " ".join(guide.split())
    assert "`tests/scenarios.yaml` has exactly `version: 1`" in normalized_guide
    assert "ordered string sequence" in normalized_guide
    assert (
        "mapping from exact operation ID to string response" in normalized_guide
    )

    version_probe = load_builtin("direct_prompt_injection")
    monkeypatch.setattr(
        builtin.yaml,
        "safe_load",
        lambda _payload: {"version": 2, "scenarios": {"unsupported": {}}},
    )
    with pytest.raises(ValueError, match="requires version 1"):
        _load_scenario(version_probe, None)
