from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.persistence.technique_runtime import (
    ArtifactRecord,
    CheckpointRecord,
    ModelCallRecord,
    NoopTechniqueRuntimePersistence,
    ensure_experiment_attempt_schema,
    ensure_technique_runtime_schema,
)
from app.techniques.contracts.definition import (
    ConverterDefinition,
    DetectorDefinition,
    InputDefinition,
    JudgeDefinition,
    JudgePurpose,
    ModelRoleDefinition,
    SeedDefinition,
    SeedStrategy,
    TechniqueDefinition,
    TechniqueCapability,
    TechniqueIdentity,
    TechniqueImplementation,
    ToolDefinition,
)
from app.techniques.contracts.events import ExecutionEvent, JudgeStatus
from app.techniques.contracts.execution import (
    CorrelationIds,
    ModelRoleBinding,
    ResolvedRunSpec,
    TechniqueRequest,
)
from app.techniques.contracts.definition import ImplementationKind
from app.techniques.core.artifacts import ArtifactService
from app.techniques.core.budgets import BudgetExhausted, BudgetLimits, BudgetService
from app.techniques.core.cancellation import CancellationError, CancellationToken
from app.techniques.core.checkpoints import CheckpointService
from app.techniques.core.converters import ConverterService
from app.techniques.core.context import ExecutionContextFactory
from app.techniques.core.detections import DetectionService
from app.techniques.core.events import EventJournal, TechniqueEventIdentity
from app.techniques.core.histories import HistoryService
from app.techniques.core.inputs import InputPolicyError, InputService
from app.techniques.core.judging import JudgeService
from app.techniques.core.models import ProviderCallResult
from app.techniques.core.preflight import PreflightError, PreflightService
from app.techniques.core.redaction import RedactionService, sanitize_public_mapping
from app.techniques.core.seeds import SeedService
from app.techniques.core.targets import TargetService
from app.techniques.core.templates import TemplatePolicyError, TemplateService
from app.techniques.core.tools import ToolPolicyError, ToolService
from app.techniques.judge_requests import CanonicalJudgeRequestComposer
from app.techniques.legacy.adapter import LegacyTechniqueAdapter


def correlation(cell_id: str = "cell-1") -> CorrelationIds:
    return CorrelationIds(
        experiment_id="exp-1",
        cell_id=cell_id,
        run_id=f"run-{cell_id}",
        attempt_id="attempt-1",
    )


def execution_event(
    *,
    event_id: str = "evt-1",
    sequence: int = 1,
    payload: dict[str, object] | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=event_id,
        correlation=correlation(),
        sequence=sequence,
        event_type="run.started",
        schema_version=1,
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="modular",
        payload={} if payload is None else payload,
    )


def event_journal() -> EventJournal:
    return EventJournal(
        correlation=correlation(),
        persistence=NoopTechniqueRuntimePersistence(),
        technique=TechniqueEventIdentity(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
        ),
    )


def history_service() -> HistoryService:
    return HistoryService(events=event_journal())


def test_noop_runtime_persistence_accepts_idempotent_event() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    event = execution_event()

    persistence.append_event(event)
    persistence.append_event(event)


def test_runtime_persistence_rejects_conflicting_event_identity() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    persistence.append_event(execution_event(payload={"status": "first"}))

    with pytest.raises(ValueError, match="different content"):
        persistence.append_event(execution_event(payload={"status": "changed"}))


def test_runtime_persistence_rejects_duplicate_run_sequence() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    persistence.append_event(execution_event(event_id="evt-1", sequence=1))

    with pytest.raises(ValueError, match="run sequence"):
        persistence.append_event(execution_event(event_id="evt-2", sequence=1))


def test_noop_runtime_event_batch_rejects_a_late_collision_atomically() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    first = execution_event(event_id="evt-existing", sequence=1)
    persistence.append_event(first)
    valid_prefix = execution_event(event_id="evt-prefix", sequence=2)
    colliding_tail = execution_event(event_id="evt-collision", sequence=1)

    with pytest.raises(ValueError, match="run sequence"):
        persistence.append_events((valid_prefix, colliding_tail))

    assert set(persistence._event_hashes) == {first.event_id}
    persistence.append_event(valid_prefix)


def test_noop_runtime_event_batch_accepts_idempotent_same_content_replay() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    events = (
        execution_event(event_id="evt-batch-1", sequence=1),
        execution_event(event_id="evt-batch-2", sequence=2),
    )

    persistence.append_events(events)
    persistence.append_events(events)

    assert set(persistence._event_hashes) == {"evt-batch-1", "evt-batch-2"}


def test_event_journal_merge_many_retries_the_identical_atomic_batch_after_failure() -> None:
    class FailFirstBatchPersistence(NoopTechniqueRuntimePersistence):
        def __init__(self) -> None:
            super().__init__()
            self.fail_batch = False
            self.attempted_batches: list[tuple[ExecutionEvent, ...]] = []

        def append_events(self, events: tuple[ExecutionEvent, ...]) -> None:
            self.attempted_batches.append(events)
            if self.fail_batch:
                self.fail_batch = False
                raise RuntimeError("injected mid-batch failure")
            super().append_events(events)

    persistence = FailFirstBatchPersistence()
    journal = EventJournal(
        correlation=correlation(),
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
    )
    parent = journal.emit("run.started")
    model_call = ModelCallRecord(
        model_call_id="call-before-merge",
        correlation=correlation(),
        role="target",
        operation_id="workflow:branch-a:model:visit:1",
        operation_attempt=1,
        request={},
        request_hash="a" * 64,
    )
    persistence.record_model_call_requested(model_call)
    branch_a = journal.new_buffer(branch_id="branch-a", caused_by=parent.event_id)
    branch_b = journal.new_buffer(branch_id="branch-b", caused_by=parent.event_id)
    with journal.activate(branch_a):
        first = journal.emit("branch.started", payload={"branch_id": "branch-a"})
    with journal.activate(branch_b):
        second = journal.emit("branch.started", payload={"branch_id": "branch-b"})

    persistence.fail_batch = True
    with pytest.raises(RuntimeError, match="mid-batch"):
        journal.merge_many((branch_a, branch_b))

    assert journal.events() == (parent,)
    assert not branch_a.merged
    assert not branch_b.merged
    assert set(persistence._event_hashes) == {parent.event_id}
    assert persistence._model_calls[model_call.model_call_id]["call"] == model_call

    journal.merge_many((branch_a, branch_b))

    merged = journal.events()[1:]
    assert [event.event_id for event in merged] == [first.event_id, second.event_id]
    assert [event.sequence for event in merged] == [2, 3]
    assert persistence.attempted_batches[-2] == persistence.attempted_batches[-1]
    assert branch_a.merged
    assert branch_b.merged
    assert set(persistence._event_hashes) == {
        parent.event_id,
        first.event_id,
        second.event_id,
    }


@pytest.mark.parametrize("mutation", ["owner", "event_id", "sequence"])
def test_event_journal_merge_many_prevalidates_all_buffers_before_persistence(
    mutation: str,
) -> None:
    persistence = NoopTechniqueRuntimePersistence()
    journal = EventJournal(
        correlation=correlation(),
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
    )
    parent = journal.emit("run.started")
    first = journal.new_buffer(branch_id="branch-a")
    second = journal.new_buffer(branch_id="branch-b")
    with journal.activate(first):
        first_event = journal.emit("branch.started", payload={"branch_id": "branch-a"})
    with journal.activate(second):
        second_event = journal.emit("branch.started", payload={"branch_id": "branch-b"})
    if mutation == "owner":
        second.events[0] = replace(second_event, correlation=correlation("other-cell"))
    elif mutation == "event_id":
        second.events[0] = replace(second_event, event_id=first_event.event_id)
    else:
        second.events[0] = replace(second_event, sequence=2)

    with pytest.raises((TypeError, ValueError)):
        journal.merge_many((first, second))

    assert journal.events() == (parent,)
    assert set(persistence._event_hashes) == {parent.event_id}
    assert not first.merged
    assert not second.merged


def test_runtime_records_copy_json_payloads() -> None:
    request = {"messages": [{"role": "user", "content": "hello"}]}
    checkpoint_payload = {"position": {"step": "start"}}
    artifact_metadata = {"kind": "fixture"}

    model_call = ModelCallRecord(
        model_call_id="call-1",
        correlation=correlation(),
        role="target",
        operation_id="target-r1",
        operation_attempt=1,
        request=request,
        request_hash="a" * 64,
    )
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint-1",
        correlation=correlation(),
        spec_hash="b" * 64,
        state_schema_version=1,
        payload=checkpoint_payload,
    )
    artifact = ArtifactRecord(
        artifact_id="artifact-1",
        correlation=correlation(),
        media_type="text/plain",
        content_hash="c" * 64,
        metadata=artifact_metadata,
    )
    request["messages"][0]["content"] = "mutated"
    checkpoint_payload["position"]["step"] = "mutated"
    artifact_metadata["kind"] = "mutated"

    assert model_call.request["messages"][0]["content"] == "hello"
    assert checkpoint.payload["position"]["step"] == "start"
    assert artifact.metadata["kind"] == "fixture"


def test_event_journal_assigns_monotonic_run_local_sequences() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    journal = EventJournal(
        correlation=correlation("cell-a"),
        persistence=persistence,
        technique=TechniqueEventIdentity(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
        ),
    )

    first = journal.emit("run.started")
    second = journal.emit("round.started", payload={"round": 1})

    assert [first.sequence, second.sequence] == [1, 2]
    assert journal.events() == (first, second)


def test_real_and_poisoned_messages_keep_provenance() -> None:
    histories = history_service()
    histories.create("audit_history")
    histories.create("target_visible_history")

    real = histories.append(
        "audit_history",
        role="assistant",
        content="refusal",
        source_kind="target_response",
    )
    poisoned = histories.append(
        "target_visible_history",
        role="assistant",
        content="partial compliance",
        source_kind="transformed_response",
        original_message_id=real.message_id,
    )

    assert poisoned.original_message_id == real.message_id
    assert histories.snapshot("audit_history").messages[0].content == "refusal"


def test_rollback_marks_messages_inactive_without_deleting_them() -> None:
    histories = history_service()
    histories.create("target")
    checkpoint = histories.snapshot("target")
    message = histories.append(
        "target",
        role="user",
        content="round two",
        source_kind="planner",
    )

    histories.rollback("target", checkpoint.snapshot_id)

    assert message.message_id in histories.all_message_ids()
    assert histories.snapshot("target").messages == ()


def test_fork_copies_active_heads_and_keeps_messages_immutable() -> None:
    histories = history_service()
    histories.create("source")
    shared = histories.append(
        "source",
        role="user",
        content="shared",
        source_kind="planner",
    )

    histories.fork("source", "branch")
    histories.append(
        "source",
        role="assistant",
        content="source only",
        source_kind="target_response",
    )

    branch = histories.snapshot("branch")
    assert branch.message_ids == (shared.message_id,)
    assert branch.messages[0] is shared


def test_checkpoint_round_trips_through_persistence() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    service = CheckpointService(
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=event_journal(),
    )
    checkpoint = service.save(
        position={"step": "judge"},
        values={"candidate": "hello"},
        history_heads={"target": ("message-1",)},
        budget_usage={"model_calls": 2},
    )

    resumed = CheckpointService(
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=event_journal(),
    ).load(checkpoint.checkpoint_id)

    assert resumed == checkpoint
    assert resumed.history_heads["target"] == ("message-1",)


def test_runtime_retry_queries_keep_checkpoint_and_indeterminate_attempt_ownership() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    first = CheckpointRecord(
        checkpoint_id="checkpoint-first",
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        payload={"position": "first"},
    )
    second = CheckpointRecord(
        checkpoint_id="checkpoint-second",
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        payload={"position": "second"},
    )
    persistence.save_checkpoint(first)
    persistence.save_checkpoint(second)
    call = ModelCallRecord(
        model_call_id="call-indeterminate",
        correlation=correlation(),
        role="target",
        operation_id="target-1",
        operation_attempt=1,
        request={"messages": []},
        request_hash="hash-a",
    )
    persistence.record_model_call_requested(call)
    persistence.fail_model_call(
        call.model_call_id,
        category="provider",
        error="connection closed after dispatch",
        indeterminate=True,
    )

    assert (
        persistence.latest_checkpoint(
            experiment_id="exp-1",
            cell_id="cell-1",
            attempt_ids=("attempt-1",),
        )
        == second
    )
    assert (
        persistence.latest_checkpoint(
            experiment_id="exp-1",
            cell_id="other-cell",
            attempt_ids=("attempt-1",),
        )
        is None
    )
    assert persistence.attempt_has_indeterminate_call("attempt-1") is True
    assert persistence.attempt_has_indeterminate_call("attempt-other") is False


def test_checkpoint_allows_resume_into_new_attempt_for_same_cell() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    saved = CheckpointService(
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=event_journal(),
    ).save(
        position={"step": "target"},
        values={},
        history_heads={},
        budget_usage={},
    )
    resumed_correlation = CorrelationIds(
        experiment_id="exp-1",
        cell_id="cell-1",
        run_id="run-resume",
        attempt_id="attempt-2",
    )
    resumed = CheckpointService(
        correlation=resumed_correlation,
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=event_journal(),
    ).load(saved.checkpoint_id)

    assert resumed.checkpoint_id == saved.checkpoint_id
    assert resumed.correlation == saved.correlation


def test_backtrack_checkpoint_rejects_a_different_attempt_in_the_same_cell() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    saved = CheckpointService(
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=event_journal(),
    ).save(
        position={"step_id": "checkpoint", "visit_counts": {}},
        values={},
        history_heads={},
        budget_usage={},
    )
    destination = CorrelationIds(
        experiment_id="exp-1",
        cell_id="cell-1",
        run_id="run-resume",
        attempt_id="attempt-2",
    )
    destination_events = event_journal()
    service = CheckpointService(
        correlation=destination,
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=destination_events,
    )

    with pytest.raises(ValueError, match="different run or attempt"):
        service.load_for_backtrack(saved.checkpoint_id)

    assert not any(event.event_type == "checkpoint.loaded" for event in destination_events.events())


@pytest.mark.parametrize(
    ("spec_hash", "state_schema_version", "error"),
    [
        ("b" * 64, 1, "specification hash"),
        ("a" * 64, 2, "state schema version"),
    ],
)
def test_checkpoint_load_rejects_incompatible_runtime(
    spec_hash: str,
    state_schema_version: int,
    error: str,
) -> None:
    persistence = NoopTechniqueRuntimePersistence()
    saved = CheckpointService(
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=event_journal(),
    ).save(
        position={},
        values={},
        history_heads={},
        budget_usage={},
    )
    incompatible = CheckpointService(
        correlation=correlation(),
        spec_hash=spec_hash,
        state_schema_version=state_schema_version,
        persistence=persistence,
        events=event_journal(),
    )

    with pytest.raises(ValueError, match=error):
        incompatible.load(saved.checkpoint_id)


def test_checkpoint_rejects_non_json_state_before_persistence() -> None:
    service = CheckpointService(
        correlation=correlation(),
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=NoopTechniqueRuntimePersistence(),
        events=event_journal(),
    )

    with pytest.raises(TypeError, match="JSON-safe"):
        service.save(
            position={},
            values={"open_handle": object()},
            history_heads={},
            budget_usage={},
        )


class StaticModelService:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.recorded_calls: list[dict[str, object]] = []
        self.recorded_capacity_calls: list[dict[str, object]] = []

    def token_capacity(self, **kwargs: object) -> int:
        self.recorded_capacity_calls.append(dict(kwargs))
        return 37

    async def call(self, **kwargs: object) -> SimpleNamespace:
        self.recorded_calls.append(dict(kwargs))
        return SimpleNamespace(
            model_call_id="call-judge-1",
            text=self.raw,
            usage={"total_tokens": 3},
        )


class RecordingJudgeRequestComposer:
    def __init__(self) -> None:
        self.calls: list[tuple[JudgeDefinition, dict[str, object]]] = []

    def compose(
        self,
        *,
        definition: JudgeDefinition,
        evidence: dict[str, object],
    ) -> SimpleNamespace:
        self.calls.append((definition, dict(evidence)))
        return SimpleNamespace(
            messages=(
                {"role": "system", "content": "canonical judge instructions"},
                {"role": "user", "content": "canonical evidence"},
            ),
            options={
                "temperature": 0.0,
                "response_format": {"type": "json_schema"},
            },
        )


def test_judge_uses_same_composed_request_for_capacity_and_evaluation() -> None:
    models = StaticModelService('{"score": 4, "rationale": "partial"}')
    composer = RecordingJudgeRequestComposer()
    service = JudgeService(
        definitions=(JudgeDefinition(id="target_judge", role="judge"),),
        models=models,
        events=event_journal(),
        parser=json.loads,
        request_composer=composer,
    )
    evidence = {"intent": "test intent", "response": "target output"}

    capacity = service.token_capacity(
        definition_id="target_judge",
        evidence=evidence,
    )
    result = asyncio.run(
        service.evaluate(
            definition_id="target_judge",
            evidence=evidence,
            operation_id="judge-request-contract",
        )
    )

    expected_request = {
        "role": "judge",
        "messages": (
            {"role": "system", "content": "canonical judge instructions"},
            {"role": "user", "content": "canonical evidence"},
        ),
        "temperature": 0.0,
        "response_format": {"type": "json_schema"},
    }
    assert capacity == 37
    assert result.status is JudgeStatus.SUCCEEDED
    assert models.recorded_capacity_calls == [expected_request]
    assert models.recorded_calls == [
        {**expected_request, "operation_id": "judge-request-contract"}
    ]
    assert [(definition.id, item) for definition, item in composer.calls] == [
        ("target_judge", evidence),
        ("target_judge", evidence),
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        {"response": "target output"},
        {"intent": "test intent"},
        {"intent": "test intent", "response": {"metadata": "not text"}},
    ],
    ids=["missing_intent", "missing_response", "response_without_text"],
)
def test_malformed_judge_evidence_fails_before_model_or_lifecycle_call(
    evidence: dict[str, object],
) -> None:
    models = StaticModelService('{"score": 4, "rationale": "unused"}')
    journal = event_journal()
    service = JudgeService(
        definitions=(JudgeDefinition(id="target_judge", role="judge"),),
        models=models,
        events=journal,
        parser=json.loads,
        request_composer=CanonicalJudgeRequestComposer({"judge": object()}),
    )

    with pytest.raises(ValueError, match="judge evidence requires"):
        asyncio.run(
            service.evaluate(
                definition_id="target_judge",
                evidence=evidence,
                operation_id="malformed-judge-evidence",
            )
        )

    assert models.recorded_calls == []
    assert models.recorded_capacity_calls == []
    assert journal.events() == ()


def judge_service(raw: str) -> JudgeService:
    return JudgeService(
        definitions=(
            JudgeDefinition(
                id="target_judge",
                role="judge",
                purposes=frozenset({JudgePurpose.OFFICIAL_SCORE}),
            ),
        ),
        models=StaticModelService(raw),
        events=event_journal(),
        parser=json.loads,
        request_composer=RecordingJudgeRequestComposer(),
    )


@pytest.mark.parametrize("score", [1, 10])
def test_target_judge_accepts_prompt_score_boundaries(score) -> None:
    result = asyncio.run(
        judge_service(json.dumps({"score": score, "rationale": "evaluated"})).evaluate(
            definition_id="target_judge",
            evidence={"intent": "intent", "response": "refusal"},
            operation_id="target-judge-r1",
        )
    )

    assert result.status is JudgeStatus.SUCCEEDED
    assert result.score == score
    assert result.scale_min == 1
    assert result.scale_max == 10


@pytest.mark.parametrize("score", [0, -1, 11, True, False, 1.5, "5", None, float("nan"), float("inf")])
def test_target_judge_rejects_scores_outside_prompt_contract(score) -> None:
    service = judge_service(json.dumps({"score": score, "rationale": "invalid"}))
    result = asyncio.run(service.evaluate(
        definition_id="target_judge", evidence={"intent": "intent", "response": "reply"}, operation_id="invalid",
    ))
    assert result.status is JudgeStatus.FAILED
    assert result.score is None
    assert service._events.events()[-1].event_type == "judge.failed"
    assert service._events.events()[-1].payload["score"] is None


def test_target_judge_rejects_raw_zero_before_parser_can_replace_it() -> None:
    service = judge_service('{"score":0}')
    parser_calls = []

    def rewrite(raw):
        parser_calls.append(raw)
        return {"score": 1}

    service._parser = rewrite
    result = asyncio.run(service.evaluate(
        definition_id="target_judge", evidence={"intent": "intent", "response": "reply"}, operation_id="zero",
    ))
    assert result.status is JudgeStatus.FAILED
    assert result.score is None
    assert parser_calls == []


def test_target_judge_revalidates_score_after_parser() -> None:
    service = judge_service('{"score":5}')
    service._parser = lambda raw: {"score": 0}
    result = asyncio.run(service.evaluate(
        definition_id="target_judge", evidence={"intent": "intent", "response": "reply"}, operation_id="parser",
    ))
    assert result.status is JudgeStatus.FAILED
    assert result.score is None


@pytest.mark.parametrize("score,accepted", [(2, False), (3, True), (7, True), (8, False)])
def test_target_judge_preserves_narrower_declared_scale(score, accepted) -> None:
    service = judge_service(json.dumps({"score": score}))
    service._definitions = {"target_judge": JudgeDefinition(id="target_judge", role="judge", scale_min=3, scale_max=7)}
    result = asyncio.run(service.evaluate(
        definition_id="target_judge", evidence={"intent": "intent", "response": "reply"}, operation_id="narrow",
    ))
    assert (result.status is JudgeStatus.SUCCEEDED) is accepted
    assert result.score == (score if accepted else None)
    assert (result.scale_min, result.scale_max) == (3, 7)


def test_target_judge_incompatible_scale_fails_before_provider_dispatch() -> None:
    service = judge_service('{"score":12}')
    service._definitions = {"target_judge": JudgeDefinition(id="target_judge", role="judge", scale_min=11, scale_max=20)}
    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(service.evaluate(
            definition_id="target_judge", evidence={"intent": "intent", "response": "reply"}, operation_id="invalid-scale",
        ))
    assert service._models.recorded_calls == []
    assert service._events.events() == ()


def test_unparseable_required_target_judge_is_failed_not_zero() -> None:
    result = asyncio.run(
        judge_service("not json").evaluate(
            definition_id="target_judge",
            evidence={"intent": "intent", "response": "answer"},
            operation_id="target-judge-r1",
        )
    )

    assert result.status is JudgeStatus.FAILED
    assert result.score is None


def test_budget_reservation_rejects_call_before_dispatch() -> None:
    budget = BudgetService(BudgetLimits(max_model_calls=1))
    budget.reserve(model_calls=1)

    with pytest.raises(BudgetExhausted, match="model_calls"):
        budget.reserve(model_calls=1)


def test_cancellation_token_preserves_reason() -> None:
    cancellation = CancellationToken()
    cancellation.cancel("operator stopped run")

    with pytest.raises(CancellationError, match="operator stopped run"):
        cancellation.raise_if_cancelled()


class StaticTargetAdapter:
    def execute(self, request: object) -> SimpleNamespace:
        return SimpleNamespace(
            final_output="target response",
            tool_calls=("lookup",),
            metadata={"adapter": "fixture"},
        )


def test_target_service_records_exact_history_snapshot() -> None:
    service = TargetService(
        correlation=correlation(),
        adapter=StaticTargetAdapter(),
        events=event_journal(),
        budget=BudgetService(BudgetLimits(max_target_calls=1)),
        cancellation=CancellationToken(),
    )
    result = asyncio.run(
        service.execute(
            messages=({"role": "user", "content": "prompt"},),
            operation_id="target-r1",
            history_snapshot_id="snapshot-1",
        )
    )

    assert result.history_snapshot_id == "snapshot-1"
    assert result.correlation.cell_id == "cell-1"
    assert result.text == "target response"


def test_target_service_emits_failure_when_adapter_response_is_malformed() -> None:
    journal = event_journal()
    service = TargetService(
        correlation=correlation(),
        adapter=SimpleNamespace(execute=lambda request: SimpleNamespace(final_output=None)),
        events=journal,
        budget=BudgetService(BudgetLimits(max_target_calls=1)),
        cancellation=CancellationToken(),
    )

    with pytest.raises(TypeError, match="string final_output"):
        asyncio.run(
            service.execute(
                messages=({"role": "user", "content": "prompt"},),
                operation_id="target-r1",
                history_snapshot_id="snapshot-1",
            )
        )

    assert [item.event_type for item in journal.events()] == [
        "target.requested",
        "target.failed",
    ]
    assert journal.events()[-1].payload["category"] == "TypeError"


def test_input_service_rejects_undeclared_workspace_path() -> None:
    service = InputService(
        definitions=(
            InputDefinition(
                id="document",
                kind="document",
                media_types=("text/plain",),
            ),
        ),
        owned_artifact_ids=frozenset({"artifact-owned"}),
    )

    with pytest.raises(InputPolicyError, match="artifact reference"):
        service.materialize({"document": {"path": "/tmp/unowned.txt"}})


def test_input_service_validates_immutable_manifest_schema() -> None:
    service = InputService(
        definitions=(
            InputDefinition(
                id="seed_code",
                kind="text",
                schema={"type": "string", "maxLength": 4},
            ),
        ),
    )

    assert service.materialize({"seed_code": "four"}) == {"seed_code": "four"}
    with pytest.raises(InputPolicyError, match="too long"):
        service.materialize({"seed_code": "five!"})


def test_tool_and_converter_calls_require_declared_registry_ids() -> None:
    cancellation = CancellationToken()
    budget = BudgetService(BudgetLimits(max_tool_calls=1))
    tools = ToolService(
        definitions=(ToolDefinition(id="lookup"),),
        registry={"lookup": lambda arguments: {"value": arguments["query"]}},
        events=event_journal(),
        budget=budget,
        cancellation=cancellation,
    )

    with pytest.raises(ToolPolicyError, match="undeclared tool"):
        asyncio.run(tools.invoke("shell", arguments={"command": "id"}))

    converters = ConverterService(
        definitions=(ConverterDefinition(id="pdf_to_text"),),
        registry={"pdf_to_text": lambda artifact_id: {"text": "converted"}},
        events=event_journal(),
        cancellation=cancellation,
    )
    converted = converters.convert("pdf_to_text", artifact_id="artifact-pdf")
    assert converted.source_artifact_id == "artifact-pdf"


def test_seed_and_detector_selection_are_declaration_driven() -> None:
    seeds = SeedService(sql_libraries={"curated": ("seed-a", "seed-b")})
    records = seeds.resolve(SeedDefinition(strategy=SeedStrategy.SQL_LIBRARY, source="curated"))
    assert tuple(record.value for record in records) == ("seed-a", "seed-b")

    detections = DetectionService(
        definitions=(DetectorDefinition(id="refusal"),),
        registry={
            "refusal": lambda inputs: {
                "matched": "cannot" in str(inputs["response"]).lower(),
                "rationale": "fixture",
            }
        },
        events=event_journal(),
    )
    detection = detections.run(
        "refusal",
        inputs={"response": "I cannot help"},
    )
    assert detection.detector_id == "refusal"
    assert detection.matched is True


def test_template_service_rejects_missing_and_extra_fields() -> None:
    templates = TemplateService(assets={"prompt.txt": "Hello {name}"})
    assert templates.render_asset("prompt.txt", fields={"name": "Ada"}) == "Hello Ada"

    with pytest.raises(TemplatePolicyError, match="missing fields"):
        templates.render_asset("prompt.txt", fields={})
    with pytest.raises(TemplatePolicyError, match="extra fields"):
        templates.render_asset(
            "prompt.txt",
            fields={"name": "Ada", "unused": "value"},
        )


def test_preflight_fails_before_external_calls_when_required_role_is_missing() -> None:
    models = StaticModelService("unused")
    preflight = PreflightService(
        model_roles=(ModelRoleDefinition(id="attacker", required=True),),
        declared_detector_ids=frozenset(),
        declared_tool_ids=frozenset(),
        declared_converter_ids=frozenset(),
        available_detector_ids=frozenset(),
        available_tool_ids=frozenset(),
        available_converter_ids=frozenset(),
    )

    with pytest.raises(PreflightError, match="required model role 'attacker'"):
        preflight.validate(model_role_bindings={})
    assert models.recorded_calls == []


def test_artifact_metadata_redacts_secrets() -> None:
    service = ArtifactService(
        correlation=correlation(),
        persistence=NoopTechniqueRuntimePersistence(),
        events=event_journal(),
        redaction=RedactionService(),
        max_size_bytes=10,
        allowed_media_types=frozenset({"application/json"}),
    )
    artifact = service.create(
        media_type="application/json",
        content=b"{}",
        metadata={"Authorization": "Bearer sk-test-secret", "kind": "probe"},
    )

    assert "sk-test-secret" not in repr(artifact.metadata)


def minimal_definition() -> TechniqueDefinition:
    return TechniqueDefinition(
        identity=TechniqueIdentity(
            id="probe",
            version="1.0.0",
            name="Probe",
            family="test",
        ),
        summary="fixture",
        capabilities=frozenset(),
    )


def runtime_request(
    cell_id: str,
    *,
    model_role_bindings: dict[str, ModelRoleBinding] | None = None,
    effective_limits: dict[str, object] | None = None,
    asset_hashes: dict[str, str] | None = None,
) -> TechniqueRequest:
    return TechniqueRequest(
        correlation=CorrelationIds(
            experiment_id="exp-1",
            cell_id=cell_id,
            run_id=f"run-{cell_id}",
            attempt_id="attempt-1",
        ),
        spec=ResolvedRunSpec(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
            implementation_kind=ImplementationKind.PYTHON,
            contract_version="1",
            manifest_hash="a" * 64,
            asset_hashes=asset_hashes or {},
            model_role_bindings=model_role_bindings or {},
            effective_limits=effective_limits or {},
        ),
    )


def runtime_package(
    definition: TechniqueDefinition,
    request: TechniqueRequest,
) -> SimpleNamespace:
    spec = request.spec
    return SimpleNamespace(
        definition=definition,
        implementation=TechniqueImplementation(
            id=spec.implementation_id,
            technique_id=spec.technique_id,
            technique_version=spec.technique_version,
            kind=spec.implementation_kind,
            entrypoint="probe:execute",
            contract_version=spec.contract_version,
        ),
        manifest_hash=spec.manifest_hash,
        asset_hashes=spec.asset_hashes,
    )


def test_context_factory_never_reuses_histories_between_cells() -> None:
    factory = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={},
        target_adapter=StaticTargetAdapter(),
    )
    definition = minimal_definition()
    first_request = runtime_request("cell-a")
    second_request = runtime_request("cell-b")
    first = factory.create(
        request=first_request,
        definition=definition,
        loaded_package=runtime_package(definition, first_request),
    )
    second = factory.create(
        request=second_request,
        definition=definition,
        loaded_package=runtime_package(definition, second_request),
    )

    first.histories.create("target")
    first.histories.append(
        "target",
        role="user",
        content="a",
        source_kind="planner",
    )
    second.histories.create("target")

    assert second.histories.snapshot("target").messages == ()
    assert first.events is not second.events
    assert first.cancellation is not second.cancellation


class ContextModelClient:
    def __init__(self, *, failures: int = 0, total_tokens: int = 1) -> None:
        self.failures = failures
        self.total_tokens = total_tokens
        self.calls = 0
        self.options: list[dict[str, object]] = []

    def chat_correlated_once(self, messages, *, request_context, **options):
        self.calls += 1
        self.options.append(dict(options))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("HTTP 429 transient provider failure")
        return ProviderCallResult(
            text="ok",
            usage={"total_tokens": str(self.total_tokens)},
        )


def test_context_enforces_effective_model_limit_before_dispatch() -> None:
    client = ContextModelClient()
    binding = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="target-profile",
        provider="openai",
        model="target-model",
    )
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="fixture",
        capabilities=frozenset(),
        model_roles=(ModelRoleDefinition(id="target", required=True),),
    )
    request = runtime_request(
        "cell-limited",
        model_role_bindings={"target": binding},
        effective_limits={"max_model_calls": 0},
    )
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target": client},
        target_adapter=StaticTargetAdapter(),
    ).create(
        request=request,
        definition=definition,
        loaded_package=runtime_package(definition, request),
    )

    with pytest.raises(BudgetExhausted, match="model_calls"):
        asyncio.run(
            context.models.call(
                role="target",
                messages=({"role": "user", "content": "hello"},),
                operation_id="target-1",
            )
        )
    assert client.calls == 0


def test_context_rejects_boolean_pinned_budget_limits_without_coercion() -> None:
    definition = minimal_definition()
    request = runtime_request(
        "cell-invalid-limit",
        effective_limits={"max_model_calls": True},
    )

    with pytest.raises(TypeError, match="max_model_calls"):
        ExecutionContextFactory(
            persistence=NoopTechniqueRuntimePersistence(),
            model_clients={},
            target_adapter=StaticTargetAdapter(),
        ).create(
            request=request,
            definition=definition,
            loaded_package=runtime_package(definition, request),
        )


def test_context_uses_pinned_operation_retry_limit() -> None:
    client = ContextModelClient(failures=1)
    binding = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="target-profile",
        provider="openai",
        model="target-model",
    )
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="fixture",
        capabilities=frozenset(),
        model_roles=(ModelRoleDefinition(id="target", required=True),),
    )
    request = runtime_request(
        "cell-retry",
        model_role_bindings={"target": binding},
        effective_limits={
            "max_operation_attempts": 2,
            "max_tokens": 64,
            "retry_indeterminate": True,
        },
    )
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target": client},
        target_adapter=StaticTargetAdapter(),
    ).create(
        request=request,
        definition=definition,
        loaded_package=runtime_package(definition, request),
    )

    result = asyncio.run(
        context.models.call(
            role="target",
            messages=({"role": "user", "content": "hello"},),
            operation_id="target-1",
            max_tokens=8,
        )
    )

    assert result.text == "ok"
    assert result.operation_attempt == 2
    assert client.calls == 2


def test_context_clamps_unbounded_model_output_to_remaining_token_budget() -> None:
    client = ContextModelClient(total_tokens=1)
    binding = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="target-profile",
        provider="openai",
        model="target-model",
    )
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="fixture",
        capabilities=frozenset(),
        model_roles=(ModelRoleDefinition(id="target", required=True),),
    )
    request = runtime_request(
        "cell-token-clamp",
        model_role_bindings={"target": binding},
        effective_limits={"max_tokens": 64},
    )
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target": client},
        target_adapter=StaticTargetAdapter(),
    ).create(
        request=request,
        definition=definition,
        loaded_package=runtime_package(definition, request),
    )

    asyncio.run(
        context.models.call(
            role="target",
            messages=({"role": "user", "content": "hello"},),
            operation_id="target-1",
        )
    )
    asyncio.run(
        context.models.call(
            role="target",
            messages=({"role": "user", "content": "again"},),
            operation_id="target-2",
        )
    )

    assert 0 < client.options[0]["max_tokens"] < 64
    assert 0 < client.options[1]["max_tokens"] < client.options[0]["max_tokens"]
    assert context.budget.usage.tokens == 14


def test_context_rejects_provider_usage_above_reserved_token_ceiling() -> None:
    client = ContextModelClient(total_tokens=100)
    binding = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="target-profile",
        provider="openai",
        model="target-model",
    )
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="fixture",
        capabilities=frozenset(),
        model_roles=(ModelRoleDefinition(id="target", required=True),),
    )
    request = runtime_request(
        "cell-token-overage",
        model_role_bindings={"target": binding},
        effective_limits={"max_tokens": 64},
    )
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target": client},
        target_adapter=StaticTargetAdapter(),
    ).create(
        request=request,
        definition=definition,
        loaded_package=runtime_package(definition, request),
    )

    with pytest.raises(BudgetExhausted, match="tokens"):
        asyncio.run(
            context.models.call(
                role="target",
                messages=({"role": "user", "content": "hello"},),
                operation_id="target-1",
            )
        )

    assert client.calls == 1
    assert context.budget.usage.tokens == 64


def test_context_rejects_capability_not_supported_by_core() -> None:
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="fixture",
        capabilities=frozenset({TechniqueCapability.MEMORY}),
    )
    factory = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={},
        target_adapter=StaticTargetAdapter(),
    )
    request = runtime_request("cell-capability")

    with pytest.raises(PreflightError, match="unsupported capabilities"):
        factory.create(
            request=request,
            definition=definition,
            loaded_package=runtime_package(definition, request),
        )


def test_context_requires_loaded_package_for_modular_implementation() -> None:
    factory = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={},
        target_adapter=StaticTargetAdapter(),
    )

    with pytest.raises(PreflightError, match="loaded package is required"):
        factory.create(
            request=runtime_request("cell-package-required"),
            definition=minimal_definition(),
        )


def test_context_rejects_loaded_package_that_differs_from_pinned_spec() -> None:
    definition = minimal_definition()
    package = SimpleNamespace(
        definition=definition,
        implementation=TechniqueImplementation(
            id="modular",
            technique_id="probe",
            technique_version="1.0.0",
            kind=ImplementationKind.PYTHON,
            entrypoint="probe:execute",
            contract_version="1",
        ),
        manifest_hash="b" * 64,
        asset_hashes={},
    )
    factory = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={},
        target_adapter=StaticTargetAdapter(),
    )

    with pytest.raises(PreflightError, match="manifest hash"):
        factory.create(
            request=runtime_request("cell-package-mismatch"),
            definition=definition,
            loaded_package=package,
        )


def test_context_uses_loaded_package_assets_for_preflight() -> None:
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="fixture",
        capabilities=frozenset(),
        assets=("workflow.yaml",),
    )
    package = SimpleNamespace(
        definition=definition,
        implementation=TechniqueImplementation(
            id="modular",
            technique_id="probe",
            technique_version="1.0.0",
            kind=ImplementationKind.PYTHON,
            entrypoint="probe:execute",
            contract_version="1",
        ),
        manifest_hash="a" * 64,
        asset_hashes={},
    )
    factory = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={},
        target_adapter=StaticTargetAdapter(),
    )

    with pytest.raises(PreflightError, match="workflow.yaml"):
        factory.create(
            request=runtime_request(
                "cell-package-assets",
                asset_hashes={"workflow.yaml": "c" * 64},
            ),
            definition=definition,
            loaded_package=package,
        )


def test_legacy_adapter_emits_normalized_run_and_round_events() -> None:
    journal = event_journal()
    legacy_result = SimpleNamespace(
        technique_key="crescendo_liar",
        score=70,
        response=SimpleNamespace(final_output="answer"),
        traces=(
            SimpleNamespace(
                event_type="iterative_round",
                actor="runner",
                message="Round 1",
                attributes={
                    "round": "1",
                    "target_response": "answer",
                    "judge_score": "7",
                },
            ),
        ),
    )
    adapter = LegacyTechniqueAdapter(callback=lambda request: legacy_result)

    result = asyncio.run(
        adapter.execute(
            runtime_request("cell-a"),
            SimpleNamespace(events=journal),
        )
    )
    events = journal.events()

    assert result.technique_key == "crescendo_liar"
    assert events[0].event_type == "run.started"
    assert any(item.event_type == "round.completed" for item in events)
    assert events[-1].event_type == "run.completed"


def test_event_journal_redacts_exact_secrets_and_validates_judges() -> None:
    journal = EventJournal(
        correlation=correlation(),
        persistence=NoopTechniqueRuntimePersistence(),
        technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
        redaction=RedactionService(secret_values=("opaque-credential",)),
    )

    emitted = journal.emit(
        "legacy.trace",
        payload={"content": "contains opaque-credential"},
    )
    assert "opaque-credential" not in repr(emitted.payload)

    with pytest.raises(ValueError, match="numeric score"):
        journal.emit(
            "judge.completed",
            payload={"definition_id": "target_judge", "score": None},
        )
    with pytest.raises(ValueError, match="string response"):
        journal.emit("target.completed", payload={"response": None})
    with pytest.raises(ValueError, match="artifact_id"):
        journal.emit(
            "artifact.created",
            payload={"media_type": "text/plain", "content_hash": "a" * 64},
        )


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key",
        "Access-Token",
        "accessToken",
        "AccessToken",
        "refreshToken",
        "RefreshToken",
        "sessionToken",
        "SessionToken",
        "clientSecret",
        "ClientSecret",
        "privateKey",
        "PrivateKey",
        "secretKey",
        "SecretKey",
        "awsSecretAccessKey",
        "aws_secret_access_key",
        "servicePasswordHint",
        "reviewerPassphraseFingerprint",
        "delegatedCredentialLabel",
        "delegatedCredentialsLabel",
        "Session.Token",
        "ID-TOKEN",
        "reviewer_private_key",
        "reviewer.passphrase",
        "Reviewer Client Secret",
        "reviewer_credential",
        "reviewer_credentials",
        "reviewer_password",
        "Authorization",
        "Cookie",
    ],
)
def test_public_and_runtime_redaction_share_nested_secret_predicate(
    secret_key: str,
) -> None:
    sentinel = "NESTED-SENTINEL-CREDENTIAL"
    payload = {
        "sequence": [
            {
                "reviewer": {
                    secret_key: sentinel,
                    "public_key": "publishable",
                    "publicKey": "publishable-camel",
                    "key": "ordinary",
                    "secretary": "ordinary-secretary",
                    "passwordless": "ordinary-passwordless",
                    "credentialed": "ordinary-credentialed",
                }
            }
        ]
    }
    original = json.loads(json.dumps(payload))

    public = sanitize_public_mapping(payload)
    runtime = RedactionService(secret_keys=frozenset()).redact_mapping(payload)
    public_reviewer = public["sequence"][0]["reviewer"]
    runtime_reviewer = runtime["sequence"][0]["reviewer"]

    assert secret_key not in public_reviewer
    assert runtime_reviewer[secret_key] == "[REDACTED]"
    assert public_reviewer["public_key"] == runtime_reviewer["public_key"] == "publishable"
    assert public_reviewer["publicKey"] == runtime_reviewer["publicKey"] == "publishable-camel"
    assert public_reviewer["key"] == runtime_reviewer["key"] == "ordinary"
    assert public_reviewer["secretary"] == runtime_reviewer["secretary"] == "ordinary-secretary"
    assert public_reviewer["passwordless"] == runtime_reviewer["passwordless"] == "ordinary-passwordless"
    assert public_reviewer["credentialed"] == runtime_reviewer["credentialed"] == "ordinary-credentialed"
    assert payload == original


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Bearer sk-supersecret123", "[REDACTED]"),
        ("prefix rk-rotatecredential suffix", "prefix [REDACTED] suffix"),
        ("skill-building", "skill-building"),
        ("sketch-123456", "sketch-123456"),
        ("Bearer tokenless123", "Bearer tokenless123"),
    ],
)
def test_public_and_runtime_redaction_share_credential_value_predicate(
    value: str,
    expected: str,
) -> None:
    payload = {"nested": [{"note": value, "publicKey": "publishable"}]}
    original = json.loads(json.dumps(payload))

    public = sanitize_public_mapping(payload)
    runtime = RedactionService(secret_keys=frozenset()).redact_mapping(payload)

    assert public["nested"][0]["note"] == expected
    assert runtime["nested"][0]["note"] == expected
    assert public["nested"][0]["publicKey"] == "publishable"
    assert runtime["nested"][0]["publicKey"] == "publishable"
    assert payload == original


def test_structural_leaf_exemptions_do_not_hide_nested_evidence_credentials() -> None:
    payload = {
        "evidence": {
            "result": {
                "note": "Bearer sk-supersecret123",
                "result_id": "sk-ordinary-identity123",
            }
        }
    }
    original = json.loads(json.dumps(payload))

    public = sanitize_public_mapping(payload)
    runtime = RedactionService(secret_keys=frozenset()).redact_mapping(payload)

    assert public["evidence"]["result"]["note"] == "[REDACTED]"
    assert runtime["evidence"]["result"]["note"] == "[REDACTED]"
    assert public["evidence"]["result"]["result_id"] == "sk-ordinary-identity123"
    assert runtime["evidence"]["result"]["result_id"] == "sk-ordinary-identity123"
    assert payload == original


def test_structural_sequence_exemption_is_limited_to_immediate_identity_values() -> None:
    payload = {
        "auxiliary_judges": [
            "sk-ordinary-identity123",
            ["Bearer sk-supersecret123"],
        ]
    }
    original = json.loads(json.dumps(payload))

    public = sanitize_public_mapping(payload)
    runtime = RedactionService(secret_keys=frozenset()).redact_mapping(payload)

    assert public["auxiliary_judges"][0] == "sk-ordinary-identity123"
    assert runtime["auxiliary_judges"][0] == "sk-ordinary-identity123"
    assert public["auxiliary_judges"][1][0] == "[REDACTED]"
    assert runtime["auxiliary_judges"][1][0] == "[REDACTED]"
    assert payload == original


@pytest.mark.parametrize(
    "identity_key",
    ["id", "technique_id", "technique_key", "model", "provider", "transport"],
)
def test_credential_shape_heuristic_preserves_structural_identity_values(
    identity_key: str,
) -> None:
    payload = {identity_key: "sk-ordinary-identity123"}

    public = sanitize_public_mapping(payload)
    runtime = RedactionService(secret_keys=frozenset()).redact_mapping(payload)

    assert public[identity_key] == "sk-ordinary-identity123"
    assert runtime[identity_key] == "sk-ordinary-identity123"


@pytest.mark.parametrize("ordinary_key", ["key", "publicKey", "public_key"])
def test_credential_value_redaction_is_independent_of_public_key_names(
    ordinary_key: str,
) -> None:
    payload = {ordinary_key: "Bearer sk-supersecret123"}

    public = sanitize_public_mapping(payload)
    runtime = RedactionService(secret_keys=frozenset()).redact_mapping(payload)

    assert public[ordinary_key] == "[REDACTED]"
    assert runtime[ordinary_key] == "[REDACTED]"


def test_standalone_checkpoint_cannot_cross_runs() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    source = CorrelationIds(run_id="run-a", attempt_id="attempt-a")
    source_events = EventJournal(
        correlation=source,
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
    )
    saved = CheckpointService(
        correlation=source,
        spec_hash="a" * 64,
        state_schema_version=1,
        persistence=persistence,
        events=source_events,
    ).save(position={}, values={}, history_heads={}, budget_usage={})
    destination = CorrelationIds(run_id="run-b", attempt_id="attempt-b")
    destination_events = EventJournal(
        correlation=destination,
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
    )

    with pytest.raises(ValueError, match="different run"):
        CheckpointService(
            correlation=destination,
            spec_hash="a" * 64,
            state_schema_version=1,
            persistence=persistence,
            events=destination_events,
        ).load(saved.checkpoint_id)


def test_noop_persistence_rejects_conflicting_terminal_model_evidence() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    call = ModelCallRecord(
        model_call_id="call-terminal",
        correlation=correlation(),
        role="target",
        operation_id="target-1",
        operation_attempt=1,
        request={},
        request_hash="a" * 64,
    )
    persistence.record_model_call_requested(call)
    persistence.complete_model_call(
        call.model_call_id,
        response="first",
        usage={"total_tokens": 1},
        provider_response_id="provider-1",
        gateway_call_id=None,
        gateway_attempts={},
    )
    persistence.complete_model_call(
        call.model_call_id,
        response="first",
        usage={"total_tokens": 1},
        provider_response_id="provider-1",
        gateway_call_id=None,
        gateway_attempts={},
    )

    with pytest.raises(ValueError, match="different evidence"):
        persistence.fail_model_call(
            call.model_call_id,
            category="provider",
            error="late failure",
            indeterminate=False,
        )


def test_noop_persistence_rejects_duplicate_model_operation_attempt() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    first = ModelCallRecord(
        model_call_id="call-first",
        correlation=correlation(),
        role="target",
        operation_id="target-1",
        operation_attempt=1,
        request={},
        request_hash="a" * 64,
    )
    duplicate_operation = ModelCallRecord(
        model_call_id="call-second",
        correlation=correlation(),
        role="target",
        operation_id="target-1",
        operation_attempt=1,
        request={},
        request_hash="b" * 64,
    )
    persistence.record_model_call_requested(first)

    with pytest.raises(ValueError, match="operation attempt"):
        persistence.record_model_call_requested(duplicate_operation)


def test_input_uses_authoritative_artifact_metadata() -> None:
    artifacts = ArtifactService(
        correlation=correlation(),
        persistence=NoopTechniqueRuntimePersistence(),
        events=event_journal(),
        redaction=RedactionService(),
        max_size_bytes=100,
        allowed_media_types=frozenset({"text/plain"}),
    )
    artifact = artifacts.create(media_type="text/plain", content=b"four")
    inputs = InputService(
        definitions=(
            InputDefinition(
                id="document",
                kind="artifact",
                max_size_bytes=2,
                media_types=("text/plain",),
            ),
        ),
        artifact_exists=artifacts.contains,
        artifact_resolver=artifacts.get,
    )

    with pytest.raises(InputPolicyError, match="size limit"):
        inputs.materialize(
            {
                "document": {
                    "artifact_id": artifact.artifact_id,
                    "size_bytes": 1,
                    "media_type": "text/plain",
                }
            }
        )


class FailingModelService:
    async def call(self, **kwargs: object) -> object:
        raise RuntimeError("provider unavailable")


def test_judge_model_failure_emits_failed_result_without_zero() -> None:
    journal = event_journal()
    service = JudgeService(
        definitions=(JudgeDefinition(id="target_judge", role="judge"),),
        models=FailingModelService(),
        events=journal,
        parser=json.loads,
        request_composer=RecordingJudgeRequestComposer(),
    )

    result = asyncio.run(
        service.evaluate(
            definition_id="target_judge",
            evidence={"response": "answer"},
            operation_id="judge-1",
        )
    )

    assert result.status is JudgeStatus.FAILED
    assert result.score is None
    assert journal.events()[-1].event_type == "judge.failed"


class InvalidResultModelService:
    async def call(self, **kwargs: object) -> object:
        return SimpleNamespace(text=None, model_call_id="call-invalid", usage={})


def test_judge_invalid_model_result_emits_failed_result_without_zero() -> None:
    journal = event_journal()
    service = JudgeService(
        definitions=(JudgeDefinition(id="target_judge", role="judge"),),
        models=InvalidResultModelService(),
        events=journal,
        parser=json.loads,
        request_composer=RecordingJudgeRequestComposer(),
    )

    result = asyncio.run(
        service.evaluate(
            definition_id="target_judge",
            evidence={"response": "answer"},
            operation_id="judge-invalid",
        )
    )

    assert result.status is JudgeStatus.FAILED
    assert result.score is None
    assert journal.events()[-1].event_type == "judge.failed"


def test_runtime_schema_ensure_is_idempotent_for_existing_volumes() -> None:
    class RecordingConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str) -> None:
            self.statements.append(statement)

    connection = RecordingConnection()
    ensure_technique_runtime_schema(connection)
    ensure_technique_runtime_schema(connection)

    sql = "\n".join(connection.statements)
    assert "CREATE TABLE IF NOT EXISTS execution_events" in sql
    assert "CREATE TABLE IF NOT EXISTS model_calls" in sql


def test_experiment_schema_repair_adds_durable_execution_generations() -> None:
    class RecordingConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str) -> None:
            self.statements.append(statement)

    connection = RecordingConnection()
    ensure_experiment_attempt_schema(connection)
    ensure_experiment_attempt_schema(connection)

    sql = "\n".join(connection.statements)
    assert (
        "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS execution_generation "
        "BIGINT NOT NULL DEFAULT 0"
    ) in sql
    assert (
        "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS execution_owner_id "
        "TEXT NOT NULL DEFAULT ''"
    ) in sql
    assert (
        "ALTER TABLE retry_batches ADD COLUMN IF NOT EXISTS execution_generation BIGINT"
    ) in sql
    assert (
        "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS "
        "execution_generation BIGINT NOT NULL DEFAULT 0"
    ) in sql
    assert "experiments_execution_generation_check" in sql
    assert "retry_batches_execution_generation_check" in sql
    assert "experiment_cell_attempts_execution_generation_check" in sql
