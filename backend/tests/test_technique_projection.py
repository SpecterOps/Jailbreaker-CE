from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.techniques.contracts.definition import (
    ResultPolicy,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
)
from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.execution import CorrelationIds, ResultDisposition
from app.techniques.core.events import TechniqueEventIdentity
from app.techniques.projection.conversation import (
    ApplicationConversationProjector,
    ConversationProjector,
)
from app.techniques.projection.metrics import MetricProjector
from app.techniques.projection.outcome import OutcomeProjector
from app.techniques.plugins.manifest import load_technique_package


CORRELATION = CorrelationIds(
    experiment_id="exp-1",
    cell_id="cell-1",
    run_id="run-1",
    attempt_id="attempt-1",
)
TECHNIQUE = TechniqueEventIdentity("future-technique", "1.0.0", "modular")


def event(
    event_id: str,
    sequence: int,
    event_type: str,
    *,
    round_number: int | None = None,
    payload: dict[str, object] | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=event_id,
        correlation=CORRELATION,
        sequence=sequence,
        event_type=event_type,
        schema_version=1,
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        technique_id=TECHNIQUE.technique_id,
        technique_version=TECHNIQUE.technique_version,
        implementation_id=TECHNIQUE.implementation_id,
        round=round_number,
        payload=payload or {},
    )


def round_events() -> tuple[ExecutionEvent, ...]:
    return (
        event("evt-1", 1, "round.started", round_number=1),
        event(
            "evt-2",
            2,
            "target.completed",
            round_number=1,
            payload={"response": "answer", "target_call_id": "target-1"},
        ),
        event(
            "evt-3",
            3,
            "judge.completed",
            round_number=1,
            payload={
                "definition_id": "target_judge",
                "status": "succeeded",
                "score": 7,
            },
        ),
    )


def test_round_projection_does_not_require_technique_id() -> None:
    view = ConversationProjector().project(
        round_events(),
        presentation={"format": "rounds"},
    )

    assert view["format"] == "rounds"
    assert view["sections"][0]["round_number"] == 1
    assert view["sections"][0]["judges"]["target_judge"]["score"] == 7


def test_projection_reads_manifest_conversation_format_key() -> None:
    view = ConversationProjector().project(
        round_events(),
        presentation={"conversation_format": "rounds"},
    )

    assert view["format"] == "rounds"
    assert view["sections"][0]["round_number"] == 1


def test_round_application_projection_attaches_run_end_judge_only_to_its_target() -> None:
    events = (
        event(
            "target-1",
            1,
            "target.completed",
            round_number=1,
            payload={"response": "setup response"},
        ),
        event(
            "target-2",
            2,
            "target.completed",
            round_number=2,
            payload={"response": "final response"},
        ),
        event(
            "judge-final",
            3,
            "judge.completed",
            payload={
                "definition_id": "target_judge",
                "status": "succeeded",
                "score": 4,
                "rationale": "final response checked",
                "target_response_event_id": "target-2",
            },
        ),
    )
    projection = ConversationProjector().project(
        events,
        presentation={"conversation_format": "rounds"},
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=events,
        history_snapshots={},
        final_target_event=events[1],
        official_judge_event=events[2],
        attack_succeeded=False,
    )

    assert conversation[0]["judge"] == {
        "status": "not_scheduled",
        "input": "",
        "output": "",
        "score": None,
        "rationale": "",
        "error": "",
    }
    assert conversation[1]["judge"] == {
        "status": "succeeded",
        "input": "",
        "output": "",
        "score": 4,
        "rationale": "final response checked",
        "error": "",
    }


@pytest.mark.parametrize("target_response_event_id", [None, "another-target"])
def test_round_application_fallback_does_not_attach_unowned_run_end_judge(
    target_response_event_id: str | None,
) -> None:
    target = event(
        "target-final",
        1,
        "target.completed",
        payload={"response": "final response"},
    )
    judge_payload: dict[str, object] = {
        "definition_id": "target_judge",
        "status": "succeeded",
        "score": 4,
        "rationale": "must not be attributed",
    }
    if target_response_event_id is not None:
        judge_payload["target_response_event_id"] = target_response_event_id
    judge = event("judge-final", 2, "judge.completed", payload=judge_payload)
    events = (target, judge)
    projection = ConversationProjector().project(
        events,
        presentation={"conversation_format": "rounds"},
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=events,
        history_snapshots={},
        final_target_event=target,
        official_judge_event=judge,
        attack_succeeded=False,
    )

    assert conversation[0]["judge"] == {
        "status": "not_scheduled",
        "input": "",
        "output": "",
        "score": None,
        "rationale": "",
        "error": "",
    }


def test_round_application_fallback_preserves_owned_failed_run_end_judge() -> None:
    target = event(
        "target-final",
        1,
        "target.completed",
        payload={"response": "final response"},
    )
    judge = event(
        "judge-final",
        2,
        "judge.failed",
        payload={
            "definition_id": "target_judge",
            "status": "failed",
            "score": None,
            "parse_status": "structured output rejected",
            "target_response_event_id": target.event_id,
        },
    )
    events = (target, judge)
    projection = ConversationProjector().project(
        events,
        presentation={"conversation_format": "rounds"},
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=events,
        history_snapshots={},
        final_target_event=target,
        official_judge_event=judge,
        attack_succeeded=False,
    )

    assert conversation[0]["judge"] == {
        "status": "failed",
        "input": "",
        "output": "",
        "score": None,
        "rationale": "",
        "error": "structured output rejected",
    }


def test_rounds_projection_appends_declared_final_artifact_with_official_judge() -> None:
    canonical_json = '{"nodes":[{"id":"root","type":"agent"}]}'
    events = (
        event(
            "target-1",
            1,
            "target.completed",
            round_number=1,
            payload={"response": "first round"},
        ),
        event(
            "target-2",
            2,
            "target.completed",
            round_number=2,
            payload={"response": "second round"},
        ),
        event(
            "architecture-map",
            3,
            "value.transformed",
            payload={"response": canonical_json},
        ),
        event(
            "judge-final",
            4,
            "judge.completed",
            payload={
                "definition_id": "target_judge",
                "status": "succeeded",
                "score": 0,
                "rationale": "No architecture was recovered.",
                "target_response_event_id": "architecture-map",
            },
        ),
    )
    presentation = {
        "conversation_format": "rounds",
        "final_artifact": {
            "source": "final_response",
            "label": "Extracted Architecture Map",
            "media_type": "application/json",
            "include_official_judge": True,
        },
    }
    projection = ConversationProjector().project(events, presentation=presentation)

    conversation = ApplicationConversationProjector().project(
        projection,
        events=events,
        history_snapshots={},
        final_target_event=events[2],
        official_judge_event=events[3],
        attack_succeeded=False,
    )

    assert [entry["kind"] for entry in conversation] == ["round", "round", "artifact"]
    assert conversation[-1] == {
        "kind": "artifact",
        "label": "Extracted Architecture Map",
        "media_type": "application/json",
        "response": canonical_json,
        "judge": {
            "status": "succeeded",
            "input": "",
            "output": "",
            "score": 0,
            "rationale": "No architecture was recovered.",
            "error": "",
        },
    }


@pytest.mark.parametrize("media_type", ["text/plain", None, "", "   ", 7, False, [], {}])
def test_final_artifact_rejects_explicit_mismatching_or_invalid_media_type(media_type) -> None:
    selected = event(
        "final-response", 1, "value.transformed",
        payload={"response": '{"answer":"unchanged"}', "media_type": media_type},
    )
    projection = ConversationProjector().project(
        (selected,),
        presentation={
            "conversation_format": "rounds",
            "final_artifact": {"source": "final_response", "media_type": "application/json"},
        },
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=(selected,),
        history_snapshots={},
        final_target_event=selected,
        official_judge_event=None,
        attack_succeeded=False,
    )

    assert [entry["kind"] for entry in conversation] == ["round"]


@pytest.mark.parametrize("metadata", [{}, {"media_type": "application/json"}], ids=["absent", "matching"])
def test_final_artifact_preserves_exact_response_with_compatible_media_type(metadata) -> None:
    canonical_json = '{ "answer": "caf\u00e9", "items": [2, 1] }\n'
    selected = event(
        "final-response", 1, "value.transformed",
        payload={"response": canonical_json, **metadata},
    )
    projection = ConversationProjector().project(
        (selected,),
        presentation={
            "conversation_format": "rounds",
            "final_artifact": {"source": "final_response", "media_type": "application/json"},
        },
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=(selected,),
        history_snapshots={},
        final_target_event=selected,
        official_judge_event=None,
        attack_succeeded=False,
    )

    assert [entry["kind"] for entry in conversation] == ["round", "artifact"]
    assert conversation[-1] == {
        "kind": "artifact", "label": "Final artifact",
        "media_type": "application/json", "response": canonical_json,
    }


def test_rounds_without_final_artifact_preserve_existing_conversation() -> None:
    events = round_events()
    projection = ConversationProjector().project(
        events,
        presentation={"conversation_format": "rounds"},
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=events,
        history_snapshots={},
        final_target_event=events[1],
        official_judge_event=events[2],
        attack_succeeded=False,
    )

    assert conversation == [
        {
            "kind": "round",
            "round_number": "1",
            "is_backtrack": False,
            "attacker": {"input": "", "output": ""},
            "target": {
                "input": "",
                "prompt": "",
                "response": "answer",
                "poisoned": "",
            },
            "judge": {
                "status": "succeeded",
                "input": "",
                "output": "",
                "score": 7,
                "rationale": "",
                "error": "",
            },
        }
    ]


def test_final_artifact_omits_official_judge_when_disabled() -> None:
    target = event(
        "target-final",
        1,
        "target.completed",
        round_number=1,
        payload={"response": "artifact body"},
    )
    judge = event(
        "judge-final",
        2,
        "judge.completed",
        payload={
            "definition_id": "target_judge",
            "score": 0,
            "target_response_event_id": target.event_id,
        },
    )
    events = (target, judge)
    projection = ConversationProjector().project(
        events,
        presentation={
            "conversation_format": "rounds",
            "final_artifact": {
                "source": "final_response",
                "include_official_judge": False,
            },
        },
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=events,
        history_snapshots={},
        final_target_event=target,
        official_judge_event=judge,
        attack_succeeded=False,
    )

    assert conversation[-1] == {
        "kind": "artifact",
        "label": "Final artifact",
        "media_type": "text/plain",
        "response": "artifact body",
    }


def test_final_artifact_is_not_fabricated_without_selected_response() -> None:
    events = round_events()
    projection = ConversationProjector().project(
        events,
        presentation={
            "conversation_format": "rounds",
            "final_artifact": {"source": "final_response"},
        },
    )

    conversation = ApplicationConversationProjector().project(
        projection,
        events=events,
        history_snapshots={},
        final_target_event=None,
        official_judge_event=None,
        attack_succeeded=False,
    )

    assert [entry["kind"] for entry in conversation] == ["round"]


def test_projection_keeps_failed_judge_score_null() -> None:
    events = (
        event("evt-1", 1, "target.completed", payload={"response": "answer"}),
        event(
            "evt-2",
            2,
            "judge.failed",
            payload={"definition_id": "target_judge", "parse_status": "failed"},
        ),
    )
    outcome = OutcomeProjector().project(
        events,
        result_policy=ResultPolicy(official_judge="target_judge"),
    )

    assert outcome.result_disposition is ResultDisposition.INCONCLUSIVE
    assert outcome.official_score is None


def test_projection_uses_latest_official_judge_for_selected_final_target() -> None:
    events = (
        event(
            "target-1",
            1,
            "target.completed",
            payload={"response": "earlier", "response_kind": "real"},
        ),
        event(
            "judge-1",
            2,
            "judge.completed",
            payload={
                "definition_id": "target_judge",
                "status": "succeeded",
                "score": 6,
                "target_response_event_id": "target-1",
            },
        ),
        event(
            "target-2",
            3,
            "target.completed",
            payload={"response": "final", "response_kind": "real"},
        ),
        event(
            "judge-2",
            4,
            "judge.failed",
            payload={
                "definition_id": "target_judge",
                "score": None,
                "parse_status": "failed",
                "target_response_event_id": "target-2",
            },
        ),
    )

    outcome = OutcomeProjector().project(
        events,
        result_policy=ResultPolicy(
            official_judge="target_judge",
            final_response="target.real_response",
            success_threshold=5,
        ),
    )

    assert outcome.final_response == "final"
    assert outcome.final_response_event_id == "target-2"
    assert outcome.official_judge_event_id == "judge-2"
    assert outcome.official_score is None
    assert outcome.result_disposition is ResultDisposition.INCONCLUSIVE


def test_metric_projection_uses_selected_attempt_once() -> None:
    metrics = MetricProjector().project_core(
        cells=(
            {
                "selected_attempt_id": "attempt-3",
                "attempts": (
                    {"attempt_id": "attempt-1", "status": "failed"},
                    {"attempt_id": "attempt-2", "status": "failed"},
                    {
                        "attempt_id": "attempt-3",
                        "status": "completed",
                        "result_disposition": "tested",
                        "score": 80,
                    },
                ),
            },
        )
    )

    assert metrics["tested_cells"] == 1
    assert metrics["average_score"] == 80
    assert metrics["total_attempts"] == 3


def test_metric_projection_enforces_canonical_status_and_disposition() -> None:
    metrics = MetricProjector().project_core(
        cells=(
            {
                "selected_attempt_id": "attempt-zero",
                "attempts": (
                    {
                        "attempt_id": "attempt-failed",
                        "execution_status": "failed",
                        "status": "completed",
                        "result_disposition": "inconclusive",
                        "failure_category": "provider",
                        "cost": 1.25,
                    },
                    {
                        "attempt_id": "attempt-zero",
                        "execution_status": "completed",
                        "status": "failed",
                        "result_disposition": "tested",
                        "score": 0,
                        "cost": 0.75,
                    },
                    {
                        "attempt_id": "attempt-inconclusive",
                        "execution_status": "completed",
                        "result_disposition": "inconclusive",
                        "cost": 0.5,
                    },
                ),
            },
        )
    )

    assert metrics == {
        "total_cells": 1,
        "tested_cells": 1,
        "inconclusive_cells": 0,
        "average_score": 0,
        "total_attempts": 3,
        "completed_attempts": 2,
        "failed_attempts": 1,
        "failure_category_counts": {"provider": 1},
        "total_attempt_cost": 2.5,
        "selected_attempt_cost": 0.75,
        "failed_attempt_cost": 1.25,
    }


def test_metric_projection_excludes_inconclusive_and_boolean_scores() -> None:
    metrics = MetricProjector().project_core(
        cells=(
            {
                "selected_attempt_id": "attempt-inconclusive",
                "attempts": (
                    {
                        "attempt_id": "attempt-inconclusive",
                        "execution_status": "completed",
                        "result_disposition": "inconclusive",
                        "score": None,
                    },
                ),
            },
            {
                "selected_attempt_id": "attempt-boolean",
                "attempts": (
                    {
                        "attempt_id": "attempt-boolean",
                        "execution_status": "completed",
                        "result_disposition": "tested",
                        "score": False,
                    },
                ),
            },
        )
    )

    assert metrics["tested_cells"] == 0
    assert metrics["inconclusive_cells"] == 1
    assert metrics["average_score"] is None
    assert metrics["total_attempt_cost"] is None


def test_metric_projection_excludes_non_finite_numeric_evidence() -> None:
    core = MetricProjector().project_core(
        cells=(
            {
                "selected_attempt_id": "attempt-nan",
                "attempts": (
                    {
                        "attempt_id": "attempt-nan",
                        "execution_status": "completed",
                        "result_disposition": "tested",
                        "score": float("nan"),
                        "cost": float("inf"),
                    },
                ),
            },
        )
    )
    declared = MetricProjector().project_declared(
        records=(
            {
                "score": float("nan"),
                "overflow_sum": 1e308,
                "ratio": 1e308,
            },
            {
                "score": float("inf"),
                "overflow_sum": 1e308,
                "ratio": 1e-308,
            },
            {"score": 4},
        ),
        definitions=(
            {
                "id": "finite_average",
                "name": "Finite average",
                "source": "score",
                "aggregation": "average",
            },
            {
                "id": "overflow_sum",
                "name": "Overflow sum",
                "source": "overflow_sum",
                "aggregation": "sum",
            },
            {
                "id": "overflow_ratio",
                "name": "Overflow ratio",
                "source": "ratio",
                "aggregation": "ratio",
            },
        ),
    )

    assert core["tested_cells"] == 0
    assert core["average_score"] is None
    assert core["total_attempt_cost"] is None
    assert declared == {
        "finite_average": 4,
        "overflow_sum": None,
        "overflow_ratio": None,
    }
    json.dumps({"core": core, "declared": declared}, allow_nan=False)


def test_declared_metrics_accept_technique_definition_manifest_mappings() -> None:
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="fixture",
        capabilities=frozenset(),
        metrics=(
            {
                "id": "average_score",
                "name": "Average score",
                "source": "score",
                "aggregation": "average",
            },
        ),
    )

    metrics = MetricProjector().project_declared(
        records=({"score": 4}, {"score": 8}),
        definitions=definition.metrics,
    )

    assert metrics == {"average_score": 6}


def test_declared_metric_group_by_is_nested_safe_and_deterministic() -> None:
    metrics = MetricProjector().project_declared(
        records=(
            {"config": {"technique_config_id": "arm-b"}, "spec_hash": "hash-b", "score": 8},
            {"config": {"technique_config_id": "arm-a"}, "spec_hash": "hash-a"},
            {"config": {"technique_config_id": "arm-b"}, "spec_hash": "hash-b", "score": 4},
        ),
        definitions=(
            {
                "id": "score_by_arm",
                "name": "Score by arm",
                "source": "score",
                "aggregation": "average",
                "group_by": ["config.technique_config_id", "spec_hash"],
            },
        ),
    )

    assert metrics == {
        "score_by_arm": [
            {
                "group": {
                    "config.technique_config_id": "arm-a",
                    "spec_hash": "hash-a",
                },
                "value": None,
            },
            {
                "group": {
                    "config.technique_config_id": "arm-b",
                    "spec_hash": "hash-b",
                },
                "value": 6,
            },
        ]
    }


def test_declared_metric_group_by_keeps_scalar_types_distinct() -> None:
    metrics = MetricProjector().project_declared(
        records=(
            {"group": 1.0, "score": 60},
            {"group": True, "score": 20},
            {"group": 1},
        ),
        definitions=(
            {
                "id": "score_by_typed_group",
                "name": "Score by typed group",
                "source": "score",
                "aggregation": "average",
                "group_by": ["group"],
            },
        ),
    )

    grouped = metrics["score_by_typed_group"]
    assert isinstance(grouped, list)
    assert len(grouped) == 3
    assert [type(item["group"]["group"]) for item in grouped] == [bool, float, int]
    assert [item["value"] for item in grouped] == [20, 60, None]


@pytest.mark.parametrize("group", [float("nan"), float("inf"), float("-inf")])
def test_declared_metric_group_by_rejects_non_finite_values(group: float) -> None:
    with pytest.raises(ValueError, match="finite JSON scalars"):
        MetricProjector().project_declared(
            records=({"group": group, "score": 1},),
            definitions=(
                {
                    "id": "invalid_group",
                    "name": "Invalid group",
                    "source": "score",
                    "aggregation": "average",
                    "group_by": ["group"],
                },
            ),
        )


def test_declared_metrics_project_the_canonical_loaded_fixture() -> None:
    package = load_technique_package(
        Path(__file__).parent / "fixtures" / "techniques" / "valid_probe",
        frozenset(TechniqueCapability),
    )

    metrics = MetricProjector().project_declared(
        records=({"target_judge": 4}, {"target_judge": 8}),
        definitions=package.definition.metrics,
    )

    assert metrics == {"score": 6}


def test_duplicate_event_id_is_projected_once() -> None:
    completed = event(
        "evt-1",
        1,
        "judge.completed",
        payload={
            "definition_id": "target_judge",
            "status": "succeeded",
            "score": 8,
        },
    )
    outcome = OutcomeProjector().project(
        (completed, completed),
        result_policy=ResultPolicy(official_judge="target_judge"),
    )

    assert outcome.official_score == 8
    assert outcome.projected_event_count == 1


def test_conflicting_duplicate_event_id_is_rejected() -> None:
    first = event("evt-1", 1, "run.started")
    conflicting = event("evt-1", 1, "run.failed")

    with pytest.raises(ValueError, match="different content"):
        OutcomeProjector().project((first, conflicting), result_policy=ResultPolicy())


def test_projection_rejects_mixed_run_attempts() -> None:
    first = event("evt-1", 1, "target.completed", payload={"response": "a"})
    other = ExecutionEvent(
        event_id="evt-2",
        correlation=CorrelationIds(run_id="run-2", attempt_id="attempt-2"),
        sequence=2,
        event_type="judge.completed",
        schema_version=1,
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        technique_id="future-technique",
        technique_version="1.0.0",
        implementation_id="modular",
        payload={"definition_id": "target_judge", "score": 8},
    )

    with pytest.raises(ValueError, match="different run attempts"):
        OutcomeProjector().project(
            (first, other),
            result_policy=ResultPolicy(official_judge="target_judge"),
        )


def test_final_response_selector_uses_declared_reference() -> None:
    events = (
        event(
            "evt-1",
            1,
            "target.completed",
            payload={"response": "real", "response_ref": "target.real"},
        ),
        event(
            "evt-2",
            2,
            "target.completed",
            payload={
                "response": "poisoned",
                "response_ref": "target.poisoned",
                "response_kind": "transformed",
            },
        ),
    )

    outcome = OutcomeProjector().project(
        events,
        result_policy=ResultPolicy(final_response="target.real"),
    )

    assert outcome.final_response == "real"


def test_final_response_selector_can_reference_a_derived_response_event() -> None:
    events = (
        event(
            "target-1",
            1,
            "target.completed",
            payload={"response": "first", "response_kind": "real"},
        ),
        event(
            "target-2",
            2,
            "target.completed",
            payload={"response": "second", "response_kind": "real"},
        ),
        event(
            "joined-response",
            3,
            "value.transformed",
            payload={
                "transform": "join",
                "response_kind": "derived",
                "response": "first second",
                "source_response_event_ids": ("target-1", "target-2"),
            },
        ),
    )

    outcome = OutcomeProjector().project(
        events,
        result_policy=ResultPolicy(final_response="joined-response"),
    )

    assert outcome.final_response == "first second"
    assert outcome.final_response_event_id == "joined-response"


def test_real_response_selector_ignores_a_later_derived_response_event() -> None:
    events = (
        event(
            "target-1",
            1,
            "target.completed",
            payload={"response": "first", "response_kind": "real"},
        ),
        event(
            "target-2",
            2,
            "target.completed",
            payload={"response": "second", "response_kind": "real"},
        ),
        event(
            "joined-response",
            3,
            "value.transformed",
            payload={
                "transform": "join",
                "response_kind": "derived",
                "response": "first second",
                "source_response_event_ids": ("target-1", "target-2"),
            },
        ),
    )

    outcome = OutcomeProjector().project(
        events,
        result_policy=ResultPolicy(final_response="target.real_response"),
    )

    assert outcome.final_response == "second"
    assert outcome.final_response_event_id == "target-2"


@pytest.mark.parametrize("transform", ["structured_inventory", "evidence_summary"])
def test_artifact_backed_derived_response_can_be_selected(transform) -> None:
    response = '{"summary":"café"}'
    events = (
        event("target-1", 1, "target.completed", payload={"response": "Evidence", "response_kind": "real"}),
        event(
            "artifact-response",
            2,
            "value.transformed",
            payload={
                "transform": transform,
                "response_kind": "derived",
                "response": response,
                "artifact_id": "artifact-1",
                "media_type": "application/json",
                "content_hash": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "source_response_event_ids": ("target-1",),
            },
        ),
    )
    outcome = OutcomeProjector().project(events, result_policy=ResultPolicy(final_response="artifact-response"))
    assert outcome.final_response == response
    assert outcome.final_response_event_id == "artifact-response"


@pytest.mark.parametrize(
    "overrides",
    [
        {"content_hash": None},
        {"content_hash": "invalid"},
        {"artifact_id": None},
        {"artifact_id": ""},
        {"media_type": None},
        {"media_type": ""},
        {"transform": ""},
        {"transform": None},
        {"response": ""},
        {"response_kind": "real"},
        {"source_response_event_ids": ()},
        {"source_response_event_ids": ("unknown",)},
        {"source_response_event_ids": ("target-1", "target-1")},
        {"source_response_event_ids": ("target-2", "target-1")},
    ],
)
def test_artifact_backed_derived_response_rejects_invalid_metadata_or_sources(overrides) -> None:
    response = '{"summary":"evidence"}'
    payload = {
        "transform": "structured_inventory",
        "response_kind": "derived",
        "response": response,
        "artifact_id": "artifact-1",
        "media_type": "application/json",
        "content_hash": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "source_response_event_ids": ("target-1",),
        **overrides,
    }
    events = (
        event("target-1", 1, "target.completed", payload={"response": "Evidence", "response_kind": "real"}),
        event("target-2", 2, "target.completed", payload={"response": "More evidence", "response_kind": "real"}),
        event("artifact-response", 3, "value.transformed", payload=payload),
    )
    with pytest.raises(ValueError, match="matched no event"):
        OutcomeProjector().project(events, result_policy=ResultPolicy(final_response="artifact-response"))


@pytest.mark.parametrize("source_kind", ["fabricated", "derived"])
def test_artifact_backed_derived_response_requires_real_sources(source_kind) -> None:
    events = (
        event("target-1", 1, "target.completed", payload={"response": "Evidence", "response_kind": source_kind}),
        event(
            "artifact-response",
            2,
            "value.transformed",
            payload={
                "transform": "structured_inventory",
                "response_kind": "derived",
                "response": "summary",
                "artifact_id": "artifact-1",
                "media_type": "text/plain",
                "content_hash": hashlib.sha256(b"summary").hexdigest(),
                "source_response_event_ids": ("target-1",),
            },
        ),
    )
    with pytest.raises(ValueError, match="matched no event"):
        OutcomeProjector().project(events, result_policy=ResultPolicy(final_response="artifact-response"))


@pytest.mark.parametrize(
    "derived_payload",
    [
        {
            "transform": "replace",
            "response_kind": "derived",
            "response": "first second",
            "source_response_event_ids": ("target-1", "target-2"),
        },
        {
            "transform": "join",
            "response_kind": "derived",
            "response": "fabricated",
            "source_response_event_ids": ("target-1", "target-2"),
        },
        {
            "transform": "join",
            "response_kind": "derived",
            "response": "first second",
            "source_response_event_ids": ("target-2", "target-1"),
        },
    ],
)
def test_derived_response_selector_rejects_untruthful_join_provenance(
    derived_payload: dict[str, object],
) -> None:
    events = (
        event("target-1", 1, "target.completed", payload={"response": "first"}),
        event("target-2", 2, "target.completed", payload={"response": "second"}),
        event("joined-response", 3, "value.transformed", payload=derived_payload),
    )

    with pytest.raises(ValueError, match="matched no event"):
        OutcomeProjector().project(
            events,
            result_policy=ResultPolicy(final_response="joined-response"),
        )


def test_derived_response_requires_an_explicitly_linked_official_judge() -> None:
    events = (
        event("target-1", 1, "target.completed", payload={"response": "first"}),
        event("target-2", 2, "target.completed", payload={"response": "second"}),
        event(
            "joined-response",
            3,
            "value.transformed",
            payload={
                "transform": "join",
                "response_kind": "derived",
                "response": "first second",
                "source_response_event_ids": ("target-1", "target-2"),
            },
        ),
        event(
            "unlinked-judge",
            4,
            "judge.completed",
            payload={
                "definition_id": "target_judge",
                "status": "succeeded",
                "score": 8,
            },
        ),
    )

    outcome = OutcomeProjector().project(
        events,
        result_policy=ResultPolicy(
            official_judge="target_judge",
            final_response="joined-response",
            success_threshold=7,
        ),
    )

    assert outcome.official_score is None
    assert outcome.official_judge_event_id is None
    assert outcome.result_disposition is ResultDisposition.INCONCLUSIVE


def test_real_response_selector_never_falls_back_to_transformed_response() -> None:
    transformed = event(
        "evt-1",
        1,
        "target.completed",
        payload={
            "response": "poisoned",
            "response_kind": "transformed",
            "response_ref": "target.poisoned",
        },
    )

    with pytest.raises(ValueError, match="real response"):
        OutcomeProjector().project(
            (transformed,),
            result_policy=ResultPolicy(final_response="target.real_response"),
        )


@pytest.mark.parametrize(
    ("format_name", "event_type", "expected_key"),
    [
        ("generation", "candidate.created", "generation_number"),
        ("tree", "branch.selected", "branch_id"),
        ("diff", "value.transformed", "event_type"),
        ("judge", "judge.completed", "event_type"),
        ("artifact", "artifact.created", "event_type"),
        ("table", "run.started", "rows"),
    ],
)
def test_declared_presentation_formats_have_format_specific_shapes(
    format_name: str,
    event_type: str,
    expected_key: str,
) -> None:
    payload: dict[str, object] = {}
    if format_name == "generation":
        payload["generation"] = 1
    elif format_name == "judge":
        payload = {"definition_id": "target_judge", "score": 7}
    elif format_name == "artifact":
        payload = {
            "artifact_id": "artifact-1",
            "media_type": "text/plain",
            "content_hash": "a" * 64,
            "size_bytes": 4,
        }
    projection_event = event("evt-format", 1, event_type, payload=payload)
    if format_name == "tree":
        projection_event = ExecutionEvent(
            event_id="evt-format",
            correlation=CORRELATION,
            sequence=1,
            event_type=event_type,
            schema_version=1,
            timestamp=datetime(2026, 8, 18, tzinfo=UTC),
            technique_id="future-technique",
            technique_version="1.0.0",
            implementation_id="modular",
            branch_id="branch-a",
            payload={},
        )

    view = ConversationProjector().project(
        (projection_event,),
        presentation={"format": format_name},
    )

    assert view["format"] == format_name
    assert expected_key in view["sections"][0]
