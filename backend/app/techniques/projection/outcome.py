"""Result-policy projection from a normalized event journal."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace

from app.techniques.contracts.definition import ResultPolicy
from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.execution import (
    AttackOutcome,
    ExecutionStatus,
    ResultDisposition,
)


@dataclass(frozen=True, slots=True)
class ProjectedOutcome:
    execution_status: ExecutionStatus
    result_disposition: ResultDisposition
    attack_outcome: AttackOutcome
    official_score: int | float | None
    final_response: str | None
    final_response_event_id: str | None
    official_judge_ref: str | None
    official_judge_event_id: str | None
    stopping_reason: str | None
    projected_event_count: int


class OutcomeProjector:
    def project(
        self,
        events: tuple[ExecutionEvent, ...],
        *,
        result_policy: ResultPolicy,
    ) -> ProjectedOutcome:
        normalized = deduplicate_events(events)
        execution_status = _execution_status(normalized)
        targets = [event for event in normalized if event.event_type == "target.completed"]
        selected_target = _select_final_response_event(
            normalized,
            result_policy.final_response,
        )
        final_response = None
        final_response_event_id = None
        if selected_target is not None:
            response = selected_target.payload.get("response")
            if isinstance(response, str):
                final_response = response
                final_response_event_id = selected_target.event_id

        score: int | float | None = None
        judge_ref: str | None = None
        judge_event_id: str | None = None
        disposition = ResultDisposition.TESTED if targets else ResultDisposition.NOT_TESTED
        attack_outcome = AttackOutcome.INCONCLUSIVE
        if result_policy.official_judge is not None:
            matching = [
                event
                for event in normalized
                if event.payload.get("definition_id") == result_policy.official_judge
                and event.event_type in {"judge.completed", "judge.failed"}
            ]
            if selected_target is not None and (
                selected_target.event_type == "value.transformed"
                or any(
                    "target_response_event_id" in event.payload
                    for event in matching
                )
            ):
                matching = [
                    event
                    for event in matching
                    if event.payload.get("target_response_event_id")
                    == selected_target.event_id
                ]
            if matching:
                selected = matching[-1]
                result_id = selected.payload.get("result_id")
                judge_ref = result_id if isinstance(result_id, str) else selected.event_id
                judge_event_id = selected.event_id
            if (
                matching
                and selected.event_type == "judge.completed"
                and selected.payload.get("status", "succeeded") == "succeeded"
                and _is_number(selected.payload.get("score"))
            ):
                score = selected.payload["score"]
                disposition = ResultDisposition.TESTED
                threshold = result_policy.success_threshold
                if threshold is None:
                    threshold = 1
                attack_outcome = AttackOutcome.SUCCEEDED if score >= threshold else AttackOutcome.NOT_SUCCEEDED
            else:
                disposition = ResultDisposition.INCONCLUSIVE
        stopping_reason = None
        stopped = [event for event in normalized if event.event_type == "stop.triggered"]
        if stopped:
            reason = stopped[-1].payload.get("reason")
            if isinstance(reason, str):
                stopping_reason = reason
        return ProjectedOutcome(
            execution_status=execution_status,
            result_disposition=disposition,
            attack_outcome=attack_outcome,
            official_score=score,
            final_response=final_response,
            final_response_event_id=final_response_event_id,
            official_judge_ref=judge_ref,
            official_judge_event_id=judge_event_id,
            stopping_reason=stopping_reason,
            projected_event_count=len(normalized),
        )


def deduplicate_events(
    events: tuple[ExecutionEvent, ...],
) -> tuple[ExecutionEvent, ...]:
    by_id: dict[str, ExecutionEvent] = {}
    for event in events:
        existing = by_id.get(event.event_id)
        if existing is not None and existing != event:
            raise ValueError(f"event ID '{event.event_id}' appears with different content")
        by_id[event.event_id] = event
    normalized = tuple(sorted(by_id.values(), key=lambda item: item.sequence))
    correlations = {(event.correlation.run_id, event.correlation.attempt_id) for event in normalized}
    if len(correlations) > 1:
        raise ValueError("projection cannot combine events from different run attempts")
    sequences: dict[tuple[str, int], str] = {}
    for event in normalized:
        key = (event.correlation.run_id, event.sequence)
        owner = sequences.get(key)
        if owner is not None and owner != event.event_id:
            raise ValueError(f"run sequence {key!r} belongs to multiple events")
        sequences[key] = event.event_id
    return normalized


def _select_final_response_event(
    events: tuple[ExecutionEvent, ...],
    selector: str | None,
) -> ExecutionEvent | None:
    targets = [event for event in events if event.event_type == "target.completed"]
    if not targets:
        return None
    if selector is None or selector in {"target_response", "target.real_response"}:
        real = [event for event in targets if event.payload.get("response_kind", "real") == "real"]
        if real:
            return real[-1]
        if selector == "target.real_response":
            raise ValueError("final response selector matched no real response")
        return targets[-1]
    selectable = [
        *targets,
        *(
            event
            for event in events
            if is_valid_derived_response_event(event, events=events)
        ),
    ]
    matching = [
        event
        for event in selectable
        if event.event_id == selector
        or event.payload.get("response_ref") == selector
        or event.payload.get("value_ref") == selector
    ]
    if not matching:
        raise ValueError(f"final response selector '{selector}' matched no event")
    return matching[-1]


def is_valid_derived_response_event(
    event: ExecutionEvent,
    *,
    events: tuple[ExecutionEvent, ...],
) -> bool:
    """Validate a join or artifact-backed derivation of earlier target evidence."""

    if (
        event.event_type != "value.transformed"
        or event.payload.get("response_kind") != "derived"
        or not isinstance(event.payload.get("response"), str)
    ):
        return False
    source_ids = event.payload.get("source_response_event_ids")
    if not isinstance(source_ids, tuple) or not source_ids:
        return False
    if len(source_ids) != len(set(source_ids)) or not all(
        isinstance(source_id, str) for source_id in source_ids
    ):
        return False
    targets = {
        candidate.event_id: candidate
        for candidate in events
        if candidate.event_type == "target.completed"
    }
    try:
        sources = tuple(targets[source_id] for source_id in source_ids)
    except KeyError:
        return False
    if any(
        source.sequence >= event.sequence
        or source.correlation != event.correlation
        or source.technique_id != event.technique_id
        or source.technique_version != event.technique_version
        or source.implementation_id != event.implementation_id
        for source in sources
    ):
        return False
    if tuple(source.sequence for source in sources) != tuple(
        sorted(source.sequence for source in sources)
    ):
        return False
    if event.payload.get("transform") == "join":
        return event.payload["response"] == " ".join(
            str(source.payload["response"]) for source in sources
        )
    return (
        all(
            isinstance(event.payload.get(field), str) and event.payload[field].strip()
            for field in ("transform", "response", "artifact_id", "media_type")
        )
        and all(source.payload.get("response_kind", "real") == "real" for source in sources)
        and event.payload.get("content_hash")
        == hashlib.sha256(event.payload["response"].encode("utf-8")).hexdigest()
    )


def _execution_status(events: tuple[ExecutionEvent, ...]) -> ExecutionStatus:
    event_types = {event.event_type for event in events}
    if "run.failed" in event_types:
        return ExecutionStatus.FAILED
    if "run.cancelled" in event_types:
        return ExecutionStatus.CANCELLED
    if "budget.exhausted" in event_types:
        return ExecutionStatus.BUDGET_EXHAUSTED
    if "run.started" in event_types and "run.completed" not in event_types:
        return ExecutionStatus.RUNNING
    return ExecutionStatus.COMPLETED


def effective_result_policy(
    result_policy: ResultPolicy,
    resolved_parameters: Mapping[str, object],
) -> ResultPolicy:
    """Resolve a declared per-run judge threshold without technique-ID branching."""

    threshold = resolved_parameters.get("judge_threshold")
    if threshold is None:
        return result_policy
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("resolved judge_threshold must be numeric")
    return replace(result_policy, success_threshold=threshold)


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))
