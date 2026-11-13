"""Translate legacy result and trace values into normalized events."""

from __future__ import annotations

from typing import Mapping

from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


def translate_legacy_trace(trace: object, events: EventJournal) -> None:
    redaction = RedactionService()
    event_type = str(getattr(trace, "event_type", "trace"))
    attributes = getattr(trace, "attributes", {})
    if not isinstance(attributes, Mapping):
        attributes = {}
    message = str(getattr(trace, "message", ""))
    actor = str(getattr(trace, "actor", "system"))
    if event_type == "iterative_round":
        round_number = _optional_int(attributes.get("round"))
        events.emit(
            "round.completed",
            round=round_number,
            payload=redaction.redact_mapping({
                "message": message,
                "actor": actor,
                "attacker_input": attributes.get("attacker_input"),
                "attacker_output": attributes.get("attacker_output"),
                "candidate_prompt": attributes.get("candidate_prompt"),
                "target_input": attributes.get("target_input"),
                "target_response": attributes.get("target_response"),
                "poisoned_response": attributes.get("poisoned_response"),
                "judge_input": attributes.get("judge_input"),
                "judge_output": attributes.get("judge_output"),
                "judge_score": _optional_number(attributes.get("judge_score")),
                "judge_rationale": attributes.get("judge_rationale"),
            }),
        )
        return
    if event_type == "judge_evaluation":
        score = _optional_number(attributes.get("judge_score"))
        failed = bool(attributes.get("judge_error")) or score is None
        events.emit(
            "judge.failed" if failed else "judge.completed",
            payload=redaction.redact_mapping({
                "definition_id": "target_judge",
                "status": "failed" if failed else "succeeded",
                "score": None if failed else score,
                "rationale": attributes.get("judge_rationale"),
                "legacy": True,
            }),
        )
        return
    events.emit(
        "legacy.trace",
        payload=redaction.redact_mapping({
            "legacy_event_type": event_type,
            "message": message,
            "actor": actor,
            "attributes": dict(attributes),
        }),
    )


def emit_legacy_result(
    result: object,
    events: EventJournal,
    *,
    include_lifecycle: bool = True,
) -> None:
    if include_lifecycle:
        events.emit("run.started", payload={"legacy": True})
    for trace in tuple(getattr(result, "traces", ())):
        translate_legacy_trace(trace, events)
    response = getattr(result, "response", None)
    final_output = getattr(response, "final_output", None)
    response_metadata = getattr(response, "metadata", {})
    response_error = (
        response_metadata.get("error")
        if isinstance(response_metadata, Mapping)
        else None
    )
    if response_error:
        events.emit(
            "target.failed",
            payload={
                "error": str(response_error),
                "legacy": True,
            },
        )
    elif isinstance(final_output, str):
        events.emit(
            "target.completed",
            payload={
                "response": RedactionService().redact_text(final_output),
                "legacy": True,
            },
        )
    events.emit("outcome.selected", payload=legacy_outcome_payload(result))
    if include_lifecycle:
        events.emit("run.completed", payload={"legacy": True})


def legacy_outcome_payload(result: object) -> dict[str, object]:
    score = getattr(result, "score", None)
    return {
        "technique_key": getattr(result, "technique_key", None),
        "score": score if _is_number(score) else None,
        "target_id": getattr(result, "target_id", None),
        "legacy": True,
    }


def _optional_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _optional_number(value: object) -> int | float | None:
    if _is_number(value):
        return value
    try:
        text = str(value)
        return float(text) if "." in text else int(text)
    except (TypeError, ValueError):
        return None


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))
