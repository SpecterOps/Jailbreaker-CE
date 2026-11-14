"""Normalized event and judge-result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import re
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from app.techniques.contracts.definition import JudgePurpose
from app.techniques.contracts.execution import CorrelationIds, _freeze_json_mapping


_EVENT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


class JudgeStatus(StrEnum):
    NOT_SCHEDULED = "not_scheduled"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class JudgeResult:
    definition_id: str
    status: JudgeStatus
    score: int | float | None = None
    result_id: str = field(default_factory=lambda: str(uuid4()))
    purpose: JudgePurpose | None = None
    subject_refs: tuple[str, ...] = ()
    scale_min: float = 0
    scale_max: float = 10
    rationale: str | None = None
    model_role: str | None = None
    profile_id: str | None = None
    policy_version: str | None = None
    prompt_version: str | None = None
    raw_input_ref: str | None = None
    raw_output_ref: str | None = None
    parse_status: str | None = None
    retry_attempts: int = 0
    token_usage: Mapping[str, object] = field(default_factory=_empty_mapping)
    latency_ms: float | None = None
    metadata: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.status is JudgeStatus.SUCCEEDED and self.score is None:
            raise ValueError("succeeded judge results require a score")
        if self.status is not JudgeStatus.SUCCEEDED and self.score is not None:
            raise ValueError("only succeeded judge results may contain a score")
        if self.scale_min > self.scale_max:
            raise ValueError("judge scale_min must not exceed scale_max")
        if self.score is not None and not self.scale_min <= self.score <= self.scale_max:
            raise ValueError("judge score must be within its declared scale")
        if self.retry_attempts < 0:
            raise ValueError("judge retry_attempts must not be negative")
        object.__setattr__(self, "subject_refs", tuple(self.subject_refs))
        object.__setattr__(self, "token_usage", _freeze_json_mapping(self.token_usage))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    event_id: str
    correlation: CorrelationIds
    sequence: int
    event_type: str
    schema_version: int
    timestamp: datetime
    technique_id: str
    technique_version: str
    implementation_id: str
    round: int | None = None
    branch_id: str | None = None
    caused_by: str | None = None
    operation_id: str | None = None
    attempt: int = 1
    payload: Mapping[str, object] = field(default_factory=_empty_mapping)

    @classmethod
    def create(
        cls,
        *,
        correlation: CorrelationIds,
        sequence: int,
        event_type: str,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
        schema_version: int = 1,
        round: int | None = None,
        branch_id: str | None = None,
        caused_by: str | None = None,
        operation_id: str | None = None,
        attempt: int = 1,
        payload: Mapping[str, object] | None = None,
    ) -> ExecutionEvent:
        return cls(
            event_id=str(uuid4()),
            correlation=correlation,
            sequence=sequence,
            event_type=event_type,
            schema_version=schema_version,
            timestamp=datetime.now(timezone.utc),
            technique_id=technique_id,
            technique_version=technique_version,
            implementation_id=implementation_id,
            round=round,
            branch_id=branch_id,
            caused_by=caused_by,
            operation_id=operation_id,
            attempt=attempt,
            payload={} if payload is None else payload,
        )

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("event sequence must be at least 1")
        if self.schema_version < 1:
            raise ValueError("event schema_version must be at least 1")
        if self.attempt < 1:
            raise ValueError("event attempt must be at least 1")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware")
        if not _EVENT_TYPE_PATTERN.fullmatch(self.event_type):
            raise ValueError(f"invalid event type '{self.event_type}'")
        _validate_standard_event(self.event_type, self.payload)
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))


def _validate_standard_event(
    event_type: str,
    payload: Mapping[str, object],
) -> None:
    if event_type == "judge.completed":
        definition_id = payload.get("definition_id")
        score = payload.get("score")
        if not isinstance(definition_id, str) or not definition_id:
            raise ValueError("judge.completed requires definition_id")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("judge.completed requires a numeric score")
        if payload.get("status", "succeeded") != "succeeded":
            raise ValueError("judge.completed status must be succeeded")
    elif event_type == "judge.failed":
        definition_id = payload.get("definition_id")
        if not isinstance(definition_id, str) or not definition_id:
            raise ValueError("judge.failed requires definition_id")
        if payload.get("score") is not None:
            raise ValueError("judge.failed must not contain a score")
    elif event_type == "target.completed":
        if not isinstance(payload.get("response"), str):
            raise ValueError("target.completed requires a string response")
    elif event_type == "artifact.created":
        for field_name in ("artifact_id", "media_type", "content_hash"):
            value = payload.get(field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"artifact.created requires {field_name}")
        size = payload.get("size_bytes")
        if size is not None and (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
        ):
            raise ValueError("artifact.created size_bytes must be a non-negative integer")
    elif event_type.startswith("model."):
        model_call_id = payload.get("model_call_id")
        if not isinstance(model_call_id, str) or not model_call_id:
            raise ValueError(f"{event_type} requires model_call_id")
