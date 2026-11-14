"""Correlated target execution with exact input-snapshot ownership."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Mapping, Protocol
from uuid import uuid4

from app.techniques.contracts.execution import CorrelationIds, _freeze_json_mapping
from app.techniques.core.budgets import BudgetService
from app.techniques.core.cancellation import CancellationToken
from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


@dataclass(frozen=True, slots=True)
class CanonicalTargetRequest:
    target_call_id: str
    correlation: CorrelationIds
    operation_id: str
    messages: tuple[Mapping[str, object], ...]
    history_snapshot_id: str
    artifact_refs: tuple[str, ...] = ()
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "messages",
            tuple(_freeze_json_mapping(message) for message in self.messages),
        )
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        object.__setattr__(self, "options", _freeze_json_mapping(self.options))


class CanonicalTargetAdapter(Protocol):
    def execute(self, request: CanonicalTargetRequest) -> object: ...


@dataclass(frozen=True, slots=True)
class TargetCallResult:
    target_call_id: str
    correlation: CorrelationIds
    operation_id: str
    text: str
    history_snapshot_id: str
    artifact_refs: tuple[str, ...]
    tool_calls: tuple[object, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))


class TargetService:
    def __init__(
        self,
        *,
        correlation: CorrelationIds,
        adapter: CanonicalTargetAdapter,
        events: EventJournal,
        budget: BudgetService,
        cancellation: CancellationToken,
        redaction: RedactionService | None = None,
    ) -> None:
        self._correlation = correlation
        self._adapter = adapter
        self._events = events
        self._budget = budget
        self._cancellation = cancellation
        self._redaction = redaction or RedactionService()

    async def execute(
        self,
        *,
        messages: tuple[Mapping[str, object], ...],
        operation_id: str,
        history_snapshot_id: str,
        artifact_refs: tuple[str, ...] = (),
        options: Mapping[str, object] | None = None,
        empty_response_fallback: str | None = None,
        round: int | None = None,
    ) -> TargetCallResult:
        self._cancellation.raise_if_cancelled()
        self._budget.reserve(target_calls=1)
        target_call_id = str(uuid4())
        request = CanonicalTargetRequest(
            target_call_id=target_call_id,
            correlation=self._correlation,
            operation_id=operation_id,
            messages=messages,
            history_snapshot_id=history_snapshot_id,
            artifact_refs=artifact_refs,
            options=options or {},
        )
        self._events.emit(
            "target.requested",
            operation_id=operation_id,
            round=round,
            payload={
                "target_call_id": target_call_id,
                "history_snapshot_id": history_snapshot_id,
                "artifact_refs": artifact_refs,
            },
        )
        try:
            response = await asyncio.to_thread(self._adapter.execute, request)
            text = getattr(response, "final_output", response)
            if not isinstance(text, str):
                raise TypeError("target adapter response must provide string final_output")
            if empty_response_fallback is not None and not text.strip():
                text = empty_response_fallback
            tool_calls = tuple(getattr(response, "tool_calls", ()))
            metadata = getattr(response, "metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {"raw_metadata": str(metadata)}
            result = TargetCallResult(
                target_call_id=target_call_id,
                correlation=self._correlation,
                operation_id=operation_id,
                text=text,
                history_snapshot_id=history_snapshot_id,
                artifact_refs=artifact_refs,
                tool_calls=tool_calls,
                metadata=self._redaction.redact_target(metadata),
            )
        except Exception as error:
            self._events.emit(
                "target.failed",
                operation_id=operation_id,
                round=round,
                payload={
                    "target_call_id": target_call_id,
                    "category": _target_failure_category(error),
                    "error": self._redaction.redact_text(str(error)),
                },
            )
            raise
        self._events.emit(
            "target.completed",
            operation_id=operation_id,
            round=round,
            payload={
                "target_call_id": target_call_id,
                "history_snapshot_id": history_snapshot_id,
                "artifact_refs": artifact_refs,
                "response": self._redaction.redact_text(text),
                "tool_calls": self._redaction.redact_target(
                    {"values": tool_calls}
                )["values"],
                "metadata": result.metadata,
            },
        )
        return result


def _target_failure_category(error: BaseException) -> str:
    explicit = getattr(error, "category", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    return type(error).__name__
