"""Thread-safe normalized event emission."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import replace
from typing import Iterator, Mapping

from app.persistence.technique_runtime import (
    TechniqueRuntimePersistence,
    execution_event_hash,
)
from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.execution import CorrelationIds
from app.techniques.core.redaction import RedactionService


@dataclass(frozen=True, slots=True)
class TechniqueEventIdentity:
    technique_id: str
    technique_version: str
    implementation_id: str


@dataclass(slots=True)
class EventBuffer:
    owner: EventJournal
    branch_id: str
    caused_by: str | None
    parent: EventBuffer | None
    events: list[ExecutionEvent]
    merged: bool = False


_ACTIVE_EVENT_BUFFER: ContextVar[EventBuffer | None] = ContextVar(
    "technique_event_buffer",
    default=None,
)


class EventJournal:
    def __init__(
        self,
        *,
        correlation: CorrelationIds,
        persistence: TechniqueRuntimePersistence,
        technique: TechniqueEventIdentity,
        redaction: RedactionService | None = None,
    ) -> None:
        self.correlation = correlation
        self.persistence = persistence
        self.technique = technique
        self._redaction = redaction or RedactionService()
        self._lock = threading.Lock()
        self._sequence = 0
        self._events: list[ExecutionEvent] = []

    def emit(
        self,
        event_type: str,
        *,
        payload: Mapping[str, object] | None = None,
        round: int | None = None,
        branch_id: str | None = None,
        caused_by: str | None = None,
        operation_id: str | None = None,
        attempt: int = 1,
    ) -> ExecutionEvent:
        active = _ACTIVE_EVENT_BUFFER.get()
        if active is not None and active.owner is self:
            event = ExecutionEvent.create(
                correlation=self.correlation,
                sequence=len(active.events) + 1,
                event_type=event_type,
                technique_id=self.technique.technique_id,
                technique_version=self.technique.technique_version,
                implementation_id=self.technique.implementation_id,
                round=round,
                branch_id=branch_id or active.branch_id,
                caused_by=caused_by or active.caused_by,
                operation_id=operation_id,
                attempt=attempt,
                payload=self._redaction.redact_mapping(payload or {}),
            )
            active.events.append(event)
            return event
        with self._lock:
            sequence = self._sequence + 1
            safe_payload = self._redaction.redact_mapping(payload or {})
            event = ExecutionEvent.create(
                correlation=self.correlation,
                sequence=sequence,
                event_type=event_type,
                technique_id=self.technique.technique_id,
                technique_version=self.technique.technique_version,
                implementation_id=self.technique.implementation_id,
                round=round,
                branch_id=branch_id,
                caused_by=caused_by,
                operation_id=operation_id,
                attempt=attempt,
                payload=safe_payload,
            )
            self.persistence.append_event(event)
            self._events.append(event)
            self._sequence = sequence
            return event

    def events(self) -> tuple[ExecutionEvent, ...]:
        active = _ACTIVE_EVENT_BUFFER.get()
        with self._lock:
            persisted = tuple(self._events)
        if active is None or active.owner is not self:
            return persisted
        lineage: list[EventBuffer] = []
        current: EventBuffer | None = active
        while current is not None:
            lineage.append(current)
            current = current.parent
        buffered = tuple(event for buffer in reversed(lineage) for event in buffer.events)
        return (*persisted, *buffered)

    def new_buffer(
        self,
        *,
        branch_id: str,
        caused_by: str | None = None,
    ) -> EventBuffer:
        if type(branch_id) is not str or not branch_id:
            raise ValueError("event buffer branch_id must be a non-empty built-in string")
        if caused_by is not None and (type(caused_by) is not str or not caused_by):
            raise ValueError("event buffer caused_by must be absent or a non-empty built-in string")
        active = _ACTIVE_EVENT_BUFFER.get()
        return EventBuffer(
            owner=self,
            branch_id=branch_id,
            caused_by=caused_by,
            parent=active if active is not None and active.owner is self else None,
            events=[],
        )

    @contextmanager
    def activate(self, buffer: EventBuffer) -> Iterator[None]:
        if type(buffer) is not EventBuffer or buffer.owner is not self or buffer.merged:
            raise ValueError("event buffer must be an unmerged EventBuffer")
        token = _ACTIVE_EVENT_BUFFER.set(buffer)
        try:
            yield
        finally:
            _ACTIVE_EVENT_BUFFER.reset(token)

    def merge_many(self, buffers: tuple[EventBuffer, ...]) -> None:
        """Atomically merge child journals in caller-declared deterministic order."""

        if type(buffers) is not tuple:
            raise TypeError("event buffers must be provided as a tuple")
        active = _ACTIVE_EVENT_BUFFER.get()
        expected_parent = active if active is not None and active.owner is self else None
        seen_buffers: set[int] = set()
        for buffer in buffers:
            if type(buffer) is not EventBuffer:
                raise TypeError("event buffers must be exact EventBuffer values")
            if id(buffer) in seen_buffers:
                raise ValueError("event buffer was declared more than once")
            seen_buffers.add(id(buffer))
            if buffer.owner is not self:
                raise ValueError("event buffer belongs to a different event journal")
            if buffer.merged:
                raise ValueError("event buffer was already merged")
            if buffer.parent is not expected_parent:
                raise ValueError("event buffer belongs to a different causal scope")
        destination = expected_parent
        if destination is not None:
            lineage_events: list[ExecutionEvent] = list(self._events)
            lineage: EventBuffer | None = destination
            while lineage is not None:
                lineage_events.extend(lineage.events)
                lineage = lineage.parent
            self._validate_buffer_events(buffers, existing=tuple(lineage_events))
            prepared = tuple(
                replace(buffered_event, sequence=len(destination.events) + offset)
                for offset, buffered_event in enumerate(
                    (event for buffer in buffers for event in buffer.events),
                    start=1,
                )
            )
            destination.events.extend(prepared)
            for buffer in buffers:
                buffer.merged = True
            return
        with self._lock:
            self._validate_buffer_events(buffers, existing=tuple(self._events))
            prepared = tuple(
                replace(buffered_event, sequence=self._sequence + offset)
                for offset, buffered_event in enumerate(
                    (event for buffer in buffers for event in buffer.events),
                    start=1,
                )
            )
            self.persistence.append_events(prepared)
            self._events.extend(prepared)
            if prepared:
                self._sequence = prepared[-1].sequence
            for buffer in buffers:
                buffer.merged = True

    def merge(self, buffers: tuple[EventBuffer, ...]) -> None:
        """Compatibility alias for callers that already provide every child buffer."""

        self.merge_many(buffers)

    def _validate_buffer_events(
        self,
        buffers: tuple[EventBuffer, ...],
        *,
        existing: tuple[ExecutionEvent, ...],
    ) -> None:
        identities = {event.event_id: execution_event_hash(event) for event in existing}
        for buffer in buffers:
            for expected_sequence, event in enumerate(buffer.events, start=1):
                if type(event) is not ExecutionEvent:
                    raise TypeError("event buffers must contain exact ExecutionEvent values")
                if type(event.event_id) is not str or not event.event_id:
                    raise ValueError("buffered event IDs must be non-empty built-in strings")
                if event.correlation != self.correlation:
                    raise ValueError("buffered event belongs to a different execution")
                if (
                    event.technique_id != self.technique.technique_id
                    or event.technique_version != self.technique.technique_version
                    or event.implementation_id != self.technique.implementation_id
                ):
                    raise ValueError("buffered event belongs to a different technique")
                if event.sequence != expected_sequence:
                    raise ValueError("buffered event sequence is not contiguous within its owner")
                event_hash = execution_event_hash(event)
                existing_hash = identities.get(event.event_id)
                if existing_hash is not None:
                    qualifier = "different content" if existing_hash != event_hash else "duplicate content"
                    raise ValueError(f"buffered event ID '{event.event_id}' has {qualifier}")
                identities[event.event_id] = event_hash
