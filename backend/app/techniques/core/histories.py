"""Immutable named histories with fork and rollback provenance."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

from app.techniques.contracts.execution import _freeze_json_value
from app.techniques.core.events import EventJournal


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    message_id: str
    history_id: str
    branch_id: str
    role: str
    content: object
    source_kind: str
    source_event_id: str | None = None
    original_message_id: str | None = None
    visibility_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("message_id", "history_id", "branch_id", "role", "source_kind"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        object.__setattr__(self, "content", _freeze_json_value(self.content))
        object.__setattr__(self, "visibility_roles", tuple(self.visibility_roles))


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    snapshot_id: str
    history_id: str
    message_ids: tuple[str, ...]
    messages: tuple[HistoryMessage, ...]


class HistoryService:
    """Owns run-local messages and the active heads of named histories."""

    def __init__(
        self,
        *,
        events: EventJournal,
        _messages: dict[str, HistoryMessage] | None = None,
        _snapshots: dict[str, HistorySnapshot] | None = None,
        _lock: threading.RLock | None = None,
        _heads: Mapping[str, tuple[str, ...]] | None = None,
        _branch_id: str | None = None,
    ) -> None:
        self._events = events
        self._lock = _lock or threading.RLock()
        self._messages = {} if _messages is None else _messages
        self._heads = {} if _heads is None else dict(_heads)
        self._snapshots = {} if _snapshots is None else _snapshots
        self._branch_id = _branch_id

    def create(self, history_id: str) -> None:
        if not history_id:
            raise ValueError("history_id must not be empty")
        with self._lock:
            if history_id in self._heads:
                raise ValueError(f"history '{history_id}' already exists")
            self._heads[history_id] = ()
            self._events.emit(
                "history.created",
                branch_id=self._branch_id or history_id,
                payload={"history_id": history_id},
            )

    def append(
        self,
        history_id: str,
        *,
        role: str,
        content: object,
        source_kind: str,
        source_event_id: str | None = None,
        original_message_id: str | None = None,
        visibility_roles: tuple[str, ...] = (),
    ) -> HistoryMessage:
        with self._lock:
            head = self._require_history(history_id)
            if original_message_id is not None and original_message_id not in self._messages:
                raise KeyError(f"original message '{original_message_id}' does not exist")
            message = HistoryMessage(
                message_id=str(uuid4()),
                history_id=history_id,
                branch_id=self._branch_id or history_id,
                role=role,
                content=content,
                source_kind=source_kind,
                source_event_id=source_event_id,
                original_message_id=original_message_id,
                visibility_roles=visibility_roles,
            )
            self._messages[message.message_id] = message
            self._heads[history_id] = (*head, message.message_id)
            self._events.emit(
                "message.appended",
                branch_id=self._branch_id or history_id,
                caused_by=source_event_id,
                payload={
                    "history_id": history_id,
                    "message_id": message.message_id,
                    "role": role,
                    "content": message.content,
                    "source_kind": source_kind,
                    "original_message_id": original_message_id,
                    "visibility_roles": visibility_roles,
                },
            )
            return message

    def snapshot(self, history_id: str) -> HistorySnapshot:
        with self._lock:
            message_ids = self._require_history(history_id)
            snapshot = HistorySnapshot(
                snapshot_id=str(uuid4()),
                history_id=history_id,
                message_ids=message_ids,
                messages=tuple(self._messages[item] for item in message_ids),
            )
            self._snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def fork(self, source_history_id: str, new_history_id: str) -> None:
        if not new_history_id:
            raise ValueError("new_history_id must not be empty")
        with self._lock:
            source_head = self._require_history(source_history_id)
            if new_history_id in self._heads:
                raise ValueError(f"history '{new_history_id}' already exists")
            self._heads[new_history_id] = tuple(source_head)
            self._events.emit(
                "history.forked",
                branch_id=self._branch_id or new_history_id,
                payload={
                    "source_history_id": source_history_id,
                    "history_id": new_history_id,
                    "message_ids": source_head,
                },
            )

    def rollback(self, history_id: str, snapshot_id: str) -> None:
        with self._lock:
            current = self._require_history(history_id)
            snapshot = self._snapshots.get(snapshot_id)
            if snapshot is None:
                raise KeyError(f"history snapshot '{snapshot_id}' does not exist")
            if snapshot.history_id != history_id:
                raise ValueError(
                    f"snapshot '{snapshot_id}' belongs to history '{snapshot.history_id}', not '{history_id}'"
                )
            if current[: len(snapshot.message_ids)] != snapshot.message_ids:
                raise ValueError(f"snapshot '{snapshot_id}' is not an ancestor of the active history")
            self._heads[history_id] = snapshot.message_ids
            rolled_back_ids = current[len(snapshot.message_ids) :]
            self._events.emit(
                "history.rolled_back",
                branch_id=self._branch_id or history_id,
                payload={
                    "history_id": history_id,
                    "snapshot_id": snapshot_id,
                    "active_message_ids": snapshot.message_ids,
                    "rolled_back_message_ids": rolled_back_ids,
                    "inactive_message_ids": rolled_back_ids,
                },
            )

    def fork_context(self, branch_id: str) -> HistoryService:
        """Copy active heads while sharing immutable message/snapshot evidence."""

        if type(branch_id) is not str or not branch_id:
            raise ValueError("branch_id must be a non-empty built-in string")
        with self._lock:
            return HistoryService(
                events=self._events,
                _messages=self._messages,
                _snapshots=self._snapshots,
                _lock=self._lock,
                _heads=self._heads,
                _branch_id=branch_id,
            )

    def restore_heads(
        self,
        heads: Mapping[str, tuple[str, ...]],
        *,
        checkpoint_id: str,
    ) -> Mapping[str, tuple[str, ...]]:
        """Atomically restore checkpoint heads and retain exact inactive evidence."""

        return self.reconcile_checkpoint_heads(
            heads,
            checkpoint_id=checkpoint_id,
            expected_inactive=None,
            commit=lambda: None,
            before_events=None,
        )

    def reconcile_checkpoint_heads(
        self,
        heads: Mapping[str, tuple[str, ...]],
        *,
        checkpoint_id: str,
        expected_inactive: Mapping[str, tuple[str, ...]] | None,
        commit: Callable[[], None],
        before_events: Callable[[], None] | None,
    ) -> Mapping[str, tuple[str, ...]]:
        """Commit history heads and a prevalidated workflow-state update together."""

        if type(checkpoint_id) is not str or not checkpoint_id:
            raise ValueError("checkpoint_id must be a non-empty built-in string")
        if not callable(commit):
            raise TypeError("checkpoint state commit must be callable")
        if before_events is not None and not callable(before_events):
            raise TypeError("checkpoint pre-restore event hook must be callable")
        with self._lock:
            normalized, inactive = self._validate_restore_heads(heads)
            if expected_inactive is not None:
                if type(expected_inactive) not in (dict, MappingProxyType):
                    raise TypeError("expected inactive history evidence must be a built-in mapping")
                expected = {history_id: tuple(message_ids) for history_id, message_ids in expected_inactive.items()}
                if expected != inactive:
                    raise ValueError("history heads changed after checkpoint validation")
            event_buffer = self._events.new_buffer(
                branch_id=self._branch_id or "checkpoint-restore",
            )
            with self._events.activate(event_buffer):
                if before_events is not None:
                    before_events()
                for history_id in sorted(inactive):
                    if not inactive[history_id]:
                        continue
                    self._events.emit(
                        "history.rolled_back",
                        branch_id=self._branch_id or history_id,
                        payload={
                            "history_id": history_id,
                            "checkpoint_id": checkpoint_id,
                            "active_message_ids": normalized[history_id],
                            "rolled_back_message_ids": inactive[history_id],
                            "inactive_message_ids": inactive[history_id],
                        },
                    )
            self._events.merge_many((event_buffer,))
            previous = self._heads
            self._heads = normalized
            try:
                commit()
            except BaseException:
                self._heads = previous
                raise
            return MappingProxyType(inactive)

    def validate_restore_heads(
        self,
        heads: Mapping[str, tuple[str, ...]],
    ) -> Mapping[str, tuple[str, ...]]:
        with self._lock:
            _, inactive = self._validate_restore_heads(heads)
            return MappingProxyType(inactive)

    def _validate_restore_heads(
        self,
        heads: Mapping[str, tuple[str, ...]],
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
        if type(heads) not in (dict, MappingProxyType):
            raise TypeError("checkpoint history heads must be a built-in mapping")
        if set(heads) != set(self._heads):
            raise ValueError("checkpoint history ownership does not match this execution")
        normalized: dict[str, tuple[str, ...]] = {}
        inactive: dict[str, tuple[str, ...]] = {}
        for history_id, raw_ids in heads.items():
            if type(history_id) is not str or not history_id:
                raise ValueError("checkpoint history IDs must be non-empty strings")
            if type(raw_ids) is not tuple or any(
                type(message_id) is not str or not message_id for message_id in raw_ids
            ):
                raise TypeError("checkpoint history heads must contain message ID tuples")
            current = self._heads[history_id]
            if current[: len(raw_ids)] != raw_ids:
                raise ValueError(f"checkpoint history head for '{history_id}' is not an active ancestor")
            if any(message_id not in self._messages for message_id in raw_ids):
                raise ValueError("checkpoint history references unknown message evidence")
            normalized[history_id] = raw_ids
            inactive[history_id] = current[len(raw_ids) :]
        return normalized, inactive

    def all_message_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._messages)

    def heads(self) -> Mapping[str, tuple[str, ...]]:
        with self._lock:
            return MappingProxyType(dict(self._heads))

    def snapshots(self) -> Mapping[str, HistorySnapshot]:
        """Return the immutable snapshots already issued during this run."""

        with self._lock:
            return MappingProxyType(dict(self._snapshots))

    def _require_history(self, history_id: str) -> tuple[str, ...]:
        try:
            return self._heads[history_id]
        except KeyError:
            raise KeyError(f"history '{history_id}' does not exist") from None
