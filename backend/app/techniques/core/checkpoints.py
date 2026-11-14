"""JSON-safe runtime checkpoints with compatibility validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from app.persistence.technique_runtime import (
    CheckpointRecord,
    TechniqueRuntimePersistence,
)
from app.techniques.contracts.execution import (
    CorrelationIds,
    _freeze_json_mapping,
)
from app.techniques.core.events import EventJournal


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ExecutionCheckpoint:
    checkpoint_id: str
    correlation: CorrelationIds
    spec_hash: str
    state_schema_version: int
    position: Mapping[str, object] = field(default_factory=_empty_mapping)
    values: Mapping[str, object] = field(default_factory=_empty_mapping)
    history_heads: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    budget_usage: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if not self.spec_hash:
            raise ValueError("spec_hash must not be empty")
        if self.state_schema_version < 1:
            raise ValueError("state_schema_version must be at least 1")
        object.__setattr__(self, "position", _freeze_json_mapping(self.position))
        object.__setattr__(self, "values", _freeze_json_mapping(self.values))
        object.__setattr__(
            self,
            "history_heads",
            _freeze_history_heads(self.history_heads),
        )
        object.__setattr__(
            self,
            "budget_usage",
            _freeze_budget_usage(self.budget_usage),
        )

    def payload(self) -> Mapping[str, object]:
        return _freeze_json_mapping(
            {
                "position": self.position,
                "values": self.values,
                "history_heads": self.history_heads,
                "budget_usage": self.budget_usage,
            }
        )

    @classmethod
    def from_record(cls, record: CheckpointRecord) -> ExecutionCheckpoint:
        payload = record.payload
        return cls(
            checkpoint_id=record.checkpoint_id,
            correlation=record.correlation,
            spec_hash=record.spec_hash,
            state_schema_version=record.state_schema_version,
            position=_mapping_field(payload, "position"),
            values=_mapping_field(payload, "values"),
            history_heads=_mapping_field(payload, "history_heads"),
            budget_usage=_mapping_field(payload, "budget_usage"),
        )


class CheckpointService:
    def __init__(
        self,
        *,
        correlation: CorrelationIds,
        spec_hash: str,
        state_schema_version: int,
        persistence: TechniqueRuntimePersistence,
        events: EventJournal,
    ) -> None:
        if state_schema_version < 1:
            raise ValueError("state_schema_version must be at least 1")
        self._correlation = correlation
        self._spec_hash = spec_hash
        self._state_schema_version = state_schema_version
        self._persistence = persistence
        self._events = events

    def save(
        self,
        *,
        position: Mapping[str, object],
        values: Mapping[str, object],
        history_heads: Mapping[str, tuple[str, ...]],
        budget_usage: Mapping[str, int | float],
    ) -> ExecutionCheckpoint:
        checkpoint = ExecutionCheckpoint(
            checkpoint_id=str(uuid4()),
            correlation=self._correlation,
            spec_hash=self._spec_hash,
            state_schema_version=self._state_schema_version,
            position=position,
            values=values,
            history_heads=history_heads,
            budget_usage=budget_usage,
        )
        payload = checkpoint.payload()
        json.dumps(_plain_json(payload), allow_nan=False)
        self._persistence.save_checkpoint(
            CheckpointRecord(
                checkpoint_id=checkpoint.checkpoint_id,
                correlation=checkpoint.correlation,
                spec_hash=checkpoint.spec_hash,
                state_schema_version=checkpoint.state_schema_version,
                payload=payload,
            )
        )
        self._events.emit(
            "checkpoint.saved",
            payload={
                "checkpoint_id": checkpoint.checkpoint_id,
                "spec_hash": checkpoint.spec_hash,
                "state_schema_version": checkpoint.state_schema_version,
            },
        )
        return checkpoint

    def load(self, checkpoint_id: str) -> ExecutionCheckpoint:
        return self._load(checkpoint_id, exact_attempt=False)

    def load_for_backtrack(self, checkpoint_id: str) -> ExecutionCheckpoint:
        """Load only a checkpoint owned by this exact run and attempt."""

        return self._load(checkpoint_id, exact_attempt=True)

    def _load(
        self,
        checkpoint_id: str,
        *,
        exact_attempt: bool,
    ) -> ExecutionCheckpoint:
        record = self._persistence.load_checkpoint(checkpoint_id)
        if record is None:
            raise KeyError(f"checkpoint '{checkpoint_id}' does not exist")
        source_cell = (
            record.correlation.experiment_id,
            record.correlation.cell_id,
        )
        destination_cell = (
            self._correlation.experiment_id,
            self._correlation.cell_id,
        )
        if source_cell != destination_cell:
            raise ValueError("checkpoint belongs to a different experiment cell")
        if source_cell == (None, None) and (
            record.correlation.run_id != self._correlation.run_id
        ):
            raise ValueError("standalone checkpoint belongs to a different run")
        if exact_attempt and record.correlation != self._correlation:
            raise ValueError("backtrack checkpoint belongs to a different run or attempt")
        if record.spec_hash != self._spec_hash:
            raise ValueError("checkpoint specification hash does not match this execution")
        if record.state_schema_version != self._state_schema_version:
            raise ValueError("checkpoint state schema version is incompatible")
        checkpoint = ExecutionCheckpoint.from_record(record)
        json.dumps(_plain_json(checkpoint.payload()), allow_nan=False)
        self._events.emit(
            "checkpoint.loaded",
            payload={"checkpoint_id": checkpoint.checkpoint_id},
        )
        return checkpoint


def _freeze_history_heads(
    value: Mapping[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    heads: dict[str, tuple[str, ...]] = {}
    for history_id, message_ids in value.items():
        if not isinstance(history_id, str):
            raise TypeError("history head keys must be strings")
        frozen_ids = tuple(message_ids)
        if not all(isinstance(message_id, str) for message_id in frozen_ids):
            raise TypeError("history heads must contain message ID strings")
        heads[history_id] = frozen_ids
    return MappingProxyType(heads)


def _freeze_budget_usage(
    value: Mapping[str, int | float],
) -> Mapping[str, int | float]:
    usage: dict[str, int | float] = {}
    for key, amount in value.items():
        if not isinstance(key, str):
            raise TypeError("budget usage keys must be strings")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError("budget usage values must be numbers")
        usage[key] = amount
    _freeze_json_mapping(usage)
    return MappingProxyType(usage)


def _mapping_field(payload: Mapping[str, object], name: str) -> Mapping:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint payload field '{name}' must be a mapping")
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value
