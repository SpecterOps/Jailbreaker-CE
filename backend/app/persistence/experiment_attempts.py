"""Immutable experiment-attempt persistence with lease-backed claims."""

from __future__ import annotations

import json
import math
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Mapping

from app.experiment_plan_integrity import (
    is_complete_resolved_spec,
    technique_config_id as canonical_technique_config_id,
    validate_normalized_technique_config,
)
from app.persistence.technique_runtime import ensure_experiment_attempt_schema
from app.persistence.runs import InMemoryExperimentExecutionAuthority
from app.techniques.contracts.execution import (
    ExecutionStatus,
    FailureCategory,
    ResultDisposition,
    _freeze_json_mapping,
)
from app.techniques.core.redaction import contains_secret_material


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _now() -> datetime:
    return datetime.now(UTC)


def _generated_id() -> str:
    return str(uuid.uuid4())


def _freeze(value: Mapping[str, object]) -> Mapping[str, object]:
    return _freeze_json_mapping(value)


@dataclass(frozen=True, slots=True)
class TechniqueConfigRecord:
    technique_config_id: str | None
    experiment_id: str
    technique_id: str
    technique_version: str
    implementation_id: str
    spec_hash: str
    preset: str | None = None
    parameters: Mapping[str, object] = field(default_factory=_empty_mapping)
    resolved_spec: Mapping[str, object] = field(default_factory=_empty_mapping)
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("experiment_id", "technique_id", "technique_version", "implementation_id", "spec_hash"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.technique_config_id == "":
            raise ValueError("technique_config_id must be absent or non-empty")
        validate_normalized_technique_config(
            {
                "technique_config_id": self.technique_config_id,
                "experiment_id": self.experiment_id,
                "technique_id": self.technique_id,
                "technique_version": self.technique_version,
                "implementation_id": self.implementation_id,
                "preset": self.preset,
                "parameters": self.parameters,
                "resolved_spec": self.resolved_spec,
                "spec_hash": self.spec_hash,
            },
            experiment_id=self.experiment_id,
        )
        object.__setattr__(self, "parameters", _freeze(self.parameters))
        object.__setattr__(self, "resolved_spec", _freeze(self.resolved_spec))


@dataclass(frozen=True, slots=True)
class CellRecord:
    cell_id: str | None
    experiment_id: str
    technique_config_id: str
    technique_key: str
    prompt_id: str
    repeat_index: int
    selected_attempt_id: str | None = None
    latest_attempt_id: str | None = None
    active_attempt_id: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("experiment_id", "technique_config_id", "technique_key", "prompt_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.cell_id == "":
            raise ValueError("cell_id must be absent or non-empty")
        if type(self.repeat_index) is not int or self.repeat_index < 1:
            raise ValueError("repeat_index must be a positive integer")


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    cell_id: str
    attempt_number: int
    parent_attempt_id: str | None
    retry_kind: str
    spec_hash: str
    execution_status: ExecutionStatus
    result_disposition: ResultDisposition
    execution_generation: int = 0
    score: int | float | None = None
    run_id: str | None = None
    failure_category: FailureCategory | None = None
    checkpoint_id: str | None = None
    promotion_policy: str = "replace_on_valid_completion"
    selection_reason: str | None = None
    late_completion: bool = False
    recoverable_cancellation: bool = False
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.cell_id or not self.retry_kind or not self.spec_hash:
            raise ValueError("attempt identity and pinned specification must not be empty")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be at least one")
        _validate_execution_generation(self.execution_generation)
        if self.promotion_policy not in {
            "diagnostic_only",
            "replace_on_valid_completion",
        }:
            raise ValueError("unsupported attempt promotion policy")
        if self.result_disposition is ResultDisposition.TESTED or self.score is not None:
            _validate_completion(
                self.execution_status,
                self.result_disposition,
                self.score,
            )


@dataclass(frozen=True, slots=True)
class OwnedAttemptCompletion:
    completed_by_worker: bool
    attempt: AttemptRecord


@dataclass(frozen=True, slots=True)
class RetryBatchRecord:
    retry_batch_id: str
    experiment_id: str
    request_id: str | None
    action: str
    completed_policy: str
    requested_by: str
    cell_ids: tuple[str, ...]
    execution_generation: int | None = None
    selection_metadata: Mapping[str, object] = field(default_factory=_empty_mapping)
    members: tuple[RetryBatchCellRecord, ...] = ()
    status: str = "requested"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    was_created: bool = field(default=True, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.retry_batch_id or not self.experiment_id or not self.action:
            raise ValueError("retry batch identity and action must not be empty")
        if self.request_id is not None and (
            not isinstance(self.request_id, str) or not self.request_id.strip()
        ):
            raise ValueError("request_id must be absent or a non-empty string")
        if not isinstance(self.requested_by, str) or not self.requested_by.strip():
            raise ValueError("requested_by must be a non-empty string")
        if self.execution_generation is not None:
            _validate_execution_generation(self.execution_generation)
        if contains_secret_material(self.request_id) or contains_secret_material(
            self.requested_by
        ):
            raise ValueError("retry batch audit fields must not contain secrets")
        if self.action not in {"resume", "restart"}:
            raise ValueError("retry action must be 'resume' or 'restart'")
        if self.completed_policy not in {
            "exclude",
            "diagnostic_only",
            "replace_on_valid_completion",
            "new_repeat",
        }:
            raise ValueError("unsupported completed-cell policy")
        object.__setattr__(self, "cell_ids", tuple(self.cell_ids))
        if contains_secret_material(self.selection_metadata):
            raise ValueError("retry batch selection metadata must not contain secrets")
        object.__setattr__(self, "selection_metadata", _freeze(self.selection_metadata))
        object.__setattr__(self, "members", tuple(self.members))


@dataclass(frozen=True, slots=True)
class RetryBatchCellRecord:
    retry_batch_id: str
    cell_id: str
    ordinal: int = 0
    status: str = "requested"
    attempt_id: str | None = None
    error_category: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.retry_batch_id or not self.cell_id:
            raise ValueError("retry batch member identity must not be empty")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("retry batch member ordinal must be a nonnegative integer")


def _validate_completion(
    execution_status: ExecutionStatus,
    result_disposition: ResultDisposition,
    score: int | float | None,
) -> None:
    if execution_status not in {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.TIMED_OUT,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.BUDGET_EXHAUSTED,
    }:
        raise ValueError("completion requires a terminal execution status")
    tested = result_disposition is ResultDisposition.TESTED
    if tested and (execution_status is not ExecutionStatus.COMPLETED or score is None):
        raise ValueError("tested results require completed execution and a numeric score")
    if not tested and score is not None:
        raise ValueError("scores are only valid for completed tested results")
    if score is not None and (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
    ):
        raise ValueError("attempt score must be a finite numeric score")


def _is_valid_selection(attempt: AttemptRecord | None) -> bool:
    return bool(
        attempt
        and attempt.execution_status is ExecutionStatus.COMPLETED
        and attempt.result_disposition is ResultDisposition.TESTED
        and not attempt.late_completion
    )


def _is_terminal(attempt: AttemptRecord) -> bool:
    return attempt.execution_status in {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.TIMED_OUT,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.BUDGET_EXHAUSTED,
    }


def _terminal_evidence(
    attempt: AttemptRecord,
) -> tuple[object, ...]:
    return (
        attempt.execution_status,
        attempt.result_disposition,
        attempt.score,
        attempt.run_id,
        attempt.failure_category,
    )


def _requested_terminal_evidence(
    *,
    attempt: AttemptRecord,
    execution_status: ExecutionStatus,
    result_disposition: ResultDisposition,
    score: int | float | None,
    run_id: str | None,
    failure_category: FailureCategory | None,
) -> tuple[object, ...]:
    return (
        execution_status,
        result_disposition,
        score,
        run_id or attempt.run_id,
        failure_category,
    )


def _is_retryable_selection(attempt: AttemptRecord | None) -> bool:
    if attempt is None:
        return True
    if _is_valid_selection(attempt):
        return False
    if attempt.execution_status is ExecutionStatus.BUDGET_EXHAUSTED:
        return False
    if attempt.execution_status is ExecutionStatus.CANCELLED:
        return attempt.recoverable_cancellation
    return (
        attempt.execution_status in {ExecutionStatus.FAILED, ExecutionStatus.TIMED_OUT}
        or attempt.result_disposition is ResultDisposition.INCONCLUSIVE
        or attempt.late_completion
    )


def _validate_positive_limit(value: int | None, *, name: str) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer")


class ExperimentAttemptPersistence(ABC):
    @abstractmethod
    def upsert_technique_config(self, record: TechniqueConfigRecord) -> TechniqueConfigRecord: ...

    @abstractmethod
    def create_cell(self, record: CellRecord) -> CellRecord: ...

    @abstractmethod
    def create_attempt(
        self,
        cell_id: str,
        *,
        retry_kind: str,
        spec_hash: str,
        execution_generation: int = 0,
        execution_owner_id: str | None = None,
        parent_attempt_id: str | None = None,
        checkpoint_id: str | None = None,
        promotion_policy: str = "replace_on_valid_completion",
        recoverable_cancellation: bool = False,
        max_attempts: int | None = None,
        max_retry_attempts_per_experiment: int | None = None,
    ) -> AttemptRecord: ...

    @abstractmethod
    def claim_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool: ...

    @abstractmethod
    def heartbeat_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool: ...

    @abstractmethod
    def abandon_attempt_claim(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool: ...

    @abstractmethod
    def complete_attempt(self, attempt_id: str, *, execution_status: ExecutionStatus, result_disposition: ResultDisposition, score: int | float | None = None, run_id: str | None = None, failure_category: FailureCategory | None = None) -> AttemptRecord: ...

    @abstractmethod
    def complete_attempt_if_owned(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
        execution_status: ExecutionStatus,
        result_disposition: ResultDisposition,
        score: int | float | None = None,
        run_id: str | None = None,
        failure_category: FailureCategory | None = None,
    ) -> OwnedAttemptCompletion: ...

    @abstractmethod
    def get_cell(self, cell_id: str) -> CellRecord: ...

    @abstractmethod
    def get_attempt(self, attempt_id: str) -> AttemptRecord: ...

    @abstractmethod
    def list_attempts(self, cell_id: str) -> tuple[AttemptRecord, ...]: ...

    @abstractmethod
    def has_active_attempts(
        self,
        experiment_id: str,
        *,
        execution_generation: int,
    ) -> bool: ...

    @abstractmethod
    def retryable_cells(self, experiment_id: str) -> tuple[CellRecord, ...]: ...

    @abstractmethod
    def recover_expired_attempt(
        self,
        cell_id: str,
        *,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> CellRecord: ...

    @abstractmethod
    def count_attempts(self, experiment_id: str, *, retries_only: bool = False) -> int: ...

    @abstractmethod
    def create_retry_batch(
        self,
        experiment_id: str,
        *,
        action: str,
        completed_policy: str = "exclude",
        cell_ids: tuple[str, ...],
        requested_by: str,
        request_id: str | None = None,
        selection_metadata: Mapping[str, object] | None = None,
    ) -> RetryBatchRecord: ...

    @abstractmethod
    def get_retry_batch(self, retry_batch_id: str) -> RetryBatchRecord: ...

    @abstractmethod
    def get_retry_batch_by_request(self, experiment_id: str, request_id: str) -> RetryBatchRecord | None: ...

    @abstractmethod
    def list_retry_batches(
        self,
        experiment_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> tuple[RetryBatchRecord, ...]: ...

    @abstractmethod
    def update_retry_batch(self, retry_batch_id: str, *, status: str) -> RetryBatchRecord: ...

    @abstractmethod
    def bind_retry_batch_generation(
        self,
        retry_batch_id: str,
        execution_generation: int,
    ) -> RetryBatchRecord: ...

    @abstractmethod
    def bind_retry_batch_attempt(self, retry_batch_id: str, cell_id: str, attempt_id: str) -> RetryBatchCellRecord: ...

    @abstractmethod
    def create_and_bind_retry_attempt(
        self,
        retry_batch_id: str,
        cell_id: str,
        *,
        retry_kind: str,
        spec_hash: str,
        execution_generation: int = 0,
        checkpoint_id: str | None = None,
        promotion_policy: str = "replace_on_valid_completion",
        recoverable_cancellation: bool = False,
        max_attempts: int | None = None,
        max_retry_attempts_per_experiment: int | None = None,
    ) -> AttemptRecord: ...

    @abstractmethod
    def complete_retry_batch_cell(self, retry_batch_id: str, cell_id: str, *, status: str, error_category: str | None) -> RetryBatchCellRecord: ...


class InMemoryExperimentAttemptPersistence(ExperimentAttemptPersistence):
    def __init__(
        self,
        *,
        execution_authority: InMemoryExperimentExecutionAuthority | None = None,
    ) -> None:
        self._configs: dict[str, TechniqueConfigRecord] = {}
        self._cells: dict[str, CellRecord] = {}
        self._attempts: dict[str, AttemptRecord] = {}
        self._cell_identities: dict[tuple[str, str, str, int], str] = {}
        self._batches: dict[str, RetryBatchRecord] = {}
        self._batch_members: dict[tuple[str, str], RetryBatchCellRecord] = {}
        self._batch_requests: dict[tuple[str, str], str] = {}
        self._execution_authority = (
            execution_authority or InMemoryExperimentExecutionAuthority()
        )
        self._lock = self._execution_authority.lock

    def attach_execution_authority(
        self,
        authority: InMemoryExperimentExecutionAuthority,
    ) -> None:
        if authority is self._execution_authority:
            return
        with self._lock:
            self._execution_authority = authority
            self._lock = authority.lock
            with authority.lock:
                for attempt in self._attempts.values():
                    if attempt.execution_status not in {
                        ExecutionStatus.QUEUED,
                        ExecutionStatus.RUNNING,
                    }:
                        continue
                    cell = self._cells[attempt.cell_id]
                    authority.active_attempts[attempt.attempt_id] = (
                        cell.experiment_id,
                        attempt.execution_generation,
                        str(
                            authority.experiments.get(cell.experiment_id, {}).get(
                                "execution_owner_id",
                                "",
                            )
                        ),
                    )

    def snapshot_plan_state(self) -> tuple[dict, ...]:
        """Capture rollback state while the shared execution authority is held."""

        with self._lock:
            return (
                dict(self._configs),
                dict(self._cells),
                dict(self._attempts),
                dict(self._cell_identities),
                dict(self._batches),
                dict(self._batch_members),
                dict(self._batch_requests),
                dict(self._execution_authority.active_attempts),
            )

    def restore_plan_state(self, snapshot: tuple[dict, ...]) -> None:
        """Restore a snapshot captured by :meth:`snapshot_plan_state`."""

        with self._lock:
            (
                self._configs,
                self._cells,
                self._attempts,
                self._cell_identities,
                self._batches,
                self._batch_members,
                self._batch_requests,
                active_attempts,
            ) = snapshot
            self._execution_authority.active_attempts.clear()
            self._execution_authority.active_attempts.update(active_attempts)

    def upsert_technique_config(self, record: TechniqueConfigRecord) -> TechniqueConfigRecord:
        with self._lock:
            if record.technique_config_id is None:
                for existing in self._configs.values():
                    if _config_content(existing) == _config_content(record):
                        return existing
                record = replace(
                    record,
                    technique_config_id=(
                        canonical_technique_config_id(
                            record.experiment_id,
                            record.resolved_spec,
                        )
                        if is_complete_resolved_spec(record.resolved_spec)
                        else _generated_id()
                    ),
                )
            existing = self._configs.get(record.technique_config_id)
            if existing is not None:
                if _config_content(existing) != _config_content(record):
                    raise ValueError("technique configuration already exists with different content")
                return existing
            self._configs[record.technique_config_id] = record
            return record

    def create_cell(self, record: CellRecord) -> CellRecord:
        with self._lock:
            _validate_planned_cell(record)
            config = self._configs.get(record.technique_config_id)
            if config is None or config.experiment_id != record.experiment_id:
                raise ValueError("technique configuration does not belong to experiment")
            identity = (record.experiment_id, record.technique_config_id, record.prompt_id, record.repeat_index)
            existing_id = self._cell_identities.get(identity)
            if existing_id is not None:
                existing = self._cells[existing_id]
                if record.cell_id not in (None, existing.cell_id) or _planned_cell_content(existing) != _planned_cell_content(record):
                    raise ValueError("cell identity already exists with different content")
                return existing
            record = replace(record, cell_id=record.cell_id or _generated_id())
            existing = self._cells.get(record.cell_id)
            if existing is not None:
                if _planned_cell_content(existing) != _planned_cell_content(record):
                    raise ValueError("cell ID already exists with different content")
                return existing
            self._cells[record.cell_id] = record
            self._cell_identities[identity] = record.cell_id
            return record

    def create_attempt(
        self,
        cell_id: str,
        *,
        retry_kind: str,
        spec_hash: str,
        execution_generation: int = 0,
        execution_owner_id: str | None = None,
        parent_attempt_id: str | None = None,
        checkpoint_id: str | None = None,
        promotion_policy: str = "replace_on_valid_completion",
        recoverable_cancellation: bool = False,
        max_attempts: int | None = None,
        max_retry_attempts_per_experiment: int | None = None,
    ) -> AttemptRecord:
        _validate_execution_generation(execution_generation)
        _validate_positive_limit(max_attempts, name="max_attempts_per_cell")
        _validate_positive_limit(
            max_retry_attempts_per_experiment,
            name="max_retry_attempts_per_experiment",
        )
        with self._lock:
            cell = self.get_cell(cell_id)
            self._require_owned_execution_locked(
                cell.experiment_id,
                execution_generation=execution_generation,
                execution_owner_id=execution_owner_id,
            )
            cell = self._recover_expired_active(cell)
            config = self._configs[cell.technique_config_id]
            if config.spec_hash != spec_hash:
                raise ValueError("attempt spec hash must match the cell's pinned spec hash")
            attempts = self.list_attempts(cell_id)
            if any(
                attempt.execution_status in {ExecutionStatus.QUEUED, ExecutionStatus.RUNNING}
                for attempt in attempts
            ):
                raise ValueError("cell already has an active or claimable attempt")
            if max_attempts is not None and len(attempts) >= max_attempts:
                raise ValueError("max_attempts_per_cell retry limit exceeded")
            if (
                retry_kind != "initial"
                and max_retry_attempts_per_experiment is not None
                and self.count_attempts(cell.experiment_id, retries_only=True)
                >= max_retry_attempts_per_experiment
            ):
                raise ValueError(
                    "max_retry_attempts_per_experiment retry limit exceeded"
                )
            parent_attempt_id = parent_attempt_id or cell.latest_attempt_id
            if parent_attempt_id is not None and self.get_attempt(parent_attempt_id).cell_id != cell_id:
                raise ValueError("parent attempt must belong to the same cell")
            attempt = AttemptRecord(
                attempt_id=_generated_id(), cell_id=cell_id,
                attempt_number=len(attempts) + 1,
                parent_attempt_id=parent_attempt_id, retry_kind=retry_kind, spec_hash=spec_hash,
                execution_status=ExecutionStatus.QUEUED,
                result_disposition=ResultDisposition.NOT_TESTED,
                execution_generation=execution_generation,
                checkpoint_id=checkpoint_id,
                promotion_policy=promotion_policy,
                recoverable_cancellation=recoverable_cancellation,
                created_at=_now(),
            )
            self._attempts[attempt.attempt_id] = attempt
            self._cells[cell_id] = replace(cell, latest_attempt_id=attempt.attempt_id)
            self._execution_authority.active_attempts[attempt.attempt_id] = (
                cell.experiment_id,
                execution_generation,
                execution_owner_id or "",
            )
            return attempt

    def claim_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool:
        _validate_lease(worker_id, lease_seconds)
        with self._lock:
            attempt = self.get_attempt(attempt_id)
            cell = self.get_cell(attempt.cell_id)
            captured_generation = (
                attempt.execution_generation
                if execution_generation is None
                else execution_generation
            )
            captured_experiment_id = experiment_id or cell.experiment_id
            if (
                captured_experiment_id != cell.experiment_id
                or captured_generation != attempt.execution_generation
                or not self._owns_execution_locked(
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
            ):
                return False
            if attempt.execution_status is not ExecutionStatus.QUEUED:
                return False
            cell = self._recover_expired_active(cell)
            if cell.active_attempt_id is not None:
                return False
            lease_expires_at = _now() + timedelta(seconds=lease_seconds)
            self._attempts[attempt_id] = replace(attempt, execution_status=ExecutionStatus.RUNNING, worker_id=worker_id, lease_expires_at=lease_expires_at, started_at=attempt.started_at or _now())
            self._cells[cell.cell_id] = replace(cell, active_attempt_id=attempt_id)
            return True

    def heartbeat_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool:
        _validate_lease(worker_id, lease_seconds)
        with self._lock:
            attempt = self.get_attempt(attempt_id)
            cell = self.get_cell(attempt.cell_id)
            now = _now()
            captured_generation = (
                attempt.execution_generation
                if execution_generation is None
                else execution_generation
            )
            captured_experiment_id = experiment_id or cell.experiment_id
            if (
                captured_experiment_id != cell.experiment_id
                or captured_generation != attempt.execution_generation
                or not self._owns_execution_locked(
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
                or cell.active_attempt_id != attempt_id
                or attempt.execution_status is not ExecutionStatus.RUNNING
                or attempt.worker_id != worker_id
                or attempt.lease_expires_at is None
                or attempt.lease_expires_at <= now
            ):
                return False
            self._attempts[attempt_id] = replace(attempt, lease_expires_at=now + timedelta(seconds=lease_seconds))
            return True

    def abandon_attempt_claim(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool:
        with self._lock:
            attempt = self.get_attempt(attempt_id)
            cell = self.get_cell(attempt.cell_id)
            captured_experiment_id = experiment_id or cell.experiment_id
            captured_generation = (
                attempt.execution_generation
                if execution_generation is None
                else execution_generation
            )
            if (
                captured_experiment_id != cell.experiment_id
                or captured_generation != attempt.execution_generation
                or not self._owns_execution_locked(
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
            ):
                return False
            if (
                cell.active_attempt_id != attempt_id
                or attempt.execution_status is not ExecutionStatus.RUNNING
                or attempt.worker_id != worker_id
            ):
                return False
            recovered = replace(
                attempt,
                execution_status=ExecutionStatus.FAILED,
                result_disposition=ResultDisposition.INCONCLUSIVE,
                failure_category=FailureCategory.WORKER_LOST,
                selection_reason="heartbeat_lost",
                worker_id=None,
                lease_expires_at=None,
                completed_at=_now(),
            )
            self._attempts[attempt_id] = recovered
            self._execution_authority.active_attempts.pop(attempt_id, None)
            selected = self._attempts.get(cell.selected_attempt_id or "")
            self._cells[cell.cell_id] = replace(
                cell,
                active_attempt_id=None,
                selected_attempt_id=(
                    attempt_id
                    if not _is_valid_selection(selected)
                    else cell.selected_attempt_id
                ),
            )
            return True

    def complete_attempt(self, attempt_id: str, *, execution_status: ExecutionStatus, result_disposition: ResultDisposition, score: int | float | None = None, run_id: str | None = None, failure_category: FailureCategory | None = None) -> AttemptRecord:
        _validate_completion(execution_status, result_disposition, score)
        with self._lock:
            return self._complete_attempt_locked(
                attempt_id,
                execution_status=execution_status,
                result_disposition=result_disposition,
                score=score,
                run_id=run_id,
                failure_category=failure_category,
            )

    def complete_attempt_if_owned(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
        execution_status: ExecutionStatus,
        result_disposition: ResultDisposition,
        score: int | float | None = None,
        run_id: str | None = None,
        failure_category: FailureCategory | None = None,
    ) -> OwnedAttemptCompletion:
        _validate_completion(execution_status, result_disposition, score)
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        with self._lock:
            attempt = self.get_attempt(attempt_id)
            cell = self.get_cell(attempt.cell_id)
            owns_live_attempt = bool(
                attempt.execution_status is ExecutionStatus.RUNNING
                and cell.active_attempt_id == attempt_id
                and attempt.worker_id == worker_id
                and attempt.lease_expires_at is not None
                and attempt.lease_expires_at > _now()
            )
            if not owns_live_attempt:
                return OwnedAttemptCompletion(False, attempt)
            captured_generation = (
                attempt.execution_generation
                if execution_generation is None
                else execution_generation
            )
            captured_experiment_id = experiment_id or cell.experiment_id
            owns_execution = bool(
                captured_experiment_id == cell.experiment_id
                and captured_generation == attempt.execution_generation
                and self._owns_execution_locked(
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
            )
            completed = self._complete_attempt_locked(
                attempt_id,
                execution_status=execution_status,
                result_disposition=result_disposition,
                score=score,
                run_id=run_id,
                failure_category=failure_category,
                force_late=not owns_execution,
            )
            return OwnedAttemptCompletion(owns_execution, completed)

    def _complete_attempt_locked(
        self,
        attempt_id: str,
        *,
        execution_status: ExecutionStatus,
        result_disposition: ResultDisposition,
        score: int | float | None,
        run_id: str | None,
        failure_category: FailureCategory | None,
        force_late: bool = False,
    ) -> AttemptRecord:
        attempt = self.get_attempt(attempt_id)
        cell = self.get_cell(attempt.cell_id)
        owns_active = cell.active_attempt_id == attempt_id
        requested = _requested_terminal_evidence(
            attempt=attempt,
            execution_status=execution_status,
            result_disposition=result_disposition,
            score=score,
            run_id=run_id,
            failure_category=failure_category,
        )
        if _is_terminal(attempt):
            if _terminal_evidence(attempt) == requested:
                return attempt
            if attempt.failure_category is not FailureCategory.WORKER_LOST:
                raise ValueError(
                    "attempt is already terminal with different terminal evidence"
                )
        late_completion = force_late or (
            not owns_active
            and (
                attempt.execution_status is not ExecutionStatus.QUEUED
                or cell.latest_attempt_id != attempt_id
                or cell.active_attempt_id is not None
            )
        )
        completed = replace(
            attempt,
            execution_status=execution_status,
            result_disposition=result_disposition,
            score=score,
            run_id=run_id or attempt.run_id,
            failure_category=failure_category,
            completed_at=_now(),
            late_completion=late_completion,
            worker_id=None,
            lease_expires_at=None,
        )
        self._attempts[attempt_id] = completed
        self._execution_authority.active_attempts.pop(attempt_id, None)
        if late_completion:
            if owns_active:
                self._cells[cell.cell_id] = replace(cell, active_attempt_id=None)
            return completed
        selected = self._attempts.get(cell.selected_attempt_id or "")
        should_promote_valid = (
            _is_valid_selection(completed)
            and completed.promotion_policy != "diagnostic_only"
        )
        should_promote_failure = not _is_valid_selection(
            selected
        ) and not _is_valid_selection(completed)
        should_promote = should_promote_valid or should_promote_failure
        selection_reason = (
            "valid_completion"
            if should_promote_valid
            else "latest_failure"
            if should_promote_failure
            else None
        )
        if selection_reason is not None:
            completed = replace(completed, selection_reason=selection_reason)
            self._attempts[attempt_id] = completed
        self._cells[cell.cell_id] = replace(
            cell,
            active_attempt_id=None if owns_active else cell.active_attempt_id,
            selected_attempt_id=(
                completed.attempt_id
                if should_promote
                else cell.selected_attempt_id
            ),
        )
        return completed

    def _owns_execution_locked(
        self,
        experiment_id: str,
        *,
        execution_generation: int,
        execution_owner_id: str | None,
    ) -> bool:
        if execution_generation == 0 and not execution_owner_id:
            return True
        experiment = self._execution_authority.experiments.get(experiment_id)
        return bool(
            experiment is not None
            and experiment.get("execution_generation", 0) == execution_generation
            and experiment.get("execution_owner_id", "") == execution_owner_id
            and experiment.get("status") in {"running", "rejudging"}
        )

    def _require_owned_execution_locked(
        self,
        experiment_id: str,
        *,
        execution_generation: int,
        execution_owner_id: str | None,
    ) -> None:
        if not self._owns_execution_locked(
            experiment_id,
            execution_generation=execution_generation,
            execution_owner_id=execution_owner_id,
        ):
            raise ValueError("attempt execution ownership does not match experiment")

    def get_cell(self, cell_id: str) -> CellRecord:
        with self._lock:
            try:
                return self._cells[cell_id]
            except KeyError:
                raise KeyError(f"cell '{cell_id}' does not exist") from None

    def get_attempt(self, attempt_id: str) -> AttemptRecord:
        with self._lock:
            try:
                return self._attempts[attempt_id]
            except KeyError:
                raise KeyError(f"attempt '{attempt_id}' does not exist") from None

    def list_attempts(self, cell_id: str) -> tuple[AttemptRecord, ...]:
        with self._lock:
            return tuple(sorted((a for a in self._attempts.values() if a.cell_id == cell_id), key=lambda attempt: attempt.attempt_number))

    def has_active_attempts(
        self,
        experiment_id: str,
        *,
        execution_generation: int,
    ) -> bool:
        _validate_execution_generation(execution_generation)
        with self._lock:
            cell_ids = {
                cell.cell_id
                for cell in self._cells.values()
                if cell.experiment_id == experiment_id
            }
            return any(
                attempt.cell_id in cell_ids
                and attempt.execution_generation == execution_generation
                and attempt.execution_status
                in {ExecutionStatus.QUEUED, ExecutionStatus.RUNNING}
                for attempt in self._attempts.values()
            )

    def retryable_cells(self, experiment_id: str) -> tuple[CellRecord, ...]:
        with self._lock:
            cells = []
            for cell in self._cells.values():
                if cell.experiment_id != experiment_id:
                    continue
                if cell.active_attempt_id is not None:
                    continue
                if _is_retryable_selection(self._attempts.get(cell.selected_attempt_id or "")):
                    cells.append(cell)
            return tuple(sorted(cells, key=lambda cell: cell.cell_id or ""))

    def recover_expired_attempt(
        self,
        cell_id: str,
        *,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> CellRecord:
        with self._lock:
            cell = self.get_cell(cell_id)
            fenced = any(
                value is not None
                for value in (
                    experiment_id,
                    execution_generation,
                    execution_owner_id,
                )
            )
            if fenced:
                if execution_generation is None:
                    raise ValueError(
                        "execution_generation is required for owned lease recovery"
                    )
                active = self._attempts.get(cell.active_attempt_id or "")
                if (
                    experiment_id != cell.experiment_id
                    or (
                        active is not None
                        and active.execution_generation != execution_generation
                    )
                    or not self._owns_execution_locked(
                        cell.experiment_id,
                        execution_generation=execution_generation,
                        execution_owner_id=execution_owner_id,
                    )
                ):
                    return cell
            return self._recover_expired_active(cell)

    def count_attempts(self, experiment_id: str, *, retries_only: bool = False) -> int:
        with self._lock:
            cell_ids = {
                cell.cell_id for cell in self._cells.values()
                if cell.experiment_id == experiment_id
            }
            return sum(
                1 for attempt in self._attempts.values()
                if attempt.cell_id in cell_ids
                and (not retries_only or attempt.retry_kind != "initial")
            )

    def create_retry_batch(
        self,
        experiment_id: str,
        *,
        action: str,
        completed_policy: str = "exclude",
        cell_ids: tuple[str, ...],
        requested_by: str,
        request_id: str | None = None,
        selection_metadata: Mapping[str, object] | None = None,
    ) -> RetryBatchRecord:
        selection_metadata = selection_metadata or {}
        with self._lock:
            unique_cell_ids = tuple(dict.fromkeys(cell_ids))
            for cell_id in unique_cell_ids:
                if self.get_cell(cell_id).experiment_id != experiment_id:
                    raise ValueError("cell does not belong to experiment")
            candidate = RetryBatchRecord(
                _generated_id(), experiment_id, request_id, action,
                completed_policy, requested_by, unique_cell_ids,
                selection_metadata=selection_metadata,
                created_at=_now(),
            )
            if request_id is not None:
                existing_id = self._batch_requests.get((experiment_id, request_id))
                if existing_id is not None:
                    existing = self._batch_with_members(self._batches[existing_id])
                    if _retry_batch_request_content(existing) != _retry_batch_request_content(candidate):
                        raise ValueError("request_id already exists for a different retry request")
                    return replace(existing, was_created=False)
            batch = candidate
            self._batches[batch.retry_batch_id] = batch
            if request_id is not None:
                self._batch_requests[(experiment_id, request_id)] = batch.retry_batch_id
            for ordinal, cell_id in enumerate(unique_cell_ids):
                self._batch_members[(batch.retry_batch_id, cell_id)] = RetryBatchCellRecord(
                    batch.retry_batch_id,
                    cell_id,
                    ordinal=ordinal,
                    created_at=_now(),
                )
            return self._batch_with_members(batch)

    def get_retry_batch(self, retry_batch_id: str) -> RetryBatchRecord:
        with self._lock:
            try:
                return self._batch_with_members(self._batches[retry_batch_id])
            except KeyError:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist") from None

    def get_retry_batch_by_request(self, experiment_id: str, request_id: str) -> RetryBatchRecord | None:
        with self._lock:
            retry_batch_id = self._batch_requests.get((experiment_id, request_id))
            return (
                self._batch_with_members(self._batches[retry_batch_id])
                if retry_batch_id is not None
                else None
            )

    def list_retry_batches(
        self,
        experiment_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> tuple[RetryBatchRecord, ...]:
        with self._lock:
            allowed = set(statuses) if statuses is not None else None
            return tuple(
                self._batch_with_members(batch)
                for batch in sorted(
                    self._batches.values(),
                    key=lambda item: (item.created_at or datetime.min.replace(tzinfo=UTC), item.retry_batch_id),
                )
                if batch.experiment_id == experiment_id
                and (allowed is None or batch.status in allowed)
            )

    def update_retry_batch(self, retry_batch_id: str, *, status: str) -> RetryBatchRecord:
        with self._lock:
            batch = self.get_retry_batch(retry_batch_id)
            if batch.status in {"completed", "failed", "partial"}:
                if batch.status != status:
                    raise ValueError("retry batch is already terminal with different status")
                return batch
            now = _now()
            updated = replace(
                self._batches[retry_batch_id],
                status=status,
                started_at=(batch.started_at or now) if status == "running" else batch.started_at,
                completed_at=now if status in {"completed", "failed", "partial"} else None,
            )
            self._batches[retry_batch_id] = updated
            return self._batch_with_members(updated)

    def bind_retry_batch_generation(
        self,
        retry_batch_id: str,
        execution_generation: int,
    ) -> RetryBatchRecord:
        _validate_positive_execution_generation(execution_generation)
        with self._lock:
            batch = self.get_retry_batch(retry_batch_id)
            if batch.execution_generation is not None:
                if batch.execution_generation != execution_generation:
                    raise ValueError(
                        "retry batch is already bound to a different execution generation"
                    )
            self._require_owned_execution_locked(
                batch.experiment_id,
                execution_generation=execution_generation,
                execution_owner_id=batch.retry_batch_id,
            )
            for member in batch.members:
                if member.attempt_id is None:
                    continue
                attempt = self.get_attempt(member.attempt_id)
                if attempt.execution_generation != execution_generation:
                    raise ValueError(
                        "associated attempt execution generation does not match retry batch"
                    )
            if batch.execution_generation is not None:
                return batch
            updated = replace(batch, execution_generation=execution_generation)
            self._batches[retry_batch_id] = replace(updated, members=())
            return updated

    def bind_retry_batch_attempt(self, retry_batch_id: str, cell_id: str, attempt_id: str) -> RetryBatchCellRecord:
        with self._lock:
            batch = self._batches[retry_batch_id]
            if batch.execution_generation is None:
                raise ValueError(
                    "retry batch execution ownership is not durably bound"
                )
            self._require_owned_execution_locked(
                batch.experiment_id,
                execution_generation=batch.execution_generation,
                execution_owner_id=batch.retry_batch_id,
            )
            member = self._get_batch_member(retry_batch_id, cell_id)
            attempt = self.get_attempt(attempt_id)
            if attempt.cell_id != cell_id:
                raise ValueError("attempt does not belong to retry batch cell")
            if (
                batch.execution_generation is not None
                and attempt.execution_generation != batch.execution_generation
            ):
                raise ValueError("attempt execution generation does not match retry batch")
            if member.attempt_id is not None and member.attempt_id != attempt_id:
                raise ValueError("retry batch cell already owns a different attempt")
            updated = replace(member, attempt_id=attempt_id, status="running")
            self._batch_members[(retry_batch_id, cell_id)] = updated
            return updated

    def create_and_bind_retry_attempt(
        self,
        retry_batch_id: str,
        cell_id: str,
        *,
        retry_kind: str,
        spec_hash: str,
        execution_generation: int = 0,
        checkpoint_id: str | None = None,
        promotion_policy: str = "replace_on_valid_completion",
        recoverable_cancellation: bool = False,
        max_attempts: int | None = None,
        max_retry_attempts_per_experiment: int | None = None,
    ) -> AttemptRecord:
        _validate_positive_execution_generation(execution_generation)
        with self._lock:
            member = self._get_batch_member(retry_batch_id, cell_id)
            batch = self._batches[retry_batch_id]
            if batch.action != retry_kind:
                raise ValueError("retry batch member belongs to a different retry attempt")
            if batch.execution_generation is None:
                raise ValueError(
                    "retry batch execution ownership is not durably bound"
                )
            if batch.execution_generation != execution_generation:
                raise ValueError("attempt execution generation does not match retry batch")
            if member.attempt_id is not None:
                attempt = self.get_attempt(member.attempt_id)
                if _bound_attempt_content(attempt) != (
                    cell_id,
                    retry_kind,
                    spec_hash,
                    checkpoint_id,
                    promotion_policy,
                    recoverable_cancellation,
                    execution_generation,
                ):
                    raise ValueError("retry batch member owns a different retry attempt")
                return attempt
            attempt = self.create_attempt(
                cell_id,
                retry_kind=retry_kind,
                spec_hash=spec_hash,
                execution_generation=execution_generation,
                execution_owner_id=batch.retry_batch_id,
                checkpoint_id=checkpoint_id,
                promotion_policy=promotion_policy,
                recoverable_cancellation=recoverable_cancellation,
                max_attempts=max_attempts,
                max_retry_attempts_per_experiment=max_retry_attempts_per_experiment,
            )
            self._batch_members[(retry_batch_id, cell_id)] = replace(
                member,
                attempt_id=attempt.attempt_id,
                status="queued",
            )
            return attempt

    def complete_retry_batch_cell(self, retry_batch_id: str, cell_id: str, *, status: str, error_category: str | None) -> RetryBatchCellRecord:
        with self._lock:
            member = self._get_batch_member(retry_batch_id, cell_id)
            if member.status in {"completed", "failed", "skipped"}:
                if (member.status, member.error_category) != (status, error_category):
                    raise ValueError("retry batch cell is already terminal with different terminal evidence")
                return member
            updated = replace(
                member,
                status=status,
                error_category=error_category,
                completed_at=_now(),
            )
            self._batch_members[(retry_batch_id, cell_id)] = updated
            return updated

    def _get_batch_member(self, retry_batch_id: str, cell_id: str) -> RetryBatchCellRecord:
        if retry_batch_id not in self._batches:
            raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
        try:
            return self._batch_members[(retry_batch_id, cell_id)]
        except KeyError:
            raise KeyError("cell does not belong to retry batch") from None

    def _batch_with_members(self, batch: RetryBatchRecord) -> RetryBatchRecord:
        members = tuple(
            self._batch_members[(batch.retry_batch_id, cell_id)]
            for cell_id in batch.cell_ids
        )
        return replace(batch, members=members)

    def _recover_expired_active(self, cell: CellRecord) -> CellRecord:
        if cell.active_attempt_id is None:
            return cell
        active = self.get_attempt(cell.active_attempt_id)
        if active.lease_expires_at is not None and active.lease_expires_at <= _now():
            self._attempts[active.attempt_id] = replace(active, execution_status=ExecutionStatus.FAILED, result_disposition=ResultDisposition.INCONCLUSIVE, failure_category=FailureCategory.WORKER_LOST, selection_reason="lease_expired", worker_id=None, lease_expires_at=None, completed_at=_now())
            self._execution_authority.active_attempts.pop(active.attempt_id, None)
            selected = self._attempts.get(cell.selected_attempt_id or "")
            cell = replace(
                cell,
                active_attempt_id=None,
                selected_attempt_id=(
                    active.attempt_id
                    if not _is_valid_selection(selected)
                    else cell.selected_attempt_id
                ),
            )
            self._cells[cell.cell_id] = cell
        return cell


def _validate_lease(worker_id: str, lease_seconds: int) -> None:
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    if lease_seconds < 0:
        raise ValueError("lease_seconds must not be negative")


def _validate_execution_generation(execution_generation: int) -> None:
    if type(execution_generation) is not int or execution_generation < 0:
        raise ValueError("execution_generation must be a non-negative integer")


def _validate_positive_execution_generation(execution_generation: int) -> None:
    if type(execution_generation) is not int or execution_generation < 1:
        raise ValueError("execution_generation must be a positive integer")


def _bound_attempt_content(attempt: AttemptRecord) -> tuple[object, ...]:
    return (
        attempt.cell_id,
        attempt.retry_kind,
        attempt.spec_hash,
        attempt.checkpoint_id,
        attempt.promotion_policy,
        attempt.recoverable_cancellation,
        attempt.execution_generation,
    )


def _config_content(record: TechniqueConfigRecord) -> tuple[object, ...]:
    return (record.experiment_id, record.technique_id, record.technique_version, record.implementation_id, record.preset, dict(record.parameters), dict(record.resolved_spec), record.spec_hash)


def _retry_batch_request_content(record: RetryBatchRecord) -> tuple[object, ...]:
    selection_mode = record.selection_metadata.get("selection_mode")
    immutable_selection = (
        None
        if selection_mode in {"automatic", "explicit"}
        else record.cell_ids
    )
    return (
        record.experiment_id,
        record.action,
        record.completed_policy,
        record.requested_by,
        immutable_selection,
        _json(record.selection_metadata),
    )


def _planned_cell_content(record: CellRecord) -> tuple[object, ...]:
    return (
        record.experiment_id,
        record.technique_config_id,
        record.technique_key,
        record.prompt_id,
        record.repeat_index,
    )


def _validate_planned_cell(record: CellRecord) -> None:
    if any(
        value is not None
        for value in (
            record.selected_attempt_id,
            record.latest_attempt_id,
            record.active_attempt_id,
        )
    ):
        raise ValueError("new cells must not include repository-owned attempt references")


class PostgreSQLExperimentAttemptPersistence(ExperimentAttemptPersistence):
    """PostgreSQL implementation. A caller must explicitly provide its test DSN."""

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("an explicit disposable PostgreSQL DSN is required")
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise ImportError("PostgreSQLExperimentAttemptPersistence requires psycopg") from None
        self._dsn = dsn
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def upsert_technique_config(self, record: TechniqueConfigRecord) -> TechniqueConfigRecord:
        with self._transaction() as connection:
            _lock_identity(
                connection,
                "config:" + (
                    record.technique_config_id
                    or _json({"content": _config_content(record)})
                ),
            )
            if record.technique_config_id is None:
                row = connection.execute(
                    "SELECT technique_config_id, experiment_id, technique_id, technique_version, implementation_id, preset, parameters, resolved_spec, spec_hash, created_at "
                    "FROM experiment_technique_configs WHERE experiment_id = %s AND technique_id = %s AND technique_version = %s "
                    "AND implementation_id = %s AND preset IS NOT DISTINCT FROM %s AND parameters = %s::jsonb "
                    "AND resolved_spec = %s::jsonb AND spec_hash = %s FOR UPDATE",
                    (record.experiment_id, record.technique_id, record.technique_version, record.implementation_id, record.preset, _json(record.parameters), _json(record.resolved_spec), record.spec_hash),
                ).fetchone()
                if row is not None:
                    return replace(record, technique_config_id=_config_from_row(row).technique_config_id)
                record = replace(
                    record,
                    technique_config_id=(
                        canonical_technique_config_id(
                            record.experiment_id,
                            record.resolved_spec,
                        )
                        if is_complete_resolved_spec(record.resolved_spec)
                        else _generated_id()
                    ),
                )
            row = connection.execute("SELECT technique_config_id, experiment_id, technique_id, technique_version, implementation_id, preset, parameters, resolved_spec, spec_hash, created_at FROM experiment_technique_configs WHERE technique_config_id = %s FOR UPDATE", (record.technique_config_id,)).fetchone()
            if row is not None:
                existing = _config_from_row(row)
                if _config_content(existing) != _config_content(record):
                    raise ValueError("technique configuration already exists with different content")
                return record
            connection.execute("INSERT INTO experiment_technique_configs (technique_config_id, experiment_id, technique_id, technique_version, implementation_id, preset, parameters, resolved_spec, spec_hash) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)", (record.technique_config_id, record.experiment_id, record.technique_id, record.technique_version, record.implementation_id, record.preset, _json(record.parameters), _json(record.resolved_spec), record.spec_hash))
            return record

    def create_cell(self, record: CellRecord) -> CellRecord:
        _validate_planned_cell(record)
        with self._transaction() as connection:
            _lock_identity(
                connection,
                "cell:" + _json(
                    {
                        "experiment_id": record.experiment_id,
                        "technique_config_id": record.technique_config_id,
                        "prompt_id": record.prompt_id,
                        "repeat_index": record.repeat_index,
                    }
                ),
            )
            if record.cell_id is not None:
                _lock_identity(connection, "cell-id:" + record.cell_id)
            config = connection.execute("SELECT experiment_id FROM experiment_technique_configs WHERE technique_config_id = %s FOR UPDATE", (record.technique_config_id,)).fetchone()
            if config is None or config[0] != record.experiment_id:
                raise ValueError("technique configuration does not belong to experiment")
            row = connection.execute(_CELL_SELECT + " WHERE experiment_id = %s AND technique_config_id = %s AND prompt_id = %s AND repeat_index = %s FOR UPDATE", (record.experiment_id, record.technique_config_id, record.prompt_id, record.repeat_index)).fetchone()
            if row is not None:
                existing = _cell_from_row(row)
                if record.cell_id not in (None, existing.cell_id) or _planned_cell_content(existing) != _planned_cell_content(record):
                    raise ValueError("cell identity already exists with different content")
                return existing
            record = replace(record, cell_id=record.cell_id or _generated_id())
            by_id = connection.execute(_CELL_SELECT + " WHERE cell_id = %s FOR UPDATE", (record.cell_id,)).fetchone()
            if by_id is not None:
                existing = _cell_from_row(by_id)
                if _planned_cell_content(existing) != _planned_cell_content(record):
                    raise ValueError("cell ID already exists with different content")
                return record
            connection.execute("INSERT INTO experiment_cells (cell_id, experiment_id, technique_key, technique_config_id, prompt_id, repeat_index) VALUES (%s, %s, %s, %s, %s, %s)", (record.cell_id, record.experiment_id, record.technique_key, record.technique_config_id, record.prompt_id, record.repeat_index))
            return record

    def create_attempt(
        self,
        cell_id: str,
        *,
        retry_kind: str,
        spec_hash: str,
        execution_generation: int = 0,
        execution_owner_id: str | None = None,
        parent_attempt_id: str | None = None,
        checkpoint_id: str | None = None,
        promotion_policy: str = "replace_on_valid_completion",
        recoverable_cancellation: bool = False,
        max_attempts: int | None = None,
        max_retry_attempts_per_experiment: int | None = None,
    ) -> AttemptRecord:
        _validate_execution_generation(execution_generation)
        if not retry_kind:
            raise ValueError("retry_kind must not be empty")
        _validate_positive_limit(max_attempts, name="max_attempts_per_cell")
        _validate_positive_limit(
            max_retry_attempts_per_experiment,
            name="max_retry_attempts_per_experiment",
        )
        with self._transaction() as connection:
            experiment_row = connection.execute(
                "SELECT experiment_id FROM experiment_cells WHERE cell_id = %s",
                (cell_id,),
            ).fetchone()
            if experiment_row is None:
                raise KeyError(f"cell '{cell_id}' does not exist")
            if not _lock_and_validate_owned_experiment(
                connection,
                str(experiment_row[0]),
                execution_generation=execution_generation,
                execution_owner_id=execution_owner_id,
            ):
                raise ValueError(
                    "attempt execution ownership does not match experiment"
                )
            row = connection.execute(_CELL_SELECT + " WHERE cell_id = %s FOR UPDATE", (cell_id,)).fetchone()
            if row is None:
                raise KeyError(f"cell '{cell_id}' does not exist")
            cell = _cell_from_row(row)
            cell = self._recover_expired_active(connection, cell)
            _lock_identity(connection, "retry-budget:" + cell.experiment_id)
            config_hash = connection.execute("SELECT spec_hash FROM experiment_technique_configs WHERE technique_config_id = %s", (cell.technique_config_id,)).fetchone()[0]
            if config_hash != spec_hash:
                raise ValueError("attempt spec hash must match the cell's pinned spec hash")
            claimable = connection.execute(
                "SELECT attempt_id FROM experiment_cell_attempts "
                "WHERE cell_id = %s AND execution_status IN ('queued', 'running') "
                "LIMIT 1 FOR UPDATE",
                (cell_id,),
            ).fetchone()
            if claimable is not None:
                raise ValueError("cell already has an active or claimable attempt")
            parent_attempt_id = parent_attempt_id or cell.latest_attempt_id
            if parent_attempt_id is not None:
                parent = connection.execute("SELECT cell_id FROM experiment_cell_attempts WHERE attempt_id = %s", (parent_attempt_id,)).fetchone()
                if parent is None or parent[0] != cell_id:
                    raise ValueError("parent attempt must belong to the same cell")
            number = connection.execute("SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM experiment_cell_attempts WHERE cell_id = %s", (cell_id,)).fetchone()[0]
            if max_attempts is not None and number > max_attempts:
                raise ValueError("max_attempts_per_cell retry limit exceeded")
            if retry_kind != "initial" and max_retry_attempts_per_experiment is not None:
                retry_count = connection.execute(
                    "SELECT COUNT(*) FROM experiment_cell_attempts AS attempt "
                    "JOIN experiment_cells AS retry_cell ON retry_cell.cell_id = attempt.cell_id "
                    "WHERE retry_cell.experiment_id = %s AND attempt.retry_kind <> 'initial'",
                    (cell.experiment_id,),
                ).fetchone()[0]
                if retry_count >= max_retry_attempts_per_experiment:
                    raise ValueError(
                        "max_retry_attempts_per_experiment retry limit exceeded"
                    )
            attempt = AttemptRecord(
                _generated_id(), cell_id, number, parent_attempt_id, retry_kind,
                spec_hash, ExecutionStatus.QUEUED, ResultDisposition.NOT_TESTED,
                execution_generation=execution_generation,
                checkpoint_id=checkpoint_id,
                promotion_policy=promotion_policy,
                recoverable_cancellation=recoverable_cancellation,
                created_at=_now(),
            )
            attempt_row = connection.execute(
                "INSERT INTO experiment_cell_attempts "
                "(attempt_id, cell_id, attempt_number, parent_attempt_id, retry_kind, "
                "spec_hash, execution_status, result_disposition, execution_generation, checkpoint_id, "
                "promotion_policy, recoverable_cancellation) "
                "VALUES (%s, %s, %s, %s, %s, %s, 'queued', 'not_tested', %s, %s, %s, %s) "
                "RETURNING " + _ATTEMPT_COLUMNS,
                (
                    attempt.attempt_id, cell_id, number, parent_attempt_id,
                    retry_kind, spec_hash, execution_generation, checkpoint_id, promotion_policy,
                    recoverable_cancellation,
                ),
            ).fetchone()
            attempt = _attempt_from_row(attempt_row)
            connection.execute("UPDATE experiment_cells SET latest_attempt_id = %s WHERE cell_id = %s", (attempt.attempt_id, cell_id))
            return attempt

    def claim_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool:
        _validate_lease(worker_id, lease_seconds)
        with self._transaction() as connection:
            identity = connection.execute(
                "SELECT cell.experiment_id, attempt.execution_generation "
                "FROM experiment_cell_attempts AS attempt "
                "JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id "
                "WHERE attempt.attempt_id = %s",
                (attempt_id,),
            ).fetchone()
            if identity is None:
                raise KeyError(f"attempt '{attempt_id}' does not exist")
            captured_experiment_id = experiment_id or str(identity[0])
            captured_generation = (
                int(identity[1])
                if execution_generation is None
                else execution_generation
            )
            if (
                captured_experiment_id != str(identity[0])
                or captured_generation != int(identity[1])
                or not _lock_and_validate_owned_experiment(
                    connection,
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
            ):
                return False
            cell, attempt = self._locked_cell_attempt(connection, attempt_id)
            if attempt.execution_status is not ExecutionStatus.QUEUED:
                return False
            cell = self._recover_expired_active(connection, cell)
            if cell.active_attempt_id is not None:
                return False
            row = connection.execute("UPDATE experiment_cell_attempts SET execution_status = 'running', worker_id = %s, lease_expires_at = now() + (%s * interval '1 second'), started_at = COALESCE(started_at, now()) WHERE attempt_id = %s AND execution_status = 'queued' RETURNING " + _ATTEMPT_COLUMNS, (worker_id, lease_seconds, attempt_id)).fetchone()
            if row is None:
                return False
            connection.execute("UPDATE experiment_cells SET active_attempt_id = %s WHERE cell_id = %s", (attempt_id, cell.cell_id))
            return True

    def heartbeat_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool:
        _validate_lease(worker_id, lease_seconds)
        with self._transaction() as connection:
            identity = connection.execute(
                "SELECT cell.experiment_id, attempt.execution_generation "
                "FROM experiment_cell_attempts AS attempt "
                "JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id "
                "WHERE attempt.attempt_id = %s",
                (attempt_id,),
            ).fetchone()
            if identity is None:
                raise KeyError(f"attempt '{attempt_id}' does not exist")
            captured_experiment_id = experiment_id or str(identity[0])
            captured_generation = (
                int(identity[1])
                if execution_generation is None
                else execution_generation
            )
            if (
                captured_experiment_id != str(identity[0])
                or captured_generation != int(identity[1])
                or not _lock_and_validate_owned_experiment(
                    connection,
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
            ):
                return False
            cell, attempt = self._locked_cell_attempt(connection, attempt_id)
            if cell.active_attempt_id != attempt_id or attempt.execution_status is not ExecutionStatus.RUNNING or attempt.worker_id != worker_id:
                return False
            return connection.execute(
                "UPDATE experiment_cell_attempts SET lease_expires_at = now() + (%s * interval '1 second') "
                "WHERE attempt_id = %s AND worker_id = %s AND execution_status = 'running' "
                "AND lease_expires_at > now()",
                (lease_seconds, attempt_id, worker_id),
            ).rowcount == 1

    def abandon_attempt_claim(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> bool:
        with self._transaction() as connection:
            identity = connection.execute(
                "SELECT cell.experiment_id, attempt.execution_generation "
                "FROM experiment_cell_attempts AS attempt "
                "JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id "
                "WHERE attempt.attempt_id = %s",
                (attempt_id,),
            ).fetchone()
            if identity is None:
                raise KeyError(f"attempt '{attempt_id}' does not exist")
            captured_experiment_id = experiment_id or str(identity[0])
            captured_generation = (
                int(identity[1])
                if execution_generation is None
                else execution_generation
            )
            if (
                captured_experiment_id != str(identity[0])
                or captured_generation != int(identity[1])
                or not _lock_and_validate_owned_experiment(
                    connection,
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
            ):
                return False
            cell, attempt = self._locked_cell_attempt(connection, attempt_id)
            if (
                cell.active_attempt_id != attempt_id
                or attempt.execution_status is not ExecutionStatus.RUNNING
                or attempt.worker_id != worker_id
            ):
                return False
            recovered = connection.execute(
                "UPDATE experiment_cell_attempts SET execution_status = 'failed', "
                "result_disposition = 'inconclusive', failure_category = 'worker_lost', "
                "selection_reason = 'heartbeat_lost', official_judge_status = 'failed', "
                "worker_id = NULL, lease_expires_at = NULL, completed_at = now() "
                "WHERE attempt_id = %s AND execution_status = 'running' "
                "AND worker_id = %s RETURNING attempt_id",
                (attempt_id, worker_id),
            ).fetchone()
            if recovered is None:
                return False
            selected = self._selected_attempt(connection, cell.selected_attempt_id)
            selected_attempt_id = (
                attempt_id
                if not _is_valid_selection(selected)
                else cell.selected_attempt_id
            )
            connection.execute(
                "UPDATE experiment_cells SET active_attempt_id = NULL, "
                "selected_attempt_id = %s WHERE cell_id = %s "
                "AND active_attempt_id = %s",
                (selected_attempt_id, cell.cell_id, attempt_id),
            )
            return True

    def complete_attempt(self, attempt_id: str, *, execution_status: ExecutionStatus, result_disposition: ResultDisposition, score: int | float | None = None, run_id: str | None = None, failure_category: FailureCategory | None = None) -> AttemptRecord:
        _validate_completion(execution_status, result_disposition, score)
        with self._transaction() as connection:
            cell, attempt = self._locked_cell_attempt(connection, attempt_id)
            owns_active = cell.active_attempt_id == attempt_id
            requested = _requested_terminal_evidence(
                attempt=attempt,
                execution_status=execution_status,
                result_disposition=result_disposition,
                score=score,
                run_id=run_id,
                failure_category=failure_category,
            )
            if _is_terminal(attempt):
                if _terminal_evidence(attempt) == requested:
                    return attempt
                if attempt.failure_category is not FailureCategory.WORKER_LOST:
                    raise ValueError("attempt is already terminal with different terminal evidence")
            late_completion = (
                not owns_active
                and (
                    attempt.execution_status is not ExecutionStatus.QUEUED
                    or cell.latest_attempt_id != attempt_id
                    or cell.active_attempt_id is not None
                )
            )
            row = connection.execute(
                "UPDATE experiment_cell_attempts SET execution_status = %s, "
                "result_disposition = %s, score = %s, run_id = COALESCE(%s, run_id), "
                "failure_category = %s, completed_at = now(), worker_id = NULL, "
                "lease_expires_at = NULL, late_completion = %s, "
                "official_judge_status = CASE WHEN %s = 'tested' THEN 'succeeded' ELSE 'failed' END "
                "WHERE attempt_id = %s RETURNING " + _ATTEMPT_COLUMNS,
                (
                    execution_status.value, result_disposition.value, score, run_id,
                    failure_category.value if failure_category else None,
                    late_completion, result_disposition.value, attempt_id,
                ),
            ).fetchone()
            completed = _attempt_from_row(row)
            if late_completion:
                return completed
            selected = self._selected_attempt(connection, cell.selected_attempt_id)
            should_promote_valid = (
                _is_valid_selection(completed)
                and completed.promotion_policy != "diagnostic_only"
            )
            should_promote_failure = not _is_valid_selection(selected) and not _is_valid_selection(completed)
            should_promote = should_promote_valid or should_promote_failure
            selection_reason = (
                "valid_completion"
                if should_promote_valid
                else "latest_failure"
                if should_promote_failure
                else None
            )
            if selection_reason is not None:
                row = connection.execute(
                    "UPDATE experiment_cell_attempts SET selection_reason = %s "
                    "WHERE attempt_id = %s RETURNING " + _ATTEMPT_COLUMNS,
                    (selection_reason, attempt_id),
                ).fetchone()
                completed = _attempt_from_row(row)
            connection.execute(
                "UPDATE experiment_cells SET active_attempt_id = "
                "CASE WHEN active_attempt_id = %s THEN NULL ELSE active_attempt_id END, "
                "selected_attempt_id = CASE WHEN %s THEN %s ELSE selected_attempt_id END "
                "WHERE cell_id = %s",
                (attempt_id, should_promote, attempt_id, cell.cell_id),
            )
            return completed

    def complete_attempt_if_owned(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
        execution_status: ExecutionStatus,
        result_disposition: ResultDisposition,
        score: int | float | None = None,
        run_id: str | None = None,
        failure_category: FailureCategory | None = None,
    ) -> OwnedAttemptCompletion:
        _validate_completion(execution_status, result_disposition, score)
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        with self._transaction() as connection:
            identity = connection.execute(
                "SELECT cell.experiment_id, attempt.execution_generation "
                "FROM experiment_cell_attempts AS attempt "
                "JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id "
                "WHERE attempt.attempt_id = %s",
                (attempt_id,),
            ).fetchone()
            if identity is None:
                raise KeyError(f"attempt '{attempt_id}' does not exist")
            captured_experiment_id = experiment_id or str(identity[0])
            captured_generation = (
                int(identity[1])
                if execution_generation is None
                else execution_generation
            )
            captured_tuple_matches_attempt = bool(
                captured_experiment_id == str(identity[0])
                and captured_generation == int(identity[1])
            )
            owns_execution = bool(
                captured_tuple_matches_attempt
                and _lock_and_validate_owned_experiment(
                    connection,
                    captured_experiment_id,
                    execution_generation=captured_generation,
                    execution_owner_id=execution_owner_id,
                )
            )
            if not captured_tuple_matches_attempt:
                return OwnedAttemptCompletion(False, self.get_attempt(attempt_id))
            cell, attempt = self._locked_cell_attempt(connection, attempt_id)
            lease_is_live = bool(
                attempt.lease_expires_at is not None
                and connection.execute(
                    "SELECT lease_expires_at > now() "
                    "FROM experiment_cell_attempts WHERE attempt_id = %s",
                    (attempt_id,),
                ).fetchone()[0]
            )
            if not (
                attempt.execution_status is ExecutionStatus.RUNNING
                and cell.active_attempt_id == attempt_id
                and attempt.worker_id == worker_id
                and lease_is_live
            ):
                return OwnedAttemptCompletion(False, attempt)

            row = connection.execute(
                "UPDATE experiment_cell_attempts SET execution_status = %s, "
                "result_disposition = %s, score = %s, "
                "run_id = COALESCE(%s, run_id), failure_category = %s, "
                "completed_at = now(), worker_id = NULL, lease_expires_at = NULL, "
                "late_completion = %s, "
                "official_judge_status = CASE WHEN %s = 'tested' "
                "THEN 'succeeded' ELSE 'failed' END "
                "WHERE attempt_id = %s AND execution_status = 'running' "
                "AND worker_id = %s AND lease_expires_at > now() RETURNING "
                + _ATTEMPT_COLUMNS,
                (
                    execution_status.value,
                    result_disposition.value,
                    score,
                    run_id,
                    failure_category.value if failure_category else None,
                    not owns_execution,
                    result_disposition.value,
                    attempt_id,
                    worker_id,
                ),
            ).fetchone()
            if row is None:
                observed = self._locked_cell_attempt(connection, attempt_id)[1]
                return OwnedAttemptCompletion(False, observed)
            completed = _attempt_from_row(row)
            if not owns_execution:
                connection.execute(
                    "UPDATE experiment_cells SET active_attempt_id = NULL "
                    "WHERE cell_id = %s AND active_attempt_id = %s",
                    (cell.cell_id, attempt_id),
                )
                return OwnedAttemptCompletion(False, completed)
            selected = self._selected_attempt(connection, cell.selected_attempt_id)
            should_promote_valid = (
                _is_valid_selection(completed)
                and completed.promotion_policy != "diagnostic_only"
            )
            should_promote_failure = not _is_valid_selection(
                selected
            ) and not _is_valid_selection(completed)
            should_promote = should_promote_valid or should_promote_failure
            selection_reason = (
                "valid_completion"
                if should_promote_valid
                else "latest_failure"
                if should_promote_failure
                else None
            )
            if selection_reason is not None:
                row = connection.execute(
                    "UPDATE experiment_cell_attempts SET selection_reason = %s "
                    "WHERE attempt_id = %s RETURNING " + _ATTEMPT_COLUMNS,
                    (selection_reason, attempt_id),
                ).fetchone()
                completed = _attempt_from_row(row)
            connection.execute(
                "UPDATE experiment_cells SET active_attempt_id = NULL, "
                "selected_attempt_id = CASE WHEN %s THEN %s "
                "ELSE selected_attempt_id END WHERE cell_id = %s "
                "AND active_attempt_id = %s",
                (
                    should_promote,
                    attempt_id,
                    cell.cell_id,
                    attempt_id,
                ),
            )
            return OwnedAttemptCompletion(True, completed)

    def get_cell(self, cell_id: str) -> CellRecord:
        with self._connection() as connection:
            row = connection.execute(_CELL_SELECT + " WHERE cell_id = %s", (cell_id,)).fetchone()
            if row is None:
                raise KeyError(f"cell '{cell_id}' does not exist")
            return _cell_from_row(row)

    def get_attempt(self, attempt_id: str) -> AttemptRecord:
        with self._connection() as connection:
            row = connection.execute("SELECT " + _ATTEMPT_COLUMNS + " FROM experiment_cell_attempts WHERE attempt_id = %s", (attempt_id,)).fetchone()
            if row is None:
                raise KeyError(f"attempt '{attempt_id}' does not exist")
            return _attempt_from_row(row)

    def list_attempts(self, cell_id: str) -> tuple[AttemptRecord, ...]:
        with self._connection() as connection:
            return tuple(_attempt_from_row(row) for row in connection.execute("SELECT " + _ATTEMPT_COLUMNS + " FROM experiment_cell_attempts WHERE cell_id = %s ORDER BY attempt_number", (cell_id,)).fetchall())

    def has_active_attempts(
        self,
        experiment_id: str,
        *,
        execution_generation: int,
    ) -> bool:
        _validate_execution_generation(execution_generation)
        with self._connection() as connection:
            return connection.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM experiment_cell_attempts AS attempt "
                "JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id "
                "WHERE cell.experiment_id = %s "
                "AND attempt.execution_generation = %s "
                "AND attempt.execution_status IN ('queued', 'running')"
                ")",
                (experiment_id, execution_generation),
            ).fetchone()[0]

    def retryable_cells(self, experiment_id: str) -> tuple[CellRecord, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                _CELL_SELECT
                + " LEFT JOIN experiment_cell_attempts selected "
                "ON selected.attempt_id = experiment_cells.selected_attempt_id "
                "WHERE experiment_cells.experiment_id = %s "
                "AND experiment_cells.active_attempt_id IS NULL "
                "AND (experiment_cells.selected_attempt_id IS NULL "
                "OR selected.late_completion "
                "OR selected.execution_status IN ('failed', 'timed_out') "
                "OR selected.result_disposition = 'inconclusive') "
                "AND COALESCE(selected.execution_status, '') <> 'budget_exhausted' "
                "AND (COALESCE(selected.execution_status, '') <> 'cancelled' "
                "OR selected.recoverable_cancellation) "
                "ORDER BY experiment_cells.cell_id",
                (experiment_id,),
            ).fetchall()
            return tuple(_cell_from_row(row) for row in rows)

    def recover_expired_attempt(
        self,
        cell_id: str,
        *,
        experiment_id: str | None = None,
        execution_generation: int | None = None,
        execution_owner_id: str | None = None,
    ) -> CellRecord:
        with self._transaction() as connection:
            fenced = any(
                value is not None
                for value in (
                    experiment_id,
                    execution_generation,
                    execution_owner_id,
                )
            )
            identity = connection.execute(
                "SELECT experiment_id FROM experiment_cells WHERE cell_id = %s",
                (cell_id,),
            ).fetchone()
            if identity is None:
                raise KeyError(f"cell '{cell_id}' does not exist")
            if fenced:
                if execution_generation is None:
                    raise ValueError(
                        "execution_generation is required for owned lease recovery"
                    )
                if (
                    experiment_id != str(identity[0])
                    or not _lock_and_validate_owned_experiment(
                        connection,
                        str(identity[0]),
                        execution_generation=execution_generation,
                        execution_owner_id=execution_owner_id,
                    )
                ):
                    row = connection.execute(
                        _CELL_SELECT + " WHERE cell_id = %s",
                        (cell_id,),
                    ).fetchone()
                    if row is None:
                        raise KeyError(f"cell '{cell_id}' does not exist")
                    return _cell_from_row(row)
            row = connection.execute(
                _CELL_SELECT + " WHERE cell_id = %s FOR UPDATE",
                (cell_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"cell '{cell_id}' does not exist")
            cell = _cell_from_row(row)
            if fenced and cell.active_attempt_id is not None:
                active_generation = connection.execute(
                    "SELECT execution_generation FROM experiment_cell_attempts "
                    "WHERE attempt_id = %s",
                    (cell.active_attempt_id,),
                ).fetchone()
                if (
                    active_generation is not None
                    and int(active_generation[0]) != execution_generation
                ):
                    return cell
            return self._recover_expired_active(connection, cell)

    def count_attempts(self, experiment_id: str, *, retries_only: bool = False) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM experiment_cell_attempts AS attempt "
                "JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id "
                "WHERE cell.experiment_id = %s "
                + ("AND attempt.retry_kind <> 'initial'" if retries_only else ""),
                (experiment_id,),
            ).fetchone()
            return int(row[0])

    def create_retry_batch(
        self,
        experiment_id: str,
        *,
        action: str,
        completed_policy: str = "exclude",
        cell_ids: tuple[str, ...],
        requested_by: str,
        request_id: str | None = None,
        selection_metadata: Mapping[str, object] | None = None,
    ) -> RetryBatchRecord:
        unique_cell_ids = tuple(dict.fromkeys(cell_ids))
        selection_metadata = selection_metadata or {}
        candidate = RetryBatchRecord(
            _generated_id(), experiment_id, request_id, action,
            completed_policy, requested_by, unique_cell_ids,
            selection_metadata=selection_metadata,
            created_at=_now(),
        )
        with self._transaction() as connection:
            if request_id is not None:
                _lock_identity(
                    connection,
                    f"retry-request:{experiment_id}:{request_id}",
                )
                row = connection.execute(
                    _RETRY_BATCH_SELECT
                    + " WHERE experiment_id = %s AND request_id = %s FOR UPDATE",
                    (experiment_id, request_id),
                ).fetchone()
                if row is not None:
                    existing = self._retry_batch_from_row(connection, row)
                    if (
                        _retry_batch_request_content(existing)
                        != _retry_batch_request_content(candidate)
                    ):
                        raise ValueError(
                            "request_id already exists for a different retry request"
                        )
                    return replace(existing, was_created=False)
            for cell_id in sorted(unique_cell_ids):
                cell_identity = connection.execute(
                    "SELECT experiment_id FROM experiment_cells WHERE cell_id = %s",
                    (cell_id,),
                ).fetchone()
                if cell_identity is not None and cell_identity[0] != experiment_id:
                    raise ValueError("cell does not belong to experiment")
            experiment = connection.execute(
                "SELECT experiment_id FROM experiments "
                "WHERE experiment_id = %s FOR UPDATE",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise KeyError(f"experiment '{experiment_id}' does not exist")
            for cell_id in sorted(unique_cell_ids):
                row = connection.execute("SELECT experiment_id FROM experiment_cells WHERE cell_id = %s FOR UPDATE", (cell_id,)).fetchone()
                if row is None or row[0] != experiment_id:
                    raise ValueError("cell does not belong to experiment")
            batch = candidate
            connection.execute(
                "INSERT INTO retry_batches "
                "(retry_batch_id, experiment_id, request_id, action, completed_policy, "
                "requested_by, selection, execution_generation) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                (
                    batch.retry_batch_id, experiment_id, request_id, action,
                    completed_policy, requested_by,
                    _json(selection_metadata),
                    batch.execution_generation,
                ),
            )
            for ordinal, cell_id in enumerate(unique_cell_ids):
                connection.execute(
                    "INSERT INTO retry_batch_cells "
                    "(retry_batch_id, cell_id, member_ordinal) VALUES (%s, %s, %s)",
                    (batch.retry_batch_id, cell_id, ordinal),
                )
            row = connection.execute(
                _RETRY_BATCH_SELECT + " WHERE retry_batch_id = %s",
                (batch.retry_batch_id,),
            ).fetchone()
            return self._retry_batch_from_row(connection, row)

    def get_retry_batch(self, retry_batch_id: str) -> RetryBatchRecord:
        with self._connection() as connection:
            row = connection.execute(
                _RETRY_BATCH_SELECT + " WHERE retry_batch_id = %s",
                (retry_batch_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
            return self._retry_batch_from_row(connection, row)

    def get_retry_batch_by_request(self, experiment_id: str, request_id: str) -> RetryBatchRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                _RETRY_BATCH_SELECT
                + " WHERE experiment_id = %s AND request_id = %s",
                (experiment_id, request_id),
            ).fetchone()
            return self._retry_batch_from_row(connection, row) if row is not None else None

    def list_retry_batches(
        self,
        experiment_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> tuple[RetryBatchRecord, ...]:
        with self._connection() as connection:
            if statuses is not None and not statuses:
                return ()
            query = _RETRY_BATCH_SELECT + " WHERE experiment_id = %s"
            parameters: tuple[object, ...] = (experiment_id,)
            if statuses is not None:
                query += " AND status = ANY(%s)"
                parameters = (experiment_id, list(statuses))
            query += " ORDER BY created_at, retry_batch_id"
            return tuple(
                self._retry_batch_from_row(connection, row)
                for row in connection.execute(query, parameters).fetchall()
            )

    def update_retry_batch(self, retry_batch_id: str, *, status: str) -> RetryBatchRecord:
        with self._transaction() as connection:
            row = connection.execute(
                _RETRY_BATCH_SELECT + " WHERE retry_batch_id = %s FOR UPDATE",
                (retry_batch_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
            batch = self._retry_batch_from_row(connection, row)
            if batch.status in {"completed", "failed", "partial"}:
                if batch.status != status:
                    raise ValueError("retry batch is already terminal with different status")
                return batch
            row = connection.execute(
                "UPDATE retry_batches SET status = %s, "
                "started_at = CASE WHEN %s = 'running' THEN COALESCE(started_at, now()) ELSE started_at END, "
                "completed_at = CASE WHEN %s IN ('completed', 'failed', 'partial') THEN now() ELSE completed_at END "
                "WHERE retry_batch_id = %s RETURNING " + _RETRY_BATCH_COLUMNS,
                (status, status, status, retry_batch_id),
            ).fetchone()
            return self._retry_batch_from_row(connection, row)

    def bind_retry_batch_generation(
        self,
        retry_batch_id: str,
        execution_generation: int,
    ) -> RetryBatchRecord:
        _validate_positive_execution_generation(execution_generation)
        with self._transaction() as connection:
            batch_identity = connection.execute(
                "SELECT experiment_id, execution_generation FROM retry_batches "
                "WHERE retry_batch_id = %s",
                (retry_batch_id,),
            ).fetchone()
            if batch_identity is None:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
            if (
                batch_identity[1] is not None
                and int(batch_identity[1]) != execution_generation
            ):
                raise ValueError(
                    "retry batch is already bound to a different execution generation"
                )
            if not _lock_and_validate_owned_experiment(
                connection,
                str(batch_identity[0]),
                execution_generation=execution_generation,
                execution_owner_id=retry_batch_id,
            ):
                raise ValueError(
                    "retry batch execution ownership does not match experiment"
                )
            row = connection.execute(
                _RETRY_BATCH_SELECT + " WHERE retry_batch_id = %s FOR UPDATE",
                (retry_batch_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
            batch = self._retry_batch_from_row(connection, row)
            if batch.execution_generation is not None:
                if batch.execution_generation != execution_generation:
                    raise ValueError(
                        "retry batch is already bound to a different execution generation"
                    )
            associated_attempts = connection.execute(
                "SELECT member.attempt_id, attempt.execution_generation "
                "FROM retry_batch_cells AS member "
                "JOIN experiment_cell_attempts AS attempt "
                "ON attempt.attempt_id = member.attempt_id "
                "WHERE member.retry_batch_id = %s "
                "ORDER BY member.member_ordinal "
                "FOR UPDATE OF member, attempt",
                (retry_batch_id,),
            ).fetchall()
            if any(
                attempt_generation != execution_generation
                for _, attempt_generation in associated_attempts
            ):
                raise ValueError(
                    "associated attempt execution generation does not match retry batch"
                )
            if batch.execution_generation is not None:
                return batch
            row = connection.execute(
                "UPDATE retry_batches SET execution_generation = %s "
                "WHERE retry_batch_id = %s RETURNING " + _RETRY_BATCH_COLUMNS,
                (execution_generation, retry_batch_id),
            ).fetchone()
            return self._retry_batch_from_row(connection, row)

    def bind_retry_batch_attempt(self, retry_batch_id: str, cell_id: str, attempt_id: str) -> RetryBatchCellRecord:
        with self._transaction() as connection:
            batch_identity = connection.execute(
                "SELECT experiment_id, execution_generation FROM retry_batches "
                "WHERE retry_batch_id = %s",
                (retry_batch_id,),
            ).fetchone()
            if batch_identity is None:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
            if batch_identity[1] is None:
                raise ValueError(
                    "retry batch execution ownership is not durably bound"
                )
            if not _lock_and_validate_owned_experiment(
                connection,
                str(batch_identity[0]),
                execution_generation=int(batch_identity[1]),
                execution_owner_id=retry_batch_id,
            ):
                raise ValueError(
                    "retry batch execution ownership does not match experiment"
                )
            batch_row = connection.execute(
                "SELECT experiment_id, execution_generation FROM retry_batches "
                "WHERE retry_batch_id = %s FOR UPDATE",
                (retry_batch_id,),
            ).fetchone()
            if batch_row is None:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
            if tuple(batch_row) != tuple(batch_identity):
                raise ValueError("retry batch ownership changed during association")
            row = connection.execute(
                _RETRY_BATCH_CELL_SELECT
                + " WHERE retry_batch_id = %s AND cell_id = %s FOR UPDATE",
                (retry_batch_id, cell_id),
            ).fetchone()
            if row is None:
                raise KeyError("cell does not belong to retry batch")
            member = _retry_batch_cell_from_row(row)
            attempt_cell = connection.execute(
                "SELECT cell_id, execution_generation FROM experiment_cell_attempts "
                "WHERE attempt_id = %s FOR UPDATE",
                (attempt_id,),
            ).fetchone()
            if attempt_cell is None or attempt_cell[0] != cell_id:
                raise ValueError("attempt does not belong to retry batch cell")
            batch_generation = batch_row[1]
            if (
                batch_generation is not None
                and attempt_cell[1] != batch_generation
            ):
                raise ValueError("attempt execution generation does not match retry batch")
            if member.attempt_id is not None and member.attempt_id != attempt_id:
                raise ValueError("retry batch cell already owns a different attempt")
            row = connection.execute(
                "UPDATE retry_batch_cells SET attempt_id = %s, status = 'running' "
                "WHERE retry_batch_id = %s AND cell_id = %s RETURNING "
                + _RETRY_BATCH_CELL_COLUMNS,
                (attempt_id, retry_batch_id, cell_id),
            ).fetchone()
            return _retry_batch_cell_from_row(row)

    def create_and_bind_retry_attempt(
        self,
        retry_batch_id: str,
        cell_id: str,
        *,
        retry_kind: str,
        spec_hash: str,
        execution_generation: int = 0,
        checkpoint_id: str | None = None,
        promotion_policy: str = "replace_on_valid_completion",
        recoverable_cancellation: bool = False,
        max_attempts: int | None = None,
        max_retry_attempts_per_experiment: int | None = None,
    ) -> AttemptRecord:
        _validate_positive_execution_generation(execution_generation)
        _validate_positive_limit(max_attempts, name="max_attempts_per_cell")
        _validate_positive_limit(
            max_retry_attempts_per_experiment,
            name="max_retry_attempts_per_experiment",
        )
        with self._transaction() as connection:
            batch_identity = connection.execute(
                "SELECT experiment_id, action, execution_generation FROM retry_batches "
                "WHERE retry_batch_id = %s",
                (retry_batch_id,),
            ).fetchone()
            if batch_identity is None:
                raise KeyError(f"retry batch '{retry_batch_id}' does not exist")
            if batch_identity[2] is None:
                raise ValueError(
                    "retry batch execution ownership is not durably bound"
                )
            if int(batch_identity[2]) != execution_generation:
                raise ValueError(
                    "attempt execution generation does not match retry batch"
                )
            if not _lock_and_validate_owned_experiment(
                connection,
                str(batch_identity[0]),
                execution_generation=execution_generation,
                execution_owner_id=retry_batch_id,
            ):
                raise ValueError(
                    "retry batch execution ownership does not match experiment"
                )
            batch_row = connection.execute(
                "SELECT experiment_id, action, execution_generation FROM retry_batches "
                "WHERE retry_batch_id = %s FOR UPDATE",
                (retry_batch_id,),
            ).fetchone()
            if batch_row is None or tuple(batch_row) != tuple(batch_identity):
                raise ValueError("retry batch ownership changed during admission")
            cell_row = connection.execute(
                _CELL_SELECT + " WHERE cell_id = %s FOR UPDATE",
                (cell_id,),
            ).fetchone()
            if cell_row is None:
                raise KeyError(f"cell '{cell_id}' does not exist")
            cell = self._recover_expired_active(connection, _cell_from_row(cell_row))
            member_row = connection.execute(
                _RETRY_BATCH_CELL_SELECT
                + " WHERE retry_batch_id = %s AND cell_id = %s FOR UPDATE",
                (retry_batch_id, cell_id),
            ).fetchone()
            if member_row is None:
                raise KeyError("cell does not belong to retry batch")
            if batch_row[0] != cell.experiment_id:
                raise ValueError("retry batch cell does not belong to experiment")
            if batch_row[1] != retry_kind:
                raise ValueError("retry batch member belongs to a different retry attempt")
            if batch_row[2] != execution_generation:
                raise ValueError("attempt execution generation does not match retry batch")
            member = _retry_batch_cell_from_row(member_row)
            if member.attempt_id is not None:
                attempt_row = connection.execute(
                    "SELECT " + _ATTEMPT_COLUMNS
                    + " FROM experiment_cell_attempts WHERE attempt_id = %s FOR UPDATE",
                    (member.attempt_id,),
                ).fetchone()
                if attempt_row is None:
                    raise ValueError("retry batch member references a missing attempt")
                attempt = _attempt_from_row(attempt_row)
                if _bound_attempt_content(attempt) != (
                    cell_id,
                    retry_kind,
                    spec_hash,
                    checkpoint_id,
                    promotion_policy,
                    recoverable_cancellation,
                    execution_generation,
                ):
                    raise ValueError("retry batch member owns a different retry attempt")
                return attempt

            _lock_identity(connection, "retry-budget:" + cell.experiment_id)
            config_row = connection.execute(
                "SELECT spec_hash FROM experiment_technique_configs "
                "WHERE technique_config_id = %s",
                (cell.technique_config_id,),
            ).fetchone()
            if config_row is None or config_row[0] != spec_hash:
                raise ValueError("attempt spec hash must match the cell's pinned spec hash")
            claimable = connection.execute(
                "SELECT attempt_id FROM experiment_cell_attempts "
                "WHERE cell_id = %s AND execution_status IN ('queued', 'running') "
                "LIMIT 1 FOR UPDATE",
                (cell_id,),
            ).fetchone()
            if claimable is not None:
                raise ValueError("cell already has an active or claimable attempt")
            number = connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 "
                "FROM experiment_cell_attempts WHERE cell_id = %s",
                (cell_id,),
            ).fetchone()[0]
            if max_attempts is not None and number > max_attempts:
                raise ValueError("max_attempts_per_cell retry limit exceeded")
            if retry_kind != "initial" and max_retry_attempts_per_experiment is not None:
                retry_count = connection.execute(
                    "SELECT COUNT(*) FROM experiment_cell_attempts AS attempt "
                    "JOIN experiment_cells AS retry_cell ON retry_cell.cell_id = attempt.cell_id "
                    "WHERE retry_cell.experiment_id = %s AND attempt.retry_kind <> 'initial'",
                    (cell.experiment_id,),
                ).fetchone()[0]
                if retry_count >= max_retry_attempts_per_experiment:
                    raise ValueError(
                        "max_retry_attempts_per_experiment retry limit exceeded"
                    )
            parent_attempt_id = cell.latest_attempt_id
            attempt = AttemptRecord(
                _generated_id(),
                cell_id,
                number,
                parent_attempt_id,
                retry_kind,
                spec_hash,
                ExecutionStatus.QUEUED,
                ResultDisposition.NOT_TESTED,
                execution_generation=execution_generation,
                checkpoint_id=checkpoint_id,
                promotion_policy=promotion_policy,
                recoverable_cancellation=recoverable_cancellation,
                created_at=_now(),
            )
            inserted_attempt = connection.execute(
                "INSERT INTO experiment_cell_attempts "
                "(attempt_id, cell_id, attempt_number, parent_attempt_id, retry_kind, "
                "spec_hash, execution_status, result_disposition, execution_generation, checkpoint_id, "
                "promotion_policy, recoverable_cancellation) "
                "VALUES (%s, %s, %s, %s, %s, %s, 'queued', 'not_tested', %s, %s, %s, %s) "
                "RETURNING " + _ATTEMPT_COLUMNS,
                (
                    attempt.attempt_id,
                    cell_id,
                    number,
                    parent_attempt_id,
                    retry_kind,
                    spec_hash,
                    execution_generation,
                    checkpoint_id,
                    promotion_policy,
                    recoverable_cancellation,
                ),
            ).fetchone()
            attempt = _attempt_from_row(inserted_attempt)
            connection.execute(
                "UPDATE experiment_cells SET latest_attempt_id = %s WHERE cell_id = %s",
                (attempt.attempt_id, cell_id),
            )
            connection.execute(
                "UPDATE retry_batch_cells SET attempt_id = %s, status = 'queued' "
                "WHERE retry_batch_id = %s AND cell_id = %s",
                (attempt.attempt_id, retry_batch_id, cell_id),
            )
            return attempt

    def complete_retry_batch_cell(self, retry_batch_id: str, cell_id: str, *, status: str, error_category: str | None) -> RetryBatchCellRecord:
        with self._transaction() as connection:
            row = connection.execute(
                _RETRY_BATCH_CELL_SELECT
                + " WHERE retry_batch_id = %s AND cell_id = %s FOR UPDATE",
                (retry_batch_id, cell_id),
            ).fetchone()
            if row is None:
                raise KeyError("cell does not belong to retry batch")
            member = _retry_batch_cell_from_row(row)
            if member.status in {"completed", "failed", "skipped"}:
                if (member.status, member.error_category) != (status, error_category):
                    raise ValueError("retry batch cell is already terminal with different terminal evidence")
                return member
            row = connection.execute(
                "UPDATE retry_batch_cells SET status = %s, error_category = %s, "
                "completed_at = now() WHERE retry_batch_id = %s AND cell_id = %s "
                "RETURNING " + _RETRY_BATCH_CELL_COLUMNS,
                (status, error_category, retry_batch_id, cell_id),
            ).fetchone()
            return _retry_batch_cell_from_row(row)

    def _connection(self):
        import psycopg

        self._ensure_schema()
        return psycopg.connect(self._dsn, autocommit=True)

    def _transaction(self):
        connection = self._connection()
        return _PostgreSQLTransaction(connection)

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if not self._schema_ready:
                with __import__("psycopg").connect(self._dsn, autocommit=True) as connection:
                    ensure_experiment_attempt_schema(connection)
                self._schema_ready = True

    def _locked_cell_attempt(self, connection, attempt_id: str) -> tuple[CellRecord, AttemptRecord]:
        row = connection.execute(
            "SELECT cell_id FROM experiment_cell_attempts WHERE attempt_id = %s",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"attempt '{attempt_id}' does not exist")
        cell_row = connection.execute(_CELL_SELECT + " WHERE cell_id = %s FOR UPDATE", (row[0],)).fetchone()
        if cell_row is None:
            raise KeyError(f"cell '{row[0]}' does not exist")
        attempt_row = connection.execute(
            "SELECT " + _ATTEMPT_COLUMNS + " FROM experiment_cell_attempts "
            "WHERE attempt_id = %s FOR UPDATE",
            (attempt_id,),
        ).fetchone()
        if attempt_row is None:
            raise KeyError(f"attempt '{attempt_id}' does not exist")
        return _cell_from_row(cell_row), _attempt_from_row(attempt_row)

    def _recover_expired_active(self, connection, cell: CellRecord) -> CellRecord:
        if cell.active_attempt_id is None:
            return cell
        active_row = connection.execute(
            "SELECT " + _ATTEMPT_COLUMNS + " FROM experiment_cell_attempts "
            "WHERE attempt_id = %s FOR UPDATE",
            (cell.active_attempt_id,),
        ).fetchone()
        if active_row is None:
            return replace(cell, active_attempt_id=None)
        active = _attempt_from_row(active_row)
        recovered = connection.execute(
            "UPDATE experiment_cell_attempts SET execution_status = 'failed', "
            "result_disposition = 'inconclusive', failure_category = 'worker_lost', "
            "selection_reason = 'lease_expired', "
            "official_judge_status = 'failed', worker_id = NULL, lease_expires_at = NULL, "
            "completed_at = now() WHERE attempt_id = %s AND execution_status = 'running' "
            "AND (lease_expires_at IS NULL OR lease_expires_at <= now()) RETURNING attempt_id",
            (active.attempt_id,),
        ).fetchone()
        if recovered is not None:
            selected = self._selected_attempt(connection, cell.selected_attempt_id)
            selected_attempt_id = (
                active.attempt_id
                if not _is_valid_selection(selected)
                else cell.selected_attempt_id
            )
            connection.execute(
                "UPDATE experiment_cells SET active_attempt_id = NULL, selected_attempt_id = %s "
                "WHERE cell_id = %s AND active_attempt_id = %s",
                (selected_attempt_id, cell.cell_id, active.attempt_id),
            )
            return replace(cell, active_attempt_id=None, selected_attempt_id=selected_attempt_id)
        return cell

    def _selected_attempt(self, connection, attempt_id: str | None) -> AttemptRecord | None:
        if attempt_id is None:
            return None
        row = connection.execute("SELECT " + _ATTEMPT_COLUMNS + " FROM experiment_cell_attempts WHERE attempt_id = %s", (attempt_id,)).fetchone()
        return _attempt_from_row(row) if row else None

    def _retry_batch_from_row(self, connection, row) -> RetryBatchRecord:
        members = tuple(
            _retry_batch_cell_from_row(member)
            for member in connection.execute(
                _RETRY_BATCH_CELL_SELECT
                + " WHERE retry_batch_id = %s ORDER BY member_ordinal",
                (row[0],),
            ).fetchall()
        )
        return RetryBatchRecord(
            retry_batch_id=row[0],
            experiment_id=row[1],
            request_id=row[2],
            action=row[3],
            completed_policy=row[4],
            requested_by=row[5],
            cell_ids=tuple(member.cell_id for member in members),
            selection_metadata=row[6],
            execution_generation=row[7],
            members=members,
            status=row[8],
            created_at=row[9],
            started_at=row[10],
            completed_at=row[11],
        )


class _PostgreSQLTransaction:
    def __init__(self, connection) -> None:
        self._connection = connection
        self._transaction = None

    def __enter__(self):
        self._transaction = self._connection.transaction()
        self._transaction.__enter__()
        return self._connection

    def __exit__(self, *args):
        try:
            return self._transaction.__exit__(*args)
        finally:
            self._connection.close()


_ATTEMPT_COLUMNS = "attempt_id, cell_id, attempt_number, parent_attempt_id, retry_kind, spec_hash, execution_status, result_disposition, execution_generation, score, run_id, failure_category, checkpoint_id, promotion_policy, selection_reason, late_completion, recoverable_cancellation, worker_id, lease_expires_at, started_at, completed_at, created_at"
_CELL_SELECT = "SELECT experiment_cells.cell_id, experiment_cells.experiment_id, experiment_cells.technique_config_id, experiment_cells.technique_key, experiment_cells.prompt_id, experiment_cells.repeat_index, experiment_cells.selected_attempt_id, experiment_cells.latest_attempt_id, experiment_cells.active_attempt_id FROM experiment_cells"
_RETRY_BATCH_COLUMNS = "retry_batch_id, experiment_id, request_id, action, completed_policy, requested_by, selection, execution_generation, status, created_at, started_at, completed_at"
_RETRY_BATCH_SELECT = "SELECT " + _RETRY_BATCH_COLUMNS + " FROM retry_batches"
_RETRY_BATCH_CELL_COLUMNS = "retry_batch_id, cell_id, member_ordinal, status, attempt_id, error_category, created_at, completed_at"
_RETRY_BATCH_CELL_SELECT = "SELECT " + _RETRY_BATCH_CELL_COLUMNS + " FROM retry_batch_cells"


def _json(value: Mapping[str, object]) -> str:
    return json.dumps(_thaw_json(value), sort_keys=True, separators=(",", ":"))


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _lock_identity(connection, identity: str) -> None:
    connection.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (identity,),
    )


# Ownership-sensitive PostgreSQL mutations use one lock order:
# request advisory lock -> existing replay batch (return-only) -> experiment ->
# retry batch -> cell/member -> attempt.  Reads used only to discover the
# experiment identity never acquire a row lock before the experiment row.
def _lock_and_validate_owned_experiment(
    connection,
    experiment_id: str,
    *,
    execution_generation: int,
    execution_owner_id: str | None,
) -> bool:
    _validate_execution_generation(execution_generation)
    if execution_generation == 0 and not execution_owner_id:
        return True
    if type(execution_owner_id) is not str or not execution_owner_id:
        return False
    row = connection.execute(
        "SELECT execution_generation, execution_owner_id, status "
        "FROM experiments WHERE experiment_id = %s FOR UPDATE",
        (experiment_id,),
    ).fetchone()
    return bool(
        row is not None
        and int(row[0]) == execution_generation
        and row[1] == execution_owner_id
        and row[2] in {"running", "rejudging"}
    )


def _config_from_row(row) -> TechniqueConfigRecord:
    return TechniqueConfigRecord(row[0], row[1], row[2], row[3], row[4], row[8], row[5], row[6], row[7], row[9])


def _cell_from_row(row) -> CellRecord:
    return CellRecord(*row)


def _attempt_from_row(row) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=row[0],
        cell_id=row[1],
        attempt_number=row[2],
        parent_attempt_id=row[3],
        retry_kind=row[4],
        spec_hash=row[5],
        execution_status=ExecutionStatus(row[6]),
        result_disposition=ResultDisposition(row[7]),
        execution_generation=row[8],
        score=row[9],
        run_id=row[10],
        failure_category=FailureCategory(row[11]) if row[11] else None,
        checkpoint_id=row[12],
        promotion_policy=row[13],
        selection_reason=row[14],
        late_completion=row[15],
        recoverable_cancellation=row[16],
        worker_id=row[17],
        lease_expires_at=row[18],
        started_at=row[19],
        completed_at=row[20],
        created_at=row[21],
    )


def _retry_batch_cell_from_row(row) -> RetryBatchCellRecord:
    return RetryBatchCellRecord(
        retry_batch_id=row[0],
        cell_id=row[1],
        ordinal=row[2],
        status=row[3],
        attempt_id=row[4],
        error_category=row[5],
        created_at=row[6],
        completed_at=row[7],
    )
