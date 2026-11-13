from __future__ import annotations

import json
import hashlib
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from app.attacklib.base import AttackTechniqueMetadata
from app.evaluation.models import TechniqueRunResult
from app.experiment_plan_integrity import (
    is_legacy_partial_technique_config,
    plain_json as canonical_plain_json,
    validate_canonical_plan_snapshot,
)
from app.persistence.technique_runtime import (
    ensure_experiment_attempt_schema,
    ensure_technique_runtime_schema,
)
from app.techniques.core.redaction import (
    contains_secret_material,
    sanitize_public_mapping,
    sanitize_public_value,
)

if TYPE_CHECKING:
    from app.api.models import RunSummary

log = logging.getLogger("persistence.runs")


class ExperimentExecutionOwnershipConflict(RuntimeError):
    """The requested execution owner cannot acquire the experiment row."""


@dataclass(frozen=True, slots=True)
class ExperimentPlanSaveResult:
    """Authoritative disposition returned by atomic plan materialization."""

    was_created: bool
    status: str
    execution_generation: int
    execution_owner_id: str


class InMemoryExperimentExecutionAuthority:
    """Shared lifecycle authority for the Noop/in-memory persistence pair."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.experiments: dict[str, dict[str, Any]] = {}
        self.active_attempts: dict[str, tuple[str, int, str]] = {}
        self.operations: dict[str, dict[str, Any]] = {}
        self.rejudge_evidence: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}


def _validate_execution_owner(owner_id: str) -> None:
    if type(owner_id) is not str or not owner_id:
        raise ValueError("owner_id must be a non-empty built-in string")


def _validate_execution_generation(generation: int, *, name: str = "generation") -> None:
    if type(generation) is not int or generation < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_execution_generation(generation: int) -> None:
    if type(generation) is not int or generation < 1:
        raise ValueError("owned operation generation must be positive")


def _validate_active_experiment_status(status: str) -> None:
    if status not in {"running", "rejudging"}:
        raise ValueError("owned active experiment status must be running or rejudging")


def _validate_terminal_experiment_status(status: str) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("owned terminal experiment status must be completed or failed")


def _validate_owned_operation_fields(
    *,
    operation_id: str,
    worker_id: str,
    lease_seconds: int,
    operation_kind: str,
) -> None:
    for name, value in (
        ("operation_id", operation_id),
        ("worker_id", worker_id),
        ("operation_kind", operation_kind),
    ):
        if type(value) is not str or not value:
            raise ValueError(f"{name} must be a non-empty built-in string")
    if type(lease_seconds) is not int or lease_seconds <= 0:
        raise ValueError("lease_seconds must be a positive integer")


class RunPersistence(ABC):
    @abstractmethod
    def save_technique_run(
        self,
        *,
        summary: Any,
        result: TechniqueRunResult,
        conversation: list[dict[str, Any]],
        technique_metadata: AttackTechniqueMetadata,
        mode: str,
        requested_by: str,
        unsafe_intent: str,
        seed_prompt: str,
        target_config: dict[str, str],
        evaluation_id: str | None = None,
    ) -> None:
        """Persist one technique run."""

    @abstractmethod
    def save_baseline_run(
        self,
        *,
        target_config: dict[str, str],
        result: dict[str, Any],
        requested_by: str,
        evaluation_id: str | None = None,
    ) -> str:
        """Persist one baseline control run and return its run ID."""

    @abstractmethod
    def list_custom_techniques(self) -> list[dict[str, Any]]:
        """Return persisted custom technique definitions."""

    @abstractmethod
    def upsert_custom_technique(
        self,
        *,
        definition: dict[str, Any],
        requested_by: str,
    ) -> dict[str, Any]:
        """Create or update one custom technique definition."""

    @abstractmethod
    def delete_custom_technique(self, technique_id: str) -> None:
        """Delete one custom technique definition."""

    @abstractmethod
    def save_experiment_dataset(
        self,
        *,
        dataset: dict[str, Any],
    ) -> None:
        """Persist a dataset with its prompts."""

    @abstractmethod
    def save_experiment(
        self,
        *,
        experiment: dict[str, Any],
    ) -> None:
        """Persist experiment metadata."""

    @abstractmethod
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update experiment status and optionally its report."""

    def reconcile_experiment_terminal_status(
        self,
        experiment_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Compatibility fallback; durable backends override with an atomic transition."""

        self.update_experiment_status(
            experiment_id,
            status,
            report=report,
            error=error,
        )
        return True

    def begin_experiment_execution(
        self,
        experiment_id: str,
        *,
        owner_id: str,
        expected_generation: int,
        active_status: str,
    ) -> int:
        raise NotImplementedError

    def update_owned_experiment_status(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        raise NotImplementedError

    def reconcile_owned_experiment_terminal_status(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        raise NotImplementedError

    def admit_owned_experiment_operation(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        operation_id: str,
        worker_id: str,
        lease_seconds: int,
        operation_kind: str,
    ) -> bool:
        raise NotImplementedError

    def finish_owned_experiment_operation(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        operation_id: str,
        worker_id: str,
        outcome: str,
    ) -> bool:
        raise NotImplementedError

    def reconcile_legacy_interrupted_experiment(
        self,
        experiment_id: str,
        *,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Fail a restored pre-generation execution through an exact legacy CAS."""

        raise NotImplementedError

    @abstractmethod
    def save_experiment_cell(
        self,
        *,
        experiment_id: str,
        cell: dict[str, Any],
        generation: int | None = None,
        owner_id: str | None = None,
    ) -> bool:
        """Persist one experiment cell result."""

    def save_experiment_rejudge(
        self,
        *,
        experiment_id: str,
        generation: int,
        owner_id: str,
        cell_id: str,
        selected_attempt_id: str,
        rejudge_result: dict[str, Any],
        original_score: Any,
        original_asr: Any,
    ) -> bool:
        """Persist additive evaluation history without changing execution state."""
        raise NotImplementedError

    @abstractmethod
    def list_experiments(self) -> list[dict[str, Any]]:
        """Return persisted experiment summaries."""

    def load_all_experiments(self) -> list[dict[str, Any]]:
        """Load full experiment state from DB (cells + conversations). Override in DB backends."""
        return []

    def list_run_history(
        self,
        *,
        filters: dict[str, str],
        limit: int = 50,
        offset: int = 0,
        sort: str = "completed_desc",
    ) -> dict[str, Any]:
        """Return persisted one-off technique runs for operator history."""
        return {"items": [], "total": 0, "limit": limit, "offset": offset, "facets": {}}

    def get_run_history_item(self, run_id: str) -> dict[str, Any] | None:
        """Return one persisted one-off technique run with child evidence."""
        return None

    @abstractmethod
    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        """Return a DB-backed seed prompt for a technique.

        When *prefer_non_default* is True (seed-agnostic techniques), the
        query favours non-default seeds so the prompt varies across runs.
        Otherwise it returns the canonical default seed first.
        """


class NoopRunPersistence(RunPersistence):
    def __init__(
        self,
        *,
        execution_authority: InMemoryExperimentExecutionAuthority | None = None,
    ) -> None:
        self._execution_authority = (
            execution_authority or InMemoryExperimentExecutionAuthority()
        )
        # Retain the private lifecycle aliases used by existing Noop fixtures
        # while making the shared authority the single source of truth.
        self._experiment_lifecycle = self._execution_authority.experiments
        self._experiment_lifecycle_lock = self._execution_authority.lock

    @property
    def execution_authority(self) -> InMemoryExperimentExecutionAuthority:
        authority = getattr(self, "_execution_authority", None)
        if authority is None:
            authority = InMemoryExperimentExecutionAuthority()
            self._execution_authority = authority
        return authority

    def _lifecycle_state(self) -> tuple[dict[str, dict[str, Any]], threading.RLock]:
        authority = self.execution_authority
        return authority.experiments, authority.lock

    def save_technique_run(
        self,
        *,
        summary: Any,
        result: TechniqueRunResult,
        conversation: list[dict[str, Any]],
        technique_metadata: AttackTechniqueMetadata,
        mode: str,
        requested_by: str,
        unsafe_intent: str,
        seed_prompt: str,
        target_config: dict[str, str],
        evaluation_id: str | None = None,
    ) -> None:
        return

    def save_baseline_run(
        self,
        *,
        target_config: dict[str, str],
        result: dict[str, Any],
        requested_by: str,
        evaluation_id: str | None = None,
    ) -> str:
        return str(result.get("run_id", ""))

    def list_custom_techniques(self) -> list[dict[str, Any]]:
        return []

    def upsert_custom_technique(
        self,
        *,
        definition: dict[str, Any],
        requested_by: str,
    ) -> dict[str, Any]:
        return dict(definition)

    def delete_custom_technique(self, technique_id: str) -> None:
        return

    def save_experiment_dataset(self, *, dataset: dict[str, Any]) -> None:
        return

    def save_experiment(self, *, experiment: dict[str, Any]) -> None:
        state, lock = self._lifecycle_state()
        stored = deepcopy(experiment)
        experiment_id = str(stored["experiment_id"])
        if "execution_generation" not in stored:
            stored["execution_generation"] = (
                1 if stored.get("status") in {"running", "rejudging"} else 0
            )
        if "execution_owner_id" not in stored:
            stored["execution_owner_id"] = (
                f"initial:{experiment_id}"
                if stored["execution_generation"] == 1
                else ""
            )
        _validate_execution_generation(stored["execution_generation"])
        if stored["execution_owner_id"]:
            _validate_execution_owner(stored["execution_owner_id"])
        with lock:
            state[experiment_id] = stored

    def save_experiment_plan(
        self,
        *,
        experiment: dict[str, Any],
        technique_configs: tuple[Any, ...],
        cells: tuple[Any, ...],
        attempt_persistence: Any,
    ) -> ExperimentPlanSaveResult:
        """Atomically simulate plan materialization for the in-memory adapters."""

        experiment_id = str(experiment["experiment_id"])
        _validated_canonical_plan_snapshot(
            experiment.get("plan"),
            experiment_id=experiment_id,
            dataset_id=str(experiment.get("dataset_id", "")),
            repeat_count=experiment.get("repeat_count", 1),
            config_rows=[
                (
                    record.technique_config_id,
                    record.technique_id,
                    record.technique_version,
                    record.implementation_id,
                    record.preset,
                    record.parameters,
                    record.resolved_spec,
                    record.spec_hash,
                )
                for record in technique_configs
            ],
            cell_rows=[
                (
                    record.cell_id,
                    record.technique_config_id,
                    record.technique_key,
                    record.prompt_id,
                    record.repeat_index,
                )
                for record in cells
            ],
        )
        state, lock = self._lifecycle_state()
        with lock:
            existing = state.get(experiment_id)
            if (
                existing is not None
                and _immutable_experiment_creation(existing)
                != _immutable_experiment_creation(experiment)
            ):
                raise ValueError(
                    "experiment plan already exists with different content"
                )
            lifecycle_snapshot = deepcopy(state)
            plan_snapshot = attempt_persistence.snapshot_plan_state()
            was_created = existing is None
            try:
                if was_created:
                    self.save_experiment(experiment=experiment)
                for record in technique_configs:
                    attempt_persistence.upsert_technique_config(record)
                for record in cells:
                    attempt_persistence.create_cell(record)
            except BaseException:
                state.clear()
                state.update(lifecycle_snapshot)
                attempt_persistence.restore_plan_state(plan_snapshot)
                raise
            authoritative = state[experiment_id]
            generation = authoritative.get("execution_generation", 0)
            owner_id = authoritative.get("execution_owner_id", "")
            _validate_execution_generation(generation)
            if owner_id:
                _validate_execution_owner(owner_id)
            return ExperimentPlanSaveResult(
                was_created=was_created,
                status=str(authoritative.get("status", "pending")),
                execution_generation=int(generation),
                execution_owner_id=str(owner_id),
            )

    def update_experiment_status(
        self, experiment_id: str, status: str,
        report: dict[str, Any] | None = None, error: str | None = None,
    ) -> None:
        state, lock = self._lifecycle_state()
        with lock:
            experiment = state.get(experiment_id)
            if experiment is None:
                return
            experiment["status"] = status
            experiment["report"] = deepcopy(report)
            experiment["error"] = error

    def begin_experiment_execution(
        self,
        experiment_id: str,
        *,
        owner_id: str,
        expected_generation: int,
        active_status: str,
    ) -> int:
        _validate_execution_owner(owner_id)
        _validate_execution_generation(
            expected_generation,
            name="expected_generation",
        )
        _validate_active_experiment_status(active_status)
        state, lock = self._lifecycle_state()
        with lock:
            experiment = state.get(experiment_id)
            if experiment is None:
                raise ExperimentExecutionOwnershipConflict(
                    "experiment execution ownership conflict: experiment does not exist"
                )
            generation = experiment.get("execution_generation", 0)
            current_owner = experiment.get("execution_owner_id", "")
            _validate_execution_generation(generation)
            if current_owner == owner_id:
                return generation
            if (
                generation != expected_generation
                or experiment.get("status") not in {"completed", "failed"}
            ):
                raise ExperimentExecutionOwnershipConflict(
                    "experiment execution ownership conflict"
                )
            generation += 1
            experiment.update(
                {
                    "execution_generation": generation,
                    "execution_owner_id": owner_id,
                    "status": active_status,
                    "report": None,
                    "error": None,
                    "completed_at": None,
                }
            )
            return generation

    def update_owned_experiment_status(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        _validate_active_experiment_status(status)
        state, lock = self._lifecycle_state()
        with lock:
            experiment = state.get(experiment_id)
            if (
                experiment is None
                or experiment.get("execution_generation", 0) != generation
                or experiment.get("execution_owner_id", "") != owner_id
                or experiment.get("status") not in {"running", "rejudging"}
            ):
                return False
            experiment["status"] = status
            experiment["report"] = deepcopy(report)
            experiment["error"] = error
            experiment["completed_at"] = None
            return True

    def reconcile_owned_experiment_terminal_status(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        _validate_terminal_experiment_status(status)
        state, lock = self._lifecycle_state()
        with lock:
            experiment = state.get(experiment_id)
            if (
                experiment is None
                or experiment.get("execution_generation", 0) != generation
                or experiment.get("execution_owner_id", "") != owner_id
            ):
                return False
            if any(
                active_experiment_id == experiment_id
                and active_generation == generation
                and active_owner_id == owner_id
                for active_experiment_id, active_generation, active_owner_id
                in self.execution_authority.active_attempts.values()
            ):
                return False
            now = datetime.now(UTC)
            if any(
                operation.get("experiment_id") == experiment_id
                and operation.get("generation") == generation
                and operation.get("owner_id") == owner_id
                and operation.get("status") == "running"
                and operation.get("lease_expires_at") is not None
                and operation["lease_expires_at"] > now
                for operation in self.execution_authority.operations.values()
            ):
                return False
            if experiment.get("status") == status:
                return True
            if experiment.get("status") not in {"running", "rejudging"}:
                return False
            experiment["status"] = status
            experiment["report"] = deepcopy(report)
            experiment["error"] = error
            experiment["completed_at"] = datetime.now(UTC).isoformat()
            return True

    def admit_owned_experiment_operation(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        operation_id: str,
        worker_id: str,
        lease_seconds: int,
        operation_kind: str,
    ) -> bool:
        _validate_positive_execution_generation(generation)
        _validate_execution_owner(owner_id)
        _validate_owned_operation_fields(
            operation_id=operation_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            operation_kind=operation_kind,
        )
        authority = self.execution_authority
        with authority.lock:
            experiment = authority.experiments.get(experiment_id)
            if (
                experiment is None
                or experiment.get("execution_generation", 0) != generation
                or experiment.get("execution_owner_id", "") != owner_id
                or experiment.get("status") not in {"running", "rejudging"}
            ):
                return False
            now = datetime.now(UTC)
            existing = authority.operations.get(operation_id)
            identity = (experiment_id, generation, owner_id, operation_kind)
            if existing is not None:
                observed = (
                    existing.get("experiment_id"),
                    existing.get("generation"),
                    existing.get("owner_id"),
                    existing.get("operation_kind"),
                )
                if observed != identity:
                    raise ValueError(
                        "owned operation identity conflicts with persisted operation"
                    )
                if existing.get("status") != "running":
                    return False
                lease_expires_at = existing.get("lease_expires_at")
                if (
                    lease_expires_at is not None
                    and lease_expires_at > now
                    and existing.get("worker_id") != worker_id
                ):
                    return False
            authority.operations[operation_id] = {
                "experiment_id": experiment_id,
                "generation": generation,
                "owner_id": owner_id,
                "operation_kind": operation_kind,
                "worker_id": worker_id,
                "status": "running",
                "lease_expires_at": now + timedelta(seconds=lease_seconds),
                "outcome": None,
            }
            return True

    def finish_owned_experiment_operation(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        operation_id: str,
        worker_id: str,
        outcome: str,
    ) -> bool:
        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        if outcome not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("owned operation outcome is unsupported")
        authority = self.execution_authority
        with authority.lock:
            experiment = authority.experiments.get(experiment_id)
            if (
                experiment is None
                or experiment.get("execution_generation", 0) != generation
                or experiment.get("execution_owner_id", "") != owner_id
                or experiment.get("status") not in {"running", "rejudging"}
            ):
                return False
            operation = authority.operations.get(operation_id)
            if operation is None:
                return False
            if (
                operation.get("experiment_id") != experiment_id
                or operation.get("generation") != generation
                or operation.get("owner_id") != owner_id
                or operation.get("worker_id") != worker_id
                or operation.get("status") != "running"
            ):
                return False
            operation["status"] = "finished"
            operation["lease_expires_at"] = None
            operation["outcome"] = outcome
            return True

    def save_experiment_rejudge(
        self,
        *,
        experiment_id: str,
        generation: int,
        owner_id: str,
        cell_id: str,
        selected_attempt_id: str,
        rejudge_result: dict[str, Any],
        original_score: Any,
        original_asr: Any,
    ) -> bool:
        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        authority = self.execution_authority
        with authority.lock:
            experiment = authority.experiments.get(experiment_id)
            if (
                experiment is None
                or experiment.get("execution_generation", 0) != generation
                or experiment.get("execution_owner_id", "") != owner_id
                or experiment.get("status") != "rejudging"
            ):
                return False
            identity = (
                experiment_id,
                cell_id,
                str(rejudge_result.get("rejudge_batch_id") or ""),
                str(rejudge_result.get("target_evaluation_id") or ""),
                str(rejudge_result.get("fabricator_evaluation_id") or ""),
            )
            evidence = {
                "selected_attempt_id": selected_attempt_id,
                "rejudge_result": deepcopy(rejudge_result),
                "original_score": deepcopy(original_score),
                "original_asr": deepcopy(original_asr),
            }
            existing = authority.rejudge_evidence.get(identity)
            if existing is not None:
                if existing != evidence:
                    raise ValueError(
                        "rejudge evaluation identity conflicts with persisted result"
                    )
                return True
            authority.rejudge_evidence[identity] = evidence
            return True

    def reconcile_legacy_interrupted_experiment(
        self,
        experiment_id: str,
        *,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        state, lock = self._lifecycle_state()
        with lock:
            experiment = state.get(experiment_id)
            if (
                experiment is None
                or experiment.get("execution_generation", 0) != 0
                or experiment.get("execution_owner_id", "") != ""
            ):
                return False
            if experiment.get("status") == "failed":
                return True
            if experiment.get("status") not in {"pending", "running", "rejudging"}:
                return False
            experiment["status"] = "failed"
            experiment["report"] = deepcopy(report)
            experiment["error"] = error
            experiment["completed_at"] = datetime.now(UTC).isoformat()
            return True

    def save_experiment_cell(
        self,
        *,
        experiment_id: str,
        cell: dict[str, Any],
        generation: int | None = None,
        owner_id: str | None = None,
    ) -> bool:
        if (generation is None) != (owner_id is None):
            raise ValueError("cell projection ownership requires generation and owner")
        state, lock = self._lifecycle_state()
        with lock:
            if generation is not None:
                _validate_execution_generation(generation)
                _validate_execution_owner(owner_id)
                experiment = state.get(experiment_id)
                if (
                    experiment is None
                    or experiment.get("execution_generation", 0) != generation
                    or experiment.get("execution_owner_id", "") != owner_id
                    or experiment.get("status") not in {"running", "rejudging"}
                ):
                    return False
            return True

    def list_experiments(self) -> list[dict[str, Any]]:
        state, lock = self._lifecycle_state()
        with lock:
            return [
                {
                    key: deepcopy(value)
                    for key, value in experiment.items()
                    if key not in {"execution_generation", "execution_owner_id"}
                }
                for experiment in state.values()
            ]

    def load_all_experiments(self) -> list[dict[str, Any]]:
        state, lock = self._lifecycle_state()
        with lock:
            return deepcopy(list(state.values()))

    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        return ""


class PostgreSQLRunPersistence(RunPersistence):
    # Execution mutations lock the experiment row before any cell, attempt, or
    # owned-operation row.  Attempt persistence extends the same order with its
    # replay-only request/batch exception documented in experiment_attempts.py.
    def __init__(self, dsn: str | None = None) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise ImportError(
                "PostgreSQLRunPersistence requires the 'psycopg' package. "
                "Install it with: uv sync --extra postgres"
            ) from None
        self._dsn = dsn or os.environ.get(
            "DATABASE_URL",
            "postgresql://jailbreaker:jailbreaker@localhost:5432/jailbreaker",
        )

    def save_technique_run(
        self,
        *,
        summary: RunSummary,
        result: TechniqueRunResult,
        conversation: list[dict[str, Any]],
        technique_metadata: AttackTechniqueMetadata,
        mode: str,
        requested_by: str,
        unsafe_intent: str,
        seed_prompt: str,
        target_config: dict[str, str],
        evaluation_id: str | None = None,
    ) -> None:
        target_model = target_config.get("model", result.response.metadata.get("model", ""))
        attacker_model = _resolve_attacker_model(target_config, mode)
        final_prompt = _extract_final_prompt(conversation)
        final_output = result.response.final_output
        run_metadata = {
            "config": _sanitize_config(target_config),
            "response_metadata": dict(result.response.metadata),
            "tool_calls": list(result.response.tool_calls),
            "memory_events": list(result.response.memory_events),
            "audit_log": [_serialize_audit_log_event(item) for item in result.audit_log],
            "requires_feedback_loop": technique_metadata.requires_feedback_loop,
            "requires_attacker_llm": technique_metadata.requires_attacker_llm,
            "requires_judge_per_round": technique_metadata.requires_judge_per_round,
            "multi_turn": technique_metadata.multi_turn,
            "tags": list(technique_metadata.tags),
        }

        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, run_kind, evaluation_id, status, mode, requested_by,
                    technique_key, technique_name, technique_family, technique_surface,
                    target_kind, target_id, target_model, target_base_url,
                    attacker_model, attacker_base_url,
                    unsafe_intent, system_prompt, seed_prompt,
                    success, score, asr, matched_detections, highest_severity,
                    final_prompt, final_output, response_metadata, run_metadata,
                    created_at, completed_at
                ) VALUES (
                    %s, 'technique', %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s::jsonb, %s::jsonb,
                    %s, %s
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    evaluation_id = EXCLUDED.evaluation_id,
                    status = EXCLUDED.status,
                    mode = EXCLUDED.mode,
                    requested_by = EXCLUDED.requested_by,
                    technique_key = EXCLUDED.technique_key,
                    technique_name = EXCLUDED.technique_name,
                    technique_family = EXCLUDED.technique_family,
                    technique_surface = EXCLUDED.technique_surface,
                    target_kind = EXCLUDED.target_kind,
                    target_id = EXCLUDED.target_id,
                    target_model = EXCLUDED.target_model,
                    target_base_url = EXCLUDED.target_base_url,
                    attacker_model = EXCLUDED.attacker_model,
                    attacker_base_url = EXCLUDED.attacker_base_url,
                    unsafe_intent = EXCLUDED.unsafe_intent,
                    system_prompt = EXCLUDED.system_prompt,
                    seed_prompt = EXCLUDED.seed_prompt,
                    success = EXCLUDED.success,
                    score = EXCLUDED.score,
                    asr = EXCLUDED.asr,
                    matched_detections = EXCLUDED.matched_detections,
                    highest_severity = EXCLUDED.highest_severity,
                    final_prompt = EXCLUDED.final_prompt,
                    final_output = EXCLUDED.final_output,
                    response_metadata = EXCLUDED.response_metadata,
                    run_metadata = EXCLUDED.run_metadata,
                    completed_at = EXCLUDED.completed_at
                """,
                (
                    summary.run_id,
                    evaluation_id,
                    summary.status,
                    mode,
                    requested_by,
                    summary.technique_key,
                    technique_metadata.name,
                    technique_metadata.family,
                    technique_metadata.surface.value,
                    summary.target_kind,
                    result.target_id,
                    target_model,
                    target_config.get("api_base_url", ""),
                    attacker_model,
                    target_config.get("attacker_api_base_url", ""),
                    unsafe_intent,
                    target_config.get("system_prompt", "You are a helpful assistant."),
                    seed_prompt,
                    summary.asr == 100,
                    summary.score,
                    summary.asr,
                    summary.matched_detections,
                    summary.highest_severity,
                    final_prompt,
                    final_output,
                    json.dumps(dict(result.response.metadata)),
                    json.dumps(run_metadata),
                    _trace_created_at(result),
                    result.completed_at.isoformat(),
                ),
            )
            self._replace_detections(conn, summary.run_id, result)
            self._replace_conversation(conn, summary.run_id, conversation)
            self._replace_traces(conn, summary.run_id, result)

    def save_baseline_run(
        self,
        *,
        target_config: dict[str, str],
        result: dict[str, Any],
        requested_by: str,
        evaluation_id: str | None = None,
    ) -> str:
        run_id = str(result.get("run_id") or f"baseline-{uuid.uuid4().hex[:12]}")
        raw_judge_score = result.get("judge_score")
        judge_score = (
            int(raw_judge_score)
            if isinstance(raw_judge_score, (int, float))
            and not isinstance(raw_judge_score, bool)
            else None
        )
        conversation = [{
            "kind": "single_shot",
            "prompt": result.get("prompt", ""),
            "response": result.get("response", ""),
            "judge": {
                "score": str(result.get("judge_score", "")),
                "rationale": result.get("judge_rationale", ""),
                "input": result.get("judge_input", ""),
                "output": result.get("judge_output", ""),
                "error": result.get("judge_error", ""),
            },
        }]
        run_metadata = {
            "config": _sanitize_config(target_config),
            "succeeded": bool(result.get("succeeded", False)),
            "judge_score": judge_score,
            "judge_rationale": result.get("judge_rationale", ""),
            "judge_error": result.get("judge_error", ""),
        }
        score = int(round((judge_score / 10) * 100)) if judge_score is not None else 0
        asr = 100 if result.get("succeeded") else 0
        now = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, run_kind, evaluation_id, status, mode, requested_by,
                    technique_key, technique_name, technique_family, technique_surface,
                    target_kind, target_id, target_model, target_base_url,
                    attacker_model, attacker_base_url,
                    unsafe_intent, system_prompt, seed_prompt,
                    success, score, asr, matched_detections, highest_severity,
                    final_prompt, final_output, response_metadata, run_metadata,
                    created_at, completed_at
                ) VALUES (
                    %s, 'baseline', %s, %s, 'live', %s,
                    'baseline', 'Baseline', 'control', 'control',
                    'chatbot', 'baseline', %s, %s,
                    %s, %s,
                    %s, %s, '',
                    %s, %s, %s, 0, 'info',
                    %s, %s, '{}'::jsonb, %s::jsonb,
                    %s, %s
                )
                """,
                (
                    run_id,
                    evaluation_id,
                    "error" if result.get("judge_error") else "completed",
                    requested_by,
                    target_config.get("model", ""),
                    target_config.get("api_base_url", ""),
                    _resolve_attacker_model(target_config, "live"),
                    target_config.get("attacker_api_base_url", ""),
                    result.get("intent", ""),
                    target_config.get("system_prompt", "You are a helpful assistant."),
                    bool(result.get("succeeded", False)),
                    score,
                    asr,
                    result.get("prompt", ""),
                    result.get("response", ""),
                    json.dumps(run_metadata),
                    now,
                    now,
                ),
            )
            self._replace_conversation(conn, run_id, conversation)

        return run_id

    def list_custom_techniques(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    technique_id,
                    name,
                    summary,
                    target_system_prompt,
                    mode,
                    prompt_template,
                    attacker_system_prompt,
                    refinement_template,
                    max_rounds,
                    created_at,
                    updated_at,
                    requested_by
                FROM custom_techniques
                ORDER BY updated_at DESC, technique_id ASC
                """
            ).fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "summary": row[2],
                "target_system_prompt": row[3],
                "mode": row[4],
                "prompt_template": row[5],
                "attacker_system_prompt": row[6],
                "refinement_template": row[7],
                "max_rounds": row[8],
                "created_at": row[9].isoformat() if row[9] else "",
                "updated_at": row[10].isoformat() if row[10] else "",
                "requested_by": row[11],
            }
            for row in rows
        ]

    def upsert_custom_technique(
        self,
        *,
        definition: dict[str, Any],
        requested_by: str,
    ) -> dict[str, Any]:
        technique_id = str(definition.get("id", "")).strip()
        if not technique_id:
            raise ValueError("Custom technique `id` is required.")

        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                INSERT INTO custom_techniques (
                    technique_id,
                    name,
                    summary,
                    target_system_prompt,
                    mode,
                    prompt_template,
                    attacker_system_prompt,
                    refinement_template,
                    max_rounds,
                    requested_by
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (technique_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    summary = EXCLUDED.summary,
                    target_system_prompt = EXCLUDED.target_system_prompt,
                    mode = EXCLUDED.mode,
                    prompt_template = EXCLUDED.prompt_template,
                    attacker_system_prompt = EXCLUDED.attacker_system_prompt,
                    refinement_template = EXCLUDED.refinement_template,
                    max_rounds = EXCLUDED.max_rounds,
                    requested_by = EXCLUDED.requested_by,
                    updated_at = now()
                RETURNING
                    technique_id,
                    name,
                    summary,
                    target_system_prompt,
                    mode,
                    prompt_template,
                    attacker_system_prompt,
                    refinement_template,
                    max_rounds,
                    created_at,
                    updated_at,
                    requested_by
                """,
                (
                    technique_id,
                    str(definition.get("name", "")).strip(),
                    str(definition.get("summary", "")).strip(),
                    str(definition.get("target_system_prompt", "")).strip(),
                    str(definition.get("mode", "single_shot")).strip() or "single_shot",
                    str(definition.get("prompt_template", "")).strip(),
                    str(definition.get("attacker_system_prompt", "")).strip(),
                    str(definition.get("refinement_template", "")).strip(),
                    int(definition.get("max_rounds", 5) or 5),
                    requested_by,
                ),
            ).fetchone()
        return {
            "id": row[0],
            "name": row[1],
            "summary": row[2],
            "target_system_prompt": row[3],
            "mode": row[4],
            "prompt_template": row[5],
            "attacker_system_prompt": row[6],
            "refinement_template": row[7],
            "max_rounds": row[8],
            "created_at": row[9].isoformat() if row[9] else "",
            "updated_at": row[10].isoformat() if row[10] else "",
            "requested_by": row[11],
        }

    def delete_custom_technique(self, technique_id: str) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute("DELETE FROM custom_techniques WHERE technique_id = %s", (technique_id,))

    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        # When prefer_non_default is True (seed-agnostic techniques),
        # sort non-default seeds first so the prompt varies across runs.
        default_order = "is_default ASC" if prefer_non_default else "is_default DESC"
        query = f"""
            SELECT prompt_text
            FROM seed_prompts
            WHERE is_active = TRUE
              AND (technique_key = %s OR technique_key IS NULL)
            ORDER BY
              CASE WHEN technique_key = %s THEN 0 ELSE 1 END,
              {default_order},
              updated_at DESC
            LIMIT 1
        """
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(query, (technique_key, technique_key)).fetchone()
        return str(row[0]) if row else ""

    def ensure_schema(self, conn) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                run_kind TEXT NOT NULL,
                evaluation_id TEXT,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                requested_by TEXT NOT NULL DEFAULT '',
                technique_key TEXT NOT NULL,
                technique_name TEXT NOT NULL DEFAULT '',
                technique_family TEXT NOT NULL DEFAULT '',
                technique_surface TEXT NOT NULL DEFAULT '',
                target_kind TEXT NOT NULL,
                target_id TEXT NOT NULL,
                target_model TEXT NOT NULL DEFAULT '',
                target_base_url TEXT NOT NULL DEFAULT '',
                attacker_model TEXT NOT NULL DEFAULT '',
                attacker_base_url TEXT NOT NULL DEFAULT '',
                unsafe_intent TEXT NOT NULL DEFAULT '',
                system_prompt TEXT NOT NULL DEFAULT '',
                seed_prompt TEXT NOT NULL DEFAULT '',
                success BOOLEAN NOT NULL DEFAULT FALSE,
                score INTEGER NOT NULL DEFAULT 0,
                asr INTEGER NOT NULL DEFAULT 0,
                matched_detections INTEGER NOT NULL DEFAULT 0,
                highest_severity TEXT NOT NULL DEFAULT 'info',
                final_prompt TEXT NOT NULL DEFAULT '',
                final_output TEXT NOT NULL DEFAULT '',
                response_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                run_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_run_detections (
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                detection_index INTEGER NOT NULL,
                detection_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                matched BOOLEAN NOT NULL,
                rationale TEXT NOT NULL DEFAULT '',
                evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
                PRIMARY KEY (run_id, detection_index)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_run_conversations (
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                entry_index INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload JSONB NOT NULL,
                PRIMARY KEY (run_id, entry_index)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_run_traces (
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                trace_index INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL DEFAULT 'system',
                message TEXT NOT NULL DEFAULT '',
                event_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
                attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
                PRIMARY KEY (run_id, trace_index)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_completed_at ON evaluation_runs (completed_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_technique_key ON evaluation_runs (technique_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_target_model ON evaluation_runs (target_model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_success ON evaluation_runs (success)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seed_prompts (
                prompt_id TEXT PRIMARY KEY,
                technique_key TEXT,
                name TEXT NOT NULL DEFAULT '',
                prompt_text TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'system',
                is_default BOOLEAN NOT NULL DEFAULT FALSE,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_seed_prompts_technique_key ON seed_prompts (technique_key)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS custom_techniques (
                technique_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                target_system_prompt TEXT NOT NULL DEFAULT '',
                mode TEXT NOT NULL DEFAULT 'single_shot',
                prompt_template TEXT NOT NULL DEFAULT '',
                attacker_system_prompt TEXT NOT NULL DEFAULT '',
                refinement_template TEXT NOT NULL DEFAULT '',
                max_rounds INTEGER NOT NULL DEFAULT 5,
                requested_by TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_custom_techniques_updated_at ON custom_techniques (updated_at DESC)")

        # Experiment tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                prompt_count INTEGER NOT NULL DEFAULT 0,
                created_by TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_dataset_prompts (
                dataset_id TEXT NOT NULL REFERENCES experiment_datasets(dataset_id) ON DELETE CASCADE,
                prompt_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                gold_answer TEXT,
                sort_order INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dataset_id, prompt_id)
            )
            """
        )
        conn.execute("ALTER TABLE experiment_dataset_prompts ADD COLUMN IF NOT EXISTS gold_answer TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                hypothesis TEXT NOT NULL DEFAULT '',
                dataset_id TEXT REFERENCES experiment_datasets(dataset_id),
                technique_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
                target_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                experiment_group TEXT NOT NULL DEFAULT '',
                mode TEXT NOT NULL DEFAULT 'live',
                status TEXT NOT NULL DEFAULT 'pending',
                total_cells INTEGER NOT NULL DEFAULT 0,
                repeat_count INTEGER NOT NULL DEFAULT 1,
                max_rounds INTEGER NOT NULL DEFAULT 5,
                completed_cells INTEGER NOT NULL DEFAULT 0,
                execution_generation BIGINT NOT NULL DEFAULT 0,
                execution_owner_id TEXT NOT NULL DEFAULT '',
                report JSONB,
                error TEXT,
                created_by TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at TIMESTAMPTZ
            )
            """
        )
        conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS experiment_group TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS repeat_count INTEGER NOT NULL DEFAULT 1")
        conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS max_rounds INTEGER NOT NULL DEFAULT 5")
        conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS plan JSONB")
        conn.execute(
            "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS "
            "execution_generation BIGINT NOT NULL DEFAULT 0"
        )
        conn.execute(
            "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS "
            "execution_owner_id TEXT NOT NULL DEFAULT ''"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at DESC)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_cells (
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                technique_key TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                repeat_index INTEGER NOT NULL DEFAULT 1,
                run_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                score INTEGER NOT NULL DEFAULT 0,
                asr INTEGER NOT NULL DEFAULT 0,
                matched_detections INTEGER NOT NULL DEFAULT 0,
                highest_severity TEXT NOT NULL DEFAULT 'info',
                error TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                completed_at TIMESTAMPTZ,
                PRIMARY KEY (experiment_id, technique_key, prompt_id, repeat_index)
            )
            """
        )
        conn.execute("ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS repeat_index INTEGER NOT NULL DEFAULT 1")
        conn.execute("ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_cells_experiment_id ON experiment_cells (experiment_id)")
        ensure_technique_runtime_schema(conn)
        ensure_experiment_attempt_schema(conn)

    def save_experiment_dataset(self, *, dataset: dict[str, Any]) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO experiment_datasets (dataset_id, name, description, prompt_count, created_by)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id) DO NOTHING
                """,
                (
                    dataset["dataset_id"],
                    dataset["name"],
                    dataset.get("description", ""),
                    dataset.get("prompt_count", len(dataset.get("prompts", []))),
                    dataset.get("created_by", "demo-user"),
                ),
            )
            for i, prompt in enumerate(dataset.get("prompts", [])):
                conn.execute(
                    """
                    INSERT INTO experiment_dataset_prompts (
                        dataset_id, prompt_id, prompt_text, category, gold_answer, sort_order
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_id, prompt_id) DO NOTHING
                    """,
                    (
                        dataset["dataset_id"],
                        prompt["id"],
                        prompt["text"],
                        prompt.get("category", "general"),
                        prompt.get("gold_answer") if prompt.get("gold_answer") not in (None, "") else None,
                        i,
                    ),
                )

    def save_experiment_rejudge(
        self,
        *,
        experiment_id: str,
        generation: int,
        owner_id: str,
        cell_id: str,
        selected_attempt_id: str,
        rejudge_result: dict[str, Any],
        original_score: Any,
        original_asr: Any,
    ) -> bool:
        """Append rejudge metadata while preserving immutable attempt ownership."""

        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        for field_name, value in (
            ("experiment_id", experiment_id),
            ("cell_id", cell_id),
            ("selected_attempt_id", selected_attempt_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        safe_result = sanitize_public_mapping(rejudge_result)
        identity = _rejudge_result_identity(safe_result)
        with self._connect() as conn:
            self.ensure_schema(conn)
            with conn.transaction():
                owned = conn.execute(
                    """
                    SELECT status
                    FROM experiments
                    WHERE experiment_id = %s
                      AND execution_generation = %s
                      AND execution_owner_id = %s
                    FOR UPDATE
                    """,
                    (experiment_id, generation, owner_id),
                ).fetchone()
                if owned is None or owned[0] != "rejudging":
                    return False
                row = conn.execute(
                    """
                    SELECT selected_attempt_id, metadata
                    FROM experiment_cells
                    WHERE experiment_id = %s AND cell_id = %s
                    FOR UPDATE
                    """,
                    (experiment_id, cell_id),
                ).fetchone()
                if row is None:
                    raise KeyError(
                        f"experiment cell '{cell_id}' does not exist in '{experiment_id}'"
                    )
                if str(row[0] or "") != selected_attempt_id:
                    raise ValueError(
                        "rejudge selected attempt ownership does not match the "
                        "persisted experiment cell"
                    )
                metadata = dict(row[1]) if isinstance(row[1], Mapping) else {}
                prior_results = metadata.get("rejudge_results", [])
                if not isinstance(prior_results, list):
                    raise ValueError("persisted rejudge_results must be an array")
                for prior in prior_results:
                    if not isinstance(prior, Mapping):
                        raise ValueError("persisted rejudge result must be an object")
                    if not _has_task19_rejudge_identity(prior):
                        continue
                    if _rejudge_result_identity(prior) != identity:
                        continue
                    if dict(prior) == safe_result:
                        return True
                    raise ValueError("rejudge evaluation identity conflicts with persisted result")

                updated_metadata = dict(metadata)
                updated_metadata.setdefault(
                    "original_score",
                    sanitize_public_value(original_score),
                )
                updated_metadata.setdefault(
                    "original_asr",
                    sanitize_public_value(original_asr),
                )
                updated_metadata["rejudge_results"] = [
                    *prior_results,
                    safe_result,
                ]
                updated = conn.execute(
                    """
                    UPDATE experiment_cells
                    SET metadata = %s::jsonb
                    WHERE experiment_id = %s AND cell_id = %s
                      AND selected_attempt_id = %s
                    RETURNING cell_id
                    """,
                    (
                        json.dumps(updated_metadata),
                        experiment_id,
                        cell_id,
                        selected_attempt_id,
                    ),
                ).fetchone()
                if updated is None:
                    raise ValueError(
                        "rejudge selected attempt ownership changed during persistence"
                    )
                return True

    def save_experiment(self, *, experiment: dict[str, Any]) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            self._insert_experiment(conn, experiment)

    def save_experiment_plan(
        self,
        *,
        experiment: dict[str, Any],
        technique_configs: tuple[Any, ...],
        cells: tuple[Any, ...],
    ) -> ExperimentPlanSaveResult:
        """Atomically persist an experiment and its complete immutable plan."""

        max_rounds = experiment.get("max_rounds", 5)
        if type(max_rounds) is not int:
            raise ValueError("max_rounds must be an integer")
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        experiment_id = str(experiment["experiment_id"])
        if any(str(record.experiment_id) != experiment_id for record in technique_configs):
            raise ValueError("planned technique configuration belongs to another experiment")
        if any(str(record.experiment_id) != experiment_id for record in cells):
            raise ValueError("planned cell belongs to another experiment")
        _validated_canonical_plan_snapshot(
            experiment.get("plan"),
            experiment_id=experiment_id,
            dataset_id=str(experiment.get("dataset_id", "")),
            repeat_count=experiment.get("repeat_count", 1),
            config_rows=[
                (
                    record.technique_config_id,
                    record.technique_id,
                    record.technique_version,
                    record.implementation_id,
                    record.preset,
                    record.parameters,
                    record.resolved_spec,
                    record.spec_hash,
                )
                for record in technique_configs
            ],
            cell_rows=[
                (
                    record.cell_id,
                    record.technique_config_id,
                    record.technique_key,
                    record.prompt_id,
                    record.repeat_index,
                )
                for record in cells
            ],
        )
        with self._connect() as conn:
            self.ensure_schema(conn)
            with conn.transaction():
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"experiment-plan:{experiment_id}",),
                )
                existing = conn.execute(
                    """
                    SELECT name, description, hypothesis, dataset_id,
                           technique_keys, target_config, mode, total_cells,
                           repeat_count, max_rounds, experiment_group,
                           created_by, plan
                    FROM experiments
                    WHERE experiment_id = %s
                    FOR UPDATE
                    """,
                    (experiment_id,),
                ).fetchone()
                expected = _immutable_experiment_creation(experiment)
                was_created = existing is None
                if was_created:
                    self._insert_experiment(conn, experiment)
                elif _immutable_experiment_row(existing) != expected:
                    raise ValueError(
                        "experiment plan already exists with different content"
                    )
                for record in sorted(
                    technique_configs,
                    key=lambda item: str(item.technique_config_id),
                ):
                    self._insert_planned_technique_config(conn, record)
                for record in sorted(cells, key=lambda item: str(item.cell_id)):
                    self._insert_planned_cell(conn, record)
                ownership = conn.execute(
                    """
                    SELECT status, execution_generation, execution_owner_id
                    FROM experiments
                    WHERE experiment_id = %s
                    FOR UPDATE
                    """,
                    (experiment_id,),
                ).fetchone()
                if ownership is None:
                    raise RuntimeError(
                        "atomic experiment plan persistence lost its experiment row"
                    )
                status, generation, owner_id = ownership
                _validate_execution_generation(generation)
                if owner_id:
                    _validate_execution_owner(owner_id)
                return ExperimentPlanSaveResult(
                    was_created=was_created,
                    status=str(status),
                    execution_generation=int(generation),
                    execution_owner_id=str(owner_id),
                )

    @staticmethod
    def _insert_experiment(conn, experiment: dict[str, Any]) -> None:
        plan = experiment.get("plan")
        if contains_secret_material(plan):
            raise ValueError("canonical experiment plan contains credential material")
        execution_generation = experiment.get(
            "execution_generation",
            1
            if experiment.get("status", "running") in {"running", "rejudging"}
            else 0,
        )
        execution_owner_id = experiment.get(
            "execution_owner_id",
            f"initial:{experiment['experiment_id']}"
            if experiment.get("status", "running") in {"running", "rejudging"}
            else "",
        )
        _validate_execution_generation(execution_generation)
        if execution_owner_id:
            _validate_execution_owner(execution_owner_id)
        conn.execute(
            """
            INSERT INTO experiments (
                experiment_id, name, description, hypothesis,
                dataset_id, technique_keys, target_config, mode,
                status, total_cells, repeat_count, max_rounds,
                experiment_group, created_by, plan,
                execution_generation, execution_owner_id
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s::jsonb, %s::jsonb, %s,
                %s, %s, %s, %s,
                %s, %s, %s::jsonb, %s, %s
            )
            ON CONFLICT (experiment_id) DO NOTHING
            """,
            (
                experiment["experiment_id"],
                experiment["name"],
                experiment.get("description", ""),
                experiment.get("hypothesis", ""),
                experiment.get("dataset_id", ""),
                json.dumps(canonical_plain_json(experiment.get("technique_keys", []))),
                json.dumps(sanitize_public_value(experiment.get("target_config", {}))),
                experiment.get("mode", "live"),
                experiment.get("status", "running"),
                experiment.get("total_cells", 0),
                int(experiment.get("repeat_count", 1) or 1),
                int(experiment.get("max_rounds", 5) or 5),
                experiment.get("experiment_group", ""),
                experiment.get("created_by", "demo-user"),
                json.dumps(canonical_plain_json(plan)),
                execution_generation,
                execution_owner_id,
            ),
        )

    @staticmethod
    def _insert_planned_technique_config(conn, record) -> None:
        result = conn.execute(
            """
            INSERT INTO experiment_technique_configs (
                technique_config_id, experiment_id, technique_id,
                technique_version, implementation_id, preset, parameters,
                resolved_spec, spec_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
            ON CONFLICT (technique_config_id) DO UPDATE
            SET technique_config_id = EXCLUDED.technique_config_id
            WHERE experiment_technique_configs.experiment_id = EXCLUDED.experiment_id
              AND experiment_technique_configs.technique_id = EXCLUDED.technique_id
              AND experiment_technique_configs.technique_version = EXCLUDED.technique_version
              AND experiment_technique_configs.implementation_id = EXCLUDED.implementation_id
              AND experiment_technique_configs.preset IS NOT DISTINCT FROM EXCLUDED.preset
              AND experiment_technique_configs.parameters = EXCLUDED.parameters
              AND experiment_technique_configs.resolved_spec = EXCLUDED.resolved_spec
              AND experiment_technique_configs.spec_hash = EXCLUDED.spec_hash
            RETURNING technique_config_id
            """,
            (
                record.technique_config_id,
                record.experiment_id,
                record.technique_id,
                record.technique_version,
                record.implementation_id,
                record.preset,
                json.dumps(canonical_plain_json(record.parameters)),
                json.dumps(canonical_plain_json(record.resolved_spec)),
                record.spec_hash,
            ),
        )
        if result.fetchone() is None:
            raise ValueError(
                "technique configuration already exists with different content"
            )

    @staticmethod
    def _insert_planned_cell(conn, record) -> None:
        result = conn.execute(
            """
            INSERT INTO experiment_cells (
                cell_id, experiment_id, technique_key, technique_config_id,
                prompt_id, repeat_index
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (cell_id) DO UPDATE SET cell_id = EXCLUDED.cell_id
            WHERE experiment_cells.experiment_id = EXCLUDED.experiment_id
              AND experiment_cells.technique_key = EXCLUDED.technique_key
              AND experiment_cells.technique_config_id = EXCLUDED.technique_config_id
              AND experiment_cells.prompt_id = EXCLUDED.prompt_id
              AND experiment_cells.repeat_index = EXCLUDED.repeat_index
            RETURNING cell_id
            """,
            (
                record.cell_id,
                record.experiment_id,
                record.technique_key,
                record.technique_config_id,
                record.prompt_id,
                record.repeat_index,
            ),
        )
        if result.fetchone() is None:
            raise ValueError("cell ID already exists with different content")

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                UPDATE experiments
                SET status = %s,
                    report = %s::jsonb,
                    error = %s,
                    completed_cells = (
                        SELECT COUNT(*) FROM experiment_cells
                        WHERE experiment_id = %s AND status != 'pending'
                    ),
                    completed_at = CASE WHEN %s IN ('completed', 'failed') THEN %s::timestamptz ELSE completed_at END
                WHERE experiment_id = %s
                """,
                (
                    status,
                    json.dumps(report) if report else None,
                    error,
                    experiment_id,
                    status,
                    now,
                    experiment_id,
                ),
            )

    def begin_experiment_execution(
        self,
        experiment_id: str,
        *,
        owner_id: str,
        expected_generation: int,
        active_status: str,
    ) -> int:
        _validate_execution_owner(owner_id)
        _validate_execution_generation(
            expected_generation,
            name="expected_generation",
        )
        _validate_active_experiment_status(active_status)
        with self._connect() as conn:
            self.ensure_schema(conn)
            with conn.transaction():
                row = conn.execute(
                    """
                    SELECT execution_generation, execution_owner_id, status
                    FROM experiments
                    WHERE experiment_id = %s
                    FOR UPDATE
                    """,
                    (experiment_id,),
                ).fetchone()
                if row is None:
                    raise ExperimentExecutionOwnershipConflict(
                        "experiment execution ownership conflict: experiment does not exist"
                    )
                generation, current_owner, current_status = row
                generation = int(generation)
                if current_owner == owner_id:
                    return generation
                if (
                    generation != expected_generation
                    or current_status not in {"completed", "failed"}
                ):
                    raise ExperimentExecutionOwnershipConflict(
                        "experiment execution ownership conflict"
                    )
                generation += 1
                updated = conn.execute(
                    """
                    UPDATE experiments
                    SET execution_generation = %s,
                        execution_owner_id = %s,
                        status = %s,
                        report = NULL,
                        error = NULL,
                        completed_at = NULL
                    WHERE experiment_id = %s
                    RETURNING execution_generation
                    """,
                    (generation, owner_id, active_status, experiment_id),
                ).fetchone()
                if updated is None:
                    raise RuntimeError("experiment execution ownership update failed")
                return int(updated[0])

    def update_owned_experiment_status(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        _validate_active_experiment_status(status)
        with self._connect() as conn:
            self.ensure_schema(conn)
            updated = conn.execute(
                """
                UPDATE experiments
                SET status = %s,
                    report = %s::jsonb,
                    error = %s,
                    completed_at = NULL
                WHERE experiment_id = %s
                  AND execution_generation = %s
                  AND execution_owner_id = %s
                  AND status IN ('running', 'rejudging')
                RETURNING status
                """,
                (
                    status,
                    json.dumps(report) if report is not None else None,
                    error,
                    experiment_id,
                    generation,
                    owner_id,
                ),
            ).fetchone()
            return updated is not None

    def admit_owned_experiment_operation(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        operation_id: str,
        worker_id: str,
        lease_seconds: int,
        operation_kind: str,
    ) -> bool:
        _validate_positive_execution_generation(generation)
        _validate_execution_owner(owner_id)
        _validate_owned_operation_fields(
            operation_id=operation_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            operation_kind=operation_kind,
        )
        with self._connect() as conn:
            self.ensure_schema(conn)
            with conn.transaction():
                experiment = conn.execute(
                    """
                    SELECT execution_generation, execution_owner_id, status
                    FROM experiments
                    WHERE experiment_id = %s
                    FOR UPDATE
                    """,
                    (experiment_id,),
                ).fetchone()
                if (
                    experiment is None
                    or int(experiment[0]) != generation
                    or experiment[1] != owner_id
                    or experiment[2] not in {"running", "rejudging"}
                ):
                    return False
                existing = conn.execute(
                    """
                    SELECT experiment_id, execution_generation,
                           execution_owner_id, operation_kind, status,
                           worker_id, lease_expires_at > now()
                    FROM experiment_execution_operations
                    WHERE operation_id = %s
                    FOR UPDATE
                    """,
                    (operation_id,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO experiment_execution_operations (
                            operation_id, experiment_id, execution_generation,
                            execution_owner_id, operation_kind, worker_id,
                            status, lease_expires_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, 'running',
                                  now() + (%s * interval '1 second'))
                        """,
                        (
                            operation_id,
                            experiment_id,
                            generation,
                            owner_id,
                            operation_kind,
                            worker_id,
                            lease_seconds,
                        ),
                    )
                    return True
                identity = (
                    str(existing[0]),
                    int(existing[1]),
                    str(existing[2]),
                    str(existing[3]),
                )
                if identity != (
                    experiment_id,
                    generation,
                    owner_id,
                    operation_kind,
                ):
                    raise ValueError(
                        "owned operation identity conflicts with persisted operation"
                    )
                if existing[4] != "running":
                    return False
                if bool(existing[6]) and existing[5] != worker_id:
                    return False
                updated = conn.execute(
                    """
                    UPDATE experiment_execution_operations
                    SET worker_id = %s,
                        lease_expires_at = now() + (%s * interval '1 second'),
                        started_at = now(), completed_at = NULL, outcome = NULL
                    WHERE operation_id = %s AND status = 'running'
                    RETURNING operation_id
                    """,
                    (worker_id, lease_seconds, operation_id),
                ).fetchone()
                return updated is not None

    def finish_owned_experiment_operation(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        operation_id: str,
        worker_id: str,
        outcome: str,
    ) -> bool:
        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty built-in string")
        if type(worker_id) is not str or not worker_id:
            raise ValueError("worker_id must be a non-empty built-in string")
        if outcome not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("owned operation outcome is unsupported")
        with self._connect() as conn:
            self.ensure_schema(conn)
            with conn.transaction():
                experiment = conn.execute(
                    """
                    SELECT execution_generation, execution_owner_id, status
                    FROM experiments
                    WHERE experiment_id = %s FOR UPDATE
                    """,
                    (experiment_id,),
                ).fetchone()
                if (
                    experiment is None
                    or int(experiment[0]) != generation
                    or experiment[1] != owner_id
                    or experiment[2] not in {"running", "rejudging"}
                ):
                    return False
                operation = conn.execute(
                    """
                    SELECT experiment_id, execution_generation,
                           execution_owner_id, worker_id, status
                    FROM experiment_execution_operations
                    WHERE operation_id = %s
                    FOR UPDATE
                    """,
                    (operation_id,),
                ).fetchone()
                if (
                    operation is None
                    or str(operation[0]) != experiment_id
                    or int(operation[1]) != generation
                    or operation[2] != owner_id
                    or operation[3] != worker_id
                    or operation[4] != "running"
                ):
                    return False
                updated = conn.execute(
                    """
                    UPDATE experiment_execution_operations
                    SET status = 'finished', outcome = %s,
                        lease_expires_at = NULL, completed_at = now()
                    WHERE operation_id = %s AND worker_id = %s
                      AND status = 'running'
                    RETURNING operation_id
                    """,
                    (outcome, operation_id, worker_id),
                ).fetchone()
                return updated is not None

    def reconcile_owned_experiment_terminal_status(
        self,
        experiment_id: str,
        *,
        generation: int,
        owner_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        _validate_execution_generation(generation)
        _validate_execution_owner(owner_id)
        _validate_terminal_experiment_status(status)
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            self.ensure_schema(conn)
            with conn.transaction():
                existing = conn.execute(
                    """
                    SELECT execution_generation, execution_owner_id, status
                    FROM experiments
                    WHERE experiment_id = %s
                    FOR UPDATE
                    """,
                    (experiment_id,),
                ).fetchone()
                if (
                    existing is None
                    or int(existing[0]) != generation
                    or existing[1] != owner_id
                ):
                    return False
                active_attempt = conn.execute(
                    """
                    SELECT 1
                    FROM experiment_cell_attempts AS attempt
                    JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id
                    WHERE cell.experiment_id = %s
                      AND attempt.execution_generation = %s
                      AND attempt.execution_status IN ('queued', 'running')
                    LIMIT 1
                    """,
                    (experiment_id, generation),
                ).fetchone()
                if active_attempt is not None:
                    return False
                active_operation = conn.execute(
                    """
                    SELECT 1
                    FROM experiment_execution_operations
                    WHERE experiment_id = %s
                      AND execution_generation = %s
                      AND execution_owner_id = %s
                      AND status = 'running'
                      AND lease_expires_at > now()
                    LIMIT 1
                    """,
                    (experiment_id, generation, owner_id),
                ).fetchone()
                if active_operation is not None:
                    return False
                if existing[2] == status:
                    return True
                if existing[2] not in {"running", "rejudging"}:
                    return False
                updated = conn.execute(
                    """
                    UPDATE experiments
                    SET status = %s,
                        report = %s::jsonb,
                        error = %s,
                        completed_cells = (
                            SELECT COUNT(*) FROM experiment_cells
                            WHERE experiment_id = %s AND status != 'pending'
                        ),
                        completed_at = %s::timestamptz
                    WHERE experiment_id = %s
                    RETURNING status
                    """,
                    (
                        status,
                        json.dumps(report) if report is not None else None,
                        error,
                        experiment_id,
                        now,
                        experiment_id,
                    ),
                ).fetchone()
                return updated is not None

    def reconcile_legacy_interrupted_experiment(
        self,
        experiment_id: str,
        *,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            self.ensure_schema(conn)
            updated = conn.execute(
                """
                UPDATE experiments
                SET status = 'failed',
                    report = %s::jsonb,
                    error = %s,
                    completed_cells = (
                        SELECT COUNT(*) FROM experiment_cells
                        WHERE experiment_id = %s AND status != 'pending'
                    ),
                    completed_at = %s::timestamptz
                WHERE experiment_id = %s
                  AND execution_generation = 0
                  AND execution_owner_id = ''
                  AND status IN ('pending', 'running', 'rejudging')
                RETURNING status
                """,
                (
                    json.dumps(report) if report is not None else None,
                    error,
                    experiment_id,
                    now,
                    experiment_id,
                ),
            ).fetchone()
            if updated is not None:
                return True
            existing = conn.execute(
                """
                SELECT execution_generation, execution_owner_id, status
                FROM experiments
                WHERE experiment_id = %s
                """,
                (experiment_id,),
            ).fetchone()
            return existing == (0, "", "failed")

    def reconcile_experiment_terminal_status(
        self,
        experiment_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        if status not in {"completed", "failed"}:
            raise ValueError("reconciled experiment status must be terminal")
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            self.ensure_schema(conn)
            updated = conn.execute(
                """
                UPDATE experiments
                SET status = %s,
                    report = %s::jsonb,
                    error = %s,
                    completed_cells = (
                        SELECT COUNT(*) FROM experiment_cells
                        WHERE experiment_id = %s AND status != 'pending'
                    ),
                    completed_at = %s::timestamptz
                WHERE experiment_id = %s
                  AND status IN ('pending', 'running', 'rejudging')
                RETURNING status
                """,
                (
                    status,
                    json.dumps(report) if report else None,
                    error,
                    experiment_id,
                    now,
                    experiment_id,
                ),
            ).fetchone()
            if updated is not None:
                return True
            existing = conn.execute(
                "SELECT status FROM experiments WHERE experiment_id = %s",
                (experiment_id,),
            ).fetchone()
            return existing is not None and existing[0] == status

    def save_experiment_cell(
        self,
        *,
        experiment_id: str,
        cell: dict[str, Any],
        generation: int | None = None,
        owner_id: str | None = None,
    ) -> bool:
        repeat_index = cell.get("repeat_index", 1)
        if type(repeat_index) is not int or repeat_index < 1:
            raise ValueError("repeat_index must be a positive integer")
        if (generation is None) != (owner_id is None):
            raise ValueError("cell projection ownership requires generation and owner")
        if generation is not None:
            _validate_execution_generation(generation)
            _validate_execution_owner(owner_id)
        now = datetime.now(UTC).isoformat()
        technique_key = str(cell.get("technique_key", ""))
        prompt_id = str(cell.get("prompt_id", ""))
        metadata = {
            key: cell[key]
            for key in (
                "experiment_group",
                "repeat_count",
                "max_rounds",
                "original_score",
                "original_asr",
                "rejudge_results",
            )
            if key in cell
        }
        with self._connect() as conn:
            self.ensure_schema(conn)
            with conn.transaction():
                if generation is None:
                    conn.execute(
                        """
                        SELECT experiment_id
                        FROM experiments
                        WHERE experiment_id = %s
                        FOR UPDATE
                        """,
                        (experiment_id,),
                    ).fetchone()
                else:
                    experiment = conn.execute(
                        """
                        SELECT execution_generation, execution_owner_id, status
                        FROM experiments
                        WHERE experiment_id = %s
                        FOR UPDATE
                        """,
                        (experiment_id,),
                    ).fetchone()
                    if (
                        experiment is None
                        or int(experiment[0]) != generation
                        or experiment[1] != owner_id
                        or experiment[2] not in {"running", "rejudging"}
                    ):
                        return False
                supplied_cell_id = str(cell.get("cell_id") or "")
                supplied_config_id = str(cell.get("technique_config_id") or "")
                if bool(supplied_cell_id) != bool(supplied_config_id):
                    raise ValueError(
                        "planned cells must include both cell_id and technique_config_id"
                    )
                if supplied_cell_id:
                    for lock_key in sorted(
                        (
                            f"experiment-cell:{experiment_id}:{supplied_cell_id}",
                            f"experiment-technique-config:{experiment_id}:{supplied_config_id}",
                        )
                    ):
                        conn.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (lock_key,),
                        )
                    config = conn.execute(
                        """
                        SELECT technique_config_id, spec_hash
                        FROM experiment_technique_configs
                        WHERE experiment_id = %s AND technique_config_id = %s
                        FOR UPDATE
                        """,
                        (experiment_id, supplied_config_id),
                    ).fetchone()
                    if config is None:
                        raise ValueError("planned technique configuration does not exist")
                    technique_config_id, spec_hash = str(config[0]), str(config[1])
                    if cell.get("spec_hash") not in (None, "", spec_hash):
                        raise ValueError(
                            "planned cell spec hash does not match pinned configuration"
                        )
                    existing = conn.execute(
                        """
                        SELECT cell_id, selected_attempt_id, latest_attempt_id, active_attempt_id,
                               run_id, status, score, asr, matched_detections,
                               highest_severity, error, metadata, completed_at,
                               technique_config_id, technique_key, prompt_id, repeat_index
                        FROM experiment_cells
                        WHERE experiment_id = %s AND cell_id = %s
                        FOR UPDATE
                        """,
                        (experiment_id, supplied_cell_id),
                    ).fetchone()
                    if existing is None:
                        raise ValueError("planned experiment cell does not exist")
                    if (
                        str(existing[13]) != technique_config_id
                        or str(existing[14]) != technique_key
                        or str(existing[15]) != prompt_id
                        or int(existing[16]) != repeat_index
                    ):
                        raise ValueError("planned experiment cell content conflicts with persisted identity")
                else:
                    lock_key = f"experiment-legacy-cell:{experiment_id}:{technique_key}:{prompt_id}:{repeat_index}"
                    conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (lock_key,),
                    )
                    config = conn.execute(
                        """
                        SELECT technique_config_id, spec_hash
                        FROM experiment_technique_configs
                        WHERE experiment_id = %s
                          AND technique_id = %s
                          AND technique_version = '0.2.0'
                          AND implementation_id = 'legacy'
                          AND parameters = '{}'::jsonb
                          AND resolved_spec ->> 'technique_key' = %s
                        ORDER BY technique_config_id
                        LIMIT 1
                        FOR UPDATE
                        """,
                        (experiment_id, technique_key, technique_key),
                    ).fetchone()
                    if config is None:
                        resolved_spec = {
                            "compatibility": "legacy_implicit",
                            "implementation_id": "legacy",
                            "technique_key": technique_key,
                        }
                        canonical = json.dumps(
                            resolved_spec,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        config_material = "\x1f".join((experiment_id, canonical))
                        technique_config_id = (
                            "etc-"
                            + hashlib.sha256(config_material.encode("utf-8")).hexdigest()[:32]
                        )
                        spec_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                        conn.execute(
                            """
                            INSERT INTO experiment_technique_configs (
                                technique_config_id, experiment_id, technique_id,
                                technique_version, implementation_id, parameters,
                                resolved_spec, spec_hash
                            ) VALUES (%s, %s, %s, '0.2.0', 'legacy', '{}'::jsonb,
                                      %s::jsonb, %s)
                            ON CONFLICT (technique_config_id) DO NOTHING
                            """,
                            (
                                technique_config_id,
                                experiment_id,
                                technique_key,
                                canonical,
                                spec_hash,
                            ),
                        )
                    else:
                        technique_config_id, spec_hash = str(config[0]), str(config[1])
                    existing = conn.execute(
                        """
                        SELECT cell_id, selected_attempt_id, latest_attempt_id, active_attempt_id,
                               run_id, status, score, asr, matched_detections,
                               highest_severity, error, metadata, completed_at,
                               technique_config_id, technique_key, prompt_id, repeat_index
                        FROM experiment_cells
                        WHERE experiment_id = %s AND technique_config_id = %s
                          AND prompt_id = %s AND repeat_index = %s
                        FOR UPDATE
                        """,
                        (experiment_id, technique_config_id, prompt_id, repeat_index),
                    ).fetchone()
                    if existing is None:
                        material = "\x1f".join(
                            (experiment_id, technique_config_id, prompt_id, str(repeat_index))
                        )
                        cell_id = (
                            "cell-"
                            + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
                        )
                        conn.execute(
                            """
                            INSERT INTO experiment_cells (
                                cell_id, experiment_id, technique_key, technique_config_id,
                                prompt_id, repeat_index
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                cell_id,
                                experiment_id,
                                technique_key,
                                technique_config_id,
                                prompt_id,
                                repeat_index,
                            ),
                        )
                        existing = (
                            cell_id, None, None, None,
                            "", "pending", None, None, 0, "info", None, {}, None,
                            technique_config_id, technique_key, prompt_id, repeat_index,
                        )

                cell_id = str(existing[0])
                selected_attempt_id = existing[1]
                latest_attempt_id = existing[2]
                active_attempt_id = existing[3]
                existing_projection = existing[4:13]

                supplied_attempt_id = str(cell.get("attempt_id") or "")
                if supplied_attempt_id:
                    if not supplied_cell_id:
                        raise ValueError(
                            "explicit attempts require a planned experiment cell"
                        )
                    attempt_row = conn.execute(
                        """
                        SELECT attempt.cell_id, owner.experiment_id,
                               attempt.execution_status, attempt.result_disposition,
                               attempt.score, attempt.run_id, attempt.late_completion,
                               attempt.completed_at
                        FROM experiment_cell_attempts AS attempt
                        JOIN experiment_cells AS owner ON owner.cell_id = attempt.cell_id
                        WHERE attempt.attempt_id = %s
                        FOR UPDATE OF attempt
                        """,
                        (supplied_attempt_id,),
                    ).fetchone()
                    if (
                        attempt_row is None
                        or str(attempt_row[0]) != cell_id
                        or str(attempt_row[1]) != experiment_id
                    ):
                        raise ValueError(
                            "explicit attempt does not belong to the supplied experiment cell"
                        )
                    supplied_selected = str(cell.get("selected_attempt_id") or "")
                    if supplied_selected and supplied_selected != str(selected_attempt_id or ""):
                        raise ValueError(
                            "explicit cell selection conflicts with persisted attempt selection"
                        )
                    if str(selected_attempt_id or "") != supplied_attempt_id:
                        # A diagnostic, failed replacement, or obsolete replay
                        # cannot overwrite the compatibility projection owned by
                        # the repository-selected attempt.
                        return False
                    if attempt_row[6]:
                        raise ValueError(
                            "a late completion cannot be projected as selected efficacy"
                        )
                    tested = (
                        attempt_row[2] == "completed"
                        and attempt_row[3] == "tested"
                        and attempt_row[4] is not None
                    )
                    supplied_run_id = str(cell.get("run_id") or "")
                    persisted_run_id = str(attempt_row[5] or "")
                    if supplied_run_id != persisted_run_id:
                        raise ValueError(
                            "explicit cell run does not match the selected attempt"
                        )
                    if cell.get("score") != (attempt_row[4] if tested else None):
                        raise ValueError(
                            "explicit cell score does not match the selected attempt"
                        )
                    projected_status = (
                        "completed"
                        if attempt_row[2] == "completed"
                        else "running"
                        if attempt_row[2] == "running"
                        else "pending"
                        if attempt_row[2] == "queued"
                        else "error"
                    )
                    supplied_status = str(cell.get("status", projected_status))
                    normalized_supplied_status = (
                        "completed"
                        if supplied_status in {"completed", "succeeded"}
                        else "running"
                        if supplied_status == "running"
                        else "pending"
                        if supplied_status in {"pending", "queued"}
                        else "error"
                    )
                    if normalized_supplied_status != projected_status:
                        raise ValueError(
                            "explicit cell status does not match the selected attempt"
                        )
                    supplied_disposition = cell.get("result_disposition")
                    if (
                        supplied_disposition not in (None, "")
                        and str(supplied_disposition) != str(attempt_row[3])
                    ):
                        raise ValueError(
                            "explicit cell disposition does not match the selected attempt"
                        )
                    projection_metadata = {
                        **metadata,
                        "_selected_attempt_projection_id": supplied_attempt_id,
                    }
                    expected_projection = (
                        persisted_run_id,
                        projected_status,
                        attempt_row[4] if tested else None,
                        cell.get("asr") if tested else None,
                        cell.get("matched_detections", 0),
                        cell.get("highest_severity", "info"),
                        cell.get("error"),
                        projection_metadata,
                    )
                    existing_metadata = (
                        existing[11] if isinstance(existing[11], dict) else {}
                    )
                    if (
                        existing_metadata.get("_selected_attempt_projection_id")
                        == supplied_attempt_id
                    ):
                        if (*existing[4:11], existing_metadata) != expected_projection:
                            raise ValueError(
                                "explicit selected-attempt projection replay conflicts "
                                "with persisted evidence"
                            )
                        return True
                    conn.execute(
                        """
                        UPDATE experiment_cells
                        SET run_id = %s, status = %s, score = %s, asr = %s,
                            matched_detections = %s, highest_severity = %s,
                            error = %s, metadata = %s::jsonb, completed_at = %s
                        WHERE cell_id = %s AND selected_attempt_id = %s
                        """,
                        (
                            persisted_run_id,
                            projected_status,
                            attempt_row[4] if tested else None,
                            cell.get("asr") if tested else None,
                            cell.get("matched_detections", 0),
                            cell.get("highest_severity", "info"),
                            cell.get("error"),
                            json.dumps(projection_metadata),
                            attempt_row[7],
                            cell_id,
                            supplied_attempt_id,
                        ),
                    )
                    return True

                if active_attempt_id is not None:
                    conn.execute(
                        """
                        UPDATE experiment_cell_attempts
                        SET execution_status = 'cancelled', result_disposition = 'inconclusive',
                            official_judge_status = 'failed', score = NULL, asr = NULL,
                            failure_category = 'cancelled', worker_id = NULL,
                            lease_expires_at = NULL, completed_at = %s
                        WHERE attempt_id = %s AND execution_status = 'running'
                        """,
                        (now, active_attempt_id),
                    )

                raw_status = str(cell.get("status", "completed"))
                raw_error = cell.get("error")
                raw_score = cell.get("score", 0)
                tested = raw_status in {"completed", "succeeded"} and raw_error in (None, "") and raw_score is not None
                execution_status = (
                    "running" if raw_status == "running"
                    else "queued" if raw_status in {"pending", "queued"}
                    else "cancelled" if raw_status in {"cancelled", "canceled"}
                    else "timed_out" if raw_status in {"timed_out", "timeout"}
                    else "completed" if raw_status in {"completed", "succeeded"}
                    else "failed"
                )
                result_disposition = "tested" if tested else ("not_tested" if execution_status in {"queued", "running"} else "inconclusive")
                judge_status = "succeeded" if tested else ("not_scheduled" if result_disposition == "not_tested" else "failed")
                attempt_number = conn.execute(
                    "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM experiment_cell_attempts WHERE cell_id = %s",
                    (cell_id,),
                ).fetchone()[0]
                attempt_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO experiment_cell_attempts (
                        attempt_id, cell_id, attempt_number, parent_attempt_id, run_id, retry_kind, spec_hash,
                        execution_status, result_disposition, official_judge_status,
                        score, asr, failure_category, selection_reason, started_at, completed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'legacy_compatibility', %s, %s)
                    """,
                    (
                        attempt_id, cell_id, attempt_number, latest_attempt_id, cell.get("run_id", ""),
                        "initial" if attempt_number == 1 else "restart", spec_hash,
                        execution_status, result_disposition, judge_status,
                        raw_score if tested else None, cell.get("asr", 0) if tested else None,
                        None,
                        now, now if execution_status not in {"queued", "running"} else None,
                    ),
                )
                selected_is_tested = False
                if selected_attempt_id is not None:
                    selected_is_tested = bool(conn.execute(
                        "SELECT result_disposition = 'tested' FROM experiment_cell_attempts WHERE attempt_id = %s",
                        (selected_attempt_id,),
                    ).fetchone()[0])
                promote = result_disposition == "tested" or not selected_is_tested
                projected_attempt_id = attempt_id if promote else str(selected_attempt_id)
                projected = conn.execute(
                    """
                    SELECT run_id, execution_status, score, asr, failure_category
                    FROM experiment_cell_attempts WHERE attempt_id = %s
                    """,
                    (projected_attempt_id,),
                ).fetchone()
                projected_status = (
                    "running" if execution_status == "running"
                    else "completed" if projected[1] == "completed"
                    else "pending" if projected[1] == "queued"
                    else "error"
                )
                if promote:
                    projection = (
                        projected[0], projected_status, projected[2], projected[3],
                        cell.get("matched_detections", 0), cell.get("highest_severity", "info"),
                        raw_error, json.dumps(metadata),
                        None if execution_status in {"queued", "running"} else now,
                    )
                else:
                    projection = existing_projection
                    projection = (*projection[:1], projected_status, *projection[2:])
                conn.execute(
                    """
                    UPDATE experiment_cells
                    SET selected_attempt_id = %s, latest_attempt_id = %s,
                        active_attempt_id = CASE WHEN %s = 'running' THEN %s ELSE NULL END,
                        run_id = %s, status = %s, score = %s, asr = %s,
                        matched_detections = %s, highest_severity = %s,
                        error = %s, metadata = %s::jsonb, completed_at = %s
                    WHERE cell_id = %s
                    """,
                    (
                        projected_attempt_id, attempt_id, execution_status, attempt_id,
                        projection[0], projection[1], projection[2], projection[3],
                        projection[4], projection[5], projection[6],
                        projection[7] if isinstance(projection[7], str) else json.dumps(projection[7]),
                        projection[8], cell_id,
                    ),
                )
                return True

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT experiment_id, name, description, hypothesis,
                       dataset_id, technique_keys, mode, status,
                       total_cells, completed_cells, report, error,
                       created_at, completed_at, repeat_count, max_rounds,
                       experiment_group
                FROM experiments
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [
            {
                "experiment_id": r[0],
                "name": r[1],
                "description": r[2],
                "hypothesis": r[3],
                "dataset_id": r[4],
                "technique_keys": r[5] if isinstance(r[5], list) else [],
                "mode": r[6],
                "status": r[7],
                "total_cells": r[8],
                "completed_cells": r[9],
                "report": r[10],
                "error": r[11],
                "created_at": r[12].isoformat() if r[12] else "",
                "completed_at": r[13].isoformat() if r[13] else None,
                "repeat_count": r[14],
                "max_rounds": r[15],
                "experiment_group": r[16],
            }
            for r in rows
        ]

    def load_all_experiments(self) -> list[dict[str, Any]]:
        """Load full experiment state from DB, including cells and conversations."""
        with self._connect() as conn:
            self.ensure_schema(conn)

            # 1. Load experiments
            exp_rows = conn.execute(
                """
                SELECT experiment_id, name, description, hypothesis,
                       dataset_id, technique_keys, target_config, mode,
                       status, total_cells, completed_cells, report, error,
                       created_at, repeat_count, max_rounds, experiment_group,
                       plan, execution_generation, execution_owner_id
                FROM experiments ORDER BY created_at
                """
            ).fetchall()

            results = []
            for er in exp_rows:
                eid = er[0]
                dataset_id = er[4]
                technique_keys = er[5] if isinstance(er[5], list) else []
                target_config = er[6] if isinstance(er[6], dict) else {}
                status = er[8]

                # 2. Load dataset + prompts
                ds_row = conn.execute(
                    "SELECT dataset_id, name, description, prompt_count FROM experiment_datasets WHERE dataset_id = %s",
                    (dataset_id,),
                ).fetchone()
                dataset = {}
                if ds_row:
                    prompt_rows = conn.execute(
                        "SELECT prompt_id, prompt_text, category, gold_answer FROM experiment_dataset_prompts WHERE dataset_id = %s ORDER BY sort_order",
                        (dataset_id,),
                    ).fetchall()
                    dataset = {
                        "dataset_id": ds_row[0],
                        "name": ds_row[1],
                        "description": ds_row[2],
                        "prompt_count": ds_row[3],
                        "prompts": [
                            {
                                "id": pr[0],
                                "text": pr[1],
                                "category": pr[2],
                                **({"gold_answer": pr[3]} if pr[3] else {}),
                            }
                            for pr in prompt_rows
                        ],
                    }

                # 3. Load cells
                cell_rows = conn.execute(
                    """
                    SELECT cell_id, technique_config_id, technique_key, prompt_id,
                           repeat_index, run_id, status, score, asr,
                           matched_detections, highest_severity, error, metadata
                    FROM experiment_cells
                    WHERE experiment_id = %s
                    ORDER BY technique_config_id, prompt_id, repeat_index
                    """,
                    (eid,),
                ).fetchall()
                config_rows = conn.execute(
                    """
                    SELECT technique_config_id, technique_id, technique_version,
                           implementation_id, preset, parameters, resolved_spec,
                           spec_hash
                    FROM experiment_technique_configs
                    WHERE experiment_id = %s
                    ORDER BY technique_config_id
                    """,
                    (eid,),
                ).fetchall()
                config_hashes = {str(row[0]): str(row[7]) for row in config_rows}
                compatibility_keys = {str(row[1]): str(row[2]) for row in cell_rows}

                # Build prompt lookup
                prompt_lookup = {
                    p["id"]: p for p in dataset.get("prompts", [])
                }

                cell_results = []
                for cr in cell_rows:
                    run_id = cr[5] or ""
                    prompt_id = cr[3]
                    prompt = prompt_lookup.get(prompt_id, {})
                    metadata = (
                        {
                            key: value
                            for key, value in cr[12].items()
                            if key != "_selected_attempt_projection_id"
                        }
                        if isinstance(cr[12], dict)
                        else {}
                    )

                    # 4. Load conversation for this cell's run
                    conversation = []
                    if run_id:
                        conv_rows = conn.execute(
                            """
                            SELECT payload FROM evaluation_run_conversations
                            WHERE run_id = %s ORDER BY entry_index
                            """,
                            (run_id,),
                        ).fetchall()
                        conversation = [
                            row[0] if isinstance(row[0], dict) else {}
                            for row in conv_rows
                        ]

                    cell = {
                        "cell_id": cr[0],
                        "technique_config_id": cr[1],
                        "spec_hash": config_hashes.get(str(cr[1]), ""),
                        "technique_key": cr[2],
                        "prompt_id": prompt_id,
                        "prompt_text": prompt.get("text", ""),
                        "prompt_category": prompt.get("category", ""),
                        "repeat_index": cr[4],
                        "run_id": run_id,
                        "status": cr[6],
                        "score": cr[7],
                        "asr": cr[8],
                        "matched_detections": cr[9],
                        "highest_severity": cr[10],
                        "error": cr[11],
                        "metadata": metadata,
                        "conversation": conversation,
                    }
                    for key, value in metadata.items():
                        if key not in cell:
                            cell[key] = value
                    cell_results.append(cell)

                plan_configs = []
                legacy_implicit_config_count = 0
                for row in config_rows:
                    resolved_spec = _sanitize_plan_value(
                        row[6] if isinstance(row[6], dict) else {}
                    )
                    if is_legacy_partial_technique_config(
                        technique_version=row[2],
                        implementation_id=row[3],
                        resolved_spec=resolved_spec,
                    ):
                        legacy_implicit_config_count += 1
                    parameters = _sanitize_plan_value(
                        row[5] if isinstance(row[5], dict) else {}
                    )
                    plan_configs.append(
                        {
                            "technique_config_id": str(row[0]),
                            "technique_key": compatibility_keys.get(
                                str(row[0]), str(row[1])
                            ),
                            "technique_id": str(row[1]),
                            "technique_version": str(row[2]),
                            "implementation_id": str(row[3]),
                            "implementation_kind": str(
                                resolved_spec.get("implementation_kind") or "legacy"
                            ),
                            "contract_version": str(
                                resolved_spec.get("contract_version") or "1"
                            ),
                            "manifest_hash": str(
                                resolved_spec.get("manifest_hash") or row[7]
                            ),
                            "asset_hashes": resolved_spec.get("asset_hashes", {}),
                            "preset": row[4],
                            "parameters": parameters,
                            "model_roles": resolved_spec.get("model_roles", {}),
                            "judge_policies": resolved_spec.get("judge_policies", {}),
                            "effective_limits": resolved_spec.get("effective_limits", {}),
                            "runtime_config": resolved_spec.get("runtime_config", {}),
                            "resolved_spec": resolved_spec,
                            "spec_hash": str(row[7]),
                        }
                    )
                plan = None
                persisted_plan = er[17] if len(er) > 17 else None
                if isinstance(persisted_plan, Mapping):
                    plan = _validated_canonical_plan_snapshot(
                        persisted_plan,
                        experiment_id=str(eid),
                        dataset_id=str(dataset_id),
                        repeat_count=int(er[14]),
                        config_rows=config_rows,
                        cell_rows=cell_rows,
                    )
                elif 0 < legacy_implicit_config_count < len(config_rows):
                    raise ValueError(
                        "experiment plan contains mixed implicit and canonical "
                        "technique configurations"
                    )
                elif plan_configs and legacy_implicit_config_count == 0:
                    plan = _validated_canonical_plan_snapshot(
                        persisted_plan,
                        experiment_id=str(eid),
                        dataset_id=str(dataset_id),
                        repeat_count=int(er[14]),
                        config_rows=config_rows,
                        cell_rows=cell_rows,
                    )
                if plan is not None:
                    snapshot_cells = {
                        str(cell["cell_id"]): cell for cell in plan["cells"]
                    }
                    for cell in cell_results:
                        snapshot = snapshot_cells[str(cell["cell_id"])]
                        cell["prompt_text"] = snapshot["prompt_text"]
                        cell["prompt_category"] = snapshot["prompt_category"]
                        if snapshot["gold_answer"] is None:
                            cell.pop("gold_answer", None)
                        else:
                            cell["gold_answer"] = snapshot["gold_answer"]

                completed_cells = sum(
                    1 for c in cell_results if c["status"] in ("completed", "error")
                )

                results.append({
                    "experiment_id": eid,
                    "name": er[1],
                    "description": er[2],
                    "hypothesis": er[3],
                    "dataset": dataset,
                    "technique_keys": technique_keys,
                    "target_config": target_config,
                    "mode": er[7],
                    "status": status,
                    "total_cells": er[9],
                    "completed_cells": completed_cells,
                    "cell_results": cell_results,
                    "report": er[11],
                    "error": er[12],
                    "created_at": er[13].isoformat() if er[13] else "",
                    "repeat_count": er[14],
                    "max_rounds": er[15],
                    "experiment_group": er[16],
                    "plan": plan,
                    "execution_generation": er[18] if len(er) > 18 else 0,
                    "execution_owner_id": er[19] if len(er) > 19 else "",
                })

            return results

    def list_run_history(
        self,
        *,
        filters: dict[str, str],
        limit: int = 50,
        offset: int = 0,
        sort: str = "completed_desc",
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit or 50), 100))
        offset = max(0, int(offset or 0))
        where_sql, params = _build_run_history_where(filters)
        order_sql = _run_history_order(sort)

        query = f"""
            SELECT
                run_id,
                status,
                mode,
                technique_key,
                technique_name,
                technique_family,
                technique_surface,
                target_kind,
                target_model,
                attacker_model,
                unsafe_intent,
                success,
                score,
                asr,
                matched_detections,
                highest_severity,
                final_prompt,
                final_output,
                created_at,
                completed_at,
                EXTRACT(EPOCH FROM (completed_at - created_at)) AS duration_seconds,
                COUNT(*) OVER() AS total_count,
                run_metadata
            FROM evaluation_runs
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT %s OFFSET %s
        """

        with self._connect() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(query, (*params, limit, offset)).fetchall()
            facets = _load_run_history_facets(conn)

        total = int(rows[0][21]) if rows else 0
        return {
            "items": [_serialize_history_row(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "facets": facets,
        }

    def get_run_history_item(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT
                    run_id,
                    run_kind,
                    evaluation_id,
                    status,
                    mode,
                    requested_by,
                    technique_key,
                    technique_name,
                    technique_family,
                    technique_surface,
                    target_kind,
                    target_model,
                    attacker_model,
                    unsafe_intent,
                    success,
                    score,
                    asr,
                    matched_detections,
                    highest_severity,
                    final_prompt,
                    final_output,
                    response_metadata,
                    run_metadata,
                    created_at,
                    completed_at,
                    EXTRACT(EPOCH FROM (completed_at - created_at)) AS duration_seconds
                FROM evaluation_runs
                WHERE run_id = %s
                  AND run_kind = 'technique'
                  AND evaluation_id IS NULL
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None

            detections = conn.execute(
                """
                SELECT detection_type, severity, matched, rationale, evidence
                FROM evaluation_run_detections
                WHERE run_id = %s
                ORDER BY detection_index
                """,
                (run_id,),
            ).fetchall()
            conversations = conn.execute(
                """
                SELECT kind, payload
                FROM evaluation_run_conversations
                WHERE run_id = %s
                ORDER BY entry_index
                """,
                (run_id,),
            ).fetchall()
            traces = conn.execute(
                """
                SELECT event_type, actor, message, event_timestamp, attributes
                FROM evaluation_run_traces
                WHERE run_id = %s
                ORDER BY trace_index
                """,
                (run_id,),
            ).fetchall()

        item = _serialize_history_detail_row(row)
        item["detections"] = [
            {
                "detection_type": det[0],
                "severity": det[1],
                "matched": bool(det[2]),
                "rationale": det[3],
                "evidence": _json_value(det[4], []),
            }
            for det in detections
        ]
        item["conversation"] = [
            _history_conversation_entry(kind, payload)
            for kind, payload in conversations
        ]
        item["traces"] = [
            {
                "event_type": trace[0],
                "actor": trace[1],
                "message": trace[2],
                "event_timestamp": _isoformat(trace[3]),
                "attributes": _json_value(trace[4], {}),
            }
            for trace in traces
        ]
        return item

    def _replace_detections(self, conn, run_id: str, result: TechniqueRunResult) -> None:
        conn.execute("DELETE FROM evaluation_run_detections WHERE run_id = %s", (run_id,))
        for idx, det in enumerate(result.detections):
            conn.execute(
                """
                INSERT INTO evaluation_run_detections (
                    run_id, detection_index, detection_type, severity, matched, rationale, evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    run_id,
                    idx,
                    det.detection_type.value,
                    det.severity.value,
                    det.matched,
                    det.rationale,
                    json.dumps(list(det.evidence)),
                ),
            )

    def _replace_conversation(self, conn, run_id: str, conversation: list[dict[str, Any]]) -> None:
        conn.execute("DELETE FROM evaluation_run_conversations WHERE run_id = %s", (run_id,))
        for idx, entry in enumerate(conversation):
            conn.execute(
                """
                INSERT INTO evaluation_run_conversations (run_id, entry_index, kind, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (run_id, idx, str(entry.get("kind", "")), json.dumps(entry)),
            )

    def _replace_traces(self, conn, run_id: str, result: TechniqueRunResult) -> None:
        conn.execute("DELETE FROM evaluation_run_traces WHERE run_id = %s", (run_id,))
        for idx, trace in enumerate(result.traces):
            conn.execute(
                """
                INSERT INTO evaluation_run_traces (
                    run_id, trace_index, event_type, actor, message, event_timestamp, attributes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    run_id,
                    idx,
                    trace.event_type,
                    trace.actor,
                    trace.message,
                    trace.timestamp.isoformat(),
                    json.dumps(dict(trace.attributes)),
                ),
            )

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)


def build_run_persistence() -> RunPersistence:
    try:
        return PostgreSQLRunPersistence()
    except Exception as exc:  # noqa: BLE001
        log.warning("Run persistence disabled: %s", exc)
        return NoopRunPersistence()


_RUN_HISTORY_FILTER_COLUMNS = {
    "technique_key": "technique_key",
    "technique_family": "technique_family",
    "technique_surface": "technique_surface",
    "status": "status",
    "highest_severity": "highest_severity",
    "mode": "mode",
    "target_kind": "target_kind",
    "target_model": "target_model",
}

_RUN_HISTORY_SORTS = {
    "completed_desc": "completed_at DESC, run_id DESC",
    "completed_asc": "completed_at ASC, run_id ASC",
    "score_desc": "score DESC, completed_at DESC",
    "score_asc": "score ASC, completed_at DESC",
    "asr_desc": "asr DESC, completed_at DESC",
    "asr_asc": "asr ASC, completed_at DESC",
    "severity_desc": """
        CASE highest_severity
            WHEN 'critical' THEN 5
            WHEN 'high' THEN 4
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 2
            ELSE 1
        END DESC,
        completed_at DESC
    """,
}


def _build_run_history_where(filters: dict[str, str]) -> tuple[str, list[Any]]:
    clauses = ["run_kind = 'technique'", "evaluation_id IS NULL"]
    params: list[Any] = []

    query = str(filters.get("q", "")).strip()
    if query:
        clauses.append(
            """
            concat_ws(
                ' ',
                run_id,
                technique_key,
                technique_name,
                technique_family,
                technique_surface,
                target_model,
                attacker_model,
                unsafe_intent,
                final_prompt,
                final_output
            ) ILIKE %s
            """
        )
        params.append(f"%{query}%")

    for key, column in _RUN_HISTORY_FILTER_COLUMNS.items():
        value = str(filters.get(key, "")).strip()
        if value:
            clauses.append(f"{column} = %s")
            params.append(value)

    success = str(filters.get("success", "")).strip().lower()
    if success in {"true", "false"}:
        clauses.append("success = %s")
        params.append(success == "true")

    completed_after = str(filters.get("completed_after", "")).strip()
    if completed_after:
        clauses.append("completed_at >= %s::timestamptz")
        params.append(completed_after)

    completed_before = str(filters.get("completed_before", "")).strip()
    if completed_before:
        if len(completed_before) == 10:
            clauses.append("completed_at < (%s::date + INTERVAL '1 day')")
        else:
            clauses.append("completed_at <= %s::timestamptz")
        params.append(completed_before)

    for key, column, operator in (
        ("min_score", "score", ">="),
        ("max_score", "score", "<="),
        ("min_asr", "asr", ">="),
        ("max_asr", "asr", "<="),
    ):
        raw = str(filters.get(key, "")).strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        clauses.append(f"{column} {operator} %s")
        params.append(value)

    return " AND ".join(clauses), params


def _run_history_order(sort: str) -> str:
    return _RUN_HISTORY_SORTS.get(sort, _RUN_HISTORY_SORTS["completed_desc"])


def _load_run_history_facets(conn) -> dict[str, list[dict[str, Any]]]:
    return {
        "techniques": [
            {"key": row[0], "label": row[1] or row[0], "count": row[2]}
            for row in conn.execute(
                """
                SELECT technique_key, technique_name, COUNT(*) AS run_count
                FROM evaluation_runs
                WHERE run_kind = 'technique' AND evaluation_id IS NULL
                GROUP BY technique_key, technique_name
                ORDER BY technique_name, technique_key
                """
            ).fetchall()
        ],
        "families": _facet_rows(conn, "technique_family"),
        "surfaces": _facet_rows(conn, "technique_surface"),
        "severities": _facet_rows(conn, "highest_severity"),
        "statuses": _facet_rows(conn, "status"),
        "modes": _facet_rows(conn, "mode"),
        "target_models": _facet_rows(conn, "target_model"),
    }


def _facet_rows(conn, column: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT {column}, COUNT(*) AS run_count
        FROM evaluation_runs
        WHERE run_kind = 'technique'
          AND evaluation_id IS NULL
          AND {column} <> ''
        GROUP BY {column}
        ORDER BY run_count DESC, {column} ASC
        """
    ).fetchall()
    return [{"key": row[0], "label": row[0], "count": row[1]} for row in rows]


def _serialize_history_row(row) -> dict[str, Any]:
    return {
        "run_id": row[0],
        "status": row[1],
        "mode": row[2],
        "technique_key": row[3],
        "technique_name": row[4],
        "technique_family": row[5],
        "technique_surface": row[6],
        "target_kind": row[7],
        "target_model": row[8],
        "attacker_model": row[9],
        "unsafe_intent": row[10],
        "success": bool(row[11]),
        "score": row[12],
        "asr": row[13],
        "matched_detections": row[14],
        "highest_severity": row[15],
        "judge_model": _history_judge_model(row[22] if len(row) > 22 else {}, row[8], row[9]),
        "final_prompt": row[16],
        "final_output": row[17],
        "created_at": _isoformat(row[18]),
        "completed_at": _isoformat(row[19]),
        "duration_seconds": _float_or_zero(row[20]),
    }


def _serialize_history_detail_row(row) -> dict[str, Any]:
    return {
        "run_id": row[0],
        "run_kind": row[1],
        "evaluation_id": row[2],
        "status": row[3],
        "mode": row[4],
        "requested_by": row[5],
        "technique_key": row[6],
        "technique_name": row[7],
        "technique_family": row[8],
        "technique_surface": row[9],
        "target_kind": row[10],
        "target_model": row[11],
        "attacker_model": row[12],
        "unsafe_intent": row[13],
        "success": bool(row[14]),
        "score": row[15],
        "asr": row[16],
        "matched_detections": row[17],
        "highest_severity": row[18],
        "judge_model": _history_judge_model(row[22], row[11], row[12]),
        "final_prompt": row[19],
        "final_output": row[20],
        "response_metadata": _json_value(row[21], {}),
        "run_metadata": _json_value(row[22], {}),
        "created_at": _isoformat(row[23]),
        "completed_at": _isoformat(row[24]),
        "duration_seconds": _float_or_zero(row[25]),
    }


def _history_conversation_entry(kind: Any, payload: Any) -> dict[str, Any]:
    value = _json_value(payload, {})
    if not isinstance(value, dict):
        value = {"value": value}
    return {**value, "kind": str(kind or value.get("kind", ""))}


def _history_judge_model(run_metadata: Any, target_model: Any, attacker_model: Any) -> str:
    metadata = _json_value(run_metadata, {})
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    if not isinstance(config, dict):
        config = {}

    mode = str(config.get("judge_mode", "") or "").strip().lower()
    if mode == "target":
        return str(config.get("model") or target_model or "")
    if mode == "custom":
        return str(config.get("judge_model") or "")
    if mode == "attacker":
        return str(config.get("attacker_model") or attacker_model or "")
    return str(config.get("attacker_model") or attacker_model or config.get("model") or target_model or "")


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _isoformat(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sanitize_config(config: dict[str, str]) -> dict[str, object]:
    return sanitize_public_mapping(config)


def _sanitize_plan_value(value: Any) -> Any:
    return sanitize_public_value(value)


def _immutable_experiment_creation(experiment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": experiment["name"],
        "description": experiment.get("description", ""),
        "hypothesis": experiment.get("hypothesis", ""),
        "dataset_id": experiment.get("dataset_id", ""),
        "technique_keys": canonical_plain_json(
            experiment.get("technique_keys", [])
        ),
        "target_config": _sanitize_plan_value(experiment.get("target_config", {})),
        "mode": experiment.get("mode", "live"),
        "total_cells": int(experiment.get("total_cells", 0) or 0),
        "repeat_count": int(experiment.get("repeat_count", 1) or 1),
        "max_rounds": int(experiment.get("max_rounds", 5) or 5),
        "experiment_group": experiment.get("experiment_group", ""),
        "created_by": experiment.get("created_by", "demo-user"),
        "plan": canonical_plain_json(experiment.get("plan")),
    }


def _immutable_experiment_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "name": row[0],
        "description": row[1],
        "hypothesis": row[2],
        "dataset_id": row[3],
        "technique_keys": canonical_plain_json(_json_value(row[4], [])),
        "target_config": _sanitize_plan_value(_json_value(row[5], {})),
        "mode": row[6],
        "total_cells": int(row[7]),
        "repeat_count": int(row[8]),
        "max_rounds": int(row[9]),
        "experiment_group": row[10],
        "created_by": row[11],
        "plan": canonical_plain_json(_json_value(row[12], None)),
    }


def _validated_canonical_plan_snapshot(
    value: object,
    *,
    experiment_id: str,
    dataset_id: str,
    repeat_count: int,
    config_rows: list[tuple],
    cell_rows: list[tuple],
) -> dict[str, Any]:
    try:
        return validate_canonical_plan_snapshot(
            value,
            expected_experiment_id=experiment_id,
            expected_dataset_id=dataset_id,
            expected_repeat_count=repeat_count,
            normalized_configs=[
                {
                    "technique_config_id": row[0],
                    "experiment_id": experiment_id,
                    "technique_id": row[1],
                    "technique_version": row[2],
                    "implementation_id": row[3],
                    "preset": row[4],
                    "parameters": row[5],
                    "resolved_spec": row[6],
                    "spec_hash": row[7],
                }
                for row in config_rows
            ],
            normalized_cells=[
                {
                    "cell_id": row[0],
                    "experiment_id": experiment_id,
                    "technique_config_id": row[1],
                    "technique_key": row[2],
                    "prompt_id": row[3],
                    "repeat_index": row[4],
                }
                for row in cell_rows
            ],
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid canonical experiment plan snapshot") from None


def _resolve_attacker_model(target_config: dict[str, str], mode: str) -> str:
    attacker_model = target_config.get("attacker_model", "")
    if attacker_model:
        return attacker_model
    if mode == "live":
        return target_config.get("model", "")
    return ""


_TASK19_REJUDGE_IDENTITY_KEYS = frozenset(
    {
        "rejudge_batch_id",
        "evaluation_run_id",
        "target_evaluation_id",
        "fabricator_evaluation_id",
    }
)


def _has_task19_rejudge_identity(value: Mapping[str, object]) -> bool:
    return any(key in value for key in _TASK19_REJUDGE_IDENTITY_KEYS)


def _optional_rejudge_evaluation_id(
    value: Mapping[str, object],
    key: str,
) -> str:
    raw_evaluation_id = value.get(key)
    if raw_evaluation_id in (None, ""):
        return ""
    if isinstance(raw_evaluation_id, str) and raw_evaluation_id.strip():
        return raw_evaluation_id.strip()
    raise ValueError(
        "rejudge result requires a batch and at least one evaluation identity"
    )


def _rejudge_result_identity(value: Mapping[str, object]) -> tuple[str, str, str]:
    raw_batch_id = value.get("rejudge_batch_id")
    batch_id = raw_batch_id.strip() if isinstance(raw_batch_id, str) else ""
    target_evaluation_id = _optional_rejudge_evaluation_id(
        value,
        "target_evaluation_id",
    )
    fabricator_evaluation_id = _optional_rejudge_evaluation_id(
        value,
        "fabricator_evaluation_id",
    )
    if not batch_id or not (target_evaluation_id or fabricator_evaluation_id):
        raise ValueError(
            "rejudge result requires a batch and at least one evaluation identity"
        )
    return batch_id, target_evaluation_id, fabricator_evaluation_id


def _extract_final_prompt(conversation: list[dict[str, Any]]) -> str:
    if not conversation:
        return ""
    last = conversation[-1]
    kind = last.get("kind")
    if kind == "single_shot":
        return str(last.get("prompt", ""))
    if kind == "round":
        target = last.get("target", {})
        return str(target.get("prompt", "") or target.get("input", ""))
    if kind == "generation":
        candidates = last.get("candidates", [])
        if candidates:
            return str(candidates[0].get("prompt", ""))
    return ""


def _trace_created_at(result: TechniqueRunResult) -> str:
    if result.traces:
        return result.traces[0].timestamp.isoformat()
    return result.completed_at.isoformat()


def _serialize_audit_log_event(item: Any) -> dict[str, Any]:
    return {
        "action": item.action,
        "actor": item.actor,
        "run_id": item.run_id,
        "timestamp": item.timestamp.isoformat(),
        "attributes": dict(item.attributes),
    }
