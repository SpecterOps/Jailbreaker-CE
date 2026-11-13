import asyncio
import csv
import io
import inspect
import json
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, "backend")

from app.api.models import RunSummary
from app.api.store import (
    EvaluationStore,
    ExperimentState,
    _StoredEvidenceJudgeModel,
    _serialize_plan,
)
from app.attacklib import get_attack_technique
from app.evaluation.models import TargetKind, TargetResponse, TechniqueRunResult
from app.experiment_plan_integrity import (
    canonical_dataset_hash,
    cell_id as canonical_cell_id,
    content_hash,
    technique_config_id,
)
from app.persistence import (
    CellRecord,
    CheckpointRecord,
    ExperimentPlanSaveResult,
    ExperimentExecutionOwnershipConflict,
    InMemoryExperimentAttemptPersistence,
    ModelCallRecord,
    NoopRunPersistence,
    NoopTechniqueRuntimePersistence,
    PostgreSQLExperimentAttemptPersistence,
    PostgreSQLRunPersistence,
    TechniqueConfigRecord,
)
from app.persistence.runs import RunPersistence
from app.profile_vault import InMemoryProfileVaultRepository, ProfileVault
from app.techniques.contracts.definition import (
    ImplementationKind,
    JudgeDefinition,
    JudgePurpose,
    ModelRoleDefinition,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
    TechniqueImplementation,
)
from app.techniques.contracts.execution import (
    CorrelationIds,
    ExecutionStatus,
    FailureCategory,
    ResultDisposition,
)
from app.techniques.core.catalog import RegisteredTechnique, TechniqueCatalog
from app.techniques.legacy.registration import definition_from_legacy


class RecordingJudgeClient:
    model = "third-judge"
    api_base_url = "http://judge.test/v1"

    def __init__(self, scores: list[int]) -> None:
        self.scores = list(scores)
        self.calls: list[list[dict[str, str]]] = []

    def chat_with_retry(self, messages, temperature=None, max_tokens=None, response_format=None):
        self.calls.append([dict(message) for message in messages])
        score = self.scores.pop(0)
        return f'{{"score": {score}, "rationale": "score {score}"}}', {"total_tokens": "1"}


class SequencedRejudgeClient:
    model = "third-judge"
    api_base_url = "http://judge.test/v1"

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[dict[str, str]]] = []

    def chat_with_retry(self, messages, temperature=None, max_tokens=None, response_format=None):
        self.calls.append([dict(message) for message in messages])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome), {"total_tokens": "1"}


def test_saved_evidence_judge_model_is_transport_only() -> None:
    class RecordingTransportClient:
        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

        def chat_with_retry(self, messages, **options):
            self.calls.append(
                ([dict(message) for message in messages], dict(options))
            )
            return '{"score": 3, "rationale": "transported"}', {
                "total_tokens": 2
            }

    client = RecordingTransportClient()
    model = _StoredEvidenceJudgeModel(client)
    messages = (
        {"role": "system", "content": "canonical judge instructions"},
        {"role": "user", "content": "canonical target evidence"},
    )
    response_format = {"type": "json_schema"}

    output = asyncio.run(
        model.call(
            role="target_judge",
            messages=messages,
            operation_id="rejudge-transport",
            temperature=0.0,
            response_format=response_format,
        )
    )

    assert output.text == '{"score": 3, "rationale": "transported"}'
    assert output.usage == {"total_tokens": 2}
    assert client.calls == [
        (
            [dict(message) for message in messages],
            {"temperature": 0.0, "response_format": response_format},
        )
    ]
    assert model.traces["rejudge-transport"]["input"] == (
        "[SYSTEM]\ncanonical judge instructions\n\n"
        "[USER]\ncanonical target evidence"
    )


class RecordingTechniqueRuntimePersistence(NoopTechniqueRuntimePersistence):
    def __init__(self) -> None:
        super().__init__()
        self.events = []

    def append_event(self, event) -> None:
        super().append_event(event)
        self.events.append(event)


class FailingRejudgeEventPersistence(NoopTechniqueRuntimePersistence):
    def append_event(self, event) -> None:
        raise RuntimeError("rejudge event write failed")


class RecordingRejudgeRunPersistence(NoopRunPersistence):
    def __init__(self, *, fail_rejudge: bool = False) -> None:
        self.rejudge_writes = []
        self.cell_writes = []
        self.fail_rejudge = fail_rejudge

    def save_experiment_rejudge(self, **kwargs) -> bool:
        if self.fail_rejudge:
            raise RuntimeError("rejudge metadata write failed")
        accepted = super().save_experiment_rejudge(**kwargs)
        if accepted:
            self.rejudge_writes.append(dict(kwargs))
        return accepted

    def save_experiment_cell(
        self,
        *,
        experiment_id,
        cell,
        generation=None,
        owner_id=None,
    ) -> bool:
        accepted = super().save_experiment_cell(
            experiment_id=experiment_id,
            cell=cell,
            generation=generation,
            owner_id=owner_id,
        )
        if accepted:
            self.cell_writes.append((experiment_id, dict(cell)))
        return accepted


class _RecordingAttemptPersistence(InMemoryExperimentAttemptPersistence):
    def __init__(self, events: list[str], *, fail_cells: bool = False) -> None:
        super().__init__()
        self.events = events
        self.fail_cells = fail_cells
        self.config_records = []
        self.cell_records = []

    def upsert_technique_config(self, record):
        self.events.append(f"config:{record.technique_config_id}")
        self.config_records.append(record)
        return super().upsert_technique_config(record)

    def create_cell(self, record):
        self.events.append(f"cell:{record.cell_id}")
        if self.fail_cells:
            raise RuntimeError("planned cell persistence failed")
        self.cell_records.append(record)
        return super().create_cell(record)


class _PlanningRunPersistence(NoopRunPersistence):
    def __init__(
        self,
        events: list[str],
        custom_techniques: list[dict] | None = None,
    ) -> None:
        self.events = events
        self.custom_techniques = list(custom_techniques or [])

    def save_experiment(self, *, experiment) -> None:
        super().save_experiment(experiment=experiment)
        self.events.append("experiment")

    def list_custom_techniques(self) -> list[dict]:
        return list(self.custom_techniques)


class _RestoringRunPersistence(NoopRunPersistence):
    def __init__(self, experiments: list[dict]) -> None:
        self.experiments = experiments

    def load_all_experiments(self) -> list[dict]:
        return list(self.experiments)


def _owned_experiment(
    experiment_id: str,
    owner_id: str,
) -> tuple[NoopRunPersistence, int]:
    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={
            "experiment_id": experiment_id,
            "status": "completed",
            "execution_generation": 0,
            "execution_owner_id": "",
        }
    )
    generation = persistence.begin_experiment_execution(
        experiment_id,
        owner_id=owner_id,
        expected_generation=0,
        active_status="running",
    )
    return persistence, generation


def test_same_execution_owner_replays_generation_without_increment() -> None:
    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={
            "experiment_id": "exp-1",
            "status": "completed",
            "execution_generation": 0,
            "execution_owner_id": "",
        }
    )

    first = persistence.begin_experiment_execution(
        "exp-1",
        owner_id="retry-batch-1",
        expected_generation=0,
        active_status="running",
    )
    replay = persistence.begin_experiment_execution(
        "exp-1",
        owner_id="retry-batch-1",
        expected_generation=0,
        active_status="running",
    )

    assert first == 1
    assert replay == 1


def test_initial_execution_generation_and_owner_are_saved_atomically() -> None:
    persistence = NoopRunPersistence()

    persistence.save_experiment(
        experiment={"experiment_id": "exp-initial", "status": "running"}
    )

    restored = persistence.load_all_experiments()[0]
    assert restored["execution_generation"] == 1
    assert restored["execution_owner_id"] == "initial:exp-initial"


def test_old_generation_cannot_terminalize_new_owner() -> None:
    persistence, first_generation = _owned_experiment(
        "exp-1",
        "retry-batch-1",
    )
    assert persistence.reconcile_owned_experiment_terminal_status(
        "exp-1",
        generation=first_generation,
        owner_id="retry-batch-1",
        status="completed",
    )
    second_generation = persistence.begin_experiment_execution(
        "exp-1",
        owner_id="retry-batch-2",
        expected_generation=first_generation,
        active_status="running",
    )

    assert not persistence.reconcile_owned_experiment_terminal_status(
        "exp-1",
        generation=first_generation,
        owner_id="retry-batch-1",
        status="failed",
        error="stale",
    )
    assert persistence.load_all_experiments()[0]["execution_generation"] == second_generation
    assert persistence.load_all_experiments()[0]["status"] == "running"


def test_different_execution_owner_cannot_take_over_active_generation() -> None:
    persistence, generation = _owned_experiment("exp-1", "retry-batch-1")

    with pytest.raises(RuntimeError, match="ownership conflict"):
        persistence.begin_experiment_execution(
            "exp-1",
            owner_id="retry-batch-2",
            expected_generation=generation,
            active_status="running",
        )

    restored = persistence.load_all_experiments()[0]
    assert restored["execution_generation"] == 1
    assert restored["execution_owner_id"] == "retry-batch-1"


@pytest.mark.parametrize("generation", [True, -1, 1.5])
def test_execution_generation_validation_rejects_boolean_negative_and_non_integer(
    generation,
) -> None:
    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={"experiment_id": "exp-1", "status": "completed"}
    )

    with pytest.raises(ValueError, match="generation"):
        persistence.begin_experiment_execution(
            "exp-1",
            owner_id="retry-batch-1",
            expected_generation=generation,
            active_status="running",
        )


def test_execution_owner_requires_a_non_empty_builtin_string() -> None:
    class Owner(str):
        pass

    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={"experiment_id": "exp-1", "status": "completed"}
    )

    for owner_id in ("", None, Owner("retry-batch-1")):
        with pytest.raises(ValueError, match="owner_id"):
            persistence.begin_experiment_execution(
                "exp-1",
                owner_id=owner_id,
                expected_generation=0,
                active_status="running",
            )


def test_owned_status_contracts_reject_unsupported_statuses() -> None:
    persistence, generation = _owned_experiment("exp-1", "retry-batch-1")

    with pytest.raises(ValueError, match="active"):
        persistence.update_owned_experiment_status(
            "exp-1",
            generation=generation,
            owner_id="retry-batch-1",
            status="completed",
        )
    with pytest.raises(ValueError, match="terminal"):
        persistence.reconcile_owned_experiment_terminal_status(
            "exp-1",
            generation=generation,
            owner_id="retry-batch-1",
            status="running",
        )


def test_owned_active_update_rejects_wrong_owner_and_generation() -> None:
    persistence, generation = _owned_experiment("exp-1", "retry-batch-1")

    assert not persistence.update_owned_experiment_status(
        "exp-1",
        generation=generation + 1,
        owner_id="retry-batch-1",
        status="rejudging",
    )
    assert not persistence.update_owned_experiment_status(
        "exp-1",
        generation=generation,
        owner_id="retry-batch-2",
        status="rejudging",
    )
    assert persistence.load_all_experiments()[0]["status"] == "running"


def test_stale_owned_operation_finish_is_rejected_after_ownership_moves() -> None:
    persistence, generation = _owned_experiment("exp-1", "retry-batch-1")
    assert persistence.admit_owned_experiment_operation(
        "exp-1",
        generation=generation,
        owner_id="retry-batch-1",
        operation_id="operation-1",
        worker_id="worker-1",
        lease_seconds=60,
        operation_kind="rejudge",
    )
    persistence.save_experiment(
        experiment={
            "experiment_id": "exp-1",
            "status": "running",
            "execution_generation": generation + 1,
            "execution_owner_id": "retry-batch-2",
        }
    )

    assert not persistence.finish_owned_experiment_operation(
        "exp-1",
        generation=generation,
        owner_id="retry-batch-1",
        operation_id="operation-1",
        worker_id="worker-1",
        outcome="succeeded",
    )


def test_active_owned_operation_blocks_terminal_reconciliation_until_finished() -> None:
    persistence, generation = _owned_experiment("exp-1", "rejudge-owner")
    assert persistence.admit_owned_experiment_operation(
        "exp-1",
        generation=generation,
        owner_id="rejudge-owner",
        operation_id="operation-1",
        worker_id="worker-1",
        lease_seconds=60,
        operation_kind="rejudge",
    )

    assert not persistence.reconcile_owned_experiment_terminal_status(
        "exp-1",
        generation=generation,
        owner_id="rejudge-owner",
        status="completed",
    )
    assert persistence.finish_owned_experiment_operation(
        "exp-1",
        generation=generation,
        owner_id="rejudge-owner",
        operation_id="operation-1",
        worker_id="worker-1",
        outcome="succeeded",
    )
    assert persistence.reconcile_owned_experiment_terminal_status(
        "exp-1",
        generation=generation,
        owner_id="rejudge-owner",
        status="completed",
    )


def test_owned_operation_same_worker_renews_lease_without_competing_admission() -> None:
    persistence, generation = _owned_experiment("exp-1", "rejudge-owner")
    arguments = {
        "generation": generation,
        "owner_id": "rejudge-owner",
        "operation_id": "operation-1",
        "lease_seconds": 60,
        "operation_kind": "rejudge",
    }
    assert persistence.admit_owned_experiment_operation(
        "exp-1",
        worker_id="worker-1",
        **arguments,
    )

    assert persistence.admit_owned_experiment_operation(
        "exp-1",
        worker_id="worker-1",
        **arguments,
    )
    assert not persistence.admit_owned_experiment_operation(
        "exp-1",
        worker_id="worker-2",
        **arguments,
    )


def test_in_memory_owned_operation_rejects_legacy_generation_zero() -> None:
    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={
            "experiment_id": "exp-legacy-active",
            "status": "running",
            "execution_generation": 0,
            "execution_owner_id": "legacy-owner",
        }
    )

    with pytest.raises(ValueError, match="positive"):
        persistence.admit_owned_experiment_operation(
            "exp-legacy-active",
            generation=0,
            owner_id="legacy-owner",
            operation_id="operation-legacy",
            worker_id="worker-legacy",
            lease_seconds=60,
            operation_kind="rejudge",
        )


def test_terminal_reconciliation_retries_same_owner_after_operation_closes(
    monkeypatch,
) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)
    state.status = "completed"
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=state._execution_generation,
        owner_id=state._execution_owner_id,
        status="completed",
    )
    owner_id = "rejudge-crash-owner"
    generation = store.persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id=owner_id,
        expected_generation=state._execution_generation,
        active_status="rejudging",
    )
    state._execution_generation = generation
    state._execution_owner_id = owner_id
    state.status = "failed"
    assert store.persistence.admit_owned_experiment_operation(
        state.experiment_id,
        generation=generation,
        owner_id=owner_id,
        operation_id="operation-crash-window",
        worker_id="worker-crash-window",
        lease_seconds=60,
        operation_kind="rejudge",
    )
    timers = _capture_retry_timers(monkeypatch)

    assert not store._reconcile_terminal_experiment_status(
        state.experiment_id,
        "failed",
        generation=generation,
        owner_id=owner_id,
        error="worker interrupted",
    )

    assert state.experiment_id in store._terminal_status_reconciliation_obligations
    assert len(timers) == 1
    assert store.persistence.load_all_experiments()[0]["status"] == "rejudging"
    assert store.persistence.finish_owned_experiment_operation(
        state.experiment_id,
        generation=generation,
        owner_id=owner_id,
        operation_id="operation-crash-window",
        worker_id="worker-crash-window",
        outcome="failed",
    )
    timers[0].fire()

    assert state.experiment_id not in store._terminal_status_reconciliation_obligations
    durable = store.persistence.load_all_experiments()[0]
    assert durable["status"] == "failed"
    assert durable["error"] == "worker interrupted"


def test_same_owner_terminal_replay_is_idempotent_and_conflict_is_rejected() -> None:
    persistence, generation = _owned_experiment("exp-1", "retry-batch-1")

    assert persistence.reconcile_owned_experiment_terminal_status(
        "exp-1",
        generation=generation,
        owner_id="retry-batch-1",
        status="completed",
        report={"winner": "first"},
    )
    assert persistence.reconcile_owned_experiment_terminal_status(
        "exp-1",
        generation=generation,
        owner_id="retry-batch-1",
        status="completed",
        report={"winner": "replay"},
    )
    assert not persistence.reconcile_owned_experiment_terminal_status(
        "exp-1",
        generation=generation,
        owner_id="retry-batch-1",
        status="failed",
        error="conflicting terminal result",
    )

    restored = persistence.load_all_experiments()[0]
    assert restored["status"] == "completed"
    assert restored["report"] == {"winner": "first"}
    assert restored.get("error") is None


def test_owned_contract_base_methods_have_no_unowned_fallback() -> None:
    with pytest.raises(NotImplementedError):
        RunPersistence.begin_experiment_execution(
            object(),
            "exp-1",
            owner_id="retry-batch-1",
            expected_generation=0,
            active_status="running",
        )


def test_public_experiment_summaries_exclude_execution_ownership() -> None:
    persistence, _ = _owned_experiment("exp-1", "retry-batch-1")

    summary = persistence.list_experiments()[0]

    assert "execution_generation" not in summary
    assert "execution_owner_id" not in summary


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_execution_owner_and_generation_match_in_memory_outcomes() -> None:
    import psycopg

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments CASCADE"
        )
        connection.execute(
            "INSERT INTO experiments "
            "(experiment_id, name, status, execution_generation, execution_owner_id) "
            "VALUES ('exp-owner-pg', 'ownership', 'completed', 0, '')"
        )
    persistence = PostgreSQLRunPersistence(dsn)

    first = persistence.begin_experiment_execution(
        "exp-owner-pg",
        owner_id="retry-batch-1",
        expected_generation=0,
        active_status="running",
    )
    replay = persistence.begin_experiment_execution(
        "exp-owner-pg",
        owner_id="retry-batch-1",
        expected_generation=0,
        active_status="running",
    )
    assert persistence.reconcile_owned_experiment_terminal_status(
        "exp-owner-pg",
        generation=first,
        owner_id="retry-batch-1",
        status="completed",
    )
    second = persistence.begin_experiment_execution(
        "exp-owner-pg",
        owner_id="retry-batch-2",
        expected_generation=first,
        active_status="running",
    )

    assert first == replay == 1
    assert second == 2
    assert not persistence.reconcile_owned_experiment_terminal_status(
        "exp-owner-pg",
        generation=first,
        owner_id="retry-batch-1",
        status="failed",
    )
    restored = persistence.load_all_experiments()[0]
    assert restored["execution_generation"] == 2
    assert restored["execution_owner_id"] == "retry-batch-2"
    assert restored["status"] == "running"


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_active_operation_blocks_terminal_reconciliation() -> None:
    import psycopg

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments CASCADE"
        )
        connection.execute(
            "INSERT INTO experiments "
            "(experiment_id, name, status, execution_generation, execution_owner_id) "
            "VALUES ('exp-operation-pg', 'operation fence', 'rejudging', 1, "
            "'rejudge-owner')"
        )
    persistence = PostgreSQLRunPersistence(dsn)

    assert persistence.admit_owned_experiment_operation(
        "exp-operation-pg",
        generation=1,
        owner_id="rejudge-owner",
        operation_id="operation-pg",
        worker_id="worker-pg",
        lease_seconds=60,
        operation_kind="rejudge",
    )
    assert persistence.admit_owned_experiment_operation(
        "exp-operation-pg",
        generation=1,
        owner_id="rejudge-owner",
        operation_id="operation-pg",
        worker_id="worker-pg",
        lease_seconds=60,
        operation_kind="rejudge",
    )
    assert not persistence.admit_owned_experiment_operation(
        "exp-operation-pg",
        generation=1,
        owner_id="rejudge-owner",
        operation_id="operation-pg",
        worker_id="competing-worker",
        lease_seconds=60,
        operation_kind="rejudge",
    )
    assert not persistence.reconcile_owned_experiment_terminal_status(
        "exp-operation-pg",
        generation=1,
        owner_id="rejudge-owner",
        status="completed",
    )
    assert persistence.finish_owned_experiment_operation(
        "exp-operation-pg",
        generation=1,
        owner_id="rejudge-owner",
        operation_id="operation-pg",
        worker_id="worker-pg",
        outcome="succeeded",
    )
    assert persistence.reconcile_owned_experiment_terminal_status(
        "exp-operation-pg",
        generation=1,
        owner_id="rejudge-owner",
        status="completed",
    )


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_stale_rejudge_is_rejected_before_cell_projection() -> None:
    import psycopg

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments CASCADE"
        )
        connection.execute(
            "INSERT INTO experiments "
            "(experiment_id, name, status, execution_generation, execution_owner_id) "
            "VALUES ('exp-rejudge-pg', 'stale rejudge', 'running', 2, "
            "'retry-owner')"
        )
    persistence = PostgreSQLRunPersistence(dsn)

    accepted = persistence.save_experiment_rejudge(
        experiment_id="exp-rejudge-pg",
        generation=1,
        owner_id="stale-rejudge-owner",
        cell_id="missing-cell-must-not-be-read",
        selected_attempt_id="stale-attempt",
        rejudge_result={
            "rejudge_batch_id": "stale-rejudge-batch",
            "target_evaluation_id": "stale-target-evaluation",
        },
        original_score=80,
        original_asr=100,
    )

    assert accepted is False


class _MutatingSnapshotAttemptPersistence(InMemoryExperimentAttemptPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_cell_id: str | None = None
        self.selected_attempt_ids: tuple[str, str] | None = None
        self.continuous = False
        self.get_cell_calls = 0
        self.list_attempt_calls = 0

    def arm(
        self,
        *,
        cell_id: str,
        selected_attempt_ids: tuple[str, str],
        continuous: bool,
    ) -> None:
        self.snapshot_cell_id = cell_id
        self.selected_attempt_ids = selected_attempt_ids
        self.continuous = continuous
        self.get_cell_calls = 0
        self.list_attempt_calls = 0

    def get_cell(self, cell_id: str):
        if cell_id == self.snapshot_cell_id:
            self.get_cell_calls += 1
        return super().get_cell(cell_id)

    def list_attempts(self, cell_id: str):
        attempts = super().list_attempts(cell_id)
        if cell_id != self.snapshot_cell_id or self.selected_attempt_ids is None:
            return attempts
        self.list_attempt_calls += 1
        if self.continuous or self.list_attempt_calls == 1:
            with self._lock:
                cell = self._cells[cell_id]
                first, second = self.selected_attempt_ids
                selected = second if cell.selected_attempt_id == first else first
                self._cells[cell_id] = replace(cell, selected_attempt_id=selected)
        return attempts


class _AtomicPlanRunPersistence(PostgreSQLRunPersistence):
    def __init__(self, dsn: str, *, fail_after_cells: int | None = None) -> None:
        self._dsn = dsn
        self.fail_after_cells = fail_after_cells
        self.events: list[str] = []
        self.persisted_experiments: list[dict] = []

    def load_all_experiments(self) -> list[dict]:
        return list(self.persisted_experiments)

    def save_experiment_dataset(self, *, dataset) -> None:
        return None

    def save_experiment(self, *, experiment) -> None:
        self.events.append("split-experiment")

    def save_experiment_plan(self, *, experiment, technique_configs, cells):
        existing = next(
            (
                item
                for item in self.persisted_experiments
                if item["experiment_id"] == experiment["experiment_id"]
            ),
            None,
        )
        was_created = existing is None
        staged = [dict(experiment)]
        self.events.append("atomic-experiment")
        for config in technique_configs:
            self.events.append(f"atomic-config:{config.technique_config_id}")
        for index, cell in enumerate(cells, 1):
            self.events.append(f"atomic-cell:{cell.cell_id}")
            if self.fail_after_cells == index:
                raise RuntimeError("atomic planned cell persistence failed")
        if existing is None:
            self.persisted_experiments = staged
            existing = staged[0]
        return ExperimentPlanSaveResult(
            was_created=was_created,
            status=str(existing["status"]),
            execution_generation=int(existing["execution_generation"]),
            execution_owner_id=str(existing["execution_owner_id"]),
        )


class _AtomicPlanAttemptPersistence(PostgreSQLExperimentAttemptPersistence):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self.events: list[str] = []

    def upsert_technique_config(self, record):
        self.events.append("split-config")
        return record

    def create_cell(self, record):
        self.events.append("split-cell")
        return record


def _planning_catalog() -> TechniqueCatalog:
    catalog = TechniqueCatalog()
    definition = TechniqueDefinition(
        identity=TechniqueIdentity(
            id="probe",
            version="1.2.3",
            name="Probe",
            family="test",
        ),
        summary="Planning test technique.",
        capabilities=frozenset({TechniqueCapability.SINGLE_SHOT}),
        contract_version="7",
        aliases=("probe-alias",),
        parameters_schema={
            "type": "object",
            "properties": {
                "threshold": {"type": "integer", "default": 2},
                "nested": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean", "default": False}},
                    "default": {},
                },
            },
            "additionalProperties": False,
        },
        presets={"high": {"threshold": 8}},
        model_roles=(
            ModelRoleDefinition(id="target", required=True),
            ModelRoleDefinition(id="target_judge", required=True),
        ),
        judges=(
            JudgeDefinition(
                id="target_judge",
                purposes=frozenset({JudgePurpose.OFFICIAL_SCORE}),
                role="target_judge",
                policy={"threshold": 0.7},
            ),
        ),
        recommended_limits={"max_rounds": 9, "max_tokens": 2048},
    )
    implementation = TechniqueImplementation(
        id="probe-python",
        technique_id="probe",
        technique_version="1.2.3",
        kind=ImplementationKind.PYTHON,
        entrypoint="tests.probe:run",
        contract_version="7",
    )
    catalog.register(
        RegisteredTechnique(
            definition=definition,
            implementation=implementation,
            manifest_hash="a" * 64,
            asset_hashes={"workflow.yaml": "b" * 64},
        )
    )
    catalog.set_default("probe", implementation_id="probe-python", version="1.2.3")
    return catalog


def _profile_vault_with_role_profiles() -> tuple[ProfileVault, dict[str, str]]:
    vault = ProfileVault(InMemoryProfileVaultRepository())
    vault.create("task-15-profile-refresh")
    profile_ids: dict[str, str] = {}
    for role in ("target", "attacker", "judge"):
        profile = vault.save_profile(
            {
                "profile_id": f"{role}-profile",
                "name": f"Planned {role}",
                "provider": f"planned-{role}-provider",
                "base_url": f"https://planned-{role}.test/v1",
                "model": f"planned-{role}-model",
                "scope": role,
                "api_key": f"stale-{role}-key",
            }
        )
        profile_ids[role] = profile["profile_id"]
    return vault, profile_ids


def _rotate_role_profiles(vault: ProfileVault, profile_ids: Mapping[str, str]) -> None:
    for role, profile_id in profile_ids.items():
        vault.save_profile(
            {
                "profile_id": profile_id,
                "name": f"Drifted {role}",
                "provider": f"drifted-{role}-provider",
                "base_url": f"https://drifted-{role}.test/v1",
                "model": f"drifted-{role}-model",
                "scope": role,
                "api_key": f"refreshed-{role}-key",
            }
        )


def _versioned_legacy_catalog() -> TechniqueCatalog:
    catalog = TechniqueCatalog()
    technique_cls = get_attack_technique("system_prompt_extraction")
    base_definition = definition_from_legacy(technique_cls)
    for version, implementation_id, digest in (
        ("1.0.0", "legacy-v1", "1" * 64),
        ("2.0.0", "legacy-v2", "2" * 64),
    ):
        definition = replace(
            base_definition,
            identity=replace(base_definition.identity, version=version),
        )
        catalog.register(
            RegisteredTechnique(
                definition=definition,
                implementation=TechniqueImplementation(
                    id=implementation_id,
                    technique_id=definition.identity.id,
                    technique_version=version,
                    kind=ImplementationKind.LEGACY,
                    entrypoint=(
                        f"{technique_cls.__module__}:{technique_cls.__qualname__}"
                    ),
                ),
                manifest_hash=digest,
            )
        )
    catalog.set_default(
        "system_prompt_extraction",
        implementation_id="legacy-v1",
        version="1.0.0",
    )
    return catalog


def _create_planned_experiment(store: EvaluationStore, **overrides):
    dataset = store.create_dataset(
        name="Planning dataset",
        prompts=[
            {"text": "intent one", "category": "cat-a"},
            {"text": "intent two", "category": "cat-b"},
        ],
    )
    request = {
        "name": "Pinned plan",
        "dataset_id": dataset["dataset_id"],
        "technique_keys": ["probe"],
        "technique_configs": [
            {"technique_key": "probe", "preset": "high", "parameters": {"threshold": 4}},
            {"technique_key": "probe", "preset": "high", "parameters": {"threshold": 6}},
        ],
        "target_config": {
            "provider": "openai",
            "target_profile_id": "target-profile",
            "model": "target-model",
            "judge_mode": "target",
            "api_key": "TOP-SECRET-TARGET-KEY",
            "access_token": "TOP-SECRET-ACCESS-TOKEN",
        },
        "mode": "demo",
        "repeat_count": 3,
        "max_rounds": 5,
    }
    request.update(overrides)
    return store.create_experiment(**request)


def _assert_secret_free(value) -> None:
    assert value != "TOP-SECRET-TARGET-KEY"
    assert value != "TOP-SECRET-ACCESS-TOKEN"
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = key.lower()
            assert "secret" not in normalized
            assert "api_key" not in normalized
            assert not normalized.endswith("access_token")
            assert not normalized.endswith("password")
            _assert_secret_free(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_secret_free(nested)


def _experiment_state(*, repeat_count: int = 3) -> ExperimentState:
    return ExperimentState(
        experiment_id="exp-repeat",
        name="Repeated Crescendo-Liar",
        description="",
        hypothesis="",
        status="running",
        mode="live",
        dataset={
            "dataset_id": "ds-1",
            "name": "dataset",
            "prompts": [
                {"id": "p1", "text": "intent one", "category": "cat-a"},
                {"id": "p2", "text": "intent two", "category": "cat-b"},
            ],
        },
        technique_keys=["crescendo_liar"],
        target_config={"model": "target-model"},
        total_cells=2 * repeat_count,
        repeat_count=repeat_count,
        max_rounds=5,
        experiment_group="crescendo-liar-main",
    )


def _attempt_reporting_store(
    monkeypatch,
    *,
    technique_configs: list[dict] | None = None,
    attempt_persistence: InMemoryExperimentAttemptPersistence | None = None,
    prompts: list[dict] | None = None,
    run_persistence: RunPersistence | None = None,
    runtime_persistence=None,
) -> tuple[EvaluationStore, InMemoryExperimentAttemptPersistence, ExperimentState]:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    attempts = attempt_persistence or InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=run_persistence or NoopRunPersistence(),
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
        technique_runtime_persistence=runtime_persistence,
    )
    dataset = store.create_dataset(
        name="Attempt reporting",
        prompts=prompts if prompts is not None else [
            {"text": "first intent", "category": "cat-a"},
            {"text": "second intent", "category": "cat-b"},
        ],
    )
    created = store.create_experiment(
        name="Attempt-aware report",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        technique_configs=technique_configs or [{"technique_key": "probe", "parameters": {"threshold": 4}}],
        target_config={"model": "target-model", "judge_threshold": "7"},
        mode="demo",
        repeat_count=1,
    )
    return store, attempts, store._experiments[created["experiment_id"]]


def _complete_reporting_attempt(
    attempts: InMemoryExperimentAttemptPersistence,
    state: ExperimentState,
    *,
    cell_index: int = 0,
    execution_status: ExecutionStatus,
    result_disposition: ResultDisposition,
    score: int | float | None = None,
    failure_category: FailureCategory | None = None,
    run_id: str | None = None,
):
    assert state.plan is not None
    planned_cell = state.plan.cells[cell_index]
    config = next(
        config
        for config in state.plan.technique_configs
        if config.technique_config_id == planned_cell.technique_config_id
    )
    attempt = attempts.create_attempt(
        planned_cell.cell_id,
        retry_kind="initial" if not attempts.list_attempts(planned_cell.cell_id) else "restart",
        spec_hash=config.spec_hash,
    )
    return attempts.complete_attempt(
        attempt.attempt_id,
        execution_status=execution_status,
        result_disposition=result_disposition,
        score=score,
        failure_category=failure_category,
        run_id=run_id,
    )


def _flattened_planned_result(
    state: ExperimentState,
    *,
    cell_index: int = 0,
    run_id: str = "run-selected",
    status: str = "completed",
    score: int | float = 80,
    asr: int = 100,
) -> dict:
    assert state.plan is not None
    planned_cell = state.plan.cells[cell_index]
    config = next(
        config
        for config in state.plan.technique_configs
        if config.technique_config_id == planned_cell.technique_config_id
    )
    return {
        "cell_id": planned_cell.cell_id,
        "technique_config_id": planned_cell.technique_config_id,
        "spec_hash": config.spec_hash,
        "technique_key": planned_cell.technique_key,
        "prompt_id": planned_cell.prompt_id,
        "prompt_text": planned_cell.prompt_text,
        "prompt_category": planned_cell.prompt_category,
        "repeat_index": planned_cell.repeat_index,
        "run_id": run_id,
        "status": status,
        "score": score,
        "asr": asr,
        "matched_detections": 0,
        "highest_severity": "info",
        "conversation": [],
        "error": "",
    }


def _finish_planned_state(state: ExperimentState) -> None:
    state.status = "completed"
    state.error = None


def _durable_retry_restart_fixture(monkeypatch):
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    persistence = NoopRunPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    source = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = source.create_dataset(
        name="Restart recovery",
        prompts=[{"text": "recover this intent", "category": "restart"}],
    )
    created = source.create_experiment(
        name="Restart recovery",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = source._experiments[created["experiment_id"]]
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.PROVIDER,
    )
    state.status = "completed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=state._execution_generation,
        owner_id=state._execution_owner_id,
        status="completed",
    )
    durable = persistence._experiment_lifecycle[state.experiment_id]
    durable["dataset"] = state.dataset
    durable["cell_results"] = list(state.cell_results)
    durable["completed_cells"] = state.completed_cells
    durable["created_at"] = state.created_at
    return persistence, attempts, state, failed


def _completed_retry_summary(**kwargs):
    return (
        RunSummary(
            run_id="run-recovered-after-restart",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=10,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ),
        object(),
    )


class _BlockingGenerationActiveCheck(InMemoryExperimentAttemptPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.block_next_check = False
        self.check_started = threading.Event()
        self.release_check = threading.Event()

    def has_active_attempts(
        self,
        experiment_id: str,
        *,
        execution_generation: int,
    ) -> bool:
        if not self.block_next_check:
            return super().has_active_attempts(
                experiment_id,
                execution_generation=execution_generation,
            )
        self.block_next_check = False
        self.check_started.set()
        assert self.release_check.wait(timeout=5)
        return True


def test_restart_resumes_null_generation_batch_before_dispatch(monkeypatch) -> None:
    persistence, attempts, state, failed = _durable_retry_restart_fixture(monkeypatch)
    batch = attempts.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        requested_by="operator",
        request_id="restart-null-generation",
    )
    assert batch.execution_generation is None
    started: list[threading.Thread] = []
    provider_calls: list[dict] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: started.append(thread),
    )

    def execute(self, **kwargs):
        provider_calls.append(kwargs)
        return _completed_retry_summary(**kwargs)

    monkeypatch.setattr(EvaluationStore, "_execute_single", execute)

    EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    assert len(started) == 1
    started[0].run()

    recovered = attempts.get_retry_batch(batch.retry_batch_id)
    durable = persistence.load_all_experiments()[0]
    assert recovered.execution_generation == durable["execution_generation"]
    assert durable["execution_owner_id"] == batch.retry_batch_id
    assert len(provider_calls) == 1


def test_restart_replays_owner_matching_null_generation_crash_window(
    monkeypatch,
) -> None:
    persistence, attempts, state, failed = _durable_retry_restart_fixture(monkeypatch)
    batch = attempts.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        requested_by="operator",
        request_id="restart-owner-null-generation",
    )
    generation = persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id=batch.retry_batch_id,
        expected_generation=state._execution_generation,
        active_status="running",
    )
    assert attempts.get_retry_batch(batch.retry_batch_id).execution_generation is None
    started: list[threading.Thread] = []
    provider_calls: list[dict] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: started.append(thread),
    )

    def execute(self, **kwargs):
        provider_calls.append(kwargs)
        return _completed_retry_summary(**kwargs)

    monkeypatch.setattr(EvaluationStore, "_execute_single", execute)

    restored = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    durable_before_worker = persistence.load_all_experiments()[0]
    rebound = attempts.get_retry_batch(batch.retry_batch_id)
    assert restored._experiments[state.experiment_id].status == "running"
    assert durable_before_worker["status"] == "running"
    assert durable_before_worker["execution_generation"] == generation
    assert durable_before_worker["execution_owner_id"] == batch.retry_batch_id
    assert rebound.execution_generation == generation
    assert len(started) == 1

    started[0].run()

    assert len(provider_calls) == 1
    assert persistence.load_all_experiments()[0]["execution_generation"] == generation


def test_restart_recovers_matching_current_generation_owner(monkeypatch) -> None:
    persistence, attempts, state, failed = _durable_retry_restart_fixture(monkeypatch)
    batch = attempts.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        requested_by="operator",
        request_id="restart-current-owner",
    )
    generation = persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id=batch.retry_batch_id,
        expected_generation=state._execution_generation,
        active_status="running",
    )
    attempts.bind_retry_batch_generation(batch.retry_batch_id, generation)
    started: list[threading.Thread] = []
    provider_calls: list[dict] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: started.append(thread),
    )

    def execute(self, **kwargs):
        provider_calls.append(kwargs)
        return _completed_retry_summary(**kwargs)

    monkeypatch.setattr(EvaluationStore, "_execute_single", execute)

    restored = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    assert restored._experiments[state.experiment_id].status == "running"
    assert len(started) == 1
    started[0].run()

    assert len(provider_calls) == 1
    assert persistence.load_all_experiments()[0]["execution_generation"] == generation


def test_restart_does_not_recover_batch_from_older_generation(monkeypatch) -> None:
    persistence, attempts, state, failed = _durable_retry_restart_fixture(monkeypatch)
    stale = attempts.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        requested_by="operator",
        request_id="restart-stale-generation",
    )
    first_generation = persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id=stale.retry_batch_id,
        expected_generation=state._execution_generation,
        active_status="running",
    )
    attempts.bind_retry_batch_generation(stale.retry_batch_id, first_generation)
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=first_generation,
        owner_id=stale.retry_batch_id,
        status="completed",
    )
    current_generation = persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id="retry-batch-current-owner",
        expected_generation=first_generation,
        active_status="running",
    )
    workers: list[tuple] = []
    provider_calls: list[dict] = []
    monkeypatch.setattr(
        EvaluationStore,
        "_start_retry_batch_worker",
        lambda self, *args, **kwargs: workers.append((args, kwargs)),
    )

    def execute(self, **kwargs):
        provider_calls.append(kwargs)
        return _completed_retry_summary(**kwargs)

    monkeypatch.setattr(EvaluationStore, "_execute_single", execute)

    EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    durable = persistence.load_all_experiments()[0]
    assert workers == []
    assert provider_calls == []
    assert durable["execution_generation"] == current_generation
    assert durable["execution_owner_id"] == "retry-batch-current-owner"


def test_restart_terminalizes_interrupted_legacy_generation_tuple(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={
            "experiment_id": "exp-9",
            "name": "Legacy restart",
            "status": "running",
            "completed_cells": 0,
            "execution_generation": 0,
            "execution_owner_id": "",
        }
    )

    restored = EvaluationStore(persistence=persistence)

    durable = persistence.load_all_experiments()[0]
    assert restored.get_experiment("exp-9")["status"] == "failed"
    assert durable["status"] == "failed"
    assert durable["execution_generation"] == 0
    assert durable["execution_owner_id"] == ""


def test_restart_recovers_queued_initial_attempt_before_owned_failure(
    monkeypatch,
) -> None:
    persistence = NoopRunPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    with monkeypatch.context() as planning:
        planning.setattr("app.api.store.threading.Thread.start", lambda _thread: None)
        original = EvaluationStore(
            persistence=persistence,
            catalog=_planning_catalog(),
            experiment_attempt_persistence=attempts,
        )
        dataset = original.create_dataset(
            name="queued initial recovery",
            prompts=[{"text": "intent"}],
        )
        created = original.create_experiment(
            name="queued initial recovery",
            dataset_id=dataset["dataset_id"],
            technique_keys=["probe"],
            target_config={"model": "target-model"},
            mode="demo",
        )
    state = original._experiments[created["experiment_id"]]
    record = persistence.load_all_experiments()[0]
    record["dataset"] = state.dataset
    record["cell_results"] = []
    record["completed_cells"] = 0
    persistence.save_experiment(experiment=record)
    planned_cell = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    queued = attempts.create_attempt(
        planned_cell.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )
    timers = _capture_retry_timers(monkeypatch)

    restored = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    public = restored.get_experiment(state.experiment_id)
    assert public is not None
    assert public["status"] == "failed"
    recovered = attempts.get_attempt(queued.attempt_id)
    assert recovered.execution_status is ExecutionStatus.FAILED
    assert recovered.failure_category is FailureCategory.WORKER_LOST
    durable = persistence.load_all_experiments()[0]
    assert durable["status"] == "failed"
    attempt_rows = durable["report"]["canonical_rows"][0]["attempts"]
    assert attempt_rows[0]["failure_category"] == FailureCategory.WORKER_LOST.value
    assert timers == []


def test_restart_recovers_expired_initial_attempt_before_owned_failure(
    monkeypatch,
) -> None:
    persistence = NoopRunPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    with monkeypatch.context() as planning:
        planning.setattr("app.api.store.threading.Thread.start", lambda _thread: None)
        original = EvaluationStore(
            persistence=persistence,
            catalog=_planning_catalog(),
            experiment_attempt_persistence=attempts,
        )
        dataset = original.create_dataset(
            name="expired initial recovery",
            prompts=[{"text": "intent"}],
        )
        created = original.create_experiment(
            name="expired initial recovery",
            dataset_id=dataset["dataset_id"],
            technique_keys=["probe"],
            target_config={"model": "target-model"},
            mode="demo",
        )
    state = original._experiments[created["experiment_id"]]
    record = persistence.load_all_experiments()[0]
    record["dataset"] = state.dataset
    record["cell_results"] = []
    record["completed_cells"] = 0
    persistence.save_experiment(experiment=record)
    planned_cell = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    attempt = attempts.create_attempt(
        planned_cell.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )
    assert attempts.claim_attempt(
        attempt.attempt_id,
        worker_id="interrupted-worker",
        lease_seconds=30,
        experiment_id=state.experiment_id,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )
    running = attempts.get_attempt(attempt.attempt_id)
    assert running.lease_expires_at is not None
    attempts._attempts[attempt.attempt_id] = replace(
        running,
        lease_expires_at=running.lease_expires_at - timedelta(seconds=31),
    )
    timers = _capture_retry_timers(monkeypatch)

    EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    recovered = attempts.get_attempt(attempt.attempt_id)
    assert recovered.execution_status is ExecutionStatus.FAILED
    assert recovered.failure_category is FailureCategory.WORKER_LOST
    durable = persistence.load_all_experiments()[0]
    assert durable["status"] == "failed"
    assert timers == []


def test_restart_waits_for_live_initial_attempt_lease_before_owned_failure(
    monkeypatch,
) -> None:
    persistence = NoopRunPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    with monkeypatch.context() as planning:
        planning.setattr("app.api.store.threading.Thread.start", lambda _thread: None)
        original = EvaluationStore(
            persistence=persistence,
            catalog=_planning_catalog(),
            experiment_attempt_persistence=attempts,
        )
        dataset = original.create_dataset(
            name="live initial recovery",
            prompts=[{"text": "intent"}],
        )
        created = original.create_experiment(
            name="live initial recovery",
            dataset_id=dataset["dataset_id"],
            technique_keys=["probe"],
            target_config={"model": "target-model"},
            mode="demo",
        )
    state = original._experiments[created["experiment_id"]]
    record = persistence.load_all_experiments()[0]
    record["dataset"] = state.dataset
    record["cell_results"] = []
    record["completed_cells"] = 0
    persistence.save_experiment(experiment=record)
    planned_cell = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    attempt = attempts.create_attempt(
        planned_cell.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )
    assert attempts.claim_attempt(
        attempt.attempt_id,
        worker_id="interrupted-worker",
        lease_seconds=30,
        experiment_id=state.experiment_id,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )
    timers = _capture_retry_timers(monkeypatch)

    EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    assert attempts.get_attempt(attempt.attempt_id).execution_status is ExecutionStatus.RUNNING
    assert persistence.load_all_experiments()[0]["status"] == "running"
    assert len(timers) == 1
    running = attempts.get_attempt(attempt.attempt_id)
    assert running.lease_expires_at is not None
    attempts._attempts[attempt.attempt_id] = replace(
        running,
        lease_expires_at=running.lease_expires_at - timedelta(seconds=31),
    )

    timers[0].fire()

    recovered = attempts.get_attempt(attempt.attempt_id)
    assert recovered.execution_status is ExecutionStatus.FAILED
    assert recovered.failure_category is FailureCategory.WORKER_LOST
    assert persistence.load_all_experiments()[0]["status"] == "failed"


def test_restart_legacy_terminalization_rejects_owned_generation() -> None:
    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={
            "experiment_id": "exp-9",
            "status": "running",
            "execution_generation": 1,
            "execution_owner_id": "initial:exp-9",
        }
    )

    assert not persistence.reconcile_legacy_interrupted_experiment("exp-9")
    assert persistence.load_all_experiments()[0]["status"] == "running"


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_legacy_restart_terminalizes_only_exact_zero_owner_tuple() -> None:
    import psycopg

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        persistence = PostgreSQLRunPersistence(dsn)
        persistence.ensure_schema(connection)
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments CASCADE"
        )
        connection.execute(
            "INSERT INTO experiments "
            "(experiment_id, name, status, execution_generation, execution_owner_id) "
            "VALUES ('exp-legacy', 'legacy', 'running', 0, ''), "
            "('exp-owned', 'owned', 'running', 1, 'initial:exp-owned')"
        )

    persistence = PostgreSQLRunPersistence(dsn)
    assert persistence.reconcile_legacy_interrupted_experiment("exp-legacy")
    assert not persistence.reconcile_legacy_interrupted_experiment("exp-owned")

    with psycopg.connect(dsn, autocommit=True) as connection:
        rows = connection.execute(
            "SELECT experiment_id, status, execution_generation, execution_owner_id "
            "FROM experiments ORDER BY experiment_id"
        ).fetchall()
    assert rows == [
        ("exp-legacy", "failed", 0, ""),
        ("exp-owned", "running", 1, "initial:exp-owned"),
    ]


def test_concurrent_owner_begin_has_exactly_one_different_owner_winner() -> None:
    persistence = NoopRunPersistence()
    persistence.save_experiment(
        experiment={
            "experiment_id": "exp-concurrent-owner",
            "status": "completed",
            "execution_generation": 0,
            "execution_owner_id": "",
        }
    )
    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, int | Exception]] = []

    def begin(owner_id: str) -> None:
        barrier.wait()
        try:
            outcome: int | Exception = persistence.begin_experiment_execution(
                "exp-concurrent-owner",
                owner_id=owner_id,
                expected_generation=0,
                active_status="running",
            )
        except Exception as exc:  # noqa: BLE001
            outcome = exc
        outcomes.append((owner_id, outcome))

    workers = [
        threading.Thread(target=begin, args=(owner_id,))
        for owner_id in ("retry-batch-a", "retry-batch-b")
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=5)

    winners = [(owner, value) for owner, value in outcomes if isinstance(value, int)]
    losers = [(owner, value) for owner, value in outcomes if isinstance(value, Exception)]
    assert winners == [(persistence.load_all_experiments()[0]["execution_owner_id"], 1)]
    assert len(losers) == 1
    assert "ownership conflict" in str(losers[0][1])


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_concurrent_owner_begin_has_one_round_trip_winner() -> None:
    import psycopg

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        persistence = PostgreSQLRunPersistence(dsn)
        persistence.ensure_schema(connection)
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments CASCADE"
        )
        connection.execute(
            "INSERT INTO experiments (experiment_id, name, status) "
            "VALUES ('exp-concurrent-owner', 'concurrent owner', 'completed')"
        )

    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, int | Exception]] = []

    def begin(owner_id: str) -> None:
        barrier.wait()
        try:
            outcome: int | Exception = PostgreSQLRunPersistence(
                dsn
            ).begin_experiment_execution(
                "exp-concurrent-owner",
                owner_id=owner_id,
                expected_generation=0,
                active_status="running",
            )
        except Exception as exc:  # noqa: BLE001
            outcome = exc
        outcomes.append((owner_id, outcome))

    workers = [
        threading.Thread(target=begin, args=(owner_id,))
        for owner_id in ("retry-batch-a", "retry-batch-b")
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=5)

    durable = PostgreSQLRunPersistence(dsn).load_all_experiments()[0]
    winners = [(owner, value) for owner, value in outcomes if isinstance(value, int)]
    losers = [(owner, value) for owner, value in outcomes if isinstance(value, Exception)]
    assert winners == [(durable["execution_owner_id"], 1)]
    assert len(losers) == 1
    assert "ownership conflict" in str(losers[0][1])
    assert durable["execution_generation"] == 1


def test_restart_generation_and_owner_remain_private_in_public_json(monkeypatch) -> None:
    persistence, attempts, state, _ = _durable_retry_restart_fixture(monkeypatch)

    restored = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    public = restored.get_experiment(state.experiment_id)

    assert "execution_generation" not in public
    assert "execution_owner_id" not in public


def test_restart_generation_completion_waits_for_current_active_attempt(
    monkeypatch,
) -> None:
    persistence, attempts, state, failed = _durable_retry_restart_fixture(monkeypatch)
    batch = attempts.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        requested_by="operator",
        request_id="restart-completion-guard",
    )
    generation = persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id=batch.retry_batch_id,
        expected_generation=state._execution_generation,
        active_status="running",
    )
    attempts.bind_retry_batch_generation(batch.retry_batch_id, generation)
    config = state.plan.technique_configs[0]
    attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=generation,
    )
    state._execution_generation = generation
    state._execution_owner_id = batch.retry_batch_id
    state.status = "completed"

    class InertTimer:
        def __init__(self, interval, function) -> None:
            self.interval = interval
            self.function = function
            self.daemon = False

        def start(self) -> None:
            return None

        def cancel(self) -> None:
            return None

    monkeypatch.setattr("app.api.store.threading.Timer", InertTimer)
    store = object.__new__(EvaluationStore)
    store.persistence = persistence
    store.experiment_attempt_persistence = attempts
    store._experiments = {state.experiment_id: state}
    store._terminal_status_reconciliation_lock = threading.Lock()
    store._terminal_status_reconciliation_timers = {}
    store._terminal_status_reconciliation_failures = {}
    store._terminal_status_reconciliation_obligations = {}

    assert not store._reconcile_terminal_experiment_status(
        state.experiment_id,
        "completed",
        generation=generation,
        owner_id=batch.retry_batch_id,
        report={"tested": 0},
    )
    assert state.status == "running"
    assert persistence.load_all_experiments()[0]["status"] == "running"


class _FailingGenerationBeginPersistence(NoopRunPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.fail_begin = False

    def begin_experiment_execution(self, *args, **kwargs):
        if self.fail_begin:
            raise RuntimeError("generation begin failed")
        return super().begin_experiment_execution(*args, **kwargs)


class _FailingGenerationBindAttempts(InMemoryExperimentAttemptPersistence):
    def bind_retry_batch_generation(self, retry_batch_id, execution_generation):
        raise RuntimeError("generation bind failed")


class _BlockingOwnedTerminalPersistence(NoopRunPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.block_next_terminal = False
        self.terminal_applied = threading.Event()
        self.release_terminal = threading.Event()

    def reconcile_owned_experiment_terminal_status(self, *args, **kwargs):
        reconciled = super().reconcile_owned_experiment_terminal_status(
            *args,
            **kwargs,
        )
        if self.block_next_terminal:
            self.block_next_terminal = False
            self.terminal_applied.set()
            assert self.release_terminal.wait(timeout=5)
        return reconciled


class _LostAcknowledgementBeginPersistence(NoopRunPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.lose_next_acknowledgement = False

    def begin_experiment_execution(self, *args, **kwargs):
        generation = super().begin_experiment_execution(*args, **kwargs)
        if self.lose_next_acknowledgement:
            self.lose_next_acknowledgement = False
            raise RuntimeError("generation acknowledgement lost")
        return generation


def _capture_retry_timers(monkeypatch):
    timers = []

    class ManualTimer:
        def __init__(self, interval, function, args=None, kwargs=None) -> None:
            self.interval = interval
            self.function = function
            self.args = tuple(args or ())
            self.kwargs = dict(kwargs or {})
            self.daemon = False
            self.cancelled = False

        def start(self) -> None:
            timers.append(self)

        def cancel(self) -> None:
            self.cancelled = True

        def fire(self) -> None:
            if not self.cancelled:
                self.cancelled = True
                self.function(*self.args, **self.kwargs)

    monkeypatch.setattr("app.api.store.threading.Timer", ManualTimer)
    return timers


def _persisted_experiment_payload(
    state: ExperimentState,
    *,
    report: dict | None,
) -> dict:
    return {
        "experiment_id": state.experiment_id,
        "name": state.name,
        "description": state.description,
        "hypothesis": state.hypothesis,
        "status": "completed",
        "mode": state.mode,
        "dataset": state.dataset,
        "technique_keys": state.technique_keys,
        "target_config": state.target_config,
        "total_cells": state.total_cells,
        "repeat_count": state.repeat_count,
        "max_rounds": state.max_rounds,
        "experiment_group": state.experiment_group,
        "completed_cells": len(state.cell_results),
        "cell_results": state.cell_results,
        "report": report,
        "error": None,
        "created_at": state.created_at,
        "plan": _serialize_plan(state.plan),
    }


def test_expected_experiment_cells_include_repeat_index() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=3)

    cells = list(store._iter_expected_experiment_cells(state))

    assert [(tech, prompt["id"], repeat) for tech, prompt, repeat in cells] == [
        ("crescendo_liar", "p1", 1),
        ("crescendo_liar", "p1", 2),
        ("crescendo_liar", "p1", 3),
        ("crescendo_liar", "p2", 1),
        ("crescendo_liar", "p2", 2),
        ("crescendo_liar", "p2", 3),
    ]


def test_experiment_plan_materializes_configurations_prompts_and_repeats(monkeypatch) -> None:
    started = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            started.append(self)

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_planning_catalog(),
    )

    payload = _create_planned_experiment(store)
    state = store._experiments[payload["experiment_id"]]

    assert state.plan is not None
    assert len(state.plan.technique_configs) == 2
    assert len(state.plan.cells) == 2 * 2 * 3
    assert payload["total_cells"] == 12
    assert len(started) == 1


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("repeat_count", True),
        ("repeat_count", "2"),
        ("max_rounds", False),
        ("max_rounds", "5"),
    ],
)
def test_create_experiment_rejects_non_integer_counts_before_persistence(
    monkeypatch,
    field: str,
    invalid: object,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    events: list[str] = []
    attempts = _RecordingAttemptPersistence(events)
    store = EvaluationStore(
        persistence=_PlanningRunPersistence(events),
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(name="Strict counts", prompts=[{"text": "intent"}])
    request = {
        "name": "Strict counts",
        "dataset_id": dataset["dataset_id"],
        "technique_keys": ["probe"],
        "target_config": {"model": "target-model"},
        "mode": "demo",
        "repeat_count": 1,
        "max_rounds": 5,
    }
    request[field] = invalid

    with pytest.raises(ValueError, match=rf"{field} must be an integer"):
        store.create_experiment(**request)

    assert events == []
    assert attempts.config_records == []
    assert attempts.cell_records == []


def test_experiment_plan_pins_behavior_and_uses_canonical_stable_identity(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_planning_catalog(),
    )

    payload = _create_planned_experiment(
        store,
        technique_configs=[
            {
                "technique_key": "probe-alias",
                "preset": "high",
                "parameters": {"nested": {"enabled": True}, "threshold": 4},
            },
            {"technique_key": "probe", "parameters": {"threshold": 6}},
        ],
        repeat_count=1,
    )
    state = store._experiments[payload["experiment_id"]]
    assert state.plan is not None
    first, distinct = state.plan.technique_configs

    assert first.technique_key == "probe-alias"
    assert first.technique_id == "probe"
    assert first.technique_version == "1.2.3"
    assert first.implementation_id == "probe-python"
    assert first.implementation_kind == "python"
    assert first.contract_version == "7"
    assert first.manifest_hash == "a" * 64
    assert first.asset_hashes == {"workflow.yaml": "b" * 64}
    assert first.preset == "high"
    assert first.parameters == {"nested": {"enabled": True}, "threshold": 4}
    assert first.model_roles["target"]["profile_id"] == "target-profile"
    assert first.model_roles["target_judge"]["model"] == "target-model"
    assert first.judge_policies["target_judge"]["policy"] == {"threshold": 0.7}
    assert first.effective_limits == {"max_rounds": 5, "max_tokens": 2048}
    assert len(first.spec_hash) == 64
    assert distinct.technique_config_id != first.technique_config_id
    first_cells = [cell for cell in state.plan.cells if cell.technique_config_id == first.technique_config_id]
    distinct_cells = [cell for cell in state.plan.cells if cell.technique_config_id == distinct.technique_config_id]
    material = "\x1f".join((state.experiment_id, first.technique_config_id, "p-001", "1"))
    assert first_cells[0].cell_id == f"cell-{sha256(material.encode('utf-8')).hexdigest()[:32]}"
    assert first_cells[0].cell_id != distinct_cells[0].cell_id

    equivalent_store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_planning_catalog(),
    )
    equivalent_payload = _create_planned_experiment(
        equivalent_store,
        technique_configs=[
            {"technique_key": "probe", "parameters": {"threshold": 6}},
            {
                "parameters": {"threshold": 4, "nested": {"enabled": True}},
                "preset": "high",
                "technique_key": "probe-alias",
            },
        ],
        repeat_count=1,
    )
    equivalent_plan = equivalent_store._experiments[equivalent_payload["experiment_id"]].plan
    assert equivalent_plan is not None
    equivalent = next(
        config
        for config in equivalent_plan.technique_configs
        if config.parameters["threshold"] == 4
    )
    assert first.technique_config_id == equivalent.technique_config_id
    assert first.spec_hash == equivalent.spec_hash
    assert state.plan.plan_hash == equivalent_plan.plan_hash
    assert [config.parameters["threshold"] for config in equivalent_plan.technique_configs] == [6, 4]


def test_experiment_plan_and_nested_values_are_immutable(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(store, repeat_count=1)
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None

    with pytest.raises(FrozenInstanceError):
        plan.repeat_count = 99
    with pytest.raises(TypeError):
        plan.technique_configs[0].parameters["threshold"] = 99
    with pytest.raises(TypeError):
        plan.technique_configs[0].parameters["nested"]["enabled"] = False


def test_plan_persists_before_worker_and_failure_starts_no_work(monkeypatch) -> None:
    events: list[str] = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            events.append("thread-created")

        def start(self) -> None:
            events.append("worker")

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    attempts = _RecordingAttemptPersistence(events)
    store = EvaluationStore(
        persistence=_PlanningRunPersistence(events),
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    payload = _create_planned_experiment(store, repeat_count=1)

    assert events[0] == "experiment"
    assert events[-2:] == ["thread-created", "worker"]
    assert all(event.startswith(("config:", "cell:")) for event in events[1:-2])
    assert attempts._attempts == {}
    _assert_secret_free(payload)
    _assert_secret_free([record.resolved_spec for record in attempts.config_records])

    failed_events: list[str] = []
    failed_store = EvaluationStore(
        persistence=_PlanningRunPersistence(failed_events),
        catalog=_planning_catalog(),
        experiment_attempt_persistence=_RecordingAttemptPersistence(
            failed_events,
            fail_cells=True,
        ),
    )
    with pytest.raises(RuntimeError, match="planned cell persistence failed"):
        _create_planned_experiment(failed_store, repeat_count=1)
    assert "thread-created" not in failed_events
    assert "worker" not in failed_events
    assert failed_store._experiments == {}


def test_public_experiment_config_recursively_drops_nested_credentials(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)

    class RecordingExperimentPersistence(_PlanningRunPersistence):
        def save_experiment(self, *, experiment) -> None:
            super().save_experiment(experiment=experiment)
            self.experiment = experiment

    persistence = RecordingExperimentPersistence([])
    store = EvaluationStore(persistence=persistence, catalog=_planning_catalog())
    dataset = store.create_dataset(name="Secrets", prompts=[{"text": "intent"}])
    sentinel = "NESTED-SENTINEL-CREDENTIAL"
    metadata = {
        "region": "us-west",
        "layers": [
            {"label": "preserved", "credentials": {"access_token": sentinel}},
            {"service_password": sentinel, "enabled": True},
            {
                "reviewer": "preserved",
                "session_token": sentinel,
                "id-token": sentinel,
                "private.key": sentinel,
                "passphrase": sentinel,
                "client_secret": sentinel,
            },
            {
                "note": "Bearer sk-supersecret123",
                "safe_note": "skill-building",
            },
        ],
    }
    target_config = {"model": "target-model", "api_key": sentinel, "metadata": metadata}

    created = store.create_experiment(
        name="Recursive redaction",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config=target_config,
        mode="demo",
    )
    payloads = (
        created,
        store.get_experiment(created["experiment_id"]),
        store.export_experiment_json(created["experiment_id"]),
        persistence.experiment,
    )

    for payload in payloads:
        assert sentinel not in json.dumps(payload, sort_keys=True)
        assert payload["target_config"]["metadata"] == {
            "layers": [
                {"label": "preserved"},
                {"enabled": True},
                {"reviewer": "preserved"},
                {"note": "[REDACTED]", "safe_note": "skill-building"},
            ],
            "region": "us-west",
        }
    assert target_config["metadata"] == metadata


def test_plan_dataset_hash_uses_normalized_planned_prompt_evidence(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    dataset = store.create_dataset(
        name="Normalized evidence",
        prompts=[
            {"text": "with reference", "gold_answer": "  expected answer  "},
            {"text": "without reference", "gold_answer": "   "},
        ],
    )
    payload = store.create_experiment(
        name="Normalized evidence",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    plan = store._experiments[payload["experiment_id"]].plan
    prompts = [
        {
            "prompt_id": cell.prompt_id,
            "prompt_text": cell.prompt_text,
            "prompt_category": cell.prompt_category,
            "gold_answer": cell.gold_answer,
        }
        for cell in plan.cells
    ]

    assert plan.dataset_hash == canonical_dataset_hash(
        dataset_id=plan.dataset_id,
        dataset_version=plan.dataset_version,
        prompts=prompts,
    )


@pytest.mark.parametrize("retry", [False, True], ids=["initial", "retry"])
def test_baseline_workers_reapply_plan_after_profile_credential_refresh(
    monkeypatch,
    retry,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence())
    dataset = store.create_dataset(name="Baseline pin", prompts=[{"text": "planned intent"}])
    payload = store.create_experiment(
        name="Baseline pin",
        dataset_id=dataset["dataset_id"],
        technique_keys=["__baseline__"],
        target_config={
            "target_profile_id": "target-profile",
            "provider": "planned-provider",
            "model": "planned-model",
            "api_base_url": "https://planned-target.test/v1",
            "judge_mode": "custom",
            "judge_profile_id": "judge-profile",
            "judge_provider": "planned-judge-provider",
            "judge_model": "planned-judge-model",
            "judge_api_base_url": "https://planned-judge.test/v1",
            "system_prompt": "planned target prompt",
            "judge_system_prompt": "planned judge prompt",
            "judge_policy": {"mode": "standard"},
            "max_tokens": "321",
        },
        mode="demo",
        max_rounds=4,
    )
    state = store._experiments[payload["experiment_id"]]
    pinned = state.plan.technique_configs[0]
    planned_cell = state.plan.cells[0]
    captured = {}

    def resolve_drifted_profile(config):
        return {
            **config,
            "api_key": "refreshed-target-key",
            "judge_api_key": "refreshed-judge-key",
            "provider": "drifted-provider",
            "model": "drifted-model",
            "api_base_url": "https://drifted-target.test/v1",
            "judge_mode": "target",
            "judge_provider": "drifted-judge-provider",
            "judge_model": "drifted-judge-model",
            "judge_api_base_url": "https://drifted-judge.test/v1",
            "system_prompt": "drifted target prompt",
            "judge_system_prompt": "drifted judge prompt",
            "judge_policy": {"mode": "rubric", "rubric": "drifted"},
            "max_tokens": "999",
            "max_rounds": "99",
        }

    class Client:
        api_base_url = "https://client.test/v1"
        system_prompt = ""

        def __init__(self, model, response):
            self.model = model
            self.response = response

        def chat_with_retry(self, _messages, **_kwargs):
            return self.response, {"total_tokens": 1}

    monkeypatch.setattr(store, "resolve_profile_config", resolve_drifted_profile)
    monkeypatch.setattr(store, "_build_chat_client", lambda config: Client(config["model"], "response"))
    monkeypatch.setattr(store, "_build_attacker_client", lambda _config: None)

    def build_judge(config, **_kwargs):
        captured.update(config)
        return Client(config["judge_model"], '{"score": 8, "rationale": "ok"}')

    monkeypatch.setattr(store, "_build_judge_client", build_judge)
    monkeypatch.setattr(store, "_correlate_legacy_client", lambda client, **_kwargs: client)
    if retry:
        _finish_planned_state(state)
        assert store.persistence.reconcile_owned_experiment_terminal_status(
            state.experiment_id,
            generation=state._execution_generation,
            owner_id=state._execution_owner_id,
            status="completed",
        )
        prompt = {
            "id": planned_cell.prompt_id,
            "text": planned_cell.prompt_text,
            "category": planned_cell.prompt_category,
        }
        store._run_experiment_retry(
            state.experiment_id,
            [(pinned, prompt, planned_cell.repeat_index)],
            {},
            "stale-key",
        )
    else:
        store._run_experiment(state.experiment_id, {}, "stale-key")

    expected = store._pinned_runtime_config(
        {
            "target_profile_id": "target-profile",
            "judge_profile_id": "judge-profile",
            "api_key": "refreshed-target-key",
            "judge_api_key": "refreshed-judge-key",
        },
        pinned,
    )
    for key in (
        "api_key", "judge_api_key", "provider", "model", "api_base_url",
        "judge_mode", "judge_provider", "judge_model", "judge_api_base_url",
        "system_prompt", "judge_system_prompt", "judge_policy", "max_tokens", "max_rounds",
    ):
        assert captured[key] == expected[key]


@pytest.mark.parametrize(
    "explicit_credentials",
    [False, True],
    ids=["profile-owned", "request-explicit"],
)
def test_initial_worker_refreshes_all_profile_credentials_and_reapplies_plan(
    monkeypatch,
    explicit_credentials,
) -> None:
    threads = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            threads.append(self)

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    vault, profiles = _profile_vault_with_role_profiles()
    custom_definition = {
        "id": "profile-refresh",
        "name": "Profile refresh",
        "mode": "iterative",
        "prompt_template": "Attack {intent}",
        "refinement_template": "Refine {last_response}",
    }
    store = EvaluationStore(
        persistence=_PlanningRunPersistence([], [custom_definition]),
        profile_vault=vault,
        catalog=_planning_catalog(),
    )
    dataset = store.create_dataset(name="Profile refresh", prompts=[{"text": "intent"}])
    target_config = {
        "target_profile_id": profiles["target"],
        "attacker_profile_id": profiles["attacker"],
        "judge_profile_id": profiles["judge"],
        "judge_mode": "custom",
    }
    if explicit_credentials:
        target_config.update(
            {
                "api_key": "explicit-target-key",
                "attacker_api_key": "explicit-attacker-key",
                "judge_api_key": "explicit-judge-key",
            }
        )
    store.create_experiment(
        name="Profile refresh",
        dataset_id=dataset["dataset_id"],
        technique_keys=["custom:profile-refresh"],
        target_config=target_config,
        mode="live",
    )
    _rotate_role_profiles(vault, profiles)
    captured: dict[str, object] = {}

    def execute_single(**kwargs):
        captured.update(kwargs["target_config"])
        return RunSummary(
            run_id="run-profile-refresh",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    monkeypatch.setattr(store, "_execute_single", execute_single)
    threads[0].target(*threads[0].args)

    expected_credentials = (
        {
            "api_key": "explicit-target-key",
            "attacker_api_key": "explicit-attacker-key",
            "judge_api_key": "explicit-judge-key",
        }
        if explicit_credentials
        else {
            "api_key": "refreshed-target-key",
            "attacker_api_key": "refreshed-attacker-key",
            "judge_api_key": "refreshed-judge-key",
        }
    )
    expected_runtime = {
        **expected_credentials,
        "provider": "planned-target-provider",
        "model": "planned-target-model",
        "api_base_url": "https://planned-target.test/v1",
        "attacker_provider": "planned-attacker-provider",
        "attacker_model": "planned-attacker-model",
        "attacker_api_base_url": "https://planned-attacker.test/v1",
        "judge_provider": "planned-judge-provider",
        "judge_model": "planned-judge-model",
        "judge_api_base_url": "https://planned-judge.test/v1",
    }
    for key, expected in expected_runtime.items():
        assert captured[key] == expected


def test_initial_baseline_worker_refreshes_profile_credentials_before_execution(
    monkeypatch,
) -> None:
    threads = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            threads.append(self)

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    vault, profiles = _profile_vault_with_role_profiles()
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        profile_vault=vault,
    )
    dataset = store.create_dataset(name="Baseline refresh", prompts=[{"text": "intent"}])
    store.create_experiment(
        name="Baseline refresh",
        dataset_id=dataset["dataset_id"],
        technique_keys=["__baseline__"],
        target_config={
            "target_profile_id": profiles["target"],
            "judge_profile_id": profiles["judge"],
            "judge_mode": "custom",
        },
        mode="live",
    )
    _rotate_role_profiles(vault, profiles)
    captured: dict[str, object] = {}

    def run_baseline(**kwargs):
        captured.update(kwargs["target_config"])
        return {
            "run_id": "run-baseline-refresh",
            "prompt": "intent",
            "response": "response",
            "judge_input": "input",
            "judge_output": "output",
            "judge_score": 0,
            "judge_rationale": "ok",
        }

    monkeypatch.setattr(store, "run_baseline", run_baseline)
    threads[0].target(*threads[0].args)

    assert captured["api_key"] == "refreshed-target-key"
    assert captured["judge_api_key"] == "refreshed-judge-key"
    assert captured["provider"] == "planned-target-provider"
    assert captured["model"] == "planned-target-model"
    assert captured["api_base_url"] == "https://planned-target.test/v1"
    assert captured["judge_provider"] == "planned-judge-provider"
    assert captured["judge_model"] == "planned-judge-model"
    assert captured["judge_api_base_url"] == "https://planned-judge.test/v1"


def test_retry_worker_preserves_explicit_credentials_for_all_profile_roles(
    monkeypatch,
) -> None:
    threads = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            threads.append(self)

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    vault, profiles = _profile_vault_with_role_profiles()
    custom_definition = {
        "id": "explicit-retry",
        "name": "Explicit retry",
        "mode": "iterative",
        "prompt_template": "Attack {intent}",
        "refinement_template": "Refine {last_response}",
    }
    store = EvaluationStore(
        persistence=_PlanningRunPersistence([], [custom_definition]),
        profile_vault=vault,
        catalog=_planning_catalog(),
    )
    dataset = store.create_dataset(name="Explicit retry", prompts=[{"text": "intent"}])
    payload = store.create_experiment(
        name="Explicit retry",
        dataset_id=dataset["dataset_id"],
        technique_keys=["custom:explicit-retry"],
        target_config={
            "target_profile_id": profiles["target"],
            "attacker_profile_id": profiles["attacker"],
            "judge_profile_id": profiles["judge"],
            "judge_mode": "custom",
        },
        mode="live",
    )
    state = store._experiments[payload["experiment_id"]]
    state.status = "failed"
    threads.clear()
    _rotate_role_profiles(vault, profiles)
    captured: dict[str, object] = {}

    def execute_single(**kwargs):
        captured.update(kwargs["target_config"])
        return RunSummary(
            run_id="run-explicit-retry",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    monkeypatch.setattr(store, "_execute_single", execute_single)
    store.retry_experiment_errors(
        state.experiment_id,
        {
            "api_key": "explicit-target-key",
            "attacker_api_key": "explicit-attacker-key",
            "judge_api_key": "explicit-judge-key",
        },
    )
    threads[0].target(*threads[0].args)

    assert captured["api_key"] == "explicit-target-key"
    assert captured["attacker_api_key"] == "explicit-attacker-key"
    assert captured["judge_api_key"] == "explicit-judge-key"
    assert captured["provider"] == "planned-target-provider"
    assert captured["attacker_provider"] == "planned-attacker-provider"
    assert captured["judge_provider"] == "planned-judge-provider"


def _save_base_role_profiles(vault: ProfileVault) -> dict[str, str]:
    profile_ids: dict[str, str] = {}
    for role in ("target", "attacker", "judge"):
        profile = vault.save_profile(
            {
                "profile_id": f"base-{role}-profile",
                "name": f"Base {role}",
                "provider": f"base-{role}-provider",
                "base_url": f"https://base-{role}.test/v1",
                "model": f"base-{role}-model",
                "scope": role,
                "api_key": f"base-{role}-key",
            }
        )
        profile_ids[role] = profile["profile_id"]
    return profile_ids


def _pin_different_arm_profiles(
    state: ExperimentState,
    arm_profiles: Mapping[str, str],
    *,
    include_attacker: bool,
    attempt_persistence: InMemoryExperimentAttemptPersistence | None = None,
) -> None:
    assert state.plan is not None
    original = state.plan.technique_configs[0]
    model_roles = {
        "target": {
            "profile_id": arm_profiles["target"],
            "provider": "planned-target-provider",
            "model": "planned-target-model",
            "transport": "openai_compatible",
        },
        "target_judge": {
            "profile_id": arm_profiles["judge"],
            "provider": "planned-judge-provider",
            "model": "planned-judge-model",
            "transport": "openai_compatible",
        },
    }
    if include_attacker:
        model_roles["attacker"] = {
            "profile_id": arm_profiles["attacker"],
            "provider": "planned-attacker-provider",
            "model": "planned-attacker-model",
            "transport": "openai_compatible",
        }
    runtime_config = {
        **original.runtime_config,
        "api_base_url": "https://planned-target.test/v1",
        "attacker_api_base_url": "https://planned-attacker.test/v1",
        "judge_api_base_url": "https://planned-judge.test/v1",
        "judge_mode": "custom",
    }
    resolved_spec = {
        **original.resolved_spec,
        "model_roles": model_roles,
        "runtime_config": runtime_config,
    }
    config_id = technique_config_id(state.experiment_id, resolved_spec)
    pinned = replace(
        original,
        technique_config_id=config_id,
        model_roles=model_roles,
        runtime_config=runtime_config,
        resolved_spec=resolved_spec,
        spec_hash=content_hash(resolved_spec),
    )
    cells = tuple(
        replace(
            cell,
            cell_id=canonical_cell_id(
                state.experiment_id,
                config_id,
                cell.prompt_id,
                cell.repeat_index,
            ),
            technique_config_id=config_id,
        )
        for cell in state.plan.cells
    )
    state.plan = replace(state.plan, technique_configs=(pinned,), cells=cells)
    if attempt_persistence is not None:
        attempt_persistence.upsert_technique_config(
            TechniqueConfigRecord(
                technique_config_id=pinned.technique_config_id,
                experiment_id=state.experiment_id,
                technique_id=pinned.technique_id,
                technique_version=pinned.technique_version,
                implementation_id=pinned.implementation_id,
                preset=pinned.preset,
                parameters=pinned.parameters,
                resolved_spec=pinned.resolved_spec,
                spec_hash=pinned.spec_hash,
            )
        )
        for cell in cells:
            attempt_persistence.create_cell(
                CellRecord(
                    cell_id=cell.cell_id,
                    experiment_id=cell.experiment_id,
                    technique_config_id=cell.technique_config_id,
                    technique_key=cell.technique_key,
                    prompt_id=cell.prompt_id,
                    repeat_index=cell.repeat_index,
                )
            )


def test_profile_owned_credentials_refresh_when_arm_pins_different_profiles(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    vault, arm_profiles = _profile_vault_with_role_profiles()
    base_profiles = _save_base_role_profiles(vault)
    store = EvaluationStore(
        persistence=_PlanningRunPersistence(
            [],
            [{"id": "profile-owner", "mode": "iterative"}],
        ),
        profile_vault=vault,
    )
    dataset = store.create_dataset(name="Profile owner", prompts=[{"text": "intent"}])
    payload = store.create_experiment(
        name="Profile owner",
        dataset_id=dataset["dataset_id"],
        technique_keys=["custom:profile-owner"],
        target_config={
            "target_profile_id": base_profiles["target"],
            "attacker_profile_id": base_profiles["attacker"],
            "judge_profile_id": base_profiles["judge"],
            "judge_mode": "custom",
        },
        mode="live",
    )
    state = store._experiments[payload["experiment_id"]]
    _pin_different_arm_profiles(state, arm_profiles, include_attacker=True)
    _rotate_role_profiles(vault, arm_profiles)
    pinned = state.plan.technique_configs[0]

    arm_config = store._pinned_runtime_config(
        {
            "target_profile_id": base_profiles["target"],
            "attacker_profile_id": base_profiles["attacker"],
            "judge_profile_id": base_profiles["judge"],
            "api_key": "base-target-key",
            "attacker_api_key": "base-attacker-key",
            "judge_api_key": "base-judge-key",
        },
        pinned,
    )
    assert "api_key" not in arm_config
    assert "attacker_api_key" not in arm_config
    assert "judge_api_key" not in arm_config

    refreshed = store.resolve_profile_config(arm_config)
    runtime = store._pinned_runtime_config(refreshed, pinned)

    assert runtime["api_key"] == "refreshed-target-key"
    assert runtime["attacker_api_key"] == "refreshed-attacker-key"
    assert runtime["judge_api_key"] == "refreshed-judge-key"
    assert runtime["provider"] == "planned-target-provider"
    assert runtime["model"] == "planned-target-model"
    assert runtime["api_base_url"] == "https://planned-target.test/v1"
    assert runtime["attacker_provider"] == "planned-attacker-provider"
    assert runtime["attacker_model"] == "planned-attacker-model"
    assert runtime["attacker_api_base_url"] == "https://planned-attacker.test/v1"
    assert runtime["judge_provider"] == "planned-judge-provider"
    assert runtime["judge_model"] == "planned-judge-model"
    assert runtime["judge_api_base_url"] == "https://planned-judge.test/v1"


@pytest.mark.parametrize("retry", [False, True], ids=["initial", "retry"])
@pytest.mark.parametrize("baseline", [False, True], ids=["technique", "baseline"])
def test_workers_carry_explicit_credential_ownership_across_different_arm_profiles(
    monkeypatch,
    retry: bool,
    baseline: bool,
) -> None:
    threads = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            threads.append(self)

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    vault, arm_profiles = _profile_vault_with_role_profiles()
    base_profiles = _save_base_role_profiles(vault)
    technique_key = "__baseline__" if baseline else "custom:explicit-owner"
    persistence = _PlanningRunPersistence(
        [],
        [] if baseline else [{"id": "explicit-owner", "mode": "iterative"}],
    )
    store = EvaluationStore(persistence=persistence, profile_vault=vault)
    dataset = store.create_dataset(name="Explicit owner", prompts=[{"text": "intent"}])
    explicit_credentials = {
        "api_key": "explicit-target-key",
        "attacker_api_key": "explicit-attacker-key",
        "judge_api_key": "explicit-judge-key",
    }
    payload = store.create_experiment(
        name="Explicit owner",
        dataset_id=dataset["dataset_id"],
        technique_keys=[technique_key],
        target_config={
            "target_profile_id": base_profiles["target"],
            "attacker_profile_id": base_profiles["attacker"],
            "judge_profile_id": base_profiles["judge"],
            "judge_mode": "custom",
            **explicit_credentials,
        },
        mode="live",
    )
    state = store._experiments[payload["experiment_id"]]
    _pin_different_arm_profiles(
        state,
        arm_profiles,
        include_attacker=not baseline,
        attempt_persistence=store.experiment_attempt_persistence,
    )
    if retry:
        state.status = "failed"
        threads.clear()
        store.retry_experiment_errors(state.experiment_id, explicit_credentials)

    boundary: dict[str, object] = {}
    runtime: dict[str, object] = {}

    if baseline:
        class Client:
            api_base_url = "https://client.test/v1"
            system_prompt = ""

            def __init__(self, model: str, response: str) -> None:
                self.model = model
                self.response = response

            def chat_with_retry(self, _messages, **_kwargs):
                return self.response, {"total_tokens": 1}

        monkeypatch.setattr(
            store,
            "_build_chat_client",
            lambda config: Client(str(config["model"]), "target response"),
        )
        monkeypatch.setattr(store, "_build_attacker_client", lambda _config: None)

        def build_judge(config, **_kwargs):
            runtime.update(config)
            return Client(
                str(config["judge_model"]),
                '{"score": 8, "rationale": "ok"}',
            )

        monkeypatch.setattr(store, "_build_judge_client", build_judge)
        monkeypatch.setattr(
            store,
            "_correlate_legacy_client",
            lambda client, **_kwargs: client,
        )
        original_run_baseline = store.run_baseline

        def run_baseline(**kwargs):
            boundary.update(kwargs)
            return original_run_baseline(**kwargs)

        monkeypatch.setattr(store, "run_baseline", run_baseline)
    else:
        def execute_custom_iterative(**kwargs):
            runtime.update(kwargs["target_config"])
            return TechniqueRunResult(
                run_id=kwargs["run_id"],
                technique_key=kwargs["technique_key"],
                target_id="live-chatbot",
                response=TargetResponse(
                    target_id="live-chatbot",
                    target_kind=TargetKind.CHATBOT,
                    final_output="response",
                ),
                detections=(),
                score=0,
                traces=(),
                audit_log=(),
            )

        monkeypatch.setattr(
            store,
            "_execute_custom_iterative",
            execute_custom_iterative,
        )
        original_execute_single = store._execute_single

        def execute_single(**kwargs):
            boundary.update(kwargs)
            return original_execute_single(**kwargs)

        monkeypatch.setattr(store, "_execute_single", execute_single)

    threads[0].target(*threads[0].args)

    expected_keys = {"api_key", "judge_api_key"}
    if not baseline:
        expected_keys.add("attacker_api_key")
    assert boundary["explicit_credential_keys"] == frozenset(
        {"api_key", "attacker_api_key", "judge_api_key"}
    )
    for key in expected_keys:
        assert runtime[key] == explicit_credentials[key]
    assert runtime["provider"] == "planned-target-provider"
    assert runtime["model"] == "planned-target-model"
    assert runtime["api_base_url"] == "https://planned-target.test/v1"
    assert runtime["judge_provider"] == "planned-judge-provider"
    assert runtime["judge_model"] == "planned-judge-model"
    assert runtime["judge_api_base_url"] == "https://planned-judge.test/v1"
    if not baseline:
        assert runtime["attacker_provider"] == "planned-attacker-provider"
        assert runtime["attacker_model"] == "planned-attacker-model"
        assert runtime["attacker_api_base_url"] == "https://planned-attacker.test/v1"
    assert "explicit_credential_keys" not in json.dumps(payload, sort_keys=True)

    runtime.clear()
    direct_config = {
        "target_profile_id": base_profiles["target"],
        "attacker_profile_id": base_profiles["attacker"],
        "judge_profile_id": base_profiles["judge"],
        "judge_mode": "custom",
        "unsafe_intent": "intent",
        **explicit_credentials,
    }
    pinned = state.plan.technique_configs[0]
    explicit_keys = frozenset(explicit_credentials)
    if baseline:
        store.run_baseline(
            target_config=direct_config,
            technique_config=pinned,
            explicit_credential_keys=explicit_keys,
        )
    else:
        store._execute_single(
            technique_key=technique_key,
            technique_config=pinned,
            target_kind="chatbot",
            mode="live",
            target_config=direct_config,
            explicit_credential_keys=explicit_keys,
        )
    for key in expected_keys:
        assert runtime[key] == explicit_credentials[key]


def test_same_dsn_postgresql_plan_materialization_uses_one_atomic_boundary(
    monkeypatch,
) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    dsn = "postgresql://atomic.test/jailbreaker"
    run_persistence = _AtomicPlanRunPersistence(dsn)
    attempts = _AtomicPlanAttemptPersistence(dsn)
    store = EvaluationStore(
        persistence=run_persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    _create_planned_experiment(store, repeat_count=1)

    assert run_persistence.events[0] == "atomic-experiment"
    persisted_plan = run_persistence.persisted_experiments[0]["plan"]
    assert persisted_plan["dataset_id"] == "ds-1"
    assert persisted_plan["dataset_version"] == "1"
    assert persisted_plan["cells"][0]["prompt_text"] == "intent one"
    assert "gold_answer" in persisted_plan["cells"][0]
    assert not any(event.startswith("split-") for event in run_persistence.events)
    assert attempts.events == []
    assert len(started) == 1


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_cross_store_same_plan_replay_restores_later_execution_generation(
    monkeypatch,
) -> None:
    import psycopg

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments, "
            "experiment_dataset_prompts, experiment_datasets CASCADE"
        )

    initial_workers = []
    background_timers = []
    real_thread_start = threading.Thread.start

    def capture_initial_worker(thread) -> None:
        target_name = getattr(thread._target, "__name__", "")
        if target_name == "_run_experiment":
            initial_workers.append(thread)
            return
        if isinstance(thread, threading.Timer):
            background_timers.append(thread)
        real_thread_start(thread)

    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        capture_initial_worker,
    )
    first_persistence = PostgreSQLRunPersistence(dsn)
    second_persistence = PostgreSQLRunPersistence(dsn)
    first_store = EvaluationStore(
        persistence=first_persistence,
        catalog=_planning_catalog(),
    )
    second_store = EvaluationStore(
        persistence=second_persistence,
        catalog=_planning_catalog(),
    )
    first_dataset = first_store.create_dataset(
        name="Atomic ownership replay",
        prompts=[{"text": "intent"}],
    )
    second_dataset = second_store.create_dataset(
        name="Atomic ownership replay",
        prompts=[{"text": "intent"}],
    )
    request = {
        "name": "Atomic ownership replay",
        "dataset_id": first_dataset["dataset_id"],
        "technique_keys": ["probe"],
        "target_config": {"model": "target-model"},
        "mode": "demo",
    }
    try:
        created = first_store.create_experiment(**request)
        experiment_id = created["experiment_id"]
        assert first_persistence.reconcile_owned_experiment_terminal_status(
            experiment_id,
            generation=1,
            owner_id=f"initial:{experiment_id}",
            status="completed",
        )
        assert first_persistence.begin_experiment_execution(
            experiment_id,
            owner_id="later-retry-owner",
            expected_generation=1,
            active_status="running",
        ) == 2
        assert second_dataset["dataset_id"] == first_dataset["dataset_id"]

        replay = second_store.create_experiment(**request)

        assert replay["status"] == "running"
        assert [thread._target.__self__ for thread in initial_workers] == [first_store]
        replayed_state = second_store._experiments[experiment_id]
        assert replayed_state._execution_generation == 2
        assert replayed_state._execution_owner_id == "later-retry-owner"
        durable = first_persistence.load_all_experiments()[0]
        assert durable["execution_generation"] == 2
        assert durable["execution_owner_id"] == "later-retry-owner"
    finally:
        for timer in background_timers:
            timer.cancel()
        for timer in background_timers:
            timer.join(timeout=1)


def test_atomic_plan_failure_is_not_restorable_and_starts_no_worker(monkeypatch) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    dsn = "postgresql://atomic.test/jailbreaker"
    run_persistence = _AtomicPlanRunPersistence(dsn, fail_after_cells=2)
    attempts = _AtomicPlanAttemptPersistence(dsn)
    store = EvaluationStore(
        persistence=run_persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    with pytest.raises(RuntimeError, match="atomic planned cell persistence failed"):
        _create_planned_experiment(store, repeat_count=1)

    restarted = EvaluationStore(
        persistence=run_persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    assert restarted.list_experiments() == []
    assert store._experiments == {}
    assert started == []
    assert attempts.events == []


def test_noop_same_plan_terminal_replay_restores_authoritative_state_without_worker(
    monkeypatch,
) -> None:
    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    persistence = NoopRunPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(
        name="Idempotent initial request",
        prompts=[{"text": "intent"}],
    )
    request = {
        "name": "Idempotent initial request",
        "dataset_id": dataset["dataset_id"],
        "technique_keys": ["probe"],
        "target_config": {"model": "target-model"},
        "mode": "demo",
    }

    created = store.create_experiment(**request)
    experiment_id = created["experiment_id"]
    assert persistence.reconcile_owned_experiment_terminal_status(
        experiment_id,
        generation=1,
        owner_id=f"initial:{experiment_id}",
        status="completed",
    )
    store._experiment_sequence = 0

    replay = store.create_experiment(**request)

    assert replay["status"] == "completed"
    assert len(workers) == 1
    assert store._experiments[experiment_id]._execution_generation == 1
    assert store._experiments[experiment_id]._execution_owner_id == (
        f"initial:{experiment_id}"
    )


def test_noop_plan_replay_refreshes_ownership_advanced_after_save_boundary(
    monkeypatch,
) -> None:
    class AdvanceAfterReplay(NoopRunPersistence):
        advance_on_replay = False

        def save_experiment_plan(self, **kwargs):
            result = super().save_experiment_plan(**kwargs)
            if not result.was_created and self.advance_on_replay:
                experiment_id = str(kwargs["experiment"]["experiment_id"])
                assert self.begin_experiment_execution(
                    experiment_id,
                    owner_id="later-owner",
                    expected_generation=result.execution_generation,
                    active_status="running",
                ) == result.execution_generation + 1
            return result

    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    persistence = AdvanceAfterReplay()
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=InMemoryExperimentAttemptPersistence(),
    )
    dataset = store.create_dataset(
        name="Replay ownership refresh",
        prompts=[{"text": "intent"}],
    )
    request = {
        "name": "Replay ownership refresh",
        "dataset_id": dataset["dataset_id"],
        "technique_keys": ["probe"],
        "target_config": {"model": "target-model"},
        "mode": "demo",
    }
    created = store.create_experiment(**request)
    experiment_id = created["experiment_id"]
    assert persistence.reconcile_owned_experiment_terminal_status(
        experiment_id,
        generation=1,
        owner_id=f"initial:{experiment_id}",
        status="completed",
    )
    persistence.advance_on_replay = True
    store._experiment_sequence = 0

    replay = store.create_experiment(**request)

    assert replay["status"] == "running"
    assert len(workers) == 1
    assert store._experiments[experiment_id]._execution_generation == 2
    assert store._experiments[experiment_id]._execution_owner_id == "later-owner"


def test_noop_plan_materialization_rolls_back_experiment_and_children(monkeypatch) -> None:
    class FailAfterSecondCell(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.created_cells = 0

        def create_cell(self, record):
            created = super().create_cell(record)
            self.created_cells += 1
            if self.created_cells == 2:
                raise RuntimeError("simulated in-memory planned cell failure")
            return created

    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    persistence = NoopRunPersistence()
    attempts = FailAfterSecondCell()
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(
        name="Rollback plan",
        prompts=[{"text": "one"}, {"text": "two"}],
    )

    with pytest.raises(RuntimeError, match="simulated in-memory planned cell failure"):
        store.create_experiment(
            name="Rollback plan",
            dataset_id=dataset["dataset_id"],
            technique_keys=["probe"],
            target_config={"model": "target-model"},
            mode="demo",
        )

    assert persistence.load_all_experiments() == []
    assert attempts._configs == {}
    assert attempts._cells == {}
    assert attempts._cell_identities == {}
    assert workers == []


def test_mismatched_postgresql_plan_repositories_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    run_persistence = _AtomicPlanRunPersistence("postgresql://one.test/db")
    attempts = _AtomicPlanAttemptPersistence("postgresql://two.test/db")
    store = EvaluationStore(
        persistence=run_persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )

    with pytest.raises(ValueError, match="same PostgreSQL DSN"):
        _create_planned_experiment(store, repeat_count=1)

    assert run_persistence.events == []
    assert attempts.events == []


def test_legacy_alias_baseline_and_custom_configurations_are_pinned(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    custom_definition = {
        "id": "probe-custom",
        "name": "Custom probe",
        "mode": "single_shot",
        "prompt_template": "Pinned {intent}",
    }
    persistence = _PlanningRunPersistence([], [custom_definition])
    store = EvaluationStore(persistence=persistence, catalog=_planning_catalog())

    payload = _create_planned_experiment(
        store,
        technique_keys=["probe-alias", "__baseline__", "custom:probe-custom"],
        technique_configs=None,
        repeat_count=1,
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    alias, baseline, custom = plan.technique_configs

    assert alias.technique_key == "probe-alias"
    assert alias.technique_id == "probe"
    assert baseline.technique_id == "__baseline__"
    assert baseline.implementation_id == "legacy-baseline"
    assert custom.parameters["definition"]["prompt_template"] == "Pinned {intent}"
    persistence.custom_techniques[0]["prompt_template"] = "Edited later {intent}"
    assert custom.parameters["definition"]["prompt_template"] == "Pinned {intent}"
    assert payload["plan"]["technique_configs"][2]["parameters"]["definition"]["prompt_template"] == "Pinned {intent}"


@pytest.mark.parametrize(
    ("attacker_profile_id", "expected_profile_id"),
    [(None, "legacy-attacker"), ("attacker-profile", "attacker-profile")],
)
def test_iterative_custom_plan_pins_attacker_backed_judge_identity(
    monkeypatch,
    attacker_profile_id,
    expected_profile_id,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    custom_definition = {
        "id": "iterative-custom",
        "name": "Iterative custom",
        "mode": "iterative",
        "prompt_template": "Attack {intent}",
        "refinement_template": "Refine {last_response}",
    }
    persistence = _PlanningRunPersistence([], [custom_definition])
    store = EvaluationStore(persistence=persistence, catalog=_planning_catalog())
    target_config = {
        "provider": "openai",
        "model": "target-model",
        "attacker_provider": "openai",
        "attacker_model": "attacker-model",
        "attacker_api_key": "attacker-refresh",
        "judge_mode": "attacker",
    }
    if attacker_profile_id is not None:
        target_config["attacker_profile_id"] = attacker_profile_id

    payload = _create_planned_experiment(
        store,
        technique_keys=["custom:iterative-custom"],
        technique_configs=None,
        target_config=target_config,
        repeat_count=1,
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    config = plan.technique_configs[0]

    assert set(config.model_roles) == {"target", "attacker", "target_judge"}
    assert config.model_roles["attacker"] == config.model_roles["target_judge"]
    assert config.model_roles["attacker"]["profile_id"] == expected_profile_id
    runtime = store._pinned_runtime_config(target_config, config)
    assert runtime["judge_mode"] == "attacker"
    assert runtime["attacker_api_key"] == "attacker-refresh"
    if attacker_profile_id is None:
        assert "attacker_profile_id" not in runtime
    else:
        assert runtime["attacker_profile_id"] == attacker_profile_id


def test_modular_crescendo_plan_rejects_distinct_attacker_backed_role_binding() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    target_config = {
        "provider": "openai",
        "model": "target-model",
        "attacker_provider": "openai",
        "attacker_profile_id": "shared-attacker",
        "attacker_model": "shared-attacker-model",
        "attacker_api_key": "provider-free-test-key",
        "judge_mode": "attacker",
    }

    with pytest.raises(ValueError, match="planner.*distinct|distinct.*planner"):
        store._catalog_technique_config(
            experiment_id="exp-distinct-crescendo-role",
            requested={
                "technique_key": "crescendo_liar",
                "implementation_id": "modular",
                "model_roles": {
                    "planner": {
                        "provider": "openai",
                        "profile_id": "other-planner",
                        "model": "other-planner-model",
                    }
                },
            },
            target_config=target_config,
            max_rounds=2,
        )


def test_modular_crescendo_pins_shared_attacker_roles_and_judge_mode() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    target_config = {
        "provider": "openai",
        "model": "target-model",
        "attacker_provider": "openai",
        "attacker_profile_id": "shared-attacker",
        "attacker_model": "shared-attacker-model",
        "attacker_api_key": "provider-free-test-key",
        "judge_mode": "attacker",
    }
    config = store._catalog_technique_config(
        experiment_id="exp-shared-crescendo-role",
        requested={
            "technique_key": "crescendo_liar",
            "implementation_id": "modular",
        },
        target_config=target_config,
        max_rounds=2,
    )

    assert config.model_roles["planner"] == config.model_roles["fabricator"]
    assert config.model_roles["target_judge"] == config.model_roles["planner"]
    runtime = store._pinned_runtime_config(target_config, config)
    assert runtime["attacker_profile_id"] == "shared-attacker"
    assert runtime["attacker_model"] == "shared-attacker-model"
    assert runtime["judge_mode"] == "attacker"


def test_modular_crescendo_live_rejects_actual_shared_client_binding_drift(
    monkeypatch,
) -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    target_config = {
        "provider": "openai",
        "model": "target-model",
        "attacker_provider": "openai",
        "attacker_profile_id": "shared-attacker",
        "attacker_model": "sealed-attacker-model",
        "attacker_api_key": "provider-free-test-key",
        "judge_mode": "attacker",
    }
    config = store._catalog_technique_config(
        experiment_id="exp-runtime-crescendo-role",
        requested={
            "technique_key": "crescendo_liar",
            "implementation_id": "modular",
        },
        target_config=target_config,
        max_rounds=1,
    )
    route = store.router.resolve(
        "crescendo_liar",
        implementation_id="modular",
    )

    class Client:
        def __init__(self, model: str) -> None:
            self.model = model

    target = Client("target-model")
    drifted_attacker = Client("different-attacker-model")
    monkeypatch.setattr(store, "_build_chat_client", lambda _config: target)
    monkeypatch.setattr(
        store,
        "_build_attacker_client",
        lambda _config: drifted_attacker,
    )
    monkeypatch.setattr(
        store,
        "_build_judge_client",
        lambda _config, **_kwargs: drifted_attacker,
    )

    with pytest.raises(ValueError, match="actual client.*planner.*sealed binding"):
        store._execute_modular_application_route(
            route=route,
            technique_config=config,
            target_kind=TargetKind.CHATBOT,
            mode="live",
            target_config=target_config,
            evaluation_id=None,
            run_id="runtime-drift-run",
            experiment_id="exp-runtime-crescendo-role",
            cell_id="runtime-drift-cell",
            attempt_id="runtime-drift-attempt",
        )


def test_modular_crescendo_pins_effective_threshold_in_judge_policy() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    config = store._catalog_technique_config(
        experiment_id="exp-threshold-crescendo",
        requested={
            "technique_key": "crescendo_liar",
            "implementation_id": "modular",
            "parameters": {"judge_threshold": 5},
        },
        target_config={
            "model": "target-model",
            "attacker_model": "attacker-model",
            "judge_mode": "attacker",
        },
        max_rounds=1,
    )

    assert config.parameters["judge_threshold"] == 5
    assert config.judge_policies["target_judge"]["threshold"] == 5
    assert config.judge_policies["result_policy"]["success_threshold"] == 5


def test_modular_crescendo_resolves_pinned_nondefault_threshold_for_execution() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    config = store._catalog_technique_config(
        experiment_id="exp-threshold-crescendo-execution",
        requested={
            "technique_key": "crescendo_liar",
            "implementation_id": "modular",
            "parameters": {"judge_threshold": 10},
        },
        target_config={
            "model": "target-model",
            "attacker_model": "attacker-model",
            "judge_mode": "attacker",
        },
        max_rounds=2,
    )

    route = store._resolve_pinned_execution_route("crescendo_liar", config)

    assert route.registered.definition.identity.id == "crescendo_liar"
    assert route.registered.implementation.id == "modular"


def test_modular_crescendo_rejects_genuinely_drifted_pinned_judge_policy() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    config = store._catalog_technique_config(
        experiment_id="exp-threshold-crescendo-drift",
        requested={
            "technique_key": "crescendo_liar",
            "implementation_id": "modular",
            "parameters": {"judge_threshold": 5},
        },
        target_config={
            "model": "target-model",
            "attacker_model": "attacker-model",
            "judge_mode": "attacker",
        },
        max_rounds=1,
    )
    drifted_policies = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in config.judge_policies.items()
    }
    drifted_policies["target_judge"]["threshold"] = 6
    drifted_policies["result_policy"]["success_threshold"] = 6
    drifted_spec = {
        **config.resolved_spec,
        "judge_policies": drifted_policies,
    }
    drifted_config = replace(
        config,
        judge_policies=drifted_policies,
        resolved_spec=drifted_spec,
        spec_hash=content_hash(drifted_spec),
    )

    with pytest.raises(
        ValueError,
        match="pinned judge and result policy does not match the exact package policy",
    ):
        store._resolve_pinned_execution_route("crescendo_liar", drifted_config)


def test_iterative_custom_retry_rejects_attacker_identity_drift(monkeypatch) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    persistence = _PlanningRunPersistence(
        [],
        [{"id": "iterative-custom", "mode": "iterative"}],
    )
    store = EvaluationStore(persistence=persistence, catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        technique_keys=["custom:iterative-custom"],
        technique_configs=None,
        target_config={
            "model": "target-model",
            "attacker_model": "attacker-model",
            "judge_mode": "attacker",
        },
        repeat_count=1,
    )
    state = store._experiments[payload["experiment_id"]]
    state.status = "failed"
    started.clear()

    with pytest.raises(ValueError, match="pinned model role 'attacker'"):
        store.retry_experiment_errors(
            state.experiment_id,
            {
                "api_key": "target-refresh",
                "attacker_model": "different-attacker",
            },
        )

    assert started == []


def test_planned_parameter_arms_execute_with_distinct_stable_cell_identity(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        technique_configs=[
            {
                "technique_key": "probe",
                "parameters": {"threshold": 4},
                "model_roles": {
                    "target": {
                        "profile_id": "low-profile",
                        "provider": "openai",
                        "model": "low-model",
                    }
                },
            },
            {
                "technique_key": "probe",
                "parameters": {"threshold": 6},
                "model_roles": {
                    "target": {
                        "profile_id": "high-profile",
                        "provider": "openai",
                        "model": "high-model",
                    }
                },
            },
        ],
    )
    state = store._experiments[payload["experiment_id"]]
    captured_thresholds = []
    captured_models = []

    def fake_execute_single(**kwargs):
        captured_thresholds.append(kwargs["target_config"]["threshold"])
        captured_models.append(kwargs["target_config"]["model"])
        return RunSummary(
            run_id=f"run-{len(captured_thresholds)}",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment(
        state.experiment_id,
        {"model": "target-model"},
        "target-key",
    )

    assert captured_thresholds == [4, 4, 6, 6]
    assert captured_models == ["low-model", "low-model", "high-model", "high-model"]
    assert [cell["cell_id"] for cell in state.cell_results] == [
        cell.cell_id for cell in state.plan.cells
    ]
    assert len({cell["technique_config_id"] for cell in state.cell_results}) == 2


def test_planned_behavioral_values_are_canonical_and_execute_per_arm(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    outcome_policy = {
        "mode": "outcomes",
        "success_definition": "original success",
        "failure_definition": "original failure",
    }
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        target_config={
            "model": "target-model",
            "system_prompt": "original system prompt",
            "judge_system_prompt": "original judge prompt",
            "judge_policy": outcome_policy,
            "temperature": "0.25",
            "max_tokens": "999",
        },
        technique_configs=[
            {
                "technique_key": "probe",
                "parameters": {"threshold": 4},
                "judge_policies": {
                    "target_judge": {"policy": {"arm": "strict", "threshold": 0.8}}
                },
                "limits": {"max_tokens": 111},
            },
            {
                "technique_key": "probe",
                "parameters": {"threshold": 6},
                "judge_policies": {
                    "target_judge": {"policy": {"arm": "lenient", "threshold": 0.6}}
                },
                "limits": {"max_tokens": 222},
            },
        ],
    )
    state = store._experiments[payload["experiment_id"]]
    assert state.plan is not None
    first, second = state.plan.technique_configs

    assert first.runtime_config == {
        "api_base_url": "https://api.openai.com/v1",
        "attacker_api_base_url": "http://localhost:1234/v1",
        "judge_policy": {**outcome_policy, "version": 1, "rubric": ""},
        "judge_api_base_url": "https://api.openai.com/v1",
        "judge_local": False,
        "judge_system_prompt": "original judge prompt",
        "max_rounds": 5,
        "max_tokens": 999,
        "system_prompt": "original system prompt",
        "temperature": 0.25,
    }
    assert second.runtime_config == first.runtime_config
    captured = []

    def fake_execute_single(**kwargs):
        captured.append(dict(kwargs["target_config"]))
        return RunSummary(
            run_id=f"run-behavior-{len(captured)}",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment(
        state.experiment_id,
        {
            "model": "target-model",
            "system_prompt": "mutated system prompt",
            "judge_policy": {"mode": "standard"},
            "temperature": "0.9",
            "max_tokens": "333",
        },
        "target-refresh",
    )

    assert [config["max_tokens"] for config in captured] == [111, 111, 222, 222]
    assert [config["judge_threshold"] for config in captured] == [8, 8, 6, 6]
    assert {config["system_prompt"] for config in captured} == {
        "original system prompt"
    }
    assert {config["temperature"] for config in captured} == {0.25}
    assert {config["judge_policy"]["success_definition"] for config in captured} == {
        "original success"
    }
    assert [
        config["judge_policies"]["target_judge"]["policy"]["arm"]
        for config in captured
    ] == ["strict", "strict", "lenient", "lenient"]


def test_per_arm_legacy_judge_policies_wrap_the_actual_judge_client(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        target_config={"model": "target-model"},
        technique_configs=[
            {
                "technique_key": "probe",
                "parameters": {"threshold": 4},
                "judge_policies": {
                    "target_judge": {
                        "policy": {
                            "mode": "rubric",
                            "rubric": "Require the secret phrase.",
                        }
                    }
                },
            },
            {
                "technique_key": "probe",
                "parameters": {"threshold": 6},
                "judge_policies": {
                    "target_judge": {
                        "policy": {
                            "mode": "outcomes",
                            "success_definition": "The phrase is present.",
                            "failure_definition": "The phrase is absent.",
                        }
                    }
                },
            },
        ],
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    compiled_system_prompts = []

    for config in plan.technique_configs:
        runtime = store._pinned_runtime_config({"model": "target-model"}, config)
        judge = store._build_judge_client(
            runtime,
            target_client=RecordingJudgeClient([]),
            attacker_client=None,
        )
        compiled_system_prompts.append(
            judge.build_judge_messages(intent="goal", response="response")[0]["content"]
        )

    assert "USER-PROVIDED RUBRIC" in compiled_system_prompts[0]
    assert "Require the secret phrase." in compiled_system_prompts[0]
    assert "SUCCESS DEFINITION" in compiled_system_prompts[1]
    assert "The phrase is present." in compiled_system_prompts[1]
    assert plan.technique_configs[0].runtime_config["judge_policy"] == {
        "version": 1,
        "mode": "rubric",
        "success_definition": "",
        "failure_definition": "",
        "rubric": "Require the secret phrase.",
    }
    assert plan.technique_configs[0].judge_policies["target_judge"]["policy"][
        "threshold"
    ] == 0.7


@pytest.mark.parametrize(
    "invalid_policy",
    [
        {"mode": "rubric", "rubric": ""},
        {
            "mode": "outcomes",
            "success_definition": "success",
            "failure_definition": "",
        },
    ],
)
def test_recognized_invalid_per_arm_legacy_judge_policy_is_rejected(
    monkeypatch,
    invalid_policy,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())

    with pytest.raises(ValueError, match="rubric|failure"):
        _create_planned_experiment(
            store,
            repeat_count=1,
            target_config={"model": "target-model"},
            technique_configs=[
                {
                    "technique_key": "probe",
                    "judge_policies": {
                        "target_judge": {"policy": invalid_policy}
                    },
                }
            ],
        )


@pytest.mark.parametrize(
    "behavioral_override",
    [
        {"system_prompt": "different system prompt"},
        {"max_tokens": "2048"},
        {"max_rounds": "9"},
        {"judge_policy": {"mode": "rubric", "rubric": "different rubric"}},
    ],
)
def test_retry_rejects_behavioral_drift_before_worker(
    monkeypatch,
    behavioral_override,
) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        target_config={
            "model": "target-model",
            "system_prompt": "original system prompt",
            "max_tokens": "1024",
            "judge_policy": {"mode": "standard"},
        },
    )
    state = store._experiments[payload["experiment_id"]]
    state.status = "failed"
    started.clear()

    with pytest.raises(ValueError, match="behavioral configuration.*experiment revision"):
        store.retry_experiment_errors(
            state.experiment_id,
            {"api_key": "target-refresh", **behavioral_override},
        )

    assert started == []


def test_direct_plan_pins_effective_role_endpoints_and_judge_local(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        target_config={
            "model": "target-model",
            "api_base_url": "https://target-original.test/v1",
            "attacker_model": "attacker-model",
            "attacker_api_key": "lm-studio",
            "attacker_api_base_url": "https://attacker-original.test/v1",
            "judge_mode": "custom",
            "judge_model": "judge-model",
            "judge_api_base_url": "https://judge-original.test/v1",
            "judge_local": "true",
        },
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    config = plan.technique_configs[0]

    assert config.runtime_config["api_base_url"] == "https://target-original.test/v1"
    assert config.runtime_config["attacker_api_base_url"] == (
        "https://attacker-original.test/v1"
    )
    assert config.runtime_config["judge_api_base_url"] == "https://judge-original.test/v1"
    assert config.runtime_config["judge_local"] is True

    runtime = store._pinned_runtime_config({}, config)
    target = store._build_chat_client(runtime)
    attacker = store._build_attacker_client(runtime)
    judge = store._build_judge_client(
        runtime,
        target_client=target,
        attacker_client=attacker,
    )
    assert target.api_base_url == "https://target-original.test/v1"
    assert attacker is not None
    assert attacker.api_base_url == "https://attacker-original.test/v1"
    assert judge.api_base_url == "https://judge-original.test/v1"


def test_reapplied_plan_endpoints_override_mutated_profile_metadata(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())

    def original_profiles(config):
        return {
            **config,
            "api_key": "original-target-secret",
            "api_base_url": "https://target-profile-original.test/v1",
            "provider": "openai",
            "model": "target-model",
            "attacker_api_key": "original-attacker-secret",
            "attacker_api_base_url": "https://attacker-profile-original.test/v1",
            "attacker_provider": "openai",
            "attacker_model": "attacker-model",
            "judge_api_key": "original-judge-secret",
            "judge_api_base_url": "https://judge-profile-original.test/v1",
            "judge_provider": "openai",
            "judge_model": "judge-model",
            "judge_mode": "custom",
        }

    monkeypatch.setattr(store, "resolve_profile_config", original_profiles)
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        mode="live",
        target_config={
            "target_profile_id": "target-profile",
            "attacker_profile_id": "attacker-profile",
            "judge_profile_id": "judge-profile",
        },
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    config = plan.technique_configs[0]
    refreshed = {
        "target_profile_id": "target-profile",
        "attacker_profile_id": "attacker-profile",
        "judge_profile_id": "judge-profile",
        "api_key": "refreshed-target-secret",
        "attacker_api_key": "refreshed-attacker-secret",
        "judge_api_key": "refreshed-judge-secret",
        "api_base_url": "https://target-profile-mutated.test/v1",
        "attacker_api_base_url": "https://attacker-profile-mutated.test/v1",
        "judge_api_base_url": "https://judge-profile-mutated.test/v1",
    }

    reapplied = store._pinned_runtime_config(refreshed, config)

    assert reapplied["api_key"] == "refreshed-target-secret"
    assert reapplied["attacker_api_key"] == "refreshed-attacker-secret"
    assert reapplied["judge_api_key"] == "refreshed-judge-secret"
    assert reapplied["api_base_url"] == "https://target-profile-original.test/v1"
    assert reapplied["attacker_api_base_url"] == (
        "https://attacker-profile-original.test/v1"
    )
    assert reapplied["judge_api_base_url"] == "https://judge-profile-original.test/v1"


def test_plan_pins_target_environment_endpoint_before_environment_changes(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    monkeypatch.setenv("TARGET_API_BASE_URL", "https://environment-original.test/v1")
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        target_config={"model": "target-model"},
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    monkeypatch.setenv("TARGET_API_BASE_URL", "https://environment-mutated.test/v1")

    runtime = store._pinned_runtime_config({}, plan.technique_configs[0])
    target = store._build_chat_client(runtime)

    assert runtime["api_base_url"] == "https://environment-original.test/v1"
    assert runtime["judge_api_base_url"] == "https://environment-original.test/v1"
    assert target.api_base_url == "https://environment-original.test/v1"


@pytest.mark.parametrize(
    "endpoint_override",
    [
        {"api_base_url": "https://target-drift.test/v1"},
        {"attacker_api_base_url": "https://attacker-drift.test/v1"},
        {"judge_api_base_url": "https://judge-drift.test/v1"},
        {"judge_local": False},
    ],
)
def test_retry_rejects_explicit_endpoint_or_judge_local_drift(
    monkeypatch,
    endpoint_override,
) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        target_config={
            "model": "target-model",
            "api_base_url": "https://target-original.test/v1",
            "attacker_api_base_url": "https://attacker-original.test/v1",
            "judge_api_base_url": "https://judge-original.test/v1",
            "judge_local": True,
            "judge_mode": "custom",
            "judge_model": "judge-model",
        },
    )
    state = store._experiments[payload["experiment_id"]]
    state.status = "failed"
    started.clear()

    with pytest.raises(ValueError, match="behavioral configuration.*experiment revision"):
        store.retry_experiment_errors(
            state.experiment_id,
            {"api_key": "refreshed-key", **endpoint_override},
        )

    assert started == []


def test_credential_only_retry_does_not_invent_changed_environment_endpoint(
    monkeypatch,
) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    monkeypatch.setenv("TARGET_API_BASE_URL", "https://environment-original.test/v1")
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        target_config={"model": "target-model"},
    )
    state = store._experiments[payload["experiment_id"]]
    state.status = "failed"
    started.clear()
    monkeypatch.setenv("TARGET_API_BASE_URL", "https://environment-mutated.test/v1")

    store.retry_experiment_errors(
        state.experiment_id,
        {"api_key": "refreshed-key"},
    )

    assert len(started) == 1


def test_execution_uses_pinned_version_after_catalog_default_changes(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    catalog = _versioned_legacy_catalog()
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=catalog)
    dataset = store.create_dataset(name="Pinned route", prompts=[{"text": "extract"}])
    payload = store.create_experiment(
        name="Pinned route",
        dataset_id=dataset["dataset_id"],
        technique_keys=["system_prompt_extraction"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    pinned = plan.technique_configs[0]
    assert (pinned.technique_version, pinned.implementation_id) == (
        "1.0.0",
        "legacy-v1",
    )
    catalog.set_default(
        "system_prompt_extraction",
        implementation_id="legacy-v2",
        version="2.0.0",
    )

    summary, _ = store._execute_single(
        technique_key="system_prompt_extraction",
        technique_config=pinned,
        target_kind="chatbot",
        mode="demo",
        target_config={"unsafe_intent": "extract"},
    )

    events = store.normalized_events_by_run[summary.run_id]
    assert {event.technique_version for event in events} == {"1.0.0"}
    assert {event.implementation_id for event in events} == {"legacy-v1"}


def test_live_execution_reapplies_pinned_model_after_profile_resolution(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_versioned_legacy_catalog(),
    )
    dataset = store.create_dataset(name="Pinned model", prompts=[{"text": "extract"}])
    payload = store.create_experiment(
        name="Pinned model",
        dataset_id=dataset["dataset_id"],
        technique_keys=["system_prompt_extraction"],
        target_config={
            "provider": "planned-provider",
            "target_profile_id": "planned-profile",
            "model": "planned-model",
        },
        mode="demo",
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    pinned = plan.technique_configs[0]
    resolved_inputs = []
    captured = {}

    matching_profile = store._pinned_runtime_config(
        {
            "target_profile_id": "planned-profile",
            "api_key": "matching-profile-refresh",
        },
        pinned,
    )
    assert matching_profile["api_key"] == "matching-profile-refresh"

    def resolve_pinned_profile(config):
        resolved_inputs.append(dict(config))
        assert "api_key" not in config
        assert config["attacker_api_key"] == "attacker-refresh"
        assert config["judge_api_key"] == "judge-refresh"
        return {
            **config,
            "provider": "vault-default-provider",
            "model": "vault-default-model",
            "api_key": "planned-profile-secret",
        }

    monkeypatch.setattr(store, "resolve_profile_config", resolve_pinned_profile)

    def capture_config(config):
        captured.update(config)
        raise RuntimeError("stop after resolved config")

    monkeypatch.setattr(store, "_build_chat_client", capture_config)

    with pytest.raises(RuntimeError, match="stop after resolved config"):
        store._execute_single(
            technique_key="system_prompt_extraction",
            technique_config=pinned,
            target_kind="chatbot",
            mode="live",
            target_config={
                "api_key": "old-base-secret",
                "attacker_api_key": "attacker-refresh",
                "judge_api_key": "judge-refresh",
            },
        )

    assert len(resolved_inputs) == 1
    assert captured["provider"] == "planned-provider"
    assert captured["model"] == "planned-model"
    assert captured["api_key"] == "planned-profile-secret"
    assert captured["attacker_api_key"] == "attacker-refresh"
    assert captured["judge_api_key"] == "judge-refresh"


def test_live_direct_config_does_not_resolve_synthetic_plan_profiles(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_versioned_legacy_catalog(),
    )
    dataset = store.create_dataset(name="Direct model", prompts=[{"text": "extract"}])
    payload = store.create_experiment(
        name="Direct model",
        dataset_id=dataset["dataset_id"],
        technique_keys=["system_prompt_extraction"],
        target_config={
            "provider": "direct-provider",
            "model": "direct-model",
            "judge_mode": "target",
        },
        mode="demo",
    )
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None
    pinned = plan.technique_configs[0]
    resolved_inputs = []
    captured = {}

    def resolve_direct_config(config):
        resolved_inputs.append(dict(config))
        assert "target_profile_id" not in config
        assert "judge_profile_id" not in config
        return {
            **config,
            "provider": "credential-default-provider",
            "model": "credential-default-model",
            "api_key": "refreshed-secret",
        }

    def capture_config(config):
        captured.update(config)
        raise RuntimeError("stop after direct config")

    monkeypatch.setattr(store, "resolve_profile_config", resolve_direct_config)
    monkeypatch.setattr(store, "_build_chat_client", capture_config)

    with pytest.raises(RuntimeError, match="stop after direct config"):
        store._execute_single(
            technique_key="system_prompt_extraction",
            technique_config=pinned,
            target_kind="chatbot",
            mode="live",
            target_config={"api_key": "refreshed-secret"},
        )

    assert len(resolved_inputs) == 1
    assert captured["provider"] == "direct-provider"
    assert captured["model"] == "direct-model"
    assert captured["judge_mode"] == "target"
    assert captured["api_key"] == "refreshed-secret"


def test_retry_rejects_model_identity_drift_before_worker(monkeypatch) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        repeat_count=1,
        technique_configs=[
            {
                "technique_key": "probe",
                "parameters": {"threshold": 4},
                "model_roles": {
                    "target": {
                        "profile_id": "low-profile",
                        "provider": "openai",
                        "model": "low-model",
                    }
                },
            },
            {
                "technique_key": "probe",
                "parameters": {"threshold": 6},
                "model_roles": {
                    "target": {
                        "profile_id": "high-profile",
                        "provider": "openai",
                        "model": "high-model",
                    }
                },
            },
        ],
    )
    state = store._experiments[payload["experiment_id"]]
    state.status = "failed"
    started.clear()

    with pytest.raises(ValueError, match="pinned model role 'target'"):
        store.retry_experiment_errors(
            state.experiment_id,
            {"api_key": "refreshed-key", "model": "different-model"},
        )

    assert started == []

    store.retry_experiment_errors(
        state.experiment_id,
        {"api_key": "refreshed-key", "model": "target-model"},
    )

    assert len(started) == 1
    retry_batch = store.get_retry_batch(
        state.experiment_id,
        started[0]._args[1],
    )
    config_by_cell = {
        cell.cell_id: cell.technique_config_id
        for cell in state.plan.cells
    }
    assert {
        config_by_cell[cell_id]
        for cell_id in retry_batch.cell_ids
    } == {config.technique_config_id for config in state.plan.technique_configs}


def test_execution_uses_pinned_prompt_snapshot_after_dataset_mutation(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=_planning_catalog())
    dataset = store.create_dataset(
        name="Pinned prompt",
        prompts=[
            {
                "text": "original intent",
                "category": "original-category",
                "gold_answer": "original answer",
            }
        ],
    )
    payload = store.create_experiment(
        name="Pinned prompt",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[payload["experiment_id"]]
    state.dataset["prompts"][0].update(
        {
            "text": "mutated intent",
            "category": "mutated-category",
            "gold_answer": "mutated answer",
        }
    )
    captured = []

    def fake_execute_single(**kwargs):
        captured.append(dict(kwargs["target_config"]))
        return RunSummary(
            run_id="run-pinned-prompt",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment(state.experiment_id, {"model": "target-model"}, "target-key")

    assert captured[0]["unsafe_intent"] == "original intent"
    assert captured[0]["gold_answer"] == "original answer"
    assert state.cell_results[0]["prompt_text"] == "original intent"
    assert state.cell_results[0]["prompt_category"] == "original-category"


def test_plan_rematerialization_is_idempotent_and_conflicting_content_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    attempts = _RecordingAttemptPersistence([])
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    payload = _create_planned_experiment(store, repeat_count=1)
    plan = store._experiments[payload["experiment_id"]].plan
    assert plan is not None

    store._persist_experiment_plan(plan)

    assert len(attempts._configs) == 2
    assert len(attempts._cells) == 4
    with pytest.raises(
        ValueError,
        match="technique configuration spec hash does not match canonical content",
    ):
        attempts.upsert_technique_config(
            replace(attempts.config_records[0], spec_hash="different-spec")
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"technique_configs": [{"technique_key": "missing"}]},
            "not registered",
        ),
        (
            {"technique_configs": [{"technique_key": "probe", "active": False}]},
            "inactive",
        ),
        (
            {"technique_configs": [{"technique_key": "probe", "preset": "missing"}]},
            "preset 'missing' is not defined",
        ),
        (
            {
                "technique_configs": [
                    {"technique_key": "probe", "parameters": {"threshold": "bad"}}
                ]
            },
            "invalid technique parameters",
        ),
        (
            {
                "target_config": {
                    "provider": "openai",
                    "target_profile_id": "target-profile",
                }
            },
            "required model role 'target'",
        ),
        (
            {
                "technique_configs": [
                    {
                        "technique_key": "probe",
                        "limits": {"access_token": "must-not-persist"},
                    }
                ]
            },
            "resolved limits must not contain credential fields",
        ),
        (
            {
                "technique_configs": [
                    {
                        "technique_key": "probe",
                        "judge_policies": {
                            "target_judge": {"auth_token": "must-not-persist"}
                        },
                    }
                ]
            },
            "judge policies must not contain credential fields",
        ),
        (
            {
                "technique_configs": [
                    {
                        "technique_key": "probe",
                        "limits": {"note": "Bearer sk-supersecret123"},
                    }
                ]
            },
            "resolved limits must not contain credential fields",
        ),
    ],
)
def test_invalid_plan_fails_before_persistence_or_worker(
    monkeypatch,
    overrides,
    message,
) -> None:
    started = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    events: list[str] = []
    store = EvaluationStore(
        persistence=_PlanningRunPersistence(events),
        catalog=_planning_catalog(),
    )

    with pytest.raises(ValueError, match=message):
        _create_planned_experiment(store, repeat_count=1, **overrides)

    assert events == []
    assert started == []
    assert store._experiments == {}


def test_safe_noncredential_configuration_text_remains_canonical(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_planning_catalog(),
    )

    created = _create_planned_experiment(
        store,
        technique_configs=[
            {"technique_key": "probe", "limits": {"note": "skill-building"}}
        ],
        repeat_count=1,
    )
    plan = store._experiments[created["experiment_id"]].plan

    assert plan is not None
    assert plan.technique_configs[0].effective_limits["note"] == "skill-building"
    assert plan.technique_configs[0].resolved_spec["effective_limits"]["note"] == (
        "skill-building"
    )


def test_dataset_preserves_gold_answer_per_datapoint() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())

    dataset = store.create_dataset(
        name="Mixed references",
        prompts=[
            {"text": "question with reference", "gold_answer": "  B  "},
            {"text": "question without reference"},
            {"text": "question with empty reference", "gold_answer": "   "},
        ],
    )

    assert dataset["prompts"][0]["gold_answer"] == "B"
    assert "gold_answer" not in dataset["prompts"][1]
    assert "gold_answer" not in dataset["prompts"][2]


def test_dataset_preserves_falsey_scalar_gold_answers() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())

    dataset = store.create_dataset(
        name="Scalar references",
        prompts=[
            {"text": "zero reference", "gold_answer": 0},
            {"text": "boolean reference", "gold_answer": False},
        ],
    )

    assert dataset["prompts"][0]["gold_answer"] == "0"
    assert dataset["prompts"][1]["gold_answer"] == "false"


def test_experiment_report_includes_wilson_ci_and_score_stddev() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())

    report = store._build_experiment_report(
        [
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 100, "asr": 100},
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 80, "asr": 100},
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 70, "asr": 100},
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 20, "asr": 0},
        ]
    )

    assert report["tested"] == 4
    assert report["overall_asr"] == 75
    assert report["asr_ci_low"] == 30.1
    assert report["asr_ci_high"] == 95.4
    assert report["score_stddev"] == 29.5
    assert report["technique_stats"]["crescendo_liar"]["asr_ci_low"] == 30.1


def test_attempt_report_uses_selected_completion_and_all_attempts(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.PROVIDER,
        run_id="run-failed-1",
    )
    _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.TIMEOUT,
        run_id="run-failed-2",
    )
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
        run_id="run-selected",
    )
    state.cell_results = [_flattened_planned_result(state)]

    report = store._build_experiment_report(state.cell_results)

    assert report["tested"] == 1
    assert report["avg_score"] == 80
    assert report["operational"]["planned_cells"] == 2
    assert report["operational"]["total_attempts"] == 3
    assert report["operational"]["retry_recovery_rate"] == 100.0
    assert report["operational"]["failure_category_counts"] == {
        "provider": 1,
        "timeout": 1,
    }
    assert report["operational"]["total_attempt_cost"] is None
    assert report["canonical_rows"][0]["selected_attempt_id"] == selected.attempt_id
    assert report["canonical_rows"][0]["attempt_count"] == 3


def test_attempt_history_projection_preserves_order_zero_and_pointer_flags(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.PROVIDER,
        run_id="run-failed",
    )
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
        run_id="run-zero",
    )

    history = store.get_experiment_cell_attempts(
        state.experiment_id,
        failed.cell_id,
    )

    assert [item["attempt_id"] for item in history["items"]] == [
        failed.attempt_id,
        selected.attempt_id,
    ]
    assert [item["attempt_number"] for item in history["items"]] == [1, 2]
    assert history["selected_attempt_id"] == selected.attempt_id
    assert history["latest_attempt_id"] == selected.attempt_id
    assert history["active_attempt_id"] is None
    assert history["items"][0]["is_selected"] is False
    assert history["items"][1]["is_selected"] is True
    assert history["items"][1]["is_latest"] is True
    assert history["items"][1]["is_active"] is False
    assert history["items"][1]["score"] == 0
    assert "worker_id" not in json.dumps(history)
    with pytest.raises(KeyError):
        store.get_experiment_cell_attempts("exp-other", failed.cell_id)


def test_attempt_history_labels_active_and_late_evidence_without_selection(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    assert state.plan is not None
    planned = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    obsolete = attempts.create_attempt(
        planned.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
    )
    assert attempts.claim_attempt(
        obsolete.attempt_id,
        worker_id="obsolete-worker",
        lease_seconds=0,
    )
    active = attempts.create_attempt(
        planned.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
    )
    assert attempts.claim_attempt(
        active.attempt_id,
        worker_id="active-worker",
        lease_seconds=60,
    )
    late = attempts.complete_attempt(
        obsolete.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=100,
    )

    history = store.get_experiment_cell_attempts(state.experiment_id, planned.cell_id)
    by_id = {item["attempt_id"]: item for item in history["items"]}

    assert late.late_completion is True
    assert by_id[obsolete.attempt_id]["late_completion"] is True
    assert by_id[obsolete.attempt_id]["is_selected"] is True
    assert by_id[active.attempt_id]["is_active"] is True
    assert by_id[active.attempt_id]["is_latest"] is True
    assert history["active_attempt_id"] == active.attempt_id


def test_retry_batch_uses_selected_attempt_not_historical_failure(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
    )
    _finish_planned_state(state)

    batch = store.create_retry_batch(state.experiment_id, action="restart")

    assert batch.cell_ids == (state.plan.cells[1].cell_id,)


@pytest.mark.parametrize("recoverable", (False, True))
def test_explicit_retry_respects_selected_cancellation_recoverability(
    monkeypatch,
    recoverable: bool,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    assert state.plan is not None
    planned = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    cancelled = attempts.create_attempt(
        planned.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
        recoverable_cancellation=recoverable,
    )
    attempts.complete_attempt(
        cancelled.attempt_id,
        execution_status=ExecutionStatus.CANCELLED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.CANCELLED,
    )
    _finish_planned_state(state)

    if not recoverable:
        with pytest.raises(ValueError, match="cancelled|cancellation|revision"):
            store.create_retry_batch(
                state.experiment_id,
                action="restart",
                cell_ids=(planned.cell_id,),
            )
        return

    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(planned.cell_id,),
    )
    assert batch.cell_ids == (planned.cell_id,)


def test_retry_batch_rejects_behavioral_unknown_and_new_repeat_changes(monkeypatch) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)
    _finish_planned_state(state)

    with pytest.raises(ValueError, match="experiment revision"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            behavioral_overrides={"judge_model": "changed"},
        )
    with pytest.raises(ValueError, match="experiment revision"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            credential_overrides={"api_base_url": "https://changed.test/v1"},
        )
    with pytest.raises(ValueError, match="new_repeat requires an experiment revision"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            completed_policy="new_repeat",
        )


def test_duplicate_retry_request_starts_only_one_worker(monkeypatch) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)
    _finish_planned_state(state)
    started: list[threading.Thread] = []
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: started.append(self))

    first = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        request_id="request-1",
        credential_overrides={
            "api_key": "secret-a",
        },
    )
    replay = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        request_id="request-1",
        credential_overrides={
            "api_key": "rotated-secret",
        },
    )

    assert replay.retry_batch_id == first.retry_batch_id
    assert len(started) == 1
    assert "secret-a" not in repr(store.get_retry_batch(state.experiment_id, first.retry_batch_id))
    assert "rotated-secret" not in repr(store.get_retry_batch(state.experiment_id, first.retry_batch_id))

    with pytest.raises(ValueError, match="different retry request"):
        store.create_retry_batch(
            state.experiment_id,
            action="resume",
            request_id="request-1",
            credential_overrides={"api_key": "secret-a"},
        )
    with pytest.raises(ValueError, match="different retry request"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            cell_ids=first.cell_ids,
            request_id="request-1",
            credential_overrides={"api_key": "secret-a"},
        )
    with pytest.raises(ValueError, match="different retry request"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            request_id="request-1",
            requested_by="different-operator",
            credential_overrides={"api_key": "secret-a"},
        )
    with pytest.raises(ValueError, match="different retry request"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            request_id="request-1",
            credential_overrides={"target_profile_id": "different-profile"},
        )


def test_initial_attempts_use_execution_generation_one(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: (
            RunSummary(
                run_id="run-initial-generation",
                technique_key=kwargs["technique_key"],
                target_kind="chatbot",
                status="completed",
                score=10,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
            ),
            object(),
        ),
    )

    store._run_experiment(state.experiment_id, {"model": "target-model"}, "")

    generations = {
        attempt.execution_generation
        for cell in state.plan.cells
        for attempt in attempts.list_attempts(cell.cell_id)
    }
    assert generations == {1}


def test_initial_worker_projects_inconclusive_placeholder_zero_as_null_and_continues(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    calls: list[str] = []

    def inconclusive(**kwargs):
        calls.append(str(kwargs["cell_id"]))
        return (
            RunSummary(
                run_id=f"run-inconclusive-{len(calls)}",
                technique_key=kwargs["technique_key"],
                target_kind="chatbot",
                status="error",
                score=0,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
                error="judge output was not parseable",
                execution_status=ExecutionStatus.FAILED,
                result_disposition=ResultDisposition.INCONCLUSIVE,
                failure_category=FailureCategory.JUDGE_PARSE,
            ),
            object(),
        )

    monkeypatch.setattr(store, "_execute_single", inconclusive)

    store._run_experiment(state.experiment_id, {"model": "target-model"}, "")

    assert calls == [cell.cell_id for cell in state.plan.cells]
    assert len(state.cell_results) == 2
    for cell in state.cell_results:
        assert cell["status"] == "error"
        assert cell["score"] is None
        assert cell["asr"] is None
        assert cell["execution_status"] == ExecutionStatus.FAILED.value
        assert cell["result_disposition"] == ResultDisposition.INCONCLUSIVE.value
        assert cell["failure_category"] == FailureCategory.JUDGE_PARSE.value
        selected = attempts.get_attempt(str(cell["selected_attempt_id"]))
        assert selected.score is None


def test_initial_worker_preserves_authoritative_tested_zero(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)

    def tested_zero(**kwargs):
        return (
            RunSummary(
                run_id=f"run-zero-{kwargs['cell_id']}",
                technique_key=kwargs["technique_key"],
                target_kind="chatbot",
                status="completed",
                score=0,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
                execution_status=ExecutionStatus.COMPLETED,
                result_disposition=ResultDisposition.TESTED,
            ),
            object(),
        )

    monkeypatch.setattr(store, "_execute_single", tested_zero)

    store._run_experiment(state.experiment_id, {"model": "target-model"}, "")

    assert len(state.cell_results) == 2
    for cell in state.cell_results:
        assert cell["status"] == "completed"
        assert cell["score"] == 0
        assert cell["asr"] == 0
        assert cell["execution_status"] == ExecutionStatus.COMPLETED.value
        assert cell["result_disposition"] == ResultDisposition.TESTED.value
        assert cell["failure_category"] is None
        selected = attempts.get_attempt(str(cell["selected_attempt_id"]))
        assert selected.score == 0


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_initial_worker_agrees_on_inconclusive_null_and_tested_zero(
    monkeypatch,
) -> None:
    import psycopg

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments, "
            "experiment_dataset_prompts, experiment_datasets CASCADE"
        )

    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    run_persistence = PostgreSQLRunPersistence(dsn)
    attempts = PostgreSQLExperimentAttemptPersistence(dsn)
    store = EvaluationStore(
        persistence=run_persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
        technique_runtime_persistence=NoopTechniqueRuntimePersistence(),
    )
    dataset = store.create_dataset(
        name="Null and zero projection",
        prompts=[{"text": "inconclusive"}, {"text": "tested zero"}],
    )
    created = store.create_experiment(
        name="Null and zero projection",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    call_count = 0

    def execute(**kwargs):
        nonlocal call_count
        call_count += 1
        inconclusive = call_count == 1
        return (
            RunSummary(
                run_id=f"run-postgresql-{call_count}",
                technique_key=kwargs["technique_key"],
                target_kind="chatbot",
                status="error" if inconclusive else "completed",
                score=0,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
                error="judge parse" if inconclusive else None,
                execution_status=(
                    ExecutionStatus.FAILED
                    if inconclusive
                    else ExecutionStatus.COMPLETED
                ),
                result_disposition=(
                    ResultDisposition.INCONCLUSIVE
                    if inconclusive
                    else ResultDisposition.TESTED
                ),
                failure_category=(
                    FailureCategory.JUDGE_PARSE if inconclusive else None
                ),
            ),
            object(),
        )

    monkeypatch.setattr(store, "_execute_single", execute)
    assert len(workers) == 1

    workers[0]._target(*workers[0]._args)

    assert call_count == 2
    assert state.status == "completed"
    assert [cell["score"] for cell in state.cell_results] == [None, 0]
    assert [cell["asr"] for cell in state.cell_results] == [None, 0]
    assert [
        attempts.list_attempts(planned.cell_id)[0].score
        for planned in state.plan.cells
    ] == [None, 0]
    persisted = next(
        experiment
        for experiment in run_persistence.load_all_experiments()
        if experiment["experiment_id"] == state.experiment_id
    )
    persisted_cells = {
        cell["prompt_text"]: cell for cell in persisted["cell_results"]
    }
    assert persisted_cells["inconclusive"]["status"] == "error"
    assert persisted_cells["inconclusive"]["score"] is None
    assert persisted_cells["inconclusive"]["asr"] is None
    assert persisted_cells["tested zero"]["status"] == "completed"
    assert persisted_cells["tested zero"]["score"] == 0
    assert persisted_cells["tested zero"]["asr"] == 0


def test_initial_completion_checks_active_generation_before_public_projection(
    monkeypatch,
) -> None:
    real_thread_start = threading.Thread.start
    attempts = _BlockingGenerationActiveCheck()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    monkeypatch.setattr(threading.Thread, "start", real_thread_start)
    monkeypatch.setattr(store, "_execute_single", _completed_retry_summary)
    attempts.block_next_check = True
    worker = threading.Thread(
        target=store._run_experiment,
        args=(state.experiment_id, {"model": "target-model"}, ""),
    )

    worker.start()
    assert attempts.check_started.wait(timeout=5)
    observed = store.get_experiment(state.experiment_id)
    attempts.release_check.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert observed["status"] == "running"
    assert state.status == "running"


def test_retry_attempt_generation_equals_bound_batch_generation(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="failed",
    )

    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: (
            RunSummary(
                run_id="run-retry-generation",
                technique_key=kwargs["technique_key"],
                target_kind="chatbot",
                status="completed",
                score=10,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
            ),
            object(),
        ),
    )

    store._run_retry_batch_once(state.experiment_id, batch.retry_batch_id, {})

    bound = attempts.get_retry_batch(batch.retry_batch_id)
    retry = attempts.list_attempts(failed.cell_id)[-1]
    assert bound.execution_generation == 2
    assert retry.execution_generation == bound.execution_generation


def test_retry_completion_checks_active_generation_before_public_projection(
    monkeypatch,
) -> None:
    real_thread_start = threading.Thread.start
    attempts = _BlockingGenerationActiveCheck()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="failed",
    )
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )
    monkeypatch.setattr(store, "_execute_single", _completed_retry_summary)
    monkeypatch.setattr(threading.Thread, "start", real_thread_start)
    attempts.block_next_check = True
    worker = threading.Thread(
        target=store._run_retry_batch_once,
        args=(state.experiment_id, batch.retry_batch_id, {}),
    )

    worker.start()
    assert attempts.check_started.wait(timeout=5)
    observed = store.get_experiment(state.experiment_id)
    attempts.release_check.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert observed["status"] == "running"
    assert state.status == "running"


def test_compatibility_retry_checks_active_generation_before_public_projection(
    monkeypatch,
) -> None:
    real_thread_start = threading.Thread.start
    attempts = _BlockingGenerationActiveCheck()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    assert state.plan is not None
    planned_cell = state.plan.cells[0]
    technique_config = next(
        config
        for config in state.plan.technique_configs
        if config.technique_config_id == planned_cell.technique_config_id
    )
    prompt = {
        "id": planned_cell.prompt_id,
        "text": planned_cell.prompt_text,
        "category": planned_cell.prompt_category,
    }
    monkeypatch.setattr(store, "_execute_single", _completed_retry_summary)
    monkeypatch.setattr(threading.Thread, "start", real_thread_start)
    state.status = "failed"
    state.error = "retryable prior execution"
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=state._execution_generation,
        owner_id=state._execution_owner_id,
        status="failed",
        error=state.error,
    )
    attempts.block_next_check = True
    worker = threading.Thread(
        target=store._run_experiment_retry,
        args=(
            state.experiment_id,
            [(technique_config, prompt, planned_cell.repeat_index)],
            {"model": "target-model"},
            "",
        ),
    )

    worker.start()
    assert attempts.check_started.wait(timeout=5)
    observed = store.get_experiment(state.experiment_id)
    attempts.release_check.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert observed["status"] == "running"
    assert state.status == "running"


def test_retry_start_failure_dispatches_no_provider_calls(monkeypatch) -> None:
    persistence = _FailingGenerationBeginPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(name="generation failure", prompts=[{"text": "intent"}])
    created = store.create_experiment(
        name="generation failure",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    state.status = "failed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="failed",
    )
    persistence.fail_begin = True
    workers = []
    provider_calls = []
    timers = _capture_retry_timers(monkeypatch)
    monkeypatch.setattr(store, "_start_retry_batch_worker", lambda *args, **kwargs: workers.append(args))
    monkeypatch.setattr(store, "_execute_single", lambda **kwargs: provider_calls.append(kwargs))

    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )

    assert batch.status == "requested"
    assert workers == []
    assert provider_calls == []
    assert batch.retry_batch_id in store._retry_recovery_timers
    assert len(timers) == 1


def test_generation_bind_failure_dispatches_no_provider_calls(monkeypatch) -> None:
    attempts = _FailingGenerationBindAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    state.status = "failed"
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="failed",
    )
    workers = []
    timers = _capture_retry_timers(monkeypatch)
    monkeypatch.setattr(store, "_start_retry_batch_worker", lambda *args, **kwargs: workers.append(args))

    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )

    assert batch.status == "requested"
    assert workers == []
    assert batch.retry_batch_id in store._retry_recovery_timers
    assert len(timers) == 1


def test_private_execution_generation_and_owner_are_not_serialized(monkeypatch) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)

    serialized = store._serialize_experiment(state)

    assert state._execution_generation == 1
    assert state._execution_owner_id == f"initial:{state.experiment_id}"
    assert "execution_generation" not in serialized
    assert "execution_owner_id" not in serialized


def test_stale_terminal_callback_cannot_finish_new_retry_generation(monkeypatch) -> None:
    persistence = _BlockingOwnedTerminalPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(name="stale terminal", prompts=[{"text": "intent"}])
    with monkeypatch.context() as planning:
        planning.setattr("app.api.store.threading.Thread.start", lambda self: None)
        created = store.create_experiment(
            name="stale terminal",
            dataset_id=dataset["dataset_id"],
            technique_keys=["probe"],
            target_config={"model": "target-model"},
            mode="demo",
        )
    monkeypatch.setattr(store, "_start_retry_batch_worker", lambda *args, **kwargs: True)
    state = store._experiments[created["experiment_id"]]
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    state.status = "failed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="failed",
    )
    first = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )
    state.status = "failed"
    persistence.block_next_terminal = True
    callback = threading.Thread(
        target=store._reconcile_terminal_experiment_status,
        args=(state.experiment_id, "failed"),
    )
    callback.start()
    assert persistence.terminal_applied.wait(timeout=5)

    second = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )
    persistence.release_terminal.set()
    callback.join(timeout=5)

    durable = persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == second.execution_generation
    assert durable["execution_owner_id"] == second.retry_batch_id
    assert durable["status"] == "running"
    assert first.execution_generation < second.execution_generation


def test_initial_worker_ownership_movement_between_cells_stops_later_dispatch(
    monkeypatch,
) -> None:
    class MoveOwnershipAfterFirstCell(NoopRunPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.moved = False

        def save_experiment_cell(
            self,
            *,
            experiment_id,
            cell,
            generation=None,
            owner_id=None,
        ) -> bool:
            if self.moved:
                return super().save_experiment_cell(
                    experiment_id=experiment_id,
                    cell=cell,
                    generation=generation,
                    owner_id=owner_id,
                )
            self.moved = True
            assert self.reconcile_owned_experiment_terminal_status(
                experiment_id,
                generation=1,
                owner_id=f"initial:{experiment_id}",
                status="completed",
            )
            assert self.begin_experiment_execution(
                experiment_id,
                owner_id="retry-batch-new-owner",
                expected_generation=1,
                active_status="running",
            ) == 2
            return super().save_experiment_cell(
                experiment_id=experiment_id,
                cell=cell,
                generation=generation,
                owner_id=owner_id,
            )

    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    persistence = MoveOwnershipAfterFirstCell()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(
        name="ownership movement",
        prompts=[{"text": "first"}, {"text": "second"}],
    )
    created = store.create_experiment(
        name="ownership movement",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    provider_calls: list[str] = []

    def execute(**kwargs):
        provider_calls.append(str(kwargs["cell_id"]))
        return _completed_retry_summary(**kwargs)

    monkeypatch.setattr(store, "_execute_single", execute)
    assert len(workers) == 1
    workers[0]._target(*workers[0]._args)

    assert provider_calls == [store._experiments[created["experiment_id"]].plan.cells[0].cell_id]
    assert store._experiments[created["experiment_id"]].cell_results == []
    durable = persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == 2
    assert durable["execution_owner_id"] == "retry-batch-new-owner"


def test_initial_dispatch_renews_exact_attempt_after_preparation_interleaving(
    monkeypatch,
) -> None:
    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    persistence = NoopRunPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(
        name="initial preparation ownership movement",
        prompts=[{"text": "intent"}],
    )
    created = store.create_experiment(
        name="initial preparation ownership movement",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    original_cell_config = store._experiment_cell_config
    moved = False

    def move_ownership_during_preparation(**kwargs):
        nonlocal moved
        prepared = original_cell_config(**kwargs)
        if moved:
            return prepared
        moved = True
        assert state.plan is not None
        attempt = attempts.list_attempts(state.plan.cells[0].cell_id)[0]
        attempts.complete_attempt(
            attempt.attempt_id,
            execution_status=ExecutionStatus.FAILED,
            result_disposition=ResultDisposition.INCONCLUSIVE,
            failure_category=FailureCategory.WORKER_LOST,
        )
        assert persistence.reconcile_owned_experiment_terminal_status(
            state.experiment_id,
            generation=1,
            owner_id=f"initial:{state.experiment_id}",
            status="failed",
        )
        assert persistence.begin_experiment_execution(
            state.experiment_id,
            owner_id="retry-owner-after-initial-preparation",
            expected_generation=1,
            active_status="running",
        ) == 2
        return prepared

    monkeypatch.setattr(store, "_experiment_cell_config", move_ownership_during_preparation)
    provider_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: provider_calls.append(kwargs),
    )

    assert len(workers) == 1
    workers[0]._target(*workers[0]._args)

    assert moved is True
    assert provider_calls == []
    durable = persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == 2
    assert durable["execution_owner_id"] == "retry-owner-after-initial-preparation"


def test_planless_generation_zero_initial_worker_dispatches_no_provider_call() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.technique_keys = ["aim"]
    state.dataset["prompts"] = [
        {"id": "p1", "text": "dataset intent", "category": "cat-a"}
    ]
    state.total_cells = 1
    store._experiments[state.experiment_id] = state
    provider_calls: list[dict[str, object]] = []
    store._execute_single = lambda **kwargs: provider_calls.append(kwargs)

    store._run_experiment(
        state.experiment_id,
        {"model": "target-model"},
        "target-key",
    )

    assert provider_calls == []
    assert state.cell_results == []


def test_planless_owned_initial_worker_dispatches_no_provider_call() -> None:
    persistence = NoopRunPersistence()
    store = EvaluationStore(persistence=persistence)
    state = _experiment_state(repeat_count=1)
    state.technique_keys = ["aim"]
    state.dataset["prompts"] = [
        {"id": "p1", "text": "dataset intent", "category": "cat-a"}
    ]
    state.total_cells = 1
    state.status = "running"
    state._execution_generation = 1
    state._execution_owner_id = f"initial:{state.experiment_id}"
    persistence.save_experiment(
        experiment={
            "experiment_id": state.experiment_id,
            "status": "running",
            "execution_generation": state._execution_generation,
            "execution_owner_id": state._execution_owner_id,
        }
    )
    store._experiments[state.experiment_id] = state
    provider_calls: list[dict[str, object]] = []
    store._execute_single = lambda **kwargs: provider_calls.append(kwargs)

    store._run_experiment(
        state.experiment_id,
        {"model": "target-model"},
        "target-key",
    )

    assert provider_calls == []
    assert state.cell_results == []


def test_manual_stop_keeps_current_generation_nonterminal_with_active_attempt(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    planned = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    queued = attempts.create_attempt(
        planned.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )

    stopped = store.stop_experiment(state.experiment_id)

    assert stopped["status"] == "running"
    assert state.stop_requested is True
    assert attempts.get_attempt(queued.attempt_id).execution_status is ExecutionStatus.QUEUED
    durable = store.persistence.load_all_experiments()[0]
    assert durable["status"] == "running"
    assert state.experiment_id not in store._terminal_status_reconciliation_obligations


def test_same_owner_lost_acknowledgement_replays_one_generation_and_worker(
    monkeypatch,
) -> None:
    persistence = _LostAcknowledgementBeginPersistence()
    attempts = InMemoryExperimentAttemptPersistence()
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(name="lost acknowledgement", prompts=[{"text": "intent"}])
    created = store.create_experiment(
        name="lost acknowledgement",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    state.status = "failed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="failed",
    )
    timers = _capture_retry_timers(monkeypatch)
    workers = []
    monkeypatch.setattr(
        store,
        "_start_retry_batch_worker",
        lambda *args, **kwargs: workers.append((args, kwargs)) or True,
    )
    persistence.lose_next_acknowledgement = True

    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )
    assert batch.execution_generation is None
    assert workers == []
    assert len(timers) == 1
    timers[0].fire()

    durable = persistence.load_all_experiments()[0]
    rebound = attempts.get_retry_batch(batch.retry_batch_id)
    assert durable["execution_generation"] == 2
    assert durable["execution_owner_id"] == batch.retry_batch_id
    assert rebound.execution_generation == 2
    assert len(workers) == 1


def test_retry_ownership_conflict_is_durable_diagnostic_without_recovery_timer(
    monkeypatch,
) -> None:
    class ConflictingRetryOwner(NoopRunPersistence):
        def begin_experiment_execution(self, experiment_id, **kwargs):
            if str(kwargs["owner_id"]) != f"initial:{experiment_id}":
                raise ExperimentExecutionOwnershipConflict(
                    "experiment execution ownership conflict"
                )
            return super().begin_experiment_execution(experiment_id, **kwargs)

    persistence = ConflictingRetryOwner()
    attempts = InMemoryExperimentAttemptPersistence()
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda _thread: None)
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(name="conflicting retry", prompts=[{"text": "intent"}])
    created = store.create_experiment(
        name="conflicting retry",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    state.status = "failed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="failed",
    )
    timers = _capture_retry_timers(monkeypatch)

    requested = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )

    diagnostic = attempts.get_retry_batch(requested.retry_batch_id)
    assert diagnostic.status == "failed"
    assert [member.error_category for member in diagnostic.members] == [
        "ownership_conflict"
    ]
    assert diagnostic.execution_generation is None
    assert timers == []
    durable = persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == 1
    assert durable["execution_owner_id"] == f"initial:{state.experiment_id}"
    assert durable["status"] == "failed"


def test_stale_bound_retry_worker_is_durable_diagnostic_without_timer_or_call(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    monkeypatch.setattr(store, "_start_retry_batch_worker", lambda *args, **kwargs: True)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )
    assert batch.execution_generation is not None
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=batch.execution_generation,
        owner_id=batch.retry_batch_id,
        status="failed",
    )
    next_generation = store.persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id="retry-new-owner",
        expected_generation=batch.execution_generation,
        active_status="running",
    )
    state._execution_generation = next_generation
    state._execution_owner_id = "retry-new-owner"
    state.status = "running"
    timers = _capture_retry_timers(monkeypatch)
    provider_calls: list[object] = []
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: provider_calls.append(kwargs),
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {},
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )

    diagnostic = attempts.get_retry_batch(batch.retry_batch_id)
    assert diagnostic.status == "failed"
    assert [member.error_category for member in diagnostic.members] == [
        "ownership_mismatch"
    ]
    assert timers == []
    assert provider_calls == []
    durable = store.persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == next_generation
    assert durable["execution_owner_id"] == "retry-new-owner"
    assert durable["status"] == "running"


def test_terminal_same_owner_retry_replay_is_diagnostic_without_local_reactivation(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    workers: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        store,
        "_start_retry_batch_worker",
        lambda *args, **kwargs: workers.append((args, kwargs)) or True,
    )
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        request_id="terminal-same-owner-replay",
    )
    assert batch.execution_generation is not None
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=batch.execution_generation,
        owner_id=batch.retry_batch_id,
        status="failed",
    )
    state.status = "failed"
    workers.clear()

    store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        request_id="terminal-same-owner-replay",
    )

    assert workers == []
    assert state.status == "failed"
    diagnostic = attempts.get_retry_batch(batch.retry_batch_id)
    assert diagnostic.status == "failed"
    assert diagnostic.members[0].error_category == "ownership_mismatch"
    durable = store.persistence.load_all_experiments()[0]
    assert durable["status"] == "failed"
    assert durable["execution_generation"] == batch.execution_generation
    assert durable["execution_owner_id"] == batch.retry_batch_id


def test_retry_dispatch_renews_exact_attempt_after_preparation_interleaving(
    monkeypatch,
) -> None:
    class MoveOwnershipOnDispatchHeartbeat(InMemoryExperimentAttemptPersistence):
        before_heartbeat = None

        def heartbeat_attempt(self, *args, **kwargs):
            callback = self.before_heartbeat
            self.before_heartbeat = None
            if callback is not None:
                callback()
            return super().heartbeat_attempt(*args, **kwargs)

    attempts = MoveOwnershipOnDispatchHeartbeat()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
    )
    assert batch.execution_generation is not None

    def move_ownership() -> None:
        durable_batch = attempts.get_retry_batch(batch.retry_batch_id)
        attempt_id = durable_batch.members[0].attempt_id
        assert attempt_id is not None
        running = attempts.get_attempt(attempt_id)
        assert running.lease_expires_at is not None
        attempts._attempts[attempt_id] = replace(
            running,
            lease_expires_at=running.lease_expires_at - timedelta(seconds=61),
        )
        attempts.recover_expired_attempt(
            failed.cell_id,
            experiment_id=state.experiment_id,
            execution_generation=batch.execution_generation,
            execution_owner_id=batch.retry_batch_id,
        )
        assert store.persistence.reconcile_owned_experiment_terminal_status(
            state.experiment_id,
            generation=batch.execution_generation,
            owner_id=batch.retry_batch_id,
            status="failed",
        )
        next_generation = store.persistence.begin_experiment_execution(
            state.experiment_id,
            owner_id="retry-owner-after-preparation",
            expected_generation=batch.execution_generation,
            active_status="running",
        )
        state._execution_generation = next_generation
        state._execution_owner_id = "retry-owner-after-preparation"
        state.status = "running"

    attempts.before_heartbeat = move_ownership
    provider_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: provider_calls.append(kwargs),
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {},
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )

    assert provider_calls == []
    durable = store.persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == batch.execution_generation + 1
    assert durable["execution_owner_id"] == "retry-owner-after-preparation"
    assert durable["status"] == "running"


@pytest.mark.parametrize(
    "attempt_state",
    ["queued", "active", "terminal-success", "terminal-failed"],
)
def test_retry_request_exact_replay_precedes_mutable_member_eligibility(
    monkeypatch,
    attempt_state,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    started: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        completed_policy="exclude",
        request_id="stable-replay-request",
        requested_by="workspace-operator",
        credential_overrides={"api_key": "first-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        promotion_policy="replace_on_valid_completion",
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    if attempt_state == "active":
        assert attempts.claim_attempt(
            bound.attempt_id,
            worker_id="winner-worker",
            lease_seconds=60,
            experiment_id=state.experiment_id,
            execution_generation=batch.execution_generation,
            execution_owner_id=batch.retry_batch_id,
        )
        attempts.bind_retry_batch_attempt(
            batch.retry_batch_id,
            failed.cell_id,
            bound.attempt_id,
        )
        attempts.update_retry_batch(batch.retry_batch_id, status="running")
    elif attempt_state.startswith("terminal"):
        succeeded = attempt_state == "terminal-success"
        attempts.complete_attempt(
            bound.attempt_id,
            execution_status=(
                ExecutionStatus.COMPLETED if succeeded else ExecutionStatus.FAILED
            ),
            result_disposition=(
                ResultDisposition.TESTED
                if succeeded
                else ResultDisposition.INCONCLUSIVE
            ),
            score=10 if succeeded else None,
            failure_category=None if succeeded else FailureCategory.INTERNAL,
        )
        attempts.complete_retry_batch_cell(
            batch.retry_batch_id,
            failed.cell_id,
            status="completed" if succeeded else "failed",
            error_category=None if succeeded else "internal",
        )
        attempts.update_retry_batch(
            batch.retry_batch_id,
            status="completed" if succeeded else "failed",
        )

    replay = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        completed_policy="exclude",
        request_id="stable-replay-request",
        requested_by="workspace-operator",
        credential_overrides={"api_key": "rotated-secret"},
    )

    assert replay.retry_batch_id == batch.retry_batch_id
    assert replay.was_created is False
    assert replay.cell_ids == (failed.cell_id,)
    assert replay.members[0].attempt_id == bound.attempt_id
    assert len(attempts.list_attempts(failed.cell_id)) == 2
    assert len(started) == 1


def test_retry_request_replay_recovers_one_atomically_bound_queued_attempt(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    started: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        request_id="recover-bound-request",
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    # Simulate process loss after durable create+bind but before claim.
    store._retry_worker_batches.clear()

    replay = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        request_id="recover-bound-request",
        credential_overrides={"api_key": "fresh-secret"},
    )

    assert replay.retry_batch_id == batch.retry_batch_id
    assert len(started) == 2
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: (
            RunSummary(
                run_id="run-recovered-bound",
                technique_key=kwargs["technique_key"],
                target_kind="chatbot",
                status="completed",
                score=10,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
            ),
            object(),
        ),
    )
    started[-1]._target(*started[-1]._args)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert member.attempt_id == bound.attempt_id
    assert len(attempts.list_attempts(failed.cell_id)) == 2


def test_recovered_queued_retry_without_transient_secret_terminalizes_safely(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    state.mode = "live"
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        request_id="lost-transient-secret",
        credential_overrides={"api_key": "ephemeral-secret"},
    )
    config = state.plan.technique_configs[0]
    queued = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    dispatched: list[object] = []
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: dispatched.append(kwargs),
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {},
        recovered_after_restart=True,
    )

    recovered = attempts.get_attempt(queued.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert dispatched == []
    assert recovered.execution_status is ExecutionStatus.FAILED
    assert member.status == "failed"
    assert member.error_category == "credential_unavailable"


def test_retry_claim_loser_preserves_winner_ownership_and_completion(
    monkeypatch,
) -> None:
    class RacingAttempts(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.raced = False
            self.winner_worker_id = "winner-process-worker"

        def claim_attempt(
            self,
            attempt_id,
            *,
            worker_id,
            lease_seconds,
            **ownership,
        ):
            if worker_id.startswith("retry:") and not self.raced:
                self.raced = True
                assert super().claim_attempt(
                    attempt_id,
                    worker_id=self.winner_worker_id,
                    lease_seconds=lease_seconds,
                    **ownership,
                )
                return False
            return super().claim_attempt(
                attempt_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                **ownership,
            )

    attempts = RacingAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    dispatched: list[object] = []
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: dispatched.append(kwargs),
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    winner = attempts.get_attempt(bound.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    cell = attempts.get_cell(failed.cell_id)
    assert dispatched == []
    assert winner.execution_status is ExecutionStatus.RUNNING
    assert winner.worker_id == attempts.winner_worker_id
    assert winner.lease_expires_at is not None
    assert cell.active_attempt_id == winner.attempt_id
    assert member.status == "running"
    completed = attempts.complete_attempt(
        winner.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=10,
    )
    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )
    final_member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert completed.late_completion is False
    assert attempts.get_cell(failed.cell_id).selected_attempt_id == completed.attempt_id
    assert final_member.status == "completed"


@pytest.mark.parametrize(
    (
        "loser_path",
        "winner_execution_status",
        "winner_result_disposition",
        "winner_score",
        "winner_failure_category",
        "expected_member_status",
        "expected_error_category",
    ),
    [
        (
            "validation",
            ExecutionStatus.COMPLETED,
            ResultDisposition.TESTED,
            10,
            None,
            "completed",
            None,
        ),
        (
            "stop",
            ExecutionStatus.COMPLETED,
            ResultDisposition.TESTED,
            10,
            None,
            "completed",
            None,
        ),
        (
            "validation",
            ExecutionStatus.FAILED,
            ResultDisposition.INCONCLUSIVE,
            None,
            FailureCategory.PROVIDER,
            "failed",
            "provider",
        ),
    ],
)
def test_retry_terminalization_loser_projects_only_concurrent_winner_evidence(
    monkeypatch,
    loser_path,
    winner_execution_status,
    winner_result_disposition,
    winner_score,
    winner_failure_category,
    expected_member_status,
    expected_error_category,
) -> None:
    class ConcurrentWinnerAttempts(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.armed_attempt_id: str | None = None
            self.raced = False

        def complete_attempt(
            self,
            attempt_id,
            *,
            execution_status,
            result_disposition,
            score=None,
            run_id=None,
            failure_category=None,
        ):
            if attempt_id == self.armed_attempt_id and not self.raced:
                self.raced = True
                super().complete_attempt(
                    attempt_id,
                    execution_status=winner_execution_status,
                    result_disposition=winner_result_disposition,
                    score=winner_score,
                    failure_category=winner_failure_category,
                )
            return super().complete_attempt(
                attempt_id,
                execution_status=execution_status,
                result_disposition=result_disposition,
                score=score,
                run_id=run_id,
                failure_category=failure_category,
            )

        def complete_attempt_if_owned(self, attempt_id, **kwargs):
            if attempt_id == self.armed_attempt_id and not self.raced:
                self.raced = True
                super().complete_attempt(
                    attempt_id,
                    execution_status=winner_execution_status,
                    result_disposition=winner_result_disposition,
                    score=winner_score,
                    failure_category=winner_failure_category,
                )
            return super().complete_attempt_if_owned(attempt_id, **kwargs)

    attempts = ConcurrentWinnerAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    attempts.armed_attempt_id = bound.attempt_id
    if loser_path == "stop":
        state.stop_requested = True
    else:
        monkeypatch.setattr(
            store,
            "_retry_cell_context",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("losing worker validation failure")
            ),
        )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    winner = attempts.get_attempt(bound.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert attempts.raced is True
    assert winner.execution_status is winner_execution_status
    assert winner.result_disposition is winner_result_disposition
    assert winner.score == winner_score
    assert winner.failure_category is winner_failure_category
    assert member.status == expected_member_status
    assert member.error_category == expected_error_category
    store._complete_retry_member_from_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        winner,
    )


def test_retry_exact_concurrent_terminal_evidence_keeps_winner_projection(
    monkeypatch,
) -> None:
    class ExactConcurrentWinnerAttempts(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.armed_attempt_id: str | None = None
            self.retry_batch_id: str | None = None
            self.cell_id: str | None = None
            self.raced = False

        def complete_attempt(
            self,
            attempt_id,
            *,
            execution_status,
            result_disposition,
            score=None,
            run_id=None,
            failure_category=None,
        ):
            if attempt_id == self.armed_attempt_id and not self.raced:
                self.raced = True
                super().complete_attempt(
                    attempt_id,
                    execution_status=ExecutionStatus.FAILED,
                    result_disposition=ResultDisposition.INCONCLUSIVE,
                    failure_category=FailureCategory.INTERNAL,
                )
                super().complete_retry_batch_cell(
                    self.retry_batch_id,
                    self.cell_id,
                    status="failed",
                    error_category="internal",
                )
            return super().complete_attempt(
                attempt_id,
                execution_status=execution_status,
                result_disposition=result_disposition,
                score=score,
                run_id=run_id,
                failure_category=failure_category,
            )

        def complete_attempt_if_owned(self, attempt_id, **kwargs):
            if attempt_id == self.armed_attempt_id and not self.raced:
                self.raced = True
                super().complete_attempt(
                    attempt_id,
                    execution_status=ExecutionStatus.FAILED,
                    result_disposition=ResultDisposition.INCONCLUSIVE,
                    failure_category=FailureCategory.INTERNAL,
                )
                super().complete_retry_batch_cell(
                    self.retry_batch_id,
                    self.cell_id,
                    status="failed",
                    error_category="internal",
                )
            return super().complete_attempt_if_owned(attempt_id, **kwargs)

    attempts = ExactConcurrentWinnerAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    attempts.armed_attempt_id = bound.attempt_id
    attempts.retry_batch_id = batch.retry_batch_id
    attempts.cell_id = failed.cell_id
    monkeypatch.setattr(
        store,
        "_retry_cell_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("losing worker validation failure")
        ),
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    winner = attempts.get_attempt(bound.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert attempts.raced is True
    assert winner.execution_status is ExecutionStatus.FAILED
    assert winner.failure_category is FailureCategory.INTERNAL
    assert member.status == "failed"
    assert member.error_category == "internal"
    store._complete_retry_member_from_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        winner,
    )


@pytest.mark.parametrize("failure_path", ["completion", "terminalization"])
def test_retry_persistence_failure_keeps_running_attempt_recoverable(
    monkeypatch,
    failure_path,
) -> None:
    class FailingCompletionAttempts(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.failing_attempt_id: str | None = None
            self.completion_calls = 0

        def complete_attempt(self, attempt_id, **kwargs):
            if attempt_id == self.failing_attempt_id:
                self.completion_calls += 1
                raise RuntimeError("injected attempt completion persistence failure")
            return super().complete_attempt(attempt_id, **kwargs)

        def complete_attempt_if_owned(self, attempt_id, **kwargs):
            if attempt_id == self.failing_attempt_id:
                self.completion_calls += 1
                raise RuntimeError("injected attempt completion persistence failure")
            return super().complete_attempt_if_owned(attempt_id, **kwargs)

    attempts = FailingCompletionAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    attempts.failing_attempt_id = bound.attempt_id
    timers = _capture_retry_timers(monkeypatch)
    if failure_path == "terminalization":
        monkeypatch.setattr(
            store,
            "_retry_cell_context",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("injected validation failure")
            ),
        )
    else:
        monkeypatch.setattr(
            store,
            "_execute_single",
            lambda **kwargs: (
                RunSummary(
                    run_id="run-before-persistence-failure",
                    technique_key=kwargs["technique_key"],
                    target_kind="chatbot",
                    status="completed",
                    score=10,
                    asr=0,
                    matched_detections=0,
                    detections=(),
                    highest_severity="info",
                    conversation=[],
                ),
                object(),
            ),
        )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    observed = attempts.get_attempt(bound.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    active_timers = [timer for timer in timers if not timer.cancelled]
    assert attempts.completion_calls >= 1
    assert observed.execution_status is ExecutionStatus.RUNNING
    assert observed.worker_id is not None
    assert observed.lease_expires_at is not None
    assert attempts.get_cell(failed.cell_id).active_attempt_id == observed.attempt_id
    assert member.status == "running"
    assert len(active_timers) == 1
    assert active_timers[0].daemon is True
    assert active_timers[0].interval > 0


def test_retry_member_persistence_failure_rearms_exact_terminal_projection(
    monkeypatch,
) -> None:
    class FailingMemberAttempts(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.fail_member_completion = False

        def complete_retry_batch_cell(self, retry_batch_id, cell_id, **kwargs):
            if self.fail_member_completion:
                raise RuntimeError("injected retry member persistence failure")
            return super().complete_retry_batch_cell(
                retry_batch_id,
                cell_id,
                **kwargs,
            )

    attempts = FailingMemberAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    timers = _capture_retry_timers(monkeypatch)
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: (
            RunSummary(
                run_id="run-member-persistence-failure",
                technique_key=kwargs["technique_key"],
                target_kind="chatbot",
                status="completed",
                score=10,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
            ),
            object(),
        ),
    )
    attempts.fail_member_completion = True

    with pytest.raises(RuntimeError, match="member persistence failure"):
        store._run_retry_batch(
            state.experiment_id,
            batch.retry_batch_id,
            {"api_key": "fresh-secret"},
        )

    completed = attempts.get_attempt(bound.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert completed.execution_status is ExecutionStatus.COMPLETED
    assert completed.result_disposition is ResultDisposition.TESTED
    assert member.status == "running"
    assert len([timer for timer in timers if not timer.cancelled]) == 1

    attempts.fail_member_completion = False
    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    projected = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert projected.status == "completed"
    assert projected.error_category is None
    assert [timer for timer in timers if not timer.cancelled] == []


def test_retry_recovery_timer_rearms_after_transient_batch_read_failure(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    timers = _capture_retry_timers(monkeypatch)
    original_get_batch = attempts.get_retry_batch
    failures_remaining = [1]

    def transient_get_batch(retry_batch_id):
        if failures_remaining[0]:
            failures_remaining[0] -= 1
            raise RuntimeError("transient retry batch read failure")
        return original_get_batch(retry_batch_id)

    monkeypatch.setattr(attempts, "get_retry_batch", transient_get_batch)
    restarted = []
    monkeypatch.setattr(
        store,
        "_start_retry_batch_worker",
        lambda *args, **kwargs: restarted.append((args, kwargs)) or True,
    )

    assert store._schedule_retry_batch_recovery(
        state.experiment_id,
        batch.retry_batch_id,
        None,
    )
    timers[0].fire()

    active_timers = [timer for timer in timers if not timer.cancelled]
    assert len(active_timers) == 1
    assert active_timers[0] is not timers[0]
    active_timers[0].fire()
    assert len(restarted) == 1


def test_retry_recovery_transient_failures_stay_bounded_and_deduplicated(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    timers = _capture_retry_timers(monkeypatch)
    monkeypatch.setattr(
        attempts,
        "get_retry_batch",
        lambda _retry_batch_id: (_ for _ in ()).throw(
            RuntimeError("repeated transient retry batch read failure")
        ),
    )

    assert store._schedule_retry_batch_recovery(
        state.experiment_id,
        batch.retry_batch_id,
        None,
    )
    intervals = []
    for _ in range(4):
        active_timers = [timer for timer in timers if not timer.cancelled]
        assert len(active_timers) == 1
        intervals.append(active_timers[0].interval)
        active_timers[0].fire()

    assert intervals == sorted(intervals)
    assert len(set(intervals)) > 1
    assert all(0 < interval <= 60.05 for interval in intervals)
    assert len([timer for timer in timers if not timer.cancelled]) == 1


def test_retry_guard_occupied_expired_lease_backs_off_and_cleans_up(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    assert attempts.claim_attempt(
        bound.attempt_id,
        worker_id="expired-guard-owner",
        lease_seconds=0,
        experiment_id=state.experiment_id,
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )
    attempts.bind_retry_batch_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        bound.attempt_id,
    )
    attempts.update_retry_batch(batch.retry_batch_id, status="running")
    expired = attempts.get_attempt(bound.attempt_id)
    assert expired.lease_expires_at is not None

    timers = _capture_retry_timers(monkeypatch)
    store._retry_worker_batches.add(batch.retry_batch_id)
    assert store._schedule_retry_batch_recovery(
        state.experiment_id,
        batch.retry_batch_id,
        expired.lease_expires_at,
    )

    intervals = []
    for _ in range(9):
        active_timers = [timer for timer in timers if not timer.cancelled]
        assert len(active_timers) == 1
        assert set(store._retry_recovery_timers) == {batch.retry_batch_id}
        assert set(store._retry_recovery_deadlines) == {batch.retry_batch_id}
        intervals.append(active_timers[0].interval)
        active_timers[0].fire()

    assert intervals == sorted(intervals)
    assert len(set(intervals)) > 3
    assert intervals[-2:] == [5.0, 5.0]

    dispatched = []
    original_worker = store._run_retry_batch_worker

    def record_worker(*args, **kwargs):
        dispatched.append((args, kwargs))
        return original_worker(*args, **kwargs)

    monkeypatch.setattr(store, "_run_retry_batch_worker", record_worker)
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: thread.run(),
    )
    store._retry_worker_batches.discard(batch.retry_batch_id)
    active_timers = [timer for timer in timers if not timer.cancelled]
    assert len(active_timers) == 1
    active_timers[0].fire()

    recovered = attempts.get_attempt(bound.attempt_id)
    final_batch = attempts.get_retry_batch(batch.retry_batch_id)
    assert len(dispatched) == 1
    assert recovered.failure_category is FailureCategory.WORKER_LOST
    assert final_batch.members[0].status == "failed"
    assert final_batch.status == "failed"
    assert [timer for timer in timers if not timer.cancelled] == []
    assert batch.retry_batch_id not in store._retry_recovery_timers
    assert batch.retry_batch_id not in store._retry_recovery_deadlines
    assert batch.retry_batch_id not in store._retry_recovery_failures


def test_retry_guard_occupied_future_lease_keeps_exact_expiry(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    assert attempts.claim_attempt(
        bound.attempt_id,
        worker_id="future-guard-owner",
        lease_seconds=10,
        experiment_id=state.experiment_id,
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )
    attempts.bind_retry_batch_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        bound.attempt_id,
    )
    attempts.update_retry_batch(batch.retry_batch_id, status="running")
    future = attempts.get_attempt(bound.attempt_id)
    assert future.lease_expires_at is not None

    timers = _capture_retry_timers(monkeypatch)
    store._retry_worker_batches.add(batch.retry_batch_id)
    store._retry_recovery_failures[batch.retry_batch_id] = 4
    assert store._schedule_retry_batch_recovery(
        state.experiment_id,
        batch.retry_batch_id,
        None,
    )
    timers[0].fire()

    active_timers = [timer for timer in timers if not timer.cancelled]
    assert len(active_timers) == 1
    assert active_timers[0].daemon is True
    assert 9 < active_timers[0].interval < 11
    expected_deadline = future.lease_expires_at + timedelta(milliseconds=50)
    actual_deadline = store._retry_recovery_deadlines[batch.retry_batch_id]
    assert abs((actual_deadline - expected_deadline).total_seconds()) < 0.001
    assert batch.retry_batch_id not in store._retry_recovery_failures

    store._cancel_retry_batch_recovery(batch.retry_batch_id)
    assert [timer for timer in timers if not timer.cancelled] == []
    assert batch.retry_batch_id not in store._retry_recovery_timers
    assert batch.retry_batch_id not in store._retry_recovery_deadlines
    assert batch.retry_batch_id not in store._retry_recovery_failures


def test_retry_exit_scan_rearms_prebind_failure_with_no_attempt(
    monkeypatch,
) -> None:
    class FailingMemberAttempts(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.fail_member_completion = False

        def complete_retry_batch_cell(self, retry_batch_id, cell_id, **kwargs):
            if self.fail_member_completion:
                raise RuntimeError("transient pre-bind member persistence failure")
            return super().complete_retry_batch_cell(
                retry_batch_id,
                cell_id,
                **kwargs,
            )

    attempts = FailingMemberAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    timers = _capture_retry_timers(monkeypatch)
    attempts.fail_member_completion = True
    monkeypatch.setattr(
        store,
        "_retry_cell_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("pre-bind validation failure")
        ),
    )

    with pytest.raises(RuntimeError, match="pre-bind member persistence failure"):
        store._run_retry_batch(
            state.experiment_id,
            batch.retry_batch_id,
            {"api_key": "fresh-secret"},
        )

    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    active_timers = [timer for timer in timers if not timer.cancelled]
    assert member.status == "requested"
    assert member.attempt_id is None
    assert len(active_timers) == 1
    assert active_timers[0].daemon is True
    assert 0 < active_timers[0].interval < 1


def test_retry_exit_scan_prefers_short_lease_over_maximum_prompt_backoff(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    first = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=0,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    second = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=1,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(first.cell_id, second.cell_id),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    running = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        first.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    assert attempts.claim_attempt(
        running.attempt_id,
        worker_id="one-second-owner",
        lease_seconds=1,
        experiment_id=state.experiment_id,
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )
    attempts.bind_retry_batch_attempt(
        batch.retry_batch_id,
        first.cell_id,
        running.attempt_id,
    )
    attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        second.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    attempts.update_retry_batch(batch.retry_batch_id, status="running")
    store._retry_recovery_failures[batch.retry_batch_id] = 16
    timers = _capture_retry_timers(monkeypatch)

    store._ensure_retry_batch_recovery_after_worker_exit(
        state.experiment_id,
        batch.retry_batch_id,
    )

    active_timers = [timer for timer in timers if not timer.cancelled]
    assert len(active_timers) == 1
    assert active_timers[0].daemon is True
    assert 0 < active_timers[0].interval < 2


def test_retry_exit_scan_prefers_no_attempt_prompt_over_long_lease(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    first = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=0,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    second = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=1,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(first.cell_id, second.cell_id),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    running = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        first.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    assert attempts.claim_attempt(
        running.attempt_id,
        worker_id="long-lease-owner",
        lease_seconds=60,
        experiment_id=state.experiment_id,
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )
    attempts.bind_retry_batch_attempt(
        batch.retry_batch_id,
        first.cell_id,
        running.attempt_id,
    )
    attempts.update_retry_batch(batch.retry_batch_id, status="running")
    timers = _capture_retry_timers(monkeypatch)

    store._ensure_retry_batch_recovery_after_worker_exit(
        state.experiment_id,
        batch.retry_batch_id,
    )

    no_attempt_member = attempts.get_retry_batch(batch.retry_batch_id).members[1]
    active_timers = [timer for timer in timers if not timer.cancelled]
    assert no_attempt_member.status == "requested"
    assert no_attempt_member.attempt_id is None
    assert len(active_timers) == 1
    assert 0 < active_timers[0].interval < 1


def test_retry_exit_scans_dedupe_earliest_timer_and_terminal_cleanup(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    first = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=0,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    second = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=1,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(first.cell_id, second.cell_id),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    running = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        first.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    assert attempts.claim_attempt(
        running.attempt_id,
        worker_id="cleanup-owner",
        lease_seconds=60,
        experiment_id=state.experiment_id,
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )
    attempts.bind_retry_batch_attempt(
        batch.retry_batch_id,
        first.cell_id,
        running.attempt_id,
    )
    attempts.update_retry_batch(batch.retry_batch_id, status="running")
    timers = _capture_retry_timers(monkeypatch)

    for _ in range(3):
        store._ensure_retry_batch_recovery_after_worker_exit(
            state.experiment_id,
            batch.retry_batch_id,
        )

    active_timers = [timer for timer in timers if not timer.cancelled]
    assert len(active_timers) == 1
    assert 0 < active_timers[0].interval < 1

    attempts.complete_attempt(
        running.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.INTERNAL,
    )
    attempts.complete_retry_batch_cell(
        batch.retry_batch_id,
        first.cell_id,
        status="failed",
        error_category="internal",
    )
    attempts.complete_retry_batch_cell(
        batch.retry_batch_id,
        second.cell_id,
        status="failed",
        error_category="internal",
    )
    attempts.update_retry_batch(batch.retry_batch_id, status="failed")
    store._ensure_retry_batch_recovery_after_worker_exit(
        state.experiment_id,
        batch.retry_batch_id,
    )

    assert [timer for timer in timers if not timer.cancelled] == []
    assert batch.retry_batch_id not in store._retry_recovery_timers
    assert batch.retry_batch_id not in store._retry_recovery_deadlines
    assert batch.retry_batch_id not in store._retry_recovery_failures


def test_foreign_retry_lease_gets_one_bounded_expiry_recheck(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    assert attempts.claim_attempt(
        bound.attempt_id,
        worker_id="old-process-worker",
        lease_seconds=60,
        experiment_id=state.experiment_id,
        execution_generation=batch.execution_generation,
        execution_owner_id=batch.retry_batch_id,
    )
    attempts.bind_retry_batch_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        bound.attempt_id,
    )
    attempts.update_retry_batch(batch.retry_batch_id, status="running")
    store._retry_worker_batches.clear()

    timers = []

    class FakeTimer:
        def __init__(self, interval, function, args=None, kwargs=None) -> None:
            self.interval = interval
            self.function = function
            self.args = tuple(args or ())
            self.kwargs = dict(kwargs or {})
            self.daemon = False
            self.cancelled = False

        def start(self) -> None:
            timers.append(self)

        def cancel(self) -> None:
            self.cancelled = True

        def fire(self) -> None:
            if not self.cancelled:
                self.function(*self.args, **self.kwargs)

    monkeypatch.setattr("app.api.store.threading.Timer", FakeTimer)
    restarted = []
    monkeypatch.setattr(
        store,
        "_start_retry_batch_worker",
        lambda *args, **kwargs: restarted.append((args, kwargs)) or True,
    )
    dispatched: list[object] = []
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: dispatched.append(kwargs),
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
        recovered_after_restart=True,
    )
    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
        recovered_after_restart=True,
    )

    assert len(timers) == 1
    assert timers[0].daemon is True
    assert timers[0].interval > 0
    assert dispatched == []
    lease_expires_at = attempts.get_attempt(bound.attempt_id).lease_expires_at
    assert lease_expires_at is not None
    clock = [lease_expires_at + timedelta(milliseconds=1)]
    monkeypatch.setattr(
        "app.persistence.experiment_attempts._now",
        lambda: clock[0],
    )
    timers[0].fire()

    assert len(restarted) == 1
    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
        recovered_after_restart=True,
    )
    recovered = attempts.get_attempt(bound.attempt_id)
    final_batch = attempts.get_retry_batch(batch.retry_batch_id)
    assert dispatched == []
    assert recovered.failure_category is FailureCategory.WORKER_LOST
    assert final_batch.members[0].status == "failed"
    assert final_batch.status == "failed"


def test_retry_recovery_supervisor_keeps_the_earliest_batch_lease_deadline(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    first = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=0,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    second = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=1,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(first.cell_id, second.cell_id),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    for cell, lease_seconds in ((first, 60), (second, 10)):
        attempt = attempts.create_and_bind_retry_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            retry_kind="restart",
            spec_hash=config.spec_hash,
            execution_generation=batch.execution_generation,
            max_attempts=5,
            max_retry_attempts_per_experiment=1000,
        )
        assert attempts.claim_attempt(
            attempt.attempt_id,
            worker_id=f"owner-{cell.cell_id}",
            lease_seconds=lease_seconds,
            experiment_id=state.experiment_id,
            execution_generation=batch.execution_generation,
            execution_owner_id=batch.retry_batch_id,
        )
        attempts.bind_retry_batch_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            attempt.attempt_id,
        )
    store._retry_worker_batches.clear()
    timers = []

    class FakeTimer:
        def __init__(self, interval, function) -> None:
            self.interval = interval
            self.function = function
            self.daemon = False
            self.cancelled = False

        def start(self) -> None:
            timers.append(self)

        def cancel(self) -> None:
            self.cancelled = True

    monkeypatch.setattr("app.api.store.threading.Timer", FakeTimer)

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
        recovered_after_restart=True,
    )

    active_timers = [timer for timer in timers if not timer.cancelled]
    assert len(active_timers) == 1
    assert active_timers[0].daemon is True
    assert active_timers[0].interval < 20


@pytest.mark.parametrize(
    ("stage", "method_name"),
    [
        ("cell-context", "_retry_cell_context"),
        ("policy", "_validate_retry_cell_policy"),
        ("runtime", "_validate_indeterminate_reissue"),
        ("profile", "_resolve_retry_profile_credentials"),
        ("limit", "_pinned_retry_limit"),
    ],
)
def test_bound_retry_validation_failure_never_leaves_terminal_member_on_queued_attempt(
    monkeypatch,
    stage,
    method_name,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    if stage == "profile":
        vault = ProfileVault(InMemoryProfileVaultRepository())
        vault.create("profile-fault-injection")
        store.profile_vault = vault
    credentials = (
        {"api_key": "fresh-secret", "target_profile_id": "profile-for-fault"}
        if stage == "profile"
        else {"api_key": "fresh-secret"}
    )
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )

    original = getattr(store, method_name)

    def injected_failure(*args, **kwargs):
        if stage != "limit" or args[1] == "max_attempts_per_cell":
            raise ValueError(f"injected {stage} validation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, method_name, injected_failure)
    dispatched: list[object] = []
    monkeypatch.setattr(
        store,
        "_execute_single",
        lambda **kwargs: dispatched.append(kwargs),
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        credentials,
    )

    persisted_attempt = attempts.get_attempt(bound.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert dispatched == []
    assert persisted_attempt.execution_status is ExecutionStatus.FAILED
    assert persisted_attempt.result_disposition is ResultDisposition.INCONCLUSIVE
    assert attempts.get_cell(failed.cell_id).active_attempt_id is None
    assert member.status == "failed"
    assert member.attempt_id == persisted_attempt.attempt_id


def test_bound_retry_stop_terminalizes_attempt_before_skipping_member(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    config = state.plan.technique_configs[0]
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    state.stop_requested = True

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    cancelled = attempts.get_attempt(bound.attempt_id)
    member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert cancelled.execution_status is ExecutionStatus.CANCELLED
    assert cancelled.failure_category is FailureCategory.CANCELLED
    assert member.status == "skipped"
    assert member.attempt_id == cancelled.attempt_id


def test_exact_execution_recovery_clears_expired_lease_before_retry_eligibility(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    planned = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    expired = attempts.create_attempt(
        planned.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )
    assert attempts.claim_attempt(
        expired.attempt_id,
        worker_id="expired-worker",
        lease_seconds=0,
        experiment_id=state.experiment_id,
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )
    store._recover_interrupted_execution_attempts(
        state,
        generation=state._execution_generation,
        owner_id=state._execution_owner_id,
    )
    _finish_planned_state(state)
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=state._execution_generation,
        owner_id=state._execution_owner_id,
        status="completed",
    )

    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(planned.cell_id,),
    )

    assert batch.cell_ids == (planned.cell_id,)
    assert attempts.get_attempt(expired.attempt_id).failure_category is FailureCategory.WORKER_LOST
    assert attempts.get_cell(planned.cell_id).active_attempt_id is None


def test_stale_retry_eligibility_cannot_recover_newer_owner_expired_attempt(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    assert state.plan is not None
    planned = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    _finish_planned_state(state)
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=state._execution_generation,
        owner_id=state._execution_owner_id,
        status="completed",
    )
    newer_owner = "newer-owner-after-stale-retry-read"
    newer_generation = store.persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id=newer_owner,
        expected_generation=state._execution_generation,
        active_status="running",
    )
    newer_attempt = attempts.create_attempt(
        planned.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
        execution_generation=newer_generation,
        execution_owner_id=newer_owner,
    )
    assert attempts.claim_attempt(
        newer_attempt.attempt_id,
        worker_id="newer-expired-worker",
        lease_seconds=0,
        experiment_id=state.experiment_id,
        execution_generation=newer_generation,
        execution_owner_id=newer_owner,
    )

    with pytest.raises(ValueError, match="active attempt"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            cell_ids=(planned.cell_id,),
        )

    preserved = attempts.get_attempt(newer_attempt.attempt_id)
    assert preserved.execution_status is ExecutionStatus.RUNNING
    assert preserved.failure_category is None
    assert attempts.get_cell(planned.cell_id).active_attempt_id == newer_attempt.attempt_id


def test_retry_budget_is_one_plan_wide_value_independent_of_arm_order(monkeypatch) -> None:
    conflicting = [
        {
            "technique_key": "probe",
            "parameters": {"threshold": 4},
            "limits": {"max_retry_attempts_per_experiment": 2},
        },
        {
            "technique_key": "probe",
            "parameters": {"threshold": 6},
            "limits": {"max_retry_attempts_per_experiment": 3},
        },
    ]
    for arms in (conflicting, list(reversed(conflicting))):
        store, attempts, state = _attempt_reporting_store(
            monkeypatch,
            technique_configs=arms,
        )
        _finish_planned_state(state)
        with pytest.raises(ValueError, match="experiment-wide retry budget.*revision"):
            store.create_retry_batch(state.experiment_id, action="restart")
        assert attempts.list_retry_batches(state.experiment_id) == ()

    store, _, state = _attempt_reporting_store(
        monkeypatch,
        technique_configs=[
            {**arm, "limits": {"max_retry_attempts_per_experiment": 4}}
            for arm in conflicting
        ],
    )
    _finish_planned_state(state)
    assert store.create_retry_batch(state.experiment_id, action="restart").cell_ids


def test_matching_profile_override_refreshes_secret_but_keeps_pinned_behavior(
    monkeypatch,
) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)
    vault = ProfileVault(InMemoryProfileVaultRepository())
    vault.create("retry-profile-refresh")
    store.profile_vault = vault
    config = state.plan.technique_configs[0]
    pinned_role = config.model_roles["target"]
    pinned_endpoint = config.runtime_config["api_base_url"]
    vault.save_profile(
        {
            "profile_id": "rotated-target-profile",
            "name": "Rotated target",
            "provider": pinned_role["provider"],
            "base_url": pinned_endpoint,
            "model": pinned_role["model"],
            "scope": "target",
            "api_key": "rotated-target-secret-a",
        }
    )
    failed = _complete_reporting_attempt(
        store.experiment_attempt_persistence,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    observed: dict[str, object] = {}

    def execute(**kwargs):
        observed.update(kwargs["target_config"])
        return RunSummary(
            run_id="run-profile-refresh",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=20,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    monkeypatch.setattr(store, "_execute_single", execute)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"profile_id": "rotated-target-profile"},
    )
    vault.save_profile(
        {
            "profile_id": "rotated-target-profile",
            "name": "Rotated target",
            "provider": pinned_role["provider"],
            "base_url": pinned_endpoint,
            "model": pinned_role["model"],
            "scope": "target",
            "api_key": "rotated-target-secret-b",
        }
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"profile_id": "rotated-target-profile"},
    )

    assert observed["api_key"] == "rotated-target-secret-b"
    assert observed["provider"] == pinned_role["provider"]
    assert observed["model"] == pinned_role["model"]
    assert observed["api_base_url"] == pinned_endpoint


def test_locked_vault_defers_profile_backed_recovery_until_unlock(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    vault = ProfileVault(InMemoryProfileVaultRepository())
    vault.create("retry-recovery-vault")
    store.profile_vault = vault
    config = state.plan.technique_configs[0]
    pinned_role = config.model_roles["target"]
    vault.save_profile(
        {
            "profile_id": "recovery-target-profile",
            "name": "Recovery target",
            "provider": pinned_role["provider"],
            "base_url": config.runtime_config["api_base_url"],
            "model": pinned_role["model"],
            "scope": "target",
            "api_key": "recovered-profile-secret",
        }
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    started: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: started.append(self),
    )
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        request_id="profile-recovery-request",
        credential_overrides={"target_profile_id": "recovery-target-profile"},
    )
    bound = attempts.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        failed.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
        execution_generation=batch.execution_generation,
        max_attempts=5,
        max_retry_attempts_per_experiment=1000,
    )
    store._retry_worker_batches.clear()
    vault.lock()
    dispatched: list[object] = []

    def execute(**kwargs):
        dispatched.append(kwargs)
        return RunSummary(
            run_id="run-after-vault-unlock",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=10,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    monkeypatch.setattr(store, "_execute_single", execute)
    store._recover_retry_batches_for_experiment(state.experiment_id)
    recovered_thread = started[-1]
    recovered_thread._target(*recovered_thread._args)

    deferred = attempts.get_attempt(bound.attempt_id)
    deferred_member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert dispatched == []
    assert deferred.execution_status is ExecutionStatus.QUEUED
    assert deferred_member.status == "queued"
    assert attempts.get_retry_batch(batch.retry_batch_id).status in {
        "requested",
        "running",
    }

    store.unlock_vault("retry-recovery-vault")
    unlocked_thread = started[-1]
    assert unlocked_thread is not recovered_thread
    unlocked_thread._target(*unlocked_thread._args)

    completed = attempts.get_attempt(bound.attempt_id)
    final_member = attempts.get_retry_batch(batch.retry_batch_id).members[0]
    assert len(dispatched) == 1
    assert dispatched[0]["target_config"]["api_key"] == "recovered-profile-secret"
    assert completed.execution_status is ExecutionStatus.COMPLETED
    assert final_member.status == "completed"


@pytest.mark.parametrize(
    "retry_entrypoint",
    ["explicit-batch", "automatic-batch", "restart-cell"],
)
def test_retry_profile_identity_validation_only_checks_selected_arm(
    monkeypatch,
    retry_entrypoint,
) -> None:
    store, attempts, state = _attempt_reporting_store(
        monkeypatch,
        technique_configs=[
            {
                "technique_key": "probe",
                "parameters": {"threshold": 4},
                "model_roles": {
                    "target": {
                        "profile_id": "arm-a",
                        "provider": "openai",
                        "model": "arm-a-model",
                    }
                },
            },
            {
                "technique_key": "probe",
                "parameters": {"threshold": 6},
                "model_roles": {
                    "target": {
                        "profile_id": "arm-b",
                        "provider": "openai",
                        "model": "arm-b-model",
                    }
                },
            },
        ],
    )
    selected_config = state.plan.technique_configs[0]
    selected_cell_index = next(
        index
        for index, cell in enumerate(state.plan.cells)
        if cell.technique_config_id == selected_config.technique_config_id
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=selected_cell_index,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    for index, cell in enumerate(state.plan.cells):
        if cell.cell_id == failed.cell_id:
            continue
        _complete_reporting_attempt(
            attempts,
            state,
            cell_index=index,
            execution_status=ExecutionStatus.COMPLETED,
            result_disposition=ResultDisposition.TESTED,
            score=0,
        )
    _finish_planned_state(state)
    vault = ProfileVault(InMemoryProfileVaultRepository())
    vault.create("selected-arm-profile")
    store.profile_vault = vault
    selected_role = selected_config.model_roles["target"]
    vault.save_profile(
        {
            "profile_id": "selected-arm-refresh",
            "name": "Selected arm refresh",
            "provider": selected_role["provider"],
            "base_url": selected_config.runtime_config["api_base_url"],
            "model": selected_role["model"],
            "scope": "target",
            "api_key": "selected-arm-secret",
        }
    )
    credentials = {"target_profile_id": "selected-arm-refresh"}

    if retry_entrypoint == "restart-cell":
        attempt = store.restart_cell(
            state.experiment_id,
            failed.cell_id,
            credential_overrides=credentials,
        )
        assert attempt.cell_id == failed.cell_id
    else:
        batch = store.create_retry_batch(
            state.experiment_id,
            action="restart",
            cell_ids=(failed.cell_id,) if retry_entrypoint == "explicit-batch" else None,
            credential_overrides=credentials,
        )
        assert batch.cell_ids == (failed.cell_id,)


@pytest.mark.parametrize("drift", ["provider", "model", "endpoint"])
def test_profile_override_behavior_mismatch_fails_before_dispatch(monkeypatch, drift) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    vault = ProfileVault(InMemoryProfileVaultRepository())
    vault.create("retry-profile-mismatch")
    store.profile_vault = vault
    config = state.plan.technique_configs[0]
    pinned_role = config.model_roles["target"]
    profile = {
        "profile_id": "mismatched-target-profile",
        "name": "Mismatch",
        "provider": pinned_role["provider"],
        "base_url": config.runtime_config["api_base_url"],
        "model": pinned_role["model"],
        "scope": "target",
        "api_key": "rotated-target-secret",
    }
    profile[
        "base_url" if drift == "endpoint" else drift
    ] = f"different-{drift}"
    vault.save_profile(profile)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    dispatched: list[object] = []
    monkeypatch.setattr(store, "_execute_single", lambda **kwargs: dispatched.append(kwargs))

    with pytest.raises(ValueError, match="experiment revision"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            cell_ids=(failed.cell_id,),
            credential_overrides={"target_profile_id": "mismatched-target-profile"},
        )

    assert dispatched == []
    assert attempts.list_retry_batches(state.experiment_id) == ()


def test_budget_exhausted_cell_is_automatic_exclusion_and_explicit_revision(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    exhausted = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.BUDGET_EXHAUSTED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.BUDGET,
    )
    _finish_planned_state(state)

    automatic = store.create_retry_batch(state.experiment_id, action="restart")
    assert exhausted.cell_id not in automatic.cell_ids
    _finish_planned_state(state)
    with pytest.raises(ValueError, match="experiment revision"):
        store.create_retry_batch(
            state.experiment_id,
            action="restart",
            cell_ids=(exhausted.cell_id,),
        )


def test_restart_cell_enforces_cell_and_experiment_retry_limits(monkeypatch) -> None:
    from app.api.store import RetryLimitExceeded

    store, attempts, state = _attempt_reporting_store(
        monkeypatch,
        technique_configs=[
            {
                "technique_key": "probe",
                "limits": {
                    "max_attempts_per_cell": 1,
                    "max_retry_attempts_per_experiment": 1,
                },
            }
        ],
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    durable_before = store.persistence.load_all_experiments()[0]

    with pytest.raises(RetryLimitExceeded, match="max_attempts_per_cell"):
        store.restart_cell(state.experiment_id, failed.cell_id)
    assert len(attempts.list_attempts(failed.cell_id)) == 1
    assert attempts.list_retry_batches(state.experiment_id) == ()
    durable_after = store.persistence.load_all_experiments()[0]
    assert (
        durable_after["status"],
        durable_after["execution_generation"],
        durable_after["execution_owner_id"],
    ) == (
        durable_before["status"],
        durable_before["execution_generation"],
        durable_before["execution_owner_id"],
    )

    store2, attempts2, state2 = _attempt_reporting_store(
        monkeypatch,
        technique_configs=[
            {
                "technique_key": "probe",
                "limits": {
                    "max_attempts_per_cell": 5,
                    "max_retry_attempts_per_experiment": 1,
                },
            }
        ],
    )
    failed2 = _complete_reporting_attempt(
        attempts2,
        state2,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    other = state2.plan.cells[1]
    config = state2.plan.technique_configs[0]
    consumed = attempts2.create_attempt(
        other.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
    )
    attempts2.complete_attempt(
        consumed.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state2)
    durable_before = store2.persistence.load_all_experiments()[0]

    with pytest.raises(RetryLimitExceeded, match="max_retry_attempts_per_experiment"):
        store2.restart_cell(state2.experiment_id, failed2.cell_id)
    assert len(attempts2.list_attempts(failed2.cell_id)) == 1
    assert attempts2.list_retry_batches(state2.experiment_id) == ()
    durable_after = store2.persistence.load_all_experiments()[0]
    assert (
        durable_after["status"],
        durable_after["execution_generation"],
        durable_after["execution_owner_id"],
    ) == (
        durable_before["status"],
        durable_before["execution_generation"],
        durable_before["execution_owner_id"],
    )


def test_restart_cell_limit_race_terminalizes_the_acquired_batch(monkeypatch) -> None:
    from app.api.store import RetryLimitExceeded

    class LimitRaceAttemptPersistence(InMemoryExperimentAttemptPersistence):
        def create_and_bind_retry_attempt(self, *args, **kwargs):
            raise ValueError("max_attempts_per_cell retry limit exceeded")

    attempts = LimitRaceAttemptPersistence()
    store, attempts, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)

    with pytest.raises(RetryLimitExceeded, match="max_attempts_per_cell"):
        store.restart_cell(state.experiment_id, failed.cell_id)

    batch = attempts.list_retry_batches(state.experiment_id)[0]
    assert batch.status == "failed"
    assert batch.members[0].status == "failed"
    assert batch.members[0].error_category == "limit"
    durable = store.persistence.load_all_experiments()[0]
    assert durable["status"] == "failed"
    assert durable["execution_generation"] == batch.execution_generation
    assert durable["execution_owner_id"] == batch.retry_batch_id


def test_restart_cell_and_compatibility_retry_bind_positive_execution_generation(
    monkeypatch,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    workers: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        store,
        "_start_retry_batch_worker",
        lambda *args, **kwargs: workers.append((args, kwargs)) or True,
    )

    restarted = store.restart_cell(state.experiment_id, failed.cell_id)

    batches = attempts.list_retry_batches(state.experiment_id)
    assert len(batches) == 1
    assert restarted.execution_generation == batches[0].execution_generation
    assert restarted.execution_generation > 0
    assert batches[0].members[0].attempt_id == restarted.attempt_id
    assert len(workers) == 1

    # The callable compatibility helper must enter through another owned batch,
    # never through its historical direct provider loop.
    attempts.complete_attempt(
        restarted.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.INTERNAL,
    )
    attempts.complete_retry_batch_cell(
        batches[0].retry_batch_id,
        failed.cell_id,
        status="failed",
        error_category="internal",
    )
    attempts.update_retry_batch(batches[0].retry_batch_id, status="failed")
    state.status = "failed"
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=batches[0].execution_generation,
        owner_id=batches[0].retry_batch_id,
        status="failed",
    )
    config = state.plan.technique_configs[0]
    planned = state.plan.cells[0]
    store._run_experiment_retry(
        state.experiment_id,
        [
            (
                config,
                {
                    "id": planned.prompt_id,
                    "text": planned.prompt_text,
                    "category": planned.prompt_category,
                    "gold_answer": planned.gold_answer,
                },
                planned.repeat_index,
            )
        ],
        {},
        "",
    )

    compatibility_batch = attempts.list_retry_batches(state.experiment_id)[-1]
    compatibility_attempt = attempts.get_attempt(
        compatibility_batch.members[0].attempt_id
    )
    assert compatibility_batch.execution_generation is not None
    assert compatibility_batch.execution_generation > batches[0].execution_generation
    assert compatibility_attempt.execution_generation == (
        compatibility_batch.execution_generation
    )


@pytest.mark.parametrize("retry_indeterminate", (False, True))
@pytest.mark.parametrize("newer_attempt", (False, True))
def test_checkpoint_resume_requires_fresh_restart_for_missing_events(monkeypatch, retry_indeterminate, newer_attempt) -> None:
    store, attempts, state = _attempt_reporting_store(
        monkeypatch, technique_configs=[{
            "technique_key": "probe", "limits": {"retry_indeterminate": retry_indeterminate},
        }],
    )
    runtime = NoopTechniqueRuntimePersistence()
    store.technique_runtime_persistence = runtime
    failed = _complete_reporting_attempt(
        attempts, state, execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    call = ModelCallRecord(
        model_call_id="call-before-crash",
        correlation=CorrelationIds(experiment_id=state.experiment_id,
            cell_id=failed.cell_id, run_id="run-before-crash", attempt_id=failed.attempt_id),
        role="target", operation_id="branch:target", operation_attempt=1,
        request={}, request_hash="hash-before-crash",
    )
    runtime.record_model_call_requested(call)
    runtime.complete_model_call(call.model_call_id, response="saved", usage={},
        provider_response_id=None, gateway_call_id=None, gateway_attempts={})
    if newer_attempt:
        _complete_reporting_attempt(attempts, state,
            execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED, score=0)
    selected_before = attempts.get_cell(failed.cell_id).selected_attempt_id
    _finish_planned_state(state)
    dispatched = []
    monkeypatch.setattr(store, "_execute_single", lambda **kwargs: dispatched.append(kwargs))
    runtime.save_checkpoint(CheckpointRecord(
        checkpoint_id="checkpoint-after-crash", correlation=call.correlation,
        spec_hash=state.plan.technique_configs[0].spec_hash, state_schema_version=1, payload={},
    ))
    with pytest.raises(ValueError, match="event.*restart.*case"):
        store.resume_cell(state.experiment_id, failed.cell_id)
    assert dispatched == []
    assert len(attempts.list_attempts(failed.cell_id)) == (2 if newer_attempt else 1)
    assert attempts.get_cell(failed.cell_id).selected_attempt_id == selected_before
    assert attempts.get_attempt(failed.attempt_id).score is None
    if newer_attempt:
        assert attempts.get_attempt(selected_before).score == 0


@pytest.fixture(params=("memory", "postgres"))
def crash_retry_repositories(request):
    if request.param == "memory":
        yield NoopRunPersistence(), InMemoryExperimentAttemptPersistence(), NoopTechniqueRuntimePersistence()
        return
    dsn = os.environ.get("TASK14_DATABASE_URL")
    if not dsn:
        pytest.skip("requires disposable Task 14 PostgreSQL")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from uuid import uuid4
    from app.persistence.technique_runtime import PostgreSQLTechniqueRuntimePersistence

    schema = f"case_retry_{uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    isolated = make_conninfo(dsn, options=f"-c search_path={schema}")
    try:
        with psycopg.connect(isolated, autocommit=True) as connection:
            for path in sorted((Path(__file__).resolve().parents[2] / "database" / "init").glob("*.sql")):
                connection.execute(path.read_text())
        yield PostgreSQLRunPersistence(isolated), PostgreSQLExperimentAttemptPersistence(isolated), PostgreSQLTechniqueRuntimePersistence(isolated)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.parametrize("call_status", ("requested", "completed", "failed"))
@pytest.mark.parametrize("route", ("cell", "batch"))
def test_crash_retry_only_reissues_case_40_and_preserves_39_completed_cases(monkeypatch, call_status, route, crash_retry_repositories):
    runs, attempts, runtime = crash_retry_repositories
    store, attempts, state = _attempt_reporting_store(
        monkeypatch, prompts=[{"text": f"case {i + 1}"} for i in range(40)],
        run_persistence=runs, attempt_persistence=attempts, runtime_persistence=runtime,
    )
    completed = []
    for i in range(39):
        saved = _complete_reporting_attempt(attempts, state, cell_index=i,
            execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED,
            score=i, run_id=f"saved-run-{i}")
        completed.append((attempts.get_cell(saved.cell_id), attempts.list_attempts(saved.cell_id)))
        row = _flattened_planned_result(state, cell_index=i, run_id=saved.run_id, score=i)
        runs.save_experiment_cell(experiment_id=state.experiment_id, cell={**row, "attempt_id": saved.attempt_id})
        state.cell_results.append(row)
    rows_before = json.dumps(state.cell_results, sort_keys=True)
    interrupted = _complete_reporting_attempt(attempts, state, cell_index=39,
        execution_status=ExecutionStatus.FAILED, result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.WORKER_LOST, run_id="crashed-run")
    row = _flattened_planned_result(state, cell_index=39, run_id="crashed-run", status="error", score=None)
    runs.save_experiment_cell(experiment_id=state.experiment_id,
        cell={**row, "attempt_id": interrupted.attempt_id, "asr": None})
    state.cell_results.append(row)
    call = ModelCallRecord(model_call_id="call-before-crash",
        correlation=CorrelationIds(experiment_id=state.experiment_id, cell_id=interrupted.cell_id,
            run_id="crashed-run", attempt_id=interrupted.attempt_id),
        role="target", operation_id="target:40", operation_attempt=1, request={}, request_hash="test")
    runtime.record_model_call_requested(call)
    if call_status == "completed":
        runtime.complete_model_call(call.model_call_id, response="partial case output", usage={},
            provider_response_id=None, gateway_call_id=None, gateway_attempts={})
    elif call_status == "failed":
        runtime.fail_model_call(call.model_call_id, category="timeout", error="interrupted", indeterminate=True)
    def snapshot_call():
        if isinstance(runtime, NoopTechniqueRuntimePersistence):
            return dict(runtime._model_calls[call.model_call_id])
        with runtime._connect() as connection:
            return connection.execute("SELECT to_jsonb(call)::text FROM model_calls AS call WHERE model_call_id = %s",
                (call.model_call_id,)).fetchone()

    old_call = snapshot_call()
    _finish_planned_state(state)
    runs.update_experiment_status(state.experiment_id, "completed")
    dispatched = []

    def execute(**kwargs):
        dispatched.append(kwargs["cell_id"])
        return _completed_retry_summary(**kwargs)

    monkeypatch.setattr(store, "_execute_single", execute)
    if route == "cell":
        retry = store.restart_cell(state.experiment_id, interrupted.cell_id)
        assert retry.parent_attempt_id == interrupted.attempt_id
        batch = attempts.list_retry_batches(state.experiment_id)[0]
    else:
        batch = store.create_retry_batch(state.experiment_id, action="restart")
    assert batch.cell_ids == (interrupted.cell_id,)
    store._run_retry_batch(state.experiment_id, batch.retry_batch_id, {})
    assert attempts.get_retry_batch(batch.retry_batch_id).status == "completed"
    assert dispatched == [interrupted.cell_id]
    assert json.dumps(state.cell_results[:39], sort_keys=True) == rows_before
    for cell, saved_attempts in completed:
        assert attempts.get_cell(cell.cell_id) == cell
        assert attempts.list_attempts(cell.cell_id) == saved_attempts
    history = attempts.list_attempts(interrupted.cell_id)
    assert len(history) == 2
    assert history[0] == interrupted
    assert history[0].score is None
    assert history[1].execution_status is ExecutionStatus.COMPLETED
    assert history[1].result_disposition is ResultDisposition.TESTED
    assert attempts.get_cell(interrupted.cell_id).selected_attempt_id == history[1].attempt_id
    assert snapshot_call() == old_call


def test_restart_blocks_indeterminate_call_unless_pinned_and_bounded(monkeypatch) -> None:
    from app.api.store import IndeterminateCallError
    from app.techniques.core.events import EventJournal, TechniqueEventIdentity

    def persist_failed_lifecycle(runtime, saved_call):
        journal = EventJournal(
            correlation=saved_call.correlation, persistence=runtime,
            technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
        )
        for event_type in ("model.requested", "model.failed"):
            journal.emit(event_type, operation_id=saved_call.operation_id,
                attempt=saved_call.operation_attempt,
                payload={"model_call_id": saved_call.model_call_id, "role": saved_call.role})

    runtime = NoopTechniqueRuntimePersistence()
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    store.technique_runtime_persistence = runtime
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    call = ModelCallRecord(
        model_call_id="call-unknown",
        correlation=CorrelationIds(
            experiment_id=state.experiment_id,
            cell_id=failed.cell_id,
            run_id="run-unknown",
            attempt_id=failed.attempt_id,
        ),
        role="target",
        operation_id="target-call",
        operation_attempt=1,
        request={"messages": []},
        request_hash="hash-unknown",
    )
    runtime.record_model_call_requested(call)
    runtime.fail_model_call(
        call.model_call_id,
        category="provider",
        error="connection closed",
        indeterminate=True,
    )
    persist_failed_lifecycle(runtime, call)
    _finish_planned_state(state)

    with pytest.raises(IndeterminateCallError, match="manual reconciliation"):
        store.restart_cell(state.experiment_id, failed.cell_id)
    assert len(attempts.list_attempts(failed.cell_id)) == 1

    allowed_store, allowed_attempts, allowed_state = _attempt_reporting_store(
        monkeypatch,
        technique_configs=[
            {
                "technique_key": "probe",
                "limits": {
                    "retry_indeterminate": True,
                    "max_attempts_per_cell": 2,
                    "max_retry_attempts_per_experiment": 1,
                },
            }
        ],
    )
    allowed_runtime = NoopTechniqueRuntimePersistence()
    allowed_store.technique_runtime_persistence = allowed_runtime
    allowed_failed = _complete_reporting_attempt(
        allowed_attempts,
        allowed_state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    allowed_call = replace(
        call,
        model_call_id="call-allowed",
        correlation=CorrelationIds(
            experiment_id=allowed_state.experiment_id,
            cell_id=allowed_failed.cell_id,
            run_id="run-allowed",
            attempt_id=allowed_failed.attempt_id,
        ),
    )
    allowed_runtime.record_model_call_requested(allowed_call)
    allowed_runtime.fail_model_call(
        allowed_call.model_call_id,
        category="provider",
        error="connection closed",
        indeterminate=True,
    )
    persist_failed_lifecycle(allowed_runtime, allowed_call)
    _finish_planned_state(allowed_state)

    restarted = allowed_store.restart_cell(
        allowed_state.experiment_id,
        allowed_failed.cell_id,
    )
    assert restarted.parent_attempt_id == allowed_failed.attempt_id


def test_resume_validates_checkpoint_before_rejecting_unsupported_implementation(monkeypatch) -> None:
    from app.api.store import ResumeCheckpointError, UnsupportedResumeError

    store, attempts, state = _attempt_reporting_store(monkeypatch)
    runtime = NoopTechniqueRuntimePersistence()
    store.technique_runtime_persistence = runtime
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)

    with pytest.raises(ResumeCheckpointError, match="checkpoint"):
        store.resume_cell(state.experiment_id, failed.cell_id)
    assert len(attempts.list_attempts(failed.cell_id)) == 1

    config = state.plan.technique_configs[0]
    runtime.save_checkpoint(
        CheckpointRecord(
            checkpoint_id="checkpoint-bad-spec",
            correlation=CorrelationIds(
                experiment_id=state.experiment_id,
                cell_id=failed.cell_id,
                run_id="run-checkpoint",
                attempt_id=failed.attempt_id,
            ),
            spec_hash="different-spec",
            state_schema_version=1,
            payload={"position": "target"},
        )
    )
    with pytest.raises(ResumeCheckpointError, match="specification"):
        store.resume_cell(state.experiment_id, failed.cell_id)
    assert len(attempts.list_attempts(failed.cell_id)) == 1

    runtime.save_checkpoint(
        CheckpointRecord(
            checkpoint_id="checkpoint-bad-schema",
            correlation=CorrelationIds(
                experiment_id=state.experiment_id,
                cell_id=failed.cell_id,
                run_id="run-checkpoint",
                attempt_id=failed.attempt_id,
            ),
            spec_hash=config.spec_hash,
            state_schema_version=99,
            payload={"position": "target"},
        )
    )
    with pytest.raises(ResumeCheckpointError, match="state schema"):
        store.resume_cell(state.experiment_id, failed.cell_id)
    assert len(attempts.list_attempts(failed.cell_id)) == 1

    runtime.save_checkpoint(
        CheckpointRecord(
            checkpoint_id="checkpoint-compatible",
            correlation=CorrelationIds(
                experiment_id=state.experiment_id,
                cell_id=failed.cell_id,
                run_id="run-checkpoint",
                attempt_id=failed.attempt_id,
            ),
            spec_hash=config.spec_hash,
            state_schema_version=1,
            payload={"position": "target"},
        )
    )
    with pytest.raises(UnsupportedResumeError, match="cannot consume checkpoints"):
        store.resume_cell(state.experiment_id, failed.cell_id)
    assert len(attempts.list_attempts(failed.cell_id)) == 1


def test_retry_worker_uses_one_explicit_attempt_and_preserves_prior_flattened_result(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        run_id="run-failed",
    )
    prior = _flattened_planned_result(
        state,
        run_id="run-failed",
        status="error",
        score=0,
    )
    prior["error"] = "provider failed"
    state.cell_results.append(prior)
    _finish_planned_state(state)
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    observed: dict[str, object] = {}

    def execute(**kwargs):
        observed.update(kwargs)
        return (
            RunSummary(
                run_id="run-retry",
                technique_key="probe",
                target_kind="chatbot",
                status="completed",
                score=10,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
            ),
            object(),
        )

    monkeypatch.setattr(store, "_execute_single", execute)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        completed_policy="replace_on_valid_completion",
        credential_overrides={"api_key": "fresh-secret"},
    )

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    current_batch = store.get_retry_batch(state.experiment_id, batch.retry_batch_id)
    member = current_batch.members[0]
    assert current_batch.status == "completed"
    assert member.status == "completed"
    assert member.attempt_id == observed["attempt_id"]
    assert observed["technique_config"] is state.plan.technique_configs[0]
    assert len(attempts.list_attempts(failed.cell_id)) == 2
    assert attempts.get_cell(failed.cell_id).selected_attempt_id == member.attempt_id
    assert prior in state.cell_results
    assert state.cell_results[-1]["attempt_id"] == member.attempt_id
    assert state.cell_results[-1]["selected_attempt_id"] == member.attempt_id


def test_retry_worker_projects_inconclusive_zero_as_null_and_preserves_tested_zero(
    monkeypatch,
    caplog,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    first_failed = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=0,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    second_failed = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=1,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    calls: list[str] = []

    def execute(**kwargs):
        cell_id = str(kwargs["cell_id"])
        calls.append(cell_id)
        inconclusive = cell_id == first_failed.cell_id
        return (
            RunSummary(
                run_id=f"run-retry-{len(calls)}",
                technique_key=str(kwargs["technique_key"]),
                target_kind="chatbot",
                status="error" if inconclusive else "completed",
                score=0,
                asr=0,
                matched_detections=0,
                detections=(),
                highest_severity="info",
                conversation=[],
                error="judge output was not parseable" if inconclusive else None,
                execution_status=(
                    ExecutionStatus.FAILED
                    if inconclusive
                    else ExecutionStatus.COMPLETED
                ),
                result_disposition=(
                    ResultDisposition.INCONCLUSIVE
                    if inconclusive
                    else ResultDisposition.TESTED
                ),
                failure_category=(
                    FailureCategory.JUDGE_PARSE if inconclusive else None
                ),
            ),
            object(),
        )

    monkeypatch.setattr(store, "_execute_single", execute)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(first_failed.cell_id, second_failed.cell_id),
        credential_overrides={"api_key": "fresh-secret"},
    )
    caplog.set_level("ERROR", logger="app.api.store")

    store._run_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
        {"api_key": "fresh-secret"},
    )

    current_batch = store.get_retry_batch(
        state.experiment_id,
        batch.retry_batch_id,
    )
    assert calls == [first_failed.cell_id, second_failed.cell_id]
    assert current_batch.status == "partial"
    assert [member.status for member in current_batch.members] == [
        "failed",
        "completed",
    ]
    retry_results = state.cell_results[-2:]
    assert retry_results[0]["status"] == "error"
    assert retry_results[0]["execution_status"] == ExecutionStatus.FAILED.value
    assert (
        retry_results[0]["result_disposition"]
        == ResultDisposition.INCONCLUSIVE.value
    )
    assert retry_results[0]["failure_category"] == FailureCategory.JUDGE_PARSE.value
    assert retry_results[0]["score"] is None
    assert retry_results[0]["asr"] is None
    assert retry_results[1]["status"] == "completed"
    assert retry_results[1]["execution_status"] == ExecutionStatus.COMPLETED.value
    assert retry_results[1]["result_disposition"] == ResultDisposition.TESTED.value
    assert retry_results[1]["failure_category"] is None
    assert retry_results[1]["score"] == 0
    assert retry_results[1]["asr"] == 0
    assert not caplog.records


def test_retry_worker_heartbeats_for_blocked_provider_lifetime_and_stops(monkeypatch) -> None:
    real_thread_start = threading.Thread.start

    class HeartbeatAttempts(InMemoryExperimentAttemptPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_count = 0
            self.heartbeat_seen = threading.Event()

        def heartbeat_attempt(
            self,
            attempt_id,
            *,
            worker_id,
            lease_seconds,
            **ownership,
        ):
            self.heartbeat_count += 1
            self.heartbeat_seen.set()
            return super().heartbeat_attempt(
                attempt_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                **ownership,
            )

    attempts = HeartbeatAttempts()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    failed = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    _finish_planned_state(state)
    batch = store.create_retry_batch(
        state.experiment_id,
        action="restart",
        cell_ids=(failed.cell_id,),
        credential_overrides={"api_key": "fresh-secret"},
    )
    release_provider = threading.Event()

    def execute(**kwargs):
        assert release_provider.wait(timeout=5)
        return RunSummary(
            run_id="run-heartbeat",
            technique_key=kwargs["technique_key"],
            target_kind="chatbot",
            status="completed",
            score=10,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    monkeypatch.setattr(store, "_execute_single", execute)
    monkeypatch.setattr("app.api.store._RETRY_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(threading.Thread, "start", real_thread_start)
    worker = threading.Thread(
        target=store._run_retry_batch,
        args=(state.experiment_id, batch.retry_batch_id, {"api_key": "fresh-secret"}),
    )
    worker.start()
    try:
        assert attempts.heartbeat_seen.wait(timeout=2)
    finally:
        release_provider.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    observed_count = attempts.heartbeat_count
    assert observed_count >= 1
    threading.Event().wait(0.05)
    assert attempts.heartbeat_count == observed_count


def test_late_selected_attempt_is_operational_evidence_not_efficacy(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    planned = state.plan.cells[0]
    config = state.plan.technique_configs[0]
    obsolete = attempts.create_attempt(
        planned.cell_id,
        retry_kind="initial",
        spec_hash=config.spec_hash,
    )
    assert attempts.claim_attempt(
        obsolete.attempt_id,
        worker_id="stale-worker",
        lease_seconds=0,
    )
    replacement = attempts.create_attempt(
        planned.cell_id,
        retry_kind="restart",
        spec_hash=config.spec_hash,
    )
    assert attempts.claim_attempt(
        replacement.attempt_id,
        worker_id="current-worker",
        lease_seconds=60,
    )
    late = attempts.complete_attempt(
        obsolete.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=100,
    )

    report = store._build_experiment_report(state.cell_results, state=state)

    assert late.late_completion is True
    assert report["tested"] == 0
    assert report["operational"]["total_attempts"] == 2
    row = next(item for item in report["canonical_rows"] if item["cell_id"] == planned.cell_id)
    assert row["late_completion"] is True


def test_legacy_retry_delegates_without_deleting_flattened_results(monkeypatch) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)
    prior = _flattened_planned_result(
        state,
        status="error",
        run_id="run-prior",
    )
    prior["error"] = "previous failure"
    state.cell_results.append(prior)
    _finish_planned_state(state)
    state.error = "Backend restarted before this experiment completed."
    threads: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda self: threads.append(self),
    )

    store.retry_experiment_errors(
        state.experiment_id,
        {"api_key": "fresh-secret"},
    )

    assert prior in state.cell_results
    assert state.error is None
    assert len(threads) == 1
    assert threads[0]._target == store._run_retry_batch_worker


def test_explicit_attempt_id_owns_retry_events_and_model_calls(monkeypatch) -> None:
    runtime = NoopTechniqueRuntimePersistence()
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=runtime,
    )

    class Client:
        model = "test-model"
        api_base_url = "https://provider.test/v1"
        system_prompt = ""

        def __init__(self, responses: list[str]) -> None:
            self.responses = list(responses)

        def chat_with_retry(self, _messages, **_kwargs):
            return self.responses.pop(0), {"total_tokens": 1}

    target = Client(["target response"])
    judge = Client(['{"score": 8, "rationale": "ok"}'])
    monkeypatch.setattr(store, "_build_chat_client", lambda _config: target)
    monkeypatch.setattr(store, "_build_attacker_client", lambda _config: None)
    monkeypatch.setattr(
        store,
        "_build_judge_client",
        lambda _config, **_kwargs: judge,
    )

    summary, _ = store._execute_single(
        technique_key="custom:ownership",
        target_kind="chatbot",
        mode="live",
        target_config={
            "unsafe_intent": "intent",
            "system_prompt": "system",
            "model": "test-model",
        },
        custom_definition={
            "id": "ownership",
            "mode": "single_shot",
            "prompt_template": "Attack {intent}",
        },
        experiment_id="exp-ownership",
        cell_id="cell-ownership",
        attempt_id="attempt-explicit",
    )

    assert summary.status == "completed"
    events = store.normalized_events_by_run[summary.run_id]
    assert events
    assert {event.correlation.attempt_id for event in events} == {"attempt-explicit"}
    assert runtime._model_calls
    assert {
        item["call"].correlation.attempt_id
        for item in runtime._model_calls.values()
    } == {"attempt-explicit"}


def test_attempt_report_excludes_selected_inconclusive_judge(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        run_id="run-selected",
    )
    state.cell_results = [_flattened_planned_result(state, score=0, asr=0)]

    report = store._build_experiment_report(state.cell_results)

    assert report["tested"] == 0
    assert report["inconclusive"] == 1
    assert report["avg_score"] is None
    assert report["overall_asr"] is None
    assert report["asr_ci_low"] is None
    assert report["asr_ci_high"] is None
    assert report["score_stddev"] is None
    assert report["score_distribution"] is None


def test_attempt_report_keeps_selected_tested_zero_out_of_errors(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
        run_id="run-selected",
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            status="error",
            score=99,
            asr=0,
        )
    ]

    report = store._build_experiment_report(state.cell_results)

    assert report["tested"] == 1
    assert report["avg_score"] == 0
    assert report["errors"] == 0
    assert report["canonical_rows"][0]["execution_status"] == "completed"
    assert report["canonical_rows"][0]["result_disposition"] == "tested"
    assert report["canonical_rows"][0]["score"] == 0
    assert report["operational"]["failed_attempts"] == 0


def test_fractional_selected_attempt_score_survives_metrics_and_export(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=35.5,
        run_id="run-fractional",
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            run_id="run-fractional",
            score=35.5,
            asr=0,
        )
    ]

    report = store._build_experiment_report(state.cell_results)
    exported = store.export_experiment_json(state.experiment_id)

    assert selected.score == 35.5
    assert report["avg_score"] == 35.5
    assert report["canonical_rows"][0]["score"] == 35.5
    assert report["canonical_rows"][0]["attempts"][0]["score"] == 35.5
    assert exported is not None
    assert exported["cell_results"][0]["score"] == 35.5


def test_attempt_report_never_selects_an_earlier_higher_score(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    higher = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=95,
        run_id="run-higher",
    )
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=20,
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            run_id="run-higher",
            score=95,
            asr=100,
        )
    ]

    report = store._build_experiment_report(state.cell_results)

    assert selected.attempt_id != higher.attempt_id
    assert report["canonical_rows"][0]["selected_attempt_id"] == selected.attempt_id
    assert report["avg_score"] == 20
    assert report["overall_asr"] == 0
    assert report["operational"]["total_attempts"] == 2


def test_attempt_report_keeps_configuration_arms_separate(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(
        monkeypatch,
        technique_configs=[
            {"technique_key": "probe", "parameters": {"threshold": 4}},
            {"technique_key": "probe", "parameters": {"threshold": 6}},
        ],
    )
    first = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=0,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=20,
        run_id="run-arm-a",
    )
    second_arm_index = next(
        index
        for index, cell in enumerate(state.plan.cells if state.plan else ())
        if cell.technique_config_id != state.plan.cells[0].technique_config_id
    )
    second = _complete_reporting_attempt(
        attempts,
        state,
        cell_index=second_arm_index,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
        run_id="run-arm-b",
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            cell_index=0,
            run_id="run-arm-a",
            score=20,
            asr=0,
        ),
        _flattened_planned_result(
            state,
            cell_index=second_arm_index,
            run_id="run-arm-b",
            score=80,
            asr=100,
        ),
    ]

    report = store._build_experiment_report(state.cell_results)

    config_stats = report["technique_config_stats"]
    assert list(config_stats) == sorted(config_stats)
    assert set(config_stats) == {
        state.plan.cells[0].technique_config_id,
        state.plan.cells[second_arm_index].technique_config_id,
    }
    assert {stats["avg_score"] for stats in config_stats.values()} == {20, 80}
    assert {stats["spec_hash"] for stats in config_stats.values()} == {
        first.spec_hash,
        second.spec_hash,
    }
    assert report["technique_stats"]["probe"]["tested"] == 2

    metrics = store.compute_experiment_metrics(state.experiment_id)
    assert metrics is not None
    assert set(metrics["technique_config_metrics"]) == set(config_stats)
    assert {stats["avg_score"] for stats in metrics["technique_config_metrics"].values()} == {20.0, 80.0}


def test_attempt_report_includes_planned_unrun_cells_and_null_rates(monkeypatch) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)

    report = store._build_experiment_report(state.cell_results)

    assert report["total_cells"] == 2
    assert report["tested"] == 0
    assert report["avg_score"] is None
    assert report["overall_asr"] is None
    assert report["score_distribution"] is None
    assert report["operational"]["planned_cells"] == 2
    assert report["operational"]["total_attempts"] == 0
    assert report["operational"]["first_attempt_completion_rate"] is None
    assert report["operational"]["retry_frequency"] is None
    assert report["operational"]["retry_recovery_rate"] is None


def test_attempt_aware_csv_and_json_append_fields_and_preserve_zero(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
        run_id="run-selected",
    )
    state.cell_results = [_flattened_planned_result(state, score=0, asr=0)]

    exported_csv = store.export_experiment_csv(state.experiment_id)
    assert exported_csv is not None
    reader = csv.DictReader(io.StringIO(exported_csv))
    rows = list(reader)
    assert reader.fieldnames is not None
    assert reader.fieldnames[-4:] == [
        "selected_attempt_id",
        "attempt_count",
        "result_disposition",
        "failure_category",
    ]
    assert {
        "experiment_id",
        "technique_key",
        "prompt_id",
        "score",
        "asr",
        "status",
        "error",
    }.issubset(reader.fieldnames)
    assert len(rows) == 2
    assert rows[0]["score"] == "0"
    assert rows[0]["selected_attempt_id"] == selected.attempt_id
    assert rows[0]["attempt_count"] == "1"
    assert rows[0]["result_disposition"] == "tested"
    assert rows[0]["failure_category"] == ""

    exported_json = store.export_experiment_json(state.experiment_id)
    assert exported_json is not None
    assert exported_json["cell_results"][0]["score"] == 0
    assert exported_json["cell_results"][0]["selected_attempt_id"] == selected.attempt_id
    assert exported_json["cell_results"][0]["attempt_count"] == 1
    assert exported_json["cell_results"][0]["result_disposition"] == "tested"
    assert exported_json["cell_results"][0]["failure_category"] is None
    assert exported_json["report"]["tested"] == 1
    assert exported_json["metrics"]["tested_cells"] == 1


def test_legacy_no_attempt_result_keeps_single_result_metrics() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.cell_results = [
        {
            "technique_key": "crescendo_liar",
            "prompt_id": "p1",
            "prompt_text": "intent one",
            "prompt_category": "cat-a",
            "repeat_index": 1,
            "status": "completed",
            "score": 80,
            "asr": 100,
            "conversation": [],
        }
    ]
    store._experiments[state.experiment_id] = state

    report = store._build_experiment_report(state.cell_results)
    metrics = store.compute_experiment_metrics(state.experiment_id)
    exported = store.export_experiment_json(state.experiment_id)

    assert report["tested"] == 1
    assert report["avg_score"] == 80
    assert report["operational"]["total_attempts"] == 1
    assert report["canonical_rows"][0]["selected_attempt_id"].startswith("compat:")
    assert metrics is not None
    assert metrics["tested_cells"] == 1
    assert metrics["overall_binary_asr"] == 100.0
    assert metrics["overall_avg_score"] == 80.0
    assert exported is not None
    assert exported["cell_results"][0]["attempt_count"] == 1


def test_attempt_outputs_are_deterministic_and_definition_versioned(monkeypatch) -> None:
    store, _, state = _attempt_reporting_store(monkeypatch)

    first_report = store._build_experiment_report(state.cell_results)
    second_report = store._build_experiment_report(state.cell_results)
    first_metrics = store.compute_experiment_metrics(state.experiment_id)
    second_metrics = store.compute_experiment_metrics(state.experiment_id)

    assert first_report == second_report
    assert first_metrics == second_metrics
    assert first_report["metric_definition_version"] == "attempt-aware-v1"
    assert first_report["metric_source_scope"] == {
        "efficacy": "selected_attempt",
        "operational": "all_attempts",
    }
    assert first_metrics is not None
    assert first_metrics["metric_definition_version"] == "attempt-aware-v1"
    assert first_metrics["metric_source_scope"] == first_report["metric_source_scope"]
    assert [
        (row["technique_config_id"], row["prompt_id"], row["repeat_index"]) for row in first_report["canonical_rows"]
    ] == sorted(
        (
            row["technique_config_id"],
            row["prompt_id"],
            row["repeat_index"],
        )
        for row in first_report["canonical_rows"]
    )


def test_json_export_uses_one_stabilized_attempt_snapshot(monkeypatch) -> None:
    attempts = _MutatingSnapshotAttemptPersistence()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    first = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=20,
        run_id="run-coherent-first",
    )
    second = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
        run_id="run-stale-second",
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            run_id="run-stale-second",
            score=80,
            asr=100,
        )
    ]
    assert state.plan is not None
    attempts.arm(
        cell_id=state.plan.cells[0].cell_id,
        selected_attempt_ids=(first.attempt_id, second.attempt_id),
        continuous=False,
    )

    exported = store.export_experiment_json(state.experiment_id)

    assert exported is not None
    row = exported["cell_results"][0]
    assert row["selected_attempt_id"] == first.attempt_id
    assert row["run_id"] == "run-coherent-first"
    assert row["score"] == 20
    assert exported["report"]["canonical_rows"] == exported["cell_results"]
    assert exported["report"]["avg_score"] == 20
    assert exported["metrics"]["canonical_rows"] == exported["cell_results"]
    assert exported["metrics"]["overall_avg_score"] == 20.0
    assert attempts.list_attempt_calls == 2
    assert attempts.get_cell_calls == 4


def test_json_export_fails_closed_after_bounded_snapshot_retries(monkeypatch) -> None:
    attempts = _MutatingSnapshotAttemptPersistence()
    store, _, state = _attempt_reporting_store(
        monkeypatch,
        attempt_persistence=attempts,
    )
    first = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=20,
        run_id="run-first",
    )
    second = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
        run_id="run-second",
    )
    assert state.plan is not None
    attempts.arm(
        cell_id=state.plan.cells[0].cell_id,
        selected_attempt_ids=(first.attempt_id, second.attempt_id),
        continuous=True,
    )

    with pytest.raises(RuntimeError, match="stable attempt history"):
        store.export_experiment_json(state.experiment_id)

    assert attempts.list_attempt_calls == 3
    assert attempts.get_cell_calls == 6


def test_attempt_report_projects_authoritative_selected_run_id(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=60,
        run_id="run-authoritative",
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            run_id="run-stale",
            score=90,
            asr=100,
        )
    ]

    report = store._build_experiment_report(state.cell_results)

    row = report["canonical_rows"][0]
    assert row["selected_attempt_id"] == selected.attempt_id
    assert row["run_id"] == "run-authoritative"
    assert row["score"] == 60


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_attempt_report_excludes_non_finite_scores_from_json(
    monkeypatch,
    bad_score: float,
) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    with pytest.raises(ValueError, match="finite numeric score"):
        _complete_reporting_attempt(
            attempts,
            state,
            execution_status=ExecutionStatus.COMPLETED,
            result_disposition=ResultDisposition.TESTED,
            score=bad_score,
            run_id="run-non-finite",
        )
    state.cell_results = [
        _flattened_planned_result(
            state,
            run_id="run-non-finite",
            score=bad_score,
            asr=100,
        )
    ]
    state.cell_results[0]["conversation"] = [
        {"kind": "round", "judge": {"score": bad_score}}
    ]

    report = store._build_experiment_report(state.cell_results)
    exported = store.export_experiment_json(state.experiment_id)

    assert report["tested"] == 0
    assert report["avg_score"] is None
    assert report["canonical_rows"][0]["score"] is None
    assert report["canonical_rows"][0]["attempts"][0]["score"] is None
    assert report["canonical_rows"][0]["attempts"][0]["execution_status"] == "queued"
    assert exported is not None
    json.dumps(exported, allow_nan=False)


@pytest.mark.parametrize(
    ("binary_success", "expected_success", "expected_asr"),
    [(True, True, 100), (False, False, 0)],
)
def test_legacy_binary_success_preserves_boolean_meaning(
    binary_success: bool,
    expected_success: bool,
    expected_asr: int,
) -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.cell_results = [
        {
            "technique_key": "crescendo_liar",
            "prompt_id": "p1",
            "prompt_text": "intent one",
            "prompt_category": "cat-a",
            "repeat_index": 1,
            "status": "completed",
            "score": 80,
            "binary_success": binary_success,
            "conversation": [],
        }
    ]
    store._experiments[state.experiment_id] = state

    report = store._build_experiment_report(state.cell_results)

    assert report["canonical_rows"][0]["success"] is expected_success
    assert report["overall_asr"] == expected_asr


def test_restore_replaces_stale_report_with_attempt_aware_projection(monkeypatch) -> None:
    _, attempts, state = _attempt_reporting_store(monkeypatch)
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=40,
        run_id="run-restored",
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            run_id="run-restored",
            score=40,
            asr=0,
        )
    ]
    persistence = _RestoringRunPersistence(
        [_persisted_experiment_payload(state, report={"legacy_metric": 100})]
    )

    restored = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )._experiments[state.experiment_id]

    assert restored.report is not None
    assert "legacy_metric" not in restored.report
    assert restored.report["metric_definition_version"] == "attempt-aware-v1"
    assert restored.report["canonical_rows"][0]["selected_attempt_id"] == selected.attempt_id
    assert restored.report["total_cells"] == len(state.plan.cells)


def test_flatten_bridge_rejects_untrusted_in_memory_cell_results(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    assert state.plan is not None
    planned = state.plan.cells[0]
    state.cell_results = [_flattened_planned_result(state)]

    store._ensure_flattened_attempt_history(state)

    assert attempts.list_attempts(planned.cell_id) == ()


def test_trusted_persistence_restore_marks_and_consumes_pre_attempt_rows_once(
    monkeypatch,
) -> None:
    source_store, _source_attempts, source_state = _attempt_reporting_store(monkeypatch)
    assert source_state.plan is not None
    source_state.cell_results = [_flattened_planned_result(source_state, score=35.5)]
    _finish_planned_state(source_state)
    restored_attempts = InMemoryExperimentAttemptPersistence()
    configs, cells = source_store._experiment_plan_records(source_state.plan)
    for config in configs:
        restored_attempts.upsert_technique_config(config)
    for cell in cells:
        restored_attempts.create_cell(cell)
    persistence = _RestoringRunPersistence(
        [_persisted_experiment_payload(source_state, report=None)]
    )

    restored_store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=restored_attempts,
    )
    restored = restored_store._experiments[source_state.experiment_id]
    restored_cell_id = source_state.plan.cells[0].cell_id

    assert "_restored_pre_attempt_cell_ids" not in inspect.signature(
        ExperimentState
    ).parameters
    assert restored._restored_pre_attempt_cell_ids == {restored_cell_id}
    assert "restored_pre_attempt" not in json.dumps(
        restored_store._serialize_experiment(restored)
    )

    restored_store._ensure_flattened_attempt_history(restored)
    first_history = restored_attempts.list_attempts(restored_cell_id)
    restored_store._ensure_flattened_attempt_history(restored)

    assert len(first_history) == 1
    assert first_history[0].score == 35.5
    assert restored_attempts.list_attempts(restored_cell_id) == first_history
    assert restored._restored_pre_attempt_cell_ids == set()


@pytest.mark.parametrize("entrypoint", ("restart_cell", "resume_cell"))
def test_direct_retry_entrypoints_materialize_trusted_restored_row_before_eligibility(
    monkeypatch,
    entrypoint,
) -> None:
    source_store, _source_attempts, source_state = _attempt_reporting_store(monkeypatch)
    assert source_state.plan is not None
    source_state.cell_results = [_flattened_planned_result(source_state, score=35.5)]
    _finish_planned_state(source_state)
    restored_attempts = InMemoryExperimentAttemptPersistence()
    configs, cells = source_store._experiment_plan_records(source_state.plan)
    for config in configs:
        restored_attempts.upsert_technique_config(config)
    for cell in cells:
        restored_attempts.create_cell(cell)
    restored_store = EvaluationStore(
        persistence=_RestoringRunPersistence(
            [_persisted_experiment_payload(source_state, report=None)]
        ),
        catalog=_planning_catalog(),
        experiment_attempt_persistence=restored_attempts,
    )
    restored = restored_store._experiments[source_state.experiment_id]
    restored_cell_id = source_state.plan.cells[0].cell_id

    with pytest.raises(ValueError, match="completed cell is excluded"):
        getattr(restored_store, entrypoint)(
            source_state.experiment_id,
            restored_cell_id,
            completed_policy="exclude",
            credential_overrides={"api_key": "fresh-secret"},
        )

    history = restored_attempts.list_attempts(restored_cell_id)
    assert len(history) == 1
    assert history[0].attempt_number == 1
    assert history[0].retry_kind == "initial"
    assert history[0].execution_status is ExecutionStatus.COMPLETED
    assert history[0].score == 35.5
    assert restored._restored_pre_attempt_cell_ids == set()


def test_restore_provenance_rejects_a_caller_replaced_compatibility_row(
    monkeypatch,
) -> None:
    source_store, _source_attempts, source_state = _attempt_reporting_store(monkeypatch)
    assert source_state.plan is not None
    source_state.cell_results = [_flattened_planned_result(source_state, score=35.5)]
    _finish_planned_state(source_state)
    restored_attempts = InMemoryExperimentAttemptPersistence()
    configs, cells = source_store._experiment_plan_records(source_state.plan)
    for config in configs:
        restored_attempts.upsert_technique_config(config)
    for cell in cells:
        restored_attempts.create_cell(cell)
    restored_store = EvaluationStore(
        persistence=_RestoringRunPersistence(
            [_persisted_experiment_payload(source_state, report=None)]
        ),
        catalog=_planning_catalog(),
        experiment_attempt_persistence=restored_attempts,
    )
    restored = restored_store._experiments[source_state.experiment_id]
    restored_cell_id = source_state.plan.cells[0].cell_id
    restored.cell_results[0] = {
        **restored.cell_results[0],
        "run_id": "caller-injected-run",
        "score": 99,
    }

    restored_store._ensure_flattened_attempt_history(restored)

    assert restored_attempts.list_attempts(restored_cell_id) == ()
    assert restored._restored_pre_attempt_cell_ids == set()


def test_restore_null_report_projects_partial_plan_with_all_pinned_cells(monkeypatch) -> None:
    _, attempts, state = _attempt_reporting_store(monkeypatch)
    _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
        run_id="run-partial",
    )
    state.cell_results = [
        _flattened_planned_result(
            state,
            run_id="run-partial",
            score=80,
            asr=100,
        )
    ]
    persistence = _RestoringRunPersistence(
        [_persisted_experiment_payload(state, report=None)]
    )

    restored = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )._experiments[state.experiment_id]

    assert restored.report is not None
    assert len(restored.report["canonical_rows"]) == len(state.plan.cells)
    assert restored.report["tested"] == 1
    assert {row["technique_version"] for row in restored.report["canonical_rows"]} == {
        "1.2.3"
    }
    assert sum(row["attempt_count"] == 0 for row in restored.report["canonical_rows"]) == 1


def test_restore_fully_unrun_plan_projects_report(monkeypatch) -> None:
    _, attempts, state = _attempt_reporting_store(monkeypatch)
    persistence = _RestoringRunPersistence(
        [_persisted_experiment_payload(state, report=None)]
    )

    restored = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )._experiments[state.experiment_id]

    assert restored.report is not None
    assert restored.report["metric_definition_version"] == "attempt-aware-v1"
    assert restored.report["total_cells"] == len(state.plan.cells)
    assert restored.report["tested"] == 0
    assert restored.report["overall_asr"] is None
    assert restored.report["operational"]["planned_cells"] == len(state.plan.cells)


def test_experiment_dataset_prompt_does_not_replace_technique_template(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence())
    dataset = store.create_dataset(
        name="AIM intent",
        prompts=[{
            "text": "dataset intent",
            "category": "cat-a",
            "gold_answer": "expected answer",
        }],
    )
    created = store.create_experiment(
        name="AIM intent",
        dataset_id=dataset["dataset_id"],
        technique_keys=["aim"],
        target_config={
            "model": "target-model",
            "seed_prompt": "must not replace AIM",
        },
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    captured_configs = []

    def fake_execute_single(**kwargs):
        captured_configs.append(kwargs["target_config"])
        return RunSummary(
            run_id="run-aim",
            technique_key="aim",
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment(
        state.experiment_id,
        {"model": "target-model", "seed_prompt": "must not replace AIM"},
        "target-key",
    )

    assert captured_configs[0]["unsafe_intent"] == "dataset intent"
    assert captured_configs[0]["gold_answer"] == "expected answer"
    assert "seed_prompt" not in captured_configs[0]


def test_planned_experiment_does_not_reapply_suppressed_seed_prompt(monkeypatch) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    custom_definition = {
        "id": "single-custom",
        "name": "Single custom",
        "mode": "single_shot",
        "prompt_template": "Attack {intent}",
    }
    persistence = _PlanningRunPersistence([], [custom_definition])
    store = EvaluationStore(persistence=persistence, catalog=_planning_catalog())
    payload = _create_planned_experiment(
        store,
        technique_keys=["custom:single-custom"],
        technique_configs=None,
        target_config={
            "model": "target-model",
            "seed_prompt": "must remain suppressed",
        },
        repeat_count=1,
    )
    captured_configs = []

    def fake_execute_custom_single(**kwargs):
        captured_configs.append(dict(kwargs["target_config"]))
        return TechniqueRunResult(
            run_id=kwargs["run_id"],
            technique_key=kwargs["technique_key"],
            target_id="demo-target",
            response=TargetResponse(
                target_id="demo-target",
                target_kind=TargetKind.CHATBOT,
                final_output="safe",
            ),
            detections=(),
            score=0,
            traces=(),
            audit_log=(),
        )

    store._execute_custom_single = fake_execute_custom_single
    state = store._experiments[payload["experiment_id"]]
    store._run_experiment(
        state.experiment_id,
        {"model": "target-model", "seed_prompt": "must remain suppressed"},
        "target-key",
    )

    assert len(captured_configs) == 2
    assert captured_configs[0]["unsafe_intent"] == "intent one"
    assert all("seed_prompt" not in config for config in captured_configs)


def test_experiment_retry_does_not_restore_dataset_prompt_as_seed(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=NoopRunPersistence())
    dataset = store.create_dataset(
        name="AIM retry intent",
        prompts=[{
            "text": "dataset intent",
            "category": "cat-a",
            "gold_answer": "expected answer",
        }],
    )
    created = store.create_experiment(
        name="AIM retry intent",
        dataset_id=dataset["dataset_id"],
        technique_keys=["aim"],
        target_config={
            "model": "target-model",
            "seed_prompt": "must not replace AIM",
        },
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    assert state.plan is not None
    planned_cell = state.plan.cells[0]
    technique_config = state.plan.technique_configs[0]
    _finish_planned_state(state)
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=state._execution_generation,
        owner_id=state._execution_owner_id,
        status="completed",
    )
    captured_configs = []

    def fake_execute_single(**kwargs):
        captured_configs.append(kwargs["target_config"])
        return RunSummary(
            run_id="run-aim-retry",
            technique_key="aim",
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment_retry(
        state.experiment_id,
        [(
            technique_config,
            {
                "id": planned_cell.prompt_id,
                "text": planned_cell.prompt_text,
                "category": planned_cell.prompt_category,
                "gold_answer": planned_cell.gold_answer,
            },
            planned_cell.repeat_index,
        )],
        {"model": "target-model", "seed_prompt": "must not replace AIM"},
        "target-key",
    )

    assert captured_configs[0]["unsafe_intent"] == "dataset intent"
    assert captured_configs[0]["gold_answer"] == "expected answer"
    assert "seed_prompt" not in captured_configs[0]


def test_experiment_cell_config_omits_missing_or_empty_gold_answers() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())

    missing = store._experiment_cell_config(
        full_config={"gold_answer": "stale answer"},
        prompt={"id": "p1", "text": "no reference"},
        max_rounds=5,
    )
    empty = store._experiment_cell_config(
        full_config={},
        prompt={"id": "p2", "text": "empty reference", "gold_answer": "   "},
        max_rounds=5,
    )

    assert "gold_answer" not in missing
    assert "gold_answer" not in empty


def test_experiment_keeps_judge_failure_cells_retryable_with_conversation(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda _thread: None)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=_planning_catalog(),
    )
    dataset = store.create_dataset(
        name="judge failure evidence",
        prompts=[{"text": "dataset intent", "category": "cat-a"}],
    )
    created = store.create_experiment(
        name="judge failure evidence",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]

    def fake_execute_single(**kwargs):
        return RunSummary(
            run_id="run-judge-error",
            technique_key=str(kwargs["technique_key"]),
            target_kind="chatbot",
            status="error",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[
                {
                    "kind": "single_shot",
                    "response": "Preserved target response.",
                    "judge": {"error": "judge provider rejected the request"},
                }
            ],
            error="judge provider rejected the request",
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment(
        state.experiment_id,
        {"model": "target-model"},
        "target-key",
        execution_generation=state._execution_generation,
        execution_owner_id=state._execution_owner_id,
    )

    cell = state.cell_results[0]
    assert cell["status"] == "error"
    assert cell["error"] == "judge provider rejected the request"
    assert cell["conversation"][0]["response"] == "Preserved target response."


def test_experiment_metrics_do_not_emit_removed_ablation() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.cell_results = [
        {
            "technique_key": "crescendo_liar",
            "prompt_id": "p1",
            "prompt_text": "intent one",
            "prompt_category": "cat-a",
            "repeat_index": 1,
            "status": "completed",
            "score": 80,
            "asr": 100,
            "conversation": [],
        }
    ]
    store._experiments[state.experiment_id] = state

    metrics = store.compute_experiment_metrics(state.experiment_id)

    assert metrics is not None
    assert metrics["ablation"] == {}


def _rejudge_fixture(monkeypatch, outcomes: list[object]):
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    runtime = RecordingTechniqueRuntimePersistence()
    store.technique_runtime_persistence = runtime
    rejudge_persistence = RecordingRejudgeRunPersistence()
    store.persistence = rejudge_persistence
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
        run_id="run-selected-execution",
    )
    cell = _flattened_planned_result(
        state,
        run_id="run-selected-execution",
        score=80,
        asr=100,
    )
    cell["attempt_id"] = selected.attempt_id
    cell["selected_attempt_id"] = selected.attempt_id
    cell["conversation"] = [
        {
            "kind": "round",
            "round_number": 1,
            "target": {
                "response": "target refusal",
                "response_message_id": "message-target-saved",
                "poisoned": "fabricated harmful response",
                "poisoned_message_id": "message-poisoned-saved",
            },
        }
    ]
    state.cell_results = [cell]
    state.status = "completed"
    state.report = store._build_experiment_report(state.cell_results)
    rejudge_persistence.save_experiment(
        experiment={
            "experiment_id": state.experiment_id,
            "status": "completed",
            "execution_generation": state._execution_generation,
            "execution_owner_id": state._execution_owner_id,
        }
    )
    judge = SequencedRejudgeClient(outcomes)
    judge_configs = []

    def build_judge(config, *, target_client, attacker_client):
        judge_configs.append(dict(config))
        return judge

    def unexpected_target_builder(*args, **kwargs):
        pytest.fail("rejudge must not build or invoke a target/attacker client")

    monkeypatch.setattr(store, "_build_chat_client", unexpected_target_builder)
    monkeypatch.setattr(store, "_build_attacker_client", unexpected_target_builder)
    monkeypatch.setattr(store, "_build_judge_client", build_judge)
    return (
        store,
        attempts,
        state,
        runtime,
        judge,
        judge_configs,
        selected,
        rejudge_persistence,
    )


def _rejudge_config() -> dict[str, str]:
    return {
        "judge_mode": "custom",
        "judge_api_key": "judge-secret-key",
        "judge_api_base_url": "http://judge.test/v1",
        "judge_model": "third-judge",
    }


def _run_owned_rejudge(
    store: EvaluationStore,
    state: ExperimentState,
    judge_config: dict[str, str],
) -> None:
    if not any(
        experiment.get("experiment_id") == state.experiment_id
        for experiment in store.persistence.load_all_experiments()
    ):
        store.persistence.save_experiment(
            experiment={
                "experiment_id": state.experiment_id,
                "status": state.status,
                "execution_generation": state._execution_generation,
                "execution_owner_id": state._execution_owner_id,
            }
        )
    owner_id = (
        f"rejudge-test:{state.experiment_id}:{state._execution_generation + 1}"
    )
    generation = store.persistence.begin_experiment_execution(
        state.experiment_id,
        owner_id=owner_id,
        expected_generation=state._execution_generation,
        active_status="rejudging",
    )
    state._execution_generation = generation
    state._execution_owner_id = owner_id
    state.status = "rejudging"
    state.error = None
    store._run_experiment_rejudge(
        state.experiment_id,
        judge_config,
        execution_generation=generation,
        execution_owner_id=owner_id,
    )


def test_rejudge_owner_begin_failure_dispatches_no_judge_calls(monkeypatch) -> None:
    store, attempts, state = _attempt_reporting_store(monkeypatch)
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
    )
    cell = _flattened_planned_result(state)
    cell["attempt_id"] = selected.attempt_id
    cell["selected_attempt_id"] = selected.attempt_id
    cell["conversation"] = [{"kind": "single_shot", "response": "saved"}]
    state.cell_results = [cell]
    state.status = "completed"
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="completed",
    )
    judge_calls = []
    workers = []
    monkeypatch.setattr(
        store.persistence,
        "begin_experiment_execution",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("rejudge ownership unavailable")
        ),
    )
    monkeypatch.setattr(
        store,
        "_build_rejudge_judge_client",
        lambda config: judge_calls.append(config),
    )
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    timers = _capture_retry_timers(monkeypatch)

    current = store.rejudge_experiment(state.experiment_id, _rejudge_config())

    assert current["status"] == "completed"
    assert judge_calls == []
    assert workers == []
    assert len(timers) == 1
    assert state.experiment_id in store._rejudge_acquisition_obligations
    assert state.status == "completed"


def test_rejudge_lost_ack_replays_same_owner_and_starts_one_worker(monkeypatch) -> None:
    class LostAckWithAudit(_LostAcknowledgementBeginPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.owners: list[str] = []

        def begin_experiment_execution(self, *args, **kwargs):
            self.owners.append(str(kwargs["owner_id"]))
            return super().begin_experiment_execution(*args, **kwargs)

    persistence = LostAckWithAudit()
    attempts = InMemoryExperimentAttemptPersistence()
    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(name="rejudge lost ack", prompts=[{"text": "intent"}])
    created = store.create_experiment(
        name="rejudge lost ack",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    selected = _complete_reporting_attempt(
        attempts,
        state,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
    )
    cell = _flattened_planned_result(state)
    cell["attempt_id"] = selected.attempt_id
    cell["selected_attempt_id"] = selected.attempt_id
    cell["conversation"] = [{"kind": "single_shot", "response": "saved"}]
    state.cell_results = [cell]
    state.status = "completed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="completed",
    )
    persistence.lose_next_acknowledgement = True
    judge = SequencedRejudgeClient(
        ['{"score": 8, "rationale": "same-owner replay"}']
    )
    monkeypatch.setattr(
        store,
        "_build_rejudge_judge_client",
        lambda _config: judge,
    )

    first = store.rejudge_experiment(state.experiment_id, _rejudge_config())
    second = store.rejudge_experiment(state.experiment_id, _rejudge_config())

    assert first["status"] == second["status"] == "rejudging"
    assert len(set(persistence.owners)) == 1
    assert len(persistence.owners) == 2
    assert len(workers) == 2  # initial experiment worker + one rejudge worker
    durable = persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == 2
    assert durable["execution_owner_id"] == persistence.owners[0]
    workers[1]._target(*workers[1]._args)
    assert len(judge.calls) == 1


def test_rejudge_persistent_transient_retains_one_bounded_same_owner_obligation(
    monkeypatch,
) -> None:
    class PersistentlyTransient(NoopRunPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.owners: list[str] = []

        def begin_experiment_execution(self, *args, **kwargs):
            self.owners.append(str(kwargs["owner_id"]))
            raise RuntimeError("rejudge begin temporarily unavailable")

    persistence = PersistentlyTransient()
    attempts = InMemoryExperimentAttemptPersistence()
    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(name="rejudge transient", prompts=[{"text": "intent"}])
    created = store.create_experiment(
        name="rejudge transient",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    state.cell_results = [_flattened_planned_result(state)]
    state.status = "completed"
    assert NoopRunPersistence.reconcile_owned_experiment_terminal_status(
        persistence,
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="completed",
    )
    timers = _capture_retry_timers(monkeypatch)

    first = store.rejudge_experiment(state.experiment_id, _rejudge_config())
    second = store.rejudge_experiment(state.experiment_id, _rejudge_config())

    assert first["status"] == second["status"] == "completed"
    assert len(set(persistence.owners)) == 1
    assert len(timers) == 1
    assert len(workers) == 1  # only the captured initial worker
    timers[0].fire()
    assert len(set(persistence.owners)) == 1
    assert len(timers) == 1
    assert state.experiment_id in store._rejudge_acquisition_obligations


def test_rejudge_persistent_confirmation_failure_schedules_only_one_recovery(
    monkeypatch,
) -> None:
    class PersistentlyUnconfirmed(NoopRunPersistence):
        def update_owned_experiment_status(self, *args, **kwargs):
            if kwargs.get("status") == "rejudging":
                raise RuntimeError("rejudge ownership confirmation unavailable")
            return super().update_owned_experiment_status(*args, **kwargs)

    persistence = PersistentlyUnconfirmed()
    attempts = InMemoryExperimentAttemptPersistence()
    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(
        name="rejudge confirmation transient",
        prompts=[{"text": "intent"}],
    )
    created = store.create_experiment(
        name="rejudge confirmation transient",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    state.cell_results = [_flattened_planned_result(state)]
    state.status = "completed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="completed",
    )
    timers = _capture_retry_timers(monkeypatch)

    store.rejudge_experiment(state.experiment_id, _rejudge_config())

    assert len(timers) == 1
    timers[0].fire()
    assert len(timers) == 1
    assert state.experiment_id in store._rejudge_acquisition_obligations
    assert len(workers) == 1  # only the captured initial worker


def test_rejudge_ownership_move_rejects_evidence_and_stops_later_judge_calls(
    monkeypatch,
) -> None:
    class MoveOwnershipAfterFirstOperation(NoopRunPersistence):
        def __init__(self) -> None:
            super().__init__()
            self.moved = False
            self.rejudge_writes: list[dict[str, object]] = []

        def finish_owned_experiment_operation(self, *args, **kwargs):
            finished = super().finish_owned_experiment_operation(*args, **kwargs)
            if finished and not self.moved:
                self.moved = True
                experiment_id = str(args[0])
                generation = int(kwargs["generation"])
                owner_id = str(kwargs["owner_id"])
                assert self.reconcile_owned_experiment_terminal_status(
                    experiment_id,
                    generation=generation,
                    owner_id=owner_id,
                    status="completed",
                )
                assert self.begin_experiment_execution(
                    experiment_id,
                    owner_id="retry-batch-after-rejudge",
                    expected_generation=generation,
                    active_status="running",
                ) == generation + 1
            return finished

        def save_experiment_rejudge(self, **kwargs):
            accepted = super().save_experiment_rejudge(**kwargs)
            if accepted:
                self.rejudge_writes.append(dict(kwargs))
            return accepted

    workers: list[threading.Thread] = []
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    persistence = MoveOwnershipAfterFirstOperation()
    attempts = InMemoryExperimentAttemptPersistence()
    store = EvaluationStore(
        persistence=persistence,
        catalog=_planning_catalog(),
        experiment_attempt_persistence=attempts,
    )
    dataset = store.create_dataset(
        name="stale rejudge",
        prompts=[{"text": "first"}, {"text": "second"}],
    )
    created = store.create_experiment(
        name="stale rejudge",
        dataset_id=dataset["dataset_id"],
        technique_keys=["probe"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    state = store._experiments[created["experiment_id"]]
    cells = []
    for cell_index in range(2):
        selected = _complete_reporting_attempt(
            attempts,
            state,
            cell_index=cell_index,
            execution_status=ExecutionStatus.COMPLETED,
            result_disposition=ResultDisposition.TESTED,
            score=80,
            run_id=f"run-selected-{cell_index}",
        )
        cell = _flattened_planned_result(
            state,
            cell_index=cell_index,
            run_id=f"run-selected-{cell_index}",
        )
        cell["attempt_id"] = selected.attempt_id
        cell["selected_attempt_id"] = selected.attempt_id
        cell["conversation"] = [
            {"kind": "single_shot", "response": f"saved {cell_index}"}
        ]
        cells.append(cell)
    state.cell_results = cells
    state.status = "completed"
    assert persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="completed",
    )
    judge = SequencedRejudgeClient(
        [
            '{"score": 2, "rationale": "first"}',
            '{"score": 3, "rationale": "second"}',
        ]
    )
    monkeypatch.setattr(store, "_build_rejudge_judge_client", lambda _config: judge)

    store.rejudge_experiment(state.experiment_id, _rejudge_config())
    assert len(workers) == 2
    workers[-1]._target(*workers[-1]._args)

    assert len(judge.calls) == 1
    assert persistence.rejudge_writes == []
    assert all("rejudge_results" not in cell for cell in state.cell_results)
    durable = persistence.load_all_experiments()[0]
    assert durable["execution_owner_id"] == "retry-batch-after-rejudge"


@pytest.mark.parametrize("finish_fails", [False, True])
def test_rejudge_cancellation_reconciles_owned_failure_and_preserves_evidence(
    monkeypatch,
    finish_fails: bool,
) -> None:
    class CancellingJudgeClient:
        model = "cancel-judge"
        api_base_url = "http://judge.test/v1"

        def __init__(self) -> None:
            self.outcomes = [
                '{"score": 2, "rationale": "first saved"}',
                asyncio.CancelledError("cancel rejudge"),
            ]
            self.calls = []

        def chat_with_retry(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome, {"total_tokens": 1}

    store, attempts, state = _attempt_reporting_store(monkeypatch)
    cells = []
    for cell_index in range(2):
        selected = _complete_reporting_attempt(
            attempts,
            state,
            cell_index=cell_index,
            execution_status=ExecutionStatus.COMPLETED,
            result_disposition=ResultDisposition.TESTED,
            score=80,
        )
        cell = _flattened_planned_result(state, cell_index=cell_index)
        cell["attempt_id"] = selected.attempt_id
        cell["selected_attempt_id"] = selected.attempt_id
        cell["conversation"] = [
            {"kind": "single_shot", "response": f"saved response {cell_index}"}
        ]
        cells.append(cell)
    state.cell_results = cells
    state.status = "completed"
    assert store.persistence.reconcile_owned_experiment_terminal_status(
        state.experiment_id,
        generation=1,
        owner_id=f"initial:{state.experiment_id}",
        status="completed",
    )
    client = CancellingJudgeClient()
    workers = []
    monkeypatch.setattr(store, "_build_rejudge_judge_client", lambda _config: client)
    monkeypatch.setattr(
        "app.api.store.threading.Thread.start",
        lambda thread: workers.append(thread),
    )
    timers = _capture_retry_timers(monkeypatch)
    if finish_fails:
        original_finish = store.persistence.finish_owned_experiment_operation

        def fail_cancelled_finish(*args, **kwargs):
            if kwargs.get("outcome") == "cancelled":
                raise RuntimeError("cancelled operation close unavailable")
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(
            store.persistence,
            "finish_owned_experiment_operation",
            fail_cancelled_finish,
        )
    store.rejudge_experiment(state.experiment_id, _rejudge_config())

    with pytest.raises(asyncio.CancelledError, match="cancel rejudge"):
        workers[0]._target(*workers[0]._args)

    if finish_fails:
        authority = store.persistence.execution_authority
        with authority.lock:
            running = next(
                operation
                for operation in authority.operations.values()
                if operation.get("status") == "running"
            )
            running["lease_expires_at"] = running["lease_expires_at"] - timedelta(
                seconds=120
            )
        assert len(timers) == 1
        timers[0].fire()

    assert len(state.cell_results[0]["rejudge_results"]) == 1
    assert "rejudge_results" not in state.cell_results[1]
    assert state.status == "failed"
    assert state.error == "cancel rejudge"
    durable = store.persistence.load_all_experiments()[0]
    assert durable["execution_generation"] == 2
    assert durable["execution_owner_id"].startswith("rejudge-batch:")
    assert durable["status"] == "failed"
    assert durable["error"] == "cancel rejudge"


def test_rejudge_records_named_evaluations_from_saved_evidence_without_target_call(
    monkeypatch,
) -> None:
    (
        store,
        _,
        state,
        runtime,
        judge,
        judge_configs,
        selected,
        rejudge_persistence,
    ) = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 2, "rationale": "target evidence"}',
            '{"score": 8, "rationale": "poisoned evidence"}',
        ],
    )
    state.dataset["prompts"][0]["gold_answer"] = "trusted answer"
    judge.api_base_url = (
        "http://url-user:url-password@judge.test/v1?api_key=query-secret"
    )

    _run_owned_rejudge(store, state, _rejudge_config())

    result = state.cell_results[0]["rejudge_results"][0]
    assert len(judge.calls) == 2
    assert result["judge_model"] == "third-judge"
    assert result["judge_api_base_url"] == "http://judge.test"
    assert result["target_score"] == 2
    assert result["fabricator_score"] == 8
    assert result["target_asr"] == 0
    assert result["fabricator_asr"] == 100
    assert result["pipeline_unsafe_asr"] == 100
    assert result["target_evaluation_id"] != result["fabricator_evaluation_id"]
    assert result["target_subject_message_id"].startswith("subject-ref:")
    assert result["fabricator_subject_message_id"].startswith("subject-ref:")
    assert result["target_subject_message_id"] != "message-target-saved"
    assert result["fabricator_subject_message_id"] != "message-poisoned-saved"
    assert "gold_answer" not in judge_configs[0]
    assert judge_configs[1]["gold_answer"] == "trusted answer"

    lifecycle = [event for event in runtime.events if event.event_type.startswith("judge.")]
    assert len(lifecycle) == 6
    assert {event.event_type for event in lifecycle} == {
        "judge.scheduled",
        "judge.running",
        "judge.completed",
    }
    completed = [event for event in lifecycle if event.event_type == "judge.completed"]
    assert {event.payload["definition_id"] for event in completed} == {
        "target_judge",
        "poisoned_context_judge",
    }
    assert {event.payload["subject_message_id"] for event in completed} == {
        "message-target-saved",
        "message-poisoned-saved",
    }
    assert {event.payload["evaluation_id"] for event in completed} == {
        result["target_evaluation_id"],
        result["fabricator_evaluation_id"],
    }
    assert all(event.correlation.experiment_id == state.experiment_id for event in lifecycle)
    assert all(event.correlation.cell_id == state.cell_results[0]["cell_id"] for event in lifecycle)
    assert all(event.correlation.attempt_id == selected.attempt_id for event in lifecycle)
    assert all(event.payload["source_run_id"] == "run-selected-execution" for event in lifecycle)
    assert all(event.correlation.run_id != "run-selected-execution" for event in lifecycle)
    assert all(str(event.payload["evaluation_id"]) in str(event.operation_id) for event in lifecycle)
    assert "judge-secret-key" not in json.dumps(
        [dict(event.payload) for event in lifecycle],
        sort_keys=True,
    )
    assert len(rejudge_persistence.rejudge_writes) == 1
    assert rejudge_persistence.rejudge_writes[0]["cell_id"] == state.cell_results[0][
        "cell_id"
    ]
    assert rejudge_persistence.rejudge_writes[0]["selected_attempt_id"] == (
        selected.attempt_id
    )
    assert rejudge_persistence.cell_writes == []
    serialized_result = json.dumps(result, sort_keys=True)
    for secret in (
        "url-user",
        "url-password",
        "query-secret",
        "message-target-saved",
        "message-poisoned-saved",
    ):
        assert secret not in serialized_result


@pytest.mark.parametrize(
    "failed_outcome",
    [RuntimeError("judge provider unavailable"), "not valid JSON"],
    ids=["provider_failure", "parse_failure"],
)
def test_rejudge_preserves_minimum_score_and_projects_failure_as_null(
    monkeypatch,
    failed_outcome: object,
) -> None:
    store, _, state, runtime, _, _, _, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 1, "rationale": "minimum score"}',
            failed_outcome,
        ],
    )

    _run_owned_rejudge(store, state, _rejudge_config())

    result = state.cell_results[0]["rejudge_results"][0]
    assert result["target_score"] == 1
    assert result["target_score_pct"] == 10
    assert result["fabricator_score"] is None
    assert result["fabricator_score_pct"] is None
    assert result["fabricator_asr"] == 0
    target_event = next(
        event
        for event in runtime.events
        if event.event_type == "judge.completed"
        and event.payload["definition_id"] == "target_judge"
    )
    poisoned_event = next(
        event
        for event in runtime.events
        if event.event_type == "judge.failed"
        and event.payload["definition_id"] == "poisoned_context_judge"
    )
    assert target_event.payload["score"] == 1
    assert poisoned_event.payload.get("score") is None


@pytest.mark.parametrize(
    "invalid_output",
    [
        '{"score": 0, "rationale": "below minimum"}',
        '{"score": 11, "rationale": "above maximum"}',
        '{"score": NaN, "rationale": "not finite"}',
        '{"score": Infinity, "rationale": "not finite"}',
        (
            '{"score": 5, "engagement": 1e999, "leakage": 1, '
            '"compliance": 1, "rationale": "overflow"}'
        ),
        (
            '{"score": 5, "engagement": 999999999999999999999999, '
            '"leakage": 1, "compliance": 1, "rationale": "huge"}'
        ),
        (
            '{"score": 5, "engagement": true, "leakage": 1, '
            '"compliance": 1, "rationale": "boolean"}'
        ),
        (
            '{"score": 5, "engagement": "2", "leakage": 1, '
            '"compliance": 1, "rationale": "string"}'
        ),
    ],
    ids=[
        "below_minimum",
        "above_maximum",
        "direct_nan",
        "direct_infinity",
        "component_infinity",
        "component_huge",
        "component_boolean",
        "component_nonnumeric",
    ],
)
def test_rejudge_invalid_numeric_output_fails_without_failing_experiment(
    monkeypatch,
    invalid_output: str,
) -> None:
    store, _, state, runtime, _, _, _, _ = _rejudge_fixture(
        monkeypatch,
        ['{"score": 1, "rationale": "minimum score"}', invalid_output],
    )

    _run_owned_rejudge(store, state, _rejudge_config())

    assert state.status == "completed"
    result = state.cell_results[0]["rejudge_results"][0]
    assert result["target_score"] == 1
    assert result["fabricator_score"] is None
    assert any(
        event.event_type == "judge.failed"
        and event.payload["definition_id"] == "poisoned_context_judge"
        and event.payload.get("score") is None
        for event in runtime.events
    )


def test_rejudge_does_not_mutate_attempts_selected_execution_or_efficacy(monkeypatch) -> None:
    store, attempts, state, _, _, _, selected, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 1, "rationale": "new target evaluation"}',
            '{"score": 2, "rationale": "new poisoned evaluation"}',
        ],
    )
    cell_id = state.cell_results[0]["cell_id"]
    before_cell = attempts.get_cell(cell_id)
    before_attempts = attempts.list_attempts(cell_id)
    before_selected = attempts.get_attempt(selected.attempt_id)
    before_execution = {
        "score": state.cell_results[0]["score"],
        "asr": state.cell_results[0]["asr"],
    }
    before_metrics = {
        key: value
        for key, value in json.loads(json.dumps(state.report, sort_keys=True)).items()
        if key != "canonical_rows"
    }

    _run_owned_rejudge(store, state, _rejudge_config())

    assert attempts.get_cell(cell_id) == before_cell
    assert attempts.list_attempts(cell_id) == before_attempts
    assert attempts.get_attempt(selected.attempt_id) == before_selected
    assert {
        "score": state.cell_results[0]["score"],
        "asr": state.cell_results[0]["asr"],
    } == before_execution
    assert {
        key: value for key, value in state.report.items() if key != "canonical_rows"
    } == before_metrics


def test_rejudge_metadata_persistence_failure_is_fail_closed(monkeypatch) -> None:
    store, attempts, state, _, _, _, selected, persistence = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 2, "rationale": "target"}',
            '{"score": 4, "rationale": "poison"}',
        ],
    )
    persistence.fail_rejudge = True
    cell_id = state.cell_results[0]["cell_id"]
    before_cell = attempts.get_cell(cell_id)
    before_attempts = attempts.list_attempts(cell_id)
    before_selected = attempts.get_attempt(selected.attempt_id)

    _run_owned_rejudge(store, state, _rejudge_config())

    assert state.status == "failed"
    assert "rejudge metadata write failed" in str(state.error)
    assert "rejudge_results" not in state.cell_results[0]
    assert "original_score" not in state.cell_results[0]
    assert "original_asr" not in state.cell_results[0]
    assert attempts.get_cell(cell_id) == before_cell
    assert attempts.list_attempts(cell_id) == before_attempts
    assert attempts.get_attempt(selected.attempt_id) == before_selected


def test_rejudge_event_persistence_failure_cannot_finish_completed(monkeypatch) -> None:
    store, attempts, state, _, _, _, selected, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 2, "rationale": "target"}',
            '{"score": 4, "rationale": "poison"}',
        ],
    )
    store.technique_runtime_persistence = FailingRejudgeEventPersistence()
    cell_id = state.cell_results[0]["cell_id"]
    before_cell = attempts.get_cell(cell_id)
    before_attempts = attempts.list_attempts(cell_id)
    before_selected = attempts.get_attempt(selected.attempt_id)

    _run_owned_rejudge(store, state, _rejudge_config())

    assert state.status == "failed"
    assert "rejudge event write failed" in str(state.error)
    assert "rejudge_results" not in state.cell_results[0]
    assert attempts.get_cell(cell_id) == before_cell
    assert attempts.list_attempts(cell_id) == before_attempts
    assert attempts.get_attempt(selected.attempt_id) == before_selected


def test_repeated_rejudge_appends_evaluations_and_keeps_subject_identity(monkeypatch) -> None:
    store, _, state, runtime, _, _, _, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 2, "rationale": "target first"}',
            '{"score": 8, "rationale": "poison first"}',
            '{"score": 3, "rationale": "target second"}',
            '{"score": 7, "rationale": "poison second"}',
        ],
    )

    _run_owned_rejudge(store, state, _rejudge_config())
    first = dict(state.cell_results[0]["rejudge_results"][0])
    first_events = tuple(runtime.events)
    _run_owned_rejudge(store, state, _rejudge_config())
    second = dict(state.cell_results[0]["rejudge_results"][1])

    assert first["rejudge_batch_id"] != second["rejudge_batch_id"]
    assert first["target_evaluation_id"] != second["target_evaluation_id"]
    assert first["fabricator_evaluation_id"] != second["fabricator_evaluation_id"]
    assert first["target_subject_message_id"] == second["target_subject_message_id"]
    assert first["fabricator_subject_message_id"] == second["fabricator_subject_message_id"]
    assert first["policy_version"] == second["policy_version"]
    assert first["policy_version"].startswith("rejudge-policy:")
    assert all(
        event.payload["policy_version"] == first["policy_version"]
        for event in runtime.events
    )
    assert tuple(runtime.events[: len(first_events)]) == first_events
    assert len(store.normalized_events_by_run) == 2
    assert sum(len(events) for events in store.normalized_events_by_run.values()) == 12


def test_rejudge_policy_version_changes_only_with_effective_policy(monkeypatch) -> None:
    store, _, state, runtime, _, _, _, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 1, "rationale": "first target"}',
            '{"score": 2, "rationale": "first poison"}',
            '{"score": 3, "rationale": "second target"}',
            '{"score": 4, "rationale": "second poison"}',
        ],
    )
    first_config = {
        **_rejudge_config(),
        "judge_system_prompt": "policy behavior one",
    }
    second_config = {
        **_rejudge_config(),
        "judge_system_prompt": "policy behavior two",
    }

    _run_owned_rejudge(store, state, first_config)
    _run_owned_rejudge(store, state, second_config)

    first, second = state.cell_results[0]["rejudge_results"]
    assert first["policy_version"] != second["policy_version"]
    assert all("judge-secret-key" not in str(event.payload["policy_version"])
               for event in runtime.events)


def test_rejudge_policy_version_tracks_gold_answer_without_exposing_it(
    monkeypatch,
) -> None:
    store, _, state, runtime, _, _, _, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 1, "rationale": "no gold target"}',
            '{"score": 2, "rationale": "no gold poison"}',
            '{"score": 3, "rationale": "first gold target"}',
            '{"score": 4, "rationale": "first gold poison"}',
            '{"score": 5, "rationale": "second gold target"}',
            '{"score": 6, "rationale": "second gold poison"}',
        ],
    )

    _run_owned_rejudge(store, state, _rejudge_config())
    state.dataset["prompts"][0]["gold_answer"] = "private trusted answer one"
    _run_owned_rejudge(store, state, _rejudge_config())
    state.dataset["prompts"][0]["gold_answer"] = "private trusted answer two"
    _run_owned_rejudge(store, state, _rejudge_config())

    versions = [
        result["policy_version"]
        for result in state.cell_results[0]["rejudge_results"]
    ]
    assert len(set(versions)) == 3
    public_payload = json.dumps(
        {
            "results": state.cell_results[0]["rejudge_results"],
            "events": [dict(event.payload) for event in runtime.events],
        },
        sort_keys=True,
    )
    assert "private trusted answer one" not in public_payload
    assert "private trusted answer two" not in public_payload


def test_rejudge_policy_version_tracks_resolved_judge_mode_and_model(
    monkeypatch,
) -> None:
    store, _, state, _, judge, _, _, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 1, "rationale": "first target"}',
            '{"score": 2, "rationale": "first poison"}',
            '{"score": 3, "rationale": "second target"}',
            '{"score": 4, "rationale": "second poison"}',
            '{"score": 5, "rationale": "third target"}',
            '{"score": 6, "rationale": "third poison"}',
        ],
    )

    _run_owned_rejudge(store, state, _rejudge_config())
    judge.model = "replacement-judge"
    _run_owned_rejudge(
        store,
        state,
        {**_rejudge_config(), "judge_model": "replacement-judge"},
    )
    _run_owned_rejudge(
        store,
        state,
        {"judge_mode": "target", "model": "replacement-judge"},
    )

    versions = [
        result["policy_version"]
        for result in state.cell_results[0]["rejudge_results"]
    ]
    assert len(set(versions)) == 3


def test_rejudge_skips_absent_saved_subjects_instead_of_judging_empty_text(
    monkeypatch,
) -> None:
    store, _, state, runtime, judge, _, _, _ = _rejudge_fixture(
        monkeypatch,
        ['{"score": 9, "rationale": "must remain unused"}'],
    )
    state.cell_results[0]["conversation"] = [
        {
            "kind": "round",
            "round_number": 1,
            "target": {"response": "", "poisoned": ""},
        }
    ]
    before = json.loads(json.dumps(state.cell_results[0], sort_keys=True))

    _run_owned_rejudge(store, state, _rejudge_config())

    assert judge.calls == []
    assert runtime.events == []
    assert state.cell_results[0] == before


def test_rejudge_uses_saved_evidence_from_execution_error_cell(monkeypatch) -> None:
    store, _, state, runtime, judge, _, _, _ = _rejudge_fixture(
        monkeypatch,
        [
            '{"score": 3, "rationale": "saved target"}',
            '{"score": 6, "rationale": "saved poison"}',
        ],
    )
    state.cell_results[0]["status"] = "error"
    state.cell_results[0]["error"] = "original judge failed after target response"

    _run_owned_rejudge(store, state, _rejudge_config())

    assert len(judge.calls) == 2
    assert len(runtime.events) == 6
    assert state.cell_results[0]["rejudge_results"][0]["target_score"] == 3


def test_rejudge_legacy_cell_uses_explicit_stable_attempt_fallback(monkeypatch) -> None:
    runtime = RecordingTechniqueRuntimePersistence()
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=runtime,
    )
    state = _experiment_state(repeat_count=1)
    state.status = "completed"
    state.cell_results = [
        {
            "technique_key": "crescendo_liar",
            "prompt_id": "p1",
            "prompt_text": "intent one",
            "prompt_category": "cat-a",
            "repeat_index": 1,
            "run_id": "run-legacy-source",
            "status": "completed",
            "score": 80,
            "asr": 100,
            "conversation": [
                {"kind": "single_shot", "response": "legacy saved response"}
            ],
        }
    ]
    store._experiments[state.experiment_id] = state
    judge = SequencedRejudgeClient(
        [
            '{"score": 4, "rationale": "legacy evidence"}',
            '{"score": 5, "rationale": "legacy evidence again"}',
        ]
    )
    monkeypatch.setattr(
        store,
        "_build_chat_client",
        lambda *args, **kwargs: pytest.fail("target builder was invoked"),
    )
    monkeypatch.setattr(
        store,
        "_build_attacker_client",
        lambda *args, **kwargs: pytest.fail("attacker builder was invoked"),
    )
    monkeypatch.setattr(
        store,
        "_build_judge_client",
        lambda config, *, target_client, attacker_client: judge,
    )

    _run_owned_rejudge(store, state, _rejudge_config())
    first_subject_id = state.cell_results[0]["rejudge_results"][0][
        "target_subject_message_id"
    ]
    _run_owned_rejudge(store, state, _rejudge_config())

    first_event = runtime.events[0]
    assert first_event.correlation.run_id != "run-legacy-source"
    assert first_event.payload["source_run_id"] == "run-legacy-source"
    assert first_event.payload["attempt_identity_kind"] == "legacy_fallback"
    assert first_event.correlation.attempt_id.startswith("legacy-attempt:")
    assert state.cell_results[0]["rejudge_results"][1]["target_subject_message_id"] == first_subject_id
