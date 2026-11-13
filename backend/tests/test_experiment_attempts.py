import os
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from app.experiment_plan_integrity import content_hash, technique_config_id
from app.techniques.contracts.execution import (
    ExecutionStatus,
    FailureCategory,
    ResultDisposition,
)


def _config(*, experiment_id: str = "exp-attempts", technique_config_id: str = "cfg-a"):
    from app.persistence.experiment_attempts import TechniqueConfigRecord

    return TechniqueConfigRecord(
        technique_config_id=technique_config_id,
        experiment_id=experiment_id,
        technique_id="direct_prompt_injection",
        technique_version="1.0.0",
        implementation_id="builtin",
        spec_hash="spec-a",
        parameters={"temperature": 0},
        resolved_spec={"model": "target-a"},
    )


def _cell(*, experiment_id: str = "exp-attempts", cell_id: str | None = "cell-a", prompt_id: str = "prompt-a"):
    from app.persistence.experiment_attempts import CellRecord

    return CellRecord(
        cell_id=cell_id,
        experiment_id=experiment_id,
        technique_config_id="cfg-a",
        technique_key="direct_prompt_injection",
        prompt_id=prompt_id,
        repeat_index=1,
    )


def _canonical_config_record():
    from app.persistence.experiment_attempts import TechniqueConfigRecord

    experiment_id = "exp-canonical-record"
    resolved_spec = {
        "technique_id": "probe",
        "technique_version": "1.2.3",
        "implementation_id": "probe-python",
        "implementation_kind": "python",
        "contract_version": "7",
        "manifest_hash": "a" * 64,
        "asset_hashes": {},
        "preset": None,
        "parameters": {"threshold": 2},
        "model_roles": {
            "target": {
                "profile_id": "target-profile",
                "provider": "openai",
                "model": "target-model",
                "transport": "direct",
            }
        },
        "judge_policies": {},
        "effective_limits": {"max_rounds": 5},
        "runtime_config": {"max_rounds": 5},
    }
    return TechniqueConfigRecord(
        technique_config_id=technique_config_id(experiment_id, resolved_spec),
        experiment_id=experiment_id,
        technique_id=resolved_spec["technique_id"],
        technique_version=resolved_spec["technique_version"],
        implementation_id=resolved_spec["implementation_id"],
        preset=resolved_spec["preset"],
        parameters=resolved_spec["parameters"],
        resolved_spec=resolved_spec,
        spec_hash=content_hash(resolved_spec),
    )


def _seed(repository, *, cell_id: str | None = "cell-a"):
    repository.upsert_technique_config(_config())
    return repository.create_cell(_cell(cell_id=cell_id))


def _set_owned_execution(
    repository,
    *,
    generation: int,
    owner_id: str,
    status: str = "running",
) -> None:
    if hasattr(repository, "_dsn"):
        import psycopg

        with psycopg.connect(repository._dsn, autocommit=True) as connection:
            connection.execute(
                "UPDATE experiments SET status = %s, execution_generation = %s, "
                "execution_owner_id = %s WHERE experiment_id = 'exp-attempts'",
                (status, generation, owner_id),
            )
        return
    authority = repository._execution_authority
    with authority.lock:
        authority.experiments["exp-attempts"] = {
            "experiment_id": "exp-attempts",
            "status": status,
            "execution_generation": generation,
            "execution_owner_id": owner_id,
        }


@pytest.mark.parametrize("execution_generation", [True, -1, 1.5])
def test_attempt_record_rejects_invalid_execution_generation(
    execution_generation,
) -> None:
    from app.persistence.experiment_attempts import AttemptRecord

    with pytest.raises(ValueError, match="execution_generation"):
        AttemptRecord(
            attempt_id="attempt-1",
            cell_id="cell-1",
            attempt_number=1,
            parent_attempt_id=None,
            retry_kind="initial",
            spec_hash="spec-1",
            execution_status=ExecutionStatus.QUEUED,
            result_disposition=ResultDisposition.NOT_TESTED,
            execution_generation=execution_generation,
        )


@pytest.mark.parametrize("execution_generation", [True, -1, 1.5])
def test_retry_batch_record_rejects_invalid_execution_generation(
    execution_generation,
) -> None:
    from app.persistence.experiment_attempts import RetryBatchRecord

    with pytest.raises(ValueError, match="execution_generation"):
        RetryBatchRecord(
            retry_batch_id="batch-1",
            experiment_id="exp-1",
            request_id=None,
            action="restart",
            completed_policy="exclude",
            requested_by="operator",
            cell_ids=(),
            execution_generation=execution_generation,
        )


def test_cell_record_rejects_boolean_repeat_index() -> None:
    with pytest.raises(ValueError, match="repeat_index must be a positive integer"):
        replace(_cell(), repeat_index=True)


def test_technique_config_record_rejects_credential_shaped_values() -> None:
    with pytest.raises(ValueError, match="credential material"):
        replace(
            _config(),
            parameters={"note": "Bearer sk-supersecret123"},
        )


def test_technique_config_record_accepts_exact_resolved_spec_projections() -> None:
    record = _canonical_config_record()

    assert record.parameters == {"threshold": 2}


def test_technique_config_record_rejects_top_level_parameter_drift() -> None:
    with pytest.raises(ValueError, match="resolved specification"):
        replace(
            _canonical_config_record(),
            parameters={"threshold": 99},
        )


def test_technique_config_record_rejects_scalar_preset_credential() -> None:
    with pytest.raises(ValueError, match="credential material"):
        replace(
            _config(),
            preset="Bearer sk-supersecret123",
        )


def test_technique_config_record_accepts_explicit_legacy_partial_spec_with_etc_id() -> None:
    from app.persistence.experiment_attempts import TechniqueConfigRecord

    record = TechniqueConfigRecord(
        technique_config_id="etc-low",
        experiment_id="exp-legacy",
        technique_id="legacy-technique",
        technique_version="0.2.0",
        implementation_id="legacy",
        parameters={"ratio": 0.5},
        resolved_spec={"parameters": {"ratio": 0.5}},
        spec_hash="spec-low",
    )

    assert record.technique_config_id == "etc-low"


def test_technique_config_record_rejects_incomplete_deterministic_identity() -> None:
    with pytest.raises(ValueError, match="resolved specification is missing canonical fields"):
        replace(
            _config(),
            technique_config_id="etc-incomplete",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("technique_config_id", "etc-forged"),
        ("spec_hash", "f" * 64),
        ("technique_version", "9.9.9"),
    ],
)
def test_technique_config_record_rejects_forged_identity_projections(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="resolved specification|canonical content"):
        replace(_canonical_config_record(), **{field: value})


@pytest.fixture(params=["memory", "postgres"])
def repository(request):
    if request.param == "memory":
        from app.persistence.experiment_attempts import InMemoryExperimentAttemptPersistence

        return InMemoryExperimentAttemptPersistence()
    if not os.environ.get("TASK14_DATABASE_URL"):
        pytest.skip("requires disposable Task 14 PostgreSQL")
    import psycopg

    from app.persistence.experiment_attempts import PostgreSQLExperimentAttemptPersistence

    dsn = os.environ["TASK14_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments CASCADE"
        )
        connection.execute("INSERT INTO experiments (experiment_id, name) VALUES ('exp-attempts', 'attempt tests')")
    return PostgreSQLExperimentAttemptPersistence(dsn)


def test_attempt_execution_generation_survives_repository_round_trip(repository) -> None:
    cell = _seed(repository)
    _set_owned_execution(
        repository,
        generation=3,
        owner_id="owner-generation-3",
    )

    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=3,
        execution_owner_id="owner-generation-3",
    )

    assert attempt.execution_generation == 3
    assert repository.get_attempt(attempt.attempt_id).execution_generation == 3
    assert repository.list_attempts(cell.cell_id)[0].execution_generation == 3


def test_active_attempt_query_is_scoped_to_current_execution_generation(repository) -> None:
    cell = _seed(repository)
    _set_owned_execution(
        repository,
        generation=3,
        owner_id="owner-generation-3",
    )
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=3,
        execution_owner_id="owner-generation-3",
    )

    assert repository.has_active_attempts("exp-attempts", execution_generation=3)
    assert not repository.has_active_attempts("exp-attempts", execution_generation=2)

    repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.PROVIDER,
    )
    assert not repository.has_active_attempts("exp-attempts", execution_generation=3)


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_migration_generation_repairs_existing_volume_without_rewriting_evidence() -> None:
    import psycopg

    from app.persistence import (
        PostgreSQLExperimentAttemptPersistence,
        PostgreSQLRunPersistence,
    )

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
            "VALUES ('exp-attempts', 'migration fixture', 'completed')"
        )

    attempts = PostgreSQLExperimentAttemptPersistence(dsn)
    cell = _seed(attempts)
    selected = attempts.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )
    attempts.complete_attempt(
        selected.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
        run_id="run-selected-before-generation",
    )
    requested = attempts.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="migration-test",
        request_id="requested-before-generation",
    )
    original_selected_attempt_id = attempts.get_cell(cell.cell_id).selected_attempt_id

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            "ALTER TABLE experiments DROP COLUMN execution_generation, "
            "DROP COLUMN execution_owner_id"
        )
        connection.execute(
            "ALTER TABLE experiment_cell_attempts DROP COLUMN execution_generation"
        )
        connection.execute(
            "ALTER TABLE retry_batches DROP COLUMN execution_generation"
        )
        PostgreSQLRunPersistence(dsn).ensure_schema(connection)
        experiment_generation, experiment_owner_id = connection.execute(
            "SELECT execution_generation, execution_owner_id FROM experiments "
            "WHERE experiment_id = 'exp-attempts'"
        ).fetchone()
        historical_attempt_generation = connection.execute(
            "SELECT execution_generation FROM experiment_cell_attempts "
            "WHERE attempt_id = %s",
            (selected.attempt_id,),
        ).fetchone()[0]
        requested_batch_generation = connection.execute(
            "SELECT execution_generation FROM retry_batches "
            "WHERE retry_batch_id = %s",
            (requested.retry_batch_id,),
        ).fetchone()[0]
        selected_attempt_id = connection.execute(
            "SELECT selected_attempt_id FROM experiment_cells WHERE cell_id = %s",
            (cell.cell_id,),
        ).fetchone()[0]

    assert experiment_generation == 0
    assert experiment_owner_id == ""
    assert historical_attempt_generation == 0
    assert requested_batch_generation is None
    assert selected_attempt_id == original_selected_attempt_id


def test_bind_retry_batch_generation_binds_once_and_accepts_exact_replay(repository) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
        request_id="generation-bind",
    )
    _set_owned_execution(
        repository,
        generation=4,
        owner_id=batch.retry_batch_id,
    )

    first = repository.bind_retry_batch_generation(batch.retry_batch_id, 4)
    replay = repository.bind_retry_batch_generation(batch.retry_batch_id, 4)

    assert batch.execution_generation is None
    assert first.execution_generation == 4
    assert replay == first
    assert repository.get_retry_batch(batch.retry_batch_id).execution_generation == 4
    with pytest.raises(ValueError, match="different execution generation"):
        repository.bind_retry_batch_generation(batch.retry_batch_id, 5)


@pytest.mark.parametrize("execution_generation", [True, -1, 0, 1.5])
def test_bind_retry_batch_generation_rejects_invalid_values(
    repository,
    execution_generation,
) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )

    with pytest.raises(ValueError, match="execution_generation"):
        repository.bind_retry_batch_generation(
            batch.retry_batch_id,
            execution_generation,
        )

    assert repository.get_retry_batch(batch.retry_batch_id).execution_generation is None


def test_bound_retry_batch_replays_request_and_attempt_generation(repository) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
        request_id="bound-request-replay",
    )
    _set_owned_execution(
        repository,
        generation=7,
        owner_id=batch.retry_batch_id,
    )
    repository.bind_retry_batch_generation(batch.retry_batch_id, 7)

    replay = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
        request_id="bound-request-replay",
    )
    attempt = repository.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=7,
    )
    attempt_replay = repository.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=7,
    )

    assert replay.retry_batch_id == batch.retry_batch_id
    assert replay.execution_generation == 7
    assert attempt.execution_generation == 7
    assert attempt_replay == attempt


def test_retry_batch_rejects_attempt_from_different_execution_generation(
    repository,
) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )
    _set_owned_execution(
        repository,
        generation=3,
        owner_id=batch.retry_batch_id,
    )
    with pytest.raises(ValueError, match="not durably bound"):
        repository.create_and_bind_retry_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            retry_kind="restart",
            spec_hash="spec-a",
            execution_generation=3,
        )
    assert repository.list_attempts(cell.cell_id) == ()

    _set_owned_execution(
        repository,
        generation=3,
        owner_id="stale-owner",
    )
    stale_attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=3,
        execution_owner_id="stale-owner",
    )
    _set_owned_execution(
        repository,
        generation=4,
        owner_id=batch.retry_batch_id,
    )
    repository.bind_retry_batch_generation(batch.retry_batch_id, 4)

    with pytest.raises(ValueError, match="execution generation"):
        repository.bind_retry_batch_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            stale_attempt.attempt_id,
        )

    assert repository.get_retry_batch(batch.retry_batch_id).members[0].attempt_id is None


def test_atomic_retry_attempt_rejects_generation_different_from_batch(
    repository,
) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )
    _set_owned_execution(
        repository,
        generation=4,
        owner_id=batch.retry_batch_id,
    )
    repository.bind_retry_batch_generation(batch.retry_batch_id, 4)

    with pytest.raises(ValueError, match="execution generation"):
        repository.create_and_bind_retry_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            retry_kind="restart",
            spec_hash="spec-a",
            execution_generation=3,
        )

    assert repository.list_attempts(cell.cell_id) == ()


def test_unbound_retry_batch_rejects_attempt_preassociation(
    repository,
) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )
    _set_owned_execution(
        repository,
        generation=3,
        owner_id="stale-owner",
    )
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=3,
        execution_owner_id="stale-owner",
    )
    with pytest.raises(ValueError, match="not durably bound"):
        repository.bind_retry_batch_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            attempt.attempt_id,
        )

    rejected = repository.get_retry_batch(batch.retry_batch_id)
    assert rejected.execution_generation is None
    assert rejected.members[0].attempt_id is None
    _set_owned_execution(
        repository,
        generation=4,
        owner_id=batch.retry_batch_id,
    )
    bound = repository.bind_retry_batch_generation(batch.retry_batch_id, 4)
    assert bound.execution_generation == 4


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_generation_binding_serializes_with_attempt_association() -> None:
    import psycopg

    from app.persistence.experiment_attempts import (
        PostgreSQLExperimentAttemptPersistence,
    )

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
            "INSERT INTO experiments (experiment_id, name) "
            "VALUES ('exp-attempts', 'attempt tests')"
        )

    repository = PostgreSQLExperimentAttemptPersistence(dsn)
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )
    _set_owned_execution(
        repository,
        generation=3,
        owner_id=batch.retry_batch_id,
    )
    repository.bind_retry_batch_generation(batch.retry_batch_id, 3)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=3,
        execution_owner_id=batch.retry_batch_id,
    )
    association_boundary = threading.Event()
    binding_backend_ready = threading.Event()
    release_association = threading.Event()
    association_backend_pids: list[int] = []
    binding_backend_pids: list[int] = []
    association_errors: list[BaseException] = []
    binding_errors: list[BaseException] = []

    class PausingConnection:
        def __init__(self, connection) -> None:
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, query, params=None):
            result = self.connection.execute(query, params)
            sql = " ".join(str(query).split())
            old_unlocked_read = sql.startswith(
                "SELECT execution_generation FROM retry_batches WHERE retry_batch_id"
            )
            new_locked_read = (
                sql.startswith("SELECT experiment_id, execution_generation")
                and "FROM retry_batches" in sql
                and "FOR UPDATE" in sql
            )
            if old_unlocked_read or new_locked_read:
                association_boundary.set()
                if not release_association.wait(timeout=10):
                    raise TimeoutError("attempt association was not released")
            return result

    class PausingPersistence(PostgreSQLExperimentAttemptPersistence):
        def _connection(self):
            self._ensure_schema()
            connection = psycopg.connect(self._dsn, autocommit=True)
            association_backend_pids.append(connection.info.backend_pid)
            return PausingConnection(connection)

    class ObservedPersistence(PostgreSQLExperimentAttemptPersistence):
        def _connection(self):
            self._ensure_schema()
            connection = psycopg.connect(self._dsn, autocommit=True)
            binding_backend_pids.append(connection.info.backend_pid)
            binding_backend_ready.set()
            return connection

    def wait_for_binding_lock_wait() -> tuple[tuple[int, ...], str | None]:
        deadline = time.monotonic() + 5
        last_snapshot: tuple[tuple[int, ...], str | None] = ((), None)
        with psycopg.connect(dsn, autocommit=True) as observer:
            while time.monotonic() < deadline:
                row = observer.execute(
                    "SELECT pg_blocking_pids(pid), wait_event_type "
                    "FROM pg_stat_activity WHERE pid = %s",
                    (binding_backend_pids[0],),
                ).fetchone()
                if row is not None:
                    last_snapshot = (tuple(row[0]), row[1])
                    if (
                        association_backend_pids[0] in last_snapshot[0]
                        and last_snapshot[1] == "Lock"
                    ):
                        return last_snapshot
                threading.Event().wait(timeout=0.01)
        return last_snapshot

    associating = PausingPersistence(dsn)
    binding = ObservedPersistence(dsn)
    associating._ensure_schema()
    binding._ensure_schema()

    def associate_attempt() -> None:
        try:
            associating.bind_retry_batch_attempt(
                batch.retry_batch_id,
                cell.cell_id,
                attempt.attempt_id,
            )
        except BaseException as error:  # pragma: no cover - relayed from worker
            association_errors.append(error)

    def bind_generation() -> None:
        try:
            binding.bind_retry_batch_generation(batch.retry_batch_id, 3)
        except BaseException as error:  # pragma: no cover - relayed from worker
            binding_errors.append(error)

    association_thread = threading.Thread(target=associate_attempt)
    binding_thread = threading.Thread(target=bind_generation)
    association_thread.start()
    assert association_boundary.wait(timeout=5)
    binding_thread.start()
    try:
        assert binding_backend_ready.wait(timeout=5)
        assert len(association_backend_pids) == 1
        assert len(binding_backend_pids) == 1
        binding_blockers, binding_wait_event_type = wait_for_binding_lock_wait()
        assert association_backend_pids[0] in binding_blockers
        assert binding_wait_event_type == "Lock"
    finally:
        release_association.set()
        association_thread.join(timeout=5)
        binding_thread.join(timeout=5)

    assert not association_thread.is_alive()
    assert not binding_thread.is_alive()
    assert association_errors == []
    assert binding_errors == []
    restored = binding.get_retry_batch(batch.retry_batch_id)
    assert restored.execution_generation == 3
    assert restored.members[0].attempt_id == attempt.attempt_id


def test_successful_retry_promotes_latest_tested_attempt_not_highest_score(repository) -> None:
    cell = _seed(repository)
    first = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(
        first.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    second = repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-a")
    repository.complete_attempt(
        second.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    completed = repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-a")
    repository.complete_attempt(
        completed.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
    )

    assert repository.get_cell(cell.cell_id).selected_attempt_id == completed.attempt_id
    assert [attempt.attempt_number for attempt in repository.list_attempts(cell.cell_id)] == [1, 2, 3]


def test_latest_failure_is_selected_until_a_tested_completion_arrives(repository) -> None:
    cell = _seed(repository)
    first = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(first.attempt_id, execution_status=ExecutionStatus.FAILED, result_disposition=ResultDisposition.INCONCLUSIVE)
    second = repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-a")
    repository.complete_attempt(second.attempt_id, execution_status=ExecutionStatus.TIMED_OUT, result_disposition=ResultDisposition.INCONCLUSIVE)

    assert repository.get_cell(cell.cell_id).selected_attempt_id == second.attempt_id


def test_diagnostic_reruns_never_displace_a_valid_selected_result(repository) -> None:
    cell = _seed(repository)
    completed = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(completed.attempt_id, execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED, score=0)
    diagnostic = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        promotion_policy="diagnostic_only",
        spec_hash="spec-a",
    )
    repository.complete_attempt(diagnostic.attempt_id, execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED, score=99)
    failed_diagnostic = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        promotion_policy="diagnostic_only",
        spec_hash="spec-a",
    )
    repository.complete_attempt(failed_diagnostic.attempt_id, execution_status=ExecutionStatus.FAILED, result_disposition=ResultDisposition.INCONCLUSIVE)

    assert repository.get_cell(cell.cell_id).selected_attempt_id == completed.attempt_id
    assert repository.retryable_cells("exp-attempts") == ()


def test_attempt_history_is_immutable_and_spec_hash_is_pinned(repository) -> None:
    cell = _seed(repository)
    initial = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(initial.attempt_id, execution_status=ExecutionStatus.FAILED, result_disposition=ResultDisposition.INCONCLUSIVE)

    with pytest.raises(ValueError, match="spec hash"):
        repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-b")
    restarted = repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-a")

    assert restarted.parent_attempt_id == initial.attempt_id
    assert repository.get_attempt(initial.attempt_id).execution_status is ExecutionStatus.FAILED


def test_claim_heartbeat_expiry_and_recovery_preserve_one_active_attempt(repository) -> None:
    cell = _seed(repository)
    first = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    assert repository.claim_attempt(first.attempt_id, worker_id="worker-a", lease_seconds=60)
    assert not repository.heartbeat_attempt(first.attempt_id, worker_id="worker-b", lease_seconds=60)
    assert repository.heartbeat_attempt(first.attempt_id, worker_id="worker-a", lease_seconds=0)
    assert not repository.heartbeat_attempt(first.attempt_id, worker_id="worker-a", lease_seconds=60)
    second = repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-a")
    assert repository.claim_attempt(second.attempt_id, worker_id="worker-b", lease_seconds=60)
    recovered = repository.get_attempt(first.attempt_id)
    assert recovered.execution_status is ExecutionStatus.FAILED
    assert recovered.result_disposition is ResultDisposition.INCONCLUSIVE
    assert recovered.failure_category is FailureCategory.WORKER_LOST
    assert repository.get_cell(cell.cell_id).selected_attempt_id == first.attempt_id
    assert repository.get_cell(cell.cell_id).active_attempt_id == second.attempt_id


def test_late_completion_cannot_clear_or_promote_over_newer_active_attempt(repository) -> None:
    cell = _seed(repository)
    obsolete = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    assert repository.claim_attempt(obsolete.attempt_id, worker_id="worker-a", lease_seconds=0)
    replacement = repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-a")
    assert repository.claim_attempt(replacement.attempt_id, worker_id="worker-b", lease_seconds=60)

    late = repository.complete_attempt(obsolete.attempt_id, execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED, score=4)

    current = repository.get_cell(cell.cell_id)
    assert late.execution_status is ExecutionStatus.COMPLETED
    assert late.result_disposition is ResultDisposition.TESTED
    assert late.late_completion is True
    assert current.active_attempt_id == replacement.attempt_id
    assert current.selected_attempt_id == obsolete.attempt_id
    assert current.latest_attempt_id == replacement.attempt_id


def test_completion_requires_tested_score_and_clears_only_owning_lease(repository) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    assert repository.claim_attempt(attempt.attempt_id, worker_id="worker-a", lease_seconds=60)
    with pytest.raises(ValueError, match="score"):
        repository.complete_attempt(attempt.attempt_id, execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED)
    with pytest.raises(ValueError, match="terminal"):
        repository.complete_attempt(attempt.attempt_id, execution_status=ExecutionStatus.RUNNING, result_disposition=ResultDisposition.NOT_TESTED)
    completed = repository.complete_attempt(attempt.attempt_id, execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED, score=0)

    assert completed.score == 0
    assert completed.worker_id is None and completed.lease_expires_at is None
    assert repository.get_cell(cell.cell_id).active_attempt_id is None


def test_owned_attempt_completion_reports_that_the_worker_wrote_terminal_evidence(
    repository,
) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="worker-a",
        lease_seconds=60,
    )

    outcome = repository.complete_attempt_if_owned(
        attempt.attempt_id,
        worker_id="worker-a",
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.INTERNAL,
    )

    assert outcome.completed_by_worker is True
    assert outcome.attempt.execution_status is ExecutionStatus.FAILED
    assert outcome.attempt.failure_category is FailureCategory.INTERNAL
    assert repository.get_cell(cell.cell_id).active_attempt_id is None


def test_stale_generation_attempt_completion_is_late_diagnostic_only() -> None:
    from app.persistence.experiment_attempts import (
        InMemoryExperimentAttemptPersistence,
    )
    from app.persistence.runs import (
        InMemoryExperimentExecutionAuthority,
        NoopRunPersistence,
    )

    authority = InMemoryExperimentExecutionAuthority()
    runs = NoopRunPersistence(execution_authority=authority)
    repository = InMemoryExperimentAttemptPersistence(
        execution_authority=authority,
    )
    runs.save_experiment(
        experiment={
            "experiment_id": "exp-attempts",
            "status": "running",
            "execution_generation": 1,
            "execution_owner_id": "initial:exp-attempts",
        }
    )
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        lease_seconds=60,
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )

    runs.save_experiment(
        experiment={
            "experiment_id": "exp-attempts",
            "status": "running",
            "execution_generation": 2,
            "execution_owner_id": "retry-batch-generation-2",
        }
    )
    outcome = repository.complete_attempt_if_owned(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=9,
        run_id="run-stale-generation-1",
    )

    assert outcome.completed_by_worker is False
    assert outcome.attempt.late_completion is True
    assert outcome.attempt.score == 9
    assert repository.get_cell(cell.cell_id).selected_attempt_id is None
    assert repository.get_cell(cell.cell_id).active_attempt_id is None


def test_terminal_reconciliation_and_attempt_insertion_have_one_valid_winner() -> None:
    from app.persistence.experiment_attempts import (
        InMemoryExperimentAttemptPersistence,
    )
    from app.persistence.runs import (
        InMemoryExperimentExecutionAuthority,
        NoopRunPersistence,
    )

    insertion_holds_authority = threading.Event()
    release_insertion = threading.Event()

    class BlockingAttemptPersistence(InMemoryExperimentAttemptPersistence):
        def _require_owned_execution_locked(self, *args, **kwargs) -> None:
            super()._require_owned_execution_locked(*args, **kwargs)
            insertion_holds_authority.set()
            assert release_insertion.wait(timeout=5)

    authority = InMemoryExperimentExecutionAuthority()
    runs = NoopRunPersistence(execution_authority=authority)
    repository = BlockingAttemptPersistence(execution_authority=authority)
    runs.save_experiment(
        experiment={
            "experiment_id": "exp-attempts",
            "status": "running",
            "execution_generation": 1,
            "execution_owner_id": "initial:exp-attempts",
        }
    )
    cell = _seed(repository)
    attempts: list[object] = []
    terminal_results: list[bool] = []
    terminal_started = threading.Event()
    terminal_finished = threading.Event()

    def insert_attempt() -> None:
        attempts.append(
            repository.create_attempt(
                cell.cell_id,
                retry_kind="initial",
                spec_hash="spec-a",
                execution_generation=1,
                execution_owner_id="initial:exp-attempts",
            )
        )

    def terminalize() -> None:
        terminal_started.set()
        terminal_results.append(
            runs.reconcile_owned_experiment_terminal_status(
                "exp-attempts",
                generation=1,
                owner_id="initial:exp-attempts",
                status="completed",
            )
        )
        terminal_finished.set()

    insertion = threading.Thread(target=insert_attempt)
    terminal = threading.Thread(target=terminalize)
    insertion.start()
    assert insertion_holds_authority.wait(timeout=5)
    terminal.start()
    assert terminal_started.wait(timeout=5)
    assert not terminal_finished.is_set()
    release_insertion.set()
    insertion.join(timeout=5)
    terminal.join(timeout=5)

    assert len(attempts) == 1
    assert terminal_results == [False]
    assert runs.load_all_experiments()[0]["status"] == "running"


def test_retry_attempt_association_rejects_stale_experiment_owner() -> None:
    from app.persistence.experiment_attempts import (
        InMemoryExperimentAttemptPersistence,
    )
    from app.persistence.runs import (
        InMemoryExperimentExecutionAuthority,
        NoopRunPersistence,
    )

    authority = InMemoryExperimentExecutionAuthority()
    runs = NoopRunPersistence(execution_authority=authority)
    repository = InMemoryExperimentAttemptPersistence(
        execution_authority=authority,
    )
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )
    runs.save_experiment(
        experiment={
            "experiment_id": "exp-attempts",
            "status": "running",
            "execution_generation": 1,
            "execution_owner_id": batch.retry_batch_id,
        }
    )
    repository.bind_retry_batch_generation(batch.retry_batch_id, 1)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id=batch.retry_batch_id,
    )
    runs.save_experiment(
        experiment={
            "experiment_id": "exp-attempts",
            "status": "running",
            "execution_generation": 2,
            "execution_owner_id": "new-owner",
        }
    )

    with pytest.raises(ValueError, match="ownership"):
        repository.bind_retry_batch_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            attempt.attempt_id,
        )

    assert repository.get_retry_batch(batch.retry_batch_id).members[0].attempt_id is None


def test_stale_attempt_owner_cannot_abandon_a_newer_execution() -> None:
    from app.persistence.experiment_attempts import (
        InMemoryExperimentAttemptPersistence,
    )
    from app.persistence.runs import (
        InMemoryExperimentExecutionAuthority,
        NoopRunPersistence,
    )

    authority = InMemoryExperimentExecutionAuthority()
    runs = NoopRunPersistence(execution_authority=authority)
    repository = InMemoryExperimentAttemptPersistence(
        execution_authority=authority,
    )
    runs.save_experiment(
        experiment={
            "experiment_id": "exp-attempts",
            "status": "running",
            "execution_generation": 1,
            "execution_owner_id": "initial:exp-attempts",
        }
    )
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        lease_seconds=60,
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    runs.save_experiment(
        experiment={
            "experiment_id": "exp-attempts",
            "status": "running",
            "execution_generation": 2,
            "execution_owner_id": "new-owner",
        }
    )

    assert not repository.abandon_attempt_claim(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    assert repository.get_attempt(attempt.attempt_id).execution_status is ExecutionStatus.RUNNING


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_terminal_reconciliation_waits_for_owned_attempt_admission() -> None:
    import psycopg

    from app.persistence import (
        PostgreSQLExperimentAttemptPersistence,
        PostgreSQLRunPersistence,
    )

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
            "VALUES ('exp-attempts', 'attempt tests', 'running', 1, "
            "'initial:exp-attempts')"
        )

    base_repository = PostgreSQLExperimentAttemptPersistence(dsn)
    cell = _seed(base_repository)
    attempt_holds_experiment = threading.Event()
    release_attempt = threading.Event()
    terminal_reached_experiment = threading.Event()
    terminal_finished = threading.Event()

    class AttemptTransaction:
        def __init__(self) -> None:
            self.connection = psycopg.connect(dsn, autocommit=True)
            self.transaction = self.connection.transaction()

        def __enter__(self):
            self.transaction.__enter__()
            connection = self.connection

            class Proxy:
                def execute(self, query, params=None):
                    result = connection.execute(query, params)
                    compact = " ".join(str(query).split())
                    if (
                        "FROM experiments WHERE experiment_id = %s FOR UPDATE"
                        in compact
                    ):
                        attempt_holds_experiment.set()
                        assert release_attempt.wait(timeout=5)
                    return result

                def __getattr__(self, name):
                    return getattr(connection, name)

            return Proxy()

        def __exit__(self, *args):
            try:
                return self.transaction.__exit__(*args)
            finally:
                self.connection.close()

    class HookedAttemptRepository(PostgreSQLExperimentAttemptPersistence):
        def _transaction(self):
            return AttemptTransaction()

    class TerminalConnection:
        def __init__(self) -> None:
            self.connection = psycopg.connect(dsn, autocommit=True)

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def transaction(self):
            return self.connection.transaction()

        def execute(self, query, params=None):
            compact = " ".join(str(query).split())
            if (
                "FROM experiments" in compact
                and "WHERE experiment_id = %s" in compact
                and "FOR UPDATE" in compact
            ):
                terminal_reached_experiment.set()
            return self.connection.execute(query, params)

    class HookedRunPersistence(PostgreSQLRunPersistence):
        def _connect(self):
            return TerminalConnection()

        def ensure_schema(self, conn) -> None:
            return

    repository = HookedAttemptRepository(dsn)
    runs = HookedRunPersistence(dsn)
    attempts: list[object] = []
    terminal_results: list[bool] = []
    failures: list[BaseException] = []

    def insert_attempt() -> None:
        try:
            attempts.append(
                repository.create_attempt(
                    cell.cell_id,
                    retry_kind="initial",
                    spec_hash="spec-a",
                    execution_generation=1,
                    execution_owner_id="initial:exp-attempts",
                )
            )
        except BaseException as exc:
            failures.append(exc)

    def terminalize() -> None:
        try:
            terminal_results.append(
                runs.reconcile_owned_experiment_terminal_status(
                    "exp-attempts",
                    generation=1,
                    owner_id="initial:exp-attempts",
                    status="completed",
                )
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            terminal_finished.set()

    insertion = threading.Thread(target=insert_attempt)
    terminal = threading.Thread(target=terminalize)
    insertion.start()
    assert attempt_holds_experiment.wait(timeout=5)
    terminal.start()
    assert terminal_reached_experiment.wait(timeout=5)
    assert not terminal_finished.is_set()
    release_attempt.set()
    insertion.join(timeout=5)
    terminal.join(timeout=5)

    assert not insertion.is_alive()
    assert not terminal.is_alive()
    assert failures == []
    assert len(attempts) == 1
    assert terminal_results == [False]
    with psycopg.connect(dsn, autocommit=True) as connection:
        assert connection.execute(
            "SELECT status FROM experiments WHERE experiment_id = 'exp-attempts'"
        ).fetchone()[0] == "running"


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_stale_generation_completion_cannot_promote_cell() -> None:
    import psycopg

    from app.persistence import PostgreSQLExperimentAttemptPersistence

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
            "VALUES ('exp-attempts', 'attempt tests', 'running', 1, "
            "'initial:exp-attempts')"
        )
    repository = PostgreSQLExperimentAttemptPersistence(dsn)
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        lease_seconds=60,
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            "UPDATE experiments SET execution_generation = 2, "
            "execution_owner_id = 'retry-batch-generation-2', status = 'running' "
            "WHERE experiment_id = 'exp-attempts'"
        )

    outcome = repository.complete_attempt_if_owned(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=9,
        run_id="run-stale-generation-1",
    )

    assert outcome.completed_by_worker is False
    assert outcome.attempt.late_completion is True
    assert outcome.attempt.score == 9
    assert repository.get_cell(cell.cell_id).selected_attempt_id is None
    assert repository.get_cell(cell.cell_id).active_attempt_id is None


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_stale_generation_cell_projection_is_rejected() -> None:
    import psycopg

    from app.persistence import (
        PostgreSQLExperimentAttemptPersistence,
        PostgreSQLRunPersistence,
    )

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
            "VALUES ('exp-attempts', 'attempt tests', 'running', 1, "
            "'initial:exp-attempts')"
        )
    repository = PostgreSQLExperimentAttemptPersistence(dsn)
    runs = PostgreSQLRunPersistence(dsn)
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        lease_seconds=60,
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    completion = repository.complete_attempt_if_owned(
        attempt.attempt_id,
        worker_id="worker-generation-1",
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=9,
        run_id="run-generation-1",
    )
    assert completion.completed_by_worker
    assert runs.reconcile_owned_experiment_terminal_status(
        "exp-attempts",
        generation=1,
        owner_id="initial:exp-attempts",
        status="completed",
    )
    assert runs.begin_experiment_execution(
        "exp-attempts",
        owner_id="retry-generation-2",
        expected_generation=1,
        active_status="running",
    ) == 2

    accepted = runs.save_experiment_cell(
        experiment_id="exp-attempts",
        generation=1,
        owner_id="initial:exp-attempts",
        cell={
            "cell_id": cell.cell_id,
            "technique_config_id": cell.technique_config_id,
            "technique_key": cell.technique_key,
            "prompt_id": cell.prompt_id,
            "repeat_index": cell.repeat_index,
            "attempt_id": attempt.attempt_id,
            "selected_attempt_id": attempt.attempt_id,
            "run_id": "run-generation-1",
            "status": "completed",
            "score": 9,
            "asr": 100,
        },
    )

    assert accepted is False
    with psycopg.connect(dsn, autocommit=True) as connection:
        projection = connection.execute(
            "SELECT status, run_id FROM experiment_cells WHERE cell_id = %s",
            (cell.cell_id,),
        ).fetchone()
    assert projection == ("pending", None)


@pytest.mark.skipif(
    not os.environ.get("TASK14_DATABASE_URL"),
    reason="requires disposable Task 14 PostgreSQL",
)
def test_postgresql_retry_request_replay_locks_batch_before_cell() -> None:
    import psycopg

    from app.persistence import PostgreSQLExperimentAttemptPersistence

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
            "VALUES ('exp-attempts', 'attempt tests', 'completed', 1, "
            "'initial:exp-attempts')"
        )
    repository = PostgreSQLExperimentAttemptPersistence(dsn)
    cell = _seed(repository)
    first = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
        request_id="stable-request",
    )
    before_batch_lock = threading.Event()

    class HookedTransaction:
        def __init__(self) -> None:
            self.connection = psycopg.connect(dsn, autocommit=True)
            self.transaction = self.connection.transaction()

        def __enter__(self):
            self.transaction.__enter__()
            connection = self.connection

            class Proxy:
                def execute(self, query, params=None):
                    compact = " ".join(str(query).split())
                    if (
                        "FROM retry_batches" in compact
                        and "request_id = %s FOR UPDATE" in compact
                    ):
                        before_batch_lock.set()
                    return connection.execute(query, params)

                def __getattr__(self, name):
                    return getattr(connection, name)

            return Proxy()

        def __exit__(self, *args):
            try:
                return self.transaction.__exit__(*args)
            finally:
                self.connection.close()

    class HookedRepository(PostgreSQLExperimentAttemptPersistence):
        def _transaction(self):
            return HookedTransaction()

    replay_repository = HookedRepository(dsn)
    blocker = psycopg.connect(dsn, autocommit=False)
    replayed: list[object] = []
    failures: list[BaseException] = []

    def replay_request() -> None:
        try:
            replayed.append(
                replay_repository.create_retry_batch(
                    "exp-attempts",
                    action="restart",
                    cell_ids=(cell.cell_id,),
                    requested_by="operator",
                    request_id="stable-request",
                )
            )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=replay_request)
    try:
        blocker.execute(
            "SELECT retry_batch_id FROM retry_batches "
            "WHERE retry_batch_id = %s FOR UPDATE",
            (first.retry_batch_id,),
        )
        worker.start()
        assert before_batch_lock.wait(timeout=5)
        with psycopg.connect(dsn, autocommit=False) as checker:
            checker.execute(
                "SELECT cell_id FROM experiment_cells "
                "WHERE cell_id = %s FOR UPDATE NOWAIT",
                (cell.cell_id,),
            )
            checker.rollback()
    finally:
        blocker.rollback()
        blocker.close()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
    assert len(replayed) == 1
    assert replayed[0].retry_batch_id == first.retry_batch_id
    assert replayed[0].was_created is False


def test_owned_attempt_completion_identifies_exact_terminal_evidence_as_observed(
    repository,
) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="worker-a",
        lease_seconds=60,
    )
    winner = repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.INTERNAL,
    )

    outcome = repository.complete_attempt_if_owned(
        attempt.attempt_id,
        worker_id="worker-a",
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.INTERNAL,
    )

    assert outcome.completed_by_worker is False
    assert outcome.attempt == winner


def test_retryable_cells_use_selected_and_active_state(repository) -> None:
    cell = _seed(repository)
    assert repository.retryable_cells("exp-attempts") == (cell,)
    attempt = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    assert repository.claim_attempt(attempt.attempt_id, worker_id="worker-a", lease_seconds=60)
    assert repository.retryable_cells("exp-attempts") == ()
    repository.complete_attempt(attempt.attempt_id, execution_status=ExecutionStatus.FAILED, result_disposition=ResultDisposition.INCONCLUSIVE)
    assert repository.retryable_cells("exp-attempts") == (repository.get_cell(cell.cell_id),)
    recovered = repository.create_attempt(cell.cell_id, retry_kind="restart", spec_hash="spec-a")
    repository.complete_attempt(recovered.attempt_id, execution_status=ExecutionStatus.COMPLETED, result_disposition=ResultDisposition.TESTED, score=0)
    assert repository.retryable_cells("exp-attempts") == ()


def test_retryable_cells_does_not_recover_expired_active_attempt(repository) -> None:
    cell = _seed(repository)
    _set_owned_execution(
        repository,
        generation=1,
        owner_id="active-owner",
    )
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id="active-owner",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="expired-active-worker",
        lease_seconds=0,
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="active-owner",
    )

    assert repository.retryable_cells("exp-attempts") == ()
    preserved = repository.get_attempt(attempt.attempt_id)
    assert preserved.execution_status is ExecutionStatus.RUNNING
    assert preserved.failure_category is None
    assert repository.get_cell(cell.cell_id).active_attempt_id == attempt.attempt_id


def test_identity_upserts_are_idempotent_and_reject_conflicts(repository) -> None:
    config = _config()
    assert repository.upsert_technique_config(config) == config
    assert repository.upsert_technique_config(config) == config
    generated_config = repository.upsert_technique_config(replace(config, technique_config_id=None))
    assert generated_config.technique_config_id
    assert repository.upsert_technique_config(replace(config, technique_config_id=None)) == generated_config
    with pytest.raises(ValueError, match="different content"):
        repository.upsert_technique_config(replace(config, spec_hash="spec-other"))
    nested = repository.upsert_technique_config(
        replace(config, technique_config_id="cfg-nested", parameters={"nested": {"enabled": True}})
    )
    assert nested.parameters == {"nested": {"enabled": True}}
    cell = repository.create_cell(_cell(cell_id=None))
    assert cell.cell_id
    assert repository.create_cell(cell) == cell
    with pytest.raises(ValueError, match="different content"):
        repository.create_cell(replace(cell, prompt_id="different-prompt"))


def test_cell_creation_replays_planned_identity_after_attempt_state_changes(repository) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )

    replayed = repository.create_cell(_cell(cell_id=cell.cell_id))

    assert replayed.selected_attempt_id == attempt.attempt_id
    assert replayed.latest_attempt_id == attempt.attempt_id
    with pytest.raises(ValueError, match="attempt references"):
        repository.create_cell(
            replace(_cell(cell_id="cell-injected", prompt_id="prompt-injected"), selected_attempt_id="attempt-injected")
        )


def test_retry_batches_are_atomic_deduplicated_and_scoped_to_experiment(repository) -> None:
    first = _seed(repository)
    second = repository.create_cell(_cell(cell_id="cell-b", prompt_id="prompt-b"))
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(second.cell_id, first.cell_id, second.cell_id),
        requested_by="operator",
        request_id="ordered-request",
    )
    replay = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(second.cell_id, first.cell_id),
        requested_by="operator",
        request_id="ordered-request",
    )

    assert batch.cell_ids == (second.cell_id, first.cell_id)
    assert tuple(member.ordinal for member in batch.members) == (0, 1)
    assert replay.retry_batch_id == batch.retry_batch_id
    assert replay.cell_ids == batch.cell_ids
    with pytest.raises(ValueError, match="does not belong"):
        repository.create_retry_batch("other-experiment", action="restart", cell_ids=(first.cell_id,), requested_by="operator")


def test_retry_batch_replay_uses_immutable_selection_not_recomputed_snapshot(
    repository,
) -> None:
    first = _seed(repository)
    second = repository.create_cell(_cell(cell_id="cell-b", prompt_id="prompt-b"))
    selection = {
        "selection_mode": "automatic",
        "requested_cell_ids": [],
    }
    created = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(second.cell_id, first.cell_id),
        requested_by="operator",
        request_id="automatic-request",
        selection_metadata=selection,
    )

    replay = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        cell_ids=(),
        requested_by="operator",
        request_id="automatic-request",
        selection_metadata=selection,
    )

    assert replay.retry_batch_id == created.retry_batch_id
    assert replay.cell_ids == (second.cell_id, first.cell_id)


def test_retry_batch_request_id_is_idempotent_and_conflicts_fail_closed(repository) -> None:
    first = _seed(repository)
    original = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        completed_policy="exclude",
        cell_ids=(first.cell_id,),
        requested_by="operator",
        request_id="request-1",
        selection_metadata={"target_profile_id": "profile-a"},
    )
    replay = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        completed_policy="exclude",
        cell_ids=(first.cell_id,),
        requested_by="operator",
        request_id="request-1",
        selection_metadata={"target_profile_id": "profile-a"},
    )

    assert replay.retry_batch_id == original.retry_batch_id
    assert original.was_created is True
    assert replay.was_created is False
    assert repository.get_retry_batch(original.retry_batch_id).cell_ids == (first.cell_id,)

    with pytest.raises(ValueError, match="different retry request"):
        repository.create_retry_batch(
            "exp-attempts",
            action="resume",
            completed_policy="exclude",
            cell_ids=(first.cell_id,),
            requested_by="operator",
            request_id="request-1",
            selection_metadata={"target_profile_id": "profile-a"},
        )
    with pytest.raises(ValueError, match="must not contain secrets"):
        repository.create_retry_batch(
            "exp-attempts",
            action="restart",
            completed_policy="exclude",
            cell_ids=(first.cell_id,),
            requested_by="operator",
            selection_metadata={"api_key": "sk-secret-value"},
        )


@pytest.mark.parametrize("audit_field", ["request_id", "requested_by"])
def test_retry_batch_rejects_secret_shaped_audit_scalars_before_persistence(
    repository,
    audit_field,
) -> None:
    cell = _seed(repository)
    arguments = {
        "action": "restart",
        "cell_ids": (cell.cell_id,),
        "requested_by": "operator",
        "request_id": "request-safe",
    }
    arguments[audit_field] = "Bearer sk-supersecret123"

    with pytest.raises(ValueError, match="must not contain secrets"):
        repository.create_retry_batch("exp-attempts", **arguments)

    assert repository.get_retry_batch_by_request(
        "exp-attempts",
        str(arguments["request_id"]),
    ) is None


def test_batch_member_attempt_binding_and_terminal_state_are_idempotent(repository) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        completed_policy="replace_on_valid_completion",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )
    _set_owned_execution(
        repository,
        generation=1,
        owner_id=batch.retry_batch_id,
    )
    repository.bind_retry_batch_generation(batch.retry_batch_id, 1)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        promotion_policy="replace_on_valid_completion",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id=batch.retry_batch_id,
    )

    repository.bind_retry_batch_attempt(batch.retry_batch_id, cell.cell_id, attempt.attempt_id)
    repository.bind_retry_batch_attempt(batch.retry_batch_id, cell.cell_id, attempt.attempt_id)
    repository.complete_retry_batch_cell(
        batch.retry_batch_id,
        cell.cell_id,
        status="completed",
        error_category=None,
    )
    repository.complete_retry_batch_cell(
        batch.retry_batch_id,
        cell.cell_id,
        status="completed",
        error_category=None,
    )

    member = repository.get_retry_batch(batch.retry_batch_id).members[0]
    assert member.attempt_id == attempt.attempt_id
    assert member.status == "completed"
    with pytest.raises(ValueError, match="different terminal evidence"):
        repository.complete_retry_batch_cell(
            batch.retry_batch_id,
            cell.cell_id,
            status="failed",
            error_category="internal",
        )


def test_atomic_retry_attempt_creation_and_binding_replays_exactly(repository) -> None:
    cell = _seed(repository)
    batch = repository.create_retry_batch(
        "exp-attempts",
        action="restart",
        completed_policy="replace_on_valid_completion",
        cell_ids=(cell.cell_id,),
        requested_by="operator",
    )
    _set_owned_execution(
        repository,
        generation=1,
        owner_id=batch.retry_batch_id,
    )
    repository.bind_retry_batch_generation(batch.retry_batch_id, 1)

    created = repository.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=1,
        promotion_policy="replace_on_valid_completion",
        max_attempts=2,
        max_retry_attempts_per_experiment=1,
    )
    replay = repository.create_and_bind_retry_attempt(
        batch.retry_batch_id,
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
        execution_generation=1,
        promotion_policy="replace_on_valid_completion",
        max_attempts=2,
        max_retry_attempts_per_experiment=1,
    )

    member = repository.get_retry_batch(batch.retry_batch_id).members[0]
    assert replay == created
    assert member.attempt_id == created.attempt_id
    assert member.status == "queued"
    assert repository.list_attempts(cell.cell_id) == (created,)

    with pytest.raises(ValueError, match="different retry attempt"):
        repository.create_and_bind_retry_attempt(
            batch.retry_batch_id,
            cell.cell_id,
            retry_kind="resume",
            spec_hash="spec-a",
            execution_generation=1,
            promotion_policy="replace_on_valid_completion",
            max_attempts=2,
            max_retry_attempts_per_experiment=1,
        )


def test_expired_attempt_recovery_is_explicit_idempotent_and_allows_successor(repository) -> None:
    cell = _seed(repository)
    expired = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )
    assert repository.claim_attempt(
        expired.attempt_id,
        worker_id="expired-worker",
        lease_seconds=0,
    )

    recovered = repository.recover_expired_attempt(cell.cell_id)
    replay = repository.recover_expired_attempt(cell.cell_id)
    successor = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        spec_hash="spec-a",
    )

    assert recovered.active_attempt_id is None
    assert replay == recovered
    assert repository.get_attempt(expired.attempt_id).failure_category is FailureCategory.WORKER_LOST
    assert successor.parent_attempt_id == expired.attempt_id
    assert [attempt.attempt_number for attempt in repository.list_attempts(cell.cell_id)] == [1, 2]


def test_expired_attempt_recovery_rejects_stale_execution_owner(repository) -> None:
    cell = _seed(repository)
    _set_owned_execution(
        repository,
        generation=1,
        owner_id="initial:exp-attempts",
    )
    expired = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    assert repository.claim_attempt(
        expired.attempt_id,
        worker_id="expired-worker",
        lease_seconds=0,
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )
    _set_owned_execution(
        repository,
        generation=2,
        owner_id="retry-batch-generation-2",
    )

    recovered = repository.recover_expired_attempt(
        cell.cell_id,
        experiment_id="exp-attempts",
        execution_generation=1,
        execution_owner_id="initial:exp-attempts",
    )

    assert recovered.active_attempt_id == expired.attempt_id
    assert (
        repository.get_attempt(expired.attempt_id).execution_status
        is ExecutionStatus.RUNNING
    )


def test_attempt_limits_do_not_consume_attempt_number_or_allow_two_queued(repository) -> None:
    cell = _seed(repository)
    first = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        max_attempts=1,
    )
    with pytest.raises(ValueError, match="claimable attempt"):
        repository.create_attempt(
            cell.cell_id,
            retry_kind="restart",
            spec_hash="spec-a",
            max_attempts=2,
        )
    repository.complete_attempt(
        first.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    with pytest.raises(ValueError, match="max_attempts_per_cell"):
        repository.create_attempt(
            cell.cell_id,
            retry_kind="restart",
            spec_hash="spec-a",
            max_attempts=1,
        )

    assert [attempt.attempt_number for attempt in repository.list_attempts(cell.cell_id)] == [1]


def test_replace_policy_promotes_only_valid_completion_and_never_highest_score(repository) -> None:
    cell = _seed(repository)
    selected = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(
        selected.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=90,
    )
    failed = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        promotion_policy="replace_on_valid_completion",
        spec_hash="spec-a",
    )
    repository.complete_attempt(
        failed.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
    )
    assert repository.get_cell(cell.cell_id).selected_attempt_id == selected.attempt_id

    replacement = repository.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        promotion_policy="replace_on_valid_completion",
        spec_hash="spec-a",
    )
    repository.complete_attempt(
        replacement.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=10,
    )
    assert repository.get_cell(cell.cell_id).selected_attempt_id == replacement.attempt_id


def test_terminal_completion_replay_is_exact_and_conflicts_fail_closed(repository) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    completed = repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
        run_id="run-a",
    )

    assert repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=0,
        run_id="run-a",
    ) == completed
    with pytest.raises(ValueError, match="different terminal evidence"):
        repository.complete_attempt(
            attempt.attempt_id,
            execution_status=ExecutionStatus.COMPLETED,
            result_disposition=ResultDisposition.TESTED,
            score=1,
            run_id="run-a",
        )


def test_fractional_attempt_score_round_trips_without_truncation(repository) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )

    completed = repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=35.5,
        run_id="fractional-run",
    )

    assert completed.score == 35.5
    assert repository.get_attempt(attempt.attempt_id).score == 35.5
    assert repository.get_cell(cell.cell_id).selected_attempt_id == attempt.attempt_id


@pytest.mark.parametrize("invalid_score", (True, float("nan"), float("inf"), -float("inf")))
def test_attempt_score_rejects_nonfinite_and_boolean_values(
    repository,
    invalid_score: object,
) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )

    with pytest.raises(ValueError, match="finite numeric score"):
        repository.complete_attempt(
            attempt.attempt_id,
            execution_status=ExecutionStatus.COMPLETED,
            result_disposition=ResultDisposition.TESTED,
            score=invalid_score,
        )


def test_recovered_worker_lost_exact_replay_does_not_become_late(repository) -> None:
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )
    assert repository.claim_attempt(
        attempt.attempt_id,
        worker_id="expired-worker",
        lease_seconds=0,
    )
    repository.recover_expired_attempt(cell.cell_id)
    recovered = repository.get_attempt(attempt.attempt_id)

    replay = repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.WORKER_LOST,
    )

    assert replay == recovered
    assert replay.late_completion is False


def test_retryable_cells_exclude_budget_and_nonrecoverable_cancellation(repository) -> None:
    first = _seed(repository)
    budget = repository.create_attempt(first.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(
        budget.attempt_id,
        execution_status=ExecutionStatus.BUDGET_EXHAUSTED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.BUDGET,
    )
    second = repository.create_cell(_cell(cell_id="cell-b", prompt_id="prompt-b"))
    cancelled = repository.create_attempt(second.cell_id, retry_kind="initial", spec_hash="spec-a")
    repository.complete_attempt(
        cancelled.attempt_id,
        execution_status=ExecutionStatus.CANCELLED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.CANCELLED,
    )
    third = repository.create_cell(_cell(cell_id="cell-c", prompt_id="prompt-c"))
    recoverable = repository.create_attempt(
        third.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
        recoverable_cancellation=True,
    )
    repository.complete_attempt(
        recoverable.attempt_id,
        execution_status=ExecutionStatus.CANCELLED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.CANCELLED,
    )

    assert repository.retryable_cells("exp-attempts") == (repository.get_cell(third.cell_id),)


def test_postgresql_attempt_score_schema_repairs_integer_and_round_trips_fraction() -> None:
    if not os.environ.get("TASK14_DATABASE_URL"):
        pytest.skip("requires disposable Task 17 PostgreSQL")
    import psycopg

    from app.persistence.experiment_attempts import (
        PostgreSQLExperimentAttemptPersistence,
    )

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
            "ALTER TABLE experiment_cell_attempts ALTER COLUMN score TYPE INTEGER "
            "USING score::INTEGER"
        )
        connection.execute(
            "INSERT INTO experiments (experiment_id, name) "
            "VALUES ('exp-attempts', 'attempt tests')"
        )

    repository = PostgreSQLExperimentAttemptPersistence(dsn)
    cell = _seed(repository)
    attempt = repository.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )
    completed = repository.complete_attempt(
        attempt.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=35.5,
    )

    assert completed.score == 35.5
    with psycopg.connect(dsn, autocommit=True) as connection:
        assert connection.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'experiment_cell_attempts' AND column_name = 'score'"
        ).fetchone() == ("double precision",)
        assert connection.execute(
            "SELECT score FROM experiment_cell_attempts WHERE attempt_id = %s",
            (attempt.attempt_id,),
        ).fetchone() == (35.5,)


def test_postgresql_explicit_attempt_projection_creates_no_duplicate_and_preserves_selection() -> None:
    if not os.environ.get("TASK14_DATABASE_URL"):
        pytest.skip("requires disposable Task 17 PostgreSQL")
    import psycopg

    from app.persistence.experiment_attempts import PostgreSQLExperimentAttemptPersistence
    from app.persistence.runs import PostgreSQLRunPersistence

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
            "INSERT INTO experiments (experiment_id, name) VALUES ('exp-attempts', 'attempt tests')"
        )
    attempts = PostgreSQLExperimentAttemptPersistence(dsn)
    runs = PostgreSQLRunPersistence(dsn)
    cell = _seed(attempts)
    selected = attempts.create_attempt(
        cell.cell_id,
        retry_kind="initial",
        spec_hash="spec-a",
    )
    attempts.complete_attempt(
        selected.attempt_id,
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        score=80,
        run_id="run-selected",
    )
    payload = {
        "cell_id": cell.cell_id,
        "technique_config_id": cell.technique_config_id,
        "spec_hash": "spec-a",
        "technique_key": cell.technique_key,
        "prompt_id": cell.prompt_id,
        "repeat_index": cell.repeat_index,
        "attempt_id": selected.attempt_id,
        "selected_attempt_id": selected.attempt_id,
        "run_id": "run-selected",
        "status": "completed",
        "score": 80,
        "asr": 100,
        "error": "",
    }

    runs.save_experiment_cell(experiment_id="exp-attempts", cell=payload)
    with psycopg.connect(dsn, autocommit=True) as connection:
        first_projection = connection.execute(
            "SELECT run_id, status, score, asr, matched_detections, "
            "highest_severity, error, metadata, completed_at "
            "FROM experiment_cells WHERE cell_id = %s",
            (cell.cell_id,),
        ).fetchone()
    runs.save_experiment_cell(experiment_id="exp-attempts", cell=payload)
    with psycopg.connect(dsn, autocommit=True) as connection:
        replayed_projection = connection.execute(
            "SELECT run_id, status, score, asr, matched_detections, "
            "highest_severity, error, metadata, completed_at "
            "FROM experiment_cells WHERE cell_id = %s",
            (cell.cell_id,),
        ).fetchone()
    assert replayed_projection == first_projection
    with pytest.raises(ValueError, match="projection replay conflicts"):
        runs.save_experiment_cell(
            experiment_id="exp-attempts",
            cell={**payload, "highest_severity": "critical"},
        )
    diagnostic = attempts.create_attempt(
        cell.cell_id,
        retry_kind="restart",
        promotion_policy="diagnostic_only",
        spec_hash="spec-a",
    )
    attempts.complete_attempt(
        diagnostic.attempt_id,
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        failure_category=FailureCategory.PROVIDER,
        run_id="run-diagnostic",
    )
    runs.save_experiment_cell(
        experiment_id="exp-attempts",
        cell={
            **payload,
            "attempt_id": diagnostic.attempt_id,
            "selected_attempt_id": selected.attempt_id,
            "run_id": "run-diagnostic",
            "status": "error",
            "score": 0,
            "asr": 0,
            "error": "provider failed",
        },
    )

    with psycopg.connect(dsn, autocommit=True) as connection:
        attempt_count = connection.execute(
            "SELECT COUNT(*) FROM experiment_cell_attempts WHERE cell_id = %s",
            (cell.cell_id,),
        ).fetchone()[0]
        projected = connection.execute(
            "SELECT selected_attempt_id, latest_attempt_id, run_id, status, score, asr "
            "FROM experiment_cells WHERE cell_id = %s",
            (cell.cell_id,),
        ).fetchone()
    assert attempt_count == 2
    assert projected == (
        selected.attempt_id,
        diagnostic.attempt_id,
        "run-selected",
        "completed",
        80,
        100,
    )


@pytest.mark.skipif(not os.environ.get("TASK14_DATABASE_URL"), reason="requires disposable PostgreSQL")
def test_postgresql_attempt_schema_repairs_phantom_and_does_not_backfill_pristine_planned_cells() -> None:
    import psycopg

    from app.persistence.experiment_attempts import PostgreSQLExperimentAttemptPersistence

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
            "INSERT INTO experiments (experiment_id, name) "
            "VALUES ('exp-planned', 'planned attempt test')"
        )
        connection.execute(
            """
            INSERT INTO experiment_technique_configs (
                technique_config_id, experiment_id, technique_id,
                technique_version, implementation_id, spec_hash
            ) VALUES (
                'cfg-planned', 'exp-planned', 'direct_prompt_injection',
                '1.0.0', 'builtin', 'spec-planned'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO experiment_cells (
                cell_id, experiment_id, technique_key, technique_config_id,
                prompt_id, repeat_index, status
            ) VALUES
            (
                'cell-planned', 'exp-planned', 'direct_prompt_injection',
                'cfg-planned', 'prompt-planned', 1, 'pending'
            ),
            (
                'cell-stale-planned', 'exp-planned', 'direct_prompt_injection',
                'cfg-planned', 'prompt-stale-planned', 1, 'pending'
            ),
            (
                'cell-enriched-planned', 'exp-planned', 'direct_prompt_injection',
                'cfg-planned', 'prompt-enriched-planned', 1, 'pending'
            ),
            (
                'cell-referenced-planned', 'exp-planned', 'direct_prompt_injection',
                'cfg-planned', 'prompt-referenced-planned', 1, 'pending'
            ),
            (
                'cell-running-legacy', 'exp-planned', 'direct_prompt_injection',
                'cfg-planned', 'prompt-running-legacy', 1, 'running'
            )
            """
        )
        artifact_ids = dict(
            connection.execute(
                """
                SELECT cell_id, 'attempt-' || md5(cell_id || E'\\x1f1')
                FROM experiment_cells
                WHERE cell_id IN (
                    'cell-stale-planned',
                    'cell-enriched-planned',
                    'cell-referenced-planned'
                )
                """
            ).fetchall()
        )
        for cell_id, checkpoint_id in (
            ("cell-stale-planned", None),
            ("cell-enriched-planned", "checkpoint-preserved"),
            ("cell-referenced-planned", None),
        ):
            connection.execute(
                """
                INSERT INTO experiment_cell_attempts (
                    attempt_id, cell_id, attempt_number, retry_kind, spec_hash,
                    execution_status, result_disposition, official_judge_status,
                    selection_reason, checkpoint_id
                ) VALUES (
                    %s, %s, 1, 'initial', 'spec-planned', 'queued',
                    'not_tested', 'not_scheduled', 'legacy_backfill', %s
                )
                """,
                (artifact_ids[cell_id], cell_id, checkpoint_id),
            )
        connection.execute(
            """
            UPDATE experiment_cells AS cell
            SET selected_attempt_id = attempt.attempt_id,
                latest_attempt_id = attempt.attempt_id
            FROM experiment_cell_attempts AS attempt
            WHERE attempt.cell_id = cell.cell_id
              AND cell.cell_id IN (
                  'cell-stale-planned',
                  'cell-enriched-planned',
                  'cell-referenced-planned'
              )
            """
        )
        connection.execute(
            """
            INSERT INTO retry_batches (
                retry_batch_id, experiment_id, action, completed_policy,
                status, requested_by
            ) VALUES (
                'retry-preserved', 'exp-planned', 'restart', 'exclude',
                'requested', 'checkpoint-test'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO retry_batch_cells (
                retry_batch_id, cell_id, member_ordinal, status, attempt_id
            ) VALUES (
                'retry-preserved', 'cell-referenced-planned', 0, 'queued', %s
            )
            """,
            (artifact_ids["cell-referenced-planned"],),
        )

    attempts = PostgreSQLExperimentAttemptPersistence(dsn)

    assert attempts.list_attempts("cell-planned") == ()
    assert attempts.list_attempts("cell-stale-planned") == ()
    enriched = attempts.list_attempts("cell-enriched-planned")
    assert len(enriched) == 1
    assert enriched[0].checkpoint_id == "checkpoint-preserved"
    referenced = attempts.list_attempts("cell-referenced-planned")
    assert len(referenced) == 1
    running = attempts.list_attempts("cell-running-legacy")
    assert len(running) == 1
    assert running[0].execution_status is ExecutionStatus.RUNNING
    assert attempts.get_cell("cell-running-legacy").active_attempt_id == running[0].attempt_id

    reloaded = PostgreSQLExperimentAttemptPersistence(dsn)
    assert reloaded.list_attempts("cell-planned") == ()
    assert reloaded.list_attempts("cell-stale-planned") == ()
    assert reloaded.list_attempts("cell-enriched-planned") == enriched
    assert reloaded.list_attempts("cell-referenced-planned") == referenced
    assert reloaded.list_attempts("cell-running-legacy") == running

    initial = attempts.create_attempt(
        "cell-stale-planned",
        retry_kind="initial",
        spec_hash="spec-planned",
    )
    assert initial.attempt_number == 1


def test_in_memory_getters_wait_for_paired_attempt_and_cell_updates() -> None:
    from app.persistence.experiment_attempts import InMemoryExperimentAttemptPersistence

    repository = InMemoryExperimentAttemptPersistence()
    cell = _seed(repository)
    attempt = repository.create_attempt(cell.cell_id, retry_kind="initial", spec_hash="spec-a")
    writer_ready = threading.Event()
    release_writer = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    observed: dict[str, object] = {}
    thread_errors: list[BaseException] = []
    thread_error_lock = threading.Lock()

    def record_error(error: BaseException) -> None:
        with thread_error_lock:
            thread_errors.append(error)

    def writer() -> None:
        try:
            with repository._lock:
                repository._attempts[attempt.attempt_id] = replace(
                    attempt,
                    execution_status=ExecutionStatus.FAILED,
                    result_disposition=ResultDisposition.INCONCLUSIVE,
                )
                writer_ready.set()
                if not release_writer.wait(timeout=10):
                    raise TimeoutError("writer was not released")
                repository._cells[cell.cell_id] = replace(
                    cell,
                    latest_attempt_id=attempt.attempt_id,
                    selected_attempt_id=attempt.attempt_id,
                )
        except BaseException as error:  # pragma: no cover - defensive thread relay
            record_error(error)

    def reader() -> None:
        try:
            if not writer_ready.wait(timeout=5):
                raise TimeoutError("writer did not enter its transaction")
            reader_started.set()
            observed["cell"] = repository.get_cell(cell.cell_id)
            observed["attempt"] = repository.get_attempt(attempt.attempt_id)
            reader_finished.set()
        except BaseException as error:  # pragma: no cover - defensive thread relay
            record_error(error)

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    assert writer_ready.wait(timeout=5)
    reader_thread.start()
    try:
        assert reader_started.wait(timeout=5)
        assert not reader_finished.wait(timeout=0.2)
    finally:
        release_writer.set()
        writer_thread.join(timeout=5)
        reader_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert not thread_errors
    assert reader_finished.is_set()
    assert observed["cell"].selected_attempt_id == attempt.attempt_id
    assert observed["attempt"].execution_status is ExecutionStatus.FAILED


def _database_init_sql() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((repo_root / "database" / "init").glob("*.sql"))
    )


def _owned_index_catalog_dimensions(connection: object, index_name: str) -> tuple[object, ...]:
    return connection.execute(
        """
        SELECT
            array_agg(actual_operator_class.oid::integer ORDER BY key_column.ordinality),
            array_agg(default_operator_class.oid::integer ORDER BY key_column.ordinality),
            array_agg(index_definition.indcollation[(key_column.ordinality - 1)::integer]::integer ORDER BY key_column.ordinality),
            array_agg(attribute.attcollation::integer ORDER BY key_column.ordinality),
            array_agg(index_definition.indoption[(key_column.ordinality - 1)::integer]::integer ORDER BY key_column.ordinality),
            index_definition.indnullsnotdistinct
        FROM pg_class index_class
        JOIN pg_namespace index_schema ON index_schema.oid = index_class.relnamespace
        JOIN pg_index index_definition ON index_definition.indexrelid = index_class.oid
        JOIN pg_am access_method ON access_method.oid = index_class.relam
        JOIN LATERAL unnest(index_definition.indkey) WITH ORDINALITY AS key_column(attnum, ordinality)
            ON key_column.ordinality <= index_definition.indnkeyatts
        JOIN pg_attribute attribute
            ON attribute.attrelid = index_definition.indrelid AND attribute.attnum = key_column.attnum
        JOIN pg_opclass actual_operator_class
            ON actual_operator_class.oid = index_definition.indclass[(key_column.ordinality - 1)::integer]
        JOIN pg_opclass default_operator_class
            ON default_operator_class.opcmethod = access_method.oid
            AND default_operator_class.opcdefault
            AND default_operator_class.opcintype = attribute.atttypid
        WHERE index_schema.nspname = current_schema() AND index_class.relname = %s
        GROUP BY index_definition.indnullsnotdistinct
        """,
        (index_name,),
    ).fetchone()


def _assert_canonical_owned_index_dimensions(dimensions: tuple[object, ...]) -> None:
    actual_opclasses, default_opclasses, actual_collations, column_collations, options, nulls_not_distinct = dimensions
    assert actual_opclasses == default_opclasses
    assert actual_collations == column_collations
    assert options == [0] * len(options)
    assert nulls_not_distinct is False


def test_attempt_schema_keeps_execution_and_result_status_separate() -> None:
    init_sql = _database_init_sql()

    assert "execution_status TEXT NOT NULL" in init_sql
    assert "result_disposition TEXT NOT NULL" in init_sql
    assert "failure_category TEXT" in init_sql


def test_attempt_schema_uses_contract_statuses_and_supports_leases() -> None:
    init_sql = _database_init_sql()

    assert "official_judge_status TEXT NOT NULL DEFAULT 'not_scheduled'" in init_sql
    assert "worker_id TEXT" in init_sql
    assert "lease_expires_at TIMESTAMPTZ" in init_sql
    assert "CHECK (execution_status IN" in init_sql
    assert "CHECK (result_disposition IN" in init_sql
    assert "CHECK (official_judge_status IN" in init_sql
    assert "experiment_cell_attempts_failure_category_check" in init_sql


@pytest.mark.skipif(
    not os.environ.get("TASK13_DATABASE_URL"),
    reason="requires disposable PostgreSQL",
)
def test_postgresql_experiment_plan_rolls_back_base_config_and_cells_together(
    monkeypatch,
) -> None:
    import psycopg

    from app.api.store import EvaluationStore
    from app.persistence import PostgreSQLRunPersistence

    dsn = os.environ["TASK13_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments, "
            "experiment_dataset_prompts, experiment_datasets CASCADE"
        )

    class FailingPlanPersistence(PostgreSQLRunPersistence):
        inserted_cells = 0

        def _insert_planned_cell(self, connection, record) -> None:
            super()._insert_planned_cell(connection, record)
            self.inserted_cells += 1
            if self.inserted_cells == 2:
                raise RuntimeError("injected atomic plan failure")

    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    persistence = FailingPlanPersistence(dsn)
    store = EvaluationStore(persistence=persistence)
    dataset = store.create_dataset(
        name="Atomic rollback",
        prompts=[{"text": "one"}, {"text": "two"}],
    )

    with pytest.raises(RuntimeError, match="injected atomic plan failure"):
        store.create_experiment(
            name="Atomic rollback",
            dataset_id=dataset["dataset_id"],
            technique_keys=["system_prompt_extraction"],
            target_config={"model": "target-model"},
            mode="demo",
        )

    with psycopg.connect(dsn, autocommit=True) as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 0
        assert connection.execute(
            "SELECT count(*) FROM experiment_technique_configs"
        ).fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM experiment_cells").fetchone()[0] == 0

    restarted = EvaluationStore(persistence=PostgreSQLRunPersistence(dsn))
    assert restarted.list_experiments() == []


@pytest.mark.skipif(
    not os.environ.get("TASK13_DATABASE_URL"),
    reason="requires disposable PostgreSQL",
)
def test_postgresql_experiment_plan_replay_rejects_conflicts_without_partial_children(
    monkeypatch,
) -> None:
    import psycopg

    from app.api.store import EvaluationStore
    from app.experiment_plan_integrity import (
        canonical_dataset_hash,
        canonical_plan_content,
        content_hash,
    )
    from app.persistence import PostgreSQLRunPersistence

    dsn = os.environ["TASK13_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments, "
            "experiment_dataset_prompts, experiment_datasets CASCADE"
        )

    created_workers = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            created_workers.append(self)

    persistence = PostgreSQLRunPersistence(dsn)
    with monkeypatch.context() as creation_patch:
        creation_patch.setattr("app.api.store.threading.Thread", FakeThread)
        store = EvaluationStore(persistence=persistence)
        dataset = store.create_dataset(name="Replay", prompts=[{"text": "original"}])
        created = store.create_experiment(
            name="Immutable replay",
            description="original description",
            hypothesis="original hypothesis",
            dataset_id=dataset["dataset_id"],
            technique_keys=["system_prompt_extraction"],
            target_config={"model": "target-model"},
            mode="demo",
        )
    assert len(created_workers) == 1
    state = store._experiments[created["experiment_id"]]
    assert state.plan is not None
    configs, cells = store._experiment_plan_records(state.plan)
    experiment = {
        "experiment_id": state.experiment_id,
        "name": state.name,
        "description": state.description,
        "hypothesis": state.hypothesis,
        "dataset_id": dataset["dataset_id"],
        "technique_keys": state.technique_keys,
        "target_config": state.target_config,
        "mode": state.mode,
        "status": "running",
        "total_cells": state.total_cells,
        "repeat_count": state.repeat_count,
        "max_rounds": state.max_rounds,
        "experiment_group": state.experiment_group,
        "plan": created["plan"],
    }
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            "UPDATE experiments SET status = 'completed', completed_cells = total_cells, "
            "report = '{\"overall_asr\": 0}'::jsonb WHERE experiment_id = %s",
            (state.experiment_id,),
        )

    persistence.save_experiment_plan(
        experiment=experiment,
        technique_configs=configs,
        cells=cells,
    )
    with psycopg.connect(dsn, autocommit=True) as connection:
        original = connection.execute(
            "SELECT name, plan, status, completed_cells, report FROM experiments "
            "WHERE experiment_id = %s",
            (state.experiment_id,),
        ).fetchone()
        original_counts = (
            connection.execute(
                "SELECT count(*) FROM experiment_technique_configs WHERE experiment_id = %s",
                (state.experiment_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT count(*) FROM experiment_cells WHERE experiment_id = %s",
                (state.experiment_id,),
            ).fetchone()[0],
        )

    undeclared_config = replace(
        configs[0],
        technique_config_id="cfg-rejected",
        spec_hash="rejected-spec",
        parameters={"rejected": True},
        resolved_spec={"parameters": {"rejected": True}},
    )
    undeclared_cell = replace(
        cells[0],
        cell_id="cell-rejected",
        technique_config_id="cfg-rejected",
        prompt_id="prompt-rejected",
    )

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        persistence.save_experiment_plan(
            experiment=experiment,
            technique_configs=configs + (undeclared_config,),
            cells=cells + (undeclared_cell,),
        )

    with psycopg.connect(dsn, autocommit=True) as connection:
        assert connection.execute(
            "SELECT name, plan, status, completed_cells, report FROM experiments "
            "WHERE experiment_id = %s",
            (state.experiment_id,),
        ).fetchone() == original
        assert (
            connection.execute(
                "SELECT count(*) FROM experiment_technique_configs WHERE experiment_id = %s",
                (state.experiment_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT count(*) FROM experiment_cells WHERE experiment_id = %s",
                (state.experiment_id,),
            ).fetchone()[0],
        ) == original_counts
        assert connection.execute(
            "SELECT count(*) FROM experiment_technique_configs "
            "WHERE technique_config_id = 'cfg-rejected'"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT count(*) FROM experiment_cells WHERE cell_id = 'cell-rejected'"
        ).fetchone()[0] == 0

    conflicting_experiment = deepcopy(experiment)
    conflicting_plan = conflicting_experiment["plan"]
    conflicting_plan["dataset_version"] = "conflicting-version"
    conflicting_plan["dataset_hash"] = canonical_dataset_hash(
        dataset_id=conflicting_plan["dataset_id"],
        dataset_version=conflicting_plan["dataset_version"],
        prompts=conflicting_plan["cells"],
    )
    conflicting_plan["plan_hash"] = content_hash(
        canonical_plan_content(
            experiment_id=conflicting_plan["experiment_id"],
            dataset_id=conflicting_plan["dataset_id"],
            dataset_version=conflicting_plan["dataset_version"],
            dataset_hash=conflicting_plan["dataset_hash"],
            technique_configs=conflicting_plan["technique_configs"],
            cells=conflicting_plan["cells"],
            repeat_count=conflicting_plan["repeat_count"],
        )
    )

    with pytest.raises(ValueError, match="experiment plan.*different content"):
        persistence.save_experiment_plan(
            experiment=conflicting_experiment,
            technique_configs=configs,
            cells=cells,
        )

    lock_attempted = threading.Event()
    finished = threading.Event()
    replay_errors: list[BaseException] = []
    lock_key = f"experiment-plan:{state.experiment_id}"

    class SignalingConnection:
        def __init__(self, connection) -> None:
            self.connection = connection

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            return self.connection.__exit__(exc_type, exc, traceback)

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, query, params=None):
            if (
                "pg_advisory_xact_lock" in str(query)
                and tuple(params or ()) == (lock_key,)
            ):
                lock_attempted.set()
            return self.connection.execute(query, params)

    class SignalingPersistence(PostgreSQLRunPersistence):
        def _connect(self):
            return SignalingConnection(super()._connect())

    signaling_persistence = SignalingPersistence(dsn)

    def replay_conflict() -> None:
        try:
            signaling_persistence.save_experiment_plan(
                experiment=conflicting_experiment,
                technique_configs=configs,
                cells=cells,
            )
        except BaseException as exc:  # pragma: no cover - relayed from worker
            replay_errors.append(exc)
        finally:
            finished.set()

    with psycopg.connect(dsn) as blocker:
        blocker.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (lock_key,),
        )
        replay_thread = threading.Thread(target=replay_conflict)
        replay_thread.start()
        try:
            assert lock_attempted.wait(timeout=5)
            assert not finished.is_set()
        finally:
            blocker.rollback()
            replay_thread.join(timeout=5)

    assert not replay_thread.is_alive()
    assert len(replay_errors) == 1
    assert isinstance(replay_errors[0], ValueError)


@pytest.mark.skipif(
    not os.environ.get("TASK13_DATABASE_URL"),
    reason="requires disposable PostgreSQL",
)
def test_postgresql_plan_snapshot_survives_mutable_dataset_row_changes(
    monkeypatch,
) -> None:
    import psycopg

    from app.api.store import EvaluationStore
    from app.persistence import PostgreSQLRunPersistence

    dsn = os.environ["TASK13_DATABASE_URL"]
    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE retry_batch_cells, retry_batches, experiment_cell_attempts, "
            "experiment_technique_configs, experiment_cells, experiments, "
            "experiment_dataset_prompts, experiment_datasets CASCADE"
        )

    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: None)
    store = EvaluationStore(persistence=PostgreSQLRunPersistence(dsn))
    dataset = store.create_dataset(
        name="Original dataset",
        description="original description",
        prompts=[
            {
                "text": "original intent",
                "category": "original-category",
                "gold_answer": "original answer",
            }
        ],
    )
    created = store.create_experiment(
        name="Immutable snapshot",
        dataset_id=dataset["dataset_id"],
        technique_keys=["system_prompt_extraction"],
        target_config={"model": "target-model"},
        mode="demo",
    )
    original_plan = created["plan"]

    with psycopg.connect(dsn, autocommit=True) as connection:
        persistence = PostgreSQLRunPersistence(dsn)
        persistence.ensure_schema(connection)
        persistence.ensure_schema(connection)
        persisted_configs = connection.execute(
            "SELECT technique_config_id, technique_version, implementation_id "
            "FROM experiment_technique_configs WHERE experiment_id = %s",
            (created["experiment_id"],),
        ).fetchall()
        assert persisted_configs == [
            (
                original_plan["technique_configs"][0]["technique_config_id"],
                original_plan["technique_configs"][0]["technique_version"],
                original_plan["technique_configs"][0]["implementation_id"],
            )
        ]
        connection.execute(
            "UPDATE experiment_datasets SET name = 'Mutated dataset', "
            "description = 'mutated description' WHERE dataset_id = %s",
            (dataset["dataset_id"],),
        )
        connection.execute(
            "UPDATE experiment_dataset_prompts SET prompt_text = 'mutated intent', "
            "category = 'mutated-category', gold_answer = 'mutated answer' "
            "WHERE dataset_id = %s",
            (dataset["dataset_id"],),
        )

    restarted = EvaluationStore(persistence=PostgreSQLRunPersistence(dsn))
    restored = restarted.get_experiment(created["experiment_id"])

    assert restored is not None
    assert restored["plan"] == original_plan
    assert restored["plan"]["dataset_id"] == dataset["dataset_id"]
    assert restored["plan"]["dataset_hash"] == original_plan["dataset_hash"]
    assert restored["plan"]["plan_hash"] == original_plan["plan_hash"]
    assert restored["plan"]["cells"][0]["prompt_text"] == "original intent"
    assert restored["plan"]["cells"][0]["prompt_category"] == "original-category"
    assert restored["plan"]["cells"][0]["gold_answer"] == "original answer"


@pytest.fixture
def legacy_migration_database_url():
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from uuid import uuid4

    dsn = os.environ.get("TASK13_DATABASE_URL")
    if not dsn:
        pytest.skip("requires disposable PostgreSQL")
    schema = f"legacy_snapshot_{uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        # Every connection opened by persistence stays in this schema. Keeping
        # the minimal legacy tables out of public allows database/container reuse.
        yield make_conninfo(dsn, options=f"-c search_path={schema}")
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.skipif(not os.environ.get("TASK13_DATABASE_URL"), reason="requires disposable PostgreSQL")
def test_seeded_legacy_cells_migrate_idempotently_and_compatibility_writer_survives(
    legacy_migration_database_url,
) -> None:
    import psycopg
    from psycopg import sql

    from app.persistence import PostgreSQLRunPersistence

    def evidence_snapshot(connection):
        # Canonical JSON strings make nested JSON and timestamps immutable too.
        # Include every column, so IDs, pointers, ordering and evidence cannot
        # silently change while schema/index repairs appear successful.
        return tuple(
            (
                table,
                tuple(connection.execute(
                    sql.SQL("SELECT to_jsonb(evidence)::text FROM {} AS evidence ORDER BY {}").format(
                        sql.Identifier(table),
                        sql.SQL(", ").join(sql.Identifier(key) for key in keys),
                    )
                ).fetchall()),
            )
            for table, keys in (
                ("experiments", ("experiment_id",)),
                ("experiment_technique_configs", ("technique_config_id",)),
                ("experiment_cells", ("cell_id",)),
                ("experiment_cell_attempts", ("cell_id", "attempt_number")),
                ("retry_batches", ("retry_batch_id",)),
                ("retry_batch_cells", ("retry_batch_id", "member_ordinal")),
                ("execution_events", ("run_id", "sequence")),
            )
        )

    def legacy_snapshot(connection):
        return tuple(connection.execute("""
            SELECT experiment_id, technique_key, prompt_id, repeat_index,
                   run_id, status, score, asr, matched_detections,
                   highest_severity, error, metadata::text, completed_at
            FROM experiment_cells ORDER BY prompt_id
        """).fetchall())

    dsn = legacy_migration_database_url
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        connection.execute(
            """CREATE TABLE experiment_cells (
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id), technique_key TEXT NOT NULL,
                prompt_id TEXT NOT NULL, repeat_index INTEGER NOT NULL DEFAULT 1, run_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending', score INTEGER NOT NULL DEFAULT 0, asr INTEGER NOT NULL DEFAULT 0, matched_detections INTEGER NOT NULL DEFAULT 0,
                highest_severity TEXT NOT NULL DEFAULT 'info', error TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb, completed_at TIMESTAMPTZ,
                CONSTRAINT legacy_cells_primary PRIMARY KEY (experiment_id, technique_key, prompt_id, repeat_index)
            )"""
        )
        connection.execute("INSERT INTO experiments VALUES ('exp-legacy')")
        connection.execute("INSERT INTO experiment_cells (experiment_id, technique_key, prompt_id, status, score, asr) VALUES ('exp-legacy', 'legacy-technique', 'zero', 'completed', 0, 0), ('exp-legacy', 'legacy-technique', 'failure', 'error', 0, 0)")
        connection.execute("ALTER TABLE experiment_cells ALTER COLUMN status DROP NOT NULL, ALTER COLUMN score DROP NOT NULL")
        connection.execute("INSERT INTO experiment_cells (experiment_id, technique_key, prompt_id, status, score, asr) VALUES ('exp-legacy', 'legacy-technique', 'missing', 'completed', NULL, 0)")
        connection.execute("""
            UPDATE experiment_cells SET run_id = 'legacy-run-' || prompt_id,
                metadata = jsonb_build_object('source', prompt_id),
                completed_at = '2026-08-31 12:00:00+00'
        """)
        legacy_before = legacy_snapshot(connection)
        persistence = PostgreSQLRunPersistence(dsn)
        persistence.ensure_schema(connection)
        assert legacy_snapshot(connection) == legacy_before
        migrated_before = evidence_snapshot(connection)
        persistence.ensure_schema(connection)
        assert evidence_snapshot(connection) == migrated_before
        assert connection.execute("SELECT count(*) FROM experiment_technique_configs").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM experiment_cell_attempts").fetchone()[0] == 3
        assert connection.execute("SELECT score, result_disposition, official_judge_status FROM experiment_cell_attempts WHERE cell_id = (SELECT cell_id FROM experiment_cells WHERE prompt_id = 'zero')").fetchone() == (0, "tested", "succeeded")
        assert connection.execute("SELECT score, result_disposition FROM experiment_cell_attempts WHERE cell_id = (SELECT cell_id FROM experiment_cells WHERE prompt_id = 'missing')").fetchone() == (None, "inconclusive")
        assert connection.execute("SELECT count(*) FROM experiment_cell_attempts WHERE failure_category IS NOT NULL AND failure_category NOT IN ('provider', 'timeout', 'judge_parse', 'validation', 'cancelled', 'budget', 'worker_lost', 'persistence', 'internal')").fetchone()[0] == 0
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute("UPDATE experiment_cell_attempts SET failure_category = 'legacy_unknown' WHERE attempt_number = 1")
        assert connection.execute("SELECT count(*) FROM experiment_cells c JOIN experiment_cell_attempts a ON (a.cell_id, a.attempt_id) = (c.cell_id, c.selected_attempt_id) WHERE c.selected_attempt_id IS NOT NULL").fetchone()[0] == 3
        connection.execute("""
            INSERT INTO experiment_cell_attempts (attempt_id, cell_id,
                attempt_number, parent_attempt_id, run_id, retry_kind, spec_hash,
                execution_status, result_disposition, official_judge_status,
                failure_category, completed_at)
            SELECT 'snapshot-retry-attempt', a.cell_id, 2, a.attempt_id,
                'snapshot-retry-run', 'restart', a.spec_hash, 'failed',
                'inconclusive', 'failed', 'timeout', '2026-08-31 12:01:00+00'
            FROM experiment_cell_attempts a JOIN experiment_cells c USING (cell_id)
            WHERE c.prompt_id = 'missing'
        """)
        connection.execute("""
            UPDATE experiment_cells SET latest_attempt_id = 'snapshot-retry-attempt'
            WHERE prompt_id = 'missing'
        """)
        connection.execute("""
            INSERT INTO retry_batches (retry_batch_id, experiment_id, request_id,
                action, selection, execution_generation)
            VALUES ('snapshot-batch', 'exp-legacy', 'snapshot-request', 'restart',
                '{"reason": "migration evidence"}', 0)
        """)
        # Deliberately differ from lexical cell/prompt order; preserving the
        # caller's member order matters independently of preserving membership.
        connection.execute("""
            INSERT INTO retry_batch_cells
                (retry_batch_id, cell_id, member_ordinal, attempt_id, status)
            SELECT 'snapshot-batch', cell_id,
                CASE prompt_id WHEN 'zero' THEN 0 WHEN 'missing' THEN 1 ELSE 2 END,
                latest_attempt_id, 'completed'
            FROM experiment_cells
        """)
        connection.execute("""
            INSERT INTO execution_events (event_id, experiment_id, cell_id,
                run_id, attempt_id, sequence, event_type, schema_version,
                event_timestamp, technique_id, technique_version,
                implementation_id, payload, event_hash)
            SELECT 'snapshot-event-' || prompt_id, experiment_id, cell_id,
                run_id, selected_attempt_id, 1,
                CASE prompt_id WHEN 'zero' THEN 'judge.completed' ELSE 'judge.failed' END, 1,
                completed_at, technique_key, '0.2.0', 'legacy',
                jsonb_build_object('score', CASE WHEN prompt_id = 'zero' THEN score END,
                    'source', prompt_id),
                'fixture-hash-' || prompt_id
            FROM experiment_cells WHERE prompt_id IN ('zero', 'failure')
        """)
        assert connection.execute("""
            SELECT c.prompt_id FROM retry_batch_cells m
            JOIN experiment_cells c USING (cell_id)
            WHERE m.retry_batch_id = 'snapshot-batch' ORDER BY m.member_ordinal
        """).fetchall() == [('zero',), ('missing',), ('failure',)]
        before_repairs = evidence_snapshot(connection)
        alternate_collation = connection.execute(
            """
            SELECT ns.nspname, coll.collname
            FROM pg_collation coll
            JOIN pg_namespace ns ON ns.oid = coll.collnamespace
            WHERE coll.collencoding IN (-1, pg_char_to_encoding(current_setting('server_encoding')))
              AND coll.collisdeterministic
              AND coll.oid <> (
                  SELECT attcollation FROM pg_attribute
                  WHERE attrelid = 'experiment_cells'::regclass AND attname = 'technique_key'
              )
            ORDER BY (coll.collprovider = 'c') DESC, coll.oid
            LIMIT 1
            """
        ).fetchone()
        nondefault_text_opclass = connection.execute(
            """
            SELECT namespace.nspname, operator_class.opcname
            FROM pg_opclass operator_class
            JOIN pg_namespace namespace ON namespace.oid = operator_class.opcnamespace
            JOIN pg_am access_method ON access_method.oid = operator_class.opcmethod
            WHERE access_method.amname = 'btree'
              AND NOT operator_class.opcdefault
              AND operator_class.opcintype = (
                  SELECT atttypid FROM pg_attribute
                  WHERE attrelid = 'experiment_cells'::regclass AND attname = 'experiment_id'
              )
            ORDER BY operator_class.oid
            LIMIT 1
            """
        ).fetchone()
        assert alternate_collation is not None
        assert nondefault_text_opclass is not None
        connection.execute("DROP INDEX uq_experiment_cells_technique_config_prompt_repeat")
        connection.execute(
            sql.SQL(
                "CREATE UNIQUE INDEX uq_experiment_cells_technique_config_prompt_repeat "
                "ON experiment_cells (experiment_id {}, technique_config_id, prompt_id, repeat_index)"
            ).format(sql.Identifier(*nondefault_text_opclass))
        )
        connection.execute("DROP INDEX IF EXISTS uq_experiment_cells_legacy_technique_prompt_repeat")
        connection.execute(
            sql.SQL(
                "CREATE UNIQUE INDEX uq_experiment_cells_legacy_technique_prompt_repeat "
                "ON experiment_cells (experiment_id, technique_key COLLATE {}, prompt_id, repeat_index)"
            ).format(sql.Identifier(*alternate_collation))
        )
        connection.execute("DROP INDEX uq_active_attempt_per_cell")
        connection.execute(
            "CREATE UNIQUE INDEX uq_active_attempt_per_cell "
            "ON experiment_cell_attempts (cell_id DESC) WHERE execution_status = 'running'"
        )
        persistence.ensure_schema(connection)
        assert evidence_snapshot(connection) == before_repairs
        repaired_dimensions = {
            "opclass": _owned_index_catalog_dimensions(
                connection, "uq_experiment_cells_technique_config_prompt_repeat"
            ),
            "options": _owned_index_catalog_dimensions(connection, "uq_active_attempt_per_cell"),
        }
        assert _owned_index_catalog_dimensions(
            connection, "uq_experiment_cells_legacy_technique_prompt_repeat"
        ) is None
        connection.execute("DROP INDEX uq_experiment_cells_technique_config_prompt_repeat")
        connection.execute(
            "CREATE UNIQUE INDEX uq_experiment_cells_technique_config_prompt_repeat "
            "ON experiment_cells (experiment_id, technique_config_id, prompt_id, repeat_index) "
            "NULLS NOT DISTINCT"
        )
        persistence.ensure_schema(connection)
        assert evidence_snapshot(connection) == before_repairs
        repaired_dimensions["nulls_not_distinct"] = _owned_index_catalog_dimensions(
            connection, "uq_experiment_cells_technique_config_prompt_repeat"
        )
        for dimensions in repaired_dimensions.values():
            _assert_canonical_owned_index_dimensions(dimensions)
        connection.execute("ALTER TABLE experiment_cell_attempts DROP CONSTRAINT experiment_cell_attempts_execution_status_check")
        connection.execute("ALTER TABLE experiment_cell_attempts ADD CONSTRAINT experiment_cell_attempts_execution_status_check CHECK (execution_status IN ('queued', 'running') OR TRUE)")
        connection.execute("ALTER TABLE experiment_cell_attempts ADD CONSTRAINT custom_attempt_limit_check CHECK (attempt_number < 1000)")
        connection.execute("DROP INDEX uq_active_attempt_per_cell")
        connection.execute("CREATE UNIQUE INDEX uq_active_attempt_per_cell ON experiment_cells (cell_id, repeat_index) WHERE status = 'running'")
        connection.execute("DROP INDEX uq_experiment_cells_technique_config_prompt_repeat")
        connection.execute("CREATE INDEX uq_experiment_cells_technique_config_prompt_repeat ON experiment_cells (prompt_id)")
        connection.execute("DROP INDEX IF EXISTS uq_experiment_cells_legacy_technique_prompt_repeat")
        connection.execute("CREATE INDEX uq_experiment_cells_legacy_technique_prompt_repeat ON experiment_cells (prompt_id)")
        connection.execute("ALTER TABLE experiment_cells DROP CONSTRAINT experiment_cells_technique_config_fkey")
        connection.execute("ALTER TABLE experiment_cells ADD CONSTRAINT experiment_cells_technique_config_fkey FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)")
        connection.execute("ALTER TABLE experiment_cell_attempts DROP CONSTRAINT experiment_cell_attempts_failure_category_check")
        connection.execute("ALTER TABLE experiment_cell_attempts ADD CONSTRAINT experiment_cell_attempts_failure_category_check CHECK (failure_category IS NULL OR failure_category IN ('internal', 'timeout', 'legacy_unknown'))")
        connection.execute("UPDATE experiment_cell_attempts SET failure_category = 'legacy_unknown' WHERE cell_id = (SELECT cell_id FROM experiment_cells WHERE prompt_id = 'failure')")
        persistence.ensure_schema(connection)
        # The invalid legacy category is intentionally repaired to its original
        # internal category; all other evidence must remain byte-for-byte equal.
        assert evidence_snapshot(connection) == before_repairs
        execution_check = connection.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = 'experiment_cell_attempts_execution_status_check'").fetchone()[0]
        assert "budget_exhausted" in execution_check and "OR true" not in execution_check
        assert connection.execute("SELECT count(*) FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = 'custom_attempt_limit_check'").fetchone()[0] == 1
        active_index = connection.execute("""SELECT indexed_table.relname, array_agg(attribute.attname ORDER BY key_column.ordinality), pg_get_expr(index_definition.indpred, index_definition.indrelid)
            FROM pg_class index_class JOIN pg_index index_definition ON index_definition.indexrelid = index_class.oid
            JOIN pg_class indexed_table ON indexed_table.oid = index_definition.indrelid
            JOIN LATERAL unnest(index_definition.indkey) WITH ORDINALITY AS key_column(attnum, ordinality) ON key_column.ordinality <= index_definition.indnkeyatts
            JOIN pg_attribute attribute ON attribute.attrelid = indexed_table.oid AND attribute.attnum = key_column.attnum
            WHERE index_class.relname = 'uq_active_attempt_per_cell'
            GROUP BY indexed_table.relname, index_definition.indpred, index_definition.indrelid""").fetchone()
        assert active_index == ("experiment_cell_attempts", ["cell_id"], "(execution_status = 'running'::text)")
        assert "technique_config_id" in connection.execute("SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = 'uq_experiment_cells_technique_config_prompt_repeat'").fetchone()[0]
        assert connection.execute("SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = 'uq_experiment_cells_legacy_technique_prompt_repeat'").fetchone() is None
        assert "experiment_technique_configs" in connection.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid = 'experiment_cells'::regclass AND conname = 'experiment_cells_technique_config_fkey'").fetchone()[0]
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "new", "run_id": "selected-run", "status": "completed", "score": 10, "asr": 20, "matched_detections": 3, "highest_severity": "critical", "experiment_group": "selected"})
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "null-first", "status": "completed", "score": None, "asr": None})
    with psycopg.connect(dsn, autocommit=True) as connection:
        first = connection.execute("SELECT c.cell_id, c.technique_config_id, c.selected_attempt_id, c.latest_attempt_id, c.active_attempt_id, c.score, a.attempt_number, a.score FROM experiment_cells c JOIN experiment_cell_attempts a ON a.attempt_id = c.selected_attempt_id WHERE c.prompt_id = 'new'").fetchone()
        assert first[2] == first[3] and first[4] is None and first[5:] == (10, 1, 10)
        assert first[0].startswith("cell-")
        assert connection.execute("SELECT c.score, a.execution_status, a.result_disposition, a.official_judge_status, a.score FROM experiment_cells c JOIN experiment_cell_attempts a ON a.attempt_id = c.selected_attempt_id WHERE c.prompt_id = 'null-first'").fetchone() == (None, "completed", "inconclusive", "failed", None)
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "new", "run_id": "lower-run", "status": "completed", "score": 0, "asr": 0, "matched_detections": 4, "highest_severity": "high", "experiment_group": "lower"})
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "new", "run_id": "discarded-run", "status": "error", "score": 0, "matched_detections": 99, "highest_severity": "low", "error": "diagnostic failed", "experiment_group": "discarded"})
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "queued-first", "status": "queued", "score": 0})
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "running", "status": "running", "score": 0})
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "running", "status": "completed", "score": 0})
    persistence.save_experiment_cell(experiment_id="exp-legacy", cell={"technique_key": "legacy-technique", "prompt_id": "running", "status": "running", "score": 0})
    with psycopg.connect(dsn, autocommit=True) as connection:
        assert connection.execute("SELECT count(*) FROM experiment_cells WHERE prompt_id = 'new'").fetchone()[0] == 1
        cell_id, selected, latest, active, score = connection.execute("SELECT cell_id, selected_attempt_id, latest_attempt_id, active_attempt_id, score FROM experiment_cells WHERE prompt_id = 'new'").fetchone()
        assert selected != latest and active is None and score == 0
        assert connection.execute("SELECT attempt_number, score FROM experiment_cell_attempts WHERE cell_id = %s ORDER BY attempt_number", (cell_id,)).fetchall() == [(1, 10), (2, 0), (3, None)]
        assert connection.execute("SELECT run_id, status, score, asr, matched_detections, highest_severity, error, metadata, completed_at IS NOT NULL FROM experiment_cells WHERE prompt_id = 'new'").fetchone() == ("lower-run", "completed", 0, 0, 4, "high", None, {"experiment_group": "lower"}, True)
        assert connection.execute("SELECT status, completed_at FROM experiment_cells WHERE prompt_id = 'queued-first'").fetchone() == ("pending", None)
        assert connection.execute("SELECT execution_status, completed_at FROM experiment_cell_attempts WHERE cell_id = (SELECT cell_id FROM experiment_cells WHERE prompt_id = 'queued-first')").fetchone() == ("queued", None)
        assert connection.execute("SELECT count(*) FROM experiment_cells c JOIN experiment_cell_attempts a ON (a.cell_id, a.attempt_id) = (c.cell_id, c.selected_attempt_id) WHERE c.selected_attempt_id IS NOT NULL").fetchone()[0] == 7
        assert connection.execute("SELECT active_attempt_id = latest_attempt_id FROM experiment_cells WHERE prompt_id = 'running'").fetchone()[0] is True
        assert connection.execute("SELECT status, score FROM experiment_cells WHERE prompt_id = 'running'").fetchone() == ("running", 0)
        assert connection.execute("SELECT execution_status FROM experiment_cell_attempts WHERE cell_id = (SELECT cell_id FROM experiment_cells WHERE prompt_id = 'running') ORDER BY attempt_number").fetchall() == [("cancelled",), ("completed",), ("running",)]

    from app.persistence import (
        CellRecord,
        PostgreSQLExperimentAttemptPersistence,
        TechniqueConfigRecord,
    )

    attempts = PostgreSQLExperimentAttemptPersistence(dsn)
    for config_id, ratio, spec_hash in (
        ("etc-low", 0.5, "spec-low"),
        ("etc-high", 0.9, "spec-high"),
    ):
        attempts.upsert_technique_config(
            TechniqueConfigRecord(
                technique_config_id=config_id,
                experiment_id="exp-legacy",
                technique_id="legacy-technique",
                technique_version="0.2.0",
                implementation_id="legacy",
                parameters={"ratio": ratio},
                resolved_spec={"parameters": {"ratio": ratio}},
                spec_hash=spec_hash,
            )
        )
        attempts.create_cell(
            CellRecord(
                cell_id=f"cell-{config_id}",
                experiment_id="exp-legacy",
                technique_config_id=config_id,
                technique_key="legacy-technique",
                prompt_id="shared-arm",
                repeat_index=1,
            )
        )
        persistence.save_experiment_cell(
            experiment_id="exp-legacy",
            cell={
                "cell_id": f"cell-{config_id}",
                "technique_config_id": config_id,
                "spec_hash": spec_hash,
                "technique_key": "legacy-technique",
                "prompt_id": "shared-arm",
                "repeat_index": 1,
                "status": "completed",
                "score": 5 if ratio == 0.5 else 9,
                "asr": 0 if ratio == 0.5 else 100,
            },
        )
    with psycopg.connect(dsn, autocommit=True) as connection:
        rows = connection.execute(
            """
            SELECT cell_id, technique_config_id, score
            FROM experiment_cells
            WHERE experiment_id = 'exp-legacy' AND prompt_id = 'shared-arm'
            ORDER BY technique_config_id
            """
        ).fetchall()
    assert rows == [
        ("cell-etc-high", "etc-high", 9),
        ("cell-etc-low", "etc-low", 5),
    ]
    with psycopg.connect(dsn, autocommit=True) as connection:
        # Repeat after compatibility writes have created attempt history,
        # selected != latest pointers, a running attempt, null and zero scores,
        # and two configurations of the same technique.
        after_writes = evidence_snapshot(connection)
        connection.execute("DROP INDEX uq_active_attempt_per_cell")
        connection.execute("CREATE INDEX uq_active_attempt_per_cell ON experiment_cell_attempts (attempt_number)")
        for _ in range(2):
            persistence.ensure_schema(connection)
            assert evidence_snapshot(connection) == after_writes
