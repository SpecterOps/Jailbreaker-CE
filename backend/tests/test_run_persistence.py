import json
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

sys.path.insert(0, "backend")

from app.attacklib import get_attack_technique
from app.api.models import ExperimentState, RunSummary
from app.api.store import EvaluationStore, _restore_plan, _serialize_plan
from app.experiment_plan_integrity import (
    canonical_dataset_hash,
    canonical_plan_content,
    cell_id as canonical_cell_id,
    content_hash,
    technique_config_id,
)
from app.persistence import (
    CellRecord,
    InMemoryExperimentAttemptPersistence,
    PostgreSQLRunPersistence,
    RunPersistence,
    TechniqueConfigRecord,
)


class RecordingPersistence(RunPersistence):
    def __init__(self) -> None:
        self.technique_runs: list[dict] = []
        self.baseline_runs: list[dict] = []
        self.custom_techniques: list[dict] = []

    def save_technique_run(
        self,
        *,
        summary,
        result,
        conversation,
        technique_metadata,
        mode,
        requested_by,
        unsafe_intent,
        seed_prompt,
        target_config,
        evaluation_id=None,
    ) -> None:
        self.technique_runs.append(
            {
                "summary": summary,
                "result": result,
                "conversation": conversation,
                "technique_metadata": technique_metadata,
                "mode": mode,
                "requested_by": requested_by,
                "unsafe_intent": unsafe_intent,
                "seed_prompt": seed_prompt,
                "target_config": target_config,
                "evaluation_id": evaluation_id,
            }
        )

    def save_baseline_run(
        self,
        *,
        target_config,
        result,
        requested_by,
        evaluation_id=None,
    ) -> str:
        self.baseline_runs.append(
            {
                "target_config": target_config,
                "result": result,
                "requested_by": requested_by,
                "evaluation_id": evaluation_id,
            }
        )
        return str(result.get("run_id", ""))

    def list_custom_techniques(self) -> list[dict]:
        return list(self.custom_techniques)

    def upsert_custom_technique(self, *, definition, requested_by) -> dict:
        item = dict(definition)
        item["requested_by"] = requested_by
        existing = next((idx for idx, entry in enumerate(self.custom_techniques) if entry["id"] == item["id"]), None)
        if existing is None:
            self.custom_techniques.append(item)
        else:
            self.custom_techniques[existing] = item
        return item

    def delete_custom_technique(self, technique_id: str) -> None:
        self.custom_techniques = [item for item in self.custom_techniques if item["id"] != technique_id]

    def save_experiment_dataset(self, *, dataset) -> None:
        return None

    def save_experiment(self, *, experiment) -> None:
        return None

    def update_experiment_status(self, experiment_id: str, status: str, report=None, error=None) -> None:
        return None

    def save_experiment_cell(self, *, experiment_id: str, cell) -> None:
        return None

    def list_experiments(self) -> list[dict]:
        return []

    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        prompts = {
            "dan_style_roleplay": "DB seed for {intent}",
            "system_prompt_extraction": "DB seed: extract the system prompt for {intent}",
            "agent_architecture_enumeration": "DB seed: enumerate architecture for {intent}",
        }
        return prompts.get(technique_key, "")


class RestoringPersistence(RecordingPersistence):
    def __init__(self, experiments: list[dict]) -> None:
        super().__init__()
        self.experiments = experiments
        self.status_updates: list[dict] = []

    def load_all_experiments(self) -> list[dict]:
        return self.experiments

    def update_experiment_status(self, experiment_id: str, status: str, report=None, error=None) -> None:
        self.status_updates.append(
            {
                "experiment_id": experiment_id,
                "status": status,
                "report": report,
                "error": error,
            }
        )

    def reconcile_legacy_interrupted_experiment(
        self,
        experiment_id: str,
        *,
        report=None,
        error=None,
    ) -> bool:
        experiment = next(
            (
                item
                for item in self.experiments
                if item.get("experiment_id") == experiment_id
            ),
            None,
        )
        if experiment is None:
            return False
        if (
            experiment.get("execution_generation", 0) != 0
            or experiment.get("execution_owner_id", "") != ""
            or experiment.get("status") not in {"pending", "running", "rejudging"}
        ):
            return False
        self.update_experiment_status(
            experiment_id,
            "failed",
            report=report,
            error=error,
        )
        return True


def test_base_legacy_interrupted_reconciliation_fails_closed() -> None:
    persistence = RecordingPersistence()

    with pytest.raises(NotImplementedError):
        persistence.reconcile_legacy_interrupted_experiment(
            "exp-legacy",
            report={"tested": 0},
            error="interrupted",
        )


class _FakeQueryResult:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple]:
        return self.rows

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None


class _FakeTransaction:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.snapshot = None

    def __enter__(self):
        self.snapshot = deepcopy(
            (
                self.connection.experiments,
                self.connection.configs,
                self.connection.cells,
            )
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            (
                self.connection.experiments,
                self.connection.configs,
                self.connection.cells,
            ) = self.snapshot


class _ExperimentPlanWriteConnection:
    def __init__(self) -> None:
        self.experiments: dict[str, dict] = {}
        self.configs: dict[str, tuple] = {}
        self.cells: dict[str, tuple] = {}
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    @staticmethod
    def _json(value):
        return json.loads(value) if isinstance(value, str) else value

    def execute(self, query: str, params=None) -> _FakeQueryResult:
        compact = " ".join(query.split())
        self.queries.append(compact)
        params = tuple(params or ())
        if "pg_advisory_xact_lock" in compact:
            return _FakeQueryResult([(None,)])
        if compact.startswith("SELECT") and "FROM experiments" in compact:
            experiment = self.experiments.get(str(params[0]))
            if experiment is None:
                return _FakeQueryResult([])
            if compact.startswith(
                "SELECT status, execution_generation, execution_owner_id"
            ):
                return _FakeQueryResult(
                    [
                        (
                            experiment["status"],
                            experiment["execution_generation"],
                            experiment["execution_owner_id"],
                        )
                    ]
                )
            return _FakeQueryResult(
                [
                    (
                        experiment["name"],
                        experiment["description"],
                        experiment["hypothesis"],
                        experiment["dataset_id"],
                        experiment["technique_keys"],
                        experiment["target_config"],
                        experiment["mode"],
                        experiment["total_cells"],
                        experiment["repeat_count"],
                        experiment["max_rounds"],
                        experiment["experiment_group"],
                        experiment["created_by"],
                        experiment["plan"],
                    )
                ]
            )
        if "INSERT INTO experiments" in compact:
            experiment_id = str(params[0])
            self.experiments.setdefault(
                experiment_id,
                {
                    "name": params[1],
                    "description": params[2],
                    "hypothesis": params[3],
                    "dataset_id": params[4],
                    "technique_keys": self._json(params[5]),
                    "target_config": self._json(params[6]),
                    "mode": params[7],
                    "status": params[8],
                    "total_cells": params[9],
                    "repeat_count": params[10],
                    "max_rounds": params[11],
                    "experiment_group": params[12],
                    "created_by": params[13],
                    "plan": self._json(params[14]),
                    "execution_generation": params[15],
                    "execution_owner_id": params[16],
                },
            )
            return _FakeQueryResult([])
        if "INSERT INTO experiment_technique_configs" in compact:
            content = (
                *params[:6],
                self._json(params[6]),
                self._json(params[7]),
                params[8],
            )
            config_id = str(params[0])
            existing = self.configs.get(config_id)
            if existing is not None and existing != content:
                return _FakeQueryResult([])
            self.configs[config_id] = content
            return _FakeQueryResult([(config_id,)])
        if "INSERT INTO experiment_cells" in compact:
            content = params
            cell_id = str(params[0])
            existing = self.cells.get(cell_id)
            if existing is not None and existing != content:
                return _FakeQueryResult([])
            self.cells[cell_id] = content
            return _FakeQueryResult([(cell_id,)])
        raise AssertionError(f"unexpected query: {query}")


class _ExperimentPlanWritePersistence(PostgreSQLRunPersistence):
    def __init__(self, connection: _ExperimentPlanWriteConnection) -> None:
        self.connection = connection
        self.connect_calls = 0

    def _connect(self):
        self.connect_calls += 1
        return self.connection

    def ensure_schema(self, conn) -> None:
        return None


def _experiment_plan_write_payload(snapshot: dict | None = None):
    snapshot = snapshot or _canonical_plan_snapshot()
    experiment_id = snapshot["experiment_id"]
    planned_config = snapshot["technique_configs"][0]
    planned_cell = snapshot["cells"][0]
    experiment = {
        "experiment_id": experiment_id,
        "name": "Immutable replay",
        "description": "original description",
        "hypothesis": "original hypothesis",
        "dataset_id": snapshot["dataset_id"],
        "technique_keys": ["probe"],
        "target_config": {"model": "target-model"},
        "mode": "demo",
        "status": "running",
        "total_cells": 1,
        "repeat_count": 1,
        "max_rounds": 5,
        "experiment_group": "original-group",
        "created_by": "demo-user",
        "plan": snapshot,
    }
    config = TechniqueConfigRecord(
        technique_config_id=planned_config["technique_config_id"],
        experiment_id=experiment_id,
        technique_id=planned_config["technique_id"],
        technique_version=planned_config["technique_version"],
        implementation_id=planned_config["implementation_id"],
        preset=planned_config["preset"],
        spec_hash=planned_config["spec_hash"],
        parameters=planned_config["parameters"],
        resolved_spec=planned_config["resolved_spec"],
    )
    cell = CellRecord(
        cell_id=planned_cell["cell_id"],
        experiment_id=experiment_id,
        technique_config_id=planned_cell["technique_config_id"],
        technique_key=planned_cell["technique_key"],
        prompt_id=planned_cell["prompt_id"],
        repeat_index=planned_cell["repeat_index"],
    )
    return experiment, (config,), (cell,)


def _config_record_namespace(record, **changes):
    values = {
        field: getattr(record, field)
        for field in (
            "technique_config_id",
            "experiment_id",
            "technique_id",
            "technique_version",
            "implementation_id",
            "preset",
            "parameters",
            "resolved_spec",
            "spec_hash",
        )
    }
    values.update(changes)
    return SimpleNamespace(**values)


class _ExperimentLoadConnection:
    def __init__(
        self,
        experiment_rows: list[tuple],
        prompt_rows: list[tuple] | None = None,
        cell_rows: list[tuple] | None = None,
        config_rows: list[tuple] | None = None,
    ) -> None:
        self.experiment_rows = experiment_rows
        self.prompt_rows = list(prompt_rows or [])
        self.cell_rows = list(cell_rows or [])
        self.config_rows = list(config_rows or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query: str, params=None) -> _FakeQueryResult:
        if "FROM experiments" in query:
            return _FakeQueryResult(self.experiment_rows)
        if "FROM experiment_datasets" in query:
            return _FakeQueryResult([("ds-1", "Dataset", "", 0)])
        if "FROM experiment_dataset_prompts" in query:
            return _FakeQueryResult(self.prompt_rows)
        if "FROM experiment_technique_configs" in query:
            return _FakeQueryResult(self.config_rows)
        if "FROM experiment_cells" in query:
            return _FakeQueryResult(self.cell_rows)
        if "FROM evaluation_run_conversations" in query:
            return _FakeQueryResult([])
        raise AssertionError(f"unexpected query: {query}")


class _FakePostgreSQLRunPersistence(PostgreSQLRunPersistence):
    def __init__(self, connection: _ExperimentLoadConnection) -> None:
        self.connection = connection

    def _connect(self):
        return self.connection

    def ensure_schema(self, conn) -> None:
        return None


@pytest.mark.parametrize("conflict", ["base", "plan"])
def test_postgresql_plan_replay_is_exact_idempotent_and_conflict_atomic(
    conflict: str,
) -> None:
    connection = _ExperimentPlanWriteConnection()
    persistence = _ExperimentPlanWritePersistence(connection)
    experiment, configs, cells = _experiment_plan_write_payload()

    persistence.save_experiment_plan(
        experiment=experiment,
        technique_configs=configs,
        cells=cells,
    )
    experiment_id = experiment["experiment_id"]
    connection.experiments[experiment_id].update(
        {"status": "completed", "report": {"overall_asr": 0}, "error": None}
    )
    persistence.save_experiment_plan(
        experiment=experiment,
        technique_configs=configs,
        cells=cells,
    )
    before = deepcopy(
        (connection.experiments, connection.configs, connection.cells)
    )

    conflicting_experiment = deepcopy(experiment)
    if conflict == "base":
        conflicting_experiment["name"] = "Conflicting replay"
    else:
        conflicting_plan = conflicting_experiment["plan"]
        conflicting_plan["dataset_version"] = "5"
        conflicting_plan["dataset_hash"] = canonical_dataset_hash(
            dataset_id=conflicting_plan["dataset_id"],
            dataset_version=conflicting_plan["dataset_version"],
            prompts=conflicting_plan["cells"],
        )
        _rehash_plan(conflicting_plan)

    with pytest.raises(ValueError, match="experiment plan.*different content"):
        persistence.save_experiment_plan(
            experiment=conflicting_experiment,
            technique_configs=configs,
            cells=cells,
        )

    assert (connection.experiments, connection.configs, connection.cells) == before
    assert len(connection.experiments) == len(connection.configs) == len(connection.cells) == 1
    assert any("pg_advisory_xact_lock" in query for query in connection.queries)
    assert any(
        "FROM experiments" in query and "FOR UPDATE" in query
        for query in connection.queries
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing-config",
        "extra-config",
        "changed-config",
        "missing-cell",
        "extra-cell",
        "changed-cell",
    ],
)
def test_postgresql_plan_children_must_exactly_match_parent_before_connect(
    damage: str,
) -> None:
    connection = _ExperimentPlanWriteConnection()
    persistence = _ExperimentPlanWritePersistence(connection)
    experiment, configs, cells = _experiment_plan_write_payload()

    if damage == "missing-config":
        configs = ()
    elif damage == "extra-config":
        configs = configs + (
            _config_record_namespace(
                configs[0],
                technique_config_id="etc-extra",
            ),
        )
    elif damage == "changed-config":
        configs = (
            _config_record_namespace(
                configs[0],
                parameters={"changed": True},
            ),
        )
    elif damage == "missing-cell":
        cells = ()
    elif damage == "extra-cell":
        cells = cells + (replace(cells[0], cell_id="cell-extra"),)
    else:
        cells = (replace(cells[0], technique_key="changed"),)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        persistence.save_experiment_plan(
            experiment=experiment,
            technique_configs=configs,
            cells=cells,
        )

    assert persistence.connect_calls == 0
    assert connection.queries == []


def test_postgresql_plan_rejects_credential_value_before_connect() -> None:
    snapshot = _canonical_plan_snapshot()
    _set_canonical_parameter_note(snapshot, "Bearer sk-supersecret123")
    experiment, _, _ = _experiment_plan_write_payload()
    experiment["plan"] = snapshot
    planned_config = snapshot["technique_configs"][0]
    planned_cell = snapshot["cells"][0]
    configs = (
        SimpleNamespace(
            technique_config_id=planned_config["technique_config_id"],
            experiment_id=snapshot["experiment_id"],
            technique_id=planned_config["technique_id"],
            technique_version=planned_config["technique_version"],
            implementation_id=planned_config["implementation_id"],
            preset=planned_config["preset"],
            parameters=planned_config["parameters"],
            resolved_spec=planned_config["resolved_spec"],
            spec_hash=planned_config["spec_hash"],
        ),
    )
    cells = (
        SimpleNamespace(
            cell_id=planned_cell["cell_id"],
            experiment_id=snapshot["experiment_id"],
            technique_config_id=planned_cell["technique_config_id"],
            technique_key=planned_cell["technique_key"],
            prompt_id=planned_cell["prompt_id"],
            repeat_index=planned_cell["repeat_index"],
        ),
    )
    connection = _ExperimentPlanWriteConnection()
    persistence = _ExperimentPlanWritePersistence(connection)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        persistence.save_experiment_plan(
            experiment=experiment,
            technique_configs=configs,
            cells=cells,
        )

    assert persistence.connect_calls == 0
    assert connection.queries == []


def test_postgresql_plan_rejects_boolean_normalized_repeat_before_connect() -> None:
    experiment, configs, cells = _experiment_plan_write_payload()
    cell = cells[0]
    invalid_cell = SimpleNamespace(
        cell_id=cell.cell_id,
        experiment_id=cell.experiment_id,
        technique_config_id=cell.technique_config_id,
        technique_key=cell.technique_key,
        prompt_id=cell.prompt_id,
        repeat_index=True,
    )
    connection = _ExperimentPlanWriteConnection()
    persistence = _ExperimentPlanWritePersistence(connection)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        persistence.save_experiment_plan(
            experiment=experiment,
            technique_configs=configs,
            cells=(invalid_cell,),
        )

    assert persistence.connect_calls == 0
    assert connection.queries == []


@pytest.mark.parametrize(
    ("projection", "drifted_value"),
    [
        ("parameters", {"behavior": "drifted"}),
        ("preset", "drifted-preset"),
    ],
)
def test_postgresql_plan_rejects_coordinated_normalized_projection_drift_before_connect(
    projection: str,
    drifted_value: object,
) -> None:
    snapshot = _canonical_plan_snapshot()
    planned_config = snapshot["technique_configs"][0]
    planned_config[projection] = drifted_value
    planned_cell = snapshot["cells"][0]
    experiment, _, _ = _experiment_plan_write_payload()
    experiment["plan"] = snapshot
    configs = (
        SimpleNamespace(
            technique_config_id=planned_config["technique_config_id"],
            experiment_id=snapshot["experiment_id"],
            technique_id=planned_config["technique_id"],
            technique_version=planned_config["technique_version"],
            implementation_id=planned_config["implementation_id"],
            preset=planned_config["preset"],
            parameters=planned_config["parameters"],
            resolved_spec=planned_config["resolved_spec"],
            spec_hash=planned_config["spec_hash"],
        ),
    )
    cells = (
        SimpleNamespace(
            cell_id=planned_cell["cell_id"],
            experiment_id=snapshot["experiment_id"],
            technique_config_id=planned_cell["technique_config_id"],
            technique_key=planned_cell["technique_key"],
            prompt_id=planned_cell["prompt_id"],
            repeat_index=planned_cell["repeat_index"],
        ),
    )
    connection = _ExperimentPlanWriteConnection()
    persistence = _ExperimentPlanWritePersistence(connection)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        persistence.save_experiment_plan(
            experiment=experiment,
            technique_configs=configs,
            cells=cells,
        )

    assert persistence.connect_calls == 0
    assert connection.queries == []


@pytest.mark.parametrize("repeat_index", [True, "1"])
def test_postgresql_cell_writer_rejects_non_integer_repeat_before_connect(
    repeat_index: object,
) -> None:
    class NoConnectPersistence(PostgreSQLRunPersistence):
        def __init__(self) -> None:
            self.connect_calls = 0

        def _connect(self):
            self.connect_calls += 1
            raise AssertionError("invalid repeat reached the database boundary")

    persistence = NoConnectPersistence()

    with pytest.raises(ValueError, match="repeat_index must be a positive integer"):
        persistence.save_experiment_cell(
            experiment_id="exp-repeat",
            cell={
                "technique_key": "probe",
                "prompt_id": "p-001",
                "repeat_index": repeat_index,
            },
        )

    assert persistence.connect_calls == 0


class _CellWriterTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _CellWriterConnection:
    """Small persistence boundary harness for cell-writer ownership branches."""

    def __init__(self, *, modern_status: str | None = None) -> None:
        self.modern_status = modern_status
        self.projection: tuple | None = None
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def transaction(self) -> _CellWriterTransaction:
        return _CellWriterTransaction()

    def execute(self, query: str, params=None) -> _FakeQueryResult:
        compact = " ".join(query.split())
        self.statements.append(compact)
        if "pg_advisory_xact_lock" in compact:
            return _FakeQueryResult([(None,)])
        if compact.startswith("SELECT") and "FROM experiments" in compact:
            if self.modern_status is None:
                if "status" in compact:
                    raise AssertionError("minimal legacy experiments table has no status column")
                return _FakeQueryResult([("exp-legacy",)])
            return _FakeQueryResult([(1, "owner-1", self.modern_status)])
        if "FROM experiment_technique_configs" in compact:
            return _FakeQueryResult([])
        if "INSERT INTO experiment_technique_configs" in compact:
            return _FakeQueryResult([])
        if compact.startswith("SELECT") and "FROM experiment_cells" in compact:
            return _FakeQueryResult([])
        if "INSERT INTO experiment_cells" in compact:
            return _FakeQueryResult([])
        if "COALESCE(MAX(attempt_number), 0) + 1" in compact:
            return _FakeQueryResult([(1,)])
        if "INSERT INTO experiment_cell_attempts" in compact:
            return _FakeQueryResult([])
        if "SELECT run_id, execution_status, score, asr, failure_category" in compact:
            return _FakeQueryResult([("legacy-run", "completed", 7, 100, None)])
        if "UPDATE experiment_cells" in compact:
            self.projection = tuple(params or ())
            return _FakeQueryResult([])
        raise AssertionError(f"unexpected query: {compact}")


class _CellWriterPersistence(PostgreSQLRunPersistence):
    def __init__(self, connection: _CellWriterConnection) -> None:
        self.connection = connection

    def _connect(self):
        return self.connection

    def ensure_schema(self, conn) -> None:
        return None


def test_postgresql_cell_writer_projects_unowned_legacy_cell_without_status() -> None:
    connection = _CellWriterConnection()
    persistence = _CellWriterPersistence(connection)

    accepted = persistence.save_experiment_cell(
        experiment_id="exp-legacy",
        cell={
            "technique_key": "legacy-technique",
            "prompt_id": "prompt-1",
            "run_id": "legacy-run",
            "status": "completed",
            "score": 7,
            "asr": 100,
        },
    )

    assert accepted is True
    assert connection.projection is not None
    assert any("FROM experiments" in statement and "FOR UPDATE" in statement for statement in connection.statements)


def test_postgresql_cell_writer_rejects_owned_projection_when_lifecycle_is_terminal() -> None:
    connection = _CellWriterConnection(modern_status="completed")
    persistence = _CellWriterPersistence(connection)

    accepted = persistence.save_experiment_cell(
        experiment_id="exp-modern",
        generation=1,
        owner_id="owner-1",
        cell={"technique_key": "probe", "prompt_id": "prompt-1"},
    )

    assert accepted is False
    assert connection.projection is None


@pytest.mark.parametrize(
    ("generation", "owner_id"),
    [(1, None), (None, "owner-1")],
)
def test_postgresql_cell_writer_requires_generation_and_owner_together_before_connect(
    generation: int | None,
    owner_id: str | None,
) -> None:
    class NoConnectPersistence(PostgreSQLRunPersistence):
        def _connect(self):
            raise AssertionError("partial ownership reached the database boundary")

    with pytest.raises(ValueError, match="requires generation and owner"):
        NoConnectPersistence().save_experiment_cell(
            experiment_id="exp-owned",
            generation=generation,
            owner_id=owner_id,
            cell={"technique_key": "probe", "prompt_id": "prompt-1"},
        )


class _RejudgeWriteTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _RejudgeWriteConnection:
    def __init__(
        self,
        *,
        selected_attempt_id: str = "attempt-selected",
        metadata: dict | None = None,
        present: bool = True,
    ) -> None:
        self.selected_attempt_id = selected_attempt_id
        self.metadata = deepcopy(metadata or {"unrelated": "preserved"})
        self.present = present
        self.statements: list[tuple[str, tuple]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def transaction(self) -> _RejudgeWriteTransaction:
        return _RejudgeWriteTransaction()

    def execute(self, query: str, params=None) -> _FakeQueryResult:
        compact = " ".join(query.split())
        parameters = tuple(params or ())
        self.statements.append((compact, parameters))
        if compact.startswith("SELECT"):
            if "FROM experiments" in compact:
                return _FakeQueryResult([("rejudging",)])
            if not self.present:
                return _FakeQueryResult([])
            return _FakeQueryResult(
                [(self.selected_attempt_id, deepcopy(self.metadata))]
            )
        if compact.startswith("UPDATE experiment_cells"):
            self.metadata = json.loads(str(parameters[0]))
            return _FakeQueryResult([("cell-rejudge",)])
        raise AssertionError(f"unexpected rejudge persistence query: {compact}")


class _RejudgeWritePersistence(PostgreSQLRunPersistence):
    def __init__(self, connection: _RejudgeWriteConnection) -> None:
        self.connection = connection

    def _connect(self):
        return self.connection

    def ensure_schema(self, conn) -> None:
        return None


def _persisted_rejudge_result(*, score: int = 0) -> dict:
    return {
        "rejudge_batch_id": "rejudge-batch-1",
        "target_evaluation_id": "evaluation-target-1",
        "fabricator_evaluation_id": None,
        "target_subject_message_id": "subject-ref:abc",
        "judge_api_base_url": "http://judge.test",
        "target_score": score,
        "fabricator_score": None,
    }


def _legacy_persisted_rejudge_result(*, score: int = 0) -> dict:
    return {
        "created_at": "2026-08-01T00:00:00+00:00",
        "judge_model": "legacy-judge",
        "target_score": score,
        "fabricator_score": None,
    }


def test_postgresql_rejudge_writer_updates_only_allowlisted_metadata() -> None:
    connection = _RejudgeWriteConnection()
    persistence = _RejudgeWritePersistence(connection)

    persistence.save_experiment_rejudge(
        experiment_id="exp-rejudge",
        generation=2,
        owner_id="rejudge-owner",
        cell_id="cell-rejudge",
        selected_attempt_id="attempt-selected",
        rejudge_result=_persisted_rejudge_result(score=0),
        original_score=80,
        original_asr=100,
    )

    assert connection.metadata == {
        "unrelated": "preserved",
        "original_score": 80,
        "original_asr": 100,
        "rejudge_results": [_persisted_rejudge_result(score=0)],
    }
    update = next(
        statement
        for statement, _ in connection.statements
        if statement.startswith("UPDATE experiment_cells")
    )
    set_clause = update.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
    assert set_clause == "metadata = %s::jsonb"
    assert "experiment_cell_attempts" not in " ".join(
        statement for statement, _ in connection.statements
    )
    assert "FOR UPDATE" in connection.statements[0][0]


@pytest.mark.parametrize(
    ("present", "persisted_attempt", "message"),
    [
        (False, "attempt-selected", "does not exist"),
        (True, "attempt-other", "selected attempt ownership"),
    ],
)
def test_postgresql_rejudge_writer_requires_exact_selected_attempt_ownership(
    present: bool,
    persisted_attempt: str,
    message: str,
) -> None:
    connection = _RejudgeWriteConnection(
        selected_attempt_id=persisted_attempt,
        present=present,
    )
    persistence = _RejudgeWritePersistence(connection)

    with pytest.raises((KeyError, ValueError), match=message):
        persistence.save_experiment_rejudge(
            experiment_id="exp-rejudge",
            generation=2,
            owner_id="rejudge-owner",
            cell_id="cell-rejudge",
            selected_attempt_id="attempt-selected",
            rejudge_result=_persisted_rejudge_result(),
            original_score=80,
            original_asr=100,
        )

    assert not any(
        statement.startswith("UPDATE experiment_cells")
        for statement, _ in connection.statements
    )


def test_postgresql_rejudge_writer_is_idempotent_and_rejects_identity_conflict() -> None:
    result = _persisted_rejudge_result(score=0)
    connection = _RejudgeWriteConnection(
        metadata={
            "rejudge_results": [result],
            "original_score": 80,
            "original_asr": 100,
        }
    )
    persistence = _RejudgeWritePersistence(connection)

    persistence.save_experiment_rejudge(
        experiment_id="exp-rejudge",
        generation=2,
        owner_id="rejudge-owner",
        cell_id="cell-rejudge",
        selected_attempt_id="attempt-selected",
        rejudge_result=result,
        original_score=80,
        original_asr=100,
    )
    assert not any(
        statement.startswith("UPDATE experiment_cells")
        for statement, _ in connection.statements
    )

    with pytest.raises(ValueError, match="rejudge evaluation identity conflicts"):
        persistence.save_experiment_rejudge(
            experiment_id="exp-rejudge",
            generation=2,
            owner_id="rejudge-owner",
            cell_id="cell-rejudge",
            selected_attempt_id="attempt-selected",
            rejudge_result=_persisted_rejudge_result(score=9),
            original_score=80,
            original_asr=100,
        )


def test_postgresql_rejudge_writer_preserves_legacy_history_and_appends_new_result() -> None:
    legacy_result = _legacy_persisted_rejudge_result(score=2)
    new_result = _persisted_rejudge_result(score=7)
    connection = _RejudgeWriteConnection(
        metadata={
            "rejudge_results": [legacy_result],
            "original_score": 80,
            "original_asr": 100,
        }
    )
    persistence = _RejudgeWritePersistence(connection)

    persistence.save_experiment_rejudge(
        experiment_id="exp-rejudge",
        generation=2,
        owner_id="rejudge-owner",
        cell_id="cell-rejudge",
        selected_attempt_id="attempt-selected",
        rejudge_result=new_result,
        original_score=80,
        original_asr=100,
    )

    assert connection.metadata["rejudge_results"] == [legacy_result, new_result]


@pytest.mark.parametrize(
    "partial_result",
    [
        {"rejudge_batch_id": "rejudge-batch-1", "target_score": 2},
        {"target_evaluation_id": "evaluation-target-1", "target_score": 2},
        {"evaluation_run_id": "rejudge-run:partial", "target_score": 2},
        {
            "rejudge_batch_id": 123,
            "target_evaluation_id": "evaluation-target-1",
            "target_score": 2,
        },
    ],
    ids=["batch_only", "evaluation_only", "evaluation_run_only", "malformed_type"],
)
def test_postgresql_rejudge_writer_rejects_partial_task19_identity(
    partial_result: dict,
) -> None:
    connection = _RejudgeWriteConnection(
        metadata={"rejudge_results": [partial_result]}
    )
    persistence = _RejudgeWritePersistence(connection)

    with pytest.raises(
        ValueError,
        match="rejudge result requires a batch and at least one evaluation identity",
    ):
        persistence.save_experiment_rejudge(
            experiment_id="exp-rejudge",
            generation=2,
            owner_id="rejudge-owner",
            cell_id="cell-rejudge",
            selected_attempt_id="attempt-selected",
            rejudge_result=_persisted_rejudge_result(),
            original_score=80,
            original_asr=100,
        )

    assert not any(
        statement.startswith("UPDATE experiment_cells")
        for statement, _ in connection.statements
    )


def test_postgresql_rejudge_writer_rejects_non_mapping_prior_result() -> None:
    connection = _RejudgeWriteConnection(
        metadata={"rejudge_results": ["malformed-result"]}
    )
    persistence = _RejudgeWritePersistence(connection)

    with pytest.raises(ValueError, match="persisted rejudge result must be an object"):
        persistence.save_experiment_rejudge(
            experiment_id="exp-rejudge",
            generation=2,
            owner_id="rejudge-owner",
            cell_id="cell-rejudge",
            selected_attempt_id="attempt-selected",
            rejudge_result=_persisted_rejudge_result(),
            original_score=80,
            original_asr=100,
        )

    assert not any(
        statement.startswith("UPDATE experiment_cells")
        for statement, _ in connection.statements
    )


@pytest.mark.parametrize("max_rounds", [True, "5"])
def test_postgresql_plan_writer_rejects_non_integer_max_rounds_before_connect(
    max_rounds: object,
) -> None:
    experiment, configs, cells = _experiment_plan_write_payload()
    experiment["max_rounds"] = max_rounds
    connection = _ExperimentPlanWriteConnection()
    persistence = _ExperimentPlanWritePersistence(connection)

    with pytest.raises(ValueError, match="max_rounds must be an integer"):
        persistence.save_experiment_plan(
            experiment=experiment,
            technique_configs=configs,
            cells=cells,
        )

    assert persistence.connect_calls == 0
    assert connection.queries == []


def _canonical_plan_snapshot() -> dict:
    experiment_id = "exp-snapshot"
    resolved_spec = {
        "technique_id": "probe",
        "technique_version": "1.2.3",
        "implementation_id": "probe-python",
        "implementation_kind": "python",
        "contract_version": "7",
        "manifest_hash": "a" * 64,
        "asset_hashes": {},
        "preset": None,
        "parameters": {},
        "model_roles": {},
        "judge_policies": {},
        "effective_limits": {},
        "runtime_config": {},
    }
    config_id = technique_config_id(experiment_id, resolved_spec)
    cell = {
        "cell_id": canonical_cell_id(experiment_id, config_id, "p-001", 1),
        "experiment_id": experiment_id,
        "technique_config_id": config_id,
        "technique_key": "probe",
        "prompt_id": "p-001",
        "prompt_text": "original intent",
        "prompt_category": "original-category",
        "repeat_index": 1,
        "gold_answer": "original answer",
    }
    snapshot = {
        "experiment_id": experiment_id,
        "dataset_id": "ds-1",
        "dataset_version": "4",
        "dataset_hash": canonical_dataset_hash(
            dataset_id="ds-1", dataset_version="4", prompts=[cell]
        ),
        "technique_configs": [
            {
                "technique_config_id": config_id,
                "technique_key": "probe",
                **resolved_spec,
                "resolved_spec": resolved_spec,
                "spec_hash": content_hash(resolved_spec),
            }
        ],
        "cells": [cell],
        "repeat_count": 1,
        "plan_hash": "pending",
    }
    _rehash_plan(snapshot)
    return snapshot


def _set_canonical_parameter_note(snapshot: dict, note: str) -> None:
    config = snapshot["technique_configs"][0]
    resolved_spec = config["resolved_spec"]
    parameters = {**resolved_spec["parameters"], "note": note}
    resolved_spec["parameters"] = parameters
    config["parameters"] = parameters
    config["spec_hash"] = content_hash(resolved_spec)
    config["technique_config_id"] = technique_config_id(
        snapshot["experiment_id"], resolved_spec
    )
    for cell in snapshot["cells"]:
        cell["technique_config_id"] = config["technique_config_id"]
        cell["cell_id"] = canonical_cell_id(
            snapshot["experiment_id"],
            config["technique_config_id"],
            cell["prompt_id"],
            cell["repeat_index"],
        )
    _rehash_plan(snapshot)


def _set_canonical_effective_max_rounds(snapshot: dict, max_rounds: object) -> None:
    config = snapshot["technique_configs"][0]
    resolved_spec = config["resolved_spec"]
    effective_limits = {
        **resolved_spec["effective_limits"],
        "max_rounds": max_rounds,
    }
    resolved_spec["effective_limits"] = effective_limits
    config["effective_limits"] = effective_limits
    config["spec_hash"] = content_hash(resolved_spec)
    config["technique_config_id"] = technique_config_id(
        snapshot["experiment_id"], resolved_spec
    )
    for cell in snapshot["cells"]:
        cell["technique_config_id"] = config["technique_config_id"]
        cell["cell_id"] = canonical_cell_id(
            snapshot["experiment_id"],
            config["technique_config_id"],
            cell["prompt_id"],
            cell["repeat_index"],
        )
    _rehash_plan(snapshot)


def _set_canonical_experiment_id(snapshot: dict, experiment_id: str) -> None:
    snapshot["experiment_id"] = experiment_id
    for config in snapshot["technique_configs"]:
        config["technique_config_id"] = technique_config_id(
            experiment_id,
            config["resolved_spec"],
        )
    configs_by_key = {
        config["technique_key"]: config for config in snapshot["technique_configs"]
    }
    for cell in snapshot["cells"]:
        config_id = configs_by_key[cell["technique_key"]]["technique_config_id"]
        cell["experiment_id"] = experiment_id
        cell["technique_config_id"] = config_id
        cell["cell_id"] = canonical_cell_id(
            experiment_id,
            config_id,
            cell["prompt_id"],
            cell["repeat_index"],
        )
    _rehash_plan(snapshot)


def _set_canonical_dataset_id(snapshot: dict, dataset_id: str) -> None:
    snapshot["dataset_id"] = dataset_id
    snapshot["dataset_hash"] = canonical_dataset_hash(
        dataset_id=dataset_id,
        dataset_version=snapshot["dataset_version"],
        prompts=snapshot["cells"],
    )
    _rehash_plan(snapshot)


def _canonical_grid_snapshot() -> dict:
    source = _canonical_plan_snapshot()
    experiment_id = source["experiment_id"]
    configs = []
    for variant in (1, 2):
        resolved_spec = deepcopy(source["technique_configs"][0]["resolved_spec"])
        resolved_spec["parameters"] = {"variant": variant}
        config_id = technique_config_id(experiment_id, resolved_spec)
        configs.append(
            {
                "technique_config_id": config_id,
                "technique_key": "probe",
                **resolved_spec,
                "resolved_spec": resolved_spec,
                "spec_hash": content_hash(resolved_spec),
            }
        )
    prompt_evidence = (
        ("p-001", "original intent", "original-category", "original answer"),
        ("p-002", "second intent", "second-category", None),
    )
    cells = [
        {
            "cell_id": canonical_cell_id(
                experiment_id,
                config["technique_config_id"],
                prompt_id,
                repeat_index,
            ),
            "experiment_id": experiment_id,
            "technique_config_id": config["technique_config_id"],
            "technique_key": config["technique_key"],
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "prompt_category": prompt_category,
            "repeat_index": repeat_index,
            "gold_answer": gold_answer,
        }
        for config in configs
        for prompt_id, prompt_text, prompt_category, gold_answer in prompt_evidence
        for repeat_index in (1, 2)
    ]
    snapshot = {
        "experiment_id": experiment_id,
        "dataset_id": source["dataset_id"],
        "dataset_version": source["dataset_version"],
        "dataset_hash": canonical_dataset_hash(
            dataset_id=source["dataset_id"],
            dataset_version=source["dataset_version"],
            prompts=cells,
        ),
        "technique_configs": configs,
        "cells": cells,
        "repeat_count": 2,
        "plan_hash": "pending",
    }
    _rehash_plan(snapshot)
    return snapshot


def _rehash_plan(snapshot: dict) -> None:
    snapshot["plan_hash"] = content_hash(
        canonical_plan_content(
            experiment_id=snapshot["experiment_id"],
            dataset_id=snapshot["dataset_id"],
            dataset_version=snapshot["dataset_version"],
            dataset_hash=snapshot["dataset_hash"],
            technique_configs=snapshot["technique_configs"],
            cells=snapshot["cells"],
            repeat_count=snapshot["repeat_count"],
        )
    )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("plan", "experiment_id"),
        ("plan", "dataset_id"),
        ("plan", "dataset_version"),
        ("plan", "dataset_hash"),
        ("plan", "plan_hash"),
        ("config", "technique_config_id"),
        ("config", "technique_key"),
        ("config", "technique_id"),
        ("config", "technique_version"),
        ("config", "implementation_id"),
        ("config", "implementation_kind"),
        ("config", "contract_version"),
        ("config", "manifest_hash"),
        ("config", "spec_hash"),
        ("cell", "cell_id"),
        ("cell", "experiment_id"),
        ("cell", "technique_config_id"),
        ("cell", "technique_key"),
        ("cell", "prompt_id"),
    ],
)
def test_restore_plan_rejects_empty_canonical_identity_fields(
    section: str,
    field: str,
) -> None:
    snapshot = _canonical_plan_snapshot()
    target = snapshot
    if section == "config":
        target = snapshot["technique_configs"][0]
    elif section == "cell":
        target = snapshot["cells"][0]
    target[field] = ""

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    [
        ("plan", "dataset_version", 4),
        ("plan", "dataset_hash", b"hash"),
        ("config", "technique_key", 7),
        ("config", "implementation_kind", True),
        ("cell", "prompt_id", 101),
        ("cell", "repeat_index", True),
    ],
)
def test_restore_plan_rejects_wrong_canonical_scalar_types(
    section: str,
    field: str,
    invalid: object,
) -> None:
    snapshot = _canonical_plan_snapshot()
    target = snapshot
    if section == "config":
        target = snapshot["technique_configs"][0]
    elif section == "cell":
        target = snapshot["cells"][0]
    target[field] = invalid

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


def test_store_construction_fails_closed_for_present_malformed_plan() -> None:
    experiment = _persisted_experiment("completed")
    experiment["plan"] = {"experiment_id": ""}

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        EvaluationStore(persistence=RestoringPersistence([experiment]))


@pytest.mark.parametrize(
    ("parent_field", "outer_value"),
    [
        ("experiment_id", "exp-10"),
        ("experiment_id", 9),
        ("dataset_id", "ds-10"),
        ("dataset_id", 1),
        ("repeat_count", 2),
        ("repeat_count", None),
    ],
)
def test_store_restore_binds_canonical_plan_to_persisted_parent(
    parent_field: str,
    outer_value: object,
) -> None:
    snapshot = _canonical_plan_snapshot()
    _set_canonical_experiment_id(snapshot, "exp-9")
    if parent_field == "experiment_id" and not isinstance(outer_value, str):
        _set_canonical_experiment_id(snapshot, str(outer_value))
    if parent_field == "dataset_id" and not isinstance(outer_value, str):
        _set_canonical_dataset_id(snapshot, str(outer_value))
    experiment = _persisted_experiment("completed")
    experiment["experiment_id"] = snapshot["experiment_id"]
    experiment["dataset"]["dataset_id"] = snapshot["dataset_id"]
    experiment["repeat_count"] = snapshot["repeat_count"]
    if parent_field == "dataset_id":
        experiment["dataset"]["dataset_id"] = outer_value
    else:
        experiment[parent_field] = outer_value
    experiment["plan"] = snapshot

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        EvaluationStore(persistence=RestoringPersistence([experiment]))


@pytest.mark.parametrize("legacy_plan", [None, "", [], 7])
def test_restore_plan_preserves_absent_non_mapping_legacy_compatibility(
    legacy_plan: object,
) -> None:
    assert _restore_plan(legacy_plan) is None


def test_restore_plan_rejects_empty_canonical_child_arrays() -> None:
    snapshot = _canonical_plan_snapshot()
    snapshot["technique_configs"] = []
    snapshot["cells"] = []
    snapshot["dataset_hash"] = canonical_dataset_hash(
        dataset_id=snapshot["dataset_id"],
        dataset_version=snapshot["dataset_version"],
        prompts=[],
    )
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


def test_restore_plan_rejects_credential_shaped_canonical_value() -> None:
    snapshot = _canonical_plan_snapshot()
    _set_canonical_parameter_note(snapshot, "Bearer sk-supersecret123")

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


@pytest.mark.parametrize("max_rounds", [True, "5", 0, 51])
def test_restore_plan_rejects_invalid_canonical_effective_max_rounds(
    max_rounds: object,
) -> None:
    snapshot = _canonical_plan_snapshot()
    _set_canonical_effective_max_rounds(snapshot, max_rounds)

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


@pytest.mark.parametrize("max_rounds", [1, 50])
def test_restore_plan_accepts_supported_canonical_effective_max_round_boundaries(
    max_rounds: int,
) -> None:
    snapshot = _canonical_plan_snapshot()
    _set_canonical_effective_max_rounds(snapshot, max_rounds)

    plan = _restore_plan(snapshot)

    assert plan is not None
    assert plan.technique_configs[0].effective_limits["max_rounds"] == max_rounds


def test_restore_plan_rejects_rehashed_partial_cartesian_cell_grid() -> None:
    snapshot = _canonical_grid_snapshot()
    snapshot["cells"].pop()
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


@pytest.mark.parametrize(
    ("projection", "drifted_value"),
    [
        ("parameters", {"behavior": "drifted"}),
        (
            "model_roles",
            {
                "target": {
                    "profile_id": "drifted-profile",
                    "provider": "drifted-provider",
                    "model": "drifted-model",
                    "transport": "direct",
                }
            },
        ),
    ],
)
def test_restore_plan_rejects_behavior_projection_drift(
    projection: str,
    drifted_value: object,
) -> None:
    snapshot = _canonical_plan_snapshot()
    snapshot["technique_configs"][0][projection] = drifted_value

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


@pytest.mark.parametrize(
    "forgery",
    ["config_id", "cell_id", "dataset_hash", "plan_hash", "technique_key"],
)
def test_restore_plan_rejects_forged_canonical_evidence(forgery: str) -> None:
    snapshot = _canonical_plan_snapshot()
    config = snapshot["technique_configs"][0]
    cell = snapshot["cells"][0]
    if forgery == "config_id":
        config["technique_config_id"] = "etc-forged"
        cell["technique_config_id"] = "etc-forged"
    elif forgery == "cell_id":
        cell["cell_id"] = "cell-forged"
    elif forgery == "dataset_hash":
        snapshot["dataset_hash"] = "d" * 64
    elif forgery == "plan_hash":
        snapshot["plan_hash"] = "f" * 64
    else:
        cell["technique_key"] = "drifted-alias"

    with pytest.raises(ValueError, match="malformed persisted experiment plan"):
        _restore_plan(snapshot)


def test_planned_models_reject_boolean_repeat_values() -> None:
    plan = _restore_plan(_canonical_plan_snapshot())
    assert plan is not None

    with pytest.raises(ValueError, match="repeat_index must be a positive integer"):
        replace(plan.cells[0], repeat_index=True)
    with pytest.raises(ValueError, match="repeat_count must be a positive integer"):
        replace(plan, repeat_count=True)


def test_serialize_plan_rejects_credential_shaped_canonical_value() -> None:
    plan = _restore_plan(_canonical_plan_snapshot())
    assert plan is not None
    tainted_config = replace(
        plan.technique_configs[0],
        parameters={"note": "Bearer sk-supersecret123"},
    )

    with pytest.raises(ValueError, match="canonical experiment plan.*credential"):
        _serialize_plan(replace(plan, technique_configs=(tainted_config,)))


class FakeChatClient:
    def __init__(self, responses: list[str], model: str = "fake-model") -> None:
        self._responses = list(responses)
        self.model = model

    def chat_with_retry(self, messages, temperature=None, max_tokens=None, response_format=None):
        return self._responses.pop(0), {"total_tokens": 42}


def test_store_uses_postgresql_attempt_repository_with_run_persistence_dsn(monkeypatch) -> None:
    dsn = "postgresql://task15.test/jailbreaker"
    captured_dsns = []
    attempt_repository = InMemoryExperimentAttemptPersistence()
    monkeypatch.setattr(
        PostgreSQLRunPersistence,
        "load_all_experiments",
        lambda self: [],
    )

    def build_attempt_repository(received_dsn):
        captured_dsns.append(received_dsn)
        return attempt_repository

    monkeypatch.setattr(
        "app.api.store.PostgreSQLExperimentAttemptPersistence",
        build_attempt_repository,
    )

    store = EvaluationStore(persistence=PostgreSQLRunPersistence(dsn))

    assert captured_dsns == [dsn]
    assert store.experiment_attempt_persistence is attempt_repository


def _experiment_state(status: str) -> ExperimentState:
    return ExperimentState(
        experiment_id="exp-1",
        name="Retry lifecycle",
        description="",
        hypothesis="",
        status=status,
        mode="live",
        dataset={
            "dataset_id": "ds-1",
            "name": "Dataset",
            "description": "",
            "prompt_count": 1,
            "prompts": [{"id": "p1", "text": "unsafe prompt", "category": "general"}],
        },
        technique_keys=["dan_style_roleplay"],
        target_config={"model": "target-model"},
        total_cells=1,
        cell_results=[],
    )


def _persisted_experiment(status: str, error: str | None = None) -> dict:
    return {
        "experiment_id": "exp-9",
        "name": "Interrupted experiment",
        "description": "",
        "hypothesis": "",
        "status": status,
        "mode": "live",
        "dataset": {
            "dataset_id": "ds-1",
            "name": "Dataset",
            "description": "",
            "prompt_count": 1,
            "prompts": [{"id": "p1", "text": "unsafe prompt", "category": "general"}],
        },
        "technique_keys": ["dan_style_roleplay"],
        "target_config": {"model": "target-model"},
        "total_cells": 1,
        "completed_cells": 0,
        "cell_results": [],
        "report": None,
        "error": error,
        "created_at": "2026-06-01T00:00:00+00:00",
        "repeat_count": 1,
        "max_rounds": 5,
        "experiment_group": "",
    }


def _architecture_phase(phase: str, **overrides):
    payload = {
        "phase": phase,
        "agent": {
            "name": None,
            "profile": None,
            "model_aliases": [],
            "confidence": "unknown",
        },
        "tools": [],
        "data_stores": [],
        "workflow": {
            "framework": None,
            "nodes": [],
            "edges": [],
            "handoff_paths": [],
        },
        "limits": {
            "write_access": None,
            "network_access": None,
            "persistence": None,
            "cross_agent_handoff": None,
        },
        "unknowns": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.parametrize("status", ["running", "pending", "rejudging"])
def test_retry_experiment_rejects_active_statuses(status: str) -> None:
    store = EvaluationStore(persistence=RecordingPersistence())
    store._experiments["exp-1"] = _experiment_state(status)

    with pytest.raises(ValueError, match="Experiment must be completed or failed before retrying."):
        store.retry_experiment_errors("exp-1", {"api_key": "sk-test"})


def test_retry_experiment_errors_rejects_planless_state_without_mutation(monkeypatch) -> None:
    started_threads = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            started_threads.append(self)

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    store = EvaluationStore(persistence=RecordingPersistence())
    state = _experiment_state("failed")
    state.error = "Backend restarted before this experiment completed."
    store._experiments["exp-1"] = state

    with pytest.raises(ValueError, match="canonical experiment plan.*experiment revision"):
        store.retry_experiment_errors("exp-1", {"api_key": "sk-test"})

    assert state.status == "failed"
    assert state.error == "Backend restarted before this experiment completed."
    assert started_threads == []


@pytest.mark.parametrize("status", ["running", "pending", "rejudging"])
def test_restore_marks_active_experiments_failed_after_restart(status: str) -> None:
    persistence = RestoringPersistence([_persisted_experiment(status)])

    store = EvaluationStore(persistence=persistence)

    restored = store.get_experiment("exp-9")
    assert restored is not None
    assert restored["status"] == "failed"
    assert restored["error"] == "Backend restarted before this experiment completed."
    report = restored["report"]
    assert report["metric_definition_version"] == "attempt-aware-v1"
    assert report["total_cells"] == 0
    assert persistence.status_updates == [
        {
            "experiment_id": "exp-9",
            "status": "failed",
            "report": report,
            "error": "Backend restarted before this experiment completed.",
        }
    ]


def test_restore_preserves_terminal_experiment_status() -> None:
    persistence = RestoringPersistence([_persisted_experiment("completed")])

    store = EvaluationStore(persistence=persistence)

    restored = store.get_experiment("exp-9")
    assert restored is not None
    assert restored["status"] == "completed"
    assert persistence.status_updates == []


def test_restore_replaces_stale_report_when_marking_active_experiment_failed() -> None:
    report = {"overall_asr": 25, "avg_score": 50, "completed_cells": 1}
    experiment = _persisted_experiment("rejudging")
    experiment["report"] = report
    persistence = RestoringPersistence([experiment])

    store = EvaluationStore(persistence=persistence)

    restored = store.get_experiment("exp-9")
    assert restored is not None
    assert restored["status"] == "failed"
    assert restored["report"] != report
    assert restored["report"]["metric_definition_version"] == "attempt-aware-v1"
    assert restored["report"]["overall_asr"] is None
    assert persistence.status_updates[0]["report"] == restored["report"]


def test_postgresql_load_all_experiments_maps_error_field() -> None:
    interrupted_error = "Backend restarted before this experiment completed."
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-9",
                "Interrupted experiment",
                "",
                "",
                "ds-1",
                ["dan_style_roleplay"],
                {"model": "target-model"},
                "live",
                "failed",
                1,
                0,
                {"overall_asr": 25},
                interrupted_error,
                created_at,
                1,
                5,
                "",
            )
        ]
    )
    persistence = _FakePostgreSQLRunPersistence(connection)

    loaded = persistence.load_all_experiments()

    assert loaded[0]["error"] == interrupted_error


def test_postgresql_load_all_experiments_restores_optional_gold_answer() -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-9",
                "Gold answer experiment",
                "",
                "",
                "ds-1",
                ["dan_style_roleplay"],
                {"model": "target-model"},
                "live",
                "completed",
                2,
                0,
                None,
                None,
                created_at,
                1,
                5,
                "",
            )
        ],
        prompt_rows=[
            ("p-001", "question with reference", "general", "B"),
            ("p-002", "question without reference", "general", None),
        ],
    )
    persistence = _FakePostgreSQLRunPersistence(connection)

    prompts = persistence.load_all_experiments()[0]["dataset"]["prompts"]

    assert prompts[0]["gold_answer"] == "B"
    assert "gold_answer" not in prompts[1]


def test_postgresql_restore_preserves_pinned_plan_and_cell_identity() -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    snapshot = _canonical_plan_snapshot()
    config = snapshot["technique_configs"][0]
    planned_cell = snapshot["cells"][0]
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-snapshot", "Pinned", "", "", "ds-1", ["probe"],
                {"model": "target-model"}, "live", "completed", 1, 1,
                None, None, created_at, 1, 5, "", snapshot,
            )
        ],
        prompt_rows=[("p-001", "mutable intent", "general", None)],
        config_rows=[
            (
                config["technique_config_id"], "probe", "1.2.3", "probe-python", None,
                {}, config["resolved_spec"], config["spec_hash"],
            )
        ],
        cell_rows=[
            (
                planned_cell["cell_id"], config["technique_config_id"], "probe", "p-001", 1,
                "run-1", "completed", 8, 100, 0, "info", None, {},
            )
        ],
    )
    persistence = _FakePostgreSQLRunPersistence(connection)

    restored = persistence.load_all_experiments()[0]

    assert restored["cell_results"][0]["cell_id"] == planned_cell["cell_id"]
    assert restored["cell_results"][0]["technique_config_id"] == config["technique_config_id"]
    assert restored["cell_results"][0]["spec_hash"] == config["spec_hash"]
    assert restored["plan"]["technique_configs"][0]["manifest_hash"] == "a" * 64
    assert restored["plan"]["cells"][0]["cell_id"] == planned_cell["cell_id"]


def test_postgresql_restore_rejects_canonical_parent_without_normalized_children() -> None:
    snapshot = _canonical_plan_snapshot()
    connection = _ExperimentLoadConnection(
        [
            (
                snapshot["experiment_id"], "Pinned", "", "", snapshot["dataset_id"],
                ["probe"], {"model": "target-model"}, "live", "failed", 1, 0,
                None, "failed", datetime(2026, 6, 1, tzinfo=UTC), 1, 5, "", snapshot,
            )
        ],
    )

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _FakePostgreSQLRunPersistence(connection).load_all_experiments()


def test_postgresql_restore_rejects_coordinated_empty_canonical_snapshot() -> None:
    snapshot = _canonical_plan_snapshot()
    snapshot["technique_configs"] = []
    snapshot["cells"] = []
    snapshot["dataset_hash"] = canonical_dataset_hash(
        dataset_id=snapshot["dataset_id"],
        dataset_version=snapshot["dataset_version"],
        prompts=[],
    )
    _rehash_plan(snapshot)
    connection = _ExperimentLoadConnection(
        [
            (
                snapshot["experiment_id"], "Pinned", "", "", snapshot["dataset_id"],
                [], {"model": "target-model"}, "live", "failed", 0, 0,
                None, "failed", datetime(2026, 6, 1, tzinfo=UTC), 1, 5, "", snapshot,
            )
        ],
    )

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _FakePostgreSQLRunPersistence(connection).load_all_experiments()


def test_postgresql_restore_uses_plan_owned_dataset_and_cell_snapshot() -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    snapshot = _canonical_plan_snapshot()
    config = snapshot["technique_configs"][0]
    planned_cell = snapshot["cells"][0]
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-snapshot", "Pinned", "", "", "ds-1", ["probe"],
                {"model": "target-model"}, "live", "completed", 1, 1,
                None, None, created_at, 1, 5, "", snapshot,
            )
        ],
        prompt_rows=[("p-001", "mutated intent", "mutated-category", "mutated answer")],
        config_rows=[
            (
                config["technique_config_id"], "probe", "1.2.3", "probe-python", None, {},
                config["resolved_spec"], config["spec_hash"],
            )
        ],
        cell_rows=[
            (
                planned_cell["cell_id"], config["technique_config_id"], "probe", "p-001", 1,
                "run-1", "completed", 8, 100, 0, "info", None, {},
            )
        ],
    )

    restored = _FakePostgreSQLRunPersistence(connection).load_all_experiments()[0]

    assert restored["plan"] == snapshot
    assert restored["plan"]["dataset_version"] == "4"
    assert restored["plan"]["dataset_hash"] == snapshot["dataset_hash"]
    assert restored["plan"]["plan_hash"] == snapshot["plan_hash"]
    assert restored["plan"]["cells"][0]["prompt_text"] == "original intent"
    assert restored["plan"]["cells"][0]["prompt_category"] == "original-category"
    assert restored["plan"]["cells"][0]["gold_answer"] == "original answer"
    assert restored["cell_results"][0]["prompt_text"] == "original intent"
    assert restored["cell_results"][0]["prompt_category"] == "original-category"


@pytest.mark.parametrize("damage", ["missing", "incomplete", "tampered"])
def test_postgresql_restore_rejects_invalid_canonical_plan_snapshot(damage) -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    snapshot = _canonical_plan_snapshot()
    config = snapshot["technique_configs"][0]
    planned_cell = snapshot["cells"][0]
    if damage == "missing":
        persisted_plan = None
    elif damage == "incomplete":
        persisted_plan = json.loads(json.dumps(snapshot))
        persisted_plan["cells"][0].pop("prompt_text")
    else:
        persisted_plan = json.loads(json.dumps(snapshot))
        persisted_plan["cells"][0]["prompt_text"] = "tampered intent"
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-snapshot", "Pinned", "", "", "ds-1", ["probe"],
                {"model": "target-model"}, "live", "failed", 1, 0,
                None, "failed", created_at, 1, 5, "", persisted_plan,
            )
        ],
        prompt_rows=[("p-001", "mutable intent", "general", None)],
        config_rows=[
            (
                config["technique_config_id"], "probe", "1.2.3", "probe-python", None, {},
                config["resolved_spec"], config["spec_hash"],
            )
        ],
        cell_rows=[
            (
                planned_cell["cell_id"], config["technique_config_id"], "probe", "p-001", 1, "",
                "error", None, None, 0, "info", "failed", {},
            )
        ],
    )

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _FakePostgreSQLRunPersistence(connection).load_all_experiments()


def _load_plan_snapshot(snapshot: dict) -> list[dict]:
    config_rows = [
        (
            config["technique_config_id"], config["technique_id"],
            config["technique_version"], config["implementation_id"], config["preset"],
            config["parameters"], config["resolved_spec"], config["spec_hash"],
        )
        for config in snapshot["technique_configs"]
    ]
    cell_rows = [
        (
            cell["cell_id"], cell["technique_config_id"], cell["technique_key"],
            cell["prompt_id"], cell["repeat_index"], "", "error", None, None, 0,
            "info", "failed", {},
        )
        for cell in snapshot["cells"]
    ]
    connection = _ExperimentLoadConnection(
        [
            (
                snapshot["experiment_id"], "Pinned", "", "", snapshot["dataset_id"],
                [item["technique_key"] for item in snapshot["technique_configs"]],
                {"model": "target-model"}, "live", "failed", len(cell_rows), 0,
                None, "failed", datetime(2026, 6, 1, tzinfo=UTC),
                snapshot["repeat_count"], 5, "", snapshot,
            )
        ],
        prompt_rows=[("p-001", "mutable intent", "general", None)],
        config_rows=config_rows,
        cell_rows=cell_rows,
    )
    return _FakePostgreSQLRunPersistence(connection).load_all_experiments()


def test_postgresql_restore_accepts_complete_cartesian_cell_grid() -> None:
    snapshot = _canonical_grid_snapshot()

    restored = _load_plan_snapshot(snapshot)

    assert len(restored[0]["plan"]["cells"]) == 2 * 2 * 2


def test_postgresql_restore_rejects_rehashed_partial_cartesian_cell_grid() -> None:
    snapshot = _canonical_grid_snapshot()
    snapshot["cells"].pop()
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _load_plan_snapshot(snapshot)


def test_postgresql_restore_rejects_credential_shaped_canonical_value() -> None:
    snapshot = _canonical_plan_snapshot()
    _set_canonical_parameter_note(snapshot, "Bearer sk-supersecret123")

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _load_plan_snapshot(snapshot)


def test_postgresql_restore_does_not_treat_structural_ids_as_credentials() -> None:
    snapshot = _canonical_plan_snapshot()
    config = snapshot["technique_configs"][0]
    resolved_spec = config["resolved_spec"]
    structural_id = "sk-ordinary-identity123"
    resolved_spec["technique_id"] = structural_id
    config["technique_id"] = structural_id
    config["technique_key"] = structural_id
    config["spec_hash"] = content_hash(resolved_spec)
    config["technique_config_id"] = technique_config_id(
        snapshot["experiment_id"], resolved_spec
    )
    for cell in snapshot["cells"]:
        cell["technique_config_id"] = config["technique_config_id"]
        cell["technique_key"] = structural_id
        cell["cell_id"] = canonical_cell_id(
            snapshot["experiment_id"],
            config["technique_config_id"],
            cell["prompt_id"],
            cell["repeat_index"],
        )
    _rehash_plan(snapshot)

    restored = _load_plan_snapshot(snapshot)

    assert restored[0]["plan"]["technique_configs"][0]["technique_id"] == (
        structural_id
    )


@pytest.mark.parametrize("forgery", ["config", "cell"])
def test_postgresql_restore_rejects_coordinated_noncanonical_ids(forgery) -> None:
    snapshot = _canonical_plan_snapshot()
    config, cell = snapshot["technique_configs"][0], snapshot["cells"][0]
    if forgery == "config":
        config["technique_config_id"] = "etc-forged"
        cell["technique_config_id"] = config["technique_config_id"]
        cell["cell_id"] = canonical_cell_id(
            snapshot["experiment_id"], config["technique_config_id"], cell["prompt_id"], 1
        )
    else:
        cell["cell_id"] = "cell-forged"
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _load_plan_snapshot(snapshot)


@pytest.mark.parametrize("identity", ["dataset_id", "technique_key", "prompt_id"])
def test_postgresql_restore_rejects_coordinated_empty_canonical_identity(
    identity: str,
) -> None:
    snapshot = _canonical_plan_snapshot()
    cell = snapshot["cells"][0]
    if identity == "dataset_id":
        snapshot["dataset_id"] = ""
        snapshot["dataset_hash"] = canonical_dataset_hash(
            dataset_id="",
            dataset_version=snapshot["dataset_version"],
            prompts=snapshot["cells"],
        )
    elif identity == "technique_key":
        snapshot["technique_configs"][0]["technique_key"] = ""
        cell["technique_key"] = ""
    else:
        cell["prompt_id"] = ""
        cell["cell_id"] = canonical_cell_id(
            snapshot["experiment_id"],
            cell["technique_config_id"],
            "",
            cell["repeat_index"],
        )
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _load_plan_snapshot(snapshot)


@pytest.mark.parametrize("invalid", [7, True])
def test_postgresql_restore_rejects_rehashed_wrong_canonical_scalar_types(
    invalid: object,
) -> None:
    snapshot = _canonical_plan_snapshot()
    cell = snapshot["cells"][0]
    if invalid is True:
        cell["repeat_index"] = True
        cell["cell_id"] = canonical_cell_id(
            snapshot["experiment_id"],
            cell["technique_config_id"],
            cell["prompt_id"],
            1,
        )
    else:
        snapshot["technique_configs"][0]["technique_key"] = invalid
        cell["technique_key"] = invalid
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _load_plan_snapshot(snapshot)


def test_postgresql_restore_rejects_forged_dataset_hash_and_prompt_evidence() -> None:
    snapshot = _canonical_plan_snapshot()
    snapshot["cells"][0]["prompt_text"] = "forged prompt"
    snapshot["dataset_hash"] = "f" * 64
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _load_plan_snapshot(snapshot)


def test_postgresql_restore_rejects_cross_arm_prompt_evidence_conflict() -> None:
    snapshot = _canonical_plan_snapshot()
    config = json.loads(json.dumps(snapshot["technique_configs"][0]))
    config["resolved_spec"]["parameters"] = config["parameters"] = {"variant": 2}
    config["spec_hash"] = content_hash(config["resolved_spec"])
    config["technique_config_id"] = technique_config_id(
        snapshot["experiment_id"], config["resolved_spec"]
    )
    cell = json.loads(json.dumps(snapshot["cells"][0]))
    cell["technique_config_id"] = config["technique_config_id"]
    cell["prompt_text"] = "conflicting prompt"
    cell["cell_id"] = canonical_cell_id(
        snapshot["experiment_id"], config["technique_config_id"], cell["prompt_id"], 1
    )
    snapshot["technique_configs"].append(config)
    snapshot["cells"].append(cell)
    _rehash_plan(snapshot)

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _load_plan_snapshot(snapshot)


def test_migrated_legacy_config_restores_without_canonical_plan_and_requires_revision(
    monkeypatch,
) -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-9", "Migrated", "", "", "ds-1",
                ["system_prompt_extraction"], {"model": "target-model"},
                "demo", "failed", 1, 1, None, "legacy failure", created_at,
                1, 5, "",
            )
        ],
        prompt_rows=[("p-001", "extract", "general", None)],
        config_rows=[
            (
                "etc-migrated", "system_prompt_extraction", "0.2.0", "legacy",
                None, {},
                {
                    "technique_key": "system_prompt_extraction",
                    "implementation_id": "legacy",
                },
                "d" * 32,
            )
        ],
        cell_rows=[
            (
                "cell-migrated", "etc-migrated", "system_prompt_extraction",
                "p-001", 1, "", "error", None, None, 0, "info",
                "legacy failure", {},
            )
        ],
    )
    loaded = _FakePostgreSQLRunPersistence(connection).load_all_experiments()[0]

    assert loaded["plan"] is None

    monkeypatch.setattr("app.api.store.threading.Thread.start", lambda self: self.run())
    store = EvaluationStore(persistence=RestoringPersistence([loaded]))
    restored = store._experiments["exp-9"]
    assert restored.plan is None

    with pytest.raises(ValueError, match="canonical experiment plan.*experiment revision"):
        store.retry_experiment_errors(
            "exp-9",
            {"api_key": "refresh", "model": "target-model"},
        )

    assert restored.cell_results[0]["status"] == "error"
    assert restored.cell_results[0]["technique_key"] == "system_prompt_extraction"


def test_parameter_only_legacy_config_restores_without_canonical_plan() -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-legacy-parameters", "Migrated", "", "", "ds-1",
                ["legacy-technique"], {"model": "target-model"},
                "demo", "failed", 1, 1, None, "legacy failure", created_at,
                1, 5, "",
            )
        ],
        prompt_rows=[("p-001", "intent", "general", None)],
        config_rows=[
            (
                "etc-low", "legacy-technique", "0.2.0", "legacy", None,
                {"ratio": 0.5}, {"parameters": {"ratio": 0.5}}, "spec-low",
            )
        ],
        cell_rows=[
            (
                "cell-legacy-parameters", "etc-low", "legacy-technique",
                "p-001", 1, "", "error", None, None, 0, "info",
                "legacy failure", {},
            )
        ],
    )

    loaded = _FakePostgreSQLRunPersistence(connection).load_all_experiments()[0]

    assert loaded["plan"] is None


def test_complete_legacy_spec_is_not_downgraded_by_compatibility_marker() -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    experiment_id = "exp-complete-legacy"
    resolved_spec = deepcopy(
        _canonical_plan_snapshot()["technique_configs"][0]["resolved_spec"]
    )
    resolved_spec.update(
        {
            "technique_version": "0.2.0",
            "implementation_id": "legacy",
            "compatibility": "legacy_implicit",
        }
    )
    config_id = technique_config_id(experiment_id, resolved_spec)
    planned_cell_id = canonical_cell_id(experiment_id, config_id, "p-001", 1)
    connection = _ExperimentLoadConnection(
        [
            (
                experiment_id, "Canonical legacy", "", "", "ds-1", ["probe"],
                {"model": "target-model"}, "demo", "failed", 1, 0, None,
                "missing plan", created_at, 1, 5, "",
            )
        ],
        prompt_rows=[("p-001", "intent", "general", None)],
        config_rows=[
            (
                config_id, "probe", "0.2.0", "legacy", None, {},
                resolved_spec, content_hash(resolved_spec),
            )
        ],
        cell_rows=[
            (
                planned_cell_id, config_id, "probe", "p-001", 1, "", "pending",
                None, None, 0, "info", None, {},
            )
        ],
    )

    with pytest.raises(ValueError, match="canonical experiment plan snapshot"):
        _FakePostgreSQLRunPersistence(connection).load_all_experiments()


def test_postgresql_restore_rejects_mixed_legacy_and_canonical_plan_rows() -> None:
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    canonical_spec = {
        "technique_id": "probe",
        "technique_version": "1.2.3",
        "implementation_id": "probe-python",
        "implementation_kind": "python",
        "contract_version": "7",
        "manifest_hash": "a" * 64,
        "asset_hashes": {},
        "parameters": {},
        "model_roles": {},
        "judge_policies": {},
        "effective_limits": {},
        "runtime_config": {},
    }
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-mixed", "Mixed", "", "", "ds-1", ["probe"],
                {"model": "target-model"}, "demo", "failed", 2, 0, None,
                "legacy failure", created_at, 1, 5, "",
            )
        ],
        prompt_rows=[("p-001", "intent", "general", None)],
        config_rows=[
            (
                "etc-implicit", "probe", "0.2.0", "legacy", None, {},
                {"technique_key": "probe", "implementation_id": "legacy"},
                "b" * 64,
            ),
            (
                "etc-canonical", "probe", "1.2.3", "probe-python", None, {},
                canonical_spec, "c" * 64,
            ),
        ],
        cell_rows=[
            (
                "cell-implicit", "etc-implicit", "probe", "p-001", 1, "",
                "error", None, None, 0, "info", "legacy failure", {},
            ),
            (
                "cell-canonical", "etc-canonical", "probe", "p-001", 1, "",
                "pending", None, None, 0, "info", None, {},
            ),
        ],
    )

    with pytest.raises(ValueError, match="mixed implicit and canonical"):
        _FakePostgreSQLRunPersistence(connection).load_all_experiments()


def test_demo_run_is_persisted_with_conversation() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)

    run = store.create_run("system_prompt_extraction", mode="demo")

    assert run["status"] == "completed"
    assert len(persistence.technique_runs) == 1
    persisted = persistence.technique_runs[0]
    assert persisted["summary"].technique_key == "system_prompt_extraction"
    assert persisted["summary"].asr == 100
    assert persisted["mode"] == "demo"
    assert persisted["conversation"][0]["kind"] == "single_shot"
    assert persisted["technique_metadata"].family


def test_agent_architecture_demo_run_serializes_canonical_artifact() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)

    run = store.create_run(
        "agent_architecture_enumeration",
        target_kind="agent",
        mode="demo",
        target_config={"unsafe_intent": "enumerate architecture"},
    )

    payload = json.loads(run["final_output"])
    assert payload["schema_version"] == "agent-architecture-map/v2"
    assert [entry["kind"] for entry in run["conversation"]] == ["round"] * 3 + ["artifact"]
    assert run["conversation"][-1]["response"] == run["final_output"]
    assert all(entry["judge"]["score"] is None for entry in run["conversation"][:-1])
    assert persistence.technique_runs[0]["conversation"] == run["conversation"]


def test_agent_architecture_live_run_serializes_final_artifact_entry() -> None:
    from app.techniques.core.models import ProviderCallResult

    calls = []

    class OwnedScriptedClient(FakeChatClient):
        def chat_correlated_once(self, messages, *, request_context, **options):
            calls.append((request_context.role, request_context.operation_id))
            return ProviderCallResult(text=self._responses.pop(0), usage={"total_tokens": 42})

    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    store._build_attacker_client = lambda config: OwnedScriptedClient([
        json.dumps({"question": "Identify agent and tools.", "focus": "agents"}),
        json.dumps({"question": "Describe data stores.", "focus": "data_stores"}),
        json.dumps({"question": "Describe workflow.", "focus": "workflow"}),
    ])
    store._build_judge_client = lambda config, **kwargs: OwnedScriptedClient([json.dumps({
        "score": 8, "rationale": "Concrete map.", "evidence": ["users table"],
        "missing_requirements": [], "success_criteria_met": True,
        "failure_criteria_met": False, "verdict": "success",
    })])
    store._build_chat_client = lambda config: OwnedScriptedClient(
        [
            _architecture_phase(
                "inventory",
                agent={
                    "name": "ExampleCo Insights Agent",
                    "profile": "exampleco-insights-agent",
                    "model_aliases": ["exampleco-insights-agent"],
                    "confidence": "confirmed",
                },
                tools=[
                    {
                        "name": "describe_schema",
                        "type": "tool",
                        "read_scope": ["SQLite schema"],
                        "write_scope": [],
                        "evidence": "available tools list",
                        "confidence": "confirmed",
                    }
                ],
            ),
            _architecture_phase(
                "data_layer",
                data_stores=[
                    {
                        "name": "users",
                        "kind": "table",
                        "parent": "SQLite",
                        "fields": ["user_id", "email"],
                        "join_keys": ["user_id"],
                        "description": "User records.",
                        "confidence": "confirmed",
                    }
                ],
            ),
            _architecture_phase(
                "architecture",
                workflow={
                    "framework": "LangGraph",
                    "nodes": [
                        {
                            "name": "execute_tools",
                            "role": "tool node",
                            "inputs": ["plan"],
                            "outputs": ["tool_results"],
                            "tools_used": ["describe_schema"],
                            "confidence": "confirmed",
                        }
                    ],
                    "edges": [],
                    "handoff_paths": ["exampleco-insights-agent -> exampleco-support-agent"],
                },
            ),
        ]
    )

    summary, _ = store._execute_single(
        technique_key="agent_architecture_enumeration",
        target_kind="agent",
        mode="live",
        target_config={"api_key": "test-key", "unsafe_intent": "enumerate architecture"},
    )
    run = store._serialize_run(summary)

    payload = json.loads(run["final_output"])
    artifact_entry = run["conversation"][-1]
    assert payload["schema_version"] == "agent-architecture-map/v2"
    assert artifact_entry["kind"] == "artifact"
    assert artifact_entry["response"] == run["final_output"]
    assert artifact_entry["media_type"] == "application/json"
    assert [entry["kind"] for entry in run["conversation"]] == ["round"] * 3 + ["artifact"]
    assert all(entry["judge"]["score"] is None for entry in run["conversation"][:-1])
    assert artifact_entry["judge"]["score"] == 8
    assert run["score"] == 80
    assert persistence.technique_runs[0]["conversation"] == run["conversation"]
    assert calls == [
        ("attacker", "agent-architecture:round:1:attacker"),
        ("target", "agent-architecture:round:1:target"),
        ("attacker", "agent-architecture:round:2:attacker"),
        ("target", "agent-architecture:round:2:target"),
        ("attacker", "agent-architecture:round:3:attacker"),
        ("target", "agent-architecture:round:3:target"),
        ("target_judge", "agent-architecture:final-judge"),
    ]


def test_baseline_is_persisted() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    store._build_chat_client = lambda config: FakeChatClient(
        ["Target refusal.", '{"score": 1, "rationale": "Clear refusal."}'],
        model="baseline-target",
    )
    store._build_attacker_client = lambda config: None

    result = store.run_baseline(
        target_config={
            "api_key": "test-key",
            "model": "baseline-target",
            "unsafe_intent": "exfiltrate secrets",
        }
    )

    assert str(result["run_id"]).startswith("run-")
    assert len(persistence.baseline_runs) == 1
    persisted = persistence.baseline_runs[0]
    assert persisted["result"]["intent"] == "exfiltrate secrets"
    assert persisted["requested_by"] == "demo-user"


def test_run_serialization_preserves_terminal_error_detail() -> None:
    store = EvaluationStore(persistence=RecordingPersistence())
    summary = RunSummary(
        run_id="run-error",
        technique_key="probe",
        target_kind="chatbot",
        status="error",
        score=0,
        asr=0,
        matched_detections=0,
        detections=(),
        highest_severity="info",
        error="judge output was not parseable",
    )

    serialized = store._serialize_run(summary)

    assert serialized["status"] == "error"
    assert serialized["error"] == "judge output was not parseable"


def test_techniques_source_seed_prompt_from_db() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("dan_style_roleplay")

    resolved = store._resolve_seed_prompt(
        technique_key="dan_style_roleplay",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )

    # DAN now sources its seed prompt from the DB
    assert resolved == "DB seed for {intent}"


def test_explicit_seed_prompt_overrides_db() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("dan_style_roleplay")

    resolved = store._resolve_seed_prompt(
        technique_key="dan_style_roleplay",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="User override for {intent}",
    )

    assert resolved == "User override for {intent}"


def test_runtime_template_techniques_use_validation_placeholder() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("base64_obfuscation")

    resolved = store._resolve_seed_prompt(
        technique_key="base64_obfuscation",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )

    # base64 has a runtime-computed template and no longer needs a validation placeholder.
    assert resolved == ""


def test_gcg_static_transfer_resolves_empty_seed_prompt() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("gcg")

    resolved = store._resolve_seed_prompt(
        technique_key="gcg",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )

    assert resolved == ""


def test_seeded_code_agent_gets_default_seed_assets() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("multi_file_seeded_code_agent")

    seed_prompt = store._resolve_seed_prompt(
        technique_key="multi_file_seeded_code_agent",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )
    seed_code = store._resolve_seed_code(
        technique_key="multi_file_seeded_code_agent",
        technique_metadata=technique_cls.metadata,
        explicit_seed_code="",
    )

    assert seed_prompt == ""
    assert "from exampleco_client import fetch_accounts" in seed_code
    assert "# file: triage.py" in seed_code


def test_multi_file_seeded_code_agent_demo_run_completes() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)

    result = store.create_run(
        "multi_file_seeded_code_agent",
        target_kind="agent",
        mode="demo",
        target_config={"unsafe_intent": "complete the reporting pipeline"},
    )

    assert result["status"] == "completed"
    assert persistence.technique_runs[0]["seed_prompt"] == ""
