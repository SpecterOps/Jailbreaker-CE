from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import pytest

from app.persistence.technique_runtime import (
    PostgreSQLTechniqueRuntimePersistence,
    TechniquePackageRecord,
)
from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.execution import CorrelationIds


DATABASE_URL = os.environ.get("TASK20_DATABASE_URL", "")
FIXTURES = Path(__file__).parent / "fixtures" / "techniques"


def _record() -> TechniquePackageRecord:
    return TechniquePackageRecord(
        package_id="task20-package",
        technique_id="task20-probe",
        technique_version="1.0.0",
        implementation_id="task20-modular",
        implementation_kind="workflow",
        contract_version="1",
        source_kind="source_controlled",
        source_location="/source/task20-probe",
        state="active",
        manifest_hash="a" * 64,
        asset_hashes={"workflow.yaml": "b" * 64},
        diagnostics=(
            {
                "stage": "discovery",
                "message": "validated",
                "source_location": "/source/task20-probe",
                "path": "",
            },
        ),
    )


def _reset_database(dsn: str) -> None:
    import psycopg

    repo_root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((repo_root / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        connection.execute(
            "TRUNCATE execution_events, execution_checkpoints, model_calls, "
            "run_artifacts, experiment_cell_attempts, experiment_cells, "
            "experiment_technique_configs, experiments, evaluation_runs, "
            "technique_versions, technique_packages CASCADE"
        )


def _execution_event(event_id: str, sequence: int) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=event_id,
        correlation=CorrelationIds(
            experiment_id=None,
            cell_id=None,
            run_id="task23-batch-run",
            attempt_id="task23-batch-attempt",
        ),
        sequence=sequence,
        event_type="legacy.trace",
        schema_version=1,
        timestamp=datetime(2026, 8, 26, tzinfo=UTC),
        technique_id="task23-probe",
        technique_version="1.0.0",
        implementation_id="task23-modular",
        payload={"sequence": sequence},
    )


@pytest.mark.skipif(not DATABASE_URL, reason="requires disposable Task 20 PostgreSQL")
def test_postgresql_execution_event_batch_is_atomic_and_idempotent() -> None:
    import psycopg

    _reset_database(DATABASE_URL)
    persistence = PostgreSQLTechniqueRuntimePersistence(DATABASE_URL)
    existing = _execution_event("task23-existing", 1)
    persistence.append_event(existing)
    valid_prefix = _execution_event("task23-valid-prefix", 2)
    colliding_tail = _execution_event("task23-colliding-tail", 1)

    with pytest.raises(psycopg.errors.UniqueViolation):
        persistence.append_events((valid_prefix, colliding_tail))

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        assert connection.execute(
            "SELECT event_id, sequence FROM execution_events WHERE run_id = %s ORDER BY sequence",
            (existing.correlation.run_id,),
        ).fetchall() == [(existing.event_id, 1)]

    completed = (
        valid_prefix,
        _execution_event("task23-valid-tail", 3),
    )
    persistence.append_events(completed)
    persistence.append_events(completed)

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        assert connection.execute(
            "SELECT event_id, sequence FROM execution_events WHERE run_id = %s ORDER BY sequence",
            (existing.correlation.run_id,),
        ).fetchall() == [
            (existing.event_id, 1),
            (valid_prefix.event_id, 2),
            (completed[1].event_id, 3),
        ]


@pytest.mark.skipif(not DATABASE_URL, reason="requires disposable Task 20 PostgreSQL")
def test_postgresql_package_upsert_diagnostics_identity_lookup_and_exact_references() -> None:
    import psycopg

    _reset_database(DATABASE_URL)
    persistence = PostgreSQLTechniqueRuntimePersistence(DATABASE_URL)
    original = _record()
    updated = replace(
        original,
        state="disabled",
        diagnostics=(
            *original.diagnostics,
            {
                "stage": "activation",
                "message": "historical failure",
                "source_location": original.source_location,
                "path": "",
            },
        ),
    )
    persistence.save_package(original)
    persistence.save_package(updated)

    loaded = persistence.load_package_by_reference(
        technique_id=original.technique_id,
        technique_version=original.technique_version,
        implementation_id=original.implementation_id,
    )

    assert loaded == updated
    with pytest.raises(ValueError, match="same version has different contents"):
        persistence.save_package(
            replace(
                updated,
                asset_hashes={"workflow.yaml": "c" * 64},
                diagnostics=(*updated.diagnostics, {"stage": "must-rollback"}),
            )
        )
    assert persistence.load_package(original.package_id) == updated

    mismatched_reference = {
        "technique_id": original.technique_id,
        "technique_version": "9.9.9",
        "implementation_id": original.implementation_id,
    }
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO experiments (experiment_id, name, plan) VALUES (%s, %s, %s::jsonb)",
            (
                "task20-reference",
                "Task 20 reference",
                json.dumps({"technique_configs": [mismatched_reference]}),
            ),
        )
        connection.execute(
            "INSERT INTO experiment_technique_configs ("
            "technique_config_id, experiment_id, technique_id, "
            "technique_version, implementation_id, spec_hash) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                "task20-config",
                "task20-reference",
                original.technique_id,
                "9.9.9",
                original.implementation_id,
                "task20-spec",
            ),
        )
    assert not persistence.has_package_references(
        technique_id=original.technique_id,
        technique_version=original.technique_version,
        implementation_id=original.implementation_id,
    )

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(
            "UPDATE experiment_technique_configs SET technique_version = %s WHERE technique_config_id = %s",
            (original.technique_version, "task20-config"),
        )
    assert persistence.has_package_references(
        technique_id=original.technique_id,
        technique_version=original.technique_version,
        implementation_id=original.implementation_id,
    )

    exact_reference = {
        "technique_id": original.technique_id,
        "technique_version": original.technique_version,
        "implementation_id": original.implementation_id,
    }
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM experiment_technique_configs WHERE technique_config_id = %s",
            ("task20-config",),
        )
        connection.execute(
            "UPDATE experiments SET plan = %s::jsonb WHERE experiment_id = %s",
            (json.dumps({"technique_configs": [exact_reference]}), "task20-reference"),
        )
    assert persistence.has_package_references(
        technique_id=original.technique_id,
        technique_version=original.technique_version,
        implementation_id=original.implementation_id,
    )


@pytest.mark.parametrize(
    ("transition", "expected_state"),
    [("disable", "disabled"), ("soft_uninstall", "uninstalled")],
)
@pytest.mark.skipif(not DATABASE_URL, reason="requires disposable Task 20 PostgreSQL")
def test_postgresql_same_path_conflict_preserves_archived_package_row(
    tmp_path: Path,
    transition: str,
    expected_state: str,
) -> None:
    api = import_module("app.techniques.plugins.loader")
    catalog_type = import_module("app.techniques.core.catalog").TechniqueCatalog
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    _reset_database(DATABASE_URL)
    persistence = PostgreSQLTechniqueRuntimePersistence(DATABASE_URL)
    root = tmp_path / "packages"
    package_root = root / "probe"
    shutil.copytree(FIXTURES / "valid_probe", package_root)
    workflow_path = package_root / "workflow.yaml"
    original_workflow = workflow_path.read_bytes()
    first = api.TechniquePackageLoader(
        catalog=catalog_type(),
        roots=(root,),
        persistence=persistence,
    )
    original = first.discover().by_id("probe")
    assert original.package is not None
    first.activate(original.package)
    getattr(first, transition)(original.reference)
    canonical_id = api._package_id(package_root)
    archived = persistence.load_package(canonical_id)
    assert archived is not None
    assert archived.state == expected_state

    workflow_path.write_text(
        "version: conflicting-postgresql-asset\nsteps: []\n",
        encoding="utf-8",
    )
    conflicting_loader = api.TechniquePackageLoader(
        catalog=catalog_type(),
        roots=(root,),
        persistence=persistence,
    )
    conflicting = conflicting_loader.discover().by_id("probe")

    assert persistence.load_package(canonical_id) == archived
    assert conflicting.package is not None
    with pytest.raises(ValueError, match="same version has different contents"):
        conflicting_loader.activate(
            conflicting.package,
            preserve_lifecycle_state=True,
        )
    assert persistence.load_package(canonical_id) == archived
    conflict_record = persistence.load_package(api._conflict_package_id(conflicting.package))
    assert conflict_record is not None
    assert conflict_record.state == api.PackageState.INVALID.value
    assert conflict_record.diagnostics[-1]["stage"] == "activation"

    workflow_path.write_bytes(original_workflow)
    restored_loader = api.TechniquePackageLoader(
        catalog=catalog_type(),
        roots=(root,),
        persistence=persistence,
    )
    restored = restored_loader.discover().by_id("probe")

    assert restored.state.value == expected_state
    assert restored.package is not None
    restored_loader.activate(restored.package, preserve_lifecycle_state=True)
    canonical = persistence.load_package_by_reference(
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="modular",
    )
    assert canonical is not None
    assert canonical.package_id == canonical_id
    assert canonical.state == expected_state
    with pytest.raises(KeyError, match="not registered"):
        restored_loader.catalog.resolve(selector_type("probe"))
    assert restored_loader.catalog.resolve(selector_type("probe", "1.0.0", "modular")).implementation.id == "modular"


@pytest.mark.skipif(not DATABASE_URL, reason="requires disposable Task 20 PostgreSQL")
def test_postgresql_loader_hard_delete_after_rediscovery_removes_fk_rows(
    tmp_path: Path,
) -> None:
    api = import_module("app.techniques.plugins.loader")
    catalog_type = import_module("app.techniques.core.catalog").TechniqueCatalog
    _reset_database(DATABASE_URL)
    persistence = PostgreSQLTechniqueRuntimePersistence(DATABASE_URL)
    root = tmp_path / "packages"
    package_root = root / "probe"
    shutil.copytree(FIXTURES / "valid_probe", package_root)
    loader = api.TechniquePackageLoader(
        catalog=catalog_type(),
        roots=(root,),
        persistence=persistence,
    )
    discovered = loader.discover().by_id("probe")
    assert discovered.package is not None
    loader.activate(discovered.package, make_default=True)
    loader.discover()

    loader.hard_delete_unreferenced(discovered.reference)

    assert persistence.load_package(api._package_id(package_root)) is None
    assert (
        persistence.load_package_by_reference(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )
        is None
    )
    assert not loader.catalog.contains("probe")


@pytest.mark.skipif(not DATABASE_URL, reason="requires disposable Task 20 PostgreSQL")
def test_postgresql_hard_delete_rechecks_references_under_lock_and_is_atomic() -> None:
    import psycopg

    _reset_database(DATABASE_URL)
    persistence = PostgreSQLTechniqueRuntimePersistence(DATABASE_URL)
    record = _record()
    persistence.save_package(record)
    lock_attempted = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

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
            if "LOCK TABLE experiments" in str(query):
                lock_attempted.set()
            return self.connection.execute(query, params)

    class SignalingPersistence(PostgreSQLTechniqueRuntimePersistence):
        def _connect(self):
            return SignalingConnection(super()._connect())

    signaling = SignalingPersistence(DATABASE_URL)
    signaling._schema_ready = True

    def hard_delete() -> None:
        try:
            signaling.hard_delete_package(record)
        except BaseException as error:  # pragma: no cover - relayed from thread
            errors.append(error)
        finally:
            finished.set()

    exact_reference = {
        "technique_id": record.technique_id,
        "technique_version": record.technique_version,
        "implementation_id": record.implementation_id,
    }
    with psycopg.connect(DATABASE_URL) as blocker:
        blocker.execute("LOCK TABLE experiments IN ROW EXCLUSIVE MODE")
        worker = threading.Thread(target=hard_delete)
        worker.start()
        try:
            assert lock_attempted.wait(timeout=5)
            assert not finished.is_set()
            blocker.execute(
                "INSERT INTO experiments (experiment_id, name, plan) VALUES (%s, %s, %s::jsonb)",
                (
                    "task20-locked-reference",
                    "Task 20 locked reference",
                    json.dumps({"technique_configs": [exact_reference]}),
                ),
            )
            blocker.commit()
        finally:
            worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert persistence.load_package(record.package_id) == record

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM experiments WHERE experiment_id = %s",
            ("task20-locked-reference",),
        )
    persistence.hard_delete_package(record)

    assert persistence.load_package(record.package_id) is None
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM technique_versions "
                "WHERE technique_id = %s AND technique_version = %s "
                "AND implementation_id = %s",
                (
                    record.technique_id,
                    record.technique_version,
                    record.implementation_id,
                ),
            ).fetchone()[0]
            == 0
        )
