"""Recovery checks use persisted evidence, never an unmerged child journal."""

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from app.persistence.technique_runtime import (
    ModelCallRecord,
    NoopTechniqueRuntimePersistence,
    PostgreSQLTechniqueRuntimePersistence,
)
from app.techniques.contracts.definition import ImplementationKind
from app.techniques.contracts.execution import CorrelationIds, ModelRoleBinding, ResolvedRunSpec
from app.techniques.core.events import EventJournal, TechniqueEventIdentity
from app.techniques.core.models import ModelService, ProviderCallResult
from app.techniques.core.redaction import RedactionService


@pytest.fixture(params=("memory", "postgres"))
def runtime(request):
    if request.param == "memory":
        persistence = NoopTechniqueRuntimePersistence()
        yield persistence, lambda: persistence
        return
    dsn = os.environ.get("TASK20_DATABASE_URL")
    if not dsn:
        pytest.skip("requires disposable PostgreSQL")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    schema = f"call_recovery_{uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated = make_conninfo(dsn, options=f"-c search_path={schema}")
        yield PostgreSQLTechniqueRuntimePersistence(isolated), lambda: PostgreSQLTechniqueRuntimePersistence(isolated)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def call_record():
    return ModelCallRecord(
        model_call_id="saved-call",
        correlation=CorrelationIds(
            experiment_id=None,
            cell_id=None,
            run_id="saved-run",
            attempt_id="saved-attempt",
        ),
        role="target",
        operation_id="branch:model:1",
        operation_attempt=1,
        request={"messages": []},
        request_hash="fixture-hash",
    )


def journal_for(persistence, call):
    return EventJournal(
        correlation=call.correlation,
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
    )


def finish_call(persistence, call, status):
    if status == "completed":
        persistence.complete_model_call(
            call.model_call_id,
            response="fixture response",
            usage={},
            provider_response_id=None,
            gateway_call_id=None,
            gateway_attempts={},
        )
    elif status == "failed":
        persistence.fail_model_call(
            call.model_call_id,
            category="timeout",
            error="fixture timeout",
            indeterminate=True,
        )


def emit_call_event(journal, call, event_type):
    return journal.emit(
        event_type,
        operation_id=call.operation_id,
        attempt=call.operation_attempt,
        payload={"model_call_id": call.model_call_id, "role": call.role},
    )


@pytest.mark.parametrize("status", ("requested", "completed", "failed"))
def test_unmerged_lifecycle_is_unreconciled_after_reopening_persistence(runtime, status):
    persistence, reopen = runtime
    call = call_record()
    journal = journal_for(persistence, call)
    persistence.record_model_call_requested(call)
    buffer = journal.new_buffer(branch_id="child")
    with journal.activate(buffer):
        emit_call_event(journal, call, "model.requested")
        finish_call(persistence, call, status)
        if status != "requested":
            emit_call_event(journal, call, f"model.{status}")

    recovered = reopen()
    assert recovered.attempt_has_unreconciled_call_events("saved-attempt") is True
    assert recovered.attempt_has_unreconciled_call_events("other-attempt") is False
    journal.merge_many((buffer,))
    assert recovered.attempt_has_unreconciled_call_events("saved-attempt") is (status == "requested")


@pytest.mark.parametrize("missing", ("requested", "terminal"))
def test_partial_root_lifecycle_requires_recovery(runtime, missing):
    persistence, reopen = runtime
    call = call_record()
    persistence.record_model_call_requested(call)
    finish_call(persistence, call, "completed")
    journal = journal_for(persistence, call)
    event_type = "model.completed" if missing == "requested" else "model.requested"
    emit_call_event(journal, call, event_type)
    assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is True


@pytest.mark.parametrize(
    "mismatch", ("run", "attempt", "experiment", "cell", "operation", "operation_attempt", "role", "terminal", "order")
)
def test_other_or_inconsistent_lifecycle_cannot_hide_recovery_gap(runtime, mismatch):
    persistence, reopen = runtime
    call = call_record()
    persistence.record_model_call_requested(call)
    finish_call(persistence, call, "completed")
    event_call = call
    if mismatch in {"run", "attempt", "experiment", "cell"}:
        event_call = replace(call, correlation=replace(call.correlation, **{f"{mismatch}_id": "other"}))
    elif mismatch == "operation":
        event_call = replace(call, operation_id="other-operation")
    elif mismatch == "operation_attempt":
        event_call = replace(call, operation_attempt=2)
    elif mismatch == "role":
        event_call = replace(call, role="attacker")
    journal = journal_for(persistence, event_call)
    types = ("model.requested", "model.failed" if mismatch == "terminal" else "model.completed")
    if mismatch == "order":
        types = tuple(reversed(types))
    for event_type in types:
        emit_call_event(journal, event_call, event_type)
    assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is True


def test_failed_event_batch_does_not_clear_recovery_gap(runtime):
    persistence, reopen = runtime
    call = call_record()
    journal = journal_for(persistence, call)
    persistence.record_model_call_requested(call)
    finish_call(persistence, call, "completed")
    existing = journal.emit("run.started")
    buffer = journal.new_buffer(branch_id="child")
    with journal.activate(buffer):
        requested = emit_call_event(journal, call, "model.requested")
        completed = emit_call_event(journal, call, "model.completed")
    if isinstance(persistence, NoopTechniqueRuntimePersistence):
        expected_error = ValueError
    else:
        from psycopg.errors import UniqueViolation

        expected_error = UniqueViolation
    with pytest.raises(expected_error):
        persistence.append_events((replace(requested, sequence=2), replace(completed, sequence=existing.sequence)))
    assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is True
    journal.merge_many((buffer,))
    assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is False


def test_real_model_gateway_leaves_recovery_gap_until_root_merge(runtime):
    persistence, reopen = runtime
    call = call_record()
    journal = journal_for(persistence, call)
    dispatched = []

    class Client:
        def chat_correlated_once(self, messages, *, request_context, **options):
            dispatched.append(request_context.model_call_id)
            return ProviderCallResult(text="fixture response")

    spec = ResolvedRunSpec(
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="modular",
        implementation_kind=ImplementationKind.WORKFLOW,
        contract_version="1",
        manifest_hash="a" * 64,
        model_role_bindings={
            "target": ModelRoleBinding.from_profile(
                logical_role="target",
                profile_id="test-profile",
                provider="openai",
                model="test-model",
            )
        },
    )
    service = ModelService(
        correlation=call.correlation,
        spec=spec,
        clients={"target": Client()},
        persistence=persistence,
        events=journal,
        redaction=RedactionService(),
    )
    parent = journal.new_buffer(branch_id="parent")
    with journal.activate(parent):
        child = journal.new_buffer(branch_id="child")
        with journal.activate(child):
            result = asyncio.run(
                service.call(role="target", messages=({"role": "user", "content": "test"},), operation_id="target:1")
            )
        journal.merge_many((child,))
        assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is True
    assert dispatched == [result.model_call_id]
    assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is True
    journal.merge_many((parent,))
    assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is False
    assert dispatched == [result.model_call_id]


@pytest.mark.parametrize("runtime", ("postgres",), indirect=True)
def test_process_exit_before_merge_leaves_durable_recovery_evidence(runtime):
    persistence, reopen = runtime
    # The child loses its entire local journal, without running cleanup code.
    # Only committed database rows are available to the recovering process.
    script = """
import os
import runpy
from app.persistence.technique_runtime import PostgreSQLTechniqueRuntimePersistence
helpers = runpy.run_path('tests/test_model_call_recovery.py')
persistence = PostgreSQLTechniqueRuntimePersistence(os.environ['RECOVERY_TEST_DSN'])
call = helpers['call_record']()
journal = helpers['journal_for'](persistence, call)
persistence.record_model_call_requested(call)
buffer = journal.new_buffer(branch_id='lost-child')
with journal.activate(buffer):
    helpers['emit_call_event'](journal, call, 'model.requested')
    helpers['finish_call'](persistence, call, 'completed')
    helpers['emit_call_event'](journal, call, 'model.completed')
    os._exit(17)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "RECOVERY_TEST_DSN": persistence._dsn},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 17, result.stderr
    assert reopen().attempt_has_unreconciled_call_events("saved-attempt") is True
    with persistence._connect() as connection:
        assert connection.execute("SELECT status, response_text FROM model_calls").fetchall() == [
            ("completed", "fixture response"),
        ]
        assert connection.execute("SELECT count(*) FROM execution_events").fetchone() == (0,)
