import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "backend")

from app.api.store import EvaluationStore
from app.persistence import NoopRunPersistence


def _store() -> EvaluationStore:
    store = EvaluationStore(persistence=NoopRunPersistence())
    store._discover_analysis_views = lambda: [
        "analysis_run_overview",
        "analysis_model_summary",
        "analysis_custom_rollup",
    ]
    return store


def test_analysis_sql_allows_direct_selects_from_approved_relations() -> None:
    store = _store()

    normalized = store._normalize_analysis_sql(
        """
        SELECT c.experiment_id, r.target_model, c.metadata->'rejudge_results' AS rejudge_results
        FROM experiment_cells c
        JOIN evaluation_runs r ON r.run_id = c.run_id
        WHERE r.final_output ILIKE '%delete%'
        ORDER BY c.experiment_id
        LIMIT 25;
        """
    )

    assert store._classify_analysis_sql(normalized) == "query"


def test_analysis_sql_allows_basic_reporting_aggregates() -> None:
    store = _store()

    normalized = store._normalize_analysis_sql(
        "SELECT target_model, COUNT(*), ROUND(AVG(score), 2) FROM evaluation_runs GROUP BY target_model"
    )

    assert store._classify_analysis_sql(normalized) == "query"


def test_analysis_sql_rejects_create_view_before_database_access() -> None:
    store = _store()
    store._analysis_connection = lambda: pytest.fail("invalid SQL should not open a DB connection")

    with pytest.raises(ValueError, match="Creating analysis views"):
        store.run_analysis_query(
            "CREATE VIEW analyst_backdoor AS SELECT * FROM evaluation_runs"
        )


def test_analysis_sql_rejects_ctes_that_can_hide_writes() -> None:
    store = _store()

    with pytest.raises(ValueError, match="WITH/CTE"):
        store._classify_analysis_sql(
            store._normalize_analysis_sql(
                "WITH deleted AS (DELETE FROM evaluation_runs RETURNING *) SELECT * FROM deleted"
            )
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * INTO exported_runs FROM evaluation_runs",
        "SELECT * FROM evaluation_runs; DROP TABLE evaluation_runs",
        "SELECT pg_read_file('/etc/passwd') FROM evaluation_runs",
        "SELECT current_setting('server_version') FROM evaluation_runs",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM (SELECT * FROM evaluation_runs) nested",
        "UPDATE evaluation_runs SET status = 'completed'",
    ],
)
def test_analysis_sql_rejects_unsafe_or_unapproved_queries(sql: str) -> None:
    store = _store()

    with pytest.raises(ValueError):
        store._classify_analysis_sql(store._normalize_analysis_sql(sql))


def test_analysis_sql_ignores_blocked_words_inside_string_literals() -> None:
    store = _store()

    normalized = store._normalize_analysis_sql(
        "SELECT run_id FROM evaluation_runs WHERE final_output = 'drop table users; delete data'"
    )

    assert store._classify_analysis_sql(normalized) == "query"


def test_analysis_sql_executes_accepted_queries_in_read_only_transaction() -> None:
    store = _store()
    commands: list[tuple[str, object]] = []

    class FakeCursor:
        description = [SimpleNamespace(name="run_id")]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            commands.append((sql, params))

        def fetchmany(self, limit):
            return [("run-1",)]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

    store._analysis_connection = lambda: FakeConnection()

    result = store.run_analysis_query("SELECT run_id FROM evaluation_runs LIMIT 1")

    assert commands[0] == ("BEGIN READ ONLY", None)
    assert commands[1] == ("SET LOCAL statement_timeout = 5000", None)
    assert commands[2] == ("SELECT run_id FROM evaluation_runs LIMIT 1", None)
    assert commands[3] == ("ROLLBACK", None)
    assert result == {
        "mode": "query",
        "columns": ["run_id"],
        "items": [{"run_id": "run-1"}],
        "row_count": 1,
    }
