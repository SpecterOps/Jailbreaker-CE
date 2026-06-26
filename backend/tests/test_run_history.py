import io
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal

sys.path.insert(0, "backend")

import app.api.server as server_module
from app.persistence import PostgreSQLRunPersistence


class InMemoryApiHandler(server_module.ApiHandler):
    def __init__(self, path: str) -> None:
        self.path = path
        self.headers = {}
        self.rfile = io.BytesIO()
        self.wfile = io.BytesIO()
        self.status = 0
        self.response_headers: dict[str, str] = {}

    def send_response(self, code, message=None):  # noqa: ANN001
        self.status = int(code)

    def send_header(self, keyword, value):  # noqa: ANN001
        self.response_headers[str(keyword)] = str(value)

    def end_headers(self):
        return


class _FakeQueryResult:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple]:
        return self.rows

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None


class _HistoryConnection:
    def __init__(self) -> None:
        self.commands: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query: str, params=None) -> _FakeQueryResult:
        self.commands.append((query, params))
        if "COUNT(*) OVER()" in query:
            return _FakeQueryResult(
                [
                    (
                        "run-7",
                        "completed",
                        "live",
                        "dan_style_roleplay",
                        "DAN Style Roleplay",
                        "human-based",
                        "text",
                        "chatbot",
                        "gpt-4o-mini",
                        "attacker-model",
                        "Extract private user data",
                        True,
                        82,
                        100,
                        2,
                        "high",
                        "Final prompt text",
                        "Final output text",
                        datetime(2026, 6, 18, 10, 0, tzinfo=UTC),
                        datetime(2026, 6, 18, 10, 1, tzinfo=UTC),
                        Decimal("61.2"),
                        1,
                        {"config": {"judge_mode": "custom", "judge_model": "judge-model"}},
                    )
                ]
            )
        if "GROUP BY technique_key, technique_name" in query:
            return _FakeQueryResult([("dan_style_roleplay", "DAN Style Roleplay", 3)])
        if "GROUP BY technique_family" in query:
            return _FakeQueryResult([("human-based", 3)])
        if "GROUP BY technique_surface" in query:
            return _FakeQueryResult([("text", 3)])
        if "GROUP BY highest_severity" in query:
            return _FakeQueryResult([("high", 2), ("low", 1)])
        if "GROUP BY status" in query:
            return _FakeQueryResult([("completed", 3)])
        if "GROUP BY mode" in query:
            return _FakeQueryResult([("live", 3)])
        if "GROUP BY target_model" in query:
            return _FakeQueryResult([("gpt-4o-mini", 3)])
        raise AssertionError(f"unexpected query: {query}")


class _DetailConnection(_HistoryConnection):
    def execute(self, query: str, params=None) -> _FakeQueryResult:
        self.commands.append((query, params))
        if "FROM evaluation_runs" in query:
            return _FakeQueryResult(
                [
                    (
                        "run-7",
                        "technique",
                        None,
                        "completed",
                        "live",
                        "demo-user",
                        "dan_style_roleplay",
                        "DAN Style Roleplay",
                        "human-based",
                        "text",
                        "chatbot",
                        "gpt-4o-mini",
                        "attacker-model",
                        "Extract private user data",
                        True,
                        82,
                        100,
                        2,
                        "high",
                        "Final prompt text",
                        "Final output text",
                        {"tokens": 42},
                        {"config": {"model": "gpt-4o-mini", "judge_mode": "custom", "judge_model": "judge-model"}},
                        datetime(2026, 6, 18, 10, 0, tzinfo=UTC),
                        datetime(2026, 6, 18, 10, 1, tzinfo=UTC),
                        Decimal("61.2"),
                    )
                ]
            )
        if "FROM evaluation_run_detections" in query:
            return _FakeQueryResult([("refusal_bypass", "high", True, "matched", ["evidence"])])
        if "FROM evaluation_run_conversations" in query:
            return _FakeQueryResult([("single_shot", {"prompt": "Final prompt text", "response": "Final output text"})])
        if "FROM evaluation_run_traces" in query:
            return _FakeQueryResult([("target_response", "target", "responded", datetime(2026, 6, 18, 10, 1, tzinfo=UTC), {})])
        raise AssertionError(f"unexpected query: {query}")


class _FakePostgreSQLRunPersistence(PostgreSQLRunPersistence):
    def __init__(self, connection) -> None:
        self.connection = connection

    def _connect(self):
        return self.connection

    def ensure_schema(self, conn) -> None:
        return None


def _get(path: str) -> tuple[int, dict]:
    handler = InMemoryApiHandler(path)
    handler.do_GET()
    return handler.status, json.loads(handler.wfile.getvalue().decode("utf-8"))


def test_api_get_run_history_parses_filters_and_pagination(monkeypatch) -> None:
    class FakeStore:
        def list_run_history(self, *, filters, limit, offset, sort):
            assert filters["q"] == "DAN"
            assert filters["technique_family"] == "human-based"
            assert filters["success"] == "true"
            assert limit == 25
            assert offset == 50
            assert sort == "score_desc"
            return {"items": [{"run_id": "run-7"}], "total": 1, "limit": limit, "offset": offset, "facets": {}}

    monkeypatch.setattr(server_module, "STORE", FakeStore())

    status, payload = _get(
        "/api/run-history?q=DAN&technique_family=human-based&success=true&limit=25&offset=50&sort=score_desc"
    )

    assert status == 200
    assert payload["items"] == [{"run_id": "run-7"}]
    assert payload["limit"] == 25
    assert payload["offset"] == 50


def test_postgresql_run_history_filters_to_one_off_technique_runs() -> None:
    conn = _HistoryConnection()
    persistence = _FakePostgreSQLRunPersistence(conn)

    payload = persistence.list_run_history(
        filters={
            "q": "private",
            "technique_family": "human-based",
            "highest_severity": "high",
            "success": "true",
            "completed_after": "2026-06-01",
            "completed_before": "2026-06-18",
        },
        limit=25,
        offset=0,
        sort="score_desc",
    )

    query, params = conn.commands[0]
    assert "run_kind = 'technique'" in query
    assert "evaluation_id IS NULL" in query
    assert "ILIKE" in query
    assert "technique_family = %s" in query
    assert "highest_severity = %s" in query
    assert "success = %s" in query
    assert "completed_at >= %s::timestamptz" in query
    assert "completed_at < (%s::date + INTERVAL '1 day')" in query
    assert "ORDER BY score DESC" in query
    assert list(params[:6]) == ["%private%", "human-based", "high", True, "2026-06-01", "2026-06-18"]
    assert payload["total"] == 1
    assert payload["items"][0]["run_id"] == "run-7"
    assert payload["items"][0]["judge_model"] == "judge-model"
    assert payload["items"][0]["duration_seconds"] == 61.2
    assert payload["facets"]["techniques"][0]["key"] == "dan_style_roleplay"


def test_postgresql_get_run_history_item_loads_detail_children() -> None:
    conn = _DetailConnection()
    persistence = _FakePostgreSQLRunPersistence(conn)

    item = persistence.get_run_history_item("run-7")

    assert item["run_id"] == "run-7"
    assert item["technique_key"] == "dan_style_roleplay"
    assert item["judge_model"] == "judge-model"
    assert item["detections"] == [
        {
            "detection_type": "refusal_bypass",
            "severity": "high",
            "matched": True,
            "rationale": "matched",
            "evidence": ["evidence"],
        }
    ]
    assert item["conversation"][0]["prompt"] == "Final prompt text"
    assert item["traces"][0]["event_type"] == "target_response"
