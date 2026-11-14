import io
import json
import sys
from datetime import UTC, datetime
from urllib.parse import quote

import pytest

sys.path.insert(0, "backend")

import app.api.server as server_module
from app.api.server import _allowed_cors_origins, _api_bind_address, _cors_origin_for_request
from app.persistence import RetryBatchCellRecord, RetryBatchRecord


class FakeHeaders:
    def __init__(self, content_lengths: list[str]) -> None:
        self._content_lengths = content_lengths

    def get(self, keyword: str, default: str = "") -> str:
        if keyword.lower() == "content-length" and self._content_lengths:
            return self._content_lengths[0]
        return default

    def get_all(self, keyword: str) -> list[str] | None:
        if keyword.lower() == "content-length":
            return list(self._content_lengths)
        return None


class InMemoryApiHandler(server_module.ApiHandler):
    def __init__(self, path: str, raw_body: bytes = b"", headers: object | None = None) -> None:
        self.path = path
        self.headers = headers or {"Content-Length": str(len(raw_body))}
        self.rfile = io.BytesIO(raw_body)
        self.wfile = io.BytesIO()
        self.status = 0
        self.response_headers: dict[str, str] = {}

    def send_response(self, code, message=None):  # noqa: ANN001
        self.status = int(code)

    def send_header(self, keyword, value):  # noqa: ANN001
        self.response_headers[str(keyword)] = str(value)

    def end_headers(self):
        return


def _post(path: str, raw_body: bytes, headers: object | None = None) -> tuple[int, dict]:
    handler = InMemoryApiHandler(path, raw_body, headers)
    handler.do_POST()
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    return handler.status, payload


def _get(path: str) -> tuple[int, dict]:
    handler = InMemoryApiHandler(path)
    handler.do_GET()
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    return handler.status, payload


def _get_status(path: str) -> int:
    handler = InMemoryApiHandler(path)
    handler.do_GET()
    return handler.status


def _retry_batch_record() -> RetryBatchRecord:
    created_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    return RetryBatchRecord(
        retry_batch_id="retry-batch-1",
        experiment_id="exp-1",
        request_id="request-1",
        action="restart",
        completed_policy="exclude",
        requested_by="workspace-user",
        cell_ids=("cell-2", "cell-1"),
        selection_metadata={
            "selection_mode": "explicit",
            "requested_cell_ids": ["cell-2", "cell-1"],
            "target_profile_id": "target-profile",
        },
        members=(
            RetryBatchCellRecord(
                retry_batch_id="retry-batch-1",
                cell_id="cell-2",
                ordinal=0,
                status="completed",
                attempt_id="attempt-2",
                created_at=created_at,
                completed_at=created_at,
            ),
            RetryBatchCellRecord(
                retry_batch_id="retry-batch-1",
                cell_id="cell-1",
                ordinal=1,
                status="failed",
                attempt_id="attempt-1",
                error_category="provider",
                created_at=created_at,
                completed_at=created_at,
            ),
        ),
        status="completed",
        started_at=created_at,
        completed_at=created_at,
        created_at=created_at,
        was_created=False,
    )


def test_cors_defaults_to_local_frontend_origins(monkeypatch) -> None:
    monkeypatch.delenv("API_CORS_ORIGINS", raising=False)

    allowed = _allowed_cors_origins()

    assert "*" not in allowed
    assert "http://localhost:3000" in allowed
    assert _cors_origin_for_request("http://localhost:3000", allowed) == "http://localhost:3000"
    assert _cors_origin_for_request("https://example.com", allowed) == ""


def test_cors_wildcard_requires_explicit_configuration() -> None:
    allowed = _allowed_cors_origins("*")

    assert _cors_origin_for_request("https://example.com", allowed) == "*"


def test_api_bind_defaults_to_loopback(monkeypatch) -> None:
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    assert _api_bind_address() == ("127.0.0.1", 8000)


def test_api_bind_can_be_overridden_for_containers(monkeypatch) -> None:
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "18000")

    assert _api_bind_address() == ("0.0.0.0", 18000)


def test_post_rejects_malformed_json_body() -> None:
    status, payload = _post("/api/datasets", b"{broken", {"Content-Length": "7"})

    assert status == 400
    assert payload == {"error": "Invalid JSON request body."}


def test_post_rejects_non_object_json_body() -> None:
    status, payload = _post("/api/datasets", b"[]", {"Content-Length": "2"})

    assert status == 400
    assert payload == {"error": "JSON request body must be an object."}


def test_post_rejects_invalid_content_length() -> None:
    status, payload = _post("/api/datasets", b"{}", {"Content-Length": "not-an-int"})

    assert status == 400
    assert payload == {"error": "Invalid Content-Length header."}


def test_post_rejects_duplicate_content_length_headers() -> None:
    status, payload = _post("/api/datasets", b"{}", FakeHeaders(["2", "2"]))

    assert status == 400
    assert payload == {"error": "Invalid Content-Length header."}


def test_bodyless_post_rejects_duplicate_content_length_headers(monkeypatch) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.lock_called = False

        def lock_vault(self) -> dict:
            self.lock_called = True
            return {"ok": True}

    fake_store = FakeStore()
    monkeypatch.setattr(server_module, "STORE", fake_store)

    status, payload = _post("/api/llm-vault/lock", b"", FakeHeaders(["0", "0"]))

    assert status == 400
    assert payload == {"error": "Invalid Content-Length header."}
    assert fake_store.lock_called is False


def test_post_rejects_signed_content_length() -> None:
    status, payload = _post("/api/datasets", b"{}", {"Content-Length": "+2"})

    assert status == 400
    assert payload == {"error": "Invalid Content-Length header."}


def test_bodyless_post_rejects_signed_content_length(monkeypatch) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.stop_called = False

        def stop_experiment(self, experiment_id: str) -> dict:
            self.stop_called = True
            return {"experiment_id": experiment_id, "status": "completed"}

    fake_store = FakeStore()
    monkeypatch.setattr(server_module, "STORE", fake_store)

    status, payload = _post("/api/experiments/exp-1/stop", b"", {"Content-Length": "+0"})

    assert status == 400
    assert payload == {"error": "Invalid Content-Length header."}
    assert fake_store.stop_called is False


def test_retry_batch_post_defaults_and_returns_safe_ordered_projection(monkeypatch) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create_retry_batch(self, experiment_id: str, **kwargs):
            self.calls.append({"experiment_id": experiment_id, **kwargs})
            return _retry_batch_record()

    fake_store = FakeStore()
    monkeypatch.setattr(server_module, "STORE", fake_store)

    status, payload = _post(
        "/api/experiments/exp-1/retry-batches",
        b'{"action":"restart"}',
    )

    assert status == 202
    assert fake_store.calls == [
        {
            "experiment_id": "exp-1",
            "action": "restart",
            "cell_ids": None,
            "completed_policy": "exclude",
            "credential_overrides": {},
            "request_id": None,
            "requested_by": "workspace-user",
        }
    ]
    item = payload["item"]
    assert item["cell_ids"] == ["cell-2", "cell-1"]
    assert [member["ordinal"] for member in item["members"]] == [0, 1]
    assert item["selection_metadata"] == {
        "selection_mode": "explicit",
        "requested_cell_ids": ["cell-2", "cell-1"],
        "target_profile_id": "target-profile",
    }
    assert item["created_at"] == "2026-08-24T12:00:00+00:00"
    assert "was_created" not in item
    encoded = json.dumps(payload)
    assert "worker" not in encoded
    assert "api_key" not in encoded


def test_retry_batch_post_preserves_explicit_selection_and_profiles(monkeypatch) -> None:
    calls = []

    class FakeStore:
        def create_retry_batch(self, experiment_id: str, **kwargs):
            calls.append({"experiment_id": experiment_id, **kwargs})
            return _retry_batch_record()

    monkeypatch.setattr(server_module, "STORE", FakeStore())
    request = {
        "action": "resume",
        "cell_ids": ["cell-2", "cell-1", "cell-2"],
        "completed_policy": "diagnostic_only",
        "credential_overrides": {
            "target_profile_id": "target-profile",
            "judge_profile_id": "judge-profile",
        },
        "request_id": "request-explicit",
    }

    status, _ = _post(
        "/api/experiments/exp-1/retry-batches",
        json.dumps(request).encode(),
    )

    assert status == 202
    assert calls == [
        {
            "experiment_id": "exp-1",
            "action": "resume",
            "cell_ids": ["cell-2", "cell-1", "cell-2"],
            "completed_policy": "diagnostic_only",
            "credential_overrides": {
                "target_profile_id": "target-profile",
                "judge_profile_id": "judge-profile",
            },
            "request_id": "request-explicit",
            "requested_by": "workspace-user",
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action": True},
        {"action": []},
        {"action": "replace"},
        {"action": "restart", "cell_ids": "cell-1"},
        {"action": "restart", "cell_ids": {}},
        {"action": "restart", "cell_ids": ["cell-1", ""]},
        {"action": "restart", "cell_ids": ["cell-1", 2]},
        {"action": "restart", "completed_policy": 1},
        {"action": "restart", "completed_policy": []},
        {"action": "restart", "completed_policy": "unknown"},
        {"action": "restart", "credential_overrides": []},
        {"action": "restart", "credential_overrides": None},
        {"action": "restart", "credential_overrides": {"model": "changed"}},
        {"action": "restart", "credential_overrides": {"api_key": 7}},
        {"action": "restart", "request_id": 7},
        {"action": "restart", "request_id": ""},
        {"action": "restart", "request_id": "   "},
        {"action": "restart", "requested_by": "attacker"},
        {"action": "restart", "target_config": {}},
    ],
)
def test_retry_batch_post_rejects_malformed_or_unknown_fields_before_store(
    monkeypatch,
    payload,
) -> None:
    class FakeStore:
        def create_retry_batch(self, *args, **kwargs):
            raise AssertionError("invalid request must not reach store")

    monkeypatch.setattr(server_module, "STORE", FakeStore())

    status, response = _post(
        "/api/experiments/exp-1/retry-batches",
        json.dumps(payload).encode(),
    )

    assert status == 400
    assert "error" in response


def test_retry_batch_and_attempt_gets_are_exact_and_experiment_scoped(monkeypatch) -> None:
    class FakeStore:
        def get_retry_batch(self, experiment_id: str, retry_batch_id: str):
            if experiment_id != "exp-1" or retry_batch_id != "retry-batch-1":
                raise KeyError("not found")
            return _retry_batch_record()

        def get_experiment_cell_attempts(self, experiment_id: str, cell_id: str):
            if experiment_id != "exp-1" or cell_id != "cell-1":
                raise KeyError("not found")
            return {
                "items": [
                    {
                        "attempt_id": "attempt-1",
                        "attempt_number": 1,
                        "score": 0,
                        "late_completion": True,
                        "is_selected": False,
                        "is_latest": True,
                        "is_active": False,
                    }
                ],
                "selected_attempt_id": None,
                "latest_attempt_id": "attempt-1",
                "active_attempt_id": None,
            }

    monkeypatch.setattr(server_module, "STORE", FakeStore())

    batch_status, batch_payload = _get(
        "/api/experiments/exp-1/retry-batches/retry-batch-1"
    )
    attempt_status, attempt_payload = _get(
        "/api/experiments/exp-1/cells/cell-1/attempts"
    )
    cross_batch_status, _ = _get(
        "/api/experiments/exp-2/retry-batches/retry-batch-1"
    )
    cross_cell_status, _ = _get(
        "/api/experiments/exp-2/cells/cell-1/attempts"
    )
    suffix_status, _ = _get(
        "/api/experiments/exp-1/cells/cell-1/attempts/extra"
    )

    assert batch_status == 200
    assert batch_payload["item"]["retry_batch_id"] == "retry-batch-1"
    assert attempt_status == 200
    assert attempt_payload["items"][0]["score"] == 0
    assert attempt_payload["items"][0]["late_completion"] is True
    assert cross_batch_status == 404
    assert cross_cell_status == 404
    assert suffix_status == 404


def test_malformed_task18_suffixes_cannot_dispatch_other_experiment_operations(
    monkeypatch,
) -> None:
    calls = []

    class FakeStore:
        def export_experiment_json(self, experiment_id: str):
            calls.append(("export-json", experiment_id))
            return {"experiment_id": experiment_id}

        def export_experiment_csv(self, experiment_id: str):
            calls.append(("export", experiment_id))
            return "csv"

        def export_experiment_full_csv(self, experiment_id: str):
            calls.append(("export-full", experiment_id))
            return "csv"

        def compute_experiment_metrics(self, experiment_id: str):
            calls.append(("metrics", experiment_id))
            return {"experiment_id": experiment_id}

        def stop_experiment(self, experiment_id: str):
            calls.append(("stop", experiment_id))
            return {"experiment_id": experiment_id}

        def rejudge_experiment(self, experiment_id: str, judge_config: dict):
            calls.append(("rejudge", experiment_id))
            return {"experiment_id": experiment_id}

    monkeypatch.setattr(server_module, "STORE", FakeStore())

    get_paths = [
        "/api/experiments/exp-a/cells/cell-a/attempts/exp-b/export",
        "/api/experiments/exp-a/retry-batches/batch-a/exp-b/export-full",
        "/api/experiments/exp-a/cells/cell-a/attempts/exp-b/metrics",
    ]
    post_paths = [
        "/api/experiments/exp-a/retry-batches/exp-b/stop",
        "/api/experiments/exp-a/cells/cell-a/attempts/exp-b/rejudge",
    ]

    statuses = [_get_status(path) for path in get_paths]
    statuses.extend(_post(path, b"{}")[0] for path in post_paths)

    assert statuses == [404, 404, 404, 404, 404]
    assert calls == []


def test_experiment_routes_decode_opaque_identifiers_exactly_once_after_matching(
    monkeypatch,
) -> None:
    calls = []

    class FakeStore:
        def get_experiment(self, experiment_id: str):
            calls.append(("detail", experiment_id))
            return {"experiment_id": experiment_id}

        def get_experiment_cell_attempts(self, experiment_id: str, cell_id: str):
            calls.append(("attempts", experiment_id, cell_id))
            return {
                "items": [],
                "selected_attempt_id": None,
                "latest_attempt_id": None,
                "active_attempt_id": None,
            }

        def get_retry_batch(self, experiment_id: str, retry_batch_id: str):
            calls.append(("batch", experiment_id, retry_batch_id))
            return _retry_batch_record()

    monkeypatch.setattr(server_module, "STORE", FakeStore())
    experiment_id = "experiment / 50%"
    cell_id = "cell / 25%"
    batch_id = "batch / 10%"
    encoded_experiment = quote(experiment_id, safe="")

    detail_status, _ = _get(f"/api/experiments/{encoded_experiment}")
    attempts_status, _ = _get(
        f"/api/experiments/{encoded_experiment}/cells/{quote(cell_id, safe='')}/attempts"
    )
    batch_status, _ = _get(
        f"/api/experiments/{encoded_experiment}/retry-batches/{quote(batch_id, safe='')}"
    )

    assert [detail_status, attempts_status, batch_status] == [200, 200, 200]
    assert calls == [
        ("detail", experiment_id),
        ("attempts", experiment_id, cell_id),
        ("batch", experiment_id, batch_id),
    ]


def test_experiment_routes_do_not_double_decode_or_decode_static_segments(
    monkeypatch,
) -> None:
    calls = []

    class FakeStore:
        def get_experiment_cell_attempts(self, experiment_id: str, cell_id: str):
            calls.append((experiment_id, cell_id))
            return {
                "items": [],
                "selected_attempt_id": None,
                "latest_attempt_id": None,
                "active_attempt_id": None,
            }

        def get_retry_batch(self, experiment_id: str, retry_batch_id: str):
            raise AssertionError("encoded static segment must not dispatch")

    monkeypatch.setattr(server_module, "STORE", FakeStore())

    once_status, _ = _get(
        "/api/experiments/exp%252Fone/cells/cell%252Fone/attempts"
    )
    confused_status, _ = _get(
        "/api/experiments/exp-1/%72etry-batches/batch-1"
    )

    assert once_status == 200
    assert calls == [("exp%2Fone", "cell%2Fone")]
    assert confused_status == 404


def test_all_experiment_subroutes_decode_their_experiment_identifier(monkeypatch) -> None:
    calls = []

    class FakeStore:
        def export_experiment_json(self, experiment_id: str):
            calls.append(("export", experiment_id))
            return {"experiment_id": experiment_id}

        def export_experiment_full_csv(self, experiment_id: str):
            calls.append(("export-full", experiment_id))
            return "column\nvalue\n"

        def compute_experiment_metrics(self, experiment_id: str):
            calls.append(("metrics", experiment_id))
            return {"experiment_id": experiment_id}

        def stop_experiment(self, experiment_id: str):
            calls.append(("stop", experiment_id))
            return {"experiment_id": experiment_id}

        def retry_experiment_errors(self, experiment_id: str, target_config: dict):
            calls.append(("retry", experiment_id))
            return {"experiment_id": experiment_id}

        def rejudge_experiment(self, experiment_id: str, judge_config: dict):
            calls.append(("rejudge", experiment_id))
            return {"experiment_id": experiment_id}

        def create_retry_batch(self, experiment_id: str, **kwargs):
            calls.append(("retry-batches", experiment_id))
            return _retry_batch_record()

    monkeypatch.setattr(server_module, "STORE", FakeStore())
    experiment_id = "experiment / 50%"
    encoded = quote(experiment_id, safe="")

    statuses = [
        _get(f"/api/experiments/{encoded}/export?format=json")[0],
        _get_status(f"/api/experiments/{encoded}/export-full"),
        _get(f"/api/experiments/{encoded}/metrics")[0],
        _post(f"/api/experiments/{encoded}/stop", b"")[0],
        _post(f"/api/experiments/{encoded}/retry", b'{"target_config":{}}')[0],
        _post(f"/api/experiments/{encoded}/rejudge", b"{}")[0],
        _post(
            f"/api/experiments/{encoded}/retry-batches",
            b'{"action":"restart"}',
        )[0],
    ]

    assert statuses == [200, 200, 200, 200, 202, 202, 202]
    assert calls == [
        ("export", experiment_id),
        ("export-full", experiment_id),
        ("metrics", experiment_id),
        ("stop", experiment_id),
        ("retry", experiment_id),
        ("rejudge", experiment_id),
        ("retry-batches", experiment_id),
    ]


def test_legacy_retry_accepts_only_target_config_and_preserves_store_validation(monkeypatch) -> None:
    calls = []

    class FakeStore:
        def retry_experiment_errors(self, experiment_id: str, target_config: dict):
            calls.append((experiment_id, target_config))
            if "model" in target_config:
                raise ValueError("retry configuration changes require an experiment revision")
            return {"experiment_id": experiment_id, "status": "running"}

    monkeypatch.setattr(server_module, "STORE", FakeStore())

    profile_status, profile_payload = _post(
        "/api/experiments/exp-1/retry",
        b'{"target_config":{"target_profile_id":"profile-1"}}',
    )
    demo_status, _ = _post(
        "/api/experiments/exp-1/retry",
        b'{"target_config":{}}',
    )
    changed_status, changed_payload = _post(
        "/api/experiments/exp-1/retry",
        b'{"target_config":{"api_key":"secret","model":"changed"}}',
    )
    unknown_status, _ = _post(
        "/api/experiments/exp-1/retry",
        b'{"target_config":{},"requested_by":"attacker"}',
    )

    assert profile_status == 202
    assert profile_payload["item"]["experiment_id"] == "exp-1"
    assert demo_status == 202
    assert changed_status == 400
    assert "experiment revision" in changed_payload["error"]
    assert unknown_status == 400
    assert calls[:2] == [
        ("exp-1", {"target_profile_id": "profile-1"}),
        ("exp-1", {}),
    ]


def test_post_rejects_underscored_content_length() -> None:
    status, payload = _post("/api/datasets", b"{}", {"Content-Length": "1_000"})

    assert status == 400
    assert payload == {"error": "Invalid Content-Length header."}
