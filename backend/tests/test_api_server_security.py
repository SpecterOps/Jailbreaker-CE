import io
import json
import sys

sys.path.insert(0, "backend")

import app.api.server as server_module
from app.api.server import _allowed_cors_origins, _api_bind_address, _cors_origin_for_request


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


def test_post_rejects_underscored_content_length() -> None:
    status, payload = _post("/api/datasets", b"{}", {"Content-Length": "1_000"})

    assert status == 400
    assert payload == {"error": "Invalid Content-Length header."}
