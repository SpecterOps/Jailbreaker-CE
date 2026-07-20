import json
import io
import sys

sys.path.insert(0, "backend")

import app.api.server as server_module
from app.api.store import EvaluationStore
from app.persistence import NoopRunPersistence
from app.profile_vault import InMemoryProfileVaultRepository, ProfileVault


class InMemoryApiHandler(server_module.ApiHandler):
    def __init__(self, path: str, body: dict | None = None) -> None:
        raw = json.dumps(body or {}).encode("utf-8")
        self.path = path
        self.headers = {"Content-Length": str(len(raw))}
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.status = 0
        self.response_headers: dict[str, str] = {}

    def send_response(self, code, message=None):  # noqa: ANN001
        self.status = int(code)

    def send_header(self, keyword, value):  # noqa: ANN001
        self.response_headers[str(keyword)] = str(value)

    def end_headers(self):
        return


def _request(method: str, path: str, payload: dict | None = None):
    handler = InMemoryApiHandler(path, payload)
    getattr(handler, f"do_{method}")()
    return handler.status, json.loads(handler.wfile.getvalue().decode("utf-8"))


def test_llm_vault_api_create_unlock_profile_lifecycle(monkeypatch) -> None:
    vault = ProfileVault(InMemoryProfileVaultRepository())
    store = EvaluationStore(persistence=NoopRunPersistence(), profile_vault=vault)
    monkeypatch.setattr(server_module, "STORE", store)

    status, payload = _request("GET", "/api/llm-vault")
    assert status == 200
    assert payload["exists"] is False
    assert payload["unlocked"] is False

    status, payload = _request(
        "POST",
        "/api/llm-vault/create",
        {"passphrase": "team-passphrase"},
    )
    assert status == 201
    assert payload["exists"] is True
    assert payload["unlocked"] is True

    status, payload = _request(
        "POST",
        "/api/llm-profiles",
        {
            "name": "OpenAI primary",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "system_prompt": "Judge against the supplied policy.",
            "scope": "all",
            "api_key": "sk-test-secret",
        },
    )
    assert status == 201
    profile = payload["item"]
    assert profile["name"] == "OpenAI primary"
    assert "system_prompt" not in profile
    assert "api_key" not in profile

    status, payload = _request("GET", "/api/llm-profiles")
    assert status == 200
    assert payload["items"] == [profile]

    status, payload = _request("POST", "/api/llm-vault/lock")
    assert status == 200
    assert payload["unlocked"] is False

    status, payload = _request(
        "POST",
        "/api/llm-vault/unlock",
        {"passphrase": "wrong"},
    )
    assert status == 400
    assert "passphrase" in payload["error"].lower()

    status, payload = _request(
        "POST",
        "/api/llm-vault/unlock",
        {"passphrase": "team-passphrase"},
    )
    assert status == 200
    assert payload["unlocked"] is True

    status, payload = _request("DELETE", f"/api/llm-profiles/{profile['profile_id']}")
    assert status == 200
    assert payload == {"ok": True}


def test_target_credentials_can_be_manual_local_or_profile() -> None:
    assert server_module._has_target_credentials({"api_key": "sk-test"}) is True
    assert server_module._has_target_credentials({"target_local": True}) is True
    assert server_module._has_target_credentials({"target_profile_id": "profile-1"}) is True
    assert server_module._has_target_credentials({"api_key": ""}) is False
