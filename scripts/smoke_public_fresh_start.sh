#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PROJECT_NAME="${SMOKE_PROJECT_NAME:-jailbreaker-smoke-$(date +%s)-$$}"
SMOKE_VAULT_PASSPHRASE="${SMOKE_VAULT_PASSPHRASE:-local-smoke-passphrase}"

export APP_ENV=development
export POSTGRES_DB=jailbreaker
export POSTGRES_USER=jailbreaker
export POSTGRES_PASSWORD=jailbreaker
export DATABASE_URL=postgresql://jailbreaker:jailbreaker@database:5432/jailbreaker
export API_BASE_URL=http://backend:8000
export FRONTEND_PORT="${SMOKE_FRONTEND_PORT:-0}"
export BACKEND_PORT="${SMOKE_BACKEND_PORT:-0}"
export DATABASE_PORT="${SMOKE_DATABASE_PORT:-0}"
export NEXT_PUBLIC_API_BASE_URL="${SMOKE_NEXT_PUBLIC_API_BASE_URL:-http://127.0.0.1:${BACKEND_PORT}}"
export API_CORS_ORIGINS="${SMOKE_API_CORS_ORIGINS:-http://127.0.0.1:${FRONTEND_PORT},http://localhost:${FRONTEND_PORT}}"

compose() {
  docker compose -p "$PROJECT_NAME" -f "$REPO_ROOT/docker-compose.yml" "$@"
}

cleanup() {
  local status=$?
  local cleanup_status=0
  trap - EXIT
  compose down -v --remove-orphans >/dev/null || cleanup_status=$?
  if (( cleanup_status != 0 )); then
    echo "Warning: failed to tear down smoke Compose project $PROJECT_NAME." >&2
    if (( status == 0 )); then
      status="$cleanup_status"
    fi
  fi
  exit "$status"
}
trap cleanup EXIT

resolve_host_port() {
  local service="$1"
  local target_port="$2"
  local mapping
  local resolved_port

  if ! mapping="$(compose port "$service" "$target_port")"; then
    echo "Failed to resolve Compose port mapping for $service:$target_port." >&2
    return 1
  fi
  if [[ ! "$mapping" =~ ^127\.0\.0\.1:([0-9]+)$ ]]; then
    echo "Malformed Compose port mapping for $service:$target_port: ${mapping:-<empty>}." >&2
    return 1
  fi

  resolved_port="${BASH_REMATCH[1]}"
  if (( 10#$resolved_port < 1 || 10#$resolved_port > 65535 )); then
    echo "Malformed Compose port mapping for $service:$target_port: $mapping." >&2
    return 1
  fi

  printf '%s\n' "$resolved_port"
}

wait_for_http_200() {
  local label="$1"
  local url="$2"
  local attempts="${SMOKE_HEALTH_ATTEMPTS:-60}"
  local delay_seconds="${SMOKE_HEALTH_DELAY_SECONDS:-2}"
  local attempt

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if python - "$url" <<'PY'
import sys
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep "$delay_seconds"
  done

  echo "$label did not return HTTP 200 at $url" >&2
  compose ps >&2 || true
  compose logs backend frontend >&2 || true
  return 1
}

compose up -d --build frontend backend database
resolved_frontend_port="$(resolve_host_port frontend 3000)"
resolved_backend_port="$(resolve_host_port backend 8000)"
resolve_host_port database 5432 >/dev/null
BACKEND_BASE_URL="http://127.0.0.1:${resolved_backend_port}"
FRONTEND_BASE_URL="http://127.0.0.1:${resolved_frontend_port}"
wait_for_http_200 "backend" "$BACKEND_BASE_URL/health"
wait_for_http_200 "frontend" "$FRONTEND_BASE_URL/"

python - "$BACKEND_BASE_URL" "$FRONTEND_BASE_URL" "$SMOKE_VAULT_PASSPHRASE" <<'PY'
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

backend_base_url = sys.argv[1]
frontend_base_url = sys.argv[2]
passphrase = sys.argv[3]


def request_json(base_url, method, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(base_url + path, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {path} failed with HTTP {exc.code}: {body}") from exc


def request_status(url):
    try:
        with urlopen(url, timeout=30) as response:
            return response.status
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GET {url} failed with HTTP {exc.code}: {body}") from exc


def require(condition, message):
    if not condition:
        raise SystemExit(message)


# GET / via frontend
frontend_status = request_status(frontend_base_url + "/")
require(frontend_status == 200, f"GET {frontend_base_url}/ returned HTTP {frontend_status}")

# GET /api/llm-vault
status, vault = request_json(backend_base_url, "GET", "/api/llm-vault")
require(status == 200, f"GET /api/llm-vault returned HTTP {status}")
require(vault.get("exists") is False, f"expected fresh vault exists=false, got {vault}")
require(vault.get("unlocked") is False, f"expected fresh vault unlocked=false, got {vault}")

# GET /api/llm-vault via frontend proxy
status, proxied_vault = request_json(frontend_base_url, "GET", "/api/llm-vault")
require(status == 200, f"GET frontend /api/llm-vault returned HTTP {status}")
require(proxied_vault == vault, f"expected frontend proxy vault response to match backend, got {proxied_vault}")

# GET /api/llm-profiles
status, profiles = request_json(backend_base_url, "GET", "/api/llm-profiles")
require(status == 200, f"GET /api/llm-profiles returned HTTP {status}")
require(profiles.get("items") == [], f"expected fresh profile list to be empty, got {profiles}")

# POST /api/llm-vault/create via frontend proxy
status, created_vault = request_json(frontend_base_url, "POST", "/api/llm-vault/create", {"passphrase": passphrase})
require(status == 201, f"POST /api/llm-vault/create returned HTTP {status}")
require(created_vault.get("exists") is True, f"expected created vault exists=true, got {created_vault}")
require(created_vault.get("unlocked") is True, f"expected created vault unlocked=true, got {created_vault}")

# GET /api/llm-vault after frontend proxy create
status, backend_created_vault = request_json(backend_base_url, "GET", "/api/llm-vault")
require(status == 200, f"GET /api/llm-vault after create returned HTTP {status}")
require(backend_created_vault.get("exists") is True, f"expected backend vault exists=true after create, got {backend_created_vault}")
require(backend_created_vault.get("unlocked") is True, f"expected backend vault unlocked=true after create, got {backend_created_vault}")

print("Public fresh-start smoke checks passed.")
PY
