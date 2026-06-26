from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from datetime import date, datetime
from decimal import Decimal

from .store import STORE

log = logging.getLogger("api.server")

_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3100",
    "http://127.0.0.1:3100",
)


def _allowed_cors_origins(raw: str | None = None) -> set[str]:
    source = os.environ.get("API_CORS_ORIGINS", "") if raw is None else raw
    if not source.strip():
        return set(_DEFAULT_CORS_ORIGINS)
    return {origin.strip().rstrip("/") for origin in source.split(",") if origin.strip()}


def _cors_origin_for_request(origin: str | None, allowed_origins: set[str]) -> str:
    if "*" in allowed_origins:
        return "*"
    normalized = (origin or "").strip().rstrip("/")
    if normalized and normalized in allowed_origins:
        return normalized
    return ""


def _api_bind_address() -> tuple[str, int]:
    host = os.environ.get("API_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("API_PORT", "8000"))
    except ValueError:
        port = 8000
    return host, port


class JsonBodyError(ValueError):
    """Raised when an HTTP request body cannot be parsed as a JSON object."""


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "JailbreakerAPI/0.1"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._write_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        route = parsed_url.path
        query = parse_qs(parsed_url.query)
        log.info("GET %s", route)
        if route == "/health":
            self._send_json({"status": "ok"})
            return
        if route == "/api/techniques":
            self._send_json({"items": STORE.list_techniques()})
            return
        if route == "/api/dashboard":
            self._send_json(STORE.dashboard())
            return
        if route == "/api/analysis":
            self._send_json({"items": STORE.list_analysis_views()})
            return
        if route == "/api/run-history":
            filters = _run_history_filters(query)
            limit = _query_int(query, "limit", default=50, minimum=1, maximum=100)
            offset = _query_int(query, "offset", default=0, minimum=0, maximum=100000)
            sort = _query_value(query, "sort", "completed_desc")
            try:
                self._send_json(STORE.list_run_history(filters=filters, limit=limit, offset=offset, sort=sort))
            except Exception as exc:  # noqa: BLE001
                log.error("GET %s -> 500: %s", route, exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if route.startswith("/api/run-history/"):
            run_id = route.split("/")[-1]
            try:
                item = STORE.get_run_history_item(run_id)
            except Exception as exc:  # noqa: BLE001
                log.error("GET %s -> 500: %s", route, exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if item is None:
                self._send_json({"error": "Run history item not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"item": item})
            return
        if route == "/api/custom-techniques":
            self._send_json({"items": STORE.list_custom_techniques()})
            return
        if route == "/api/llm-vault":
            self._send_json(STORE.vault_status())
            return
        if route == "/api/llm-profiles":
            self._send_json({"items": STORE.list_llm_profiles()})
            return
        if route.startswith("/api/analysis/"):
            view_key = route.split("/")[-1]
            try:
                result = STORE.get_analysis_view(view_key)
            except KeyError:
                self._send_json({"error": "Analysis view not found."}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("GET %s -> 500: %s", route, exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)
            return
        if route == "/api/datasets":
            self._send_json({"items": STORE.list_datasets()})
            return
        if route == "/api/experiments":
            self._send_json({"items": STORE.list_experiments()})
            return
        if route.startswith("/api/experiments/") and route.endswith("/export"):
            experiment_id = route.split("/")[-2]
            query = urlparse(self.path).query
            fmt = "csv"
            for part in query.split("&"):
                if part.startswith("format="):
                    fmt = part.split("=", 1)[1]
            if fmt == "json":
                data = STORE.export_experiment_json(experiment_id)
                if data is None:
                    self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(data)
            else:
                csv_data = STORE.export_experiment_csv(experiment_id)
                if csv_data is None:
                    self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                body = csv_data.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self._write_cors_headers()
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{experiment_id}.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        if route.startswith("/api/experiments/") and route.endswith("/export-full"):
            experiment_id = route.split("/")[-2]
            csv_data = STORE.export_experiment_full_csv(experiment_id)
            if csv_data is None:
                self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                return
            body = csv_data.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._write_cors_headers()
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{experiment_id}-full.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route.startswith("/api/experiments/") and route.endswith("/metrics"):
            experiment_id = route.split("/")[-2]
            result = STORE.compute_experiment_metrics(experiment_id)
            if result is None:
                self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
            return
        if route.startswith("/api/experiments/"):
            experiment_id = route.split("/")[-1]
            result = STORE.get_experiment(experiment_id)
            if result is None:
                self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"item": result})
            return
        if route.startswith("/api/runs/"):
            run_id = route.split("/")[-1]
            result = STORE.get_run(run_id)
            if result is None:
                log.warning("GET %s -> 404", route)
                self._send_json({"error": "Run not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"item": result})
            return
        if route.startswith("/api/evaluations/"):
            eval_id = route.split("/")[-1]
            result = STORE.get_evaluation(eval_id)
            if result is None:
                log.warning("GET %s -> 404", route)
                self._send_json({"error": "Evaluation not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"item": result})
            return
        log.warning("GET %s -> 404", route)
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        log.info("POST %s", route)
        if not self._validate_content_length_or_400():
            return

        if route == "/api/runs":
            payload = self._read_json_or_400()
            if payload is None:
                return
            technique_key = str(payload.get("technique_key", ""))
            target_kind = str(payload.get("target_kind", "chatbot"))
            mode = str(payload.get("mode", "demo"))
            target_config = payload.get("target_config") or {}
            custom_definition = payload.get("custom_definition")
            if not technique_key:
                log.warning("POST /api/runs -> 400: missing technique_key")
                self._send_json({"error": "`technique_key` is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            if mode not in ("demo", "live"):
                log.warning("POST /api/runs -> 400: invalid mode=%s", mode)
                self._send_json({"error": "`mode` must be 'demo' or 'live'."}, status=HTTPStatus.BAD_REQUEST)
                return
            log.info("POST /api/runs technique=%s mode=%s target_kind=%s", technique_key, mode, target_kind)
            try:
                run = STORE.create_run(
                    technique_key=technique_key,
                    target_kind=target_kind,
                    mode=mode,
                    target_config=target_config,
                    custom_definition=custom_definition if isinstance(custom_definition, dict) else None,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/runs -> 400: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            status_code = HTTPStatus.ACCEPTED if run.get("status") == "running" else HTTPStatus.CREATED
            log.info("POST /api/runs -> %d run_id=%s status=%s", status_code, run.get("run_id"), run.get("status"))
            self._send_json({"item": run}, status=status_code)
            return

        if route == "/api/baseline":
            payload = self._read_json_or_400()
            if payload is None:
                return
            target_config = payload.get("target_config") or {}
            if not _has_target_credentials(target_config):
                log.warning("POST /api/baseline -> 400: missing api_key")
                self._send_json(
                    {"error": "`target_config.api_key` or `target_config.target_profile_id` is required."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if not target_config.get("unsafe_intent"):
                log.warning("POST /api/baseline -> 400: missing unsafe_intent")
                self._send_json(
                    {"error": "`target_config.unsafe_intent` is required."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            log.info("POST /api/baseline intent=%s", target_config.get("unsafe_intent", "")[:60])
            try:
                result = STORE.run_baseline(target_config=target_config)
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/baseline -> 400: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            log.info("POST /api/baseline -> 200 score=%s", result.get("judge_score"))
            self._send_json({"item": result})
            return

        if route == "/api/evaluations":
            payload = self._read_json_or_400()
            if payload is None:
                return
            mode = str(payload.get("mode", "live"))
            target_kind = str(payload.get("target_kind", "chatbot"))
            target_config = payload.get("target_config") or {}
            technique_keys = payload.get("technique_keys") or []

            if mode not in ("demo", "live"):
                log.warning("POST /api/evaluations -> 400: invalid mode=%s", mode)
                self._send_json(
                    {"error": "`mode` must be 'demo' or 'live'."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if mode == "live" and not _has_target_credentials(target_config):
                log.warning("POST /api/evaluations -> 400: missing api_key")
                self._send_json(
                    {"error": "`target_config.api_key` or `target_config.target_profile_id` is required for live mode."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            log.info(
                "POST /api/evaluations mode=%s target_kind=%s techniques=%s",
                mode, target_kind, len(technique_keys) or "all",
            )
            try:
                result = STORE.create_evaluation(
                    mode=mode,
                    target_kind=target_kind,
                    target_config=target_config,
                    technique_keys=technique_keys or None,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/evaluations -> 400: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            log.info("POST /api/evaluations -> 201 eval_id=%s total=%s", result.get("eval_id"), result.get("total"))
            self._send_json({"item": result}, status=HTTPStatus.CREATED)
            return

        if route == "/api/datasets":
            payload = self._read_json_or_400()
            if payload is None:
                return
            name = str(payload.get("name", "")).strip()
            description = str(payload.get("description", "")).strip()
            prompts = payload.get("prompts")
            if not name:
                self._send_json({"error": "`name` is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(prompts, list) or not prompts:
                self._send_json({"error": "`prompts` must be a non-empty list."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                dataset = STORE.create_dataset(name=name, description=description, prompts=prompts)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"item": dataset}, status=HTTPStatus.CREATED)
            return

        if route.startswith("/api/experiments/") and route.endswith("/stop"):
            experiment_id = route.split("/")[-2]
            result = STORE.stop_experiment(experiment_id)
            if result is None:
                self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"item": result})
            return

        if route.startswith("/api/experiments/") and route.endswith("/retry"):
            experiment_id = route.split("/")[-2]
            payload = self._read_json_or_400()
            if payload is None:
                return
            target_config = payload.get("target_config") or {}
            if not _has_target_credentials(target_config):
                self._send_json(
                    {"error": "`target_config.api_key` or `target_config.target_profile_id` is required."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                result = STORE.retry_experiment_errors(experiment_id, target_config)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if result is None:
                self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"item": result}, status=HTTPStatus.ACCEPTED)
            return

        if route.startswith("/api/experiments/") and route.endswith("/rejudge"):
            experiment_id = route.split("/")[-2]
            payload = self._read_json_or_400()
            if payload is None:
                return
            judge_config = payload.get("judge_config") or payload.get("target_config") or {}
            try:
                result = STORE.rejudge_experiment(experiment_id, judge_config)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if result is None:
                self._send_json({"error": "Experiment not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"item": result}, status=HTTPStatus.ACCEPTED)
            return

        if route == "/api/experiments/import":
            payload = self._read_json_or_400()
            if payload is None:
                return
            try:
                result = STORE.import_experiment(payload)
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/experiments/import -> 500: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"item": result}, status=HTTPStatus.CREATED)
            return

        if route == "/api/experiments":
            payload = self._read_json_or_400()
            if payload is None:
                return
            name = str(payload.get("name", "")).strip()
            description = str(payload.get("description", "")).strip()
            hypothesis = str(payload.get("hypothesis", "")).strip()
            dataset_id = str(payload.get("dataset_id", "")).strip()
            technique_keys = payload.get("technique_keys") or []
            target_config = payload.get("target_config") or {}
            mode = str(payload.get("mode", "live")).strip()
            repeat_count = payload.get("repeat_count", 1)
            max_rounds = payload.get("max_rounds", 5)
            experiment_group = str(payload.get("experiment_group", "")).strip()

            if not name:
                self._send_json({"error": "`name` is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            if not dataset_id:
                self._send_json({"error": "`dataset_id` is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            if not technique_keys:
                self._send_json({"error": "`technique_keys` must not be empty."}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                result = STORE.create_experiment(
                    name=name,
                    description=description,
                    hypothesis=hypothesis,
                    dataset_id=dataset_id,
                    technique_keys=technique_keys,
                    target_config=target_config,
                    mode=mode,
                    repeat_count=repeat_count,
                    max_rounds=max_rounds,
                    experiment_group=experiment_group,
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/experiments -> 500: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"item": result}, status=HTTPStatus.ACCEPTED)
            return

        if route == "/api/analysis/query":
            payload = self._read_json_or_400()
            if payload is None:
                return
            sql = str(payload.get("sql", ""))
            try:
                result = STORE.run_analysis_query(sql)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/analysis/query -> 500: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)
            return

        if route == "/api/llm-vault/create":
            payload = self._read_json_or_400()
            if payload is None:
                return
            try:
                result = STORE.create_vault(str(payload.get("passphrase", "")))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/llm-vault/create -> 500: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result, status=HTTPStatus.CREATED)
            return

        if route == "/api/llm-vault/unlock":
            payload = self._read_json_or_400()
            if payload is None:
                return
            try:
                result = STORE.unlock_vault(str(payload.get("passphrase", "")))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/llm-vault/unlock -> 500: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)
            return

        if route == "/api/llm-vault/lock":
            self._send_json(STORE.lock_vault())
            return

        if route == "/api/llm-profiles":
            payload = self._read_json_or_400()
            if payload is None:
                return
            try:
                item = STORE.save_llm_profile(payload)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/llm-profiles -> 500: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"item": item}, status=HTTPStatus.CREATED)
            return

        if route == "/api/custom-techniques":
            payload = self._read_json_or_400()
            if payload is None:
                return
            definition = payload.get("definition")
            if not isinstance(definition, dict):
                self._send_json({"error": "`definition` is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                item = STORE.save_custom_technique(definition=definition)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("POST /api/custom-techniques -> 500: %s", exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"item": item}, status=HTTPStatus.CREATED)
            return

        log.warning("POST %s -> 404", route)
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        log.info("DELETE %s", route)
        if route.startswith("/api/custom-techniques/"):
            technique_id = route.split("/")[-1]
            try:
                STORE.delete_custom_technique(technique_id)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("DELETE %s -> 500: %s", route, exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"ok": True})
            return
        if route.startswith("/api/llm-profiles/"):
            profile_id = route.split("/")[-1]
            try:
                STORE.delete_llm_profile(profile_id)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("DELETE %s -> 500: %s", route, exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"ok": True})
            return
        log.warning("DELETE %s -> 404", route)
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # Suppress default BaseHTTPRequestHandler stderr logging; we use our own.
        return

    def _read_json_or_400(self) -> dict[str, object] | None:
        try:
            return self._read_json()
        except JsonBodyError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return None

    def _validate_content_length_or_400(self) -> bool:
        try:
            self._content_length()
        except JsonBodyError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return False
        return True

    def _content_length(self) -> int:
        get_all = getattr(self.headers, "get_all", None)
        if callable(get_all):
            raw_lengths = get_all("Content-Length") or []
            if len(raw_lengths) > 1:
                raise JsonBodyError("Invalid Content-Length header.")
            raw_length = raw_lengths[0] if raw_lengths else "0"
        else:
            raw_length = self.headers.get("Content-Length", "0")

        normalized = str(raw_length).strip()
        if not normalized.isascii() or not normalized.isdecimal():
            raise JsonBodyError("Invalid Content-Length header.")
        return int(normalized)

    def _read_json(self) -> dict[str, object]:
        length = self._content_length()
        if length == 0:
            return {}

        raw = self.rfile.read(length)
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JsonBodyError("Invalid JSON request body.") from exc
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise JsonBodyError("Invalid JSON request body.") from exc
        if not isinstance(payload, dict):
            raise JsonBodyError("JSON request body must be an object.")
        return payload

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self._write_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_cors_headers(self) -> None:
        allowed_origin = _cors_origin_for_request(
            self.headers.get("Origin"),
            _allowed_cors_origins(),
        )
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")


def _query_value(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key) or []
    if not values:
        return default
    return str(values[0]).strip()


def _query_int(
    query: dict[str, list[str]],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = _query_value(query, key, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _run_history_filters(query: dict[str, list[str]]) -> dict[str, str]:
    keys = (
        "q",
        "technique_key",
        "technique_family",
        "technique_surface",
        "status",
        "highest_severity",
        "mode",
        "target_kind",
        "target_model",
        "success",
        "completed_after",
        "completed_before",
        "min_score",
        "max_score",
        "min_asr",
        "max_asr",
    )
    return {key: _query_value(query, key) for key in keys if _query_value(query, key)}


def run() -> None:
    host, port = _api_bind_address()
    server = ThreadingHTTPServer((host, port), ApiHandler)
    log.info("Backend API listening on http://%s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    run()


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _has_target_credentials(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    if config.get("target_profile_id"):
        return True
    if config.get("target_local"):
        return True
    return bool(config.get("api_key"))
