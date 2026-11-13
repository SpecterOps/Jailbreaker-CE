"""Shared value conversion, fingerprints, and redacted report serialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType

from app.techniques.core.redaction import is_secret_key, redact_credential_text, sanitize_public_value

_RESPONSE_KEYS = frozenset(
    {
        "response",
        "raw_response",
        "model_response",
        "target_response",
        "final_response",
        "final_output",
        "raw_output",
        "judge_output",
        "attacker_output",
        "poisoned_response",
    }
)


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_plain(item) for item in value]
        return sorted(normalized, key=_sort_key)
    return value


def _sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalized_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _content_fingerprint(value: str) -> dict[str, object]:
    normalized = _normalized_text(value)
    return {
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "length": len(normalized),
    }


def _as_mapping(value: object, *, label: str) -> dict[str, object]:
    plain = _plain(value)
    if not isinstance(plain, Mapping):
        raise TypeError(f"{label} must be a mapping or dataclass")
    return {str(key): item for key, item in plain.items()}


def _safe_evidence(value: object) -> object:
    sanitized = sanitize_public_value(_plain(value))
    if isinstance(sanitized, Mapping):
        return {str(key): _safe_evidence(item) for key, item in sanitized.items()}
    if isinstance(sanitized, list):
        return [_safe_evidence(item) for item in sanitized]
    if isinstance(sanitized, str):
        return _content_fingerprint(sanitized)
    return sanitized


def _sanitize_report_value(value: object, *, key: str | None = None) -> object:
    plain = _plain(value)
    if key is not None and key.lower() in _RESPONSE_KEYS and isinstance(plain, str):
        return _content_fingerprint(plain)
    if isinstance(plain, Mapping):
        return {
            str(child_key): _sanitize_report_value(item, key=str(child_key))
            for child_key, item in sorted(plain.items(), key=lambda pair: str(pair[0]))
            if not is_secret_key(child_key)
        }
    if isinstance(plain, list):
        return [_sanitize_report_value(item) for item in plain]
    if isinstance(plain, str):
        safe = redact_credential_text(plain)
        return _content_fingerprint(safe) if len(safe) > 160 else safe
    if plain is None or isinstance(plain, (bool, int, float)):
        return plain
    return str(plain)
