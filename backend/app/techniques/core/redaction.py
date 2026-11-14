"""Central recursive redaction for persisted runtime evidence."""

from __future__ import annotations

import json
import math
import re
from typing import Mapping

from app.techniques.contracts.execution import _freeze_json_mapping


_REDACTED = "[REDACTED]"
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:bearer\s+)?(?:sk|rk|pk)-[a-z0-9_-]{6,}(?![a-z0-9_-])"
)
_STRONG_SECRET_KEY_SEGMENTS = frozenset(
    {"credential", "credentials", "passphrase", "password", "secret"}
)
_SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "client_secret",
        "id_token",
        "password",
        "passphrase",
        "private_key",
        "secret",
        "secret_key",
        "session_token",
        "token",
        "access_token",
        "auth_token",
        "bearer_token",
        "refresh_token",
        "x_api_key",
    }
)
_STRUCTURAL_STRING_KEYS = frozenset(
    {
        "id",
        "model",
        "provider",
        "technique_key",
        "transport",
        "version",
    }
)
_STRUCTURAL_STRING_SEQUENCE_KEYS = frozenset(
    {"auxiliary_judges", "technique_keys"}
)


def is_secret_key(key: object) -> bool:
    normalized = _normalize_key(str(key))
    segments = frozenset(normalized.split("_"))
    return (
        normalized in _SECRET_KEY_NAMES
        or bool(segments & _STRONG_SECRET_KEY_SEGMENTS)
        or normalized.endswith(
            (
                "_api_key",
                "_authorization",
                "_cookie",
                "_password",
                "_passphrase",
                "_private_key",
                "_credential",
                "_credentials",
                "_secret",
                "_secret_key",
                "_token",
                "_access_token",
                "_auth_token",
                "_bearer_token",
                "_refresh_token",
            )
        )
    )


def contains_credential_text(value: str) -> bool:
    """Return whether text contains a bounded credential-shaped token."""
    return _CREDENTIAL_PATTERN.search(value) is not None


def redact_credential_text(value: str) -> str:
    """Redact bounded credential-shaped tokens from text."""
    return _CREDENTIAL_PATTERN.sub(_REDACTED, value)


def contains_secret_material(value: object) -> bool:
    """Return whether nested canonical/public material contains a secret."""
    return _contains_secret_material(value, inspect_strings=True)


def _contains_secret_material(value: object, *, inspect_strings: bool) -> bool:
    if isinstance(value, Mapping):
        return any(
            is_secret_key(key)
            or _contains_secret_material(
                item,
                inspect_strings=_inspect_child_strings(key, item),
            )
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(
            _contains_secret_material(
                item,
                inspect_strings=_inspect_sequence_item_strings(
                    item,
                    inspect_strings=inspect_strings,
                ),
            )
            for item in value
        )
    return isinstance(value, str) and inspect_strings and contains_credential_text(value)


def sanitize_public_value(value: object) -> object:
    """Return a deterministic JSON-safe copy without credential fields."""
    return _sanitize_public_value(value, inspect_strings=True)


def _sanitize_public_value(value: object, *, inspect_strings: bool) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_public_value(
                item,
                inspect_strings=_inspect_child_strings(key, item),
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not is_secret_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_public_value(
                item,
                inspect_strings=_inspect_sequence_item_strings(
                    item,
                    inspect_strings=inspect_strings,
                ),
            )
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        items = [
            _sanitize_public_value(
                item,
                inspect_strings=_inspect_sequence_item_strings(
                    item,
                    inspect_strings=inspect_strings,
                ),
            )
            for item in value
        ]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        text = getattr(value, "value")
        return redact_credential_text(text) if inspect_strings else text
    if isinstance(value, str):
        return redact_credential_text(value) if inspect_strings else value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def sanitize_public_mapping(value: Mapping[object, object]) -> dict[str, object]:
    sanitized = sanitize_public_value(value)
    if not isinstance(sanitized, dict):  # pragma: no cover - invariant
        raise TypeError("expected a mapping")
    return sanitized


class RedactionService:
    def __init__(
        self,
        *,
        secret_keys: frozenset[str] = _SECRET_KEY_NAMES,
        secret_values: tuple[str, ...] = (),
    ) -> None:
        self._secret_keys = frozenset(_normalize_key(key) for key in secret_keys)
        self._secret_values = tuple(
            sorted(
                (value for value in secret_values if value),
                key=len,
                reverse=True,
            )
        )

    def redact_text(self, value: str) -> str:
        return self._redact_text(value, inspect_credentials=True)

    def _redact_text(self, value: str, *, inspect_credentials: bool) -> str:
        redacted = redact_credential_text(value) if inspect_credentials else value
        for secret in self._secret_values:
            redacted = redacted.replace(secret, _REDACTED)
        return redacted

    def redact_mapping(
        self,
        value: Mapping[str, object],
    ) -> Mapping[str, object]:
        redacted = self._redact_value(value, inspect_strings=True)
        if not isinstance(redacted, Mapping):  # pragma: no cover - invariant
            raise TypeError("expected a mapping")
        return _freeze_json_mapping(redacted)

    def redact_target(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return self.redact_mapping(value)

    def redact_judge(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return self.redact_mapping(value)

    def redact_detector(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return self.redact_mapping(value)

    def redact_seed(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return self.redact_mapping(value)

    def redact_tool(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return self.redact_mapping(value)

    def redact_converter(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return self.redact_mapping(value)

    def redact_artifact(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return self.redact_mapping(value)

    def _redact_value(self, value: object, *, inspect_strings: bool) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): (
                    _REDACTED
                    if is_secret_key(key)
                    or _normalize_key(str(key)) in self._secret_keys
                    else self._redact_value(
                        item,
                        inspect_strings=_inspect_child_strings(key, item),
                    )
                )
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [
                self._redact_value(
                    item,
                    inspect_strings=_inspect_sequence_item_strings(
                        item,
                        inspect_strings=inspect_strings,
                    ),
                )
                for item in value
            ]
        if isinstance(value, str):
            return self._redact_text(
                value,
                inspect_credentials=inspect_strings,
            )
        return value


def _is_structural_string_key(key: object) -> bool:
    normalized = _normalize_key(str(key))
    return (
        normalized in _STRUCTURAL_STRING_KEYS
        or normalized.endswith(("_hash", "_hashes", "_id", "_ids", "_version"))
    )


def _inspect_child_strings(key: object, value: object) -> bool:
    if isinstance(value, Mapping):
        return True
    normalized = _normalize_key(str(key))
    if isinstance(value, (list, tuple, set, frozenset)):
        return normalized not in _STRUCTURAL_STRING_SEQUENCE_KEYS
    return not _is_structural_string_key(key)


def _inspect_sequence_item_strings(
    value: object,
    *,
    inspect_strings: bool,
) -> bool:
    """Limit a structural sequence exemption to its immediate string values."""
    return inspect_strings if isinstance(value, str) else True


def _normalize_key(value: str) -> str:
    word_bounded = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value.strip())
    word_bounded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", word_bounded)
    return re.sub(r"[^a-z0-9]+", "_", word_bounded.lower()).strip("_")
