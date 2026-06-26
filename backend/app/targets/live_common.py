"""Shared helpers for live target adapters."""

from __future__ import annotations

import json


_FLOAT_GENERATION_KEYS = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
}

_STRING_GENERATION_KEYS = {
    "stop",
    "logit_bias",
}

_ALLOWED_GENERATION_KEYS = _FLOAT_GENERATION_KEYS | _STRING_GENERATION_KEYS


def _parse_logit_bias(value: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def parse_generation_overrides(overrides: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in overrides.items():
        if key not in _ALLOWED_GENERATION_KEYS:
            continue
        if key in _FLOAT_GENERATION_KEYS:
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = str(value)
        elif key == "logit_bias":
            logit_bias = _parse_logit_bias(value)
            if logit_bias is not None:
                parsed[key] = logit_bias
        else:
            parsed[key] = str(value)
    return parsed


def messages_to_trace_input(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
