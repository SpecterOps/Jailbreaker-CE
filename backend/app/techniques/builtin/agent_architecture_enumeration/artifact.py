"""Artifact boundary for modular agent-architecture enumeration."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Mapping

from jsonschema import Draft202012Validator

from app.attacklib.architecture_inventory import build_architecture_artifact


MODULAR_ARCHITECTURE_SCHEMA_VERSION = "agent-architecture-map/v2"
_TECHNIQUE_ID = "agent_architecture_enumeration"
_PHASES = ("inventory", "data_layer", "architecture")
_PHASE_ORDER = {phase: index for index, phase in enumerate(_PHASES)}
_CONFIDENCE_LEVELS = ("confirmed", "inferred", "unknown")
_MAPPABLE_SECTIONS = ("agents", "tools", "data_stores", "workflow", "limits")
_ROUND_PROVENANCE_FIELDS = ("round_number", "phase", "question")

TARGET_RESPONSE_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "phase",
        "agent",
        "tools",
        "data_stores",
        "workflow",
        "limits",
        "unknowns",
    ],
    "properties": {
        "phase": {"type": "string", "enum": list(_PHASES)},
        "agent": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "profile", "model_aliases", "confidence"],
            "properties": {
                "name": {"type": ["string", "null"]},
                "profile": {"type": ["string", "null"]},
                "model_aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {
                    "type": "string",
                    "enum": list(_CONFIDENCE_LEVELS),
                },
            },
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "type",
                    "read_scope",
                    "write_scope",
                    "evidence",
                    "confidence",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "tool",
                            "connector",
                            "retriever",
                            "database",
                            "memory",
                            "handoff",
                            "unknown",
                        ],
                    },
                    "read_scope": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "write_scope": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": list(_CONFIDENCE_LEVELS),
                    },
                },
            },
        },
        "data_stores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "kind",
                    "parent",
                    "fields",
                    "join_keys",
                    "description",
                    "confidence",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "database",
                            "schema",
                            "table",
                            "collection",
                            "index",
                            "knowledge_base",
                            "unknown",
                        ],
                    },
                    "parent": {"type": ["string", "null"]},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "join_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": list(_CONFIDENCE_LEVELS),
                    },
                },
            },
        },
        "workflow": {
            "type": "object",
            "additionalProperties": False,
            "required": ["framework", "nodes", "edges", "handoff_paths"],
            "properties": {
                "framework": {"type": ["string", "null"]},
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "name",
                            "role",
                            "inputs",
                            "outputs",
                            "tools_used",
                            "confidence",
                        ],
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                            "inputs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "outputs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "tools_used": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {
                                "type": "string",
                                "enum": list(_CONFIDENCE_LEVELS),
                            },
                        },
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["from", "to", "condition", "confidence"],
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "condition": {"type": ["string", "null"]},
                            "confidence": {
                                "type": "string",
                                "enum": list(_CONFIDENCE_LEVELS),
                            },
                        },
                    },
                },
                "handoff_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "limits": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "write_access",
                "network_access",
                "persistence",
                "cross_agent_handoff",
            ],
            "properties": {
                "write_access": {"type": ["string", "null"]},
                "network_access": {"type": ["string", "null"]},
                "persistence": {"type": ["string", "null"]},
                "cross_agent_handoff": {"type": ["string", "null"]},
            },
        },
        "unknowns": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_TARGET_RESPONSE_VALIDATOR = Draft202012Validator(TARGET_RESPONSE_SCHEMA)


def build_modular_architecture_artifact(
    rounds: list[dict[str, object]],
) -> dict[str, object]:
    normalized_rounds = sorted(
        (_normalize_round(round_item) for round_item in rounds),
        key=_round_merge_key,
    )
    artifact = build_architecture_artifact(
        technique=_TECHNIQUE_ID,
        rounds=normalized_rounds,
        schema_version=MODULAR_ARCHITECTURE_SCHEMA_VERSION,
    )
    quality = artifact["quality"]
    if isinstance(quality, dict):
        mapped_sections = quality.get("mapped_sections", [])
        quality["missing_sections"] = [
            section for section in _MAPPABLE_SECTIONS if section not in mapped_sections
        ]
    return artifact


def stable_modular_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _round_merge_key(round_item: dict[str, object]) -> tuple[int, int, str]:
    round_number = round_item.get("round_number")
    if (
        isinstance(round_number, int)
        and not isinstance(round_number, bool)
        and round_number > 0
    ):
        return (0, round_number, stable_modular_json(round_item))
    return (
        1,
        _PHASE_ORDER.get(str(round_item.get("phase")), len(_PHASE_ORDER)),
        stable_modular_json(round_item),
    )


def _normalize_round(round_item: dict[str, object]) -> dict[str, object]:
    normalized = {
        field: deepcopy(round_item[field])
        for field in _ROUND_PROVENANCE_FIELDS
        if field in round_item
    }
    response, raw_response, parse_error = _validated_response(round_item)
    if response is not None:
        normalized["response"] = response
        normalized["parse_error"] = None
    else:
        normalized["raw_response"] = raw_response
        normalized["parse_error"] = parse_error
    return normalized


def _validated_response(
    round_item: dict[str, object],
) -> tuple[dict[str, object] | None, str, str | None]:
    supplied_error = round_item.get("parse_error")
    response = round_item.get("response")
    raw_response = round_item.get("raw_response")

    if supplied_error is not None:
        return None, _raw_text(response, raw_response), _redacted_error(supplied_error)

    if isinstance(response, dict):
        candidate = response
        raw_text = stable_modular_json(response)
    elif isinstance(raw_response, str):
        raw_text = raw_response
        try:
            decoded = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            return None, raw_response, "invalid_json"
        if not isinstance(decoded, dict):
            return None, raw_response, "invalid_response_schema"
        candidate = decoded
    else:
        return None, _raw_text(response, raw_response), "missing_response"

    if not _response_is_valid(candidate, round_item.get("phase")):
        return None, raw_text, "invalid_response_schema"
    return _copy_declared_response(candidate), "", None


def _response_is_valid(candidate: dict[str, object], phase: object) -> bool:
    if any(_TARGET_RESPONSE_VALIDATOR.iter_errors(candidate)):
        return False
    return not isinstance(phase, str) or candidate["phase"] == phase


def _copy_declared_response(candidate: dict[str, object]) -> dict[str, object]:
    properties = TARGET_RESPONSE_SCHEMA["properties"]
    assert isinstance(properties, dict)
    return {
        field: deepcopy(candidate[field])
        for field in properties
        if field in candidate
    }


def _raw_text(response: object, raw_response: object) -> str:
    if isinstance(raw_response, str):
        return raw_response
    if isinstance(response, dict):
        return stable_modular_json(response)
    if response is None:
        return ""
    return str(response)


def _redacted_error(error: object) -> str:
    text = str(error).casefold()
    if "json" in text:
        return "invalid_json"
    if "schema" in text or "valid" in text:
        return "invalid_response_schema"
    return "invalid_response"
