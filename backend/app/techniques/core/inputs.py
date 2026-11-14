"""Manifest-declared input materialization without host-path access."""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping

import jsonschema

from app.techniques.contracts.definition import InputDefinition
from app.techniques.contracts.execution import _freeze_json_mapping


class InputPolicyError(ValueError):
    pass


class InputService:
    def __init__(
        self,
        *,
        definitions: tuple[InputDefinition, ...],
        owned_artifact_ids: frozenset[str] = frozenset(),
        artifact_exists: Callable[[str], bool] | None = None,
        artifact_resolver: Callable[[str], object | None] | None = None,
    ) -> None:
        self._definitions = MappingProxyType(
            {definition.id: definition for definition in definitions}
        )
        self._owned_artifact_ids = owned_artifact_ids
        self._artifact_exists = artifact_exists
        self._artifact_resolver = artifact_resolver

    def materialize(self, values: Mapping[str, object]) -> Mapping[str, object]:
        unknown = set(values) - set(self._definitions)
        if unknown:
            raise InputPolicyError(f"undeclared inputs: {sorted(unknown)!r}")
        missing = [
            item.id
            for item in self._definitions.values()
            if item.required and item.id not in values
        ]
        if missing:
            raise InputPolicyError(f"missing required inputs: {missing!r}")
        materialized: dict[str, object] = {}
        for input_id, value in values.items():
            definition = self._definitions[input_id]
            materialized[input_id] = self._materialize_value(definition, value)
        return _freeze_json_mapping(materialized)

    def _materialize_value(self, definition: InputDefinition, value: object) -> object:
        items = value if isinstance(value, (list, tuple)) else (value,)
        if definition.max_count is not None and len(items) > definition.max_count:
            raise InputPolicyError(
                f"input '{definition.id}' exceeds maximum count {definition.max_count}"
            )
        for item in items:
            if isinstance(item, Mapping) and "path" in item:
                raise InputPolicyError(
                    f"input '{definition.id}' requires an owned artifact reference; "
                    "host paths are not accepted"
                )
            if isinstance(item, Mapping) and "artifact_id" in item:
                artifact_id = item["artifact_id"]
                is_owned = artifact_id in self._owned_artifact_ids or (
                    isinstance(artifact_id, str)
                    and self._artifact_exists is not None
                    and self._artifact_exists(artifact_id)
                )
                if not is_owned:
                    raise InputPolicyError(
                        f"artifact reference '{artifact_id}' is not owned by this run"
                    )
                authoritative = (
                    self._artifact_resolver(artifact_id)
                    if isinstance(artifact_id, str)
                    and self._artifact_resolver is not None
                    else None
                )
                media_type = _artifact_field(authoritative, "media_type")
                size = _artifact_field(authoritative, "size_bytes")
                if authoritative is None and (
                    definition.media_types or definition.max_size_bytes is not None
                ):
                    raise InputPolicyError(
                        f"artifact reference '{artifact_id}' has no authoritative metadata"
                    )
                if (
                    definition.media_types
                    and media_type not in definition.media_types
                ):
                    raise InputPolicyError(
                        f"input '{definition.id}' media type '{media_type}' is not allowed"
                    )
                if (
                    definition.max_size_bytes is not None
                    and isinstance(size, int)
                    and size > definition.max_size_bytes
                ):
                    raise InputPolicyError(
                        f"input '{definition.id}' exceeds its size limit"
                    )
        if definition.schema:
            try:
                jsonschema.validate(value, _plain_schema(definition.schema))
            except jsonschema.ValidationError as error:
                raise InputPolicyError(
                    f"input '{definition.id}' does not match its schema: {error.message}"
                ) from error
        return value


def _artifact_field(artifact: object | None, name: str) -> object:
    if artifact is None:
        return None
    if isinstance(artifact, Mapping):
        return artifact.get(name)
    return getattr(artifact, name, None)


def _plain_schema(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_schema(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_schema(item) for item in value]
    return value
