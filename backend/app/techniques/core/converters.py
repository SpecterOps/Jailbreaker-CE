"""Declared converter registry with source-artifact provenance."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

import jsonschema

from app.techniques.contracts.definition import ConverterDefinition
from app.techniques.contracts.execution import _freeze_json_mapping
from app.techniques.core.cancellation import CancellationToken
from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


class ConverterPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConversionResult:
    conversion_id: str
    converter_id: str
    source_artifact_id: str
    output: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _freeze_json_mapping(self.output))


class ConverterService:
    def __init__(
        self,
        *,
        definitions: tuple[ConverterDefinition, ...],
        registry: Mapping[str, Callable[[str], Mapping[str, object]]],
        events: EventJournal,
        cancellation: CancellationToken,
        redaction: RedactionService | None = None,
    ) -> None:
        self._definitions = MappingProxyType({item.id: item for item in definitions})
        self._registry = MappingProxyType(dict(registry))
        self._events = events
        self._cancellation = cancellation
        self._redaction = redaction or RedactionService()

    def convert(self, converter_id: str, *, artifact_id: str) -> ConversionResult:
        definition = self._definitions.get(converter_id)
        if definition is None:
            raise ConverterPolicyError(f"undeclared converter '{converter_id}'")
        implementation = self._registry.get(converter_id)
        if implementation is None:
            raise ConverterPolicyError(f"converter '{converter_id}' is not registered")
        self._cancellation.raise_if_cancelled()
        output = implementation(artifact_id)
        if not isinstance(output, Mapping):
            raise ConverterPolicyError(
                f"converter '{converter_id}' returned a non-mapping result"
            )
        if definition.output_schema:
            try:
                jsonschema.validate(output, definition.output_schema)
            except jsonschema.ValidationError as error:
                raise ConverterPolicyError(
                    f"converter '{converter_id}' result failed schema validation: "
                    f"{error.message}"
                ) from error
        result = ConversionResult(str(uuid4()), converter_id, artifact_id, output)
        self._events.emit(
            "converter.completed",
            operation_id=result.conversion_id,
            payload={
                "conversion_id": result.conversion_id,
                "converter_id": converter_id,
                "source_artifact_id": artifact_id,
                "output": self._redaction.redact_converter(result.output),
            },
        )
        return result
