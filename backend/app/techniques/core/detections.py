"""Declaration-driven detector invocation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

import jsonschema

from app.techniques.contracts.definition import DetectorDefinition
from app.techniques.contracts.execution import _freeze_json_mapping
from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


class DetectionPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DetectionEvidence:
    detection_id: str
    detector_id: str
    matched: bool
    rationale: str
    output: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _freeze_json_mapping(self.output))


class DetectionService:
    def __init__(
        self,
        *,
        definitions: tuple[DetectorDefinition, ...],
        registry: Mapping[
            str,
            Callable[[Mapping[str, object]], Mapping[str, object]],
        ],
        events: EventJournal,
        redaction: RedactionService | None = None,
    ) -> None:
        self._definitions = MappingProxyType({item.id: item for item in definitions})
        self._registry = MappingProxyType(dict(registry))
        self._events = events
        self._redaction = redaction or RedactionService()

    def run(
        self,
        detector_id: str,
        *,
        inputs: Mapping[str, object],
    ) -> DetectionEvidence:
        definition = self._definitions.get(detector_id)
        if definition is None:
            raise DetectionPolicyError(f"undeclared detector '{detector_id}'")
        implementation = self._registry.get(detector_id)
        if implementation is None:
            raise DetectionPolicyError(f"detector '{detector_id}' is not registered")
        missing = set(definition.inputs) - set(inputs)
        if missing:
            raise DetectionPolicyError(
                f"detector '{detector_id}' is missing inputs {sorted(missing)!r}"
            )
        output = implementation(inputs)
        if definition.output_schema:
            try:
                jsonschema.validate(output, definition.output_schema)
            except jsonschema.ValidationError as error:
                raise DetectionPolicyError(
                    f"detector '{detector_id}' output is invalid: {error.message}"
                ) from error
        result = DetectionEvidence(
            detection_id=str(uuid4()),
            detector_id=detector_id,
            matched=bool(output.get("matched", False)),
            rationale=str(output.get("rationale", "")),
            output=output,
        )
        self._events.emit(
            "detector.completed",
            operation_id=result.detection_id,
            payload={
                "detection_id": result.detection_id,
                "detector_id": detector_id,
                "matched": result.matched,
                "rationale": result.rationale,
                "output": self._redaction.redact_detector(result.output),
            },
        )
        return result
