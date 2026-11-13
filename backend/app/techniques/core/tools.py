"""Manifest-declared access to source-controlled runtime tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

import jsonschema

from app.techniques.contracts.definition import ToolDefinition
from app.techniques.contracts.execution import _freeze_json_mapping
from app.techniques.core.budgets import BudgetService
from app.techniques.core.cancellation import CancellationToken
from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


class ToolPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    tool_call_id: str
    tool_id: str
    output: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _freeze_json_mapping(self.output))


class ToolService:
    def __init__(
        self,
        *,
        definitions: tuple[ToolDefinition, ...],
        registry: Mapping[str, Callable[[Mapping[str, object]], Mapping[str, object]]],
        events: EventJournal,
        budget: BudgetService,
        cancellation: CancellationToken,
        redaction: RedactionService | None = None,
    ) -> None:
        self._definitions = MappingProxyType({item.id: item for item in definitions})
        self._registry = MappingProxyType(dict(registry))
        self._events = events
        self._budget = budget
        self._cancellation = cancellation
        self._redaction = redaction or RedactionService()

    async def invoke(
        self,
        tool_id: str,
        *,
        arguments: Mapping[str, object],
    ) -> ToolCallResult:
        definition = self._definitions.get(tool_id)
        if definition is None:
            raise ToolPolicyError(f"undeclared tool '{tool_id}'")
        implementation = self._registry.get(tool_id)
        if implementation is None:
            raise ToolPolicyError(f"tool '{tool_id}' is not registered")
        self._validate(arguments, definition.input_schema, tool_id, "arguments")
        self._cancellation.raise_if_cancelled()
        self._budget.reserve(tool_calls=1)
        call_id = str(uuid4())
        self._events.emit(
            "tool.requested",
            operation_id=call_id,
            payload={
                "tool_call_id": call_id,
                "tool_id": tool_id,
                "arguments": self._redaction.redact_tool(arguments),
            },
        )
        try:
            output = await asyncio.to_thread(implementation, arguments)
            if not isinstance(output, Mapping):
                raise ToolPolicyError(f"tool '{tool_id}' returned a non-mapping result")
            self._validate(output, definition.output_schema, tool_id, "result")
        except Exception as error:
            self._events.emit(
                "tool.failed",
                operation_id=call_id,
                payload={
                    "tool_call_id": call_id,
                    "tool_id": tool_id,
                    "error": self._redaction.redact_text(str(error)),
                },
            )
            raise
        result = ToolCallResult(call_id, tool_id, output)
        self._events.emit(
            "tool.completed",
            operation_id=call_id,
            payload={
                "tool_call_id": call_id,
                "tool_id": tool_id,
                "output": self._redaction.redact_tool(result.output),
            },
        )
        return result

    def capture(self, tool_id: str, *, output: Mapping[str, object]) -> ToolCallResult:
        if tool_id not in self._definitions:
            raise ToolPolicyError(f"undeclared tool '{tool_id}'")
        result = ToolCallResult(str(uuid4()), tool_id, output)
        self._events.emit(
            "tool.captured",
            operation_id=result.tool_call_id,
            payload={
                "tool_call_id": result.tool_call_id,
                "tool_id": tool_id,
                "output": self._redaction.redact_tool(result.output),
            },
        )
        return result

    @staticmethod
    def _validate(
        value: Mapping[str, object],
        schema: Mapping[str, object],
        tool_id: str,
        label: str,
    ) -> None:
        if not schema:
            return
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError as error:
            raise ToolPolicyError(
                f"tool '{tool_id}' {label} failed schema validation: {error.message}"
            ) from error
