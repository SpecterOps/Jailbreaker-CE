"""Explicit judge lifecycle with validated scores distinct from failure."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from app.judge_policy import (
    judge_policy_from_config,
    validate_configurable_judge_output,
)
from app.judge_scores import validate_runtime_judge_score
from app.techniques.contracts.definition import JudgeDefinition, JudgePurpose
from app.techniques.contracts.events import JudgeResult, JudgeStatus
from app.techniques.core.budgets import BudgetExhausted, BudgetService
from app.techniques.core.cancellation import CancellationError, CancellationToken
from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


class JudgeModelService(Protocol):
    async def call(self, **kwargs: object) -> object: ...

    def token_capacity(self, **kwargs: object) -> int: ...


@dataclass(frozen=True, slots=True)
class JudgeModelRequest:
    """One fully compiled request shared by judge sizing and execution."""

    messages: tuple[Mapping[str, str], ...]
    options: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("judge model request requires at least one message")
        frozen_messages: list[Mapping[str, str]] = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise TypeError("judge model request roles must be non-empty text")
            if not isinstance(content, str):
                raise TypeError("judge model request content must be text")
            frozen_messages.append(
                MappingProxyType({"role": role, "content": content})
            )
        options = dict(self.options)
        reserved = {"messages", "operation_id", "role"}.intersection(options)
        if reserved:
            raise ValueError(
                "judge model request options contain reserved keys: "
                + ", ".join(sorted(reserved))
            )
        object.__setattr__(self, "messages", tuple(frozen_messages))
        object.__setattr__(self, "options", MappingProxyType(options))


class JudgeRequestComposer(Protocol):
    def compose(
        self,
        *,
        definition: JudgeDefinition,
        evidence: Mapping[str, object],
    ) -> JudgeModelRequest: ...


class JudgeService:
    def __init__(
        self,
        *,
        definitions: tuple[JudgeDefinition, ...],
        models: JudgeModelService,
        events: EventJournal,
        parser: Callable[[str], object],
        request_composer: JudgeRequestComposer,
        budget: BudgetService | None = None,
        cancellation: CancellationToken | None = None,
        redaction: RedactionService | None = None,
    ) -> None:
        self._definitions = MappingProxyType({item.id: item for item in definitions})
        self._models = models
        self._events = events
        self._parser = parser
        self._request_composer = request_composer
        self._budget = budget
        self._cancellation = cancellation
        self._redaction = redaction or RedactionService()

    def token_capacity(
        self,
        *,
        definition_id: str,
        evidence: Mapping[str, object],
    ) -> int:
        """Return the bound judge model's token ceiling without scheduling it."""

        definition = self._definitions.get(definition_id)
        if definition is None:
            raise KeyError(f"judge definition '{definition_id}' does not exist")
        request = self._compose_request(definition=definition, evidence=evidence)
        return self._models.token_capacity(
            role=definition.role or "judge",
            messages=request.messages,
            **request.options,
        )

    async def evaluate(
        self,
        *,
        definition_id: str,
        evidence: Mapping[str, object],
        operation_id: str,
        provenance: Mapping[str, object] | None = None,
        policy_version: str | None = None,
        round: int | None = None,
    ) -> JudgeResult:
        definition = self._definitions.get(definition_id)
        if definition is None:
            raise KeyError(f"judge definition '{definition_id}' does not exist")
        scale_min = max(1, definition.scale_min)
        scale_max = min(10, definition.scale_max)
        if scale_min > scale_max:
            raise ValueError("judge declared scale must overlap the platform range 1 through 10")
        if policy_version is not None and (
            not isinstance(policy_version, str) or not policy_version.strip()
        ):
            raise ValueError("judge policy_version must be a non-empty string")
        request = self._compose_request(definition=definition, evidence=evidence)
        event_provenance = {
            key: value
            for key, value in (provenance or {}).items()
            if key
            not in {
                "definition_id",
                "parse_status",
                "rationale",
                "result_id",
                "score",
                "status",
                "policy_version",
            }
        }
        if policy_version is not None:
            event_provenance["policy_version"] = policy_version
        if self._cancellation is not None:
            self._cancellation.raise_if_cancelled()
        if self._budget is not None:
            self._budget.reserve(judge_calls=1)
        self._events.emit(
            "judge.scheduled",
            operation_id=operation_id,
            round=round,
            payload={**event_provenance, "definition_id": definition_id},
        )
        self._events.emit(
            "judge.running",
            operation_id=operation_id,
            round=round,
            payload={**event_provenance, "definition_id": definition_id},
        )
        started = time.monotonic()
        try:
            model_result = await self._models.call(
                role=definition.role or "judge",
                messages=request.messages,
                operation_id=operation_id,
                **request.options,
            )
        except (CancellationError, BudgetExhausted):
            raise
        except Exception as error:
            result = JudgeResult(
                definition_id=definition.id,
                status=JudgeStatus.FAILED,
                score=None,
                purpose=_primary_purpose(definition),
                scale_min=scale_min,
                scale_max=scale_max,
                model_role=definition.role,
                policy_version=policy_version,
                parse_status=self._redaction.redact_text(
                    f"model_failed: {type(error).__name__}: {error}"
                ),
                latency_ms=(time.monotonic() - started) * 1000,
            )
            self._events.emit(
                "judge.failed",
                operation_id=operation_id,
                round=round,
                payload={
                    **event_provenance,
                    "definition_id": definition_id,
                    "result_id": result.result_id,
                    "parse_status": result.parse_status,
                    "score": None,
                },
            )
            return result
        raw = getattr(model_result, "text", None)
        latency_ms = (time.monotonic() - started) * 1000
        try:
            if not isinstance(raw, str):
                raise TypeError("judge model result must provide string text")
            declared_policy = judge_policy_from_config(
                {"judge_policy": definition.policy}
            )
            if not declared_policy.is_standard:
                validate_configurable_judge_output(raw)
            raw_result = json.loads(raw)
            if not isinstance(raw_result, Mapping):
                raise ValueError("judge result must be an object")
            validate_runtime_judge_score(raw_result.get("score"))
            parsed = self._parser(raw)
            if not isinstance(parsed, Mapping):
                raise ValueError("parsed judge result must be an object")
            score = validate_runtime_judge_score(parsed.get("score"))
            rationale = parsed.get("rationale")
            if rationale is not None and not isinstance(rationale, str):
                raise ValueError("parsed judge rationale must be a string")
            result = JudgeResult(
                definition_id=definition.id,
                status=JudgeStatus.SUCCEEDED,
                score=score,
                purpose=_primary_purpose(definition),
                scale_min=scale_min,
                scale_max=scale_max,
                rationale=(
                    self._redaction.redact_text(rationale)
                    if rationale is not None
                    else None
                ),
                model_role=definition.role,
                policy_version=policy_version,
                raw_output_ref=getattr(model_result, "model_call_id", None),
                parse_status="parsed",
                token_usage=getattr(model_result, "usage", {}),
                latency_ms=latency_ms,
            )
        except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
            result = JudgeResult(
                definition_id=definition.id,
                status=JudgeStatus.FAILED,
                score=None,
                purpose=_primary_purpose(definition),
                scale_min=scale_min,
                scale_max=scale_max,
                model_role=definition.role,
                policy_version=policy_version,
                raw_output_ref=getattr(model_result, "model_call_id", None),
                parse_status=self._redaction.redact_text(f"failed: {error}"),
                token_usage=getattr(model_result, "usage", {}),
                latency_ms=latency_ms,
            )
            self._events.emit(
                "judge.failed",
                operation_id=operation_id,
                round=round,
                payload={
                    **event_provenance,
                    "definition_id": definition_id,
                    "result_id": result.result_id,
                    "parse_status": result.parse_status,
                    "score": None,
                },
            )
            return result
        self._events.emit(
            "judge.completed",
            operation_id=operation_id,
            round=round,
            payload={
                **event_provenance,
                "definition_id": definition_id,
                "result_id": result.result_id,
                "status": result.status.value,
                "score": result.score,
                "rationale": result.rationale,
            },
        )
        return result

    def _compose_request(
        self,
        *,
        definition: JudgeDefinition,
        evidence: Mapping[str, object],
    ) -> JudgeModelRequest:
        request = self._request_composer.compose(
            definition=definition,
            evidence=evidence,
        )
        return JudgeModelRequest(
            messages=tuple(request.messages),
            options=request.options,
        )


def _primary_purpose(definition: JudgeDefinition) -> JudgePurpose | None:
    if JudgePurpose.OFFICIAL_SCORE in definition.purposes:
        return JudgePurpose.OFFICIAL_SCORE
    return next(iter(sorted(definition.purposes, key=str)), None)
