"""Correlated, retry-aware gateway over OpenAI-compatible model clients."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol
from uuid import uuid4

from app.persistence.technique_runtime import (
    ModelCallRecord,
    TechniqueRuntimePersistence,
)
from app.techniques.contracts.execution import (
    CorrelationIds,
    ModelTransport,
    ResolvedRunSpec,
    _freeze_json_mapping,
)
from app.techniques.core.budgets import BudgetExhausted, BudgetService
from app.techniques.core.cancellation import CancellationToken
from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ProviderRequestContext:
    correlation: CorrelationIds
    model_call_id: str
    role: str
    operation_id: str
    technique_id: str
    technique_version: str
    implementation_id: str
    transport: ModelTransport

    def litellm_headers(self) -> Mapping[str, str]:
        if self.transport is not ModelTransport.LITELLM_PROXY:
            return MappingProxyType({})
        metadata = {
            "experiment_id": self.correlation.experiment_id,
            "cell_id": self.correlation.cell_id,
            "run_id": self.correlation.run_id,
            "attempt_id": self.correlation.attempt_id,
            "operation_id": self.operation_id,
            "model_call_id": self.model_call_id,
            "role": self.role,
            "technique_id": self.technique_id,
            "technique_version": self.technique_version,
            "implementation_id": self.implementation_id,
        }
        return MappingProxyType(
            {
                "x-litellm-trace-id": self.correlation.attempt_id,
                "x-litellm-spend-logs-metadata": json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ProviderCallResult:
    text: str
    usage: Mapping[str, object] = field(default_factory=_empty_mapping)
    provider_response_id: str | None = None
    gateway_call_id: str | None = None
    gateway_attempts: Mapping[str, int] = field(default_factory=dict)
    tool_calls: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _freeze_json_mapping(self.usage))
        object.__setattr__(
            self,
            "tool_calls",
            tuple(_freeze_json_mapping(item) for item in self.tool_calls),
        )
        object.__setattr__(self, "gateway_attempts", MappingProxyType(dict(self.gateway_attempts)))


class CorrelatedChatClient(Protocol):
    def chat_correlated_once(
        self,
        messages: list[dict[str, str]],
        *,
        request_context: ProviderRequestContext,
        **options: object,
    ) -> ProviderCallResult: ...


@dataclass(frozen=True, slots=True)
class CallRetryPolicy:
    max_attempts: int = 1
    retry_indeterminate: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def allows(self, failure: ProviderFailure, operation_attempt: int) -> bool:
        return (
            operation_attempt < self.max_attempts
            and failure.retryable
            and (not failure.indeterminate or self.retry_indeterminate)
        )


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    category: str
    retryable: bool
    indeterminate: bool


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    model_call_id: str
    correlation: CorrelationIds
    role: str
    operation_id: str
    operation_attempt: int
    text: str
    usage: Mapping[str, object]
    provider_response_id: str | None = None
    gateway_call_id: str | None = None
    gateway_attempts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _freeze_json_mapping(self.usage))
        object.__setattr__(self, "gateway_attempts", MappingProxyType(dict(self.gateway_attempts)))


class ModelService:
    def __init__(
        self,
        *,
        correlation: CorrelationIds,
        spec: ResolvedRunSpec,
        clients: Mapping[str, CorrelatedChatClient],
        persistence: TechniqueRuntimePersistence,
        events: EventJournal,
        redaction: RedactionService,
        retry_policy: CallRetryPolicy = CallRetryPolicy(),
        budget: BudgetService | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        self.correlation = correlation
        self.spec = spec
        self._clients = MappingProxyType(dict(clients))
        self._persistence = persistence
        self._events = events
        self._redaction = redaction
        self._retry_policy = retry_policy
        self._budget = budget
        self._cancellation = cancellation

    def output_token_limit(self, *, role: str) -> int:
        """Expose the bound client's finite completion limit to target callers."""
        if self.spec.model_role_bindings.get(role) is None or role not in self._clients:
            raise KeyError(f"model role '{role}' has no bound client")
        return _client_default_max_tokens(self._clients[role])

    def token_capacity(
        self,
        *,
        role: str,
        messages: tuple[dict[str, str], ...],
        **options: object,
    ) -> int:
        """Return the exact token ceiling a call would reserve, without charging it."""

        if self.spec.model_role_bindings.get(role) is None:
            raise KeyError(f"model role '{role}' is not bound")
        client = self._clients.get(role)
        if client is None:
            raise KeyError(f"model role '{role}' has no client")
        if self._budget is None or self._budget.remaining("tokens") is None:
            requested_completion = _positive_token_option(
                options.get("max_tokens", _client_default_max_tokens(client))
            )
            return _message_token_ceiling(messages) + (requested_completion or 0)
        _, reserved_tokens, _ = _bounded_model_options(
            messages,
            options,
            self._budget,
            default_max_tokens=_client_default_max_tokens(client),
        )
        return reserved_tokens

    async def call(
        self,
        *,
        role: str,
        messages: tuple[dict[str, str], ...],
        operation_id: str,
        **options: object,
    ) -> ModelCallResult:
        binding = self.spec.model_role_bindings.get(role)
        if binding is None:
            raise KeyError(f"model role '{role}' is not bound")
        client = self._clients.get(role)
        if client is None:
            raise KeyError(f"model role '{role}' has no client")

        for operation_attempt in range(1, self._retry_policy.max_attempts + 1):
            if self._cancellation is not None:
                self._cancellation.raise_if_cancelled()
            call_options, reserved_tokens, prompt_tokens = _bounded_model_options(
                messages,
                options,
                self._budget,
                default_max_tokens=_client_default_max_tokens(client),
            )
            if self._budget is not None:
                self._budget.reserve(
                    model_calls=1,
                    tokens=reserved_tokens,
                )
            model_call_id = str(uuid4())
            redacted_request = self._redaction.redact_mapping(
                {
                    "messages": messages,
                    "options": call_options,
                }
            )
            call = ModelCallRecord(
                model_call_id=model_call_id,
                correlation=self.correlation,
                role=role,
                operation_id=operation_id,
                operation_attempt=operation_attempt,
                request=redacted_request,
                request_hash=_canonical_hash(redacted_request),
            )
            self._persistence.record_model_call_requested(call)
            self._events.emit(
                "model.requested",
                operation_id=operation_id,
                attempt=operation_attempt,
                payload={
                    "model_call_id": model_call_id,
                    "role": role,
                    "operation_attempt": operation_attempt,
                },
            )
            context = ProviderRequestContext(
                correlation=self.correlation,
                model_call_id=model_call_id,
                role=role,
                operation_id=operation_id,
                technique_id=self.spec.technique_id,
                technique_version=self.spec.technique_version,
                implementation_id=self.spec.implementation_id,
                transport=binding.transport,
            )
            try:
                provider = await asyncio.to_thread(
                    client.chat_correlated_once,
                    [dict(message) for message in messages],
                    request_context=context,
                    **call_options,
                )
            except Exception as error:
                failure = classify_provider_error(error)
                redacted_error = self._redaction.redact_text(str(error))
                self._persistence.fail_model_call(
                    model_call_id,
                    category=failure.category,
                    error=redacted_error,
                    indeterminate=failure.indeterminate,
                )
                self._events.emit(
                    "model.failed",
                    operation_id=operation_id,
                    attempt=operation_attempt,
                    payload={
                        "model_call_id": model_call_id,
                        "category": failure.category,
                        "indeterminate": failure.indeterminate,
                    },
                )
                if self._budget is not None and not failure.indeterminate:
                    self._budget.reconcile(
                        reserved_tokens=reserved_tokens,
                        actual_tokens=0,
                    )
                if not self._retry_policy.allows(failure, operation_attempt):
                    raise
                continue

            persisted_response = self._redaction.redact_text(provider.text)
            persisted_usage = self._redaction.redact_mapping(provider.usage)
            self._persistence.complete_model_call(
                model_call_id,
                response=persisted_response,
                usage=persisted_usage,
                provider_response_id=provider.provider_response_id,
                gateway_call_id=provider.gateway_call_id,
                gateway_attempts=provider.gateway_attempts,
            )
            self._events.emit(
                "model.completed",
                operation_id=operation_id,
                attempt=operation_attempt,
                payload={
                    "model_call_id": model_call_id,
                    "gateway_call_id": provider.gateway_call_id,
                },
            )
            if self._budget is not None:
                actual_tokens = _usage_number(
                    provider.usage.get("total_tokens"),
                    default=reserved_tokens,
                )
                self._budget.reconcile(
                    reserved_tokens=reserved_tokens,
                    actual_tokens=max(prompt_tokens, actual_tokens),
                )
            return ModelCallResult(
                model_call_id=model_call_id,
                correlation=self.correlation,
                role=role,
                operation_id=operation_id,
                operation_attempt=operation_attempt,
                text=provider.text,
                usage=provider.usage,
                provider_response_id=provider.provider_response_id,
                gateway_call_id=provider.gateway_call_id,
                gateway_attempts=provider.gateway_attempts,
            )
        raise AssertionError("retry loop exhausted without returning or raising")


def classify_provider_error(error: Exception) -> ProviderFailure:
    if isinstance(error, TimeoutError):
        return ProviderFailure("timeout", retryable=True, indeterminate=True)
    # This classifier is called only after the provider boundary was dispatched.
    # No current transport contract proves safe non-completion, so every failure
    # at this point is both potentially replayable and indeterminate.
    return ProviderFailure("provider", retryable=True, indeterminate=True)


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_estimate(options: Mapping[str, object]) -> int:
    value = options.get("max_tokens", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _bounded_model_options(
    messages: tuple[dict[str, str], ...],
    options: Mapping[str, object],
    budget: BudgetService | None,
    *,
    default_max_tokens: int,
) -> tuple[dict[str, object], int, int]:
    bounded = dict(options)
    if budget is None or budget.limits.max_tokens is None:
        reserved = _token_estimate(bounded)
        return bounded, reserved, 0

    remaining = budget.remaining("tokens")
    if remaining is None:  # pragma: no cover - guarded by the finite limit above
        reserved = _token_estimate(bounded)
        return bounded, reserved, 0
    remaining_tokens = int(remaining)
    prompt_ceiling = _message_token_ceiling(messages)
    completion_capacity = remaining_tokens - prompt_ceiling
    requested_completion = _positive_token_option(
        bounded.get("max_tokens", default_max_tokens)
    )
    if completion_capacity < 1:
        raise BudgetExhausted(
            "tokens",
            limit=budget.limits.max_tokens,
            requested=budget.usage.tokens + prompt_ceiling + 1,
        )
    completion_ceiling = min(
        requested_completion if requested_completion is not None else completion_capacity,
        completion_capacity,
    )
    bounded["max_tokens"] = completion_ceiling
    return bounded, prompt_ceiling + completion_ceiling, prompt_ceiling


def _client_default_max_tokens(client: object) -> int:
    value = getattr(client, "max_tokens", 1024)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1024
    return value


def _positive_token_option(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_tokens must be a positive integer")
    return value


def _message_token_ceiling(messages: tuple[dict[str, str], ...]) -> int:
    """Return a provider-neutral conservative prompt-token reservation.

    OpenAI-compatible providers do not expose a common pre-dispatch tokenizer.
    Reserve four framing tokens per message plus one token per four UTF-8 bytes;
    provider-reported overages are still rejected during reconciliation.
    """

    return sum(
        4
        + (
            sum(len(str(value).encode("utf-8")) for value in message.values())
            + 3
        )
        // 4
        for message in messages
    )


def _usage_number(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return default
        return parsed if parsed >= 0 else default
    return default


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value
