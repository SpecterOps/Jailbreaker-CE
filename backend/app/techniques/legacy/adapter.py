"""Temporary adapter from legacy callbacks to the modular event boundary."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
from typing import Callable, Mapping, NoReturn
from uuid import uuid4

from app.llm_client import ChatCompletionResult
from app.persistence.technique_runtime import ModelCallRecord, TechniqueRuntimePersistence
from app.techniques.contracts.execution import (
    CorrelationIds,
    ModelRoleBinding,
    TechniqueRequest,
)
from app.techniques.core.context import ExecutionContext
from app.techniques.core.events import EventJournal
from app.techniques.core.models import ProviderFailure, ProviderRequestContext
from app.techniques.core.redaction import RedactionService
from app.techniques.legacy.translation import emit_legacy_result


class LegacyTechniqueAdapter:
    def __init__(self, *, callback: Callable[[TechniqueRequest], object]) -> None:
        self._callback = callback

    async def execute(
        self,
        request: TechniqueRequest,
        context: ExecutionContext,
    ) -> object:
        context.events.emit("run.started", payload={"legacy": True})
        try:
            result = await asyncio.to_thread(self._callback, request)
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:
            context.events.emit(
                "run.failed",
                payload={
                    "legacy": True,
                    "category": type(error).__name__,
                    "error": context.redaction.redact_text(str(error)),
                },
            )
            raise
        emit_legacy_result(result, context.events, include_lifecycle=False)
        context.events.emit("run.completed", payload={"legacy": True})
        return result


class LegacyCorrelatedChatClient:
    """ChatClient-compatible wrapper that persists every legacy dispatch first."""

    def __init__(
        self,
        *,
        client: object,
        correlation: CorrelationIds,
        binding: ModelRoleBinding,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
        persistence: TechniqueRuntimePersistence,
        events: EventJournal,
        redaction: RedactionService | None = None,
    ) -> None:
        self._client = client
        self._correlation = correlation
        self._binding = binding
        self._technique_id = technique_id
        self._technique_version = technique_version
        self._implementation_id = implementation_id
        self._persistence = persistence
        self._events = events
        self._redaction = redaction or RedactionService()
        self._operation_lock = threading.Lock()
        self._operation_sequence = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)

    def prepare_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        preparer = getattr(self._client, "prepare_messages", None)
        if callable(preparer):
            return preparer(messages)
        return [dict(message) for message in messages]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, Mapping[str, object]]:
        result = self.chat_raw(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )
        return result.content, result.usage

    def chat_raw(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> ChatCompletionResult:
        try:
            return self._dispatch_raw(
                messages,
                operation_id=self._next_operation_id(),
                operation_attempt=1,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )
        except Exception as error:
            self._raise_redacted(error)

    def chat_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        response_format: dict | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, Mapping[str, object]]:
        operation_id = self._next_operation_id()
        for operation_attempt in range(1, max_retries + 2):
            try:
                result = self._dispatch_raw(
                    messages,
                    operation_id=operation_id,
                    operation_attempt=operation_attempt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    extra_body=extra_body,
                )
                return result.content, result.usage
            except Exception as error:
                failure = _classify_legacy_provider_error(error)
                if (
                    operation_attempt > max_retries
                    or not failure.retryable
                    or failure.indeterminate
                ):
                    self._raise_redacted(error)
        raise AssertionError("legacy retry loop exhausted")

    def _dispatch_raw(
        self,
        messages: list[dict[str, str]],
        *,
        operation_id: str,
        operation_attempt: int,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict | None,
        extra_body: dict[str, object] | None,
    ) -> ChatCompletionResult:
        model_call_id = str(uuid4())
        options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "extra_body": extra_body,
        }
        redacted_request = self._redaction.redact_mapping(
            {"messages": messages, "options": options}
        )
        self._persistence.record_model_call_requested(
            ModelCallRecord(
                model_call_id=model_call_id,
                correlation=self._correlation,
                role=self._binding.logical_role,
                operation_id=operation_id,
                operation_attempt=operation_attempt,
                request=redacted_request,
                request_hash=_canonical_hash(redacted_request),
            )
        )
        self._events.emit(
            "model.requested",
            operation_id=operation_id,
            attempt=operation_attempt,
            payload={
                "model_call_id": model_call_id,
                "role": self._binding.logical_role,
                "operation_attempt": operation_attempt,
                "legacy": True,
            },
        )
        context = ProviderRequestContext(
            correlation=self._correlation,
            model_call_id=model_call_id,
            role=self._binding.logical_role,
            operation_id=operation_id,
            technique_id=self._technique_id,
            technique_version=self._technique_version,
            implementation_id=self._implementation_id,
            transport=self._binding.transport,
        )
        try:
            result = self._chat_once(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
                request_context=context,
            )
        except Exception as error:
            failure = _classify_legacy_provider_error(error)
            self._persistence.fail_model_call(
                model_call_id,
                category=failure.category,
                error=self._redaction.redact_text(str(error)),
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
                    "legacy": True,
                },
            )
            raise
        headers = result.response_headers
        gateway_call_id = None
        gateway_attempts: dict[str, int] = {}
        if self._binding.transport.value == "litellm_proxy":
            gateway_call_id = headers.get("x-litellm-call-id")
            gateway_attempts = {
                "attempted_retries": _header_int(
                    headers,
                    "x-litellm-attempted-retries",
                ),
                "attempted_fallbacks": _header_int(
                    headers,
                    "x-litellm-attempted-fallbacks",
                ),
            }
        self._persistence.complete_model_call(
            model_call_id,
            response=self._redaction.redact_text(result.content),
            usage=self._redaction.redact_mapping(result.usage),
            provider_response_id=result.provider_response_id,
            gateway_call_id=gateway_call_id,
            gateway_attempts=gateway_attempts,
        )
        self._events.emit(
            "model.completed",
            operation_id=operation_id,
            attempt=operation_attempt,
            payload={
                "model_call_id": model_call_id,
                "gateway_call_id": gateway_call_id,
                "legacy": True,
            },
        )
        return result

    def _chat_once(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict | None,
        extra_body: dict[str, object] | None,
        request_context: ProviderRequestContext,
    ) -> ChatCompletionResult:
        chat_raw = getattr(self._client, "chat_raw", None)
        if callable(chat_raw) and _delegated_method_supported(self._client, "chat_raw"):
            return chat_raw(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
                _request_context=request_context,
            )

        chat_with_retry = getattr(self._client, "chat_with_retry", None)
        if not callable(chat_with_retry):
            raise TypeError("legacy model client must provide chat_raw or chat_with_retry")
        options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": 0,
            "response_format": response_format,
            "extra_body": extra_body,
        }
        parameters = inspect.signature(chat_with_retry).parameters
        accepts_options = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        supported_options = (
            options
            if accepts_options
            else {key: value for key, value in options.items() if key in parameters}
        )
        content, usage = chat_with_retry(messages, **supported_options)
        return ChatCompletionResult(
            content=content,
            usage=dict(usage),
            raw_response={},
            choice={},
            message={"content": content},
            finish_reason="",
            refusal=None,
        )

    def _next_operation_id(self) -> str:
        with self._operation_lock:
            self._operation_sequence += 1
            return f"legacy-{self._binding.logical_role}-{self._operation_sequence}"

    def _raise_redacted(self, error: Exception) -> NoReturn:
        safe_message = self._redaction.redact_text(str(error))
        if safe_message == str(error):
            raise error
        if isinstance(error, TimeoutError):
            raise TimeoutError(safe_message) from error
        raise RuntimeError(safe_message) from error


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _classify_legacy_provider_error(error: Exception) -> ProviderFailure:
    """Preserve the legacy adapter's established message-based retry policy."""

    if isinstance(error, TimeoutError):
        return ProviderFailure("timeout", retryable=True, indeterminate=True)
    message = str(error).lower()
    retryable = any(code in message for code in ("429", "500", "502", "503", "504"))
    return ProviderFailure("provider", retryable=retryable, indeterminate=False)


def _delegated_method_supported(client: object, method_name: str) -> bool:
    """Return whether a wrapper's delegated method exists all the way down.

    Compatibility wrappers such as ``JudgePolicyClient`` expose ``chat_raw``
    even when the injected legacy client only implements ``chat_with_retry``.
    Calling that facade would fail before a provider dispatch. In that case we
    deliberately use the wrapper's supported retry method with retries disabled.
    """

    wrapped = getattr(client, "_client", None)
    if wrapped is None or wrapped is client:
        return callable(getattr(client, method_name, None))
    return callable(getattr(wrapped, method_name, None)) and _delegated_method_supported(
        wrapped,
        method_name,
    )


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _header_int(headers: Mapping[str, str], key: str) -> int:
    try:
        return int(headers.get(key, "0"))
    except (TypeError, ValueError):
        return 0
