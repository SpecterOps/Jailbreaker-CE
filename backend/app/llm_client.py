"""Shared LLM chat client used by both target and attacker adapters.

Includes both synchronous (``ChatClient``) and asynchronous
(``AsyncChatClient``) variants. The async client wraps the sync one
via ``asyncio.to_thread`` — no extra HTTP dependencies required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Mapping, Sequence

log = logging.getLogger("llm_client")


_RESERVED_EXTRA_BODY_KEYS = {
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "response_format",
}
_ALLOWED_API_BASE_URL_SCHEMES = {"http", "https"}


def prepare_judge_messages(
    judge_client: object,
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Prepare one message list for both an outbound call and its run trace."""

    preparer = getattr(judge_client, "prepare_messages", None)
    if callable(preparer):
        return preparer([dict(message) for message in messages])
    return [dict(message) for message in messages]


def _safe_extra_body(extra_body: dict[str, object] | None) -> dict[str, object]:
    if not extra_body:
        return {}
    return {
        key: value
        for key, value in extra_body.items()
        if key not in _RESERVED_EXTRA_BODY_KEYS
    }


def _validate_api_base_url(api_base_url: str) -> str:
    normalized = api_base_url.strip()
    try:
        parsed = urllib.parse.urlsplit(normalized)
    except ValueError as exc:
        raise ValueError("`api_base_url` must be an absolute http or https URL.") from exc
    if parsed.scheme not in _ALLOWED_API_BASE_URL_SCHEMES or not parsed.netloc:
        raise ValueError("`api_base_url` must be an absolute http or https URL.")
    return normalized


def _make_ssl_context(url: str) -> ssl.SSLContext | None:
    skip_keywords = ("bedrock", "litellm", "localhost", "127.0.0.1", "internal")
    if any(kw in url.lower() for kw in skip_keywords):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None


@dataclass(slots=True, frozen=True)
class ChatCompletionResult:
    content: str
    usage: dict[str, str]
    raw_response: dict
    choice: dict
    message: dict
    finish_reason: str
    refusal: str | None
    tool_calls: tuple[dict, ...] = ()


@dataclass(slots=True)
class ChatClient:
    """Stateless OpenAI-compatible chat completions client."""

    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    extra_headers: dict[str, str] = field(default_factory=dict)
    system_prompt: str = ""

    def __post_init__(self) -> None:
        self.api_base_url = _validate_api_base_url(self.api_base_url)

    def prepare_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Compose the configured prompt without mutating or duplicating input."""

        configured_prompt = str(self.system_prompt or "").strip()
        prepared = [dict(message) for message in messages]
        if not configured_prompt:
            return prepared
        if prepared and prepared[0].get("role") == "system":
            existing = str(prepared[0].get("content", "") or "").strip()
            if existing == configured_prompt or existing.startswith(f"{configured_prompt}\n\n"):
                return prepared
            prepared[0]["content"] = (
                f"{configured_prompt}\n\n{existing}" if existing else configured_prompt
            )
            return prepared
        prepared.insert(0, {"role": "system", "content": configured_prompt})
        return prepared

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Send messages and return (response_text, usage_dict).

        response_format: optional OpenAI/LM Studio structured-output spec, e.g.
            {"type": "json_schema", "json_schema": {"name": "...", "strict": true,
             "schema": {...}}}
        When provided, the model is constrained to emit JSON matching the schema.
        """
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
        """Send messages and preserve the raw chat-completion response."""
        url = f"{self.api_base_url.rstrip('/')}/chat/completions"
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # OpenAI's newer models (gpt-4o, gpt-5.*) require max_completion_tokens
        # instead of max_tokens. Use max_completion_tokens when hitting the
        # actual OpenAI API; keep max_tokens for all other providers.
        is_openai_api = "api.openai.com" in self.api_base_url
        token_key = "max_completion_tokens" if is_openai_api else "max_tokens"

        payload: dict = {
            "model": self.model,
            "messages": self.prepare_messages(messages),
            "temperature": temp,
            token_key: tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(_safe_extra_body(extra_body))

        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        ssl_ctx = _make_ssl_context(url)

        log.info("POST %s model=%s", url, self.model)

        try:
            with urllib.request.urlopen(req, timeout=90, context=ssl_ctx) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            log.error("HTTP %d from %s: %s", exc.code, url, error_body[:500])
            raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc

        choice = data["choices"][0]
        message = choice.get("message", {}) or {}
        raw_content = message.get("content", "")

        # Handle the Anthropic/Bedrock content-block format where content is
        # a list of {type, text} blocks instead of a plain string.
        if isinstance(raw_content, list):
            text_parts = [
                block.get("text", "")
                for block in raw_content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = "".join(text_parts)
        elif isinstance(raw_content, str):
            text = raw_content
        else:
            text = str(raw_content) if raw_content else ""

        # Diagnostic: when content is empty, log the FULL raw response so we
        # can see what the upstream actually sent (refusal field? content
        # filter? other format?).
        refusal = message.get("refusal")
        if not text:
            finish_reason = choice.get("finish_reason", "?")
            log.warning(
                "EMPTY CONTENT — finish_reason=%s refusal=%r raw_content_type=%s",
                finish_reason, refusal, type(raw_content).__name__,
            )
            log.warning("EMPTY CONTENT — full response: %s", raw[:2000])
            # If there's a refusal field with text, use that as the response.
            if isinstance(refusal, str) and refusal.strip():
                text = refusal

        usage = {k: str(v) for k, v in data.get("usage", {}).items()}
        raw_tool_calls = message.get("tool_calls", ())
        tool_calls = (
            tuple(item for item in raw_tool_calls if isinstance(item, dict))
            if isinstance(raw_tool_calls, list)
            else ()
        )

        log.info("response model=%s tokens=%s len=%d", self.model, usage.get("total_tokens", "?"), len(text))
        log.info("response body: %s", text[:500])

        return ChatCompletionResult(
            content=text,
            usage=usage,
            raw_response=data,
            choice=choice,
            message=message,
            finish_reason=str(choice.get("finish_reason", "")),
            refusal=refusal if isinstance(refusal, str) else None,
            tool_calls=tool_calls,
        )

    def chat_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        response_format: dict | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Chat with simple retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    extra_body=extra_body,
                )
            except RuntimeError as exc:
                last_exc = exc
                if attempt < max_retries and ("429" in str(exc) or "500" in str(exc) or "502" in str(exc)):
                    wait = 2 ** attempt
                    log.warning("Retry %d/%d after %ds: %s", attempt + 1, max_retries, wait, exc)
                    time.sleep(wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]


class AsyncChatClient:
    """Async wrapper around ``ChatClient``.

    Uses ``asyncio.to_thread`` to run the synchronous HTTP call in a
    thread-pool executor. This keeps the stdlib-only approach (no aiohttp)
    while allowing concurrent LLM calls via ``asyncio.gather``.

    The concurrency limit prevents overwhelming the LLM API with too many
    parallel requests.
    """

    def __init__(self, client: ChatClient, *, max_concurrency: int = 5) -> None:
        self._client = client
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def prepare_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Expose the wrapped client's outbound messages for accurate traces."""

        return self._client.prepare_messages(messages)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, str]]:
        async with self._semaphore:
            return await asyncio.to_thread(
                self._client.chat,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )

    async def chat_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        response_format: dict | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, str]]:
        async with self._semaphore:
            return await asyncio.to_thread(
                self._client.chat_with_retry,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                response_format=response_format,
                extra_body=extra_body,
            )
