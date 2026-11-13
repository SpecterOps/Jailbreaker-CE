"""Deterministic fixtures for legacy-versus-modular parity runs."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar

from app.techniques.contracts.execution import _freeze_json_mapping
from app.techniques.core.catalog import TechniqueSelector
from app.techniques.core.models import ProviderCallResult, ProviderRequestContext
from app.techniques.core.router import ExecutionRoute, ExecutionRouter
from devtools.technique_parity.randomness import seeded_random


_T = TypeVar("_T")
_U = TypeVar("_U")


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return _freeze_json_mapping(value)


def _freeze_messages(
    messages: Iterable[Mapping[str, str]],
) -> tuple[Mapping[str, str], ...]:
    return tuple(MappingProxyType(dict(message)) for message in messages)


@dataclass(frozen=True, slots=True)
class ScriptedCall:
    """One accepted call made to a :class:`ScriptedRoleClient`."""

    role: str
    messages: tuple[Mapping[str, str], ...]
    options: Mapping[str, object]
    operation_id: str
    call_index: int


@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    """A queued provider result with optional semantic call expectations."""

    text: str
    expected_operation_id: str | None = None
    expected_messages: tuple[Mapping[str, str], ...] | None = None
    expected_options: Mapping[str, object] | None = None
    usage: Mapping[str, object] = field(default_factory=_empty_mapping)
    provider_response_id: str | None = None
    gateway_call_id: str | None = None
    gateway_attempts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("scripted response text must be a string")
        if self.expected_operation_id is not None and not self.expected_operation_id:
            raise ValueError("expected_operation_id must be absent or non-empty")
        if self.expected_messages is not None:
            object.__setattr__(
                self,
                "expected_messages",
                _freeze_messages(self.expected_messages),
            )
        if self.expected_options is not None:
            object.__setattr__(
                self,
                "expected_options",
                _freeze_mapping(self.expected_options),
            )
        object.__setattr__(self, "usage", _freeze_mapping(self.usage))
        object.__setattr__(
            self,
            "gateway_attempts",
            MappingProxyType(dict(self.gateway_attempts)),
        )

    def provider_result(self) -> ProviderCallResult:
        return ProviderCallResult(
            text=self.text,
            usage=self.usage,
            provider_response_id=self.provider_response_id,
            gateway_call_id=self.gateway_call_id,
            gateway_attempts=self.gateway_attempts,
        )


QueuedResponse = str | ScriptedResponse | ProviderCallResult


class ScriptedRoleClient:
    """A strict, ordered fake for one logical model role.

    Role and per-response expectations are checked before the queue item is
    consumed. This makes a surprising call fail at its source instead of
    shifting every later scripted response.
    """

    def __init__(
        self,
        *,
        role: str,
        responses: Iterable[QueuedResponse],
        default_max_tokens: int = 1024,
    ) -> None:
        if not isinstance(role, str) or not role:
            raise ValueError("scripted client role must be a non-empty string")
        if type(default_max_tokens) is not int or default_max_tokens < 1:
            raise ValueError("default_max_tokens must be a positive integer")
        self.role = role
        self.default_max_tokens = default_max_tokens
        self._responses = deque(responses)
        self._calls: list[ScriptedCall] = []

    @property
    def calls(self) -> tuple[ScriptedCall, ...]:
        return tuple(self._calls)

    @property
    def remaining(self) -> int:
        return len(self._responses)

    def assert_exhausted(self) -> None:
        remaining = self.remaining
        if remaining:
            label = "response" if remaining == 1 else "responses"
            raise AssertionError(f"{remaining} scripted {label} remain unconsumed for role '{self.role}'")

    def chat_correlated_once(
        self,
        messages: list[dict[str, str]],
        *,
        request_context: ProviderRequestContext,
        **options: object,
    ) -> ProviderCallResult:
        if request_context.role != self.role:
            raise AssertionError(
                f"scripted client expected role '{self.role}', got '{request_context.role}'"
            )
        if not self._responses:
            raise AssertionError(f"scripted response queue exhausted for role '{self.role}'")

        queued = self._responses[0]
        if not isinstance(queued, (str, ScriptedResponse, ProviderCallResult)):
            raise TypeError(
                "scripted responses must be strings, ScriptedResponse, or ProviderCallResult"
            )
        frozen_messages = _freeze_messages(messages)
        frozen_options = _freeze_mapping(options)
        scripted = queued if isinstance(queued, ScriptedResponse) else None
        if scripted is not None:
            if (
                scripted.expected_operation_id is not None
                and scripted.expected_operation_id != request_context.operation_id
            ):
                raise AssertionError(
                    "scripted response expected operation "
                    f"'{scripted.expected_operation_id}', got '{request_context.operation_id}'"
                )
            if scripted.expected_messages is not None and scripted.expected_messages != frozen_messages:
                raise AssertionError("scripted response received unexpected messages")
            if scripted.expected_options is not None and scripted.expected_options != frozen_options:
                raise AssertionError("scripted response received unexpected options")

        self._responses.popleft()
        self._calls.append(
            ScriptedCall(
                role=self.role,
                messages=frozen_messages,
                options=frozen_options,
                operation_id=request_context.operation_id,
                call_index=len(self._calls) + 1,
            )
        )
        if isinstance(queued, ProviderCallResult):
            return queued
        if isinstance(queued, ScriptedResponse):
            return queued.provider_result()
        if isinstance(queued, str):
            return ProviderCallResult(text=queued)
        raise AssertionError("validated scripted response type was not handled")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class ParityScenario:
    """Pinned selectors and seed shared by a pair of comparison arms."""

    technique_id: str
    legacy_implementation: str
    modular_implementation: str
    seed: int
    technique_version: str | None = None
    inputs: Mapping[str, object] = field(default_factory=_empty_mapping)
    experiment_plan: object | None = None

    def __post_init__(self) -> None:
        for name in ("technique_id", "legacy_implementation", "modular_implementation"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.seed) is not int:
            raise TypeError("parity scenario seed must be an integer")
        object.__setattr__(self, "inputs", _freeze_mapping(self.inputs))

    @property
    def legacy_selector(self) -> TechniqueSelector:
        return TechniqueSelector(
            self.technique_id,
            technique_version=self.technique_version,
            implementation_id=self.legacy_implementation,
        )

    @property
    def modular_selector(self) -> TechniqueSelector:
        return TechniqueSelector(
            self.technique_id,
            technique_version=self.technique_version,
            implementation_id=self.modular_implementation,
        )

    def resolve(self, router: ExecutionRouter) -> tuple[ExecutionRoute, ExecutionRoute]:
        return (
            router.resolve(
                self.technique_id,
                technique_version=self.technique_version,
                implementation_id=self.legacy_implementation,
            ),
            router.resolve(
                self.technique_id,
                technique_version=self.technique_version,
                implementation_id=self.modular_implementation,
            ),
        )

    def run(
        self,
        legacy_arm: Callable[[], _T],
        modular_arm: Callable[[], _U],
    ) -> tuple[_T, _U]:
        with seeded_random(self.seed):
            legacy = legacy_arm()
            random.seed(self.seed)
            modular = modular_arm()
            return legacy, modular

    async def run_async(
        self,
        legacy_arm: Callable[[], Awaitable[_T]],
        modular_arm: Callable[[], Awaitable[_U]],
    ) -> tuple[_T, _U]:
        """Run on the caller's loop; callers must exclude other global-RNG users.

        This unused fixture helper does not share the synchronous runtime lock:
        a thread lock held across awaits cannot serialize tasks on the same loop.
        """

        state = random.getstate()
        try:
            random.seed(self.seed)
            legacy = await legacy_arm()
            random.seed(self.seed)
            modular = await modular_arm()
            return legacy, modular
        finally:
            random.setstate(state)
