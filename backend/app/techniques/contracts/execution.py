"""Immutable run specifications and executor boundary for modular techniques."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

from app.techniques.contracts.definition import ImplementationKind

if TYPE_CHECKING:
    from app.techniques.core.context import ExecutionContext


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("JSON contract values must use finite numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON contract mappings must use string keys")
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive invariant
        raise TypeError("expected a mapping")
    return frozen


def _freeze_string_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise TypeError("expected a string-to-string mapping")
    return MappingProxyType(dict(value))


class ModelTransport(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    LITELLM_PROXY = "litellm_proxy"


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ResultDisposition(StrEnum):
    TESTED = "tested"
    INCONCLUSIVE = "inconclusive"
    NOT_TESTED = "not_tested"


class FailureCategory(StrEnum):
    PROVIDER = "provider"
    TIMEOUT = "timeout"
    JUDGE_PARSE = "judge_parse"
    VALIDATION = "validation"
    CANCELLED = "cancelled"
    BUDGET = "budget"
    WORKER_LOST = "worker_lost"
    PERSISTENCE = "persistence"
    INTERNAL = "internal"


class AttackOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    NOT_SUCCEEDED = "not_succeeded"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class CorrelationIds:
    run_id: str
    attempt_id: str
    experiment_id: str | None = None
    cell_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "attempt_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in ("experiment_id", "cell_id"):
            value = getattr(self, name)
            if value is not None and not value:
                raise ValueError(f"{name} must be absent or non-empty")


@dataclass(frozen=True, slots=True)
class ModelRoleBinding:
    logical_role: str
    profile_id: str
    provider: str
    model: str
    transport: ModelTransport

    @classmethod
    def from_profile(
        cls,
        *,
        logical_role: str,
        profile_id: str,
        provider: str,
        model: str,
    ) -> ModelRoleBinding:
        normalized_provider = provider.strip().lower()
        transport = (
            ModelTransport.LITELLM_PROXY
            if normalized_provider in {"litellm", "litellm_proxy"}
            else ModelTransport.OPENAI_COMPATIBLE
        )
        return cls(
            logical_role=logical_role,
            profile_id=profile_id,
            provider=normalized_provider,
            model=model,
            transport=transport,
        )

    def __post_init__(self) -> None:
        for name in ("logical_role", "profile_id", "provider", "model"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class ResolvedRunSpec:
    technique_id: str
    technique_version: str
    implementation_id: str
    implementation_kind: ImplementationKind
    contract_version: str
    manifest_hash: str
    asset_hashes: Mapping[str, str] = field(default_factory=dict)
    resolved_parameters: Mapping[str, object] = field(default_factory=_empty_mapping)
    model_role_bindings: Mapping[str, ModelRoleBinding] = field(default_factory=dict)
    judge_policies: Mapping[str, object] = field(default_factory=_empty_mapping)
    effective_limits: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_hashes", _freeze_string_mapping(self.asset_hashes))
        for name in ("resolved_parameters", "judge_policies", "effective_limits"):
            object.__setattr__(
                self,
                name,
                _freeze_json_mapping(getattr(self, name)),
            )
        role_bindings = dict(self.model_role_bindings)
        if not all(
            isinstance(key, str)
            and isinstance(binding, ModelRoleBinding)
            and key == binding.logical_role
            for key, binding in role_bindings.items()
        ):
            raise ValueError(
                "model role binding keys must match ModelRoleBinding.logical_role"
            )
        object.__setattr__(
            self,
            "model_role_bindings",
            MappingProxyType(role_bindings),
        )


@dataclass(frozen=True, slots=True)
class TechniqueRequest:
    correlation: CorrelationIds
    spec: ResolvedRunSpec
    inputs: Mapping[str, object] = field(default_factory=_empty_mapping)
    artifact_refs: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _freeze_json_mapping(self.inputs))
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TechniqueOutcome:
    execution_status: ExecutionStatus
    result_disposition: ResultDisposition
    attack_outcome: AttackOutcome
    final_response_ref: str | None = None
    official_judge_ref: str | None = None
    stopping_reason: str | None = None
    auxiliary_judge_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    failure_category: FailureCategory | None = None
    outputs: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "auxiliary_judge_refs",
            tuple(self.auxiliary_judge_refs),
        )
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        object.__setattr__(self, "outputs", _freeze_json_mapping(self.outputs))


@runtime_checkable
class TechniqueExecutor(Protocol):
    async def execute(
        self,
        request: TechniqueRequest,
        context: "ExecutionContext",
    ) -> TechniqueOutcome: ...
