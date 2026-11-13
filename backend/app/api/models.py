from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Mapping

from app.techniques.contracts.execution import (
    ExecutionStatus,
    FailureCategory,
    ResultDisposition,
)


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive invariant
        raise TypeError("expected a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class ExperimentTechniqueConfig:
    technique_config_id: str
    technique_key: str
    technique_id: str
    technique_version: str
    implementation_id: str
    implementation_kind: str
    contract_version: str
    manifest_hash: str
    asset_hashes: Mapping[str, object] = field(default_factory=_empty_mapping)
    preset: str | None = None
    parameters: Mapping[str, object] = field(default_factory=_empty_mapping)
    model_roles: Mapping[str, object] = field(default_factory=_empty_mapping)
    judge_policies: Mapping[str, object] = field(default_factory=_empty_mapping)
    effective_limits: Mapping[str, object] = field(default_factory=_empty_mapping)
    runtime_config: Mapping[str, object] = field(default_factory=_empty_mapping)
    resolved_spec: Mapping[str, object] = field(default_factory=_empty_mapping)
    spec_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "technique_config_id",
            "technique_key",
            "technique_id",
            "technique_version",
            "implementation_id",
            "implementation_kind",
            "contract_version",
            "manifest_hash",
            "spec_hash",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in (
            "asset_hashes",
            "parameters",
            "model_roles",
            "judge_policies",
            "effective_limits",
            "runtime_config",
            "resolved_spec",
        ):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PlannedCell:
    cell_id: str
    experiment_id: str
    technique_config_id: str
    technique_key: str
    prompt_id: str
    prompt_text: str
    prompt_category: str
    repeat_index: int
    gold_answer: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "experiment_id",
            "technique_config_id",
            "technique_key",
            "prompt_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if type(self.repeat_index) is not int or self.repeat_index < 1:
            raise ValueError("repeat_index must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    experiment_id: str
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    technique_configs: tuple[ExperimentTechniqueConfig, ...]
    cells: tuple[PlannedCell, ...]
    repeat_count: int
    plan_hash: str

    def __post_init__(self) -> None:
        for name in ("experiment_id", "dataset_id", "dataset_version", "dataset_hash", "plan_hash"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if type(self.repeat_count) is not int or self.repeat_count < 1:
            raise ValueError("repeat_count must be a positive integer")
        object.__setattr__(self, "technique_configs", tuple(self.technique_configs))
        object.__setattr__(self, "cells", tuple(self.cells))


@dataclass(slots=True)
class RunSummary:
    run_id: str
    technique_key: str
    target_kind: str
    status: str
    score: int | float
    asr: int
    matched_detections: int
    detections: tuple[str, ...]
    highest_severity: str
    final_output: str = ""
    conversation: list[dict] | None = None
    error: str = ""
    # Internal canonical terminal evidence. Compatibility serialization remains
    # intentionally unchanged until its persistence schema is migrated.
    execution_status: ExecutionStatus | None = None
    result_disposition: ResultDisposition | None = None
    failure_category: FailureCategory | None = None
    experiment_id: str | None = None
    cell_id: str | None = None
    attempt_id: str | None = None


@dataclass(slots=True)
class EvaluationState:
    eval_id: str
    status: str
    mode: str
    target_config: dict[str, str]
    technique_keys: list[str]
    completed_keys: list[str] = field(default_factory=list)
    current_key: str | None = None
    results: list[dict[str, object]] = field(default_factory=list)
    report: dict[str, object] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(slots=True)
class ExperimentState:
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    status: str
    mode: str
    dataset: dict
    technique_keys: list[str]
    target_config: dict[str, str]
    total_cells: int
    repeat_count: int = 1
    max_rounds: int = 5
    experiment_group: str = ""
    completed_cells: int = 0
    current_technique: str | None = None
    current_prompt_id: str | None = None
    current_repeat_index: int | None = None
    cell_results: list[dict] = field(default_factory=list)
    report: dict | None = None
    error: str | None = None
    stop_requested: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    plan: ExperimentPlan | None = None
    _execution_generation: int = field(default=0, repr=False)
    _execution_owner_id: str = field(default="", repr=False)
    _restored_pre_attempt_cell_ids: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _restored_pre_attempt_row_hashes: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
