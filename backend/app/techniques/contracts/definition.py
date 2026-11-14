"""Immutable definition values supplied by versioned technique packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


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


def _freeze_mapping_tuple(
    values: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(_freeze_mapping(value) for value in values)


class ImplementationKind(StrEnum):
    LEGACY = "legacy"
    WORKFLOW = "workflow"
    PYTHON = "python"


class TechniqueCapability(StrEnum):
    SINGLE_SHOT = "single_shot"
    ITERATIVE = "iterative"
    MULTIPLE_HISTORIES = "multiple_histories"
    MULTIPLE_JUDGES = "multiple_judges"
    POPULATION_SEARCH = "population_search"
    PARALLEL_BRANCHES = "parallel_branches"
    BACKTRACKING = "backtracking"
    RESPONSE_TRANSFORMATION = "response_transformation"
    STRUCTURED_OUTPUT = "structured_output"
    DOCUMENTS = "documents"
    WORKSPACE_FILES = "workspace_files"
    TOOLS = "tools"
    MULTIMODAL = "multimodal"
    MEMORY = "memory"
    CONVERTERS = "converters"


class JudgeSchedule(StrEnum):
    AFTER_TARGET = "after_target"
    EVERY_ROUND = "every_round"
    FINAL_ROUND = "final_round"
    RUN_END = "run_end"
    ON_DEMAND = "on_demand"


class JudgePurpose(StrEnum):
    OFFICIAL_SCORE = "official_score"
    SUCCESS_CONTROL = "success_control"
    STOPPING_CONTROL = "stopping_control"
    BACKTRACKING_CONTROL = "backtracking_control"
    CANDIDATE_RANKING = "candidate_ranking"
    DIAGNOSTICS = "diagnostics"


class SeedStrategy(StrEnum):
    NONE = "none"
    SUPPLIED = "supplied"
    PACKAGED_ASSET = "packaged_asset"
    SQL_LIBRARY = "sql_library"
    RUNTIME = "runtime"


@dataclass(frozen=True, slots=True)
class TechniqueIdentity:
    id: str
    version: str
    name: str
    family: str


@dataclass(frozen=True, slots=True)
class ModelRoleDefinition:
    id: str
    required: bool = True
    description: str = ""
    capabilities: frozenset[str] = frozenset()
    options: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "options", _freeze_mapping(self.options))


@dataclass(frozen=True, slots=True)
class InputDefinition:
    id: str
    kind: str
    required: bool = False
    media_types: tuple[str, ...] = ()
    max_count: int | None = None
    max_size_bytes: int | None = None
    schema: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "media_types", tuple(self.media_types))
        object.__setattr__(self, "schema", _freeze_mapping(self.schema))


@dataclass(frozen=True, slots=True)
class HistoryDefinition:
    id: str
    visible_to: tuple[str, ...] = ()
    max_messages: int | None = None
    max_tokens: int | None = None
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "visible_to", tuple(self.visible_to))
        object.__setattr__(self, "extensions", _freeze_mapping(self.extensions))


@dataclass(frozen=True, slots=True)
class JudgeDefinition:
    id: str
    schedule: JudgeSchedule = JudgeSchedule.RUN_END
    purposes: frozenset[JudgePurpose] = frozenset()
    role: str | None = None
    required: bool = True
    evidence: tuple[str, ...] = ()
    threshold: float | None = None
    scale_min: float = 0
    scale_max: float = 10
    policy: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "purposes", frozenset(self.purposes))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "policy", _freeze_mapping(self.policy))


@dataclass(frozen=True, slots=True)
class DetectorDefinition:
    id: str
    inputs: tuple[str, ...] = ()
    output_schema: Mapping[str, object] = field(default_factory=_empty_mapping)
    policy: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "output_schema", _freeze_mapping(self.output_schema))
        object.__setattr__(self, "policy", _freeze_mapping(self.policy))


@dataclass(frozen=True, slots=True)
class SeedDefinition:
    strategy: SeedStrategy = SeedStrategy.NONE
    source: str | None = None
    required: bool = False
    options: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze_mapping(self.options))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    id: str
    input_schema: Mapping[str, object] = field(default_factory=_empty_mapping)
    output_schema: Mapping[str, object] = field(default_factory=_empty_mapping)
    policy: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", _freeze_mapping(self.input_schema))
        object.__setattr__(self, "output_schema", _freeze_mapping(self.output_schema))
        object.__setattr__(self, "policy", _freeze_mapping(self.policy))


@dataclass(frozen=True, slots=True)
class ConverterDefinition:
    id: str
    input_schema: Mapping[str, object] = field(default_factory=_empty_mapping)
    output_schema: Mapping[str, object] = field(default_factory=_empty_mapping)
    policy: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", _freeze_mapping(self.input_schema))
        object.__setattr__(self, "output_schema", _freeze_mapping(self.output_schema))
        object.__setattr__(self, "policy", _freeze_mapping(self.policy))


@dataclass(frozen=True, slots=True)
class ResultPolicy:
    official_judge: str | None = None
    final_response: str | None = None
    success_threshold: float | None = None
    auxiliary_judges: tuple[str, ...] = ()
    stopping: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "auxiliary_judges", tuple(self.auxiliary_judges))
        object.__setattr__(self, "stopping", _freeze_mapping(self.stopping))


@dataclass(frozen=True, slots=True)
class TechniqueDefinition:
    identity: TechniqueIdentity
    summary: str
    capabilities: frozenset[TechniqueCapability]
    contract_version: str = "1"
    api_version: str = "1"
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    target_surfaces: tuple[str, ...] = ()
    parameters_schema: Mapping[str, object] = field(default_factory=_empty_mapping)
    presets: Mapping[str, object] = field(default_factory=_empty_mapping)
    inputs: tuple[InputDefinition, ...] = ()
    model_roles: tuple[ModelRoleDefinition, ...] = ()
    histories: tuple[HistoryDefinition, ...] = ()
    judges: tuple[JudgeDefinition, ...] = ()
    detectors: tuple[DetectorDefinition, ...] = ()
    seed: SeedDefinition = field(default_factory=SeedDefinition)
    tools: tuple[ToolDefinition, ...] = ()
    converters: tuple[ConverterDefinition, ...] = ()
    result_policy: ResultPolicy = field(default_factory=ResultPolicy)
    presentation: Mapping[str, object] = field(default_factory=_empty_mapping)
    metrics: tuple[Mapping[str, object], ...] = ()
    recommended_limits: Mapping[str, object] = field(default_factory=_empty_mapping)
    assets: tuple[str, ...] = ()
    state_schema_version: int = 1
    result_schema_version: int = 1
    source: Mapping[str, object] = field(default_factory=_empty_mapping)
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        for name in (
            "tags",
            "aliases",
            "target_surfaces",
            "inputs",
            "model_roles",
            "histories",
            "judges",
            "detectors",
            "tools",
            "converters",
            "assets",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "parameters_schema",
            "presets",
            "presentation",
            "recommended_limits",
            "source",
            "extensions",
        ):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        object.__setattr__(self, "metrics", _freeze_mapping_tuple(self.metrics))


@dataclass(frozen=True, slots=True)
class TechniqueImplementation:
    id: str
    technique_id: str
    technique_version: str
    kind: ImplementationKind
    entrypoint: str
    contract_version: str = "1"
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extensions", _freeze_mapping(self.extensions))
