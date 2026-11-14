"""Per-attempt composition of controlled modular runtime services."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from app.persistence.technique_runtime import TechniqueRuntimePersistence
from app.techniques.contracts.definition import TechniqueCapability, TechniqueDefinition
from app.techniques.contracts.execution import ResolvedRunSpec, TechniqueRequest
from app.techniques.core.artifacts import ArtifactService
from app.techniques.core.budgets import BudgetLimits, BudgetService
from app.techniques.core.cancellation import CancellationToken
from app.techniques.core.checkpoints import CheckpointService
from app.techniques.core.converters import ConverterService
from app.techniques.core.detections import DetectionService
from app.techniques.core.events import EventJournal, TechniqueEventIdentity
from app.techniques.core.histories import HistoryService
from app.techniques.core.inputs import InputService
from app.techniques.core.judging import JudgeService
from app.techniques.core.models import CallRetryPolicy, CorrelatedChatClient, ModelService
from app.techniques.core.preflight import PreflightService, validate_package_provenance
from app.techniques.core.redaction import RedactionService
from app.techniques.core.seeds import SeedService
from app.techniques.core.targets import CanonicalTargetAdapter, TargetService
from app.techniques.core.templates import TemplateService
from app.techniques.core.tools import ToolService
from app.techniques.judge_requests import CanonicalJudgeRequestComposer


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    spec: ResolvedRunSpec
    loaded_package: object | None
    inputs: InputService
    models: ModelService
    targets: TargetService
    tools: ToolService
    converters: ConverterService
    judges: JudgeService
    detections: DetectionService
    seeds: SeedService
    templates: TemplateService
    preflight: PreflightService
    redaction: RedactionService
    histories: HistoryService
    artifacts: ArtifactService
    events: EventJournal
    budget: BudgetService
    checkpoints: CheckpointService
    cancellation: CancellationToken
    scope_id: str | None = None

    def fork(self, scope_id: str) -> ExecutionContext:
        """Fork branch-local mutable history heads over shared run services."""

        if type(scope_id) is not str or not scope_id:
            raise ValueError("context scope_id must be a non-empty built-in string")
        return replace(
            self,
            histories=self.histories.fork_context(scope_id),
            scope_id=scope_id,
        )


class ExecutionContextFactory:
    """Shares immutable registries but creates all mutable run state afresh."""

    def __init__(
        self,
        *,
        persistence: TechniqueRuntimePersistence,
        model_clients: Mapping[str, CorrelatedChatClient],
        target_adapter: CanonicalTargetAdapter,
        tool_registry: Mapping[str, Callable] | None = None,
        converter_registry: Mapping[str, Callable] | None = None,
        detector_registry: Mapping[str, Callable] | None = None,
        template_assets: Mapping[str, str] | None = None,
        supplied_seeds: tuple[str, ...] = (),
        packaged_seed_assets: Mapping[str, tuple[str, ...]] | None = None,
        sql_seed_libraries: Mapping[str, tuple[str, ...]] | None = None,
        runtime_seed_resolvers: Mapping[str, Callable] | None = None,
        budget_limits: BudgetLimits = BudgetLimits(),
        max_artifact_size_bytes: int = 10 * 1024 * 1024,
        allowed_artifact_media_types: frozenset[str] = frozenset(
            {"application/json", "application/octet-stream", "text/plain"}
        ),
        supported_capabilities: frozenset[TechniqueCapability] = frozenset(
            {
                TechniqueCapability.SINGLE_SHOT,
                TechniqueCapability.ITERATIVE,
                TechniqueCapability.MULTIPLE_HISTORIES,
                TechniqueCapability.MULTIPLE_JUDGES,
                TechniqueCapability.BACKTRACKING,
                TechniqueCapability.PARALLEL_BRANCHES,
                TechniqueCapability.POPULATION_SEARCH,
                TechniqueCapability.RESPONSE_TRANSFORMATION,
                TechniqueCapability.STRUCTURED_OUTPUT,
                TechniqueCapability.DOCUMENTS,
                TechniqueCapability.WORKSPACE_FILES,
                TechniqueCapability.TOOLS,
                TechniqueCapability.MULTIMODAL,
                TechniqueCapability.CONVERTERS,
            }
        ),
        redaction: RedactionService | None = None,
    ) -> None:
        self._persistence = persistence
        self._model_clients = dict(model_clients)
        self._target_adapter = target_adapter
        self._tool_registry = dict(tool_registry or {})
        self._converter_registry = dict(converter_registry or {})
        self._detector_registry = dict(detector_registry or {})
        self._template_assets = dict(template_assets or {})
        self._supplied_seeds = supplied_seeds
        self._packaged_seed_assets = dict(packaged_seed_assets or {})
        self._sql_seed_libraries = dict(sql_seed_libraries or {})
        self._runtime_seed_resolvers = dict(runtime_seed_resolvers or {})
        self._budget_limits = budget_limits
        self._max_artifact_size_bytes = max_artifact_size_bytes
        self._allowed_artifact_media_types = allowed_artifact_media_types
        self._supported_capabilities = supported_capabilities
        self._redaction = redaction

    def create(
        self,
        *,
        request: TechniqueRequest,
        definition: TechniqueDefinition,
        loaded_package: object | None = None,
        run_preflight: bool = True,
        target_surface: str | None = None,
    ) -> ExecutionContext:
        validate_package_provenance(
            spec=request.spec,
            definition=definition,
            loaded_package=loaded_package,
        )
        correlation = request.correlation
        spec = request.spec
        redaction = self._redaction or RedactionService()
        events = EventJournal(
            correlation=correlation,
            persistence=self._persistence,
            technique=TechniqueEventIdentity(
                spec.technique_id,
                spec.technique_version,
                spec.implementation_id,
            ),
            redaction=redaction,
        )
        cancellation = CancellationToken()
        budget = BudgetService(_resolved_budget_limits(self._budget_limits, spec))
        histories = HistoryService(events=events)
        models = ModelService(
            correlation=correlation,
            spec=spec,
            clients=self._model_clients,
            persistence=self._persistence,
            events=events,
            redaction=redaction,
            retry_policy=_resolved_retry_policy(spec),
            budget=budget,
            cancellation=cancellation,
        )
        artifacts = ArtifactService(
            correlation=correlation,
            persistence=self._persistence,
            events=events,
            redaction=redaction,
            max_size_bytes=self._max_artifact_size_bytes,
            allowed_media_types=self._allowed_artifact_media_types,
            budget=budget,
        )
        inputs = InputService(
            definitions=definition.inputs,
            artifact_exists=artifacts.contains,
            artifact_resolver=artifacts.get,
        )
        inputs.materialize(request.inputs)
        tools = ToolService(
            definitions=definition.tools,
            registry=self._tool_registry,
            events=events,
            budget=budget,
            cancellation=cancellation,
            redaction=redaction,
        )
        converters = ConverterService(
            definitions=definition.converters,
            registry=self._converter_registry,
            events=events,
            cancellation=cancellation,
            redaction=redaction,
        )
        detections = DetectionService(
            definitions=definition.detectors,
            registry=self._detector_registry,
            events=events,
            redaction=redaction,
        )
        seeds = SeedService(
            supplied=self._supplied_seeds,
            packaged_assets=self._packaged_seed_assets,
            sql_libraries=self._sql_seed_libraries,
            runtime_resolvers=self._runtime_seed_resolvers,
        )
        templates = TemplateService(assets=self._template_assets)
        preflight = PreflightService(
            model_roles=definition.model_roles,
            declared_detector_ids=frozenset(item.id for item in definition.detectors),
            declared_tool_ids=frozenset(item.id for item in definition.tools),
            declared_converter_ids=frozenset(item.id for item in definition.converters),
            available_detector_ids=frozenset(self._detector_registry),
            available_tool_ids=frozenset(self._tool_registry),
            available_converter_ids=frozenset(self._converter_registry),
            available_model_roles=frozenset(self._model_clients),
            supported_capabilities=self._supported_capabilities,
        )
        if run_preflight:
            preflight.validate(
                model_role_bindings=spec.model_role_bindings,
                implementation_kind=spec.implementation_kind,
                capabilities=definition.capabilities,
                resolved_parameters=spec.resolved_parameters,
                parameter_schema=definition.parameters_schema,
                required_assets=definition.assets,
                available_assets=frozenset(
                    getattr(loaded_package, "asset_hashes", spec.asset_hashes)
                ),
                target_surface=target_surface,
                supported_target_surfaces=definition.target_surfaces,
            )
            seeds.resolve(definition.seed)
        judges = JudgeService(
            definitions=definition.judges,
            models=models,
            events=events,
            parser=json.loads,
            request_composer=CanonicalJudgeRequestComposer(self._model_clients),
            budget=budget,
            cancellation=cancellation,
            redaction=redaction,
        )
        checkpoints = CheckpointService(
            correlation=correlation,
            spec_hash=_resolved_spec_hash(spec),
            state_schema_version=definition.state_schema_version,
            persistence=self._persistence,
            events=events,
        )
        targets = TargetService(
            correlation=correlation,
            adapter=self._target_adapter,
            events=events,
            budget=budget,
            cancellation=cancellation,
            redaction=redaction,
        )
        return ExecutionContext(
            spec=spec,
            loaded_package=loaded_package,
            inputs=inputs,
            models=models,
            targets=targets,
            tools=tools,
            converters=converters,
            judges=judges,
            detections=detections,
            seeds=seeds,
            templates=templates,
            preflight=preflight,
            redaction=redaction,
            histories=histories,
            artifacts=artifacts,
            events=events,
            budget=budget,
            checkpoints=checkpoints,
            cancellation=cancellation,
        )


def _resolved_spec_hash(spec: ResolvedRunSpec) -> str:
    payload = {
        "technique_id": spec.technique_id,
        "technique_version": spec.technique_version,
        "implementation_id": spec.implementation_id,
        "implementation_kind": spec.implementation_kind.value,
        "contract_version": spec.contract_version,
        "manifest_hash": spec.manifest_hash,
        "asset_hashes": spec.asset_hashes,
        "resolved_parameters": spec.resolved_parameters,
        "model_role_bindings": {
            role: {
                "logical_role": binding.logical_role,
                "profile_id": binding.profile_id,
                "provider": binding.provider,
                "model": binding.model,
                "transport": binding.transport.value,
            }
            for role, binding in spec.model_role_bindings.items()
        },
        "judge_policies": spec.judge_policies,
        "effective_limits": spec.effective_limits,
    }
    encoded = json.dumps(
        _plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_budget_limits(
    system_limits: BudgetLimits,
    spec: ResolvedRunSpec,
) -> BudgetLimits:
    values: dict[str, int | float | None] = {}
    for field_name in BudgetLimits.__dataclass_fields__:
        system_value = getattr(system_limits, field_name)
        requested = spec.effective_limits.get(field_name)
        if field_name in spec.effective_limits:
            if field_name == "max_cost":
                if type(requested) not in (int, float) or not math.isfinite(requested):
                    raise TypeError("max_cost must be a finite number")
            elif type(requested) is not int:
                raise TypeError(f"{field_name} must be a built-in integer")
        if system_value is None:
            values[field_name] = requested
        elif requested is None:
            values[field_name] = system_value
        else:
            values[field_name] = min(system_value, requested)
    return BudgetLimits(**values)


def _resolved_retry_policy(spec: ResolvedRunSpec) -> CallRetryPolicy:
    raw_attempts = spec.effective_limits.get("max_operation_attempts", 1)
    max_attempts = (
        raw_attempts
        if isinstance(raw_attempts, int)
        and not isinstance(raw_attempts, bool)
        and raw_attempts >= 1
        else 1
    )
    return CallRetryPolicy(
        max_attempts=max_attempts,
        retry_indeterminate=spec.effective_limits.get("retry_indeterminate") is True,
    )


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value
