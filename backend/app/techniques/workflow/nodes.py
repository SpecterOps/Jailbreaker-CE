"""Core handlers for the bounded declarative workflow interpreter."""

from __future__ import annotations

import base64
import json
import string
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Awaitable, Callable, Mapping, Protocol

from app.techniques.contracts.definition import SeedDefinition
from app.techniques.contracts.events import ExecutionEvent
from app.techniques.core.context import ExecutionContext
from app.techniques.core.templates import TemplateService
from app.techniques.workflow.conditions import ConditionEvaluator
from app.techniques.workflow.definition import (
    StepDefinition,
    WorkflowDefinition,
    WorkflowState,
)


class WorkflowNodeError(RuntimeError):
    """A categorized ordinary node failure eligible for a typed error edge."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        indeterminate: bool = False,
    ) -> None:
        self.category = category
        self.indeterminate = indeterminate
        super().__init__(message)


class UnsupportedWorkflowNodeError(RuntimeError):
    """A validated future node reached before its runtime is implemented."""


@dataclass(frozen=True, slots=True)
class NodeResult:
    values: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    next_step_id: str | None = None
    finish_evidence: Mapping[str, object] | None = None


class NodeHandler(Protocol):
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult: ...


def operation_id(step: StepDefinition, state: WorkflowState) -> str:
    visit = state.visit_counts.get(step.id, 0)
    if state.scope_id is not None:
        return f"workflow:{state.scope_id}:{step.id}:visit:{visit}"
    return f"workflow:{step.id}:visit:{visit}"


class RenderHandler:
    def __init__(self, templates: TemplateService | None) -> None:
        self._templates = templates

    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        options = step.options
        output = _output_name(options, step.id)
        raw_fields = options.get("fields")
        if raw_fields is None:
            fields = {
                **dict(_optional_mapping_at(state.values, "inputs")),
                **dict(_optional_mapping_at(state.values, "parameters")),
            }
        else:
            fields = _reference_mapping(raw_fields, state)
        allow_unused_fields = options.get("allow_unused_fields", False)
        if not isinstance(allow_unused_fields, bool):
            raise ValueError("render allow_unused_fields must be a boolean")
        strip_trailing_newline = options.get("strip_trailing_newline", False)
        if not isinstance(strip_trailing_newline, bool):
            raise ValueError("render strip_trailing_newline must be a boolean")
        asset = options.get("asset", options.get("template_asset"))
        text = options.get("text", options.get("template"))
        template_from = options.get("template_from")
        if sum(value is not None for value in (asset, text, template_from)) != 1:
            raise ValueError(
                "render requires exactly one of asset, text, or template_from"
            )
        if asset is not None:
            asset_id = _string(asset, "render asset")
            if self._templates is None:
                raise WorkflowNodeError(
                    "render",
                    "render asset requires a verified or explicitly trusted template snapshot",
                )
            rendered = self._templates.render_asset(
                asset_id,
                fields=fields,
            )
        elif template_from is not None:
            template = resolve_reference(
                _string(template_from, "render template_from"),
                state.values,
            )
            rendered = _render_inline(
                _string(template, "resolved render template"),
                fields,
                allow_unused_fields=allow_unused_fields,
            )
        else:
            rendered = _render_inline(
                _string(text, "render text"),
                fields,
                allow_unused_fields=allow_unused_fields,
            )
        if strip_trailing_newline:
            rendered = rendered.removesuffix("\n")
        return NodeResult({output: rendered}, step.next_step_id)


class InputMaterializeHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        raw = step.options.get("inputs")
        values = _mapping_at(state.values, "inputs") if raw is None else _reference_mapping(raw, state)
        materialized = context.inputs.materialize(values)
        return NodeResult(
            {_output_name(step.options, "inputs"): materialized},
            step.next_step_id,
        )


class SeedResolveHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        definition = getattr(
            getattr(context.loaded_package, "definition", None),
            "seed",
            None,
        )
        if not isinstance(definition, SeedDefinition):
            raise ValueError("seed resolution requires the pinned package definition")
        seeds = tuple(
            {
                "value": record.value,
                "strategy": record.strategy.value,
                "source": record.source,
                "index": record.index,
            }
            for record in context.seeds.resolve(definition)
        )
        return NodeResult(
            {_output_name(step.options, "seeds"): seeds},
            step.next_step_id,
        )


class ModelCallHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        role = _string(step.options.get("role"), "model role")
        messages = _messages(step.options.get("messages"), state)
        raw_options = step.options.get("call_options", {})
        call_options = _mapping(raw_options, "model call_options")
        result = await context.models.call(
            role=role,
            messages=messages,
            operation_id=operation_id(step, state),
            **dict(call_options),
        )
        value = {
            "model_call_id": result.model_call_id,
            "role": result.role,
            "operation_id": result.operation_id,
            "operation_attempt": result.operation_attempt,
            "text": result.text,
            "usage": result.usage,
            "provider_response_id": result.provider_response_id,
            "gateway_call_id": result.gateway_call_id,
            "gateway_attempts": result.gateway_attempts,
        }
        return NodeResult(
            {_output_name(step.options, step.id): value},
            step.next_step_id,
        )


class TargetCallHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        _preflight_target_history_append(step, context)
        messages, snapshot_id = _target_messages(step, state, context)
        artifact_refs = _string_refs(step.options.get("artifact_refs", ()), state)
        target_options = _mapping(step.options.get("call_options", {}), "target call_options")
        round_number = step.options.get("round")
        if round_number is not None and (
            type(round_number) is not int or round_number < 1
        ):
            raise ValueError("target call round must be a positive integer")
        result = await context.targets.execute(
            messages=messages,
            operation_id=operation_id(step, state),
            history_snapshot_id=snapshot_id,
            artifact_refs=artifact_refs,
            options=target_options,
            round=round_number,
        )
        completed_event = _latest_operation_event(
            context,
            "target.completed",
            operation_id(step, state),
        )
        _append_target_exchange(step, state, context, messages, result.text)
        value = {
            "target_call_id": result.target_call_id,
            "operation_id": result.operation_id,
            "text": result.text,
            "response_kind": "real",
            "history_snapshot_id": result.history_snapshot_id,
            "metadata": result.metadata,
            "event_id": completed_event.event_id,
        }
        if result.artifact_refs:
            value["artifact_refs"] = result.artifact_refs
        if result.tool_calls:
            value["tool_calls"] = result.tool_calls
        return NodeResult(
            {_output_name(step.options, "target_response"): value},
            step.next_step_id,
        )


class ToolCallHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        tool_id = _string(step.options.get("tool"), "tool ID")
        arguments = _reference_mapping(step.options.get("arguments", {}), state)
        result = await context.tools.invoke(tool_id, arguments=arguments)
        value = {
            "tool_call_id": result.tool_call_id,
            "tool_id": result.tool_id,
            "output": result.output,
        }
        return NodeResult(
            {_output_name(step.options, step.id): value},
            step.next_step_id,
        )


class ConvertHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        converter_id = _string(step.options.get("converter"), "converter ID")
        source_ref = _string(
            step.options.get("artifact_ref", step.options.get("artifact_from")),
            "converter artifact_ref",
        )
        artifact_id = _referenced_string(source_ref, state, "converter artifact")
        result = context.converters.convert(converter_id, artifact_id=artifact_id)
        value = {
            "conversion_id": result.conversion_id,
            "converter_id": result.converter_id,
            "source_artifact_id": result.source_artifact_id,
            "output": result.output,
        }
        return NodeResult(
            {_output_name(step.options, step.id): value},
            step.next_step_id,
        )


class JudgeHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        definition_id = _string(
            step.options.get("definition", step.options.get("definition_id")),
            "judge definition",
        )
        evidence = _reference_mapping(step.options.get("evidence", {}), state)
        policy_version = step.options.get("policy_version")
        if policy_version is not None:
            policy_version = _string(policy_version, "judge policy_version")
        provenance = {
            **_reference_mapping(step.options.get("provenance", {}), state),
            "workflow_step_id": step.id,
        }
        result = await context.judges.evaluate(
            definition_id=definition_id,
            evidence=evidence,
            operation_id=operation_id(step, state),
            provenance=provenance,
            policy_version=policy_version,
        )
        judge_event = _latest_result_event(
            context,
            result.result_id,
            ("judge.completed", "judge.failed"),
        )
        value = {
            "result_id": result.result_id,
            "definition_id": result.definition_id,
            "status": result.status.value,
            "score": result.score,
            "rationale": result.rationale,
            "parse_status": result.parse_status,
            "raw_output_ref": result.raw_output_ref,
            "event_id": judge_event.event_id,
        }
        if result.subject_refs:
            value["subject_refs"] = result.subject_refs
        judges = dict(_optional_mapping_at(state.values, "judges"))
        judges[definition_id] = value
        values: dict[str, object] = {"judges": judges}
        output = step.options.get("output")
        if output is not None:
            output_name = _string(output, "judge output")
            if output_name == "judges":
                raise ValueError("judge output must not replace the judges namespace")
            values[output_name] = _plain_json(value)
        return NodeResult(values, step.next_step_id)


class DetectorHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        detector_id = _string(
            step.options.get("detector", step.options.get("detector_id")),
            "detector ID",
        )
        inputs = _reference_mapping(step.options.get("inputs", {}), state)
        result = context.detections.run(detector_id, inputs=inputs)
        value = {
            "detection_id": result.detection_id,
            "detector_id": result.detector_id,
            "matched": result.matched,
            "rationale": result.rationale,
            "output": result.output,
        }
        detections = dict(_optional_mapping_at(state.values, "detections"))
        detections[detector_id] = value
        values: dict[str, object] = {"detections": detections}
        output = step.options.get("output")
        if output is not None:
            output_name = _string(output, "detector output")
            if output_name == "detections":
                raise ValueError("detector output must not replace the detections namespace")
            values[output_name] = _plain_json(value)
        return NodeResult(values, step.next_step_id)


class HistoryCreateHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        history_id = _string(step.options.get("history"), "history ID")
        _require_declared_history(context, history_id)
        context.histories.create(history_id)
        return NodeResult(
            {_output_name(step.options, step.id): {"history_id": history_id}},
            step.next_step_id,
        )


class HistoryAppendHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        history_id = _string(step.options.get("history"), "history ID")
        _require_declared_history(context, history_id)
        role = _string(step.options.get("role"), "history message role")
        content = _content(step.options, state)
        original = step.options.get("original_message_ref")
        original_message_id = None
        if original is not None:
            original_value = resolve_reference(_string(original, "original_message_ref"), state.values)
            if type(original_value) in (dict, MappingProxyType):
                original_value = original_value.get("message_id")
            original_message_id = _string(original_value, "original message ID")
        visibility = _string_tuple(step.options.get("visibility_roles", ()), "visibility_roles")
        message = context.histories.append(
            history_id,
            role=role,
            content=content,
            source_kind="workflow",
            original_message_id=original_message_id,
            visibility_roles=visibility,
        )
        return NodeResult(
            {_output_name(step.options, step.id): _message_value(message)},
            step.next_step_id,
        )


class HistorySnapshotHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        history_id = _string(step.options.get("history"), "history ID")
        _require_declared_history(context, history_id)
        snapshot = context.histories.snapshot(history_id)
        value = {
            "snapshot_id": snapshot.snapshot_id,
            "history_id": snapshot.history_id,
            "message_ids": snapshot.message_ids,
            "messages": tuple(_message_value(item) for item in snapshot.messages),
        }
        return NodeResult(
            {_output_name(step.options, step.id): value},
            step.next_step_id,
        )


class TransformHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        source_ref = _string(
            step.options.get("source", step.options.get("source_from")),
            "transform source",
        )
        output = _string(step.options.get("output"), "transform output")
        source_path = source_ref[len("values.") :] if source_ref.startswith("values.") else source_ref
        if output in state.values or output == source_path.split(".", 1)[0]:
            raise ValueError("transform output must not overwrite its source")
        source = resolve_reference(source_ref, state.values)
        kind = _string(step.options.get("kind", "identity"), "transform kind")
        transformed = _transform(kind, source, step.options)
        value = {
            "value": transformed,
            "source_ref": source_ref,
            "transform": kind,
        }
        context.events.emit(
            "value.transformed",
            operation_id=operation_id(step, state),
            payload={
                "source_ref": source_ref,
                "value_ref": output,
                "transform": kind,
            },
        )
        return NodeResult({output: value}, step.next_step_id)


class ArtifactCreateHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        media_type = _string(step.options.get("media_type"), "artifact media_type")
        content = _content(step.options, state)
        if type(content) is bytes:
            encoded = content
        elif type(content) is str:
            encoded = content.encode("utf-8")
        else:
            encoded = json.dumps(
                _plain_json(content),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        metadata = _reference_mapping(step.options.get("metadata", {}), state)
        artifact = context.artifacts.create(
            media_type=media_type,
            content=encoded,
            metadata=metadata,
        )
        value = {
            "artifact_id": artifact.artifact_id,
            "media_type": artifact.media_type,
            "content_hash": artifact.content_hash,
            "size_bytes": artifact.size_bytes,
            "metadata": artifact.metadata,
        }
        return NodeResult(
            {_output_name(step.options, step.id): value},
            step.next_step_id,
        )


class BranchHandler:
    def __init__(self, evaluator: ConditionEvaluator | None = None) -> None:
        self._evaluator = evaluator or ConditionEvaluator()

    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        for index, branch in enumerate(step.branches):
            if branch.condition is None or self._evaluator.evaluate(branch.condition, state.values):
                context.events.emit(
                    "branch.selected",
                    operation_id=operation_id(step, state),
                    payload={
                        "step_id": step.id,
                        "branch_index": index,
                        "next_step_id": branch.next_step_id,
                    },
                )
                return NodeResult(next_step_id=branch.next_step_id)
        raise ValueError(f"branch step '{step.id}' matched no declared branch")


class CheckpointHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        checkpoint = context.checkpoints.save(
            position={
                "step_id": step.id,
                "next_step_id": step.next_step_id,
                "visit_counts": state.visit_counts,
                "control_counts": state.control_counts,
                "scope_id": state.scope_id,
            },
            values=state.values,
            history_heads=context.histories.heads(),
            budget_usage=asdict(context.budget.usage),
        )
        value = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "spec_hash": checkpoint.spec_hash,
            "state_schema_version": checkpoint.state_schema_version,
        }
        return NodeResult(
            {_output_name(step.options, step.id): value},
            step.next_step_id,
        )


class FinishHandler:
    def __init__(
        self,
        *,
        validate: Callable[[StepDefinition, WorkflowState], Mapping[str, object]],
    ) -> None:
        self._validate = validate

    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        evidence = self._validate(step, state)
        reason = step.options.get("reason")
        if reason is not None:
            context.events.emit(
                "stop.triggered",
                operation_id=operation_id(step, state),
                payload={"reason": _string(reason, "finish reason")},
            )
        return NodeResult(
            values={_output_name(step.options, "finish"): evidence},
            next_step_id=None,
            finish_evidence=evidence,
        )


StepReplay = Callable[[StepDefinition, WorkflowState, ExecutionContext, int], Awaitable[NodeResult]]

_RETRY_SAFE_NODE_TYPES = frozenset(
    {
        "convert",
        "detector",
        "input_materialize",
        "judge",
        "model_call",
        "render",
        "seed_resolve",
        "target_call",
        "tool_call",
    }
)


class RetryHandler:
    def __init__(self, *, workflow: WorkflowDefinition, replay: StepReplay) -> None:
        self._workflow = workflow
        self._replay = replay

    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        target_id = _string(step.options.get("step"), "retry step")
        target = self._workflow.steps.get(target_id)
        if target is None:
            raise ValueError(f"retry target step '{target_id}' does not exist")
        if target.type not in _RETRY_SAFE_NODE_TYPES:
            raise ValueError(f"retry target node type '{target.type}' is not explicitly replay-safe")
        if target.error_step_id is not None or target.branches:
            raise ValueError("retry target must not declare control-flow edges")
        attempts = step.options.get("max_attempts")
        if type(attempts) is not int or not 1 <= attempts <= 100:
            raise ValueError("retry max_attempts must be an integer from 1 to 100")
        pinned = context.spec.effective_limits.get("max_workflow_retry_attempts")
        if pinned is not None:
            if type(pinned) is not int or pinned < 1:
                raise ValueError("pinned max_workflow_retry_attempts must be positive")
            attempts = min(attempts, pinned)
        retry_on = _string_tuple(step.options.get("retry_on"), "retry_on")
        if not retry_on:
            raise ValueError("retry_on must declare at least one failure category")
        for attempt in range(1, attempts + 1):
            try:
                result = await self._replay(target, state, context, attempt)
            except WorkflowNodeError as error:
                if error.category not in retry_on or attempt == attempts:
                    raise
                if error.indeterminate and context.spec.effective_limits.get("retry_indeterminate") is not True:
                    raise
                continue
            return NodeResult(result.values, step.next_step_id)
        raise AssertionError("bounded retry loop exhausted")


class UnsupportedHandler:
    def __init__(self, node_type: str) -> None:
        self._node_type = node_type

    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        raise UnsupportedWorkflowNodeError(f"workflow node type '{self._node_type}' is reserved for Task 23")


def core_handlers(
    *,
    workflow: WorkflowDefinition,
    replay: StepReplay,
    finish_validator: Callable[[StepDefinition, WorkflowState], Mapping[str, object]],
    template_assets: Mapping[str, str] | None,
) -> Mapping[str, NodeHandler]:
    from app.techniques.workflow.advanced import (
        BacktrackHandler,
        CandidateSelectHandler,
        LoopHandler,
        ParallelHandler,
        PopulationMapHandler,
    )

    templates = None if template_assets is None else TemplateService(assets=template_assets)
    handlers: dict[str, NodeHandler] = {
        "artifact_create": ArtifactCreateHandler(),
        "backtrack": BacktrackHandler(workflow=workflow),
        "branch": BranchHandler(),
        "candidate_select": CandidateSelectHandler(),
        "checkpoint": CheckpointHandler(),
        "convert": ConvertHandler(),
        "detector": DetectorHandler(),
        "finish": FinishHandler(validate=finish_validator),
        "history_append": HistoryAppendHandler(),
        "history_create": HistoryCreateHandler(),
        "history_snapshot": HistorySnapshotHandler(),
        "input_materialize": InputMaterializeHandler(),
        "judge": JudgeHandler(),
        "loop": LoopHandler(),
        "model_call": ModelCallHandler(),
        "parallel": ParallelHandler(workflow=workflow, replay=replay),
        "population_map": PopulationMapHandler(workflow=workflow, replay=replay),
        "render": RenderHandler(templates),
        "retry": RetryHandler(workflow=workflow, replay=replay),
        "seed_resolve": SeedResolveHandler(),
        "target_call": TargetCallHandler(),
        "tool_call": ToolCallHandler(),
        "transform": TransformHandler(),
    }
    return MappingProxyType(handlers)


def resolve_reference(path: str, values: Mapping[str, object]) -> object:
    if type(path) is not str or not path:
        raise ValueError("value reference must be a non-empty built-in string")
    if path == "values":
        raise ValueError("value reference must name a value below the values namespace")
    if path.startswith("values."):
        path = path[len("values.") :]
    current: object = values
    for part in path.split("."):
        if not part:
            raise KeyError(f"workflow value '{path}' does not exist")
        if type(current) in (dict, MappingProxyType):
            if part not in current:
                raise KeyError(f"workflow value '{path}' does not exist")
            current = current[part]
            continue
        if type(current) is tuple and part.isascii() and part.isdecimal():
            if len(part) > 1 and part.startswith("0"):
                raise KeyError(f"workflow value '{path}' does not exist")
            index = int(part)
            if index >= len(current):
                raise KeyError(f"workflow value '{path}' does not exist")
            current = current[index]
            continue
        raise KeyError(f"workflow value '{path}' does not exist")
    return current


def resolve_mapping(
    value: object,
    state: WorkflowState,
) -> Mapping[str, object]:
    """Resolve a declarative workflow mapping against the current value state."""

    return _reference_mapping(value, state)


def _reference_mapping(value: object, state: WorkflowState) -> Mapping[str, object]:
    source = _mapping(value, "reference mapping")
    resolved: dict[str, object] = {}
    for key, item in source.items():
        name = _string(key, "reference field")
        if name.endswith("_from"):
            name = name[: -len("_from")]
            if not name:
                raise ValueError("reference field before '_from' must not be empty")
        resolved[name] = _plain_json(
            resolve_reference(item, state.values) if type(item) is str else _declared_value(item, state)
        )
    return resolved


def _declared_value(value: object, state: WorkflowState) -> object:
    if type(value) in (dict, MappingProxyType) and tuple(value) == ("ref",):
        return resolve_reference(_string(value["ref"], "value ref"), state.values)
    if type(value) in (dict, MappingProxyType) and tuple(value) == ("value",):
        return value["value"]
    return value


def _messages(value: object, state: WorkflowState) -> tuple[dict[str, str], ...]:
    raw_messages = _tuple(value, "messages")
    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        message = _mapping(raw, "message")
        role = _string(message.get("role"), "message role")
        content = _message_content(message, state)
        if type(content) is not str:
            raise TypeError("model message content must resolve to a string")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("messages must not be empty")
    return tuple(messages)


def _target_messages(
    step: StepDefinition,
    state: WorkflowState,
    context: ExecutionContext,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    history_id = step.options.get("history")
    if history_id is not None:
        declared_history = _string(history_id, "history ID")
        _require_declared_history(context, declared_history)
        snapshot = context.histories.snapshot(declared_history)
    else:
        raw_messages = _tuple(step.options.get("messages"), "messages")
        generated_history = f"workflow.{step.id}.{state.visit_counts.get(step.id, 0)}"
        context.histories.create(generated_history)
        for raw in raw_messages:
            message = _mapping(raw, "message")
            context.histories.append(
                generated_history,
                role=_string(message.get("role"), "message role"),
                content=_message_content(message, state),
                source_kind="workflow",
            )
        snapshot = context.histories.snapshot(generated_history)
    messages = tuple({"role": message.role, "content": message.content} for message in snapshot.messages)
    if not messages:
        raise ValueError("target call history must not be empty")
    return messages, snapshot.snapshot_id


def _append_target_exchange(
    step: StepDefinition,
    state: WorkflowState,
    context: ExecutionContext,
    messages: tuple[Mapping[str, object], ...],
    response: str,
) -> None:
    raw_history = step.options.get("append_to_history")
    if raw_history is None:
        return
    history_id = _string(raw_history, "append_to_history")
    _require_declared_history(context, history_id)
    if history_id not in context.histories.heads():
        context.histories.create(history_id)
    if step.options.get("history") != history_id:
        for message in messages:
            context.histories.append(
                history_id,
                role=_string(message.get("role"), "target message role"),
                content=message.get("content"),
                source_kind="workflow",
            )
    completed_event = _latest_operation_event(
        context,
        "target.completed",
        operation_id(step, state),
    )
    context.histories.append(
        history_id,
        role="assistant",
        content=response,
        source_kind="target",
        source_event_id=completed_event.event_id,
    )


def _preflight_target_history_append(
    step: StepDefinition,
    context: ExecutionContext,
) -> None:
    raw_history = step.options.get("append_to_history")
    if raw_history is None:
        return
    _require_declared_history(
        context,
        _string(raw_history, "append_to_history"),
    )


def _latest_operation_event(
    context: ExecutionContext,
    event_type: str,
    expected_operation_id: str,
) -> ExecutionEvent:
    event = next(
        (
            item
            for item in reversed(context.events.events())
            if item.event_type == event_type and item.operation_id == expected_operation_id
        ),
        None,
    )
    if event is None:
        raise WorkflowNodeError(
            "evidence",
            f"{event_type} emitted no event with immutable operation ownership",
            indeterminate=True,
        )
    return event


def _latest_result_event(
    context: ExecutionContext,
    result_id: str,
    event_types: tuple[str, ...],
) -> ExecutionEvent:
    event = next(
        (
            item
            for item in reversed(context.events.events())
            if item.event_type in event_types and item.payload.get("result_id") == result_id
        ),
        None,
    )
    if event is None:
        raise WorkflowNodeError(
            "evidence",
            "judge service returned evidence without an owned lifecycle event",
            indeterminate=True,
        )
    return event


def _message_content(message: Mapping[str, object], state: WorkflowState) -> object:
    reference_keys = tuple(key for key in ("content_ref", "content_from") if key in message)
    has_content = "content" in message
    if len(reference_keys) + int(has_content) != 1:
        raise ValueError("message requires exactly one of content, content_ref, or content_from")
    if reference_keys:
        key = reference_keys[0]
        return resolve_reference(_string(message[key], f"message {key}"), state.values)
    return _declared_value(message["content"], state)


def _content(options: Mapping[str, object], state: WorkflowState) -> object:
    reference_keys = tuple(key for key in ("content_ref", "content_from") if key in options)
    has_content = "content" in options
    if len(reference_keys) + int(has_content) != 1:
        raise ValueError("node requires exactly one of content, content_ref, or content_from")
    if reference_keys:
        key = reference_keys[0]
        return resolve_reference(_string(options[key], key), state.values)
    return _declared_value(options["content"], state)


def _transform(kind: str, source: object, options: Mapping[str, object]) -> object:
    if kind == "identity":
        return source
    if type(source) is not str:
        raise TypeError(f"transform '{kind}' requires a string source")
    if kind == "prefix":
        return _string(options.get("value"), "transform value") + source
    if kind == "suffix":
        return source + _string(options.get("value"), "transform value")
    if kind == "strip":
        return source.strip()
    if kind == "base64":
        return base64.b64encode(source.encode("utf-8")).decode("ascii")
    if kind == "reverse":
        return source[::-1]
    if kind == "replace":
        old = _string(options.get("old"), "transform old")
        new = _string(options.get("new"), "transform new")
        count = options.get("count", -1)
        if type(count) is not int or count < -1:
            raise ValueError("transform replace count must be -1 or a non-negative integer")
        return source.replace(old, new, count)
    raise ValueError(f"unsupported declarative transform '{kind}'")


def _render_inline(
    template: str,
    fields: Mapping[str, object],
    *,
    allow_unused_fields: bool = False,
) -> str:
    required: set[str] = set()
    for _, field_name, format_spec, conversion in string.Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name.isidentifier() or format_spec or conversion:
            raise ValueError("inline template fields must be simple identifiers")
        required.add(field_name)
    missing = required - set(fields)
    extra = set(fields) - required
    if missing:
        raise KeyError(f"template missing fields: {sorted(missing)!r}")
    if extra and not allow_unused_fields:
        raise ValueError(f"template has extra fields: {sorted(extra)!r}")
    return template.format_map(dict(fields))


def _string_refs(value: object, state: WorkflowState) -> tuple[str, ...]:
    if type(value) is str:
        resolved = resolve_reference(value, state.values)
        return _string_tuple(resolved, "artifact_refs")
    refs: list[str] = []
    for item in _tuple(value, "artifact_refs"):
        resolved = _declared_value(item, state)
        refs.append(_string(resolved, "artifact reference"))
    return tuple(refs)


def _referenced_string(path: str, state: WorkflowState, label: str) -> str:
    value = resolve_reference(path, state.values)
    if type(value) in (dict, MappingProxyType):
        value = value.get("artifact_id")
    return _string(value, label)


def _message_value(message: object) -> Mapping[str, object]:
    value = {
        "message_id": getattr(message, "message_id"),
        "history_id": getattr(message, "history_id"),
        "branch_id": getattr(message, "branch_id"),
        "role": getattr(message, "role"),
        "content": getattr(message, "content"),
        "source_kind": getattr(message, "source_kind"),
        "source_event_id": getattr(message, "source_event_id"),
        "original_message_id": getattr(message, "original_message_id"),
    }
    visibility_roles = getattr(message, "visibility_roles")
    if visibility_roles:
        value["visibility_roles"] = visibility_roles
    return value


def _require_declared_history(
    context: ExecutionContext,
    history_id: str,
) -> None:
    definition = getattr(context.loaded_package, "definition", None)
    declared = getattr(definition, "histories", ())
    if history_id not in {item.id for item in declared if type(getattr(item, "id", None)) is str}:
        raise WorkflowNodeError(
            "history",
            f"history '{history_id}' is not declared by the pinned technique",
        )


def _output_name(options: Mapping[str, object], default: str) -> str:
    return _string(options.get("output", default), "node output")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) not in (dict, MappingProxyType):
        raise TypeError(f"{label} must be a mapping")
    return value


def _tuple(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a list")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty built-in string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    items = _tuple(value, label)
    return tuple(_string(item, label) for item in items)


def _mapping_at(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = resolve_reference(name, values)
    return _mapping(value, name)


def _optional_mapping_at(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = values.get(name, {})
    return _mapping(value, name)


def _plain_json(value: object) -> object:
    if type(value) in (dict, MappingProxyType):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_json(item) for item in value]
    return value


__all__ = [
    "NodeHandler",
    "NodeResult",
    "UnsupportedWorkflowNodeError",
    "WorkflowNodeError",
    "core_handlers",
    "operation_id",
    "resolve_mapping",
    "resolve_reference",
]
