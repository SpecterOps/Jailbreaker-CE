"""Deterministic async interpreter for validated declarative workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from app.techniques.contracts.definition import ResultPolicy
from app.techniques.contracts.execution import (
    ExecutionStatus,
    TechniqueOutcome,
    TechniqueRequest,
)
from app.techniques.core.budgets import BudgetExhausted
from app.techniques.core.cancellation import CancellationError
from app.techniques.core.context import ExecutionContext
from app.techniques.core.inputs import InputPolicyError
from app.techniques.core.templates import TemplatePolicyError
from app.techniques.projection.outcome import OutcomeProjector
from app.techniques.workflow.definition import (
    StepDefinition,
    WorkflowDefinition,
    WorkflowState,
)
from app.techniques.workflow.nodes import (
    NodeHandler,
    NodeResult,
    UnsupportedWorkflowNodeError,
    WorkflowNodeError,
    core_handlers,
    operation_id,
    resolve_reference,
)


_TERMINAL_CONTROL = (CancellationError, BudgetExhausted, asyncio.CancelledError)
_INDETERMINATE_SERVICE_NODE_TYPES = frozenset(
    {
        "convert",
        "detector",
        "judge",
        "seed_resolve",
        "target_call",
        "tool_call",
    }
)


class WorkflowErrorPolicyError(RuntimeError):
    """An error edge was absent, untyped, or did not match its failure."""


@dataclass(frozen=True, slots=True)
class _VerifiedWorkflowBinding:
    workflow: WorkflowDefinition
    result_policy: ResultPolicy
    template_assets: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.workflow) is not WorkflowDefinition:
            raise TypeError("verified workflow must be a WorkflowDefinition")
        if not isinstance(self.result_policy, ResultPolicy):
            raise TypeError("verified result policy must be a ResultPolicy")
        if not all(type(key) is str and type(value) is str for key, value in self.template_assets.items()):
            raise TypeError("verified template assets must be a string mapping")
        object.__setattr__(self, "template_assets", MappingProxyType(dict(self.template_assets)))


class WorkflowExecutor:
    """Execute only immutable definitions produced by ``load_workflow``."""

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition | None = None,
        result_policy: ResultPolicy | None = None,
        handlers: Mapping[str, NodeHandler] | None = None,
        trusted_template_assets: Mapping[str, str] | None = None,
    ) -> None:
        if workflow is not None and type(workflow) is not WorkflowDefinition:
            raise TypeError("workflow must be a validated WorkflowDefinition")
        if result_policy is not None and type(result_policy) is not ResultPolicy:
            raise TypeError("result_policy must be a ResultPolicy")
        self._workflow = workflow
        self._result_policy = result_policy
        self._custom_handlers = MappingProxyType(dict(handlers or {}))
        self._trusted_template_assets = (
            None if trusted_template_assets is None else MappingProxyType(dict(trusted_template_assets))
        )

    async def execute(
        self,
        request: TechniqueRequest,
        context: ExecutionContext,
        *,
        workflow: WorkflowDefinition | None = None,
        result_policy: ResultPolicy | None = None,
    ) -> TechniqueOutcome:
        return await self._execute_selected(
            request,
            context,
            workflow=self._workflow if workflow is None else workflow,
            result_policy=self._result_policy if result_policy is None else result_policy,
            template_assets=self._trusted_template_assets,
        )

    async def _execute_verified(
        self,
        request: TechniqueRequest,
        context: ExecutionContext,
        binding: _VerifiedWorkflowBinding,
    ) -> TechniqueOutcome:
        if type(binding) is not _VerifiedWorkflowBinding:
            raise TypeError("workflow package binding is not verified")
        return await self._execute_selected(
            request,
            context,
            workflow=binding.workflow,
            result_policy=binding.result_policy,
            template_assets=binding.template_assets,
        )

    async def _execute_selected(
        self,
        request: TechniqueRequest,
        context: ExecutionContext,
        *,
        workflow: WorkflowDefinition | None,
        result_policy: ResultPolicy | None,
        template_assets: Mapping[str, str] | None,
    ) -> TechniqueOutcome:
        selected_workflow = workflow
        if type(selected_workflow) is not WorkflowDefinition:
            raise TypeError("WorkflowExecutor requires an already validated WorkflowDefinition")
        selected_policy = _context_result_policy(context) if result_policy is None else result_policy
        if type(selected_policy) is not ResultPolicy:
            raise TypeError("workflow result policy must be a ResultPolicy")

        state = WorkflowState.initial(selected_workflow)
        state.apply(
            {
                "inputs": request.inputs,
                "parameters": request.spec.resolved_parameters,
                "metadata": request.metadata,
                "artifact_refs": request.artifact_refs,
            }
        )
        projector = OutcomeProjector()
        handlers: dict[str, NodeHandler] = {}

        def validate_finish(
            step: StepDefinition,
            finish_state: WorkflowState,
        ) -> Mapping[str, object]:
            projected = projector.project(
                context.events.events(),
                result_policy=selected_policy,
            )
            if selected_policy.final_response is not None and projected.final_response is None:
                raise WorkflowNodeError(
                    "finish",
                    "declared final response matched no completed target evidence",
                )
            final_response_from = step.options.get("final_response_from")
            if final_response_from is not None:
                referenced = _evidence_mapping(
                    resolve_reference(
                        _required_string(final_response_from, "finish final_response_from"),
                        finish_state.values,
                    ),
                    "final response evidence",
                )
                if (
                    referenced.get("event_id") != projected.final_response_event_id
                    or referenced.get("response_kind") != "real"
                ):
                    raise WorkflowNodeError(
                        "finish",
                        "finish final response evidence does not match the real response selected by policy",
                    )

            official_judge_from = step.options.get("official_judge_from")
            if selected_policy.official_judge is not None:
                if projected.official_judge_ref is None or projected.official_judge_event_id is None:
                    raise WorkflowNodeError(
                        "finish",
                        "declared official judge evidence is missing or failed",
                    )
                if official_judge_from is not None:
                    referenced = _evidence_mapping(
                        resolve_reference(
                            _required_string(
                                official_judge_from,
                                "finish official_judge_from",
                            ),
                            finish_state.values,
                        ),
                        "official judge evidence",
                    )
                    if (
                        referenced.get("definition_id") != selected_policy.official_judge
                        or referenced.get("status") != "succeeded"
                        or not _is_score(referenced.get("score"))
                        or referenced.get("result_id") != projected.official_judge_ref
                        or referenced.get("event_id") != projected.official_judge_event_id
                    ):
                        raise WorkflowNodeError(
                            "finish",
                            "finish official judge evidence does not match the succeeded judge selected by policy",
                        )
            elif official_judge_from is not None:
                raise WorkflowNodeError(
                    "finish",
                    "finish official judge evidence is not declared by the result policy",
                )

            evidence: dict[str, object] = {}
            if projected.final_response_event_id is not None:
                evidence["final_response_event_id"] = projected.final_response_event_id
            if projected.official_judge_ref is not None:
                evidence["official_judge_result_id"] = projected.official_judge_ref
            if projected.official_judge_event_id is not None:
                evidence["official_judge_event_id"] = projected.official_judge_event_id
            return MappingProxyType(evidence)

        async def replay(
            step: StepDefinition,
            replay_state: WorkflowState,
            replay_context: ExecutionContext,
            retry_attempt: int,
        ) -> NodeResult:
            return await self._execute_step(
                step=step,
                state=replay_state,
                context=replay_context,
                handlers=handlers,
                retry_attempt=retry_attempt,
            )

        handlers.update(
            core_handlers(
                workflow=selected_workflow,
                replay=replay,
                finish_validator=validate_finish,
                template_assets=template_assets,
            )
        )
        handlers.update(self._custom_handlers)

        context.events.emit(
            "run.started",
            payload={
                "workflow_version": selected_workflow.version,
                "start_step_id": selected_workflow.start,
            },
        )
        finish_evidence: Mapping[str, object] | None = None
        try:
            while state.current_step_id is not None:
                context.cancellation.raise_if_cancelled()
                step = selected_workflow.steps[state.current_step_id]
                try:
                    result = await self._execute_step(
                        step=step,
                        state=state,
                        context=context,
                        handlers=handlers,
                    )
                except _TERMINAL_CONTROL:
                    raise
                except Exception as error:
                    if step.type == "finish":
                        raise
                    state.current_step_id = _error_transition(step, error)
                else:
                    state.apply(result.values)
                    if step.type == "finish":
                        if result.finish_evidence is None:
                            raise WorkflowNodeError(
                                "completion",
                                "workflow finish produced no immutable evidence",
                            )
                        finish_evidence = MappingProxyType(dict(result.finish_evidence))
                    state.current_step_id = result.next_step_id
            projected = projector.project(
                context.events.events(),
                result_policy=selected_policy,
            )
            if finish_evidence is None:
                raise WorkflowNodeError(
                    "completion",
                    "workflow terminated without finish evidence",
                )
            auxiliary_refs = _auxiliary_judge_refs(
                context.events.events(),
                selected_policy,
            )
            artifact_refs = tuple(
                artifact_id
                for event in context.events.events()
                if event.event_type == "artifact.created"
                for artifact_id in (event.payload.get("artifact_id"),)
                if type(artifact_id) is str
            )
            outcome = TechniqueOutcome(
                execution_status=ExecutionStatus.COMPLETED,
                result_disposition=projected.result_disposition,
                attack_outcome=projected.attack_outcome,
                final_response_ref=projected.final_response_event_id,
                official_judge_ref=projected.official_judge_ref,
                stopping_reason=projected.stopping_reason,
                auxiliary_judge_refs=auxiliary_refs,
                artifact_refs=artifact_refs,
                outputs={
                    "official_score": projected.official_score,
                    "final_response": projected.final_response,
                    "finish_evidence": dict(finish_evidence),
                    "projected_event_count": projected.projected_event_count + 1,
                },
            )
        except CancellationError:
            context.events.emit(
                "run.cancelled",
                payload={"category": "cancellation"},
            )
            raise
        except asyncio.CancelledError:
            context.events.emit(
                "run.cancelled",
                payload={"category": "task_cancellation"},
            )
            raise
        except BudgetExhausted as error:
            context.events.emit(
                "budget.exhausted",
                payload={
                    "resource": error.resource,
                    "limit": error.limit,
                    "requested": error.requested,
                },
            )
            raise
        except Exception as error:
            _emit_run_failed_once(context, error)
            raise

        context.events.emit("run.completed", payload={})
        return outcome

    async def _execute_step(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
        handlers: Mapping[str, NodeHandler],
        retry_attempt: int = 1,
    ) -> NodeResult:
        context.cancellation.raise_if_cancelled()
        state.increment_visit(step.id)
        step_operation_id = operation_id(step, state)
        context.events.emit(
            "step.started",
            operation_id=step_operation_id,
            attempt=retry_attempt,
            payload={"step_id": step.id, "step_type": step.type},
        )
        handler = handlers.get(step.type)
        if handler is None:
            error: Exception = UnsupportedWorkflowNodeError(
                f"workflow node type '{step.type}' has no registered handler"
            )
            _emit_step_failed(
                context,
                step,
                step_operation_id,
                retry_attempt,
                error,
            )
            raise error
        try:
            result = await handler.execute(step=step, state=state, context=context)
        except _TERMINAL_CONTROL as error:
            _emit_step_failed(
                context,
                step,
                step_operation_id,
                retry_attempt,
                error,
                category="terminal_control",
            )
            raise
        except UnsupportedWorkflowNodeError as error:
            _emit_step_failed(
                context,
                step,
                step_operation_id,
                retry_attempt,
                error,
                category="unsupported_node",
            )
            raise
        except WorkflowNodeError as error:
            _emit_step_failed(
                context,
                step,
                step_operation_id,
                retry_attempt,
                error,
                category=error.category,
            )
            raise
        except Exception as error:
            normalized = _normalize_node_error(
                step,
                error,
                context=context,
                step_operation_id=step_operation_id,
            )
            _emit_step_failed(
                context,
                step,
                step_operation_id,
                retry_attempt,
                normalized,
                category=normalized.category,
            )
            raise normalized from error
        if type(result) is not NodeResult:
            error = WorkflowNodeError(
                step.type,
                f"handler for '{step.type}' returned an invalid result",
            )
            _emit_step_failed(
                context,
                step,
                step_operation_id,
                retry_attempt,
                error,
                category=error.category,
            )
            raise error
        context.events.emit(
            "step.completed",
            operation_id=step_operation_id,
            attempt=retry_attempt,
            payload={
                "step_id": step.id,
                "step_type": step.type,
                "output_refs": tuple(result.values),
            },
        )
        return result


def _context_result_policy(context: ExecutionContext) -> ResultPolicy:
    definition = getattr(context.loaded_package, "definition", None)
    policy = getattr(definition, "result_policy", None)
    return policy if isinstance(policy, ResultPolicy) else ResultPolicy()


def _error_transition(step: StepDefinition, error: Exception) -> str:
    if step.error_step_id is None:
        raise error
    if not isinstance(error, WorkflowNodeError):
        raise WorkflowErrorPolicyError(f"step '{step.id}' cannot route an untyped failure") from error
    declared = step.options.get("error_types")
    if type(declared) is not tuple or not declared:
        raise WorkflowErrorPolicyError(f"step '{step.id}' error edge requires non-empty error_types") from error
    if any(type(item) is not str or not item for item in declared):
        raise WorkflowErrorPolicyError(f"step '{step.id}' error_types must contain non-empty strings") from error
    if error.category not in declared:
        raise WorkflowErrorPolicyError(
            f"step '{step.id}' error edge does not accept category '{error.category}'"
        ) from error
    return step.error_step_id


def _normalize_node_error(
    step: StepDefinition,
    error: Exception,
    *,
    context: ExecutionContext,
    step_operation_id: str,
) -> WorkflowNodeError:
    category = step.type
    indeterminate = False
    if step.type == "model_call":
        indeterminate = _model_failure_is_indeterminate(
            context,
            step_operation_id,
            error,
        )
    elif step.type in _INDETERMINATE_SERVICE_NODE_TYPES:
        # Each of these nodes can cross a callback/provider boundary before a
        # failure is observed. Replay therefore requires the pinned run's
        # explicit retry_indeterminate permission, even for policy-shaped errors.
        indeterminate = True
    elif isinstance(error, (InputPolicyError, TemplatePolicyError)):
        indeterminate = False
    return WorkflowNodeError(
        category,
        f"workflow node '{step.id}' failed with {type(error).__name__}",
        indeterminate=indeterminate,
    )


def _model_failure_is_indeterminate(
    context: ExecutionContext,
    step_operation_id: str,
    error: Exception,
) -> bool:
    owned_events = tuple(
        event
        for event in reversed(context.events.events())
        if event.operation_id == step_operation_id and event.event_type in {"model.failed", "model.requested"}
    )
    for event in owned_events:
        if event.event_type == "model.failed":
            value = event.payload.get("indeterminate")
            if type(value) is bool:
                return value
        if event.event_type == "model.requested":
            return True
    return not isinstance(error, (KeyError, TypeError, ValueError))


def _emit_step_failed(
    context: ExecutionContext,
    step: StepDefinition,
    step_operation_id: str,
    attempt: int,
    error: BaseException,
    *,
    category: str | None = None,
) -> None:
    context.events.emit(
        "step.failed",
        operation_id=step_operation_id,
        attempt=attempt,
        payload={
            "step_id": step.id,
            "step_type": step.type,
            "category": category or _safe_failure_category(error),
            "error_type": type(error).__name__,
        },
    )


def _safe_failure_category(error: BaseException) -> str:
    if isinstance(error, WorkflowNodeError):
        return error.category
    if isinstance(error, UnsupportedWorkflowNodeError):
        return "unsupported_node"
    if isinstance(error, WorkflowErrorPolicyError):
        return "error_policy"
    return "workflow"


def _emit_run_failed_once(context: ExecutionContext, error: BaseException) -> None:
    if any(event.event_type == "run.failed" for event in context.events.events()):
        return
    context.events.emit(
        "run.failed",
        payload={"category": _safe_failure_category(error)},
    )


def _evidence_mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) not in (dict, MappingProxyType):
        raise WorkflowNodeError("finish", f"finish {label} must resolve to owned evidence")
    return value


def _required_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise WorkflowNodeError("finish", f"{label} must be a non-empty string")
    return value


def _is_score(value: object) -> bool:
    return type(value) in (int, float)


def _auxiliary_judge_refs(events: tuple[object, ...], policy: ResultPolicy) -> tuple[str, ...]:
    refs: list[str] = []
    for definition_id in policy.auxiliary_judges:
        matching = [
            event
            for event in events
            if getattr(event, "event_type", None) == "judge.completed"
            and event.payload.get("definition_id") == definition_id
        ]
        if matching:
            result_id = matching[-1].payload.get("result_id")
            refs.append(result_id if type(result_id) is str else matching[-1].event_id)
    return tuple(refs)


__all__ = ["WorkflowErrorPolicyError", "WorkflowExecutor"]
