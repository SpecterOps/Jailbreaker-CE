"""Bounded advanced control-flow handlers for declarative workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Awaitable, Callable, Generic, Iterable, Mapping, TypeVar

from app.techniques.core.budgets import BudgetExhausted, BudgetReservation
from app.techniques.core.cancellation import CancellationError
from app.techniques.core.context import ExecutionContext
from app.techniques.core.events import EventBuffer
from app.techniques.workflow.conditions import ConditionEvaluator
from app.techniques.workflow.definition import (
    StepDefinition,
    WorkflowDefinition,
    WorkflowState,
)
from app.techniques.workflow.nodes import (
    NodeResult,
    WorkflowNodeError,
    operation_id,
    resolve_mapping,
    resolve_reference,
)


MAX_PARALLEL_BRANCHES = 64
MAX_POPULATION_SIZE = 1_000
MAX_BACKTRACKS = 100
MAX_ROUNDS = 10_000

StepReplay = Callable[
    [StepDefinition, WorkflowState, ExecutionContext, int],
    Awaitable[NodeResult],
]


@dataclass(frozen=True, slots=True)
class _BranchPlan:
    branch_id: str
    step: StepDefinition
    values: Mapping[str, object]
    capacity: Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class _BranchOutcome:
    branch_id: str
    result: NodeResult | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _CandidatePlan:
    candidate_id: str
    value: object
    provenance: Mapping[str, object]
    step: StepDefinition
    capacity: Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    plan: _CandidatePlan
    result: NodeResult | None = None
    error: BaseException | None = None


_PlanT = TypeVar("_PlanT")
_OutcomeT = TypeVar("_OutcomeT")


@dataclass(frozen=True, slots=True)
class _FanoutAdapter(Generic[_PlanT, _OutcomeT]):
    identity: Callable[[_PlanT], str]
    capacity: Callable[[_PlanT], Mapping[str, int | float]]
    run: Callable[[_PlanT, BudgetReservation, EventBuffer], Awaitable[_OutcomeT]]
    error: Callable[[_OutcomeT], BaseException | None]
    from_exception: Callable[[_PlanT, BaseException], _OutcomeT]
    task_name_prefix: str


class ParallelHandler:
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
        plans = self._plans(step, state, context)
        maximum = _effective_limit(
            context,
            option=step.options.get("max_branches"),
            pinned_name="max_parallel_branches",
            hard=MAX_PARALLEL_BRANCHES,
        )
        if len(plans) > maximum:
            raise WorkflowNodeError(
                "parallel",
                f"parallel branch count {len(plans)} exceeds limit {maximum}",
            )
        state.increment_control(
            "parallel_branches",
            maximum=maximum,
            amount=len(plans),
        )
        return await self._dispatch(
            step=step,
            state=state,
            context=context,
            plans=plans,
        )

    def _plans(
        self,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> tuple[_BranchPlan, ...]:
        raw_branches = _tuple(step.options.get("branches"), "parallel branches")
        if not raw_branches:
            raise ValueError("parallel branches must not be empty")
        default_capacity = _capacity_mapping(
            step.options.get("per_branch_budget", {}),
            "parallel per_branch_budget",
        )
        plans: list[_BranchPlan] = []
        seen: set[str] = set()
        for raw in raw_branches:
            branch = _mapping(raw, "parallel branch")
            branch_id = _string(branch.get("id"), "parallel branch ID")
            if branch_id in seen:
                raise ValueError(f"duplicate parallel branch ID '{branch_id}'")
            seen.add(branch_id)
            target_id = _string(branch.get("step"), "parallel branch step")
            target = self._workflow.steps.get(target_id)
            if target is None:
                raise ValueError(f"parallel branch step '{target_id}' does not exist")
            values = _mapping(branch.get("values", {}), "parallel branch values")
            inferred = _step_capacity(target, state, context, self._workflow)
            declared = {
                **default_capacity,
                **_capacity_mapping(
                    branch.get("budget", {}),
                    "parallel branch budget",
                ),
            }
            capacity = _bounded_capacity(inferred, declared)
            plans.append(
                _BranchPlan(
                    branch_id=branch_id,
                    step=target,
                    values=MappingProxyType(dict(values)),
                    capacity=MappingProxyType(capacity),
                )
            )
        return tuple(sorted(plans, key=lambda item: item.branch_id))

    async def _dispatch(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
        plans: tuple[_BranchPlan, ...],
    ) -> NodeResult:
        parent_operation = operation_id(step, state)
        failure_policy = _string(
            step.options.get("failure_policy", "fail_fast"),
            "parallel failure_policy",
        )
        if failure_policy not in {"collect", "fail_fast"}:
            raise ValueError("parallel failure_policy must be 'collect' or 'fail_fast'")
        parent_event_id = _causal_step_event_id(context, parent_operation)

        async def run(
            plan: _BranchPlan,
            reservation: BudgetReservation,
            buffer: EventBuffer,
        ) -> _BranchOutcome:
            return await self._run_branch(
                plan=plan,
                parent_state=state,
                parent_context=context,
                reservation=reservation,
                buffer=buffer,
                parent_operation=parent_operation,
            )

        outcomes = await _execute_bounded_fanout(
            context=context,
            plans=plans,
            parent_event_id=parent_event_id,
            failure_policy=failure_policy,
            adapter=_FanoutAdapter[
                _BranchPlan,
                _BranchOutcome,
            ](
                identity=lambda plan: plan.branch_id,
                capacity=lambda plan: plan.capacity,
                run=run,
                error=lambda outcome: outcome.error,
                from_exception=lambda plan, error: _BranchOutcome(
                    plan.branch_id,
                    error=error,
                ),
                task_name_prefix="workflow-branch",
            ),
        )

        branch_values = tuple(_branch_value(outcome) for outcome in outcomes)
        successful = tuple(outcome.branch_id for outcome in outcomes if outcome.result is not None)
        selected_branch_id = successful[0] if successful else None
        if selected_branch_id is not None:
            context.events.emit(
                "branch.selected",
                operation_id=parent_operation,
                branch_id=selected_branch_id,
                payload={
                    "step_id": step.id,
                    "branch_id": selected_branch_id,
                    "selection_policy": "declared_id",
                },
            )
        value = {
            "branches": branch_values,
            "selected_branch_id": selected_branch_id,
        }
        context.events.emit(
            "parallel.completed",
            operation_id=parent_operation,
            payload=value,
        )
        return NodeResult(
            {_output_name(step, step.id): value},
            step.next_step_id,
        )

    async def _run_branch(
        self,
        *,
        plan: _BranchPlan,
        parent_state: WorkflowState,
        parent_context: ExecutionContext,
        reservation: BudgetReservation,
        buffer: EventBuffer,
        parent_operation: str,
    ) -> _BranchOutcome:
        scope_id = plan.branch_id if parent_state.scope_id is None else f"{parent_state.scope_id}/{plan.branch_id}"
        child_state = parent_state.fork(
            scope_id,
            values={
                **dict(plan.values),
                "branch": {"id": plan.branch_id},
            },
        )
        child_context = parent_context.fork(scope_id)
        with parent_context.events.activate(buffer), parent_context.budget.activate(reservation):
            parent_context.events.emit(
                "branch.started",
                operation_id=parent_operation,
                payload={"branch_id": plan.branch_id, "step_id": plan.step.id},
            )
            try:
                result = await self._replay(
                    plan.step,
                    child_state,
                    child_context,
                    1,
                )
            except BaseException as error:
                parent_context.events.emit(
                    "branch.failed",
                    operation_id=parent_operation,
                    payload={
                        "branch_id": plan.branch_id,
                        "step_id": plan.step.id,
                        "category": _error_category(error),
                    },
                )
                return _BranchOutcome(plan.branch_id, error=error)
            parent_context.events.emit(
                "branch.completed",
                operation_id=parent_operation,
                payload={"branch_id": plan.branch_id, "step_id": plan.step.id},
            )
            return _BranchOutcome(plan.branch_id, result=result)


class BacktrackHandler:
    def __init__(self, *, workflow: WorkflowDefinition) -> None:
        self._workflow = workflow

    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        maximum = _effective_limit(
            context,
            option=step.options.get("max_backtracks"),
            pinned_name="max_backtracks",
            hard=MAX_BACKTRACKS,
        )
        try:
            state.validate_control("backtracks", maximum=maximum)
        except RuntimeError as error:
            raise WorkflowNodeError("backtrack", str(error)) from error
        checkpoint_id = _checkpoint_id(step, state)
        checkpoint = context.checkpoints.load_for_backtrack(checkpoint_id)
        position = _mapping(checkpoint.position, "checkpoint position")
        inactive = context.histories.validate_restore_heads(checkpoint.history_heads)

        def emit_started(count: int | None) -> None:
            if count is None:  # pragma: no cover - consume_control always returns it
                raise AssertionError("backtrack reconciliation did not consume its counter")
            context.events.emit(
                "backtrack.started",
                operation_id=operation_id(step, state),
                payload={
                    "checkpoint_id": checkpoint_id,
                    "backtrack_count": count,
                    "inactive_message_ids": inactive,
                },
            )

        try:
            restored_inactive, count = state.reconcile_checkpoint(
                position=position,
                values=checkpoint.values,
                workflow=self._workflow,
                histories=context.histories,
                history_heads=checkpoint.history_heads,
                checkpoint_id=checkpoint_id,
                expected_inactive=inactive,
                consume_control=("backtracks", maximum, 1),
                before_history_events=emit_started,
            )
        except RuntimeError as error:
            raise WorkflowNodeError("backtrack", str(error)) from error
        if count is None:  # pragma: no cover - consume_control always returns it
            raise AssertionError("backtrack reconciliation did not consume its counter")
        value = {
            "checkpoint_id": checkpoint_id,
            "backtrack_count": count,
            "inactive_message_ids": restored_inactive,
        }
        return NodeResult(
            {_output_name(step, step.id): value},
            step.next_step_id,
        )


class LoopHandler:
    def __init__(self, evaluator: ConditionEvaluator | None = None) -> None:
        self._evaluator = evaluator or ConditionEvaluator()

    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        if not step.branches:
            raise ValueError("loop requires declared conditional and exit branches")
        matched = next(
            (
                (index, branch)
                for index, branch in enumerate(step.branches)
                if branch.condition is None or self._evaluator.evaluate(branch.condition, state.values)
            ),
            None,
        )
        if matched is None:
            raise ValueError(f"loop step '{step.id}' matched no declared transition")
        branch_index, branch = matched
        if branch.condition is None:
            context.events.emit(
                "stop.triggered",
                operation_id=operation_id(step, state),
                payload={
                    "reason": "loop_condition_false",
                    "step_id": step.id,
                    "next_step_id": branch.next_step_id,
                },
            )
            return NodeResult(next_step_id=branch.next_step_id)

        maximum = _effective_limit(
            context,
            option=step.options.get("max_rounds"),
            pinned_name="max_rounds",
            hard=MAX_ROUNDS,
        )
        counter = f"rounds:{step.id}"
        try:
            state.validate_control(counter, maximum=maximum)
            round_number = state.increment_control(counter, maximum=maximum)
        except RuntimeError as error:
            raise WorkflowNodeError("loop", str(error)) from error
        context.events.emit(
            "round.started",
            operation_id=operation_id(step, state),
            round=round_number,
            payload={
                "round": round_number,
                "step_id": step.id,
                "branch_index": branch_index,
                "next_step_id": branch.next_step_id,
            },
        )
        return NodeResult(
            {
                _output_name(step, "round"): {
                    "number": round_number,
                    "step_id": step.id,
                }
            },
            branch.next_step_id,
        )


class PopulationMapHandler:
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
        plans = self._plans(step, state, context)
        maximum = _effective_limit(
            context,
            option=step.options.get("max_population_size"),
            pinned_name="max_population_size",
            hard=MAX_POPULATION_SIZE,
        )
        if len(plans) > maximum:
            raise WorkflowNodeError(
                "population_map",
                f"population size {len(plans)} exceeds limit {maximum}",
            )
        state.increment_control(
            "population_candidates",
            maximum=maximum,
            amount=len(plans),
        )
        generation = _positive_limit(
            step.options.get("generation", 1),
            "population generation",
            minimum=1,
        )
        return await self._dispatch(
            step=step,
            state=state,
            context=context,
            generation=generation,
            plans=plans,
        )

    def _plans(
        self,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> tuple[_CandidatePlan, ...]:
        has_inline = "candidates" in step.options
        has_reference = "candidates_from" in step.options
        if has_inline == has_reference:
            raise ValueError("population_map requires exactly one of candidates or candidates_from")
        source = (
            step.options["candidates"]
            if has_inline
            else resolve_reference(
                _string(step.options["candidates_from"], "candidates_from"),
                state.values,
            )
        )
        raw_candidates = _tuple(source, "population candidates")
        if not raw_candidates:
            raise ValueError("population candidates must not be empty")
        target_id = _string(step.options.get("step"), "population candidate step")
        target = self._workflow.steps.get(target_id)
        if target is None:
            raise ValueError(f"population candidate step '{target_id}' does not exist")
        default_capacity = _capacity_mapping(
            step.options.get("per_candidate_budget", {}),
            "population per_candidate_budget",
        )
        plans: list[_CandidatePlan] = []
        seen: set[str] = set()
        for raw in raw_candidates:
            candidate = _mapping(raw, "population candidate")
            candidate_id = _string(candidate.get("id"), "candidate ID")
            if candidate_id in seen:
                raise ValueError(f"duplicate candidate ID '{candidate_id}'")
            seen.add(candidate_id)
            if "value" not in candidate:
                raise ValueError(f"candidate '{candidate_id}' requires a value")
            provenance = _mapping(
                candidate.get("provenance", {}),
                "candidate provenance",
            )
            scope_id = candidate_id if state.scope_id is None else f"{state.scope_id}/{candidate_id}"
            capacity_state = state.fork(
                scope_id,
                values={
                    "candidate": {
                        "id": candidate_id,
                        "value": candidate["value"],
                        "provenance": provenance,
                    }
                },
            )
            inferred = _step_capacity(
                target,
                capacity_state,
                context,
                self._workflow,
            )
            declared = {
                **default_capacity,
                **_capacity_mapping(
                    candidate.get("budget", {}),
                    "candidate budget",
                ),
            }
            plans.append(
                _CandidatePlan(
                    candidate_id=candidate_id,
                    value=candidate["value"],
                    provenance=MappingProxyType(dict(provenance)),
                    step=target,
                    capacity=MappingProxyType(_bounded_capacity(inferred, declared)),
                )
            )
        return tuple(sorted(plans, key=lambda item: item.candidate_id))

    async def _dispatch(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
        generation: int,
        plans: tuple[_CandidatePlan, ...],
    ) -> NodeResult:
        parent_operation = operation_id(step, state)
        failure_policy = _string(
            step.options.get("failure_policy", "fail_fast"),
            "population failure_policy",
        )
        if failure_policy not in {"collect", "fail_fast"}:
            raise ValueError("population failure_policy must be 'collect' or 'fail_fast'")
        parent_event_id = _causal_step_event_id(context, parent_operation)

        def before_tasks() -> None:
            context.events.emit(
                "generation.started",
                operation_id=parent_operation,
                round=generation,
                payload={"generation": generation, "candidate_count": len(plans)},
            )

        async def run(
            plan: _CandidatePlan,
            reservation: BudgetReservation,
            buffer: EventBuffer,
        ) -> _CandidateOutcome:
            return await self._run_candidate(
                plan=plan,
                generation=generation,
                parent_state=state,
                parent_context=context,
                reservation=reservation,
                buffer=buffer,
                parent_operation=parent_operation,
            )

        outcomes = await _execute_bounded_fanout(
            context=context,
            plans=plans,
            parent_event_id=parent_event_id,
            failure_policy=failure_policy,
            before_tasks=before_tasks,
            adapter=_FanoutAdapter[
                _CandidatePlan,
                _CandidateOutcome,
            ](
                identity=lambda plan: plan.candidate_id,
                capacity=lambda plan: plan.capacity,
                run=run,
                error=lambda outcome: outcome.error,
                from_exception=lambda plan, error: _CandidateOutcome(
                    plan,
                    error=error,
                ),
                task_name_prefix="workflow-candidate",
            ),
        )
        candidate_values = tuple(_candidate_value(outcome) for outcome in outcomes)
        value = {
            "generation": generation,
            "candidates": candidate_values,
        }
        context.events.emit(
            "generation.completed",
            operation_id=parent_operation,
            round=generation,
            payload=value,
        )
        return NodeResult(
            {_output_name(step, step.id): value},
            step.next_step_id,
        )

    async def _run_candidate(
        self,
        *,
        plan: _CandidatePlan,
        generation: int,
        parent_state: WorkflowState,
        parent_context: ExecutionContext,
        reservation: BudgetReservation,
        buffer: EventBuffer,
        parent_operation: str,
    ) -> _CandidateOutcome:
        scope_id = (
            plan.candidate_id if parent_state.scope_id is None else f"{parent_state.scope_id}/{plan.candidate_id}"
        )
        candidate_value = {
            "id": plan.candidate_id,
            "generation": generation,
            "value": plan.value,
            "provenance": plan.provenance,
        }
        child_state = parent_state.fork(
            scope_id,
            values={"candidate": candidate_value},
        )
        child_context = parent_context.fork(scope_id)
        with parent_context.events.activate(buffer), parent_context.budget.activate(reservation):
            parent_context.events.emit(
                "candidate.created",
                operation_id=parent_operation,
                round=generation,
                payload={
                    "generation": generation,
                    "candidate_id": plan.candidate_id,
                    "provenance": plan.provenance,
                },
            )
            try:
                result = await self._replay(
                    plan.step,
                    child_state,
                    child_context,
                    1,
                )
            except BaseException as error:
                parent_context.events.emit(
                    "candidate.failed",
                    operation_id=parent_operation,
                    round=generation,
                    payload={
                        "generation": generation,
                        "candidate_id": plan.candidate_id,
                        "category": _error_category(error),
                    },
                )
                return _CandidateOutcome(plan, error=error)
            parent_context.events.emit(
                "candidate.completed",
                operation_id=parent_operation,
                round=generation,
                payload={
                    "generation": generation,
                    "candidate_id": plan.candidate_id,
                },
            )
            return _CandidateOutcome(plan, result=result)


class CandidateSelectHandler:
    async def execute(
        self,
        *,
        step: StepDefinition,
        state: WorkflowState,
        context: ExecutionContext,
    ) -> NodeResult:
        population = _mapping(
            resolve_reference(
                _string(
                    step.options.get("candidates_from"),
                    "candidate selection candidates_from",
                ),
                state.values,
            ),
            "candidate population",
        )
        generation = _positive_limit(
            population.get("generation"),
            "candidate generation",
            minimum=1,
        )
        candidates = _tuple(population.get("candidates"), "candidate population")
        maximum = _effective_limit(
            context,
            option=step.options.get("max_population_size"),
            pinned_name="max_population_size",
            hard=MAX_POPULATION_SIZE,
        )
        if len(candidates) > maximum:
            raise WorkflowNodeError(
                "candidate_select",
                f"population size {len(candidates)} exceeds limit {maximum}",
            )
        candidate_ids = tuple(
            _string(
                _mapping(candidate, "candidate selection entry").get("candidate_id"),
                "candidate selection ID",
            )
            for candidate in candidates
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate selection IDs must be unique")
        judge_path = _string(step.options.get("judge_from"), "judge_from")
        definition_id = _string(
            step.options.get("judge_definition"),
            "judge_definition",
        )
        event_by_id = {event.event_id: event for event in context.events.events()}
        ranked = tuple(
            sorted(
                (
                    _rank_candidate(
                        candidate,
                        judge_path=judge_path,
                        definition_id=definition_id,
                        event_by_id=event_by_id,
                    )
                    for candidate in candidates
                ),
                key=_ranking_key,
            )
        )
        eligible = tuple(item for item in ranked if item["rank_status"] == "eligible")
        selected = eligible[0] if eligible else None
        selected_id = (
            None
            if selected is None
            else _string(
                selected.get("candidate_id"),
                "selected candidate ID",
            )
        )
        if selected is not None:
            selected_event_id = _string(
                selected.get("event_id"),
                "selected judge event ID",
            )
            context.events.emit(
                "candidate.selected",
                operation_id=operation_id(step, state),
                round=generation,
                branch_id=selected_id,
                caused_by=selected_event_id,
                payload={
                    "generation": generation,
                    "candidate_id": selected_id,
                    "score": selected["score"],
                    "judge_definition": definition_id,
                    "judge_result_id": selected["result_id"],
                },
            )
        value = {
            "generation": generation,
            "selected_candidate_id": selected_id,
            "score": None if selected is None else selected["score"],
            "status": "selected" if selected is not None else "no_successful_scores",
            "ranking": ranked,
        }
        context.events.emit(
            "candidate.selection_completed",
            operation_id=operation_id(step, state),
            round=generation,
            payload=value,
        )
        return NodeResult(
            {_output_name(step, step.id): value},
            step.next_step_id,
        )


async def _execute_bounded_fanout(
    *,
    context: ExecutionContext,
    plans: tuple[_PlanT, ...],
    parent_event_id: str,
    failure_policy: str,
    adapter: _FanoutAdapter[_PlanT, _OutcomeT],
    before_tasks: Callable[[], None] | None = None,
) -> tuple[_OutcomeT, ...]:
    identities = tuple(adapter.identity(plan) for plan in plans)
    if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
        raise ValueError("fan-out plans must have unique identities in deterministic order")
    aggregate = context.budget.reserve_capacity(**_sum_capacities(adapter.capacity(plan) for plan in plans))
    reservations: dict[str, BudgetReservation] = {}
    tasks: dict[str, asyncio.Task[_OutcomeT]] = {}
    buffers: dict[str, EventBuffer] = {}
    try:
        for plan, identity in zip(plans, identities, strict=True):
            reservations[identity] = aggregate.reserve_capacity(**dict(adapter.capacity(plan)))
        if before_tasks is not None:
            before_tasks()
        buffers = {
            identity: context.events.new_buffer(
                branch_id=identity,
                caused_by=parent_event_id,
            )
            for identity in identities
        }
        plan_by_id = dict(zip(identities, plans, strict=True))
        tasks = {
            identity: asyncio.create_task(
                adapter.run(
                    plan_by_id[identity],
                    reservations[identity],
                    buffers[identity],
                ),
                name=f"{adapter.task_name_prefix}:{identity}",
            )
            for identity in identities
        }
        try:
            outcomes = await _join_fanout_tasks(
                tasks,
                plans=plan_by_id,
                failure_policy=failure_policy,
                adapter=adapter,
            )
        except BaseException:
            await _cancel_and_join(tasks.values())
            context.events.merge_many(tuple(buffers[identity] for identity in identities))
            raise
        context.events.merge_many(tuple(buffers[identity] for identity in identities))
        _raise_fanout_error(
            outcomes,
            failure_policy=failure_policy,
            adapter=adapter,
        )
        return outcomes
    finally:
        await _cancel_and_join(tasks.values())
        for reservation in reservations.values():
            reservation.close()
        aggregate.close()


async def _join_fanout_tasks(
    tasks: Mapping[str, asyncio.Task[_OutcomeT]],
    *,
    plans: Mapping[str, _PlanT],
    failure_policy: str,
    adapter: _FanoutAdapter[_PlanT, _OutcomeT],
) -> tuple[_OutcomeT, ...]:
    pending: set[asyncio.Task[_OutcomeT]] = set(tasks.values())
    while pending:
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )
        completed = tuple(task.result() for task in done if not task.cancelled() and task.exception() is None)
        errors = tuple(error for outcome in completed for error in (adapter.error(outcome),) if error is not None)
        terminal = any(
            isinstance(
                error,
                (CancellationError, BudgetExhausted, asyncio.CancelledError),
            )
            for error in errors
        )
        unexpected_failure = any(_task_exception(task) is not None for task in done)
        if terminal or unexpected_failure or (failure_policy == "fail_fast" and errors):
            await _cancel_and_join(pending)
            break
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    outcomes: list[_OutcomeT] = []
    for identity, result in zip(tasks, results, strict=True):
        if isinstance(result, BaseException):
            outcomes.append(adapter.from_exception(plans[identity], result))
        else:
            outcomes.append(result)
    return tuple(outcomes)


def _raise_fanout_error(
    outcomes: tuple[_OutcomeT, ...],
    *,
    failure_policy: str,
    adapter: _FanoutAdapter[_PlanT, _OutcomeT],
) -> None:
    errors = tuple(error for outcome in outcomes for error in (adapter.error(outcome),) if error is not None)
    terminal = next(
        (error for error in errors if isinstance(error, (CancellationError, BudgetExhausted))),
        None,
    )
    if terminal is not None:
        raise terminal
    if errors and failure_policy == "fail_fast":
        ordinary = next(
            (error for error in errors if not isinstance(error, asyncio.CancelledError)),
            None,
        )
        if ordinary is not None:
            raise ordinary
        raise errors[0]
    cancelled = next(
        (error for error in errors if isinstance(error, asyncio.CancelledError)),
        None,
    )
    if cancelled is not None:
        raise cancelled


async def _cancel_and_join(tasks: Iterable[asyncio.Task[object]]) -> None:
    task_values = tuple(tasks)
    for task in task_values:
        if not task.done():
            task.cancel()
    if task_values:
        await asyncio.gather(*task_values, return_exceptions=True)


def _task_exception(task: asyncio.Task[object]) -> BaseException | None:
    if task.cancelled():
        return asyncio.CancelledError()
    return task.exception()


def _branch_value(outcome: _BranchOutcome) -> Mapping[str, object]:
    if outcome.result is not None:
        return MappingProxyType(
            {
                "branch_id": outcome.branch_id,
                "status": "succeeded",
                "values": dict(outcome.result.values),
            }
        )
    return MappingProxyType(
        {
            "branch_id": outcome.branch_id,
            "status": "failed",
            "values": {},
            "failure_category": _error_category(outcome.error),
        }
    )


def _candidate_value(outcome: _CandidateOutcome) -> Mapping[str, object]:
    value: dict[str, object] = {
        "candidate_id": outcome.plan.candidate_id,
        "provenance": outcome.plan.provenance,
    }
    if outcome.result is not None:
        value.update(
            {
                "status": "succeeded",
                "values": dict(outcome.result.values),
            }
        )
    else:
        value.update(
            {
                "status": "failed",
                "values": {},
                "failure_category": _error_category(outcome.error),
            }
        )
    return MappingProxyType(value)


def _rank_candidate(
    raw_candidate: object,
    *,
    judge_path: str,
    definition_id: str,
    event_by_id: Mapping[str, object],
) -> Mapping[str, object]:
    candidate = _mapping(raw_candidate, "candidate selection entry")
    candidate_id = _string(candidate.get("candidate_id"), "candidate selection ID")
    base: dict[str, object] = {
        "candidate_id": candidate_id,
        "score": None,
        "rank_status": "candidate_failed",
    }
    if candidate.get("status") != "succeeded":
        return MappingProxyType(base)
    values = _mapping(candidate.get("values"), "candidate values")
    try:
        raw_judge = resolve_reference(judge_path, values)
    except KeyError:
        base["rank_status"] = "missing_judge"
        return MappingProxyType(base)
    judge = _mapping(raw_judge, "candidate judge evidence")
    if judge.get("status") != "succeeded":
        base["rank_status"] = "judge_failed"
        return MappingProxyType(base)
    score = judge.get("score")
    event_id = judge.get("event_id")
    result_id = judge.get("result_id")
    event = event_by_id.get(event_id) if type(event_id) is str else None
    valid_event = (
        getattr(event, "event_type", None) == "judge.completed"
        and getattr(event, "branch_id", None) == candidate_id
        and getattr(event, "payload", {}).get("definition_id") == definition_id
        and getattr(event, "payload", {}).get("result_id") == result_id
        and getattr(event, "payload", {}).get("score") == score
    )
    if (
        judge.get("definition_id") != definition_id
        or type(score) not in (int, float)
        or type(result_id) is not str
        or not valid_event
    ):
        base["rank_status"] = "invalid_judge_evidence"
        return MappingProxyType(base)
    base.update(
        {
            "score": score,
            "rank_status": "eligible",
            "event_id": event_id,
            "result_id": result_id,
        }
    )
    return MappingProxyType(base)


def _ranking_key(item: Mapping[str, object]) -> tuple[int, float, str]:
    eligible = item.get("rank_status") == "eligible"
    score = item.get("score")
    return (
        0 if eligible else 1,
        -float(score) if eligible and type(score) in (int, float) else 0,
        str(item["candidate_id"]),
    )


def _step_capacity(
    step: StepDefinition,
    state: WorkflowState,
    context: ExecutionContext,
    workflow: WorkflowDefinition,
    seen: tuple[str, ...] = (),
) -> dict[str, int | float]:
    if step.id in seen:
        raise ValueError("advanced reservation targets must not recurse")
    path = (*seen, step.id)
    attempts = _positive_limit(
        context.spec.effective_limits.get("max_operation_attempts", 1),
        "max_operation_attempts",
        minimum=1,
    )
    if step.type == "model_call":
        return {
            "model_calls": attempts,
            "tokens": attempts * _model_token_capacity(step, state, context),
        }
    if step.type == "judge":
        return {
            "judge_calls": 1,
            "model_calls": attempts,
            "tokens": attempts * _judge_token_capacity(step, state, context),
        }
    if step.type == "target_call":
        return {"target_calls": 1}
    if step.type == "tool_call":
        return {"tool_calls": 1}
    if step.type == "parallel":
        raw_branches = _tuple(step.options.get("branches"), "parallel branches")
        default_capacity = _capacity_mapping(
            step.options.get("per_branch_budget", {}),
            "parallel per_branch_budget",
        )
        capacities: list[Mapping[str, int | float]] = []
        for raw in raw_branches:
            branch = _mapping(raw, "parallel branch")
            branch_id = _string(branch.get("id"), "parallel branch ID")
            target_id = _string(branch.get("step"), "parallel branch step")
            target = workflow.steps.get(target_id)
            if target is None:
                raise ValueError(f"parallel branch step '{target_id}' does not exist")
            branch_values = _mapping(
                branch.get("values", {}),
                "parallel branch values",
            )
            child_state = state.fork(
                f"{state.scope_id}/{branch_id}" if state.scope_id else branch_id,
                values={
                    **dict(branch_values),
                    "branch": {"id": branch_id},
                },
            )
            inferred = _step_capacity(
                target,
                child_state,
                context,
                workflow,
                path,
            )
            declared = {
                **default_capacity,
                **_capacity_mapping(
                    branch.get("budget", {}),
                    "parallel branch budget",
                ),
            }
            capacities.append(_bounded_capacity(inferred, declared))
        return _sum_capacities(capacities)
    if step.type == "population_map":
        has_inline = "candidates" in step.options
        has_reference = "candidates_from" in step.options
        if has_inline == has_reference:
            raise ValueError("population_map requires exactly one of candidates or candidates_from")
        source = (
            step.options["candidates"]
            if has_inline
            else resolve_reference(
                _string(step.options["candidates_from"], "candidates_from"),
                state.values,
            )
        )
        target_id = _string(step.options.get("step"), "population candidate step")
        target = workflow.steps.get(target_id)
        if target is None:
            raise ValueError(f"population candidate step '{target_id}' does not exist")
        default_capacity = _capacity_mapping(
            step.options.get("per_candidate_budget", {}),
            "population per_candidate_budget",
        )
        capacities = []
        for raw in _tuple(source, "population candidates"):
            candidate = _mapping(raw, "population candidate")
            candidate_id = _string(candidate.get("id"), "candidate ID")
            if "value" not in candidate:
                raise ValueError(f"candidate '{candidate_id}' requires a value")
            provenance = _mapping(
                candidate.get("provenance", {}),
                "candidate provenance",
            )
            child_state = state.fork(
                f"{state.scope_id}/{candidate_id}" if state.scope_id else candidate_id,
                values={
                    "candidate": {
                        "id": candidate_id,
                        "value": candidate["value"],
                        "provenance": provenance,
                    }
                },
            )
            inferred = _step_capacity(
                target,
                child_state,
                context,
                workflow,
                path,
            )
            declared = {
                **default_capacity,
                **_capacity_mapping(
                    candidate.get("budget", {}),
                    "candidate budget",
                ),
            }
            capacities.append(_bounded_capacity(inferred, declared))
        return _sum_capacities(capacities)
    if step.type == "retry":
        target_id = _string(step.options.get("step"), "retry step")
        target = workflow.steps.get(target_id)
        if target is None:
            raise ValueError(f"retry target step '{target_id}' does not exist")
        attempts = _positive_limit(
            step.options.get("max_attempts"),
            "retry max_attempts",
            minimum=1,
        )
        pinned = context.spec.effective_limits.get("max_workflow_retry_attempts")
        if pinned is not None:
            attempts = min(
                attempts,
                _positive_limit(
                    pinned,
                    "pinned max_workflow_retry_attempts",
                    minimum=1,
                ),
            )
        capacity = _step_capacity(target, state, context, workflow, path)
        return {resource: amount * attempts for resource, amount in capacity.items()}
    return {}


def _model_token_capacity(
    step: StepDefinition,
    state: WorkflowState,
    context: ExecutionContext,
) -> int:
    raw_messages = _tuple(step.options.get("messages"), "model messages")
    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        message = _mapping(raw, "model message")
        content = message.get("content")
        if "content_from" in message:
            content = resolve_reference(
                _string(message["content_from"], "message content_from"),
                state.values,
            )
        elif "content_ref" in message:
            content = resolve_reference(
                _string(message["content_ref"], "message content_ref"),
                state.values,
            )
        role = _string(message.get("role"), "model message role")
        if type(content) is not str:
            raise TypeError("parallel model message content must resolve to a string")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("model messages must not be empty")
    model_role = _string(step.options.get("role"), "model role")
    call_options = _mapping(step.options.get("call_options", {}), "model call_options")
    return context.models.token_capacity(
        role=model_role,
        messages=tuple(messages),
        **dict(call_options),
    )


def _judge_token_capacity(
    step: StepDefinition,
    state: WorkflowState,
    context: ExecutionContext,
) -> int:
    definition_id = _string(
        step.options.get("definition", step.options.get("definition_id")),
        "judge definition",
    )
    evidence = resolve_mapping(step.options.get("evidence", {}), state)
    return context.judges.token_capacity(
        definition_id=definition_id,
        evidence=evidence,
    )


def _bounded_capacity(
    inferred: Mapping[str, int | float],
    declared: Mapping[str, int | float],
) -> dict[str, int | float]:
    capacity = dict(inferred)
    for resource, amount in declared.items():
        if amount < inferred.get(resource, 0):
            raise ValueError(f"declared {resource} capacity is below inferred worst-case work")
        capacity[resource] = amount
    return capacity


def _sum_capacities(
    capacities: Iterable[Mapping[str, int | float]],
) -> dict[str, int | float]:
    total: dict[str, int | float] = {}
    for capacity in capacities:
        for resource, amount in capacity.items():
            total[resource] = total.get(resource, 0) + amount
    return total


def _capacity_mapping(value: object, label: str) -> dict[str, int | float]:
    mapping = _mapping(value, label)
    allowed = {
        "artifact_bytes",
        "cost",
        "judge_calls",
        "model_calls",
        "target_calls",
        "tokens",
        "tool_calls",
    }
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{label} has unknown resources {sorted(unknown)!r}")
    capacity: dict[str, int | float] = {}
    for resource, amount in mapping.items():
        if resource == "cost":
            if type(amount) not in (int, float) or amount < 0:
                raise ValueError(f"{label} cost must be a non-negative number")
        elif type(amount) is not int or amount < 0:
            raise ValueError(f"{label} {resource} must be a non-negative built-in integer")
        capacity[resource] = amount
    return capacity


def _effective_limit(
    context: ExecutionContext,
    *,
    option: object,
    pinned_name: str,
    hard: int,
) -> int:
    limits = [hard]
    if option is not None:
        limits.append(_positive_limit(option, pinned_name, minimum=0))
    pinned = context.spec.effective_limits.get(pinned_name)
    if pinned is not None:
        limits.append(_positive_limit(pinned, f"pinned {pinned_name}", minimum=0))
    return min(limits)


def _positive_limit(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        comparator = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{label} must be a {comparator} built-in integer")
    return value


def _error_category(error: BaseException | None) -> str:
    if isinstance(error, WorkflowNodeError):
        return error.category
    if isinstance(error, BudgetExhausted):
        return "budget"
    if isinstance(error, (CancellationError, asyncio.CancelledError)):
        return "cancellation"
    return "workflow"


def _causal_step_event_id(
    context: ExecutionContext,
    parent_operation: str,
) -> str:
    event = next(
        (
            item
            for item in reversed(context.events.events())
            if item.event_type == "step.started" and item.operation_id == parent_operation
        ),
        None,
    )
    if event is None:
        raise WorkflowNodeError(
            "evidence",
            "advanced workflow node has no causal step event",
        )
    return event.event_id


def _checkpoint_id(step: StepDefinition, state: WorkflowState) -> str:
    reference = _string(
        step.options.get("checkpoint_from", step.options.get("checkpoint")),
        "backtrack checkpoint reference",
    )
    value = resolve_reference(reference, state.values)
    if type(value) in (dict, MappingProxyType):
        value = value.get("checkpoint_id")
    return _string(value, "backtrack checkpoint ID")


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


def _output_name(step: StepDefinition, default: str) -> str:
    return _string(step.options.get("output", default), "node output")


__all__ = [
    "MAX_BACKTRACKS",
    "MAX_PARALLEL_BRANCHES",
    "MAX_POPULATION_SIZE",
    "MAX_ROUNDS",
    "BacktrackHandler",
    "CandidateSelectHandler",
    "LoopHandler",
    "ParallelHandler",
    "PopulationMapHandler",
]
