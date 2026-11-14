"""Immutable, bounded definitions for declarative technique workflows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Collection, Mapping

import yaml
from yaml.events import (
    AliasEvent,
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
)

from app.techniques.workflow.conditions import (
    MAX_CONDITION_DEPTH,
    MAX_CONDITION_NODES,
    MAX_JSON_COLLECTION_ITEMS,
    MAX_JSON_NODES,
    MAX_JSON_STRING_LENGTH,
    SUPPORTED_CONDITION_OPERATORS,
    Condition,
    _MISSING,
    _extract_builtin_dict,
    _freeze_json_mapping,
    _require_builtin_string,
)

if TYPE_CHECKING:
    from app.techniques.core.histories import HistoryService


JsonPath = tuple[str | int, ...]
MAX_STEP_VISITS = 10_000
MAX_WORKFLOW_STEPS = 10_000
_MAX_YAML_DEPTH = 128
REGISTERED_NODE_TYPES = frozenset(
    {
        "artifact_create",
        "backtrack",
        "branch",
        "candidate_select",
        "checkpoint",
        "convert",
        "detector",
        "finish",
        "history_append",
        "history_create",
        "history_snapshot",
        "input_materialize",
        "judge",
        "loop",
        "model_call",
        "parallel",
        "population_map",
        "render",
        "retry",
        "seed_resolve",
        "target_call",
        "tool_call",
        "transform",
    }
)
_FLAT_STEP_OPTIONS = frozenset(
    {
        "allow_unused_fields",
        "append_to_history",
        "definition_id",
        "evidence",
        "final_response_from",
        "messages",
        "official_judge_from",
        "output",
        "provenance",
        "success_threshold",
        "template_asset",
        "template_from",
    }
)


@dataclass(frozen=True, slots=True)
class WorkflowDiagnostic:
    stage: str
    path: JsonPath
    message: str

    def __post_init__(self) -> None:
        _require_builtin_string(self.stage, "diagnostic stage", allow_empty=False)
        _require_builtin_string(self.message, "diagnostic message", allow_empty=False)
        if type(self.path) is not tuple:
            raise TypeError("diagnostic path must be a tuple")
        for part in self.path:
            if type(part) is str:
                _require_builtin_string(part, "diagnostic path field", allow_empty=False)
            elif type(part) is not int or part < 0:
                raise TypeError("diagnostic path components must be built-in strings or non-negative integers")


class WorkflowValidationError(ValueError):
    def __init__(self, diagnostics: tuple[WorkflowDiagnostic, ...]) -> None:
        self.diagnostics = tuple(sorted(diagnostics, key=_diagnostic_sort_key))
        super().__init__("; ".join(_render_diagnostic(item) for item in self.diagnostics))


@dataclass(frozen=True, slots=True)
class BranchDefinition:
    next_step_id: str
    condition: Condition | None = None

    def __post_init__(self) -> None:
        _require_builtin_string(
            self.next_step_id,
            "branch next step ID",
            allow_empty=False,
        )
        if self.condition is not None and type(self.condition) is not Condition:
            raise TypeError("branch condition must be a Condition or None")

    @property
    def next(self) -> str:
        return self.next_step_id


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class StepDefinition:
    id: str
    type: str
    next_step_id: str | None = None
    error_step_id: str | None = None
    branches: tuple[BranchDefinition, ...] = ()
    options: Mapping[str, object] = field(default_factory=_empty_mapping)
    _error_source_field: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_builtin_string(self.id, "step ID", allow_empty=False)
        _require_builtin_string(self.type, "step type", allow_empty=False)
        for name in ("next_step_id", "error_step_id"):
            value = getattr(self, name)
            if value is not None:
                _require_builtin_string(value, name, allow_empty=False)
        error_source = self._error_source_field
        if self.error_step_id is not None and error_source is None:
            error_source = "error"
        if error_source is not None:
            _require_builtin_string(error_source, "error source field", allow_empty=False)
            if error_source not in {"error", "on_error"}:
                raise ValueError("error source field must be 'error' or 'on_error'")
            if self.error_step_id is None:
                raise ValueError("error source field requires an error step ID")
        object.__setattr__(self, "_error_source_field", error_source)
        if type(self.branches) is not tuple:
            raise TypeError("step branches must be a tuple")
        if any(type(branch) is not BranchDefinition for branch in self.branches):
            raise TypeError("step branches must be BranchDefinition values")
        object.__setattr__(self, "options", _freeze_json_mapping(self.options))

    @property
    def next(self) -> str | None:
        return self.next_step_id

    @property
    def error(self) -> str | None:
        return self.error_step_id

    @property
    def error_source_field(self) -> str | None:
        return self._error_source_field


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    version: int
    start: str
    steps: Mapping[str, StepDefinition]
    max_step_visits: int = 1
    diagnostics: tuple[WorkflowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("workflow version must be integer 1")
        try:
            _require_builtin_string(self.start, "workflow start", allow_empty=False)
        except TypeError as error:
            raise ValueError(str(error)) from error
        _validate_max_step_visits(self.max_step_visits)
        if type(self.steps) not in (dict, MappingProxyType):
            raise TypeError("workflow steps must be a mapping")
        steps: dict[str, StepDefinition] = {}
        for step_id, step in self.steps.items():
            _require_builtin_string(step_id, "workflow step key", allow_empty=False)
            if type(step) is not StepDefinition:
                raise TypeError("workflow steps must be StepDefinition values")
            if step_id != step.id:
                raise ValueError("workflow step keys must match StepDefinition.id")
            steps[step_id] = step
        if len(steps) > MAX_WORKFLOW_STEPS:
            raise ValueError(f"maximum workflow step count of {MAX_WORKFLOW_STEPS} exceeded")
        if type(self.diagnostics) is not tuple or any(
            type(item) is not WorkflowDiagnostic for item in self.diagnostics
        ):
            raise TypeError("workflow diagnostics must be a tuple of WorkflowDiagnostic values")
        object.__setattr__(self, "steps", MappingProxyType(steps))


class WorkflowVisitLimitError(RuntimeError):
    pass


class _CounterLedger:
    def __init__(self, counts: Mapping[str, int] | None = None) -> None:
        self._counts = dict(counts or {})
        self._lock = threading.Lock()

    def increment(
        self,
        key: str,
        maximum: int,
        *,
        error: str,
        amount: int = 1,
    ) -> int:
        with self._lock:
            count = self._counts.get(key, 0) + amount
            if count > maximum:
                raise WorkflowVisitLimitError(error)
            self._counts[key] = count
            return count

    def validate(self, key: str, maximum: int, *, error: str, amount: int = 1) -> None:
        with self._lock:
            if self._counts.get(key, 0) + amount > maximum:
                raise WorkflowVisitLimitError(error)

    def fork(self) -> _CounterLedger:
        with self._lock:
            return _CounterLedger(self._counts)

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(dict(self._counts))


@dataclass(slots=True, init=False)
class WorkflowState:
    """Mutable JSON-safe execution state; definitions remain immutable."""

    current_step_id: str | None
    values: Mapping[str, object]
    visit_counts: Mapping[str, int]
    control_counts: Mapping[str, int]
    scope_id: str | None
    _max_step_visits: int
    _visit_ledger: _CounterLedger
    _control_ledger: _CounterLedger

    def __init__(
        self,
        current_step_id: str | None,
        values: Mapping[str, object] | None = None,
        visit_counts: Mapping[str, int] | None = None,
        *,
        max_step_visits: int = 1,
        control_counts: Mapping[str, int] | None = None,
        scope_id: str | None = None,
        _visit_ledger: _CounterLedger | None = None,
        _control_ledger: _CounterLedger | None = None,
    ) -> None:
        self.current_step_id = current_step_id
        self._max_step_visits = max_step_visits
        self.values = {} if values is None else values
        self.visit_counts = {} if visit_counts is None else visit_counts
        self.control_counts = {} if control_counts is None else control_counts
        self.scope_id = scope_id
        object.__setattr__(
            self,
            "_visit_ledger",
            _visit_ledger or _CounterLedger(self.visit_counts),
        )
        object.__setattr__(
            self,
            "_control_ledger",
            _control_ledger or _CounterLedger(self.control_counts),
        )

    def __setattr__(self, name: str, value: object) -> None:
        if name == "current_step_id":
            if value is not None and (type(value) is not str or not value):
                raise ValueError("current_step_id must be absent or a non-empty built-in str")
        elif name == "_max_step_visits":
            if hasattr(self, "_max_step_visits"):
                raise AttributeError("max_step_visits is immutable")
            _validate_max_step_visits(value)
        elif name == "values":
            value = _freeze_json_mapping(value)  # type: ignore[arg-type]
        elif name == "visit_counts":
            value = _freeze_visit_counts(value)  # type: ignore[arg-type]
        elif name == "control_counts":
            value = _freeze_visit_counts(value)  # type: ignore[arg-type]
        elif name == "scope_id":
            if value is not None and (type(value) is not str or not value):
                raise ValueError("scope_id must be absent or a non-empty built-in str")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_max_step_visits":
            raise AttributeError("max_step_visits is immutable")
        object.__delattr__(self, name)

    @property
    def max_step_visits(self) -> int:
        return self._max_step_visits

    @property
    def visits(self) -> Mapping[str, int]:
        return self.visit_counts

    @classmethod
    def initial(cls, workflow: WorkflowDefinition) -> "WorkflowState":
        if type(workflow) is not WorkflowDefinition:
            raise TypeError("workflow must be a WorkflowDefinition")
        return cls(
            current_step_id=workflow.start,
            max_step_visits=workflow.max_step_visits,
        )

    def apply(self, values: Mapping[str, object]) -> None:
        frozen = _freeze_json_mapping(values)
        updated = dict(self.values)
        updated.update(frozen)
        self.values = updated

    def fork(
        self,
        scope_id: str,
        *,
        values: Mapping[str, object] | None = None,
    ) -> WorkflowState:
        if type(scope_id) is not str or not scope_id:
            raise ValueError("scope_id must be a non-empty built-in str")
        child = WorkflowState(
            current_step_id=self.current_step_id,
            values=self.values,
            visit_counts=self.visit_counts,
            max_step_visits=self.max_step_visits,
            control_counts=self.control_counts,
            scope_id=scope_id,
            _visit_ledger=self._visit_ledger.fork(),
            _control_ledger=self._control_ledger.fork(),
        )
        if values:
            child.apply(values)
        return child

    def increment_visit(self, step_id: str, *, maximum: int | None = None) -> int:
        if type(step_id) is not str or not step_id:
            raise ValueError("step_id must be a non-empty built-in str")
        if maximum is not None:
            try:
                _validate_max_step_visits(maximum)
            except ValueError as error:
                raise ValueError("maximum visit count must be a positive integer") from error
        effective_maximum = self.max_step_visits if maximum is None else min(self.max_step_visits, maximum)
        if effective_maximum < 1:  # pragma: no cover - _validate_max_step_visits
            raise ValueError("maximum visit count must be a positive integer")
        self._visit_ledger.increment(
            step_id,
            effective_maximum,
            error=f"visit limit exceeded for step '{step_id}'",
        )
        count = self.visit_counts.get(step_id, 0) + 1
        updated = dict(self.visit_counts)
        updated[step_id] = count
        self.visit_counts = updated
        return count

    def increment_control(
        self,
        name: str,
        *,
        maximum: int,
        amount: int = 1,
    ) -> int:
        if type(name) is not str or not name:
            raise ValueError("control counter name must be a non-empty built-in str")
        if type(maximum) is not int or maximum < 0:
            raise ValueError("control counter maximum must be a non-negative built-in integer")
        if type(amount) is not int or amount < 1:
            raise ValueError("control counter amount must be a positive built-in integer")
        self._control_ledger.increment(
            name,
            maximum,
            error=f"control limit exceeded for '{name}'",
            amount=amount,
        )
        count = self.control_counts.get(name, 0) + amount
        updated = dict(self.control_counts)
        updated[name] = count
        self.control_counts = updated
        return count

    def validate_control(
        self,
        name: str,
        *,
        maximum: int,
        amount: int = 1,
    ) -> None:
        if type(name) is not str or not name:
            raise ValueError("control counter name must be a non-empty built-in str")
        if type(maximum) is not int or maximum < 0:
            raise ValueError("control counter maximum must be a non-negative built-in integer")
        if type(amount) is not int or amount < 1:
            raise ValueError("control counter amount must be a positive built-in integer")
        self._control_ledger.validate(
            name,
            maximum,
            error=f"control limit exceeded for '{name}'",
            amount=amount,
        )

    def reconcile_checkpoint(
        self,
        *,
        position: Mapping[str, object],
        values: Mapping[str, object],
        workflow: WorkflowDefinition,
        histories: HistoryService | None = None,
        history_heads: Mapping[str, tuple[str, ...]] | None = None,
        checkpoint_id: str | None = None,
        expected_inactive: Mapping[str, tuple[str, ...]] | None = None,
        consume_control: tuple[str, int, int] | None = None,
        before_history_events: Callable[[int | None], None] | None = None,
    ) -> tuple[Mapping[str, tuple[str, ...]], int | None]:
        """Atomically restore checkpoint values and reconcile monotonic ledgers."""

        if type(workflow) is not WorkflowDefinition:
            raise TypeError("workflow must be an exact WorkflowDefinition")
        raw_position = _extract_builtin_dict(position, "checkpoint position")
        scope_id = raw_position.get("scope_id")
        if scope_id is not None and (type(scope_id) is not str or not scope_id):
            raise ValueError("checkpoint scope must be absent or a non-empty built-in string")
        if scope_id != self.scope_id:
            raise ValueError("checkpoint scope does not belong to this workflow context")
        required_position_keys = {
            "step_id",
            "next_step_id",
            "scope_id",
            "visit_counts",
            "control_counts",
        }
        if set(raw_position) != required_position_keys:
            raise ValueError("checkpoint position fields do not match the workflow state schema")
        step_id = raw_position["step_id"]
        if type(step_id) is not str or step_id not in workflow.steps:
            raise ValueError("checkpoint position does not belong to this workflow")
        next_step_id = raw_position["next_step_id"]
        if next_step_id is not None and (type(next_step_id) is not str or next_step_id not in workflow.steps):
            raise ValueError("checkpoint next position does not belong to this workflow")

        checkpoint_visits = _checkpoint_counter_mapping(
            raw_position["visit_counts"],
            label="visit",
        )
        checkpoint_controls = _checkpoint_counter_mapping(
            raw_position["control_counts"],
            label="control",
        )
        _validate_checkpoint_visit_owners(checkpoint_visits, workflow)
        _validate_checkpoint_control_owners(checkpoint_controls, workflow)
        _validate_checkpoint_visit_owners(self.visit_counts, workflow)
        _validate_checkpoint_control_owners(self.control_counts, workflow)

        private_visits = self._visit_ledger.snapshot()
        private_controls = self._control_ledger.snapshot()
        if private_visits != self.visit_counts:
            raise ValueError("private visit ledger does not match its public projection")
        if private_controls != self.control_counts:
            raise ValueError("private control ledger does not match its public projection")

        restored_values = _freeze_json_mapping(values)
        merged_visits = _merge_monotonic_counts(
            self.visit_counts,
            checkpoint_visits,
        )
        if any(count > self.max_step_visits for count in merged_visits.values()):
            raise ValueError("checkpoint visit count exceeds the workflow visit limit")
        merged_controls = dict(
            _merge_monotonic_counts(
                self.control_counts,
                checkpoint_controls,
            )
        )
        consumed_count: int | None = None
        if consume_control is not None:
            if type(consume_control) is not tuple or len(consume_control) != 3:
                raise TypeError("checkpoint control consumption must be a three-item tuple")
            name, maximum, amount = consume_control
            if type(name) is not str or not name:
                raise ValueError("control counter name must be a non-empty built-in str")
            if type(maximum) is not int or maximum < 0:
                raise ValueError("control counter maximum must be a non-negative built-in integer")
            if type(amount) is not int or amount < 1:
                raise ValueError("control counter amount must be a positive built-in integer")
            _validate_checkpoint_control_owners({name: 0}, workflow)
            consumed_count = merged_controls.get(name, 0) + amount
            if consumed_count > maximum:
                raise WorkflowVisitLimitError(f"control limit exceeded for '{name}'")
            merged_controls[name] = consumed_count
        frozen_visits = _freeze_visit_counts(dict(merged_visits))
        frozen_controls = _freeze_visit_counts(merged_controls)
        new_visit_ledger = _CounterLedger(frozen_visits)
        new_control_ledger = _CounterLedger(frozen_controls)

        history_arguments = (
            histories,
            history_heads,
            checkpoint_id,
            expected_inactive,
        )
        if any(item is not None for item in history_arguments) and any(item is None for item in history_arguments):
            raise ValueError("checkpoint history reconciliation requires complete evidence")
        if before_history_events is not None and histories is None:
            raise ValueError("checkpoint pre-restore events require history reconciliation")

        def commit() -> None:
            object.__setattr__(self, "values", restored_values)
            object.__setattr__(self, "visit_counts", frozen_visits)
            object.__setattr__(self, "control_counts", frozen_controls)
            object.__setattr__(self, "_visit_ledger", new_visit_ledger)
            object.__setattr__(self, "_control_ledger", new_control_ledger)

        inactive: Mapping[str, tuple[str, ...]] = MappingProxyType({})
        if histories is None:
            commit()
        else:
            inactive = histories.reconcile_checkpoint_heads(
                history_heads,  # type: ignore[arg-type]
                checkpoint_id=checkpoint_id,  # type: ignore[arg-type]
                expected_inactive=expected_inactive,
                commit=commit,
                before_events=(
                    None if before_history_events is None else lambda: before_history_events(consumed_count)
                ),
            )
        return inactive, consumed_count


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    pass


@dataclass(slots=True)
class _YamlCollectionFrame:
    is_mapping: bool
    item_count: int = 0
    expecting_key: bool = True


def _construct_mapping(
    loader: _DuplicateKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise yaml.YAMLError("workflow YAML mapping keys must be strings")
        if key in mapping:
            raise yaml.YAMLError(f"duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_workflow(
    source: Mapping[str, object] | str | bytes | Path,
    *,
    allowed_node_types: Collection[str] | None = None,
) -> WorkflowDefinition:
    """Load and semantically validate an immutable workflow contract.

    A mapping is treated as already parsed data; strings and bytes are YAML
    documents, while ``Path`` values are safe-loaded from an asset on disk.
    """

    payload = _load_payload(source)
    try:
        frozen_payload = _freeze_json_mapping(payload)
    except (TypeError, ValueError) as error:
        _raise("structure", (), str(error))
    allowed = _normalize_node_types(allowed_node_types)
    return _build_workflow(frozen_payload, allowed)


def _load_payload(source: Mapping[str, object] | str | bytes | Path) -> Mapping[str, object]:
    if type(source) in (dict, MappingProxyType):
        return source
    if isinstance(source, Path):
        try:
            content: str | bytes = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            _raise("yaml", (), f"cannot read workflow YAML: {error}")
    elif type(source) in (str, bytes):
        content = source
    else:  # pragma: no cover - protected by the public type annotation
        raise TypeError("workflow source must be a mapping, YAML text, bytes, or Path")
    _validate_yaml_events(content)
    try:
        payload = yaml.load(content, Loader=_DuplicateKeySafeLoader)
    except RecursionError:
        _raise("yaml", (), "invalid workflow YAML: parser recursion limit exceeded")
    except (ValueError, OverflowError, yaml.YAMLError) as error:
        _raise("yaml", (), f"invalid workflow YAML: {error}")
    if type(payload) is not dict:
        _raise("structure", (), "workflow root must be a mapping with string keys")
    return payload


def _validate_yaml_events(content: str | bytes) -> None:
    collections: list[_YamlCollectionFrame] = []
    node_count = 0
    try:
        for event in yaml.parse(content, Loader=_DuplicateKeySafeLoader):
            if isinstance(event, AliasEvent):
                _raise("yaml", (), "YAML aliases are not allowed")
            is_collection_start = isinstance(event, (MappingStartEvent, SequenceStartEvent))
            if isinstance(event, ScalarEvent) or is_collection_start:
                node_count += 1
                if node_count > MAX_JSON_NODES:
                    _raise(
                        "yaml",
                        (),
                        f"maximum YAML node count of {MAX_JSON_NODES} exceeded",
                    )
                _count_yaml_collection_item(collections)
            if isinstance(event, ScalarEvent) and len(event.value) > MAX_JSON_STRING_LENGTH:
                _raise(
                    "yaml",
                    (),
                    f"maximum YAML scalar length of {MAX_JSON_STRING_LENGTH} exceeded",
                )
            if is_collection_start:
                depth = len(collections) + 1
                if depth > _MAX_YAML_DEPTH:
                    _raise(
                        "yaml",
                        (),
                        f"maximum YAML depth of {_MAX_YAML_DEPTH} exceeded",
                    )
                collections.append(_YamlCollectionFrame(is_mapping=isinstance(event, MappingStartEvent)))
            elif isinstance(event, (MappingEndEvent, SequenceEndEvent)):
                collections.pop()
    except RecursionError:
        _raise("yaml", (), "invalid workflow YAML: parser recursion limit exceeded")
    except yaml.YAMLError as error:
        _raise("yaml", (), f"invalid workflow YAML: {error}")


def _count_yaml_collection_item(collections: list[_YamlCollectionFrame]) -> None:
    if not collections:
        return
    frame = collections[-1]
    if frame.is_mapping:
        if frame.expecting_key:
            frame.item_count += 1
        frame.expecting_key = not frame.expecting_key
    else:
        frame.item_count += 1
    if frame.item_count > MAX_JSON_COLLECTION_ITEMS:
        _raise(
            "yaml",
            (),
            f"maximum YAML collection size of {MAX_JSON_COLLECTION_ITEMS} exceeded",
        )


def _normalize_node_types(allowed_node_types: Collection[str] | None) -> frozenset[str]:
    if allowed_node_types is None:
        return REGISTERED_NODE_TYPES
    allowed_values = tuple(allowed_node_types)
    if not all(type(item) is str and item for item in allowed_values):
        raise ValueError("allowed workflow node types must be non-empty strings")
    return frozenset(allowed_values)


def _build_workflow(
    payload: Mapping[str, object],
    allowed_node_types: frozenset[str],
) -> WorkflowDefinition:
    _reject_unknown_keys(payload, {"version", "start", "steps", "max_step_visits"}, ())
    version = payload.get("version")
    if isinstance(version, bool) or type(version) is not int or version != 1:
        _raise("schema", ("version",), "workflow version must be integer 1")
    start = _required_id(payload, "start", ())
    steps_value = payload.get("steps")
    if not isinstance(steps_value, tuple):
        _raise("schema", ("steps",), "workflow steps must be a list")
    if not steps_value:
        _raise("schema", ("steps",), "workflow steps must not be empty")
    if len(steps_value) > MAX_WORKFLOW_STEPS:
        _raise(
            "schema",
            ("steps",),
            f"maximum workflow step count of {MAX_WORKFLOW_STEPS} exceeded",
        )
    bound_declared = "max_step_visits" in payload
    max_step_visits = _max_step_visits(payload.get("max_step_visits", 1))

    steps, source_indexes = _parse_steps(steps_value, allowed_node_types)
    _validate_targets(steps, source_indexes)
    if start not in steps:
        _raise("graph", ("start",), f"unknown start step '{start}'")
    if _has_cycle(steps, tuple(steps)) and not bound_declared:
        _raise("graph", (), "cycle requires an explicit max_step_visits")
    diagnostics = _unreachable_diagnostics(steps, source_indexes, start)
    return WorkflowDefinition(
        version=version,
        start=start,
        steps=steps,
        max_step_visits=max_step_visits,
        diagnostics=diagnostics,
    )


def _parse_steps(
    values: tuple[object, ...],
    allowed_node_types: frozenset[str],
) -> tuple[dict[str, StepDefinition], Mapping[str, int]]:
    steps: dict[str, StepDefinition] = {}
    indexes: dict[str, int] = {}
    for index, raw_step in enumerate(values):
        path = ("steps", index)
        step = _required_mapping(raw_step, path, "step must be a mapping")
        _reject_unknown_keys(
            step,
            {
                "id",
                "type",
                "next",
                "error",
                "on_error",
                "branches",
                "options",
                "config",
                *_FLAT_STEP_OPTIONS,
            },
            path,
        )
        step_id = _required_id(step, "id", path)
        if step_id in steps:
            _raise("graph", (*path, "id"), f"duplicate step ID '{step_id}'")
        node_type = _required_id(step, "type", path)
        if node_type not in allowed_node_types:
            _raise("schema", (*path, "type"), f"unsupported workflow node type '{node_type}'")
        if "error" in step and "on_error" in step:
            _raise("schema", path, "a step may declare either error or on_error, not both")
        next_step_id = _optional_target(step, "next", path)
        error_field = "error" if "error" in step else "on_error" if "on_error" in step else None
        error_step_id = _optional_target(step, error_field, path) if error_field is not None else None
        branches = _parse_branches(step["branches"], (*path, "branches")) if "branches" in step else ()
        options = _parse_options(step, path)
        step_definition = StepDefinition(
            id=step_id,
            type=node_type,
            next_step_id=next_step_id,
            error_step_id=error_step_id,
            branches=branches,
            options=options,
        )
        if error_field is not None:
            # Source spelling is loader metadata, not public constructor input.
            object.__setattr__(step_definition, "_error_source_field", error_field)
        steps[step_id] = step_definition
        indexes[step_id] = index
    return steps, MappingProxyType(indexes)


def _parse_options(step: Mapping[str, object], path: JsonPath) -> Mapping[str, object]:
    if "options" in step and "config" in step:
        _raise("schema", path, "a step may declare either options or config, not both")
    nested_field = "options" if "options" in step else "config" if "config" in step else None
    flat = {key: step[key] for key in _FLAT_STEP_OPTIONS if key in step}
    if nested_field is not None and flat:
        _raise("schema", path, "a step may declare nested or flat options, not both")
    value = flat if nested_field is None else step[nested_field]
    return _required_mapping(
        value, (*path, nested_field or "options"), "step options must be a mapping",
    )


def _parse_branches(value: object, path: JsonPath) -> tuple[BranchDefinition, ...]:
    if not isinstance(value, tuple):
        _raise("schema", path, "branches must be a list")
    if not value:
        _raise("schema", path, "branches must not be empty")
    branches: list[BranchDefinition] = []
    default_seen = False
    for index, raw_branch in enumerate(value):
        branch_path = (*path, index)
        branch = _required_mapping(raw_branch, branch_path, "branch must be a mapping")
        _reject_unknown_keys(branch, {"condition", "next"}, branch_path)
        next_step_id = _required_id(branch, "next", branch_path)
        raw_condition = branch.get("condition", _MISSING)
        if raw_condition is _MISSING:
            if default_seen:
                _raise("schema", branch_path, "a branch may declare only one default target")
            default_seen = True
            if index != len(value) - 1:
                _raise("schema", branch_path, "the default branch must be last")
            condition = None
        else:
            if default_seen:
                _raise("schema", branch_path, "the default branch must be last")
            condition = _parse_condition(raw_condition, (*branch_path, "condition"))
        branches.append(BranchDefinition(next_step_id=next_step_id, condition=condition))
    return tuple(branches)


def _parse_condition(
    value: object,
    path: JsonPath,
    *,
    enforce_limits: bool = True,
) -> Condition:
    if enforce_limits:
        _validate_condition_limits(value, path)
    data = _required_mapping(value, path, "condition must be a mapping")
    _reject_unknown_keys(data, {"path", "operator", "value", "conditions"}, path)
    operator = data.get("operator")
    if not isinstance(operator, str) or not operator:
        _raise("schema", (*path, "operator"), "condition operator must be a non-empty string")
    if operator not in SUPPORTED_CONDITION_OPERATORS:
        _raise("schema", (*path, "operator"), f"unsupported condition operator '{operator}'")
    raw_path = data.get("path")
    if raw_path is not None and (not isinstance(raw_path, str) or not _valid_path(raw_path)):
        _raise("schema", (*path, "path"), "condition path must be a non-empty dotted string")

    compound = operator in {"all", "any", "not"}
    if compound:
        if raw_path is not None or "value" in data:
            _raise("schema", path, f"operator '{operator}' accepts only conditions")
        children_value = data.get("conditions")
        if not isinstance(children_value, tuple) or not children_value:
            _raise("schema", path, f"operator '{operator}' requires a non-empty conditions list")
        if operator == "not" and len(children_value) != 1:
            _raise("schema", path, "operator 'not' requires exactly one condition")
        return Condition(
            operator=operator,
            conditions=tuple(
                _parse_condition(
                    item,
                    (*path, "conditions", index),
                    enforce_limits=False,
                )
                for index, item in enumerate(children_value)
            ),
        )

    if raw_path is None:
        _raise("schema", (*path, "path"), f"operator '{operator}' requires a path")
    if "conditions" in data:
        _raise("schema", path, f"operator '{operator}' does not accept conditions")
    requires_value = operator in {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"}
    if requires_value and "value" not in data:
        _raise("schema", (*path, "value"), f"operator '{operator}' requires a value")
    if not requires_value and "value" in data:
        _raise("schema", (*path, "value"), f"operator '{operator}' does not accept a value")
    return Condition(
        path=raw_path,
        operator=operator,
        value=data["value"] if "value" in data else _MISSING,
    )


def _validate_condition_limits(value: object, path: JsonPath) -> None:
    pending: list[tuple[object, JsonPath, int]] = [(value, path, 1)]
    node_count = 0
    while pending:
        current, current_path, depth = pending.pop()
        node_count += 1
        if node_count > MAX_CONDITION_NODES:
            _raise(
                "schema",
                current_path,
                f"maximum condition node count of {MAX_CONDITION_NODES} exceeded",
            )
        if depth > MAX_CONDITION_DEPTH:
            _raise(
                "schema",
                current_path,
                f"maximum condition depth of {MAX_CONDITION_DEPTH} exceeded",
            )
        if type(current) not in (dict, MappingProxyType):
            continue
        operator = current.get("operator")
        if operator not in {"all", "any", "not"}:
            continue
        children = current.get("conditions")
        if type(children) is not tuple:
            continue
        pending.extend((child, (*current_path, "conditions", index), depth + 1) for index, child in enumerate(children))


def _validate_targets(steps: Mapping[str, StepDefinition], indexes: Mapping[str, int]) -> None:
    diagnostics: list[WorkflowDiagnostic] = []
    for step_id, step in steps.items():
        index = indexes[step_id]
        targets: tuple[tuple[str, str | None, JsonPath], ...] = (
            ("next", step.next_step_id, ("steps", index, "next")),
            (
                step.error_source_field or "error",
                step.error_step_id,
                ("steps", index, step.error_source_field or "error"),
            ),
            *tuple(
                ("branch", branch.next_step_id, ("steps", index, "branches", branch_index, "next"))
                for branch_index, branch in enumerate(step.branches)
            ),
        )
        for kind, target, path in targets:
            if target is not None and target not in steps:
                diagnostics.append(WorkflowDiagnostic("graph", path, f"unknown {kind} step '{target}'"))
    if diagnostics:
        raise WorkflowValidationError(tuple(diagnostics))


def _has_cycle(steps: Mapping[str, StepDefinition], ordered_ids: tuple[str, ...]) -> bool:
    unvisited, visiting, visited = 0, 1, 2
    state = {step_id: unvisited for step_id in ordered_ids}
    for root in ordered_ids:
        if state[root] != unvisited:
            continue
        state[root] = visiting
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            step_id, target_index = stack[-1]
            targets = _targets(steps[step_id])
            if target_index == len(targets):
                state[step_id] = visited
                stack.pop()
                continue
            stack[-1] = (step_id, target_index + 1)
            target = targets[target_index]
            if target is None:
                continue
            if state[target] == visiting:
                return True
            if state[target] == unvisited:
                state[target] = visiting
                stack.append((target, 0))
    return False


def _unreachable_diagnostics(
    steps: Mapping[str, StepDefinition],
    indexes: Mapping[str, int],
    start: str,
) -> tuple[WorkflowDiagnostic, ...]:
    reachable: set[str] = set()
    pending: deque[str] = deque((start,))
    while pending:
        step_id = pending.popleft()
        if step_id in reachable:
            continue
        reachable.add(step_id)
        pending.extend(target for target in _targets(steps[step_id]) if target is not None)
    return tuple(
        WorkflowDiagnostic(
            "graph",
            ("steps", indexes[step_id]),
            f"unreachable step '{step_id}'",
        )
        for step_id in steps
        if step_id not in reachable
    )


def _targets(step: StepDefinition) -> tuple[str | None, ...]:
    return (
        step.next_step_id,
        step.error_step_id,
        *(branch.next_step_id for branch in step.branches),
    )


def _max_step_visits(value: object) -> int:
    try:
        return _validate_max_step_visits(value)
    except ValueError as error:
        _raise("schema", ("max_step_visits",), str(error))


def _validate_max_step_visits(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_STEP_VISITS:
        raise ValueError(f"max_step_visits must be an integer from 1 to {MAX_STEP_VISITS}")
    return value


def _required_id(value: Mapping[str, object], key: str, path: JsonPath) -> str:
    result = value.get(key)
    if type(result) is not str or not result.strip():
        _raise("schema", (*path, key), f"{key} must be a non-empty string")
    if len(result) > MAX_JSON_STRING_LENGTH:
        _raise(
            "schema",
            (*path, key),
            f"{key} exceeds maximum JSON string length of {MAX_JSON_STRING_LENGTH}",
        )
    return result


def _optional_target(value: Mapping[str, object], key: str, path: JsonPath) -> str | None:
    if key not in value:
        return None
    result = value[key]
    if type(result) is not str or not result.strip():
        _raise("schema", (*path, key), f"{key} must be a non-empty string")
    if len(result) > MAX_JSON_STRING_LENGTH:
        _raise(
            "schema",
            (*path, key),
            f"{key} exceeds maximum JSON string length of {MAX_JSON_STRING_LENGTH}",
        )
    return result


def _required_mapping(value: object, path: JsonPath, message: str) -> Mapping[str, object]:
    if type(value) not in (dict, MappingProxyType):
        _raise("schema", path, message)
    return value


def _reject_unknown_keys(
    value: Mapping[str, object],
    allowed: set[str],
    path: JsonPath,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _raise("schema", path, f"unknown field '{unknown[0]}'")


def _valid_path(path: str) -> bool:
    return bool(path) and all(part for part in path.split("."))


def _freeze_visit_counts(value: Mapping[str, int]) -> Mapping[str, int]:
    mapping = _extract_builtin_dict(value, "workflow visit counts")
    counts: dict[str, int] = {}
    for key, count in mapping.items():
        if type(key) is not str or not key:
            raise TypeError("workflow visit count keys must be non-empty built-in strings")
        if type(count) is not int or count < 0:
            raise TypeError("workflow visit counts must be non-negative integers")
        counts[key] = count
    return MappingProxyType(counts)


def _checkpoint_counter_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, int]:
    mapping = _extract_builtin_dict(value, f"checkpoint {label} counts")  # type: ignore[arg-type]
    counts: dict[str, int] = {}
    for key, count in mapping.items():
        if type(key) is not str or not key:
            raise ValueError(f"checkpoint {label} count keys must be non-empty built-in strings")
        if type(count) is not int or count < 0:
            raise ValueError(f"checkpoint {label} counts must be non-negative built-in integers")
        counts[key] = count
    return MappingProxyType(counts)


def _validate_checkpoint_visit_owners(
    counts: Mapping[str, int],
    workflow: WorkflowDefinition,
) -> None:
    unknown = sorted(set(counts) - set(workflow.steps))
    if unknown:
        raise ValueError(f"checkpoint visit count '{unknown[0]}' does not belong to this workflow")


def _validate_checkpoint_control_owners(
    counts: Mapping[str, int],
    workflow: WorkflowDefinition,
) -> None:
    workflow_types = {step.type for step in workflow.steps.values()}
    for key in counts:
        if key == "backtracks" and "backtrack" in workflow_types:
            continue
        if key == "parallel_branches" and "parallel" in workflow_types:
            continue
        if key == "population_candidates" and "population_map" in workflow_types:
            continue
        if key.startswith("rounds:"):
            step_id = key.removeprefix("rounds:")
            step = workflow.steps.get(step_id)
            if step is not None and step.type == "loop":
                continue
        raise ValueError(f"checkpoint control count '{key}' does not belong to this workflow scope")


def _merge_monotonic_counts(
    current: Mapping[str, int],
    checkpoint: Mapping[str, int],
) -> Mapping[str, int]:
    return MappingProxyType(
        {key: max(current.get(key, 0), checkpoint.get(key, 0)) for key in sorted(set(current) | set(checkpoint))}
    )


def _diagnostic_sort_key(item: WorkflowDiagnostic) -> tuple[str, ...]:
    return tuple(str(part) for part in item.path) + (item.stage, item.message)


def _render_diagnostic(item: WorkflowDiagnostic) -> str:
    path = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in item.path)
    return f"{item.stage} {path}: {item.message}"


def _raise(stage: str, path: JsonPath, message: str) -> None:
    raise WorkflowValidationError((WorkflowDiagnostic(stage, path, message),))
