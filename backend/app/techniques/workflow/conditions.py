"""Data-only condition contracts and a fail-closed evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from gc import get_referents
from types import MappingProxyType
from typing import Mapping


MAX_JSON_DEPTH = 512
MAX_JSON_NODES = 100_000
MAX_JSON_COLLECTION_ITEMS = 20_000
MAX_JSON_STRING_LENGTH = 1_000_000
MAX_CONDITION_DEPTH = 64
MAX_CONDITION_NODES = 4_096

_MISSING = object()
_JSON_MAPPING_TYPES = (dict, MappingProxyType)
SUPPORTED_CONDITION_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "contains",
        "exists",
        "truthy",
        "all",
        "any",
        "not",
    }
)


class JsonValueLimitError(ValueError):
    """A JSON-safe workflow input exceeded a declared structural limit."""


def _require_builtin_string(
    value: object,
    field: str,
    *,
    allow_empty: bool = True,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a built-in str")
    if len(value) > MAX_JSON_STRING_LENGTH:
        raise JsonValueLimitError(f"{field} exceeds maximum JSON string length of {MAX_JSON_STRING_LENGTH}")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _freeze_json_value(value: object) -> object:
    """Copy a bounded JSON tree into immutable containers without callbacks."""

    _validate_json_value(value)
    return _freeze_valid_json_value(value)


def _extract_builtin_dict(value: object, field: str) -> dict[object, object]:
    """Return only an exact dict without invoking mapping-level hooks."""

    if type(value) is dict:
        return value
    if type(value) is MappingProxyType:
        # MappingProxyType does not expose its backing mapping. GC referents let
        # us verify and extract an exact dict without calling proxy operations
        # that could dispatch to hooks on an attacker-controlled dict subclass.
        referents = get_referents(value)
        if len(referents) == 1 and type(referents[0]) is dict:
            return referents[0]
    raise TypeError(f"{field} must be a built-in dict or a mapping proxy backed by a built-in dict")


def _validate_json_value(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    node_count = 0
    while pending:
        current, depth = pending.pop()
        node_count += 1
        if node_count > MAX_JSON_NODES:
            raise JsonValueLimitError(f"maximum JSON node count of {MAX_JSON_NODES} exceeded")
        if depth > MAX_JSON_DEPTH:
            raise JsonValueLimitError(f"maximum JSON depth of {MAX_JSON_DEPTH} exceeded")

        kind = type(current)
        if current is None or kind in (str, bool, int):
            if kind is str:
                _require_builtin_string(current, "JSON string")
            continue
        if kind is float:
            if not math.isfinite(current):
                raise TypeError("JSON-safe values must use finite numbers")
            continue
        if kind in _JSON_MAPPING_TYPES:
            mapping = _extract_builtin_dict(current, "JSON mapping")
            container_id = id(current)
            if container_id in seen_containers:
                raise TypeError("JSON-safe values must not repeat container references")
            seen_containers.add(container_id)
            if len(mapping) > MAX_JSON_COLLECTION_ITEMS:
                raise JsonValueLimitError(f"maximum JSON collection size of {MAX_JSON_COLLECTION_ITEMS} exceeded")
            for key, item in mapping.items():
                _require_builtin_string(key, "JSON mapping key")
                pending.append((item, depth + 1))
            continue
        if kind in (list, tuple):
            container_id = id(current)
            if container_id in seen_containers:
                raise TypeError("JSON-safe values must not repeat container references")
            seen_containers.add(container_id)
            if len(current) > MAX_JSON_COLLECTION_ITEMS:
                raise JsonValueLimitError(f"maximum JSON collection size of {MAX_JSON_COLLECTION_ITEMS} exceeded")
            pending.extend((item, depth + 1) for item in current)
            continue
        raise TypeError(f"value of type {kind.__name__} is not JSON-safe")


def _freeze_valid_json_value(value: object) -> object:
    """Iteratively copy a previously validated JSON tree into immutable values."""

    result: dict[str, object] = {}
    pending: list[tuple[str, object, object, object]] = [("value", value, result, "result")]
    while pending:
        action, current, parent, key = pending.pop()
        if action == "freeze_mapping":
            assert type(current) is dict  # validated internal accumulator
            assert type(parent) is dict or type(parent) is list
            parent[key] = MappingProxyType(current)  # type: ignore[index]
            continue
        if action == "freeze_sequence":
            assert type(current) is list  # validated internal accumulator
            assert type(parent) is dict or type(parent) is list
            parent[key] = tuple(current)  # type: ignore[index]
            continue

        kind = type(current)
        if current is None or kind in (str, bool, int, float):
            assert type(parent) is dict or type(parent) is list
            parent[key] = current  # type: ignore[index]
            continue
        if kind in _JSON_MAPPING_TYPES:
            mapping = _extract_builtin_dict(current, "JSON mapping")
            target: dict[str, object] = {}
            pending.append(("freeze_mapping", target, parent, key))
            pending.extend(("value", item, target, item_key) for item_key, item in reversed(tuple(mapping.items())))
            continue
        if kind in (list, tuple):
            target = [None] * len(current)
            pending.append(("freeze_sequence", target, parent, key))
            pending.extend(("value", item, target, index) for index, item in reversed(tuple(enumerate(current))))
            continue
        raise AssertionError("JSON values were validated before freezing")  # pragma: no cover

    return result["result"]


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    _extract_builtin_dict(value, "JSON mapping")
    frozen = _freeze_json_value(value)
    if type(frozen) not in _JSON_MAPPING_TYPES:  # pragma: no cover - defensive invariant
        raise TypeError("expected a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class Condition:
    """A structured predicate; it contains data, never source code."""

    path: str | None = None
    operator: str = "truthy"
    value: object = _MISSING
    conditions: tuple["Condition", ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.path is not None:
            _require_builtin_string(self.path, "condition path")
        _require_builtin_string(self.operator, "condition operator", allow_empty=False)
        if type(self.conditions) is not tuple:
            raise TypeError("condition children must be a tuple")
        if any(type(item) is not Condition for item in self.conditions):
            raise TypeError("condition children must be Condition values")
        if self.value is not _MISSING:
            object.__setattr__(self, "value", _freeze_json_value(self.value))
        _validate_condition_tree(self)

    def canonical_mapping(self) -> Mapping[str, object]:
        """Return the JSON-safe semantic fields used to bind this condition."""

        value: dict[str, object] = {"operator": self.operator}
        if self.path is not None:
            value["path"] = self.path
        if self.value is not _MISSING:
            value["value"] = self.value
        if self.conditions:
            value["conditions"] = tuple(
                condition.canonical_mapping() for condition in self.conditions
            )
        return MappingProxyType(value)


class ConditionEvaluator:
    """Evaluate :class:`Condition` objects against JSON-safe runtime values."""

    def evaluate(self, condition: Condition, values: Mapping[str, object]) -> bool:
        if type(condition) is not Condition:
            raise TypeError("condition must be a Condition")
        if type(values) not in _JSON_MAPPING_TYPES:
            return False
        try:
            frozen_values = _freeze_json_mapping(values)
        except (TypeError, ValueError):
            return False
        return self._evaluate(condition, frozen_values)

    def _evaluate(self, condition: Condition, values: Mapping[str, object]) -> bool:
        operator = condition.operator
        if operator not in SUPPORTED_CONDITION_OPERATORS:
            raise ValueError(f"unsupported condition operator '{operator}'")

        if operator in {"all", "any", "not"}:
            return self._evaluate_compound(condition, values)

        if condition.path is None or not _valid_path(condition.path):
            raise ValueError(f"condition operator '{operator}' requires a valid path")
        found, actual = _resolve_path(values, condition.path)
        if not found:
            return False
        if operator == "exists":
            return True
        if operator == "truthy":
            return _json_truthy(actual)
        if condition.value is _MISSING:
            raise ValueError(f"condition operator '{operator}' requires a value")

        if operator == "eq":
            return _json_equal(actual, condition.value)
        if operator == "ne":
            return not _json_equal(actual, condition.value)
        if operator in {"gt", "gte", "lt", "lte"}:
            return _ordered_comparison(operator, actual, condition.value)
        if operator == "in":
            return _contains(condition.value, actual)
        if operator == "contains":
            return _contains(actual, condition.value)
        raise AssertionError("condition operator dispatch is incomplete")  # pragma: no cover

    def _evaluate_compound(self, condition: Condition, values: Mapping[str, object]) -> bool:
        if condition.path is not None or condition.value is not _MISSING:
            raise ValueError(f"compound condition operator '{condition.operator}' accepts only conditions")
        children = condition.conditions
        if condition.operator == "not":
            if len(children) != 1:
                raise ValueError("condition operator 'not' requires exactly one child")
            return not self._evaluate(children[0], values)
        if not children:
            raise ValueError(f"condition operator '{condition.operator}' requires at least one child")
        if condition.operator == "all":
            return all(self._evaluate(child, values) for child in children)
        return any(self._evaluate(child, values) for child in children)


def _validate_condition_tree(root: Condition) -> None:
    pending: list[tuple[Condition, int]] = [(root, 1)]
    node_count = 0
    while pending:
        current, depth = pending.pop()
        node_count += 1
        if node_count > MAX_CONDITION_NODES:
            raise JsonValueLimitError(f"maximum condition node count of {MAX_CONDITION_NODES} exceeded")
        if depth > MAX_CONDITION_DEPTH:
            raise JsonValueLimitError(f"maximum condition depth of {MAX_CONDITION_DEPTH} exceeded")
        pending.extend((child, depth + 1) for child in current.conditions)


def _valid_path(path: str) -> bool:
    return bool(path) and all(segment for segment in path.split("."))


def _resolve_path(values: Mapping[str, object], path: str) -> tuple[bool, object]:
    current: object = values
    for segment in path.split("."):
        if type(current) in _JSON_MAPPING_TYPES:
            if segment not in current:
                return False, _MISSING
            current = current[segment]
            continue
        if type(current) is tuple:
            index = _list_index(segment)
            if index is None or index >= len(current):
                return False, _MISSING
            current = current[index]
            continue
        return False, _MISSING
    return True, current


def _list_index(segment: str) -> int | None:
    if not segment or not segment.isascii() or not segment.isdecimal():
        return None
    if len(segment) > 1 and segment.startswith("0"):
        return None
    return int(segment)


def _json_truthy(value: object) -> bool:
    if value is None or value is False:
        return False
    if type(value) is bool:
        return value
    if type(value) in (int, float):
        return value != 0
    if type(value) is str or type(value) is tuple or type(value) in _JSON_MAPPING_TYPES:
        return len(value) > 0
    return False  # pragma: no cover - inputs were frozen before dispatch


def _json_equal(left: object, right: object) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left is right
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is str or left is None:
        return type(left) is type(right) and left == right
    if type(left) is tuple and type(right) is tuple:
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) in _JSON_MAPPING_TYPES and type(right) in _JSON_MAPPING_TYPES:
        return (
            len(left) == len(right)
            and set(left) == set(right)
            and all(_json_equal(left[key], right[key]) for key in left)
        )
    return False


def _ordered_comparison(operator: str, actual: object, expected: object) -> bool:
    if type(actual) in (int, float) and type(expected) in (int, float):
        pass
    elif type(actual) is str and type(expected) is str:
        pass
    else:
        return False
    if operator == "gt":
        return actual > expected  # type: ignore[operator]
    if operator == "gte":
        return actual >= expected  # type: ignore[operator]
    if operator == "lt":
        return actual < expected  # type: ignore[operator]
    return actual <= expected  # type: ignore[operator]


def _contains(container: object, item: object) -> bool:
    if type(container) is str:
        return type(item) is str and item in container
    if type(container) is tuple:
        return any(_json_equal(candidate, item) for candidate in container)
    if type(container) in _JSON_MAPPING_TYPES:
        return type(item) is str and item in container
    return False
