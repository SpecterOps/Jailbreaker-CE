"""Safe fixed-operation metric projection."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

from app.techniques.contracts.presentation import MetricDefinition


def _numbers(values: Iterable[object]) -> list[int | float]:
    return [value for value in values if _is_number(value)]


def count_values(values: list[object], _: Mapping[str, object]) -> int:
    return len(values)


def sum_values(values: list[object], _: Mapping[str, object]) -> int | float:
    return sum(_numbers(values))


def average_values(values: list[object], _: Mapping[str, object]) -> float | None:
    numbers = _numbers(values)
    return sum(numbers) / len(numbers) if numbers else None


def minimum(values: list[object], _: Mapping[str, object]) -> int | float | None:
    numbers = _numbers(values)
    return min(numbers) if numbers else None


def maximum(values: list[object], _: Mapping[str, object]) -> int | float | None:
    numbers = _numbers(values)
    return max(numbers) if numbers else None


def first_value(values: list[object], _: Mapping[str, object]) -> object:
    return values[0] if values else None


def last_value(values: list[object], _: Mapping[str, object]) -> object:
    return values[-1] if values else None


def rate_values(values: list[object], options: Mapping[str, object]) -> float | None:
    expected = options.get("equals", True)
    return sum(value == expected for value in values) / len(values) if values else None


def difference(values: list[object], _: Mapping[str, object]) -> int | float | None:
    numbers = _numbers(values)
    return numbers[-1] - numbers[0] if len(numbers) >= 2 else None


def ratio(values: list[object], _: Mapping[str, object]) -> float | None:
    numbers = _numbers(values)
    if len(numbers) < 2 or numbers[1] == 0:
        return None
    return numbers[0] / numbers[1]


MetricOperation = Callable[[list[object], Mapping[str, object]], object]
OPERATIONS: Mapping[str, MetricOperation] = {
    "count": count_values,
    "sum": sum_values,
    "average": average_values,
    "minimum": minimum,
    "maximum": maximum,
    "rate": rate_values,
    "difference": difference,
    "ratio": ratio,
    "first": first_value,
    "last": last_value,
}


class MetricProjector:
    def project_core(
        self,
        *,
        cells: tuple[Mapping[str, object], ...],
    ) -> dict[str, int | float | None]:
        selected_scores: list[int | float] = []
        total_attempts = 0
        completed_attempts = 0
        failed_attempts = 0
        inconclusive_cells = 0
        failure_category_counts: dict[str, int] = {}
        attempt_costs: list[int | float] = []
        selected_costs: list[int | float] = []
        failed_costs: list[int | float] = []
        for cell in cells:
            attempts_value = cell.get("attempts", ())
            attempts = (
                tuple(attempt for attempt in attempts_value if isinstance(attempt, Mapping))
                if isinstance(attempts_value, Iterable) and not isinstance(attempts_value, (str, bytes, Mapping))
                else ()
            )
            total_attempts += len(attempts)
            selected_id = cell.get("selected_attempt_id")
            selected_attempt: Mapping[str, object] | None = None
            for attempt in attempts:
                execution_status = _enum_value(attempt.get("execution_status", attempt.get("status")))
                if execution_status == "completed":
                    completed_attempts += 1
                if execution_status in {"failed", "timed_out"}:
                    failed_attempts += 1
                failure_category = _enum_value(attempt.get("failure_category"))
                if failure_category:
                    failure_category_counts[failure_category] = failure_category_counts.get(failure_category, 0) + 1
                cost = attempt.get("cost")
                if _is_number(cost):
                    attempt_costs.append(cost)
                    if execution_status in {"failed", "timed_out"} or failure_category:
                        failed_costs.append(cost)
                if attempt.get("attempt_id") != selected_id:
                    continue
                selected_attempt = attempt
                if _is_number(cost):
                    selected_costs.append(cost)
            if selected_attempt is None:
                continue
            execution_status = _enum_value(selected_attempt.get("execution_status", selected_attempt.get("status")))
            disposition = _enum_value(selected_attempt.get("result_disposition", selected_attempt.get("disposition")))
            if disposition == "inconclusive":
                inconclusive_cells += 1
            score = selected_attempt.get("score")
            if execution_status == "completed" and disposition == "tested" and _is_number(score):
                selected_scores.append(score)
        return {
            "total_cells": len(cells),
            "tested_cells": len(selected_scores),
            "inconclusive_cells": inconclusive_cells,
            "average_score": _apply_operation(
                average_values,
                list(selected_scores),
                {},
            ),
            "total_attempts": total_attempts,
            "completed_attempts": completed_attempts,
            "failed_attempts": failed_attempts,
            "failure_category_counts": dict(sorted(failure_category_counts.items())),
            "total_attempt_cost": (
                _apply_operation(sum_values, list(attempt_costs), {})
                if attempt_costs
                else None
            ),
            "selected_attempt_cost": (
                _apply_operation(sum_values, list(selected_costs), {})
                if selected_costs
                else None
            ),
            "failed_attempt_cost": (
                _apply_operation(sum_values, list(failed_costs), {})
                if failed_costs
                else None
            ),
        }

    def project_declared(
        self,
        *,
        records: tuple[Mapping[str, object], ...],
        definitions: tuple[MetricDefinition | Mapping[str, object], ...],
    ) -> dict[str, object]:
        projected: dict[str, object] = {}
        for raw_definition in definitions:
            definition = _coerce_metric_definition(raw_definition)
            operation = OPERATIONS.get(definition.aggregation)
            if operation is None:
                raise ValueError(f"metric '{definition.id}' uses unsupported operation '{definition.aggregation}'")
            _validate_metric_paths(definition)
            filtered = [
                record
                for record in records
                if all(_resolve(record, key) == value for key, value in definition.filters.items())
            ]
            if not definition.group_by:
                values = [_resolve(record, definition.source) for record in filtered]
                values = [
                    value
                    for value in values
                    if value is not _MISSING and not _is_non_finite_float(value)
                ]
                projected[definition.id] = _apply_operation(
                    operation,
                    values,
                    definition.options,
                )
                continue

            groups: dict[
                tuple[tuple[type[object], object], ...],
                tuple[tuple[object, ...], list[object]],
            ] = {}
            for record in filtered:
                group = tuple(_resolve(record, path) for path in definition.group_by)
                if _MISSING in group:
                    continue
                for value in group:
                    _validate_group_value(definition.id, value)
                identity = tuple((type(value), value) for value in group)
                _, values = groups.setdefault(identity, (group, []))
                source = _resolve(record, definition.source)
                if source is not _MISSING and not _is_non_finite_float(source):
                    values.append(source)
            projected[definition.id] = [
                {
                    "group": dict(zip(definition.group_by, group, strict=True)),
                    "value": _apply_operation(
                        operation,
                        values,
                        definition.options,
                    ),
                }
                for group, values in sorted(
                    groups.values(),
                    key=lambda item: _group_sort_key(item[0]),
                )
            ]
        return projected


def _coerce_metric_definition(
    value: MetricDefinition | Mapping[str, object],
) -> MetricDefinition:
    if isinstance(value, MetricDefinition):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("metric definition must be a mapping or MetricDefinition")
    metric_id = _required_string(value, "id")
    source = _required_string(value, "source")
    aggregation_value = value.get("aggregation", "average")
    if not isinstance(aggregation_value, str) or not aggregation_value:
        raise ValueError(
            f"metric '{metric_id}' aggregation must be a non-empty string"
        )
    aggregation = aggregation_value
    name = value.get("name", metric_id)
    if not isinstance(name, str) or not name:
        raise ValueError(f"metric '{metric_id}' name must be a non-empty string")
    description = value.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"metric '{metric_id}' description must be a string")
    unit = value.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError(f"metric '{metric_id}' unit must be a string or null")
    group_by_value = value.get("group_by", ())
    if not isinstance(group_by_value, (tuple, list)) or not all(
        isinstance(item, str) and _valid_path(item) for item in group_by_value
    ):
        raise ValueError(f"metric '{metric_id}' group_by must be a list of safe field paths")
    filters = value.get("filters", {})
    options = value.get("options", {})
    if not isinstance(filters, Mapping) or not isinstance(options, Mapping):
        raise ValueError(f"metric '{metric_id}' filters and options must be mappings")
    return MetricDefinition(
        id=metric_id,
        name=name,
        source=source,
        aggregation=aggregation,
        description=description,
        unit=unit,
        group_by=tuple(group_by_value),
        filters=filters,
        options=options,
    )


def _required_string(value: Mapping[str, object], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field:
        raise ValueError(f"metric {key!r} must be a non-empty string")
    return field


_MISSING = object()


def _resolve(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _enum_value(value: object) -> str | None:
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, str) and raw else None


def _is_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and not _is_non_finite_float(value)
    )


def _is_non_finite_float(value: object) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def _apply_operation(
    operation: MetricOperation,
    values: list[object],
    options: Mapping[str, object],
) -> object:
    try:
        projected = operation(values, options)
    except OverflowError:
        return None
    return _finite_projection_value(projected)


def _finite_projection_value(value: object) -> object:
    if _is_non_finite_float(value):
        return None
    if isinstance(value, Mapping):
        return {
            key: _finite_projection_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_finite_projection_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_finite_projection_value(item) for item in value)
    return value


def _valid_path(path: object) -> bool:
    return isinstance(path, str) and bool(path) and all(path.split("."))


def _validate_metric_paths(definition: MetricDefinition) -> None:
    paths = (definition.source, *definition.group_by, *definition.filters)
    if not all(_valid_path(path) for path in paths):
        raise ValueError(f"metric '{definition.id}' uses an invalid field path")


def _validate_group_value(metric_id: str, value: object) -> None:
    if (
        value is not None
        and not isinstance(value, (str, int, float, bool))
    ) or _is_non_finite_float(value):
        raise ValueError(
            f"metric '{metric_id}' group values must be finite JSON scalars"
        )


def _group_sort_key(group: tuple[object, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(value).__name__, repr(value)) for value in group)
