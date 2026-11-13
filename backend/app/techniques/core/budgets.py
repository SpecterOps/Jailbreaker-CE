"""Prospective hard-limit accounting for controlled runtime operations."""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Iterator


class BudgetExhausted(RuntimeError):
    def __init__(self, resource: str, *, limit: int | float, requested: int | float) -> None:
        self.resource = resource
        self.limit = limit
        self.requested = requested
        super().__init__(
            f"budget exhausted for {resource}: requested cumulative "
            f"{requested}, limit is {limit}"
        )


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_model_calls: int | None = None
    max_target_calls: int | None = None
    max_judge_calls: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    max_artifact_bytes: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            if item.name == "max_cost":
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise TypeError("max_cost must be a finite number or None")
            elif type(value) is not int:
                raise TypeError(f"{item.name} must be a built-in integer or None")
            if value < 0:
                raise ValueError(f"{item.name} must not be negative")


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    model_calls: int = 0
    target_calls: int = 0
    judge_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost: float = 0
    artifact_bytes: int = 0


_LIMIT_NAMES = {
    "model_calls": "max_model_calls",
    "target_calls": "max_target_calls",
    "judge_calls": "max_judge_calls",
    "tool_calls": "max_tool_calls",
    "tokens": "max_tokens",
    "cost": "max_cost",
    "artifact_bytes": "max_artifact_bytes",
}


_ACTIVE_RESERVATION: ContextVar[object | None] = ContextVar(
    "technique_budget_reservation",
    default=None,
)


class BudgetService:
    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self._usage = BudgetUsage()
        self._lock = threading.Lock()

    @property
    def usage(self) -> BudgetUsage:
        with self._lock:
            return self._usage

    def remaining(self, resource: str) -> int | float | None:
        if resource not in _LIMIT_NAMES:
            raise KeyError(f"unknown budget resource: {resource!r}")
        active = self._active_reservation()
        if active is not None:
            return active.remaining(resource)
        with self._lock:
            limit = getattr(self.limits, _LIMIT_NAMES[resource])
            if limit is None:
                return None
            return limit - getattr(self._usage, resource)

    def reserve(self, **amounts: int | float) -> BudgetUsage:
        active = self._active_reservation()
        if active is not None:
            active.reserve(**amounts)
            return self.usage
        return self._reserve_root(amounts)

    def reserve_capacity(self, **amounts: int | float) -> BudgetReservation:
        """Reserve worst-case capacity for bounded child work before dispatch."""

        active = self._active_reservation()
        if active is not None:
            return active.reserve_capacity(**amounts)
        normalized = _validate_amounts(amounts)
        self._reserve_root(normalized)
        return BudgetReservation(self, normalized)

    @contextmanager
    def activate(self, reservation: BudgetReservation) -> Iterator[None]:
        if type(reservation) is not BudgetReservation or reservation.owner is not self:
            raise ValueError("budget reservation belongs to a different budget manager")
        reservation._require_open()
        token = _ACTIVE_RESERVATION.set(reservation)
        try:
            yield
        finally:
            _ACTIVE_RESERVATION.reset(token)

    def _reserve_root(self, amounts: dict[str, int | float]) -> BudgetUsage:
        amounts = _validate_amounts(amounts)
        with self._lock:
            prospective = {name: getattr(self._usage, name) + amounts.get(name, 0) for name in _LIMIT_NAMES}
            for resource, limit_name in _LIMIT_NAMES.items():
                limit = getattr(self.limits, limit_name)
                if limit is not None and prospective[resource] > limit:
                    raise BudgetExhausted(
                        resource,
                        limit=limit,
                        requested=prospective[resource],
                    )
            self._usage = BudgetUsage(**prospective)
            return self._usage

    def reconcile(
        self,
        *,
        reserved_tokens: int = 0,
        actual_tokens: int = 0,
        reserved_cost: float = 0,
        actual_cost: float = 0,
    ) -> BudgetUsage:
        active = self._active_reservation()
        if active is not None:
            active.reconcile(
                reserved_tokens=reserved_tokens,
                actual_tokens=actual_tokens,
                reserved_cost=reserved_cost,
                actual_cost=actual_cost,
            )
            return self.usage
        _validate_reconciliation(
            reserved_tokens=reserved_tokens,
            actual_tokens=actual_tokens,
            reserved_cost=reserved_cost,
            actual_cost=actual_cost,
        )
        with self._lock:
            prospective = replace(
                self._usage,
                tokens=self._usage.tokens - reserved_tokens + actual_tokens,
                cost=self._usage.cost - reserved_cost + actual_cost,
            )
            for resource in ("tokens", "cost"):
                limit = getattr(self.limits, _LIMIT_NAMES[resource])
                requested = getattr(prospective, resource)
                if limit is not None and requested > limit:
                    raise BudgetExhausted(
                        resource,
                        limit=limit,
                        requested=requested,
                    )
            self._usage = prospective
            return self._usage

    def _refund(self, amounts: dict[str, int | float]) -> None:
        with self._lock:
            prospective = {name: getattr(self._usage, name) - amounts.get(name, 0) for name in _LIMIT_NAMES}
            if any(value < 0 for value in prospective.values()):
                raise RuntimeError("budget reservation refund exceeds charged usage")
            self._usage = BudgetUsage(**prospective)

    def _active_reservation(self) -> BudgetReservation | None:
        active = _ACTIVE_RESERVATION.get()
        return active if isinstance(active, BudgetReservation) and active.owner is self else None


class BudgetReservation:
    """A composable slice of capacity already charged to a hard budget."""

    def __init__(
        self,
        parent: BudgetService | BudgetReservation,
        allocated: dict[str, int | float],
    ) -> None:
        self.owner = parent if type(parent) is BudgetService else parent.owner
        self._parent = parent
        self._allocated = {name: allocated.get(name, 0) for name in _LIMIT_NAMES}
        self._consumed = {name: 0 for name in _LIMIT_NAMES}
        self._lock = threading.RLock()
        self._closed = False

    def remaining(self, resource: str) -> int | float:
        if resource not in _LIMIT_NAMES:
            raise KeyError(f"unknown budget resource: {resource!r}")
        with self._lock:
            self._require_open()
            return self._allocated[resource] - self._consumed[resource]

    def reserve(self, **amounts: int | float) -> None:
        normalized = _validate_amounts(amounts)
        with self._lock:
            self._require_open()
            prospective = {name: self._consumed[name] + normalized.get(name, 0) for name in _LIMIT_NAMES}
            for resource in _LIMIT_NAMES:
                if prospective[resource] > self._allocated[resource]:
                    raise BudgetExhausted(
                        resource,
                        limit=self._allocated[resource],
                        requested=prospective[resource],
                    )
            self._consumed = prospective

    def reserve_capacity(self, **amounts: int | float) -> BudgetReservation:
        normalized = _validate_amounts(amounts)
        self.reserve(**normalized)
        return BudgetReservation(self, normalized)

    def reconcile(
        self,
        *,
        reserved_tokens: int = 0,
        actual_tokens: int = 0,
        reserved_cost: float = 0,
        actual_cost: float = 0,
    ) -> None:
        _validate_reconciliation(
            reserved_tokens=reserved_tokens,
            actual_tokens=actual_tokens,
            reserved_cost=reserved_cost,
            actual_cost=actual_cost,
        )
        with self._lock:
            self._require_open()
            prospective = dict(self._consumed)
            prospective["tokens"] += actual_tokens - reserved_tokens
            prospective["cost"] += actual_cost - reserved_cost
            for resource in ("tokens", "cost"):
                if prospective[resource] < 0:
                    raise RuntimeError("budget reconciliation exceeds reservation usage")
                if prospective[resource] > self._allocated[resource]:
                    raise BudgetExhausted(
                        resource,
                        limit=self._allocated[resource],
                        requested=prospective[resource],
                    )
            self._consumed = prospective

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            unused = {name: self._allocated[name] - self._consumed[name] for name in _LIMIT_NAMES}
            self._closed = True
        self._parent._refund(unused)

    def _refund(self, amounts: dict[str, int | float]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot refund a closed budget reservation")
            prospective = {name: self._consumed[name] - amounts.get(name, 0) for name in _LIMIT_NAMES}
            if any(value < 0 for value in prospective.values()):
                raise RuntimeError("budget reservation refund exceeds charged usage")
            self._consumed = prospective

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("budget reservation is closed")


def _validate_amounts(
    amounts: dict[str, int | float],
) -> dict[str, int | float]:
    unknown = set(amounts) - set(_LIMIT_NAMES)
    if unknown:
        raise KeyError(f"unknown budget resources: {sorted(unknown)!r}")
    normalized: dict[str, int | float] = {}
    for resource, amount in amounts.items():
        if resource == "cost":
            if type(amount) not in (int, float) or not math.isfinite(amount):
                raise TypeError("cost reservation must be a finite number")
        elif type(amount) is not int:
            raise TypeError(f"{resource} reservation must be a built-in integer")
        if amount < 0:
            raise ValueError("budget reservations must not be negative")
        normalized[resource] = amount
    return normalized


def _validate_reconciliation(
    *,
    reserved_tokens: int,
    actual_tokens: int,
    reserved_cost: float,
    actual_cost: float,
) -> None:
    for label, value in (
        ("reserved_tokens", reserved_tokens),
        ("actual_tokens", actual_tokens),
    ):
        if type(value) is not int:
            raise TypeError(f"{label} must be a built-in integer")
    for label, value in (
        ("reserved_cost", reserved_cost),
        ("actual_cost", actual_cost),
    ):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise TypeError(f"{label} must be a finite number")
    if min(reserved_tokens, actual_tokens, reserved_cost, actual_cost) < 0:
        raise ValueError("budget reconciliation values must not be negative")
