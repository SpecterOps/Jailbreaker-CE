"""Catalog-routed, technique-agnostic parity arm execution seams."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias

from app.techniques.contracts.definition import ImplementationKind
from app.techniques.contracts.execution import TechniqueRequest
from app.techniques.core.router import ExecutionRoute, ExecutionRouter
from devtools.technique_parity.randomness import seeded_random
from app.techniques.workflow.conditions import Condition

if TYPE_CHECKING:
    from devtools.technique_parity.compare import ParityPolicy, ParityReport


@dataclass(frozen=True, slots=True)
class ParityArmRequest:
    """One fully resolved arm request supplied to a scenario-owned runner."""

    route: ExecutionRoute
    mode: str
    policy: ParityPolicy
    dataset_id: str | None
    repeat_count: int
    profile_ids: Mapping[str, str]
    seed: int = 0
    _route_execution: _BoundRouteExecution | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.mode not in {"deterministic", "live"}:
            raise ValueError("parity arm mode must be deterministic or live")
        if type(self.repeat_count) is not int or self.repeat_count < 1:
            raise ValueError("parity arm repeat_count must be positive")
        object.__setattr__(
            self,
            "profile_ids",
            MappingProxyType(dict(sorted(self.profile_ids.items()))),
        )

    async def _execute_exact_route(
        self,
        execution_request: object,
        context: object,
    ) -> object:
        """Execute only this active runtime-issued arm's exact route."""

        execution = self._route_execution
        if type(execution) is not _BoundRouteExecution:
            raise ValueError("parity arm request has no active runtime-issued route execution")
        return await execution.execute(self, execution_request, context)


class ParityArmRunner(Protocol):
    """Generic scenario runner; dispatch is owned outside parity core."""

    def __call__(
        self,
        request: ParityArmRequest,
    ) -> object | Awaitable[object]: ...


ParityRuntimeFactory: TypeAlias = Callable[[], "ParityRuntime"]
_ParityRouteExecutor: TypeAlias = Callable[
    [ExecutionRouter, ExecutionRoute, TechniqueRequest, object],
    object | Awaitable[object],
]


class ParityArmExecutionError(RuntimeError):
    """A runtime-owned arm runner failed after its session was consumed."""


class ParityArmConfigurationError(ValueError):
    """Parity execution configuration failed after its session was consumed."""


@dataclass(frozen=True, slots=True)
class _DeterministicArmBinding:
    arm_labels: tuple[str, str]
    route_object_ids: tuple[int, int]
    registered_object_ids: tuple[int, int]
    definition_object_ids: tuple[int, int]
    route_owner_object_ids: tuple[int, int]
    workflow_binding_object_ids: tuple[int, int]
    workflow_binding_digests: tuple[str | None, str | None]
    route_digests: tuple[str, str]
    route_owners: tuple[object | None, object | None] = field(compare=False)
    workflow_bindings: tuple[object | None, object | None] = field(compare=False)


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _DeterministicArmSession:
    """One runtime-issued, exact, single-claim legacy/modular route session."""

    legacy_route: ExecutionRoute
    modular_route: ExecutionRoute
    _authority_ref: weakref.ReferenceType[_DeterministicArmAuthority]


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _ClaimedDeterministicArms:
    """Internal proof that an exact arm session was consumed before execution."""

    legacy_route: ExecutionRoute
    modular_route: ExecutionRoute
    _authority_ref: weakref.ReferenceType[_DeterministicArmAuthority]


class _DeterministicArmAuthority:
    """Runtime-scoped weak issuance state for exact comparison route objects."""

    def __init__(self) -> None:
        self._issued: weakref.WeakKeyDictionary[
            _DeterministicArmSession,
            _DeterministicArmBinding,
        ] = weakref.WeakKeyDictionary()
        self._claimed: weakref.WeakKeyDictionary[
            _ClaimedDeterministicArms,
            _DeterministicArmBinding,
        ] = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()

    def issue(
        self,
        routes: tuple[ExecutionRoute, ExecutionRoute],
    ) -> _DeterministicArmSession:
        binding = _arm_binding(routes)
        session = object.__new__(_DeterministicArmSession)
        object.__setattr__(session, "legacy_route", routes[0])
        object.__setattr__(session, "modular_route", routes[1])
        object.__setattr__(session, "_authority_ref", weakref.ref(self))
        with self._lock:
            self._issued[session] = binding
        return session

    def peek(
        self,
        session: object,
    ) -> tuple[ExecutionRoute, ExecutionRoute]:
        if type(session) is not _DeterministicArmSession:
            raise TypeError("deterministic routes require a runtime-issued arm session")
        with self._lock:
            binding = self._issued.get(session)
        if binding is None:
            raise ValueError("deterministic arm session was not issued or is stale")
        try:
            routes = _session_routes(session)
            matches = _arm_binding_matches(routes, binding)
        except (AttributeError, TypeError, ValueError):
            self.invalidate(session)
            raise
        if not matches:
            self.invalidate(session)
            raise ValueError("deterministic arm session route binding was mutated")
        return routes

    def claim(self, session: object) -> _ClaimedDeterministicArms:
        routes = self.peek(session)
        with self._lock:
            binding = self._issued.pop(session, None)
        if binding is None:
            raise ValueError("deterministic arm session was already consumed")
        claimed = object.__new__(_ClaimedDeterministicArms)
        object.__setattr__(claimed, "legacy_route", routes[0])
        object.__setattr__(claimed, "modular_route", routes[1])
        object.__setattr__(claimed, "_authority_ref", weakref.ref(self))
        with self._lock:
            self._claimed[claimed] = binding
        return claimed

    def peek_claimed(
        self,
        claimed: object,
    ) -> tuple[ExecutionRoute, ExecutionRoute]:
        if type(claimed) is not _ClaimedDeterministicArms:
            raise TypeError("deterministic comparison requires claimed runtime arms")
        with self._lock:
            binding = self._claimed.get(claimed)
        if binding is None:
            raise ValueError("deterministic arm claim was not issued or is stale")
        try:
            routes = _session_routes(claimed)
            matches = _arm_binding_matches(routes, binding)
        except (AttributeError, TypeError, ValueError):
            self.invalidate_claimed(claimed)
            raise
        if not matches:
            self.invalidate_claimed(claimed)
            raise ValueError("claimed deterministic arm route binding was mutated")
        return routes

    def consume_claimed(
        self,
        claimed: object,
    ) -> tuple[ExecutionRoute, ExecutionRoute]:
        routes = self.peek_claimed(claimed)
        with self._lock:
            binding = self._claimed.pop(claimed, None)
        if binding is None:
            raise ValueError("deterministic arm claim was already consumed")
        return routes

    def invalidate_claimed(self, claimed: object) -> None:
        if type(claimed) is not _ClaimedDeterministicArms:
            return
        with self._lock:
            self._claimed.pop(claimed, None)

    def invalidate(self, session: object) -> None:
        if type(session) is not _DeterministicArmSession:
            return
        with self._lock:
            self._issued.pop(session, None)


class _BoundRouteExecution:
    """One active arm request's exact route execution closure."""

    def __init__(
        self,
        *,
        authority: _DeterministicArmAuthority,
        claimed: _ClaimedDeterministicArms,
        arm_index: int,
        route: ExecutionRoute,
        router: ExecutionRouter,
        executor: _ParityRouteExecutor | None,
    ) -> None:
        self._authority = authority
        self._claimed = claimed
        self._arm_index = arm_index
        self._route = route
        self._router = router
        self._executor = executor
        self._request: ParityArmRequest | None = None
        self._active = False
        self._lock = threading.RLock()

    def bind(self, request: ParityArmRequest) -> None:
        with self._lock:
            if self._request is not None or self._active:
                raise ValueError("exact route execution was already bound")
            if request.route is not self._route:
                raise ValueError("exact route execution request does not own its route")
            self._request = request
            self._active = True

    async def execute(
        self,
        request: ParityArmRequest,
        execution_request: object,
        context: object,
    ) -> object:
        with self._lock:
            if not self._active:
                raise ValueError("exact route execution is no longer active")
            if request is not self._request or request.route is not self._route:
                raise ValueError("exact route execution is not owned by this arm request")
        try:
            routes = self._authority.peek_claimed(self._claimed)
        except (TypeError, ValueError):
            self.invalidate()
            raise
        if routes[self._arm_index] is not self._route:
            self.invalidate()
            self._authority.invalidate_claimed(self._claimed)
            raise ValueError("exact route execution arm binding was mutated")
        if type(execution_request) is not TechniqueRequest:
            raise TypeError("exact route execution requires a TechniqueRequest")
        executor = self._executor
        if executor is None:
            self.invalidate()
            self._authority.invalidate_claimed(self._claimed)
            raise ValueError("no exact parity route executor is registered")
        result = executor(
            self._router,
            self._route,
            execution_request,
            context,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    def invalidate(self) -> None:
        with self._lock:
            self._active = False


@dataclass(frozen=True, slots=True)
class _VerifiedPairBinding:
    arm_labels: tuple[str, str]
    route_digests: tuple[str, str]
    evidence_digests: tuple[str, str]


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _VerifiedDeterministicPair:
    """One runtime-issued, immutable, single-consumption comparison pair."""

    arm_labels: tuple[str, str]
    legacy_route_identity: Mapping[str, object]
    modular_route_identity: Mapping[str, object]
    legacy_evidence: Mapping[str, object]
    modular_evidence: Mapping[str, object]
    _authority_ref: weakref.ReferenceType[_VerifiedPairAuthority]


class _VerifiedPairAuthority:
    """Runtime-scoped weak issuance state for deterministic comparisons."""

    def __init__(self) -> None:
        self._issued: weakref.WeakKeyDictionary[
            _VerifiedDeterministicPair,
            _VerifiedPairBinding,
        ] = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()

    def issue(
        self,
        evidence: tuple[Mapping[str, object], Mapping[str, object]],
        route_identities: tuple[Mapping[str, object], Mapping[str, object]],
    ) -> _VerifiedDeterministicPair:
        legacy_evidence = _freeze_mapping(evidence[0])
        modular_evidence = _freeze_mapping(evidence[1])
        legacy_identity = _freeze_mapping(route_identities[0])
        modular_identity = _freeze_mapping(route_identities[1])
        pair = object.__new__(_VerifiedDeterministicPair)
        object.__setattr__(pair, "arm_labels", ("legacy", "modular"))
        object.__setattr__(pair, "legacy_route_identity", legacy_identity)
        object.__setattr__(pair, "modular_route_identity", modular_identity)
        object.__setattr__(pair, "legacy_evidence", legacy_evidence)
        object.__setattr__(pair, "modular_evidence", modular_evidence)
        object.__setattr__(pair, "_authority_ref", weakref.ref(self))
        binding = _VerifiedPairBinding(
            arm_labels=pair.arm_labels,
            route_digests=(_digest(legacy_identity), _digest(modular_identity)),
            evidence_digests=(_digest(legacy_evidence), _digest(modular_evidence)),
        )
        with self._lock:
            self._issued[pair] = binding
        return pair

    def consume(
        self,
        pair: _VerifiedDeterministicPair,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        if type(pair) is not _VerifiedDeterministicPair:
            raise TypeError("deterministic pair was not issued by a parity runtime")
        with self._lock:
            binding = self._issued.pop(pair, None)
        if binding is None:
            raise ValueError("deterministic pair was not issued or was already consumed")
        try:
            arm_labels = pair.arm_labels
            legacy_identity = pair.legacy_route_identity
            modular_identity = pair.modular_route_identity
            legacy_evidence = pair.legacy_evidence
            modular_evidence = pair.modular_evidence
            actual = _VerifiedPairBinding(
                arm_labels=arm_labels,
                route_digests=(
                    _digest(legacy_identity),
                    _digest(modular_identity),
                ),
                evidence_digests=(
                    _digest(legacy_evidence),
                    _digest(modular_evidence),
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("deterministic pair binding was mutated") from exc
        if actual != binding:
            raise ValueError("deterministic pair arm, route, or evidence digest was mutated")
        return legacy_evidence, modular_evidence


@dataclass(frozen=True, slots=True)
class ParityRuntime:
    """Resolved catalog/router and an optional scenario execution seam."""

    router: ExecutionRouter
    arm_runner: ParityArmRunner | None = None
    _route_executor: _ParityRouteExecutor | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _pair_authority: _VerifiedPairAuthority = field(
        default_factory=_VerifiedPairAuthority,
        init=False,
        repr=False,
        compare=False,
    )
    _arm_authority: _DeterministicArmAuthority = field(
        default_factory=_DeterministicArmAuthority,
        init=False,
        repr=False,
        compare=False,
    )

    def issue_deterministic_arms(
        self,
        technique_id: str,
        *,
        legacy_implementation_id: str,
        modular_implementation_id: str,
        technique_version: str | None = None,
    ) -> _DeterministicArmSession:
        """Resolve and issue one exact, runtime-owned legacy/modular arm session."""

        routes = (
            self.router.resolve(
                technique_id,
                technique_version=technique_version,
                implementation_id=legacy_implementation_id,
            ),
            self.router.resolve(
                technique_id,
                technique_version=technique_version,
                implementation_id=modular_implementation_id,
            ),
        )
        self._validate_deterministic_arm_kinds(routes)
        return self._arm_authority.issue(routes)

    def deterministic_arm_routes(
        self,
        session: object,
    ) -> tuple[ExecutionRoute, ExecutionRoute]:
        """Inspect exact issued routes without granting comparison authorization."""

        self._require_arm_authority(session)
        return self._arm_authority.peek(session)

    def compare_deterministic_captures(
        self,
        captures: tuple[object, object],
        session: object,
        *,
        policy: ParityPolicy,
    ) -> ParityReport:
        """Validate and compare one exact legacy/modular capture pair."""

        from devtools.technique_parity.compare import ParityComparator

        pair = self._prepare_deterministic_pair(captures, session)
        return ParityComparator()._compare_verified_deterministic_pair(  # noqa: SLF001
            pair,
            policy=policy,
        )

    def execute_and_compare_deterministic(
        self,
        session: object,
        *,
        policy: ParityPolicy,
        dataset_id: str | None,
        repeat_count: int,
        profile_ids: Mapping[str, str],
        seed: int,
    ) -> tuple[ParityReport, tuple[object, object]]:
        """Consume exact arms, execute both, and compare their captured evidence."""

        from devtools.technique_parity.compare import ParityComparator

        claimed = self._claim_deterministic_arms(session)
        runner = self.arm_runner
        if runner is None:
            self._arm_authority.invalidate_claimed(claimed)
            raise ParityArmConfigurationError(
                "no generic parity arm runner is registered"
            )
        try:
            captures = self._execute_claimed_arms(
                claimed,
                runner,
                mode="deterministic",
                policy=policy,
                dataset_id=dataset_id,
                repeat_count=repeat_count,
                profile_ids=profile_ids,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - preserve opaque runner boundary
            self._arm_authority.invalidate_claimed(claimed)
            raise ParityArmExecutionError("a parity arm runner failed") from exc
        pair = self._pair_from_claimed_captures(captures, claimed)
        report = ParityComparator()._compare_verified_deterministic_pair(  # noqa: SLF001
            pair,
            policy=policy,
        )
        return report, (captures[0], captures[1])

    def compare_live_captures(
        self,
        captures: tuple[object, object],
        session: object,
        *,
        policy: ParityPolicy,
    ) -> ParityReport:
        """Consume exact arms and compare pre-captured live samples."""

        claimed = self._claim_deterministic_arms(session)
        return self._compare_live_from_claimed(captures, claimed, policy=policy)

    def execute_and_compare_live(
        self,
        session: object,
        *,
        policy: ParityPolicy,
        dataset_id: str | None,
        repeat_count: int,
        profile_ids: Mapping[str, str],
        seed: int,
    ) -> tuple[ParityReport, tuple[object, object]]:
        """Consume exact arms, execute both, and compare their live samples."""

        claimed = self._claim_deterministic_arms(session)
        runner = self.arm_runner
        if runner is None:
            self._arm_authority.invalidate_claimed(claimed)
            raise ParityArmConfigurationError(
                "no generic parity arm runner is registered"
            )
        try:
            captures = self._execute_claimed_arms(
                claimed,
                runner,
                mode="live",
                policy=policy,
                dataset_id=dataset_id,
                repeat_count=repeat_count,
                profile_ids=profile_ids,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - preserve opaque runner boundary
            self._arm_authority.invalidate_claimed(claimed)
            raise ParityArmExecutionError("a parity arm runner failed") from exc
        report = self._compare_live_from_claimed(captures, claimed, policy=policy)
        return report, (captures[0], captures[1])

    def _execute_claimed_arms(
        self,
        claimed: _ClaimedDeterministicArms,
        runner: ParityArmRunner,
        *,
        mode: str,
        policy: ParityPolicy,
        dataset_id: str | None,
        repeat_count: int,
        profile_ids: Mapping[str, str],
        seed: int,
    ) -> tuple[object, object]:
        captures: list[object] = []
        for arm_index in range(2):
            routes = self._arm_authority.peek_claimed(claimed)
            route = routes[arm_index]
            execution = _BoundRouteExecution(
                authority=self._arm_authority,
                claimed=claimed,
                arm_index=arm_index,
                route=route,
                router=self.router,
                executor=self._route_executor,
            )
            request = ParityArmRequest(
                route=route,
                mode=mode,
                policy=policy,
                dataset_id=dataset_id,
                repeat_count=repeat_count,
                profile_ids=profile_ids,
                seed=seed,
            )
            object.__setattr__(request, "_route_execution", execution)
            execution.bind(request)
            try:
                if (
                    mode == "deterministic"
                    and route.registered.implementation.kind is ImplementationKind.LEGACY
                ):
                    with seeded_random(seed):
                        captures.append(_invoke_arm_runner(runner, request))
                else:
                    captures.append(_invoke_arm_runner(runner, request))
            finally:
                execution.invalidate()
        return captures[0], captures[1]

    def _prepare_deterministic_pair(
        self,
        captures: tuple[object, object],
        session: object,
    ) -> _VerifiedDeterministicPair:
        claimed = self._claim_deterministic_arms(session)
        return self._pair_from_claimed_captures(captures, claimed)

    def _pair_from_claimed_captures(
        self,
        captures: tuple[object, object],
        claimed: object,
    ) -> _VerifiedDeterministicPair:
        self._require_arm_authority(claimed)
        routes = self._arm_authority.consume_claimed(claimed)
        evidence = tuple(
            _verified_capture_mapping(capture, route)
            for capture, route in zip(captures, routes, strict=True)
        )
        return self._pair_authority.issue(
            (evidence[0], evidence[1]),
            (parity_route_identity(routes[0]), parity_route_identity(routes[1])),
        )

    def _compare_live_from_claimed(
        self,
        captures: tuple[object, object],
        claimed: object,
        *,
        policy: ParityPolicy,
    ) -> ParityReport:
        from devtools.technique_parity.compare import compare_live_samples

        self._require_arm_authority(claimed)
        routes = self._arm_authority.consume_claimed(claimed)
        evidence = tuple(
            _verified_capture_payload(capture, route)
            for capture, route in zip(captures, routes, strict=True)
        )
        if not all(
            isinstance(samples, Sequence)
            and not isinstance(samples, (str, bytes, bytearray))
            for samples in evidence
        ):
            raise TypeError("live arm evidence must be sample sequences")
        return compare_live_samples(evidence[0], evidence[1], policy=policy)

    def validate_policy_threshold(
        self,
        policy: ParityPolicy,
        session: object,
        *,
        require_declared: bool,
    ) -> None:
        """Bind a preloaded policy threshold to both resolved result policies."""

        try:
            routes = self.deterministic_arm_routes(session)
            thresholds = tuple(
                route.registered.definition.result_policy.success_threshold
                for route in routes
            )
            if require_declared and any(threshold is None for threshold in thresholds):
                raise ValueError("live parity routes must declare success thresholds")
            declared = tuple(threshold for threshold in thresholds if threshold is not None)
            if declared and len({float(threshold) for threshold in declared}) != 1:
                raise ValueError("resolved parity routes disagree on success threshold")
            if declared and float(policy.success_threshold) != float(declared[0]):
                raise ValueError("parity policy success threshold mismatches resolved routes")
        except (TypeError, ValueError):
            self._arm_authority.invalidate(session)
            raise

    def _invalidate_deterministic_arm_session(self, session: object) -> None:
        """Permanently discard an issued session after an external pre-run failure."""

        self._require_arm_authority(session)
        self._arm_authority.invalidate(session)

    def _validate_deterministic_arm_kinds(
        self,
        routes: tuple[ExecutionRoute, ExecutionRoute],
    ) -> None:
        legacy, modular = routes
        if legacy.registered.implementation.kind is not ImplementationKind.LEGACY:
            raise ValueError("deterministic legacy arm must resolve a legacy implementation")
        if modular.registered.implementation.kind is ImplementationKind.LEGACY:
            raise ValueError("deterministic modular arm must not resolve a legacy implementation")
        if legacy.registered is modular.registered:
            raise ValueError("deterministic comparison arms must resolve distinct routes")

    def _claim_deterministic_arms(
        self,
        session: object,
    ) -> _ClaimedDeterministicArms:
        self._require_arm_authority(session)
        claimed = self._arm_authority.claim(session)
        self._validate_deterministic_arm_kinds(
            self._arm_authority.peek_claimed(claimed)
        )
        return claimed

    def _require_arm_authority(self, value: object) -> None:
        if type(value) not in {
            _DeterministicArmSession,
            _ClaimedDeterministicArms,
        }:
            raise TypeError("deterministic routes require a runtime-issued arm session")
        try:
            authority = value._authority_ref()  # type: ignore[attr-defined]  # noqa: SLF001
        except (AttributeError, TypeError) as exc:
            raise TypeError("deterministic arm session has no issuance authority") from exc
        if authority is not self._arm_authority:
            raise ValueError("deterministic arm session belongs to another runtime")


def _session_routes(
    value: _DeterministicArmSession | _ClaimedDeterministicArms,
) -> tuple[ExecutionRoute, ExecutionRoute]:
    try:
        routes = (value.legacy_route, value.modular_route)
    except AttributeError as exc:
        raise ValueError("deterministic arm session route binding was mutated") from exc
    if not all(type(route) is ExecutionRoute for route in routes):
        raise TypeError("deterministic arm session requires exact execution routes")
    return routes


def _arm_binding(
    routes: tuple[ExecutionRoute, ExecutionRoute],
) -> _DeterministicArmBinding:
    owners = (routes[0]._owner, routes[1]._owner)  # noqa: SLF001
    workflow_bindings = (
        routes[0]._workflow_binding,  # noqa: SLF001
        routes[1]._workflow_binding,  # noqa: SLF001
    )
    return _DeterministicArmBinding(
        arm_labels=("legacy", "modular"),
        route_object_ids=(id(routes[0]), id(routes[1])),
        registered_object_ids=(
            id(routes[0].registered),
            id(routes[1].registered),
        ),
        definition_object_ids=(
            id(routes[0].registered.definition),
            id(routes[1].registered.definition),
        ),
        route_owner_object_ids=(id(owners[0]), id(owners[1])),
        workflow_binding_object_ids=(
            id(workflow_bindings[0]),
            id(workflow_bindings[1]),
        ),
        workflow_binding_digests=(
            _authority_digest(workflow_bindings[0]),
            _authority_digest(workflow_bindings[1]),
        ),
        route_digests=(
            _digest(parity_route_identity(routes[0])),
            _digest(parity_route_identity(routes[1])),
        ),
        route_owners=owners,
        workflow_bindings=workflow_bindings,
    )


def _arm_binding_matches(
    routes: tuple[ExecutionRoute, ExecutionRoute],
    expected: _DeterministicArmBinding,
) -> bool:
    actual = _arm_binding(routes)
    return (
        actual == expected
        and all(
            route._owner is owner  # noqa: SLF001
            for route, owner in zip(routes, expected.route_owners, strict=True)
        )
        and all(
            route._workflow_binding is binding  # noqa: SLF001
            for route, binding in zip(
                routes,
                expected.workflow_bindings,
                strict=True,
            )
        )
    )


def _authority_digest(value: object | None) -> str | None:
    return None if value is None else _digest(value)


def _invoke_arm_runner(runner: object, request: ParityArmRequest) -> object:
    if not callable(runner):
        raise TypeError("parity arm runner must be callable")
    result = runner(request)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def default_parity_runtime() -> ParityRuntime:
    """Build the operational catalog without promoting any modular default."""

    from app.persistence.technique_runtime import NoopTechniqueRuntimePersistence
    from app.techniques import build_default_catalog
    from devtools.technique_parity.builtin import (
        _execute_builtin_parity_route,
        run_builtin_parity_arm,
    )

    return ParityRuntime(
        router=ExecutionRouter(
            build_default_catalog(
                persistence=NoopTechniqueRuntimePersistence(),
            )
        ),
        arm_runner=run_builtin_parity_arm,
        _route_executor=_execute_builtin_parity_route,
    )


def parity_route_identity(route: ExecutionRoute) -> dict[str, object]:
    """Return the exact selector and pinned package/spec identity for a route."""

    registered = route.registered
    definition = registered.definition
    implementation = registered.implementation
    specification = {
        "definition": _plain(definition),
        "implementation": _plain(implementation),
        "manifest_hash": registered.manifest_hash,
        "asset_hashes": _plain(registered.asset_hashes),
    }
    encoded = json.dumps(
        specification,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "technique_id": definition.identity.id,
        "technique_version": definition.identity.version,
        "implementation_id": implementation.id,
        "contract_version": implementation.contract_version,
        "manifest_hash": registered.manifest_hash,
        "asset_hashes": _plain(registered.asset_hashes),
        "spec_hash": hashlib.sha256(encoded).hexdigest(),
    }


def parity_route_contract(route: ExecutionRoute) -> dict[str, object]:
    """Return route-owned deterministic contract evidence for one arm."""

    registered = route.registered
    definition = _plain(registered.definition)
    resolved_spec = {
        "definition": definition,
        "implementation_contract_version": registered.implementation.contract_version,
        "manifest_hash": registered.manifest_hash,
        "asset_hashes": _plain(registered.asset_hashes),
    }
    encoded = json.dumps(
        resolved_spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "route_identity": parity_route_identity(route),
        "definition": definition,
        "resolved_spec": {
            "spec_hash": hashlib.sha256(encoded).hexdigest(),
            "implementation_contract_version": registered.implementation.contract_version,
            "manifest_hash": registered.manifest_hash,
            "asset_hashes": _plain(registered.asset_hashes),
        },
    }


def _consume_verified_deterministic_pair(
    pair: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Consume a pair through its runtime-scoped issuance authority exactly once."""

    if type(pair) is not _VerifiedDeterministicPair:
        raise TypeError("deterministic comparison requires a runtime-issued pair")
    try:
        authority = pair._authority_ref()  # noqa: SLF001 - internal capability binding
    except (AttributeError, TypeError) as exc:
        raise TypeError("deterministic comparison pair has no issuance authority") from exc
    if type(authority) is not _VerifiedPairAuthority:
        raise TypeError("deterministic comparison pair has no live issuance authority")
    return authority.consume(pair)


def _verified_capture_mapping(
    capture: object,
    route: ExecutionRoute,
) -> Mapping[str, object]:
    raw_evidence = _verified_capture_payload(capture, route)
    evidence = _plain(raw_evidence)
    if not isinstance(evidence, Mapping):
        raise TypeError("deterministic arm evidence must be a mapping")
    snapshot = {str(key): item for key, item in evidence.items()}
    expected_contract = parity_route_contract(route)
    supplied_contract = snapshot.get("contract")
    if supplied_contract is not None:
        if not isinstance(supplied_contract, Mapping):
            raise ValueError("deterministic contract evidence must be a mapping")
        for field in ("route_identity", "definition", "resolved_spec"):
            if field in supplied_contract and supplied_contract[field] != expected_contract[field]:
                raise ValueError(
                    "deterministic contract evidence does not match resolved route"
                )
    return _freeze_mapping({**snapshot, "contract": expected_contract})


def _verified_capture_payload(capture: object, route: ExecutionRoute) -> object:
    if not isinstance(capture, Mapping):
        raise TypeError("parity capture must be a mapping")
    identity = capture.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("parity capture requires embedded identity")
    if dict(identity) != parity_route_identity(route):
        raise ValueError("parity capture identity does not match the resolved route")
    if "evidence" not in capture:
        raise ValueError("parity capture requires evidence")
    return capture["evidence"]


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_value(_plain(value))
    if not isinstance(frozen, Mapping):  # pragma: no cover - mapping input invariant
        raise TypeError("verified deterministic evidence must be a mapping")
    return frozen


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Condition):
        return _plain(value.canonical_mapping())
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(
                items,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        return items
    return value
