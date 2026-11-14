"""Implementation-kind routing over the versioned technique catalog."""

from __future__ import annotations

import hashlib
import inspect
import os
import stat
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Mapping
from weakref import WeakValueDictionary

from app.techniques.contracts.definition import ImplementationKind
from app.techniques.contracts.execution import (
    TechniqueExecutor,
    TechniqueOutcome,
    TechniqueRequest,
)
from app.techniques.core.catalog import (
    RegisteredTechnique,
    TechniqueCatalog,
    TechniqueSelector,
)
from app.techniques.plugins.manifest import LoadedTechniquePackage
from app.techniques.workflow.definition import WorkflowDefinition, load_workflow
from app.techniques.workflow.executor import (
    WorkflowExecutor,
    _VerifiedWorkflowBinding,
)


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class ExecutionRoute:
    """A public legacy route or a router-sealed package execution route."""

    registered: RegisteredTechnique
    _workflow_binding: _VerifiedWorkflowBinding | None
    _python_binding: _VerifiedPythonBinding | None
    _owner: object | None

    def __init__(self, registered: RegisteredTechnique) -> None:
        object.__setattr__(self, "registered", registered)
        object.__setattr__(self, "_workflow_binding", None)
        object.__setattr__(self, "_python_binding", None)
        object.__setattr__(self, "_owner", None)

    @classmethod
    def _verified_workflow(
        cls,
        registered: RegisteredTechnique,
        binding: _VerifiedWorkflowBinding,
        owner: object,
    ) -> ExecutionRoute:
        route = cls(registered)
        object.__setattr__(route, "_workflow_binding", binding)
        object.__setattr__(route, "_owner", owner)
        return route

    @classmethod
    def _verified_python(
        cls,
        registered: RegisteredTechnique,
        binding: _VerifiedPythonBinding,
        owner: object,
    ) -> ExecutionRoute:
        route = cls(registered)
        object.__setattr__(route, "_python_binding", binding)
        object.__setattr__(route, "_owner", owner)
        return route

    @property
    def workflow(self) -> WorkflowDefinition | None:
        binding = self._workflow_binding
        return None if binding is None else binding.workflow

    @property
    def sealed_template_assets(self) -> Mapping[str, str]:
        """Immutable package text captured when this route was issued."""

        if self._workflow_binding is not None:
            return self._workflow_binding.template_assets
        if self._python_binding is not None:
            return self._python_binding.template_assets
        return MappingProxyType({})

    @property
    def is_legacy(self) -> bool:
        return self.registered.implementation.kind is ImplementationKind.LEGACY


class ExecutionRouter:
    def __init__(self, catalog: TechniqueCatalog) -> None:
        self.catalog = catalog
        self._route_owner = object()
        self._issued_routes: WeakValueDictionary[int, ExecutionRoute] = (
            WeakValueDictionary()
        )

    def resolve(
        self,
        technique_id: str,
        *,
        technique_version: str | None = None,
        implementation_id: str | None = None,
    ) -> ExecutionRoute:
        registered = self.catalog.resolve(
            TechniqueSelector(
                technique_id=technique_id,
                technique_version=technique_version,
                implementation_id=implementation_id,
            )
        )
        kind = registered.implementation.kind
        if kind is ImplementationKind.LEGACY:
            return ExecutionRoute(registered)
        if kind is ImplementationKind.WORKFLOW:
            route = ExecutionRoute._verified_workflow(
                registered,
                _snapshot_registered_workflow(registered),
                self._route_owner,
            )
        elif kind is ImplementationKind.PYTHON:
            if not isinstance(registered.package, LoadedTechniquePackage):
                # Metadata-only registrations remain resolvable for offline
                # contract comparison, but cannot pass this router's issued
                # package-route execution boundary.
                return ExecutionRoute(registered)
            route = ExecutionRoute._verified_python(
                registered,
                _snapshot_registered_python(registered),
                self._route_owner,
            )
        else:  # pragma: no cover - ImplementationKind is closed
            raise ValueError(f"unsupported modular implementation kind '{kind}'")
        self._issued_routes[id(route)] = route
        return route

    async def execute(
        self,
        route: ExecutionRoute,
        request: TechniqueRequest,
        context: object,
        *,
        executors: Mapping[ImplementationKind, TechniqueExecutor],
    ) -> TechniqueOutcome:
        """Dispatch only on implementation kind, never on technique ID."""

        _validate_route_ownership(route, request)
        kind = route.registered.implementation.kind
        if kind is ImplementationKind.LEGACY:
            executor = executors.get(kind)
            if executor is None:
                raise LookupError(f"no executor is registered for implementation kind '{kind}'")
            return await executor.execute(request, context)
        self._validate_issued_route(route)
        package = route.registered.package
        if not isinstance(package, LoadedTechniquePackage):
            raise ValueError("modular execution requires an exact loaded package")
        if getattr(context, "loaded_package", None) is not package:
            raise ValueError("execution context does not own the routed loaded package")
        if kind is ImplementationKind.WORKFLOW:
            binding = route._workflow_binding
            executor = executors.get(kind)
            if binding is None:
                raise ValueError("workflow execution requires a verified package route")
            if type(executor) is not WorkflowExecutor:
                raise TypeError("workflow implementation kind requires an exact WorkflowExecutor")
            return await executor._execute_verified(  # noqa: SLF001 - package binding boundary
                request,
                context,  # type: ignore[arg-type]
                binding,
            )
        if kind is ImplementationKind.PYTHON:
            if kind in executors:
                raise TypeError("Python executor injection is not permitted for a package route")
            binding = route._python_binding
            if binding is None:
                raise ValueError("Python execution requires a verified package route")
            _revalidate_python_binding(package, binding)
            executor = _instantiate_python_executor(binding)
            return await executor.execute(request, context)  # type: ignore[arg-type]
        raise ValueError(f"unsupported modular implementation kind '{kind}'")

    def template_assets(self, route: ExecutionRoute) -> Mapping[str, str]:
        """Return only the immutable assets sealed into this exact issued route."""

        self._validate_issued_route(route)
        if route._workflow_binding is not None:
            return route._workflow_binding.template_assets
        if route._python_binding is not None:
            return route._python_binding.template_assets
        raise ValueError("legacy routes do not own sealed template assets")

    def _validate_issued_route(self, route: ExecutionRoute) -> None:
        if (
            route._owner is not self._route_owner
            or self._issued_routes.get(id(route)) is not route
        ):
            raise ValueError("modular execution requires this router's exact issued route")


@dataclass(frozen=True, slots=True)
class _VerifiedPythonBinding:
    module_name: str
    class_name: str
    entrypoint_asset: str
    source: bytes
    source_hash: str
    template_assets: Mapping[str, str]


def _snapshot_registered_python(
    registered: RegisteredTechnique,
) -> _VerifiedPythonBinding:
    package = registered.package
    if not isinstance(package, LoadedTechniquePackage):
        raise ValueError("Python registration requires an exact loaded package")
    module_name, separator, class_name = registered.implementation.entrypoint.partition(":")
    if (
        not separator
        or not class_name.isidentifier()
        or not module_name
        or any(not item.isidentifier() for item in module_name.split("."))
    ):
        raise ValueError("Python entrypoint must be module:ClassName")
    entrypoint_asset = PurePosixPath(*module_name.split(".")).with_suffix(".py").as_posix()
    expected_hash = package.asset_hashes.get(entrypoint_asset)
    if expected_hash is None:
        raise ValueError("Python entrypoint is not a pinned package asset")
    source = _read_confined_asset(package, entrypoint_asset, label="Python entrypoint")
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError("pinned Python entrypoint hash does not match catalog registration")
    try:
        compile(source, str(package.root / entrypoint_asset), "exec")
    except (SyntaxError, ValueError) as error:
        raise ValueError("Python entrypoint source cannot be compiled") from error
    template_assets: dict[str, str] = {}
    for asset_id, asset_hash in sorted(package.asset_hashes.items()):
        content = source if asset_id == entrypoint_asset else _read_confined_asset(
            package,
            asset_id,
            label="package",
        )
        if hashlib.sha256(content).hexdigest() != asset_hash:
            raise ValueError("pinned package asset hash does not match catalog registration")
        try:
            template_assets[asset_id] = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return _VerifiedPythonBinding(
        module_name=module_name,
        class_name=class_name,
        entrypoint_asset=entrypoint_asset,
        source=source,
        source_hash=expected_hash,
        template_assets=MappingProxyType(template_assets),
    )


def _revalidate_python_binding(
    package: LoadedTechniquePackage,
    binding: _VerifiedPythonBinding,
) -> None:
    for asset_id, expected_hash in package.asset_hashes.items():
        content = _read_confined_asset(package, asset_id, label="pinned package")
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"pinned package asset '{asset_id}' changed after route issuance")
    source = _read_confined_asset(
        package,
        binding.entrypoint_asset,
        label="Python entrypoint",
    )
    if hashlib.sha256(source).hexdigest() != binding.source_hash or source != binding.source:
        raise ValueError("pinned Python entrypoint changed after route issuance")


def _instantiate_python_executor(binding: _VerifiedPythonBinding) -> TechniqueExecutor:
    module = ModuleType(
        f"_jailbreaker_technique_{binding.module_name}_{binding.source_hash}"
    )
    module.__file__ = binding.entrypoint_asset
    module.__package__ = ""
    exec(  # noqa: S102 - the route admits only validated, source-controlled packages
        compile(binding.source, binding.entrypoint_asset, "exec"),
        module.__dict__,
    )
    executor_type = getattr(module, binding.class_name, None)
    if not isinstance(executor_type, type):
        raise ValueError("Python entrypoint class was not found in the issued source")
    executor = executor_type()
    if not isinstance(executor, TechniqueExecutor) or not inspect.iscoroutinefunction(
        executor.execute
    ):
        raise TypeError("Python entrypoint must implement async TechniqueExecutor")
    return executor


def _snapshot_registered_workflow(
    registered: RegisteredTechnique,
) -> _VerifiedWorkflowBinding:
    package = registered.package
    if not isinstance(package, LoadedTechniquePackage):
        raise ValueError("workflow registration requires an exact loaded package")
    entrypoint = PurePosixPath(registered.implementation.entrypoint).as_posix()
    workflow_bytes = _read_confined_asset(package, entrypoint, label="workflow")
    expected_workflow_hash = package.asset_hashes.get(entrypoint)
    if expected_workflow_hash is None:
        raise ValueError("workflow entrypoint is not a pinned package asset")
    if hashlib.sha256(workflow_bytes).hexdigest() != expected_workflow_hash:
        raise ValueError("pinned workflow asset hash does not match catalog registration")
    workflow = load_workflow(workflow_bytes)
    template_assets: dict[str, str] = {}
    text_asset_ids = set(_workflow_template_asset_ids(workflow))
    fallback_asset = registered.definition.seed.options.get("fallback_asset")
    if fallback_asset is not None:
        if type(fallback_asset) is not str or not fallback_asset:
            raise ValueError("seed fallback asset must be a non-empty string")
        text_asset_ids.add(PurePosixPath(fallback_asset).as_posix())
    for asset_id in sorted(text_asset_ids):
        if asset_id not in package.definition.assets or asset_id not in package.asset_hashes:
            raise ValueError(f"workflow render references no pinned package template asset '{asset_id}'")
        content = _read_confined_asset(package, asset_id, label="template")
        if hashlib.sha256(content).hexdigest() != package.asset_hashes[asset_id]:
            raise ValueError("pinned package template asset hash does not match catalog registration")
        try:
            template_assets[asset_id] = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"package template asset '{asset_id}' is not valid UTF-8") from error
    return _VerifiedWorkflowBinding(
        workflow=workflow,
        result_policy=registered.definition.result_policy,
        template_assets=MappingProxyType(template_assets),
    )


def _workflow_template_asset_ids(workflow: WorkflowDefinition) -> tuple[str, ...]:
    assets: set[str] = set()
    for step in workflow.steps.values():
        if step.type != "render":
            continue
        asset = step.options.get("asset", step.options.get("template_asset"))
        if asset is None:
            continue
        if type(asset) is not str or not asset:
            raise ValueError(f"render step '{step.id}' template asset must be a non-empty string")
        assets.add(PurePosixPath(asset).as_posix())
    return tuple(sorted(assets))


def _read_confined_asset(
    package: LoadedTechniquePackage,
    reference: str,
    *,
    label: str,
) -> bytes:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY") or os.open not in os.supports_dir_fd:
        raise ValueError("secure confined package reads are unavailable on this platform")
    path = PurePosixPath(reference)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} reference is not a confined package asset")
    root = package.root
    try:
        if root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError(f"{label} reference is not a confined package asset")
    except OSError as error:
        raise ValueError(f"{label} reference is not a confined package asset") from error

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow = os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags | nofollow)
        descriptors.append(current)
        for part in path.parts[:-1]:
            current = os.open(part, directory_flags | nofollow, dir_fd=current)
            descriptors.append(current)
        asset_fd = os.open(
            path.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
            dir_fd=current,
        )
        descriptors.append(asset_fd)
        before = os.fstat(asset_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} reference is not a confined package asset")
        chunks: list[bytes] = []
        while chunk := os.read(asset_fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(asset_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"{label} package asset changed while it was read")
        return b"".join(chunks)
    except OSError as error:
        raise ValueError(f"{label} reference is not a confined package asset") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _validate_route_ownership(
    route: ExecutionRoute,
    request: TechniqueRequest,
) -> None:
    registered = route.registered
    spec = request.spec
    expected = (
        registered.definition.identity.id,
        registered.definition.identity.version,
        registered.implementation.id,
        registered.implementation.kind,
        registered.implementation.contract_version,
        registered.manifest_hash,
        registered.asset_hashes,
    )
    actual = (
        spec.technique_id,
        spec.technique_version,
        spec.implementation_id,
        spec.implementation_kind,
        spec.contract_version,
        spec.manifest_hash,
        spec.asset_hashes,
    )
    if actual != expected:
        raise ValueError("execution route does not match the pinned run specification")
