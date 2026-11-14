"""Deterministic discovery and lifecycle management for source-controlled packages."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import logging
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Mapping

import yaml

from app.persistence.technique_runtime import (
    NoopTechniqueRuntimePersistence,
    TechniquePackageRecord,
    TechniqueRuntimePersistence,
)
from app.techniques.contracts.definition import ImplementationKind, TechniqueCapability
from app.techniques.contracts.execution import TechniqueExecutor
from app.techniques.core.catalog import TechniqueCatalog, TechniqueRegistrationConflict
from app.techniques.core.redaction import redact_credential_text
from app.techniques.plugins.manifest import (
    LoadedTechniquePackage,
    ManifestValidationError,
    load_technique_package,
)


SOURCE_CONTROLLED_PACKAGE_ROOTS = (
    Path(__file__).resolve().parents[1] / "builtin",
    Path(__file__).resolve().parents[1] / "packages",
)
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parents[1] / "defaults.yaml"
log = logging.getLogger("techniques.plugins.loader")


class PackageState(StrEnum):
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    ACTIVE = "active"
    INVALID = "invalid"
    INCOMPATIBLE = "incompatible"
    DISABLED = "disabled"
    UNINSTALLED = "uninstalled"


@dataclass(frozen=True, slots=True)
class PackageReference:
    technique_id: str
    technique_version: str
    implementation_id: str

    def __post_init__(self) -> None:
        for name in ("technique_id", "technique_version", "implementation_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class PackageDiagnostic:
    stage: str
    message: str
    source_location: str
    reference: PackageReference | None = None
    path: tuple[str | int, ...] = ()

    def as_persistence_value(self) -> Mapping[str, str]:
        return {
            "stage": self.stage,
            "message": self.message,
            "source_location": self.source_location,
            "path": ".".join(str(item) for item in self.path),
        }


@dataclass(frozen=True, slots=True)
class WorkflowEntrypoint:
    """A constrained asset reference; workflow execution is owned by Task 21+."""

    package_root: Path
    asset_path: Path

    def __post_init__(self) -> None:
        root = self.package_root.resolve()
        asset = self.asset_path.resolve()
        try:
            asset.relative_to(root)
        except ValueError as error:
            raise ValueError("workflow entrypoint resolves outside its package root") from error
        object.__setattr__(self, "package_root", root)
        object.__setattr__(self, "asset_path", asset)


@dataclass(frozen=True, slots=True)
class DiscoveredPackage:
    root: Path
    reference: PackageReference
    state: PackageState
    package: LoadedTechniquePackage | None
    diagnostics: tuple[PackageDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class PackageDiscoveryResult:
    packages: tuple[DiscoveredPackage, ...]
    diagnostics: tuple[PackageDiagnostic, ...]

    def by_id(self, technique_id: str) -> DiscoveredPackage:
        matches = [
            package for package in self.packages if package.reference.technique_id == technique_id
        ]
        if len(matches) != 1:
            raise KeyError(f"expected one discovered package for technique '{technique_id}'")
        return matches[0]


class PackageActivationError(RuntimeError):
    pass


def activation_failure_diagnostic(
    package: LoadedTechniquePackage,
    error: Exception,
) -> PackageDiagnostic:
    """Describe rejection consistently for startup logs and persisted diagnostics."""

    reference = _reference_for(package)
    message = (
        f"Activation failed for technique '{reference.technique_id}' "
        f"version '{reference.technique_version}' "
        f"implementation '{reference.implementation_id}' from '{package.root}': {error}. "
    )
    if isinstance(error, TechniqueRegistrationConflict):
        sources = sorted({
            f"{registered.package.root if registered.package is not None else registered.implementation.entrypoint} "
            f"(implementation '{registered.implementation.id}')"
            for registered in error.conflicting_registrations
        })
        message += (
            f"Conflicting registrations: {'; '.join(sources)}. "
            "Remove the duplicate package source or give changed contents a new version."
        )
    else:
        message += "Correct the package error and restart."
    return PackageDiagnostic(
        stage="activation",
        message=redact_credential_text(message),
        source_location=redact_credential_text(str(package.root)),
        reference=reference,
    )


class PackageInUseError(RuntimeError):
    pass


PackageSelfTest = Callable[[LoadedTechniquePackage, object], object]


class TechniquePackageLoader:
    """Own lifecycle facts in persistence, not in process-local loader state."""

    def __init__(
        self,
        *,
        catalog: TechniqueCatalog,
        roots: Iterable[Path] = SOURCE_CONTROLLED_PACKAGE_ROOTS,
        persistence: TechniqueRuntimePersistence | None = None,
        supported_capabilities: frozenset[TechniqueCapability] = frozenset(
            TechniqueCapability
        ),
        self_test: PackageSelfTest | object | None = None,
        defaults_path: Path = DEFAULT_OVERRIDES_PATH,
    ) -> None:
        self.catalog = catalog
        self._roots = tuple(sorted({root.resolve() for root in roots}, key=str))
        self._persistence = persistence or NoopTechniqueRuntimePersistence()
        self._supported_capabilities = supported_capabilities
        self._self_test = self_test
        self._defaults_path = defaults_path.resolve()
        self._records: dict[PackageReference, TechniquePackageRecord] = {}
        self._states: dict[PackageReference, PackageState] = {}
        self._diagnostics: list[PackageDiagnostic] = []

    def discover(self) -> PackageDiscoveryResult:
        """Validate every package independently in deterministic root/path order."""

        discovered: list[DiscoveredPackage] = []
        self._diagnostics = []
        for root in self._roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir(), key=lambda path: path.name):
                if not child.is_dir():
                    continue
                package_root = child.resolve()
                try:
                    package_root.relative_to(root)
                except ValueError:
                    continue
                discovered.append(self.validate(package_root))
        return PackageDiscoveryResult(
            packages=tuple(discovered),
            diagnostics=tuple(self._diagnostics),
        )

    def validate(self, package_root: Path) -> DiscoveredPackage:
        """Validate one package without changing catalog selection or defaults."""

        root = package_root.resolve()
        reference = _manifest_reference(root)
        package_id = _package_id(root)
        try:
            package = load_technique_package(
                root,
                supported_capabilities=self._supported_capabilities,
            )
        except ManifestValidationError as error:
            diagnostics = tuple(
                PackageDiagnostic(
                    stage=item.stage,
                    message=item.message,
                    source_location=str(root),
                    reference=reference,
                    path=item.path,
                )
                for item in error.diagnostics
            )
            discovered = DiscoveredPackage(
                root=root,
                reference=reference,
                state=PackageState.INVALID,
                package=None,
                diagnostics=diagnostics,
            )
            package_id = self._invalid_discovery_package_id(
                discovered,
                package_id=package_id,
            )
            self._persist(discovered, package_id=package_id)
            return discovered
        except Exception as error:  # noqa: BLE001 - a package must remain isolated
            diagnostic = PackageDiagnostic(
                stage="discovery",
                message=str(error),
                source_location=str(root),
                reference=reference,
            )
            discovered = DiscoveredPackage(
                root=root,
                reference=reference,
                state=PackageState.INVALID,
                package=None,
                diagnostics=(diagnostic,),
            )
            package_id = self._invalid_discovery_package_id(
                discovered,
                package_id=package_id,
            )
            self._persist(discovered, package_id=package_id)
            return discovered

        reference = _reference_for(package)
        package_id, persisted = self._persistence_identity(package)
        state = _restored_state(
            persisted,
            reference=reference,
            manifest_hash=package.manifest_hash,
            asset_hashes=package.asset_hashes,
        )
        discovered = DiscoveredPackage(
            root=root,
            reference=reference,
            state=state,
            package=package,
        )
        self._persist(discovered, package_id=package_id)
        return discovered

    def activate(
        self,
        package: LoadedTechniquePackage,
        *,
        make_default: bool = False,
        preserve_lifecycle_state: bool = False,
    ) -> None:
        """Self-test, register, persist, and optionally select as one logical unit."""

        reference = _reference_for(package)
        previous_catalog = self.catalog.snapshot()
        package_id, matching_record = self._persistence_identity(package)
        prior_record = self._persistence.load_package(package_id)
        prior_loader_record = self._records.get(reference)
        prior_state = self._states.get(reference)
        restored_state = (
            _restored_state(
                matching_record,
                reference=reference,
                manifest_hash=package.manifest_hash,
                asset_hashes=package.asset_hashes,
            )
            if preserve_lifecycle_state
            else None
        )
        state = (
            restored_state
            if restored_state in {PackageState.DISABLED, PackageState.UNINSTALLED}
            else PackageState.ACTIVE
        )
        activation_persisted = False
        try:
            entrypoint = self.resolve_entrypoint(package)
            self._run_self_test(package, entrypoint)
            self.catalog.register(_registered_package(package))
            selectable = state not in {PackageState.DISABLED, PackageState.UNINSTALLED}
            self.catalog.set_selectable(
                reference.technique_id,
                version=reference.technique_version,
                implementation_id=reference.implementation_id,
                selectable=selectable,
            )
            discovered = DiscoveredPackage(
                root=package.root,
                reference=reference,
                state=state,
                package=package,
            )
            self._persist(discovered, package_id=package_id)
            activation_persisted = True
            if make_default and selectable:
                self.catalog.set_default(
                    reference.technique_id,
                    version=reference.technique_version,
                    implementation_id=reference.implementation_id,
                )
        except Exception as error:  # noqa: BLE001 - normalize lifecycle failures
            self.catalog.restore(previous_catalog)
            try:
                self._restore_persisted_record(prior_record, package_id=package_id)
            except Exception as storage_error:  # noqa: BLE001 - retain original rejection
                log.warning(
                    "Could not restore package state after activation failure: %s Storage error: %s",
                    activation_failure_diagnostic(package, error).message,
                    redact_credential_text(str(storage_error)),
                )
            if prior_loader_record is None:
                self._records.pop(reference, None)
            else:
                self._records[reference] = prior_loader_record
            if prior_state is None:
                self._states.pop(reference, None)
            else:
                self._states[reference] = prior_state
            if not activation_persisted:
                self._persist_activation_failure(
                    package,
                    reference=reference,
                    package_id=package_id,
                    error=error,
                )
            if isinstance(error, (PackageActivationError, ValueError)):
                raise
            raise PackageActivationError(str(error)) from error

    def resolve_entrypoint(self, package: LoadedTechniquePackage) -> object:
        """Resolve data workflows without imports and Python only from configured roots."""

        implementation = package.implementation
        if implementation.kind is ImplementationKind.WORKFLOW:
            asset = (package.root / implementation.entrypoint).resolve()
            return WorkflowEntrypoint(package.root, asset)
        if implementation.kind is ImplementationKind.PYTHON:
            return self._load_python_executor(package)
        raise PackageActivationError(
            f"package '{implementation.technique_id}' has unsupported "
            f"implementation kind '{implementation.kind}'"
        )

    def disable(self, reference: PackageReference) -> None:
        self._set_lifecycle_state(reference, PackageState.DISABLED)

    def soft_uninstall(self, reference: PackageReference) -> None:
        self._set_lifecycle_state(reference, PackageState.UNINSTALLED)

    def hard_delete_unreferenced(self, reference: PackageReference) -> None:
        transient_record = self._record_for(reference)
        durable_registration = self._persistence.load_package_by_reference(
            technique_id=reference.technique_id,
            technique_version=reference.technique_version,
            implementation_id=reference.implementation_id,
        )
        record = durable_registration or transient_record
        if self._persistence.has_package_references(
            technique_id=reference.technique_id,
            technique_version=reference.technique_version,
            implementation_id=reference.implementation_id,
        ):
            raise PackageInUseError("package is referenced by persisted runs")
        snapshot = self.catalog.snapshot()
        try:
            if durable_registration is not None:
                self.catalog.unregister(
                    reference.technique_id,
                    version=reference.technique_version,
                    implementation_id=reference.implementation_id,
                )
            self._persistence.hard_delete_package(record)
        except ValueError as error:
            self.catalog.restore(snapshot)
            raise PackageInUseError("package is referenced by persisted runs") from error
        except Exception:
            self.catalog.restore(snapshot)
            raise
        self._records.pop(reference, None)
        self._states.pop(reference, None)

    def state(self, reference: PackageReference) -> PackageState:
        if reference not in self._states:
            raise KeyError(f"package {reference!r} is not known to this loader")
        return self._states[reference]

    def apply_default_overrides(self) -> tuple[PackageDiagnostic, ...]:
        """Apply validated promotion overrides after all package registrations exist."""

        try:
            payload = yaml.safe_load(self._defaults_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            return self._persist_default_diagnostic(
                PackageDiagnostic(
                    stage="defaults",
                    message=f"cannot read defaults: {error}",
                    source_location=str(self._defaults_path),
                )
            )
        if payload is None:
            return ()
        if not isinstance(payload, list):
            return self._persist_default_diagnostic(
                PackageDiagnostic(
                    stage="defaults",
                    message="defaults.yaml must contain a list",
                    source_location=str(self._defaults_path),
                )
            )
        diagnostics: list[PackageDiagnostic] = []
        for index, override in enumerate(payload):
            if not isinstance(override, Mapping):
                diagnostics.extend(
                    self._persist_default_diagnostic(
                        PackageDiagnostic(
                            stage="defaults",
                            message="default override must be a mapping",
                            source_location=str(self._defaults_path),
                            path=(index,),
                        )
                    )
                )
                continue
            try:
                reference = PackageReference(
                    technique_id=_required_override_value(override, "technique_id"),
                    technique_version=_required_override_value(
                        override, "technique_version"
                    ),
                    implementation_id=_required_override_value(
                        override, "implementation_id"
                    ),
                )
                self.catalog.set_default(
                    reference.technique_id,
                    version=reference.technique_version,
                    implementation_id=reference.implementation_id,
                )
            except (KeyError, ValueError) as error:
                reference = _override_reference(override)
                diagnostics.extend(
                    self._persist_default_diagnostic(
                        PackageDiagnostic(
                            stage="defaults",
                            message=str(error),
                            source_location=str(self._defaults_path),
                            reference=reference,
                            path=(index,),
                        )
                    )
                )
        return tuple(diagnostics)

    def _set_lifecycle_state(
        self,
        reference: PackageReference,
        state: PackageState,
    ) -> None:
        record = self._record_for(reference)
        snapshot = self.catalog.snapshot()
        updated = replace(record, state=state.value)
        try:
            self.catalog.set_selectable(
                reference.technique_id,
                version=reference.technique_version,
                implementation_id=reference.implementation_id,
                selectable=False,
            )
            self._persistence.save_package(updated)
        except Exception:
            self.catalog.restore(snapshot)
            raise
        self._records[reference] = updated
        self._states[reference] = state

    def _persist(self, discovered: DiscoveredPackage, *, package_id: str) -> None:
        record = _persistence_record(discovered, package_id=package_id)
        existing = self._persistence.load_package(package_id)
        if existing is not None:
            record = replace(
                record,
                diagnostics=_merge_diagnostics(
                    existing.diagnostics,
                    record.diagnostics,
                ),
            )
        self._persistence.save_package(record)
        self._records[discovered.reference] = record
        self._states[discovered.reference] = discovered.state
        self._diagnostics.extend(discovered.diagnostics)

    def _persist_activation_failure(
        self,
        package: LoadedTechniquePackage,
        *,
        reference: PackageReference,
        package_id: str,
        error: Exception,
    ) -> None:
        """Keep a failed activation diagnosable without hiding its root error."""

        failure = DiscoveredPackage(
            root=package.root,
            reference=reference,
            state=PackageState.INVALID,
            package=package,
            diagnostics=(
                activation_failure_diagnostic(package, error),
            ),
        )
        try:
            self._persist(failure, package_id=package_id)
        except Exception as storage_error:  # noqa: BLE001 - preserve the activation failure
            log.warning(
                "Could not persist package activation diagnostic: %s Storage error: %s",
                failure.diagnostics[0].message,
                redact_credential_text(str(storage_error)),
            )

    def _record_for(self, reference: PackageReference) -> TechniquePackageRecord:
        record = self._records.get(reference)
        if record is None:
            raise KeyError(f"package {reference!r} is not registered")
        return record

    def _persistence_identity(
        self,
        package: LoadedTechniquePackage,
    ) -> tuple[str, TechniquePackageRecord | None]:
        package_id = _package_id(package.root)
        reference = _reference_for(package)
        direct = self._persistence.load_package(package_id)
        direct_matches = _record_matches_contents(
            direct,
            reference=reference,
            manifest_hash=package.manifest_hash,
            asset_hashes=package.asset_hashes,
        )
        exact = self._persistence.load_package_by_reference(
            technique_id=reference.technique_id,
            technique_version=reference.technique_version,
            implementation_id=reference.implementation_id,
        )
        exact_matches = _record_matches_contents(
            exact,
            reference=reference,
            manifest_hash=package.manifest_hash,
            asset_hashes=package.asset_hashes,
        )
        if direct_matches and (
            exact is None
            or exact.package_id != package_id
            or exact_matches
        ):
            return package_id, direct
        if exact_matches:
            return exact.package_id, exact
        occupied_by_registration = (
            direct is not None and direct.is_registration
        ) or (
            exact is not None and exact.package_id == package_id
        )
        if not occupied_by_registration:
            return package_id, None
        conflict_id = _conflict_package_id(package)
        conflict = self._persistence.load_package(conflict_id)
        if _record_matches_contents(
            conflict,
            reference=reference,
            manifest_hash=package.manifest_hash,
            asset_hashes=package.asset_hashes,
        ):
            return conflict_id, conflict
        return conflict_id, None

    def _invalid_discovery_package_id(
        self,
        discovered: DiscoveredPackage,
        *,
        package_id: str,
    ) -> str:
        direct = self._persistence.load_package(package_id)
        if direct is None:
            return package_id
        reference = discovered.reference
        exact = self._persistence.load_package_by_reference(
            technique_id=reference.technique_id,
            technique_version=reference.technique_version,
            implementation_id=reference.implementation_id,
        )
        if direct.is_registration or (
            exact is not None and exact.package_id == package_id
        ):
            return _invalid_conflict_package_id(discovered)
        return package_id

    def _restore_persisted_record(
        self,
        record: TechniquePackageRecord | None,
        *,
        package_id: str,
    ) -> None:
        if record is not None:
            self._persistence.save_package(record)
            return
        current = self._persistence.load_package(package_id)
        if current is not None:
            self._persistence.hard_delete_package(current)

    def _run_self_test(
        self,
        package: LoadedTechniquePackage,
        entrypoint: object,
    ) -> None:
        if self._self_test is None:
            return
        try:
            if callable(self._self_test):
                result = self._self_test(package, entrypoint)
            else:
                result = self._self_test.run(package, entrypoint)
            if inspect.isawaitable(result):
                raise TypeError("package self-test must complete synchronously")
        except Exception as error:  # noqa: BLE001 - contract boundary
            raise PackageActivationError(str(error)) from error

    def _load_python_executor(self, package: LoadedTechniquePackage) -> TechniqueExecutor:
        if not any(_is_within(package.root, root) for root in self._roots):
            raise PackageActivationError(
                "Python package entrypoints require a configured source-controlled root"
            )
        module_name, separator, class_name = package.implementation.entrypoint.partition(":")
        if (
            not separator
            or not class_name
            or not class_name.isidentifier()
            or not module_name
            or any(not item.isidentifier() for item in module_name.split("."))
        ):
            raise PackageActivationError("Python entrypoint must be module:ClassName")
        module_path = (package.root / Path(*module_name.split("."))).with_suffix(".py")
        module_path = module_path.resolve()
        if not _is_within(module_path, package.root) or not module_path.is_file():
            raise PackageActivationError("Python entrypoint module is outside the package root")
        module = _load_module(module_path)
        executor_type = getattr(module, class_name, None)
        if not isinstance(executor_type, type):
            raise PackageActivationError("Python entrypoint class was not found")
        try:
            executor = executor_type()
        except Exception as error:  # noqa: BLE001 - package constructor boundary
            raise PackageActivationError(f"cannot instantiate Python entrypoint: {error}") from error
        if not isinstance(executor, TechniqueExecutor) or not inspect.iscoroutinefunction(
            executor.execute
        ):
            raise PackageActivationError(
                "Python entrypoint must implement the async TechniqueExecutor protocol"
            )
        return executor

    def _persist_default_diagnostic(
        self,
        diagnostic: PackageDiagnostic,
    ) -> tuple[PackageDiagnostic, ...]:
        reference = diagnostic.reference
        if reference is not None and reference in self._records:
            record = self._records[reference]
            updated = replace(
                record,
                diagnostics=(*record.diagnostics, diagnostic.as_persistence_value()),
            )
            self._persistence.save_package(updated)
            self._records[reference] = updated
        else:
            source = diagnostic.source_location
            package_id = "defaults-" + hashlib.sha256(source.encode()).hexdigest()
            existing = self._persistence.load_package(package_id)
            self._persistence.save_package(
                TechniquePackageRecord(
                    package_id=package_id,
                    technique_id=reference.technique_id if reference else "defaults",
                    technique_version=reference.technique_version if reference else "",
                    implementation_id=reference.implementation_id if reference else "",
                    implementation_kind="",
                    contract_version="",
                    source_kind="defaults",
                    source_location=source,
                    state=PackageState.INVALID.value,
                    manifest_hash=_file_hash(self._defaults_path),
                    diagnostics=(
                        *(existing.diagnostics if existing is not None else ()),
                        diagnostic.as_persistence_value(),
                    ),
                    is_registration=False,
                )
            )
        self._diagnostics.append(diagnostic)
        return (diagnostic,)


def _registered_package(package: LoadedTechniquePackage):
    from app.techniques.core.catalog import RegisteredTechnique

    return RegisteredTechnique(
        definition=package.definition,
        implementation=package.implementation,
        manifest_hash=package.manifest_hash,
        asset_hashes=package.asset_hashes,
        package=package,
    )


def _reference_for(package: LoadedTechniquePackage) -> PackageReference:
    return PackageReference(
        technique_id=package.definition.identity.id,
        technique_version=package.definition.identity.version,
        implementation_id=package.implementation.id,
    )


def _manifest_reference(root: Path) -> PackageReference:
    try:
        payload = yaml.safe_load((root / "technique.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        payload = {}
    technique = payload.get("technique", {}) if isinstance(payload, Mapping) else {}
    implementation = payload.get("implementation", {}) if isinstance(payload, Mapping) else {}
    return PackageReference(
        technique_id=str(technique.get("id") or root.name.replace("_", "-")),
        technique_version=str(technique.get("version") or "unknown"),
        implementation_id=str(implementation.get("id") or "unknown"),
    )


def _persistence_record(
    discovered: DiscoveredPackage,
    *,
    package_id: str,
) -> TechniquePackageRecord:
    package = discovered.package
    if package is None:
        return TechniquePackageRecord(
            package_id=package_id,
            technique_id=discovered.reference.technique_id,
            technique_version=discovered.reference.technique_version,
            implementation_id=discovered.reference.implementation_id,
            implementation_kind="",
            contract_version="",
            source_kind="source_controlled",
            source_location=str(discovered.root),
            state=discovered.state.value,
            manifest_hash=_file_hash(discovered.root / "technique.yaml"),
            diagnostics=tuple(item.as_persistence_value() for item in discovered.diagnostics),
            is_registration=False,
        )
    return TechniquePackageRecord(
        package_id=package_id,
        technique_id=discovered.reference.technique_id,
        technique_version=discovered.reference.technique_version,
        implementation_id=discovered.reference.implementation_id,
        implementation_kind=package.implementation.kind.value,
        contract_version=package.implementation.contract_version,
        source_kind="source_controlled",
        source_location=str(discovered.root),
        state=discovered.state.value,
        manifest_hash=package.manifest_hash,
        asset_hashes=package.asset_hashes,
        diagnostics=tuple(item.as_persistence_value() for item in discovered.diagnostics),
        is_registration=discovered.state
        in {PackageState.ACTIVE, PackageState.DISABLED, PackageState.UNINSTALLED},
    )


def _restored_state(
    record: TechniquePackageRecord | None,
    *,
    reference: PackageReference,
    manifest_hash: str,
    asset_hashes: Mapping[str, str],
) -> PackageState:
    if not _record_matches_contents(
        record,
        reference=reference,
        manifest_hash=manifest_hash,
        asset_hashes=asset_hashes,
    ):
        return PackageState.VALIDATED
    try:
        persisted = PackageState(record.state)
    except ValueError:
        return PackageState.VALIDATED
    if persisted in {PackageState.DISABLED, PackageState.UNINSTALLED}:
        return persisted
    return PackageState.VALIDATED


def _record_matches_contents(
    record: TechniquePackageRecord | None,
    *,
    reference: PackageReference,
    manifest_hash: str,
    asset_hashes: Mapping[str, str],
) -> bool:
    return record is not None and (
        record.technique_id,
        record.technique_version,
        record.implementation_id,
        record.manifest_hash,
        dict(record.asset_hashes),
    ) == (
        reference.technique_id,
        reference.technique_version,
        reference.implementation_id,
        manifest_hash,
        dict(asset_hashes),
    )


def _merge_diagnostics(
    existing: tuple[Mapping[str, str], ...],
    current: tuple[Mapping[str, str], ...],
) -> tuple[Mapping[str, str], ...]:
    merged: list[Mapping[str, str]] = []
    for diagnostic in (*existing, *current):
        if not any(dict(item) == dict(diagnostic) for item in merged):
            merged.append(diagnostic)
    return tuple(merged)


def _package_id(root: Path) -> str:
    return "source-" + hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()


def _conflict_package_id(package: LoadedTechniquePackage) -> str:
    reference = _reference_for(package)
    return _metadata_package_id(
        root=package.root,
        reference=reference,
        manifest_hash=package.manifest_hash,
        asset_hashes=package.asset_hashes,
    )


def _invalid_conflict_package_id(discovered: DiscoveredPackage) -> str:
    return _metadata_package_id(
        root=discovered.root,
        reference=discovered.reference,
        manifest_hash=_file_hash(discovered.root / "technique.yaml"),
        asset_hashes={},
    )


def _metadata_package_id(
    *,
    root: Path,
    reference: PackageReference,
    manifest_hash: str,
    asset_hashes: Mapping[str, str],
) -> str:
    identity = {
        "source_location": str(root.resolve()),
        "technique_id": reference.technique_id,
        "technique_version": reference.technique_version,
        "implementation_id": reference.implementation_id,
        "manifest_hash": manifest_hash,
        "asset_hashes": dict(asset_hashes),
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "source-conflict-" + hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_module(path: Path) -> ModuleType:
    module_name = "jailbreaker_source_package_" + hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PackageActivationError("cannot load Python entrypoint module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_override_value(override: Mapping[object, object], key: str) -> str:
    value = override.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"default override requires non-empty '{key}'")
    return value


def _override_reference(override: Mapping[object, object]) -> PackageReference | None:
    values = tuple(override.get(key) for key in (
        "technique_id",
        "technique_version",
        "implementation_id",
    ))
    if all(isinstance(value, str) and value for value in values):
        return PackageReference(*values)
    return None
