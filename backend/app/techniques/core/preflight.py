"""Aggregated preflight validation before any external work is claimed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jsonschema

from app.techniques.contracts.definition import (
    ImplementationKind,
    ModelRoleDefinition,
    TechniqueCapability,
    TechniqueDefinition,
)
from app.techniques.contracts.execution import ResolvedRunSpec


@dataclass(frozen=True, slots=True)
class PreflightDiagnostic:
    code: str
    message: str


class PreflightError(ValueError):
    def __init__(self, diagnostics: tuple[PreflightDiagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__("; ".join(item.message for item in diagnostics))


class PreflightService:
    def __init__(
        self,
        *,
        model_roles: tuple[ModelRoleDefinition, ...],
        declared_detector_ids: frozenset[str],
        declared_tool_ids: frozenset[str],
        declared_converter_ids: frozenset[str],
        available_detector_ids: frozenset[str],
        available_tool_ids: frozenset[str],
        available_converter_ids: frozenset[str],
        available_model_roles: frozenset[str] = frozenset(),
        supported_capabilities: frozenset[TechniqueCapability] = frozenset(),
        supported_implementation_kinds: frozenset[ImplementationKind] = frozenset(
            ImplementationKind
        ),
    ) -> None:
        self._model_roles = model_roles
        self._declared_detector_ids = declared_detector_ids
        self._declared_tool_ids = declared_tool_ids
        self._declared_converter_ids = declared_converter_ids
        self._available_detector_ids = available_detector_ids
        self._available_tool_ids = available_tool_ids
        self._available_converter_ids = available_converter_ids
        self._available_model_roles = available_model_roles
        self._supported_capabilities = supported_capabilities
        self._supported_implementation_kinds = supported_implementation_kinds

    def validate(
        self,
        *,
        model_role_bindings: Mapping[str, object],
        implementation_kind: ImplementationKind | None = None,
        capabilities: frozenset[TechniqueCapability] = frozenset(),
        resolved_parameters: Mapping[str, object] | None = None,
        parameter_schema: Mapping[str, object] | None = None,
        required_assets: tuple[str, ...] = (),
        available_assets: frozenset[str] = frozenset(),
        target_surface: str | None = None,
        supported_target_surfaces: tuple[str, ...] = (),
    ) -> None:
        diagnostics: list[PreflightDiagnostic] = []
        for role in self._model_roles:
            if role.required and role.id not in model_role_bindings:
                diagnostics.append(
                    PreflightDiagnostic(
                        "missing_model_role",
                        f"required model role '{role.id}' is not bound",
                    )
                )
            elif role.required and role.id not in self._available_model_roles:
                diagnostics.append(
                    PreflightDiagnostic(
                        "missing_model_client",
                        f"required model role '{role.id}' has no available client",
                    )
                )
        if (
            implementation_kind is not None
            and implementation_kind not in self._supported_implementation_kinds
        ):
            diagnostics.append(
                PreflightDiagnostic(
                    "unsupported_implementation",
                    f"implementation kind '{implementation_kind}' is unsupported",
                )
            )
        unsupported = capabilities - self._supported_capabilities
        if unsupported:
            diagnostics.append(
                PreflightDiagnostic(
                    "unsupported_capability",
                    f"unsupported capabilities: {sorted(map(str, unsupported))!r}",
                )
            )
        self._check_registry(
            diagnostics,
            "detector",
            self._declared_detector_ids,
            self._available_detector_ids,
        )
        self._check_registry(
            diagnostics,
            "tool",
            self._declared_tool_ids,
            self._available_tool_ids,
        )
        self._check_registry(
            diagnostics,
            "converter",
            self._declared_converter_ids,
            self._available_converter_ids,
        )
        missing_assets = set(required_assets) - set(available_assets)
        if missing_assets:
            diagnostics.append(
                PreflightDiagnostic(
                    "missing_asset",
                    f"required assets are unavailable: {sorted(missing_assets)!r}",
                )
            )
        if (
            target_surface is not None
            and supported_target_surfaces
            and target_surface not in supported_target_surfaces
        ):
            diagnostics.append(
                PreflightDiagnostic(
                    "unsupported_target_surface",
                    f"target surface '{target_surface}' is unsupported",
                )
            )
        if parameter_schema:
            try:
                jsonschema.validate(
                    _plain_json(resolved_parameters or {}),
                    _plain_json(parameter_schema),
                )
            except jsonschema.ValidationError as error:
                diagnostics.append(
                    PreflightDiagnostic(
                        "invalid_parameters",
                        f"resolved parameters are invalid: {error.message}",
                    )
                )
        if diagnostics:
            raise PreflightError(tuple(diagnostics))

    @staticmethod
    def _check_registry(
        diagnostics: list[PreflightDiagnostic],
        kind: str,
        declared: frozenset[str],
        available: frozenset[str],
    ) -> None:
        missing = declared - available
        if missing:
            diagnostics.append(
                PreflightDiagnostic(
                    f"missing_{kind}",
                    f"declared {kind} IDs are unavailable: {sorted(missing)!r}",
                )
            )


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def validate_package_provenance(
    *,
    spec: ResolvedRunSpec,
    definition: TechniqueDefinition,
    loaded_package: object | None,
) -> None:
    """Verify that execution inputs match the immutable run specification."""

    diagnostics: list[PreflightDiagnostic] = []
    _compare_provenance(
        diagnostics,
        "technique ID",
        spec.technique_id,
        definition.identity.id,
    )
    _compare_provenance(
        diagnostics,
        "technique version",
        spec.technique_version,
        definition.identity.version,
    )
    _compare_provenance(
        diagnostics,
        "contract version",
        spec.contract_version,
        definition.contract_version,
    )

    if (
        spec.implementation_kind is not ImplementationKind.LEGACY
        and loaded_package is None
    ):
        diagnostics.append(
            PreflightDiagnostic(
                "missing_loaded_package",
                "a loaded package is required for modular implementations",
            )
        )
    elif loaded_package is not None:
        package_definition = getattr(loaded_package, "definition", None)
        package_implementation = getattr(loaded_package, "implementation", None)
        if not isinstance(package_definition, TechniqueDefinition):
            diagnostics.append(
                PreflightDiagnostic(
                    "invalid_loaded_package",
                    "loaded package has no valid technique definition",
                )
            )
        else:
            _compare_provenance(
                diagnostics,
                "loaded package definition",
                definition,
                package_definition,
            )
            _compare_provenance(
                diagnostics,
                "loaded package technique ID",
                spec.technique_id,
                package_definition.identity.id,
            )
            _compare_provenance(
                diagnostics,
                "loaded package technique version",
                spec.technique_version,
                package_definition.identity.version,
            )
            _compare_provenance(
                diagnostics,
                "loaded package contract version",
                spec.contract_version,
                package_definition.contract_version,
            )
        if package_implementation is None:
            diagnostics.append(
                PreflightDiagnostic(
                    "invalid_loaded_package",
                    "loaded package has no implementation",
                )
            )
        else:
            for label, pinned, actual in (
                ("implementation ID", spec.implementation_id, getattr(package_implementation, "id", None)),
                ("implementation kind", spec.implementation_kind, getattr(package_implementation, "kind", None)),
                ("implementation technique ID", spec.technique_id, getattr(package_implementation, "technique_id", None)),
                ("implementation technique version", spec.technique_version, getattr(package_implementation, "technique_version", None)),
                ("implementation contract version", spec.contract_version, getattr(package_implementation, "contract_version", None)),
            ):
                _compare_provenance(diagnostics, label, pinned, actual)
        _compare_provenance(
            diagnostics,
            "manifest hash",
            spec.manifest_hash,
            getattr(loaded_package, "manifest_hash", None),
        )
        _compare_provenance(
            diagnostics,
            "asset hashes",
            dict(spec.asset_hashes),
            dict(getattr(loaded_package, "asset_hashes", {}) or {}),
        )

    if diagnostics:
        raise PreflightError(tuple(diagnostics))


def _compare_provenance(
    diagnostics: list[PreflightDiagnostic],
    label: str,
    pinned: object,
    actual: object,
) -> None:
    if pinned != actual:
        diagnostics.append(
            PreflightDiagnostic(
                "provenance_mismatch",
                f"pinned {label} {pinned!r} does not match loaded value {actual!r}",
            )
        )
