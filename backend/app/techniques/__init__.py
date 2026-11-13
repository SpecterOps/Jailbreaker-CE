"""Versioned, modular technique contracts and runtime boundaries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

log = logging.getLogger("techniques")

if TYPE_CHECKING:
    from app.techniques.legacy.registration import (
        definition_from_legacy,
        register_legacy_techniques,
    )

__all__ = [
    "build_default_catalog",
    "definition_from_legacy",
    "register_legacy_techniques",
]


def __getattr__(name: str) -> Any:
    if name in {"definition_from_legacy", "register_legacy_techniques"}:
        from app.techniques import legacy

        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_default_catalog(*, discover_modular: bool = True, persistence=None):
    """Build legacy defaults, then discover modular packages without promotion."""

    from app.persistence import build_technique_runtime_persistence
    from app.techniques.core.catalog import TechniqueCatalog
    from app.techniques.legacy.registration import register_legacy_techniques
    from app.techniques.plugins.loader import (
        PackageActivationError,
        PackageState,
        TechniquePackageLoader,
        activation_failure_diagnostic,
    )

    catalog = TechniqueCatalog()
    register_legacy_techniques(catalog)
    if not discover_modular:
        return catalog

    loader = TechniquePackageLoader(
        catalog=catalog,
        persistence=(
            build_technique_runtime_persistence()
            if persistence is None
            else persistence
        ),
    )
    discovery = loader.discover()
    for discovered in discovery.packages:
        if discovered.package is not None and discovered.state in {
            PackageState.VALIDATED,
            PackageState.DISABLED,
            PackageState.UNINSTALLED,
        }:
            try:
                loader.activate(
                    discovered.package,
                    preserve_lifecycle_state=True,
                )
            except Exception as error:  # noqa: BLE001 - isolate each startup package
                log.warning("%s", activation_failure_diagnostic(discovered.package, error).message)
                continue
    default_diagnostics = loader.apply_default_overrides()
    if default_diagnostics:
        detail = "; ".join(
            f"{item.source_location}: {item.message}"
            for item in default_diagnostics
        )
        raise PackageActivationError(
            f"technique defaults promotion could not be proven: {detail}"
        )
    return catalog
