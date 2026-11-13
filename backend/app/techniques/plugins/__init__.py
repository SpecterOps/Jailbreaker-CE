"""Import-safe public exports for technique manifests and package lifecycle."""

from __future__ import annotations

from importlib import import_module


_MANIFEST_EXPORTS = {
    "LoadedTechniquePackage",
    "ManifestDiagnostic",
    "ManifestValidationError",
    "load_technique_package",
}
_LOADER_EXPORTS = {
    "DEFAULT_OVERRIDES_PATH",
    "SOURCE_CONTROLLED_PACKAGE_ROOTS",
    "DiscoveredPackage",
    "PackageActivationError",
    "PackageDiagnostic",
    "PackageDiscoveryResult",
    "PackageInUseError",
    "PackageReference",
    "PackageState",
    "TechniquePackageLoader",
    "WorkflowEntrypoint",
}
__all__ = sorted(_MANIFEST_EXPORTS | _LOADER_EXPORTS)


def __getattr__(name: str) -> object:
    if name in _MANIFEST_EXPORTS:
        module_name = "app.techniques.plugins.manifest"
    elif name in _LOADER_EXPORTS:
        module_name = "app.techniques.plugins.loader"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
