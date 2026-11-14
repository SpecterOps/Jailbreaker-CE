"""Compatibility boundary for existing attack-library implementations."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "LegacyTechniqueAdapter": ("app.techniques.legacy.adapter", "LegacyTechniqueAdapter"),
    "build_default_catalog": ("app.techniques.legacy.registration", "build_default_catalog"),
    "definition_from_legacy": (
        "app.techniques.legacy.registration",
        "definition_from_legacy",
    ),
    "emit_legacy_result": (
        "app.techniques.legacy.translation",
        "emit_legacy_result",
    ),
    "register_legacy_techniques": (
        "app.techniques.legacy.registration",
        "register_legacy_techniques",
    ),
    "translate_legacy_trace": (
        "app.techniques.legacy.translation",
        "translate_legacy_trace",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
