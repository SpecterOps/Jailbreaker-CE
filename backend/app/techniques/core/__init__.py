"""Generic runtime services for legacy and modular techniques.

Exports are lazy so leaf services can depend on each other without making an
OpenAI-compatible client import initialize the complete runtime graph.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ArtifactService": ("app.techniques.core.artifacts", "ArtifactService"),
    "BudgetExhausted": ("app.techniques.core.budgets", "BudgetExhausted"),
    "BudgetLimits": ("app.techniques.core.budgets", "BudgetLimits"),
    "BudgetService": ("app.techniques.core.budgets", "BudgetService"),
    "CallRetryPolicy": ("app.techniques.core.models", "CallRetryPolicy"),
    "CancellationToken": (
        "app.techniques.core.cancellation",
        "CancellationToken",
    ),
    "CheckpointService": ("app.techniques.core.checkpoints", "CheckpointService"),
    "CorrelationScope": ("app.techniques.core.correlation", "CorrelationScope"),
    "ConverterService": ("app.techniques.core.converters", "ConverterService"),
    "DetectionService": ("app.techniques.core.detections", "DetectionService"),
    "EventJournal": ("app.techniques.core.events", "EventJournal"),
    "ExecutionCheckpoint": (
        "app.techniques.core.checkpoints",
        "ExecutionCheckpoint",
    ),
    "ExecutionContext": ("app.techniques.core.context", "ExecutionContext"),
    "ExecutionContextFactory": (
        "app.techniques.core.context",
        "ExecutionContextFactory",
    ),
    "ExecutionRoute": ("app.techniques.core.router", "ExecutionRoute"),
    "ExecutionRouter": ("app.techniques.core.router", "ExecutionRouter"),
    "HistoryMessage": ("app.techniques.core.histories", "HistoryMessage"),
    "HistoryService": ("app.techniques.core.histories", "HistoryService"),
    "HistorySnapshot": ("app.techniques.core.histories", "HistorySnapshot"),
    "InputService": ("app.techniques.core.inputs", "InputService"),
    "JudgeService": ("app.techniques.core.judging", "JudgeService"),
    "ModelCallResult": ("app.techniques.core.models", "ModelCallResult"),
    "ModelService": ("app.techniques.core.models", "ModelService"),
    "ProviderCallResult": ("app.techniques.core.models", "ProviderCallResult"),
    "ProviderRequestContext": (
        "app.techniques.core.models",
        "ProviderRequestContext",
    ),
    "PreflightService": ("app.techniques.core.preflight", "PreflightService"),
    "RedactionService": ("app.techniques.core.redaction", "RedactionService"),
    "RegisteredTechnique": ("app.techniques.core.catalog", "RegisteredTechnique"),
    "TechniqueCatalog": ("app.techniques.core.catalog", "TechniqueCatalog"),
    "TechniqueEventIdentity": (
        "app.techniques.core.events",
        "TechniqueEventIdentity",
    ),
    "TechniqueSelector": ("app.techniques.core.catalog", "TechniqueSelector"),
    "SeedService": ("app.techniques.core.seeds", "SeedService"),
    "TargetService": ("app.techniques.core.targets", "TargetService"),
    "TemplateService": ("app.techniques.core.templates", "TemplateService"),
    "ToolService": ("app.techniques.core.tools", "ToolService"),
    "current_correlation": (
        "app.techniques.core.correlation",
        "current_correlation",
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
