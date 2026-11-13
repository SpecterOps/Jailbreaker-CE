"""Persistence boundaries with import-safe lazy public exports."""

from __future__ import annotations

from importlib import import_module


_RUN_EXPORTS = {
    "ExperimentExecutionOwnershipConflict",
    "ExperimentPlanSaveResult",
    "NoopRunPersistence",
    "PostgreSQLRunPersistence",
    "RunPersistence",
    "build_run_persistence",
}
_RUNTIME_EXPORTS = {
    "ArtifactRecord",
    "CheckpointRecord",
    "ModelCallRecord",
    "NoopTechniqueRuntimePersistence",
    "PostgreSQLTechniqueRuntimePersistence",
    "TechniquePackageRecord",
    "TechniqueRuntimePersistence",
    "build_technique_runtime_persistence",
}
_ATTEMPT_EXPORTS = {
    "AttemptRecord",
    "CellRecord",
    "ExperimentAttemptPersistence",
    "InMemoryExperimentAttemptPersistence",
    "OwnedAttemptCompletion",
    "PostgreSQLExperimentAttemptPersistence",
    "RetryBatchCellRecord",
    "RetryBatchRecord",
    "TechniqueConfigRecord",
}
__all__ = sorted(_RUN_EXPORTS | _RUNTIME_EXPORTS | _ATTEMPT_EXPORTS)


def __getattr__(name: str) -> object:
    if name in _RUN_EXPORTS:
        module_name = "app.persistence.runs"
    elif name in _RUNTIME_EXPORTS:
        module_name = "app.persistence.technique_runtime"
    elif name in _ATTEMPT_EXPORTS:
        module_name = "app.persistence.experiment_attempts"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
