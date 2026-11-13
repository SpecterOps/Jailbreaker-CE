"""Strict, deterministic loading for declarative technique packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from app.techniques.contracts.definition import (
    ConverterDefinition,
    DetectorDefinition,
    HistoryDefinition,
    ImplementationKind,
    InputDefinition,
    JudgeDefinition,
    JudgePurpose,
    JudgeSchedule,
    ModelRoleDefinition,
    ResultPolicy,
    SeedDefinition,
    SeedStrategy,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
    TechniqueImplementation,
    ToolDefinition,
)


JsonPath = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class ManifestDiagnostic:
    stage: str
    path: JsonPath
    message: str


@dataclass(frozen=True, slots=True)
class LoadedTechniquePackage:
    root: Path
    definition: TechniqueDefinition
    implementation: TechniqueImplementation
    manifest_hash: str
    asset_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_hashes", MappingProxyType(dict(self.asset_hashes)))


class ManifestValidationError(ValueError):
    def __init__(self, diagnostics: tuple[ManifestDiagnostic, ...]) -> None:
        self.diagnostics = tuple(sorted(diagnostics, key=_diagnostic_sort_key))
        super().__init__("; ".join(_render_diagnostic(item) for item in self.diagnostics))


_STRING_ARRAY = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "uniqueItems": True,
}
_OBJECT = {"type": "object"}


MANIFEST_SCHEMA: Mapping[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["api_version", "contract_version", "technique", "implementation"],
    "properties": {
        "api_version": {"type": "string", "minLength": 1},
        "contract_version": {"type": "string", "minLength": 1},
        "technique": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "id",
                "version",
                "name",
                "family",
                "summary",
                "capabilities",
            ],
            "properties": {
                "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
                "version": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "family": {"type": "string", "minLength": 1},
                "summary": {"type": "string", "minLength": 1},
                "capabilities": {
                    **_STRING_ARRAY,
                    "items": {"enum": [item.value for item in TechniqueCapability]},
                },
                "tags": _STRING_ARRAY,
                "aliases": _STRING_ARRAY,
                "target_surfaces": _STRING_ARRAY,
            },
        },
        "implementation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "kind", "entrypoint"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "kind": {"enum": [item.value for item in ImplementationKind]},
                "entrypoint": {"type": "string", "minLength": 1},
                "extensions": _OBJECT,
            },
        },
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema": {"type": "string", "minLength": 1},
                "presets": _OBJECT,
            },
        },
        "inputs": {"type": "array", "items": {"$ref": "#/$defs/input"}},
        "model_roles": {"type": "array", "items": {"$ref": "#/$defs/model_role"}},
        "histories": {"type": "array", "items": {"$ref": "#/$defs/history"}},
        "judges": {"type": "array", "items": {"$ref": "#/$defs/judge"}},
        "detectors": {"type": "array", "items": {"$ref": "#/$defs/detector"}},
        "seed": {"$ref": "#/$defs/seed"},
        "tools": {"type": "array", "items": {"$ref": "#/$defs/tool"}},
        "converters": {"type": "array", "items": {"$ref": "#/$defs/tool"}},
        "result_policy": {"$ref": "#/$defs/result_policy"},
        "presentation": _OBJECT,
        "metrics": {"type": "array", "items": _OBJECT},
        "recommended_limits": _OBJECT,
        "assets": _STRING_ARRAY,
        "state_schema_version": {"type": "integer", "minimum": 1},
        "result_schema_version": {"type": "integer", "minimum": 1},
        "source": _OBJECT,
        "extensions": _OBJECT,
    },
    "$defs": {
        "input": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "kind"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "kind": {"type": "string", "minLength": 1},
                "required": {"type": "boolean"},
                "media_types": _STRING_ARRAY,
                "max_count": {"type": "integer", "minimum": 1},
                "max_size_bytes": {"type": "integer", "minimum": 1},
                "schema": _OBJECT,
            },
        },
        "model_role": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "required": {"type": "boolean"},
                "description": {"type": "string"},
                "capabilities": _STRING_ARRAY,
                "options": _OBJECT,
            },
        },
        "history": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "visible_to": _STRING_ARRAY,
                "max_messages": {"type": "integer", "minimum": 1},
                "max_tokens": {"type": "integer", "minimum": 1},
                "extensions": _OBJECT,
            },
        },
        "judge": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "schedule": {"enum": [item.value for item in JudgeSchedule]},
                "purposes": {
                    **_STRING_ARRAY,
                    "items": {"enum": [item.value for item in JudgePurpose]},
                },
                "role": {"type": ["string", "null"]},
                "required": {"type": "boolean"},
                "evidence": _STRING_ARRAY,
                "threshold": {"type": ["number", "null"]},
                "scale_min": {"type": "number"},
                "scale_max": {"type": "number"},
                "policy": _OBJECT,
            },
        },
        "detector": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "inputs": _STRING_ARRAY,
                "output_schema": _OBJECT,
                "policy": _OBJECT,
            },
        },
        "seed": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strategy": {"enum": [item.value for item in SeedStrategy]},
                "source": {"type": ["string", "null"]},
                "required": {"type": "boolean"},
                "options": _OBJECT,
            },
        },
        "tool": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "input_schema": _OBJECT,
                "output_schema": _OBJECT,
                "policy": _OBJECT,
            },
        },
        "result_policy": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "official_judge": {"type": ["string", "null"]},
                "final_response": {"type": ["string", "null"]},
                "success_threshold": {"type": ["number", "null"]},
                "auxiliary_judges": _STRING_ARRAY,
                "stopping": _OBJECT,
            },
        },
    },
}


def load_technique_package(
    root: Path,
    supported_capabilities: frozenset[TechniqueCapability],
) -> LoadedTechniquePackage:
    package_root = root.resolve()
    manifest_path = package_root / "technique.yaml"
    manifest_bytes = _read_manifest(manifest_path)
    payload = _load_yaml(manifest_bytes)
    _validate_schema(payload)
    _validate_references(payload, package_root)
    _validate_capabilities(payload, supported_capabilities)
    _validate_entrypoint(payload, package_root)

    parameter_schema = _load_parameter_schema(payload, package_root)
    _validate_presets(payload, parameter_schema)
    definition, implementation = _build_contracts(payload, parameter_schema)
    asset_hashes = _hash_assets(payload, package_root)
    return LoadedTechniquePackage(
        root=package_root,
        definition=definition,
        implementation=implementation,
        manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        asset_hashes=asset_hashes,
    )


def _read_manifest(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        _raise("structure", (), f"cannot read technique.yaml: {error}")


def _load_yaml(content: bytes) -> dict[str, object]:
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError as error:
        _raise("structure", (), f"invalid YAML: {error}")
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        _raise("structure", (), "manifest root must be a mapping with string keys")
    return payload


def _validate_schema(payload: Mapping[str, object]) -> None:
    diagnostics = tuple(
        ManifestDiagnostic(
            stage="schema",
            path=tuple(error.absolute_path),
            message=error.message,
        )
        for error in Draft202012Validator(MANIFEST_SCHEMA).iter_errors(payload)
    )
    if diagnostics:
        raise ManifestValidationError(diagnostics)


def _validate_references(payload: Mapping[str, object], root: Path) -> None:
    diagnostics: list[ManifestDiagnostic] = []
    _validate_final_artifact_presentation(payload, diagnostics)
    judges = _list_of_mappings(payload.get("judges", []))
    judge_ids = {str(item["id"]) for item in judges}
    roles = _list_of_mappings(payload.get("model_roles", []))
    role_ids = {str(item["id"]) for item in roles}
    result = _mapping(payload.get("result_policy", {}))

    official_judge = result.get("official_judge")
    if official_judge is not None and official_judge not in judge_ids:
        diagnostics.append(
            ManifestDiagnostic(
                "references",
                ("result_policy", "official_judge"),
                f"official judge '{official_judge}' is not declared",
            )
        )
    for index, judge_id in enumerate(result.get("auxiliary_judges", [])):
        if judge_id not in judge_ids:
            diagnostics.append(
                ManifestDiagnostic(
                    "references",
                    ("result_policy", "auxiliary_judges", index),
                    f"auxiliary judge '{judge_id}' is not declared",
                )
            )
    for index, judge in enumerate(judges):
        role = judge.get("role")
        if role is not None and role not in role_ids:
            diagnostics.append(
                ManifestDiagnostic(
                    "references",
                    ("judges", index, "role"),
                    f"judge role '{role}' is not declared",
                )
            )
    for history_index, history in enumerate(
        _list_of_mappings(payload.get("histories", []))
    ):
        for role_index, role in enumerate(history.get("visible_to", [])):
            if role not in role_ids:
                diagnostics.append(
                    ManifestDiagnostic(
                        "references",
                        ("histories", history_index, "visible_to", role_index),
                        f"history-visible role '{role}' is not declared",
                    )
                )

    for path, reference in _asset_references(payload, include_entrypoint=False):
        _, error = _resolve_asset(root, reference)
        if error is not None:
            diagnostics.append(ManifestDiagnostic("references", path, error))

    if diagnostics:
        raise ManifestValidationError(tuple(diagnostics))


def _validate_final_artifact_presentation(
    payload: Mapping[str, object],
    diagnostics: list[ManifestDiagnostic],
) -> None:
    presentation = _mapping(payload.get("presentation", {}))
    if "final_artifact" not in presentation:
        return

    path = ("presentation", "final_artifact")
    final_artifact = presentation["final_artifact"]
    if not isinstance(final_artifact, Mapping):
        diagnostics.append(
            ManifestDiagnostic("references", path, "must be a mapping")
        )
        return

    allowed_keys = {
        "source",
        "label",
        "media_type",
        "include_official_judge",
    }
    for key in final_artifact:
        if key not in allowed_keys:
            diagnostics.append(
                ManifestDiagnostic("references", path, f"unknown key {key!r}")
            )

    if final_artifact.get("source") != "final_response":
        diagnostics.append(
            ManifestDiagnostic("references", path, "source must be final_response")
        )
    for key in ("label", "media_type"):
        if key not in final_artifact:
            continue
        value = final_artifact[key]
        if not isinstance(value, str) or not value.strip():
            diagnostics.append(
                ManifestDiagnostic(
                    "references",
                    path,
                    f"{key} must be non-empty text",
                )
            )
    if (
        "include_official_judge" in final_artifact
        and not isinstance(final_artifact["include_official_judge"], bool)
    ):
        diagnostics.append(
            ManifestDiagnostic(
                "references",
                path,
                "include_official_judge must be boolean",
            )
        )


def _validate_capabilities(
    payload: Mapping[str, object],
    supported: frozenset[TechniqueCapability],
) -> None:
    supported_values = {item.value for item in supported}
    technique = _mapping(payload["technique"])
    diagnostics = tuple(
        ManifestDiagnostic(
            "capabilities",
            ("technique", "capabilities", index),
            f"unsupported capability '{capability}'",
        )
        for index, capability in enumerate(technique["capabilities"])
        if capability not in supported_values
    )
    if diagnostics:
        raise ManifestValidationError(diagnostics)


def _validate_entrypoint(payload: Mapping[str, object], root: Path) -> None:
    implementation = _mapping(payload["implementation"])
    kind = ImplementationKind(str(implementation["kind"]))
    if kind is ImplementationKind.LEGACY:
        return
    reference = _entrypoint_asset(kind, str(implementation["entrypoint"]))
    resolved, error = _resolve_asset(root, reference)
    if error is None and resolved is not None and not resolved.is_file():
        error = f"implementation entrypoint '{reference}' does not exist"
    if error is not None:
        _raise("entrypoint", ("implementation", "entrypoint"), error)


def _load_parameter_schema(
    payload: Mapping[str, object], root: Path
) -> Mapping[str, object]:
    parameters = _mapping(payload.get("parameters", {}))
    reference = parameters.get("schema")
    if reference is None:
        return {}
    path, error = _resolve_asset(root, str(reference))
    if error is not None or path is None:
        _raise("references", ("parameters", "schema"), error or "invalid schema path")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _raise("schema", ("parameters", "schema"), f"invalid parameter schema: {error}")
    if not isinstance(loaded, dict):
        _raise("schema", ("parameters", "schema"), "parameter schema must be an object")
    try:
        Draft202012Validator.check_schema(loaded)
    except SchemaError as error:
        _raise("schema", ("parameters", "schema"), f"invalid parameter schema: {error}")
    return loaded


def _validate_presets(
    payload: Mapping[str, object], parameter_schema: Mapping[str, object]
) -> None:
    parameters = _mapping(payload.get("parameters", {}))
    presets = _mapping(parameters.get("presets", {}))
    validator = Draft202012Validator(parameter_schema)
    diagnostics: list[ManifestDiagnostic] = []
    for preset_name, preset in presets.items():
        for error in validator.iter_errors(preset):
            diagnostics.append(
                ManifestDiagnostic(
                    "schema",
                    ("parameters", "presets", preset_name, *error.absolute_path),
                    error.message,
                )
            )
    if diagnostics:
        raise ManifestValidationError(tuple(diagnostics))


def _build_contracts(
    payload: Mapping[str, object], parameter_schema: Mapping[str, object]
) -> tuple[TechniqueDefinition, TechniqueImplementation]:
    technique = _mapping(payload["technique"])
    implementation_data = _mapping(payload["implementation"])
    parameters = _mapping(payload.get("parameters", {}))
    result_data = _mapping(payload.get("result_policy", {}))
    seed_data = _mapping(payload.get("seed", {}))
    identity = TechniqueIdentity(
        id=str(technique["id"]),
        version=str(technique["version"]),
        name=str(technique["name"]),
        family=str(technique["family"]),
    )
    definition = TechniqueDefinition(
        identity=identity,
        summary=str(technique["summary"]),
        capabilities=frozenset(
            TechniqueCapability(str(item)) for item in technique["capabilities"]
        ),
        contract_version=str(payload["contract_version"]),
        api_version=str(payload["api_version"]),
        tags=tuple(technique.get("tags", [])),
        aliases=tuple(technique.get("aliases", [])),
        target_surfaces=tuple(technique.get("target_surfaces", [])),
        parameters_schema=parameter_schema,
        presets=_mapping(parameters.get("presets", {})),
        inputs=tuple(
            InputDefinition(
                id=str(item["id"]),
                kind=str(item["kind"]),
                required=bool(item.get("required", False)),
                media_types=tuple(item.get("media_types", [])),
                max_count=_optional_int(item.get("max_count")),
                max_size_bytes=_optional_int(item.get("max_size_bytes")),
                schema=_mapping(item.get("schema", {})),
            )
            for item in _list_of_mappings(payload.get("inputs", []))
        ),
        model_roles=tuple(
            ModelRoleDefinition(
                id=str(item["id"]),
                required=bool(item.get("required", True)),
                description=str(item.get("description", "")),
                capabilities=frozenset(str(value) for value in item.get("capabilities", [])),
                options=_mapping(item.get("options", {})),
            )
            for item in _list_of_mappings(payload.get("model_roles", []))
        ),
        histories=tuple(
            HistoryDefinition(
                id=str(item["id"]),
                visible_to=tuple(item.get("visible_to", [])),
                max_messages=_optional_int(item.get("max_messages")),
                max_tokens=_optional_int(item.get("max_tokens")),
                extensions=_mapping(item.get("extensions", {})),
            )
            for item in _list_of_mappings(payload.get("histories", []))
        ),
        judges=tuple(
            JudgeDefinition(
                id=str(item["id"]),
                schedule=JudgeSchedule(str(item.get("schedule", "run_end"))),
                purposes=frozenset(
                    JudgePurpose(str(value)) for value in item.get("purposes", [])
                ),
                role=_optional_str(item.get("role")),
                required=bool(item.get("required", True)),
                evidence=tuple(item.get("evidence", [])),
                threshold=_optional_float(item.get("threshold")),
                scale_min=float(item.get("scale_min", 0)),
                scale_max=float(item.get("scale_max", 10)),
                policy=_mapping(item.get("policy", {})),
            )
            for item in _list_of_mappings(payload.get("judges", []))
        ),
        detectors=tuple(
            DetectorDefinition(
                id=str(item["id"]),
                inputs=tuple(item.get("inputs", [])),
                output_schema=_mapping(item.get("output_schema", {})),
                policy=_mapping(item.get("policy", {})),
            )
            for item in _list_of_mappings(payload.get("detectors", []))
        ),
        seed=SeedDefinition(
            strategy=SeedStrategy(str(seed_data.get("strategy", "none"))),
            source=_optional_str(seed_data.get("source")),
            required=bool(seed_data.get("required", False)),
            options=_mapping(seed_data.get("options", {})),
        ),
        tools=tuple(
            _build_tool(ToolDefinition, item)
            for item in _list_of_mappings(payload.get("tools", []))
        ),
        converters=tuple(
            _build_tool(ConverterDefinition, item)
            for item in _list_of_mappings(payload.get("converters", []))
        ),
        result_policy=ResultPolicy(
            official_judge=_optional_str(result_data.get("official_judge")),
            final_response=_optional_str(result_data.get("final_response")),
            success_threshold=_optional_float(result_data.get("success_threshold")),
            auxiliary_judges=tuple(result_data.get("auxiliary_judges", [])),
            stopping=_mapping(result_data.get("stopping", {})),
        ),
        presentation=_mapping(payload.get("presentation", {})),
        metrics=tuple(_list_of_mappings(payload.get("metrics", []))),
        recommended_limits=_mapping(payload.get("recommended_limits", {})),
        assets=tuple(payload.get("assets", [])),
        state_schema_version=int(payload.get("state_schema_version", 1)),
        result_schema_version=int(payload.get("result_schema_version", 1)),
        source=_mapping(payload.get("source", {})),
        extensions=_mapping(payload.get("extensions", {})),
    )
    implementation = TechniqueImplementation(
        id=str(implementation_data["id"]),
        technique_id=identity.id,
        technique_version=identity.version,
        kind=ImplementationKind(str(implementation_data["kind"])),
        entrypoint=str(implementation_data["entrypoint"]),
        contract_version=str(payload["contract_version"]),
        extensions=_mapping(implementation_data.get("extensions", {})),
    )
    return definition, implementation


def _hash_assets(payload: Mapping[str, object], root: Path) -> Mapping[str, str]:
    references = {
        reference for _, reference in _asset_references(payload, include_entrypoint=True)
    }
    hashes: dict[str, str] = {}
    for reference in sorted(references):
        path, error = _resolve_asset(root, reference)
        if error is not None or path is None or not path.is_file():
            continue  # Validation stages already provide the user-facing diagnostic.
        hashes[Path(reference).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return MappingProxyType(hashes)


def _asset_references(
    payload: Mapping[str, object], *, include_entrypoint: bool
) -> tuple[tuple[JsonPath, str], ...]:
    references: list[tuple[JsonPath, str]] = []
    parameters = _mapping(payload.get("parameters", {}))
    if "schema" in parameters:
        references.append((("parameters", "schema"), str(parameters["schema"])))
    for index, reference in enumerate(payload.get("assets", [])):
        references.append((("assets", index), str(reference)))
    if include_entrypoint:
        implementation = _mapping(payload["implementation"])
        kind = ImplementationKind(str(implementation["kind"]))
        if kind is not ImplementationKind.LEGACY:
            references.append(
                (
                    ("implementation", "entrypoint"),
                    _entrypoint_asset(kind, str(implementation["entrypoint"])),
                )
            )
    return tuple(references)


def _entrypoint_asset(kind: ImplementationKind, entrypoint: str) -> str:
    if kind is ImplementationKind.WORKFLOW:
        return entrypoint
    module = entrypoint.partition(":")[0]
    if module.endswith(".py") or "/" in module:
        return module
    return f"{module.replace('.', '/')}.py"


def _resolve_asset(root: Path, reference: str) -> tuple[Path | None, str | None]:
    candidate = (root / reference).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, f"asset '{reference}' resolves outside the package root"
    if not candidate.is_file():
        return None, f"asset '{reference}' does not exist"
    return candidate, None


def _build_tool(
    constructor: type[ToolDefinition] | type[ConverterDefinition],
    item: Mapping[str, object],
) -> ToolDefinition | ConverterDefinition:
    return constructor(
        id=str(item["id"]),
        input_schema=_mapping(item.get("input_schema", {})),
        output_schema=_mapping(item.get("output_schema", {})),
        policy=_mapping(item.get("policy", {})),
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _diagnostic_sort_key(item: ManifestDiagnostic) -> tuple[str, ...]:
    return tuple(str(part) for part in item.path) + (item.stage, item.message)


def _render_diagnostic(item: ManifestDiagnostic) -> str:
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in item.path
    )
    return f"{item.stage} {path}: {item.message}"


def _raise(stage: str, path: JsonPath, message: str) -> None:
    raise ManifestValidationError((ManifestDiagnostic(stage, path, message),))
