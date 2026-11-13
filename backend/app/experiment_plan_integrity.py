"""Canonical identity and hash derivation for immutable experiment plans."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from hashlib import sha256

from app.techniques.core.redaction import contains_secret_material


_RESOLVED_SPEC_IDENTITY_FIELDS = (
    "technique_id",
    "technique_version",
    "implementation_id",
    "implementation_kind",
    "contract_version",
    "manifest_hash",
)
_RESOLVED_SPEC_MAPPING_FIELDS = (
    "asset_hashes",
    "parameters",
    "model_roles",
    "judge_policies",
    "effective_limits",
    "runtime_config",
)
_RESOLVED_SPEC_PROJECTION_FIELDS = (
    *_RESOLVED_SPEC_IDENTITY_FIELDS,
    "asset_hashes",
    "preset",
    "parameters",
    "model_roles",
    "judge_policies",
    "effective_limits",
    "runtime_config",
)
_NORMALIZED_CONFIG_PROJECTION_FIELDS = (
    "technique_id",
    "technique_version",
    "implementation_id",
    "preset",
    "parameters",
)


def is_complete_resolved_spec(value: object) -> bool:
    """Return whether a spec asserts the complete canonical projection contract."""
    if not isinstance(value, Mapping):
        return False
    return {
        *_RESOLVED_SPEC_IDENTITY_FIELDS,
        *_RESOLVED_SPEC_MAPPING_FIELDS,
        "preset",
    } <= set(value)


def is_legacy_partial_technique_config(
    *,
    technique_version: object,
    implementation_id: object,
    resolved_spec: object,
) -> bool:
    """Return whether normalized evidence is an incomplete migration record.

    Historical compatibility rows are identified by their explicit legacy
    version/implementation pair and incomplete resolved specification.  A full
    specification remains canonical even when it retains compatibility
    metadata from migration.
    """
    return (
        str(technique_version) == "0.2.0"
        and str(implementation_id) == "legacy"
        and isinstance(resolved_spec, Mapping)
        and not is_complete_resolved_spec(resolved_spec)
    )


def plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [plain_json(item) for item in value]
        return sorted(items, key=canonical_json)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return getattr(value, "value")
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def content_hash(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def technique_config_id(experiment_id: str, resolved_spec: object) -> str:
    material = "\x1f".join((experiment_id, canonical_json(resolved_spec)))
    return f"etc-{sha256(material.encode('utf-8')).hexdigest()[:32]}"


def validate_normalized_technique_config(
    value: Mapping[str, object],
    *,
    experiment_id: str,
    require_canonical: bool = False,
) -> dict[str, object]:
    """Validate a normalized config record against its authoritative spec.

    Task 14's partial records remain legacy-compatible. That includes explicit
    ``0.2.0``/``legacy`` records whose historical opaque IDs use an ``etc-``
    prefix. Secrets are rejected for every record, while full specs and
    nonlegacy deterministic ``etc-`` identities opt into the canonical contract.
    """
    required = {
        "technique_config_id",
        "experiment_id",
        "technique_id",
        "technique_version",
        "implementation_id",
        "preset",
        "parameters",
        "resolved_spec",
        "spec_hash",
    }
    if not required <= set(value):
        raise ValueError("technique configuration is missing canonical fields")
    if contains_secret_material(value):
        raise ValueError("technique configuration must not contain credential material")

    config = plain_json(value)
    if not isinstance(config, dict):  # pragma: no cover - Mapping normalized above
        raise TypeError("technique configuration must be a mapping")
    if (
        not isinstance(experiment_id, str)
        or not experiment_id
        or config["experiment_id"] != experiment_id
        or not isinstance(config["experiment_id"], str)
    ):
        raise ValueError("technique configuration experiment identity is invalid")
    for field in (*_NORMALIZED_CONFIG_PROJECTION_FIELDS[:3], "spec_hash"):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"{field} must be a non-empty string")
    config_id = config["technique_config_id"]
    if config_id is not None and (not isinstance(config_id, str) or not config_id):
        raise ValueError("technique_config_id must be absent or non-empty")
    if config["preset"] is not None and (not isinstance(config["preset"], str) or not config["preset"]):
        raise ValueError("preset must be a non-empty string or null")
    if not isinstance(config["parameters"], dict):
        raise ValueError("parameters must be a mapping")
    resolved_spec = config["resolved_spec"]
    if not isinstance(resolved_spec, dict):
        raise ValueError("resolved_spec must be a mapping")
    complete_spec = is_complete_resolved_spec(resolved_spec)
    is_explicit_legacy_partial = is_legacy_partial_technique_config(
        technique_version=config["technique_version"],
        implementation_id=config["implementation_id"],
        resolved_spec=resolved_spec,
    )
    is_canonical = (
        require_canonical
        or complete_spec
        or (isinstance(config_id, str) and config_id.startswith("etc-") and not is_explicit_legacy_partial)
    )
    if not is_canonical:
        return config
    _validate_resolved_spec(resolved_spec)

    if any(config[field] != plain_json(resolved_spec[field]) for field in _NORMALIZED_CONFIG_PROJECTION_FIELDS):
        raise ValueError("technique configuration projections do not match resolved specification")
    expected_hash = content_hash(resolved_spec)
    if config["spec_hash"] != expected_hash:
        raise ValueError("technique configuration spec hash does not match canonical content")
    expected_id = technique_config_id(experiment_id, resolved_spec)
    if config_id is not None and config_id != expected_id:
        raise ValueError("technique configuration ID does not match canonical content")
    return config


def _validate_resolved_spec(resolved_spec: Mapping[str, object]) -> None:
    required = {
        *_RESOLVED_SPEC_IDENTITY_FIELDS,
        *_RESOLVED_SPEC_MAPPING_FIELDS,
        "preset",
    }
    if not required <= set(resolved_spec):
        raise ValueError("resolved specification is missing canonical fields")
    for field in _RESOLVED_SPEC_IDENTITY_FIELDS:
        if not isinstance(resolved_spec[field], str) or not resolved_spec[field]:
            raise ValueError(f"resolved specification {field} must be non-empty")
    if resolved_spec["preset"] is not None and (
        not isinstance(resolved_spec["preset"], str) or not resolved_spec["preset"]
    ):
        raise ValueError("resolved specification preset must be a string or null")
    for field in _RESOLVED_SPEC_MAPPING_FIELDS:
        if not isinstance(resolved_spec[field], Mapping):
            raise ValueError(f"resolved specification {field} must be a mapping")
    effective_limits = resolved_spec["effective_limits"]
    if "max_rounds" in effective_limits:
        max_rounds = effective_limits["max_rounds"]
        if type(max_rounds) is not int or not 1 <= max_rounds <= 50:
            raise ValueError("resolved specification effective_limits.max_rounds must be an integer between 1 and 50")


def _validate_canonical_technique_config(
    value: Mapping[str, object],
    *,
    experiment_id: str,
) -> dict[str, object]:
    required = {
        "technique_config_id",
        "technique_key",
        *_RESOLVED_SPEC_PROJECTION_FIELDS,
        "resolved_spec",
        "spec_hash",
    }
    if not required <= set(value):
        raise ValueError("canonical technique configuration is incomplete")
    config = validate_normalized_technique_config(
        {
            "technique_config_id": value["technique_config_id"],
            "experiment_id": experiment_id,
            "technique_id": value["technique_id"],
            "technique_version": value["technique_version"],
            "implementation_id": value["implementation_id"],
            "preset": value["preset"],
            "parameters": value["parameters"],
            "resolved_spec": value["resolved_spec"],
            "spec_hash": value["spec_hash"],
        },
        experiment_id=experiment_id,
        require_canonical=True,
    )
    technique_key = value["technique_key"]
    if not isinstance(technique_key, str) or not technique_key:
        raise ValueError("technique_key must be a non-empty string")
    resolved_spec = config["resolved_spec"]
    if any(plain_json(value[field]) != plain_json(resolved_spec[field]) for field in _RESOLVED_SPEC_PROJECTION_FIELDS):
        raise ValueError("technique configuration projections do not match resolved specification")
    canonical = plain_json(value)
    if not isinstance(canonical, dict):  # pragma: no cover - Mapping normalized above
        raise TypeError("canonical technique configuration must be a mapping")
    return canonical


def cell_id(
    experiment_id: str,
    technique_config_id: str,
    prompt_id: str,
    repeat_index: int,
) -> str:
    material = "\x1f".join((experiment_id, technique_config_id, prompt_id, str(repeat_index)))
    return f"cell-{sha256(material.encode('utf-8')).hexdigest()[:32]}"


def canonical_prompt_snapshot(prompt: Mapping[str, object]) -> dict[str, object]:
    raw_gold_answer = prompt.get("gold_answer")
    gold_answer = str(raw_gold_answer).strip() if raw_gold_answer is not None else ""
    return {
        "id": str(prompt.get("id") or prompt.get("prompt_id") or ""),
        "text": str(prompt.get("text") or prompt.get("prompt_text") or ""),
        "category": str(prompt.get("category") if "category" in prompt else prompt.get("prompt_category", "general")),
        "gold_answer": gold_answer or None,
    }


def canonical_dataset_snapshot(
    *,
    dataset_id: str,
    dataset_version: str,
    prompts: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    by_id: dict[str, dict[str, object]] = {}
    for prompt in prompts:
        evidence = canonical_prompt_snapshot(prompt)
        prompt_id = str(evidence["id"])
        if not prompt_id:
            raise ValueError("canonical dataset prompt id must not be empty")
        previous = by_id.get(prompt_id)
        if previous is not None and previous != evidence:
            raise ValueError("canonical dataset prompt evidence is inconsistent")
        by_id[prompt_id] = evidence
    return {
        "dataset_id": dataset_id,
        "version": dataset_version,
        "prompts": [by_id[prompt_id] for prompt_id in sorted(by_id)],
    }


def canonical_dataset_hash(
    *,
    dataset_id: str,
    dataset_version: str,
    prompts: Iterable[Mapping[str, object]],
) -> str:
    return content_hash(
        canonical_dataset_snapshot(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            prompts=prompts,
        )
    )


def canonical_plan_content(
    *,
    experiment_id: str,
    dataset_id: str,
    dataset_version: str,
    dataset_hash: str,
    technique_configs: Iterable[Mapping[str, object]],
    cells: Iterable[Mapping[str, object]],
    repeat_count: int,
) -> dict[str, object]:
    configs = list(technique_configs)
    planned_cells = list(cells)
    return {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "technique_configs": [
            plain_json(item["resolved_spec"])
            for item in sorted(
                configs,
                key=lambda item: str(item["technique_config_id"]),
            )
        ],
        "cells": [
            {
                key: plain_json(item[key])
                for key in (
                    "cell_id",
                    "technique_config_id",
                    "technique_key",
                    "prompt_id",
                    "prompt_text",
                    "prompt_category",
                    "repeat_index",
                    "gold_answer",
                )
            }
            for item in sorted(
                planned_cells,
                key=lambda item: str(item["cell_id"]),
            )
        ],
        "repeat_count": repeat_count,
    }


def validate_canonical_plan_snapshot(
    value: Mapping[str, object],
    *,
    expected_experiment_id: str | None = None,
    expected_dataset_id: str | None = None,
    expected_repeat_count: int | None = None,
    normalized_configs: Iterable[Mapping[str, object]] | None = None,
    normalized_cells: Iterable[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate every derived identity and projection in a canonical plan."""
    if contains_secret_material(value):
        raise ValueError("canonical experiment plan contains credential material")
    plan = plain_json(value)
    if not isinstance(plan, dict):  # pragma: no cover - Mapping normalized above
        raise TypeError("canonical experiment plan must be a mapping")
    required_plan_keys = {
        "experiment_id",
        "dataset_id",
        "dataset_version",
        "dataset_hash",
        "technique_configs",
        "cells",
        "repeat_count",
        "plan_hash",
    }
    if not required_plan_keys <= set(plan):
        raise ValueError("canonical experiment plan is incomplete")
    for field in (
        "experiment_id",
        "dataset_id",
        "dataset_version",
        "dataset_hash",
        "plan_hash",
    ):
        if not isinstance(plan[field], str) or not plan[field]:
            raise ValueError(f"canonical experiment plan {field} must be non-empty")
    experiment_id = plan["experiment_id"]
    dataset_id = plan["dataset_id"]
    repeat_count = plan["repeat_count"]
    if type(repeat_count) is not int or repeat_count < 1:
        raise ValueError("repeat_count must be a positive integer")
    if expected_experiment_id is not None and (
        not isinstance(expected_experiment_id, str)
        or not expected_experiment_id
        or experiment_id != expected_experiment_id
    ):
        raise ValueError("canonical plan experiment identity does not match")
    if expected_dataset_id is not None and (
        not isinstance(expected_dataset_id, str) or not expected_dataset_id or dataset_id != expected_dataset_id
    ):
        raise ValueError("canonical plan dataset identity does not match")
    if expected_repeat_count is not None and (
        type(expected_repeat_count) is not int or expected_repeat_count < 1 or repeat_count != expected_repeat_count
    ):
        raise ValueError("canonical plan repeat count does not match")

    raw_configs = plan["technique_configs"]
    raw_cells = plan["cells"]
    if not isinstance(raw_configs, list) or not raw_configs or not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("canonical plans require configurations and cells")

    configs_by_id: dict[str, dict[str, object]] = {}
    for raw_config in raw_configs:
        if not isinstance(raw_config, Mapping):
            raise ValueError("canonical technique configurations must be mappings")
        config = _validate_canonical_technique_config(
            raw_config,
            experiment_id=experiment_id,
        )
        config_id = config["technique_config_id"]
        if config_id in configs_by_id:
            raise ValueError("canonical technique configuration IDs must be unique")
        configs_by_id[config_id] = config

    cells_by_id: dict[str, dict[str, object]] = {}
    required_cell_keys = {
        "cell_id",
        "experiment_id",
        "technique_config_id",
        "technique_key",
        "prompt_id",
        "prompt_text",
        "prompt_category",
        "repeat_index",
        "gold_answer",
    }
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping) or not required_cell_keys <= set(raw_cell):
            raise ValueError("canonical planned cell is incomplete")
        cell = plain_json(raw_cell)
        if not isinstance(cell, dict):  # pragma: no cover - Mapping normalized above
            raise TypeError("canonical planned cell must be a mapping")
        for field in (
            "cell_id",
            "experiment_id",
            "technique_config_id",
            "technique_key",
            "prompt_id",
        ):
            if not isinstance(cell[field], str) or not cell[field]:
                raise ValueError(f"canonical planned cell {field} must be non-empty")
        if not isinstance(cell["prompt_text"], str) or not isinstance(cell["prompt_category"], str):
            raise ValueError("canonical prompt text and category must be strings")
        if cell["gold_answer"] is not None and not isinstance(cell["gold_answer"], str):
            raise ValueError("canonical gold answer must be a string or null")
        repeat_index = cell["repeat_index"]
        config_id = cell["technique_config_id"]
        config = configs_by_id.get(config_id)
        if (
            type(repeat_index) is not int
            or not 1 <= repeat_index <= repeat_count
            or cell["experiment_id"] != experiment_id
            or config is None
            or cell["technique_key"] != config["technique_key"]
            or cell["cell_id"] != cell_id(experiment_id, config_id, cell["prompt_id"], repeat_index)
        ):
            raise ValueError("canonical planned cell identity is inconsistent")
        prompt = canonical_prompt_snapshot(cell)
        if (
            cell["prompt_text"] != prompt["text"]
            or cell["prompt_category"] != prompt["category"]
            or cell["gold_answer"] != prompt["gold_answer"]
        ):
            raise ValueError("canonical planned cell prompt evidence is inconsistent")
        if cell["cell_id"] in cells_by_id:
            raise ValueError("canonical planned cell IDs must be unique")
        cells_by_id[cell["cell_id"]] = cell

    validate_exact_cell_grid(
        technique_config_ids=configs_by_id,
        cells=cells_by_id.values(),
        repeat_count=repeat_count,
    )
    independent_dataset_hash = canonical_dataset_hash(
        dataset_id=dataset_id,
        dataset_version=plan["dataset_version"],
        prompts=cells_by_id.values(),
    )
    if plan["dataset_hash"] != independent_dataset_hash:
        raise ValueError("canonical dataset hash does not match prompt evidence")
    plan_content = canonical_plan_content(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        dataset_version=plan["dataset_version"],
        dataset_hash=independent_dataset_hash,
        technique_configs=configs_by_id.values(),
        cells=cells_by_id.values(),
        repeat_count=repeat_count,
    )
    if plan["plan_hash"] != content_hash(plan_content):
        raise ValueError("canonical plan hash does not match canonical content")

    if normalized_configs is not None:
        normalized_configs_by_id: dict[str, dict[str, object]] = {}
        for raw_config in normalized_configs:
            config = validate_normalized_technique_config(
                raw_config,
                experiment_id=experiment_id,
                require_canonical=True,
            )
            config_id = config["technique_config_id"]
            if not isinstance(config_id, str) or config_id in normalized_configs_by_id:
                raise ValueError("normalized technique configuration IDs are invalid")
            normalized_configs_by_id[config_id] = config
        if set(normalized_configs_by_id) != set(configs_by_id):
            raise ValueError("normalized technique configurations do not match plan")
        for config_id, normalized in normalized_configs_by_id.items():
            snapshot = configs_by_id[config_id]
            if any(
                normalized[field] != snapshot[field]
                for field in (
                    *_NORMALIZED_CONFIG_PROJECTION_FIELDS,
                    "resolved_spec",
                    "spec_hash",
                )
            ):
                raise ValueError("normalized technique configuration does not match plan")

    if normalized_cells is not None:
        normalized_cells_by_id: dict[str, dict[str, object]] = {}
        for raw_cell in normalized_cells:
            cell = plain_json(raw_cell)
            if not isinstance(cell, dict):
                raise ValueError("normalized planned cells must be mappings")
            required = {
                "cell_id",
                "experiment_id",
                "technique_config_id",
                "technique_key",
                "prompt_id",
                "repeat_index",
            }
            if not required <= set(cell):
                raise ValueError("normalized planned cell is incomplete")
            for field in (
                "cell_id",
                "experiment_id",
                "technique_config_id",
                "technique_key",
                "prompt_id",
            ):
                if not isinstance(cell[field], str) or not cell[field]:
                    raise ValueError(f"normalized planned cell {field} is invalid")
            repeat_index = cell["repeat_index"]
            if (
                type(repeat_index) is not int
                or not 1 <= repeat_index <= repeat_count
                or cell["experiment_id"] != experiment_id
                or cell["technique_config_id"] not in configs_by_id
                or cell["cell_id"]
                != cell_id(
                    experiment_id,
                    cell["technique_config_id"],
                    cell["prompt_id"],
                    repeat_index,
                )
                or cell["cell_id"] in normalized_cells_by_id
            ):
                raise ValueError("normalized planned cell identity is inconsistent")
            normalized_cells_by_id[cell["cell_id"]] = cell
        if set(normalized_cells_by_id) != set(cells_by_id):
            raise ValueError("normalized planned cells do not match plan")
        for cell_id_value, normalized in normalized_cells_by_id.items():
            snapshot = cells_by_id[cell_id_value]
            if any(
                normalized[field] != snapshot[field]
                for field in (
                    "experiment_id",
                    "technique_config_id",
                    "technique_key",
                    "prompt_id",
                    "repeat_index",
                )
            ):
                raise ValueError("normalized planned cell does not match plan")

    return plan


def validate_exact_cell_grid(
    *,
    technique_config_ids: Iterable[str],
    cells: Iterable[Mapping[str, object]],
    repeat_count: int,
) -> None:
    """Require the exact configuration × prompt × repeat Cartesian grid."""
    config_ids = tuple(technique_config_ids)
    if (
        type(repeat_count) is not int
        or repeat_count < 1
        or not config_ids
        or any(not isinstance(config_id, str) or not config_id for config_id in config_ids)
        or len(set(config_ids)) != len(config_ids)
    ):
        raise ValueError("invalid canonical cell-grid dimensions")

    coordinates: list[tuple[str, str, int]] = []
    prompt_ids: set[str] = set()
    for cell in cells:
        config_id = cell.get("technique_config_id")
        prompt_id = cell.get("prompt_id")
        repeat_index = cell.get("repeat_index")
        if (
            not isinstance(config_id, str)
            or not config_id
            or not isinstance(prompt_id, str)
            or not prompt_id
            or type(repeat_index) is not int
            or not 1 <= repeat_index <= repeat_count
        ):
            raise ValueError("invalid canonical planned-cell coordinate")
        coordinates.append((config_id, prompt_id, repeat_index))
        prompt_ids.add(prompt_id)

    required = {
        (config_id, prompt_id, repeat_index)
        for config_id in config_ids
        for prompt_id in prompt_ids
        for repeat_index in range(1, repeat_count + 1)
    }
    actual = set(coordinates)
    if not prompt_ids or len(actual) != len(coordinates) or actual != required:
        raise ValueError("canonical planned cells must equal the configuration × prompt × repeat grid")
