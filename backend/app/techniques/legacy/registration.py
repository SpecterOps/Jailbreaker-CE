"""Translate legacy attack descriptors into the modular catalog contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from app.attacklib.base import AttackTechnique
from app.attacklib.config import default_technique_config
from app.attacklib.registry import list_attack_techniques
from app.techniques.contracts.definition import (
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
)
from app.techniques.core.catalog import RegisteredTechnique, TechniqueCatalog


LEGACY_TECHNIQUE_VERSION = "0.2.0"
LEGACY_IMPLEMENTATION_ID = "legacy"


def definition_from_legacy(
    technique_cls: type[AttackTechnique],
) -> TechniqueDefinition:
    metadata = technique_cls.metadata
    config = default_technique_config(metadata.key)
    class_import_path = _class_import_path(technique_cls)
    capabilities = {
        TechniqueCapability.ITERATIVE
        if metadata.multi_turn or metadata.requires_feedback_loop
        else TechniqueCapability.SINGLE_SHOT
    }
    if metadata.requires_image:
        capabilities.add(TechniqueCapability.MULTIMODAL)
    if "population-based" in metadata.tags:
        capabilities.add(TechniqueCapability.POPULATION_SEARCH)
        capabilities.add(TechniqueCapability.PARALLEL_BRANCHES)
    elif "parallel-branches" in metadata.tags:
        capabilities.add(TechniqueCapability.PARALLEL_BRANCHES)

    inputs = [InputDefinition(id="intent", kind="text", required=True)]
    if metadata.requires_seed_prompt:
        inputs.append(InputDefinition(id="seed_prompt", kind="text", required=True))
    if metadata.requires_seed_code:
        inputs.append(InputDefinition(id="seed_code", kind="text", required=True))
    if metadata.requires_image:
        inputs.append(
            InputDefinition(
                id="image",
                kind="artifact",
                required=True,
                media_types=("image/*",),
                max_count=1,
            )
        )

    model_roles = [
        ModelRoleDefinition(id="target", required=True),
        ModelRoleDefinition(id="target_judge", required=False),
    ]
    if metadata.requires_attacker_llm:
        model_roles.append(ModelRoleDefinition(id="attacker", required=True))

    detector_ids = tuple(detector.value for detector in config.enabled_detectors)
    legacy_extension = {
        "surface": metadata.surface.value,
        "automation_level": metadata.automation_level.value,
        "source_paper": metadata.source_paper,
        "requires_image": metadata.requires_image,
        "requires_seed_prompt": metadata.requires_seed_prompt,
        "requires_seed_code": metadata.requires_seed_code,
        "requires_white_box_access": metadata.requires_white_box_access,
        "requires_feedback_loop": metadata.requires_feedback_loop,
        "requires_attacker_llm": metadata.requires_attacker_llm,
        "requires_judge_per_round": metadata.requires_judge_per_round,
        "multi_turn": metadata.multi_turn,
        "detector_defaults": detector_ids,
        "default_max_turns": config.max_turns,
        "class_import_path": class_import_path,
    }
    return TechniqueDefinition(
        identity=TechniqueIdentity(
            id=metadata.key,
            version=LEGACY_TECHNIQUE_VERSION,
            name=metadata.name,
            family=metadata.family,
        ),
        summary=metadata.summary,
        capabilities=frozenset(capabilities),
        tags=metadata.tags,
        target_surfaces=(metadata.surface.value,),
        inputs=tuple(inputs),
        model_roles=tuple(model_roles),
        histories=(
            HistoryDefinition(
                id="audit_history",
                visible_to=("target_judge",),
            ),
        ),
        judges=(
            JudgeDefinition(
                id="target_judge",
                schedule=(
                    JudgeSchedule.EVERY_ROUND
                    if metadata.requires_judge_per_round
                    else JudgeSchedule.RUN_END
                ),
                purposes=frozenset({JudgePurpose.OFFICIAL_SCORE}),
                role="target_judge",
                required=False,
            ),
        ),
        detectors=tuple(
            DetectorDefinition(id=detector_id, inputs=("target_response",))
            for detector_id in detector_ids
        ),
        seed=SeedDefinition(
            strategy=(
                SeedStrategy.SUPPLIED
                if metadata.requires_seed_prompt or metadata.requires_seed_code
                else SeedStrategy.NONE
            )
        ),
        result_policy=ResultPolicy(
            official_judge="target_judge",
            final_response="target_response",
            success_threshold=7,
        ),
        presentation={
            "conversation_format": (
                "generation"
                if TechniqueCapability.POPULATION_SEARCH in capabilities
                else "tree"
                if TechniqueCapability.PARALLEL_BRANCHES in capabilities
                else "rounds"
                if metadata.multi_turn or metadata.requires_feedback_loop
                else "generation"
            )
        },
        recommended_limits={"max_rounds": config.max_turns},
        source={"kind": "legacy", "paper": metadata.source_paper},
        extensions={"legacy": legacy_extension},
    )


def register_legacy_techniques(catalog: TechniqueCatalog) -> None:
    for technique_cls in list_attack_techniques():
        definition = definition_from_legacy(technique_cls)
        class_import_path = _class_import_path(technique_cls)
        implementation = TechniqueImplementation(
            id=LEGACY_IMPLEMENTATION_ID,
            technique_id=definition.identity.id,
            technique_version=definition.identity.version,
            kind=ImplementationKind.LEGACY,
            entrypoint=class_import_path,
            extensions={"adapter": "attacklib"},
        )
        registered = RegisteredTechnique(
            definition=definition,
            implementation=implementation,
            manifest_hash=_legacy_manifest_hash(technique_cls),
        )
        catalog.register(registered)
        catalog.set_default(
            definition.identity.id,
            implementation_id=LEGACY_IMPLEMENTATION_ID,
            version=LEGACY_TECHNIQUE_VERSION,
        )


def build_default_catalog(*, discover_modular: bool = True) -> TechniqueCatalog:
    catalog = TechniqueCatalog()
    register_legacy_techniques(catalog)
    # Modular discovery is introduced in Checkpoint 5. Accepting the flag now
    # keeps this construction API stable while all defaults remain legacy.
    _ = discover_modular
    return catalog


def _legacy_manifest_hash(technique_cls: type[AttackTechnique]) -> str:
    metadata = technique_cls.metadata
    config = default_technique_config(metadata.key)
    payload = {
        "adapter_contract": "legacy-v1",
        "class_import_path": _class_import_path(technique_cls),
        "metadata": asdict(metadata),
        "config": {
            "enabled_detectors": [item.value for item in config.enabled_detectors],
            "max_turns": config.max_turns,
            "tags": list(config.tags),
            "parameters": config.parameters,
        },
        "technique_version": LEGACY_TECHNIQUE_VERSION,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _class_import_path(technique_cls: type[AttackTechnique]) -> str:
    return f"{technique_cls.__module__}:{technique_cls.__qualname__}"
