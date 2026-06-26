from __future__ import annotations

from app.evaluation.models import DetectionType, TechniqueConfig, TechniqueDescriptor

from .registry import get_attack_technique, list_attack_techniques


DEFAULT_DETECTORS_BY_SURFACE: dict[str, tuple[DetectionType, ...]] = {
    "text": (
        DetectionType.REFUSAL_BYPASS,
        DetectionType.SYSTEM_PROMPT_LEAKAGE,
        DetectionType.INDIRECT_PROMPT_INJECTION,
    ),
    "code-agent": (
        DetectionType.REFUSAL_BYPASS,
        DetectionType.SYSTEM_PROMPT_LEAKAGE,
        DetectionType.INDIRECT_PROMPT_INJECTION,
        DetectionType.TOOL_MISUSE,
        DetectionType.MEMORY_PERSISTENCE,
        DetectionType.CROSS_AGENT_PROPAGATION,
        DetectionType.AGENT_ARCHITECTURE_DISCLOSURE,
    ),
}

# Customer-service techniques declare family="customer-service" but use
# surface=TEXT. We map per-technique detectors instead of per-surface.
CS_DETECTORS_BY_TECHNIQUE: dict[str, tuple[DetectionType, ...]] = {
    "cs_off_topic_request": (DetectionType.SCOPE_VIOLATION,),
    "cs_domain_boundary_probe": (DetectionType.SCOPE_VIOLATION,),
    "cs_verbosity": (DetectionType.EXCESSIVE_VERBOSITY,),
    "cs_verbosity_elicitation": (DetectionType.EXCESSIVE_VERBOSITY,),
    "cs_code_generation": (DetectionType.UNAUTHORIZED_CAPABILITY, DetectionType.SCOPE_VIOLATION),
    "cs_unauthorized_capability": (DetectionType.UNAUTHORIZED_CAPABILITY, DetectionType.SCOPE_VIOLATION),
    "cs_role_boundary": (DetectionType.ROLE_BOUNDARY_VIOLATION,),
    "cs_escalation_handling": (DetectionType.ROLE_BOUNDARY_VIOLATION,),
    "cs_pii_handling": (DetectionType.PII_EXPOSURE,),
    "cs_data_exfiltration": (DetectionType.PII_EXPOSURE, DetectionType.SYSTEM_PROMPT_LEAKAGE),
}


def default_technique_config(technique_key: str) -> TechniqueConfig:
    technique_cls = get_attack_technique(technique_key)
    # CS techniques use per-technique detectors instead of per-surface.
    if technique_key in CS_DETECTORS_BY_TECHNIQUE:
        detectors = CS_DETECTORS_BY_TECHNIQUE[technique_key]
    else:
        detectors = DEFAULT_DETECTORS_BY_SURFACE[technique_cls.metadata.surface.value]
    max_turns = 3 if technique_cls.metadata.multi_turn else 1
    return TechniqueConfig(
        technique_key=technique_key,
        enabled_detectors=detectors,
        max_turns=max_turns,
        tags=technique_cls.metadata.tags,
    )


def list_technique_descriptors() -> list[TechniqueDescriptor]:
    descriptors = []
    for technique_cls in list_attack_techniques():
        descriptors.append(
            TechniqueDescriptor(
                metadata=technique_cls.metadata,
                config=default_technique_config(technique_cls.metadata.key),
            )
        )
    return descriptors
