from __future__ import annotations

from .base import AttackTechnique
from .customer_service_attacks import (
    CodeGenerationTest,
    DataExfiltrationTest,
    DomainBoundaryProbeTest,
    EscalationHandlingTest,
    OffTopicRequestTest,
    PIIHandlingTest,
    RoleBoundaryTest,
    UnauthorizedCapabilityTest,
    VerbosityElicitationTest,
    VerbosityTest,
)
from .code_agent_attacks import (
    AgentArchitectureEnumerationAttack,
    ContextPoisoningAttack,
    MultiFileSeededCodeAgentAttack,
    PromptOnlyCodeAgentAttack,
    SingleFileSeededCodeAgentAttack,
    ToolMisuseAttack,
)
from .pyrit_attacks import (
    ChunkedRequestAttack,
    ContextComplianceAttack,
    FlipAttack,
    ManyShotJailbreakAttack,
    SkeletonKeyAttack,
    ViolentDurianAttack,
    XPIADocumentInjectionAttack,
)
from .text_attacks import (
    AimAttack,
    AutoDanAttack,
    Base64ObfuscationAttack,
    CrescendoAttack,
    CrescendoLiarAttack,
    DanStyleRoleplayAttack,
    DevmodeV2Attack,
    DirectPromptInjectionAttack,
    GCGAttack,
    GPTFuzzAttack,
    GenerationExploitationAttack,
    IndirectPromptInjectionAttack,
    MultilingualObfuscationAttack,
    NestedScenarioAttack,
    PairAttack,
    SystemPromptExtractionAttack,
    TAPAttack,
    ZuluObfuscationAttack,
)


ACTIVE_ATTACKS: tuple[type[AttackTechnique], ...] = (
    DanStyleRoleplayAttack,
    AimAttack,
    DevmodeV2Attack,
    DirectPromptInjectionAttack,
    IndirectPromptInjectionAttack,
    SystemPromptExtractionAttack,
    NestedScenarioAttack,
    Base64ObfuscationAttack,
    ZuluObfuscationAttack,
    MultilingualObfuscationAttack,
    AutoDanAttack,
    GPTFuzzAttack,
    GCGAttack,
    PairAttack,
    TAPAttack,
    CrescendoAttack,
    CrescendoLiarAttack,
    SkeletonKeyAttack,
    ManyShotJailbreakAttack,
    FlipAttack,
    ContextComplianceAttack,
    ChunkedRequestAttack,
    ViolentDurianAttack,
    GenerationExploitationAttack,
    PromptOnlyCodeAgentAttack,
    SingleFileSeededCodeAgentAttack,
    MultiFileSeededCodeAgentAttack,
    ToolMisuseAttack,
    ContextPoisoningAttack,
    AgentArchitectureEnumerationAttack,
    XPIADocumentInjectionAttack,
    # Customer-service testing suite
    OffTopicRequestTest,
    DomainBoundaryProbeTest,
    VerbosityTest,
    VerbosityElicitationTest,
    CodeGenerationTest,
    UnauthorizedCapabilityTest,
    RoleBoundaryTest,
    EscalationHandlingTest,
    PIIHandlingTest,
    DataExfiltrationTest,
)


PRIMARY_ATTACK_REGISTRY: dict[str, type[AttackTechnique]] = {
    attack.key(): attack
    for attack in ACTIVE_ATTACKS
}

ATTACK_REGISTRY: dict[str, type[AttackTechnique]] = dict(PRIMARY_ATTACK_REGISTRY)


def list_attack_techniques() -> list[type[AttackTechnique]]:
    return sorted(PRIMARY_ATTACK_REGISTRY.values(), key=lambda item: item.metadata.name)


def get_attack_technique(key: str) -> type[AttackTechnique]:
    try:
        return ATTACK_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown attack technique: {key}") from exc
