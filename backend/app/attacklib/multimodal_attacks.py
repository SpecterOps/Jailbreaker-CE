from __future__ import annotations

from .base import (
    AttackBuildRequest,
    AttackCase,
    AttackSurface,
    AttackTechnique,
    AttackTechniqueMetadata,
    AutomationLevel,
)


class _SafeMultimodalAttack(AttackTechnique):
    guidance: str = ""

    def build_case(self, request: AttackBuildRequest) -> AttackCase:
        return self._build_safe_case(
            request,
            instructions=self.guidance,
            assets={"image_reference": request.image_reference or ""},
            notes=(
                "This class models a multimodal attack family for authorized evaluation.",
                "It does not create image prompts or attack samples.",
            ),
        )


class FigStepAttack(_SafeMultimodalAttack):
    metadata = AttackTechniqueMetadata(
        key="figstep",
        name="FigStep",
        surface=AttackSurface.MULTIMODAL,
        family="manual-black-box",
        summary="Black-box multimodal attack family using harmful text embedded in images.",
        source_paper="JPRO",
        automation_level=AutomationLevel.MANUAL,
        requires_image=True,
        tags=("image-text", "black-box"),
    )
    guidance = "Store image reference provenance and evaluation constraints only."


class VisualRoleplayAttack(_SafeMultimodalAttack):
    metadata = AttackTechniqueMetadata(
        key="visual_roleplay",
        name="Visual-Roleplay",
        surface=AttackSurface.MULTIMODAL,
        family="manual-black-box",
        summary="Visual role-play framing adapted from text jailbreak techniques.",
        source_paper="JPRO",
        automation_level=AutomationLevel.MANUAL,
        requires_image=True,
        tags=("role-play", "visual"),
    )
    guidance = "Represent the scenario type and moderation expectations only."


class MirageNarrativeAttack(_SafeMultimodalAttack):
    metadata = AttackTechniqueMetadata(
        key="mirage_narrative",
        name="MIRAGE-Style Narrative Attack",
        surface=AttackSurface.MULTIMODAL,
        family="manual-black-box",
        summary="Narrative visual storytelling approach that distributes content across turns.",
        source_paper="JPRO",
        automation_level=AutomationLevel.SEMI_AUTOMATIC,
        requires_image=True,
        multi_turn=True,
        tags=("narrative", "multi-turn"),
    )
    guidance = "Capture sequence metadata for a narrative-style multimodal case."


class JPRORoleplaySituationAttack(_SafeMultimodalAttack):
    metadata = AttackTechniqueMetadata(
        key="jpro_roleplay_situation",
        name="JPRO Role-Playing and Situational Setting",
        surface=AttackSurface.MULTIMODAL,
        family="multi-agent-multimodal",
        summary="Scenario framing tactic from JPRO's planner tactic library.",
        source_paper="JPRO",
        automation_level=AutomationLevel.AUTOMATIC,
        requires_image=True,
        multi_turn=True,
        tags=("jpro", "role-play", "planner"),
    )
    guidance = "Model tactic selection and case metadata without generating prompts."


class JPROMetaphoricalFrameworkAttack(_SafeMultimodalAttack):
    metadata = AttackTechniqueMetadata(
        key="jpro_metaphorical_framework",
        name="JPRO Metaphorical Framework",
        surface=AttackSurface.MULTIMODAL,
        family="multi-agent-multimodal",
        summary="Indirect metaphor-based multimodal tactic from JPRO.",
        source_paper="JPRO",
        automation_level=AutomationLevel.AUTOMATIC,
        requires_image=True,
        multi_turn=True,
        tags=("jpro", "metaphor"),
    )
    guidance = "Track indirect framing as metadata only."


class JPROIterativeImageInquiryAttack(_SafeMultimodalAttack):
    metadata = AttackTechniqueMetadata(
        key="jpro_iterative_image_inquiry",
        name="JPRO Iterative Image Inquiry",
        surface=AttackSurface.MULTIMODAL,
        family="multi-agent-multimodal",
        summary="Multi-turn tactic that distributes content across repeated image-grounded queries.",
        source_paper="JPRO",
        automation_level=AutomationLevel.AUTOMATIC,
        requires_image=True,
        multi_turn=True,
        tags=("jpro", "iterative", "multi-turn"),
    )
    guidance = "Use this to configure turn budgets and trace expectations."


class JPROMaliciousContentSegmentationAttack(_SafeMultimodalAttack):
    metadata = AttackTechniqueMetadata(
        key="jpro_malicious_content_segmentation",
        name="JPRO Malicious Content Segmentation",
        surface=AttackSurface.MULTIMODAL,
        family="multi-agent-multimodal",
        summary="Splits risky semantics between image and text so neither looks unsafe alone.",
        source_paper="JPRO",
        automation_level=AutomationLevel.AUTOMATIC,
        requires_image=True,
        multi_turn=True,
        tags=("jpro", "segmentation", "cross-modal"),
    )
    guidance = "Use for metadata and evaluation planning only."
