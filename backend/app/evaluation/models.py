from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from app.attacklib.base import AttackCase, AttackTechniqueMetadata


def utc_now() -> datetime:
    return datetime.now(UTC)


class TargetKind(StrEnum):
    CHATBOT = "chatbot"
    AGENT = "agent"


class DetectionType(StrEnum):
    REFUSAL_BYPASS = "refusal_bypass"
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    INDIRECT_PROMPT_INJECTION = "indirect_prompt_injection"
    TOOL_MISUSE = "tool_misuse"
    MEMORY_PERSISTENCE = "memory_persistence"
    CROSS_AGENT_PROPAGATION = "cross_agent_propagation"
    AGENT_ARCHITECTURE_DISCLOSURE = "agent_architecture_disclosure"
    # Customer-service detectors
    SCOPE_VIOLATION = "scope_violation"
    EXCESSIVE_VERBOSITY = "excessive_verbosity"
    UNAUTHORIZED_CAPABILITY = "unauthorized_capability"
    ROLE_BOUNDARY_VIOLATION = "role_boundary_violation"
    PII_EXPOSURE = "pii_exposure"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True, frozen=True)
class TechniqueConfig:
    technique_key: str
    enabled_detectors: tuple[DetectionType, ...]
    max_turns: int = 1
    tags: tuple[str, ...] = ()
    parameters: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TechniqueDescriptor:
    metadata: AttackTechniqueMetadata
    config: TechniqueConfig


@dataclass(slots=True, frozen=True)
class ExecutionRequest:
    run_id: str
    target_id: str
    requested_by: str
    case: AttackCase
    config: TechniqueConfig
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True, frozen=True)
class TraceEvent:
    event_type: str
    message: str
    actor: str = "system"
    timestamp: datetime = field(default_factory=utc_now)
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AuditLogEvent:
    action: str
    actor: str
    run_id: str
    timestamp: datetime = field(default_factory=utc_now)
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TargetResponse:
    target_id: str
    target_kind: TargetKind
    final_output: str
    traces: tuple[TraceEvent, ...] = ()
    tool_calls: tuple[str, ...] = ()
    memory_events: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DetectionResult:
    detection_type: DetectionType
    severity: Severity
    matched: bool
    rationale: str
    evidence: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class TechniqueRunResult:
    run_id: str
    technique_key: str
    target_id: str
    response: TargetResponse
    detections: tuple[DetectionResult, ...]
    score: int
    traces: tuple[TraceEvent, ...]
    audit_log: tuple[AuditLogEvent, ...]
    completed_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True, frozen=True)
class ReportSummary:
    run_id: str
    target_id: str
    total_techniques: int
    matched_detections: int
    highest_severity: Severity
    score: int


@dataclass(slots=True, frozen=True)
class EvaluationReport:
    summary: ReportSummary
    results: tuple[TechniqueRunResult, ...]


@dataclass(slots=True, frozen=True)
class ImportedSuiteCase:
    technique_key: str
    unsafe_intent_label: str
    seed_prompt: str | None = None
    seed_code: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ImportedSuite:
    suite_name: str
    source: str
    cases: tuple[ImportedSuiteCase, ...]
