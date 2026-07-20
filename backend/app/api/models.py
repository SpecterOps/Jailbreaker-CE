from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class RunSummary:
    run_id: str
    technique_key: str
    target_kind: str
    status: str
    score: int
    asr: int
    matched_detections: int
    detections: tuple[str, ...]
    highest_severity: str
    final_output: str = ""
    conversation: list[dict] | None = None
    error: str = ""


@dataclass(slots=True)
class EvaluationState:
    eval_id: str
    status: str
    mode: str
    target_config: dict[str, str]
    technique_keys: list[str]
    completed_keys: list[str] = field(default_factory=list)
    current_key: str | None = None
    results: list[dict[str, object]] = field(default_factory=list)
    report: dict[str, object] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(slots=True)
class ExperimentState:
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    status: str
    mode: str
    dataset: dict
    technique_keys: list[str]
    target_config: dict[str, str]
    total_cells: int
    repeat_count: int = 1
    max_rounds: int = 5
    experiment_group: str = ""
    completed_cells: int = 0
    current_technique: str | None = None
    current_prompt_id: str | None = None
    current_repeat_index: int | None = None
    cell_results: list[dict] = field(default_factory=list)
    report: dict | None = None
    error: str | None = None
    stop_requested: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
