from __future__ import annotations

from dataclasses import dataclass, field

from app.evaluation.models import ExecutionRequest, TargetKind, TargetResponse, TraceEvent

from .base import TargetAdapter, TargetDescriptor


@dataclass(slots=True)
class RecordedAgentAdapter(TargetAdapter):
    descriptor: TargetDescriptor
    recorded_output: str = "Agent output unavailable."
    recorded_traces: tuple[TraceEvent, ...] = field(default_factory=tuple)
    tool_calls: tuple[str, ...] = field(default_factory=tuple)
    memory_events: tuple[str, ...] = field(default_factory=tuple)

    def execute(self, request: ExecutionRequest) -> TargetResponse:
        return TargetResponse(
            target_id=self.descriptor.target_id,
            target_kind=TargetKind.AGENT,
            final_output=self.recorded_output,
            traces=self.recorded_traces,
            tool_calls=self.tool_calls,
            memory_events=self.memory_events,
            metadata={
                "adapter": "recorded-agent",
                "technique_key": request.case.technique_key,
            },
        )
