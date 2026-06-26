from __future__ import annotations

from dataclasses import dataclass, field

from app.evaluation.models import ExecutionRequest, TargetKind, TargetResponse, TraceEvent

from .base import TargetAdapter, TargetDescriptor


@dataclass(slots=True)
class RecordedChatbotAdapter(TargetAdapter):
    descriptor: TargetDescriptor
    recorded_output: str = "Target output unavailable."
    recorded_traces: tuple[TraceEvent, ...] = field(default_factory=tuple)

    def execute(self, request: ExecutionRequest) -> TargetResponse:
        return TargetResponse(
            target_id=self.descriptor.target_id,
            target_kind=TargetKind.CHATBOT,
            final_output=self.recorded_output,
            traces=self.recorded_traces,
            metadata={
                "adapter": "recorded-chatbot",
                "technique_key": request.case.technique_key,
            },
        )
