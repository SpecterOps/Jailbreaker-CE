from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.evaluation.models import ExecutionRequest, TargetKind, TargetResponse


@dataclass(slots=True, frozen=True)
class TargetDescriptor:
    target_id: str
    name: str
    kind: TargetKind
    capabilities: tuple[str, ...]


class TargetAdapter(ABC):
    descriptor: TargetDescriptor

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> TargetResponse:
        """Execute a safe evaluation request against a connected target."""
