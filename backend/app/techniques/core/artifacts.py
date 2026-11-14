"""Typed, size-bounded runtime artifact creation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping
from uuid import uuid4

from app.persistence.technique_runtime import ArtifactRecord, TechniqueRuntimePersistence
from app.techniques.contracts.execution import CorrelationIds
from app.techniques.core.budgets import BudgetService
from app.techniques.core.events import EventJournal
from app.techniques.core.redaction import RedactionService


class ArtifactPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeArtifact:
    artifact_id: str
    correlation: CorrelationIds
    media_type: str
    content_hash: str
    size_bytes: int
    metadata: Mapping[str, object]


class ArtifactService:
    def __init__(
        self,
        *,
        correlation: CorrelationIds,
        persistence: TechniqueRuntimePersistence,
        events: EventJournal,
        redaction: RedactionService,
        max_size_bytes: int,
        allowed_media_types: frozenset[str],
        budget: BudgetService | None = None,
    ) -> None:
        self._correlation = correlation
        self._persistence = persistence
        self._events = events
        self._redaction = redaction
        self._max_size_bytes = max_size_bytes
        self._allowed_media_types = allowed_media_types
        self._budget = budget
        self._artifacts: dict[str, RuntimeArtifact] = {}

    def create(
        self,
        *,
        media_type: str,
        content: bytes,
        metadata: Mapping[str, object] | None = None,
    ) -> RuntimeArtifact:
        if media_type not in self._allowed_media_types:
            raise ArtifactPolicyError(f"media type '{media_type}' is not allowed")
        if len(content) > self._max_size_bytes:
            raise ArtifactPolicyError(
                f"artifact size {len(content)} exceeds limit {self._max_size_bytes}"
            )
        if self._budget is not None:
            self._budget.reserve(artifact_bytes=len(content))
        artifact_id = str(uuid4())
        content_hash = hashlib.sha256(content).hexdigest()
        safe_metadata = self._redaction.redact_artifact(metadata or {})
        artifact = RuntimeArtifact(
            artifact_id=artifact_id,
            correlation=self._correlation,
            media_type=media_type,
            content_hash=content_hash,
            size_bytes=len(content),
            metadata=safe_metadata,
        )
        self._persistence.save_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                correlation=self._correlation,
                media_type=media_type,
                content_hash=content_hash,
                metadata={**safe_metadata, "size_bytes": len(content)},
            )
        )
        self._artifacts[artifact_id] = artifact
        self._events.emit(
            "artifact.created",
            payload={
                "artifact_id": artifact_id,
                "media_type": media_type,
                "content_hash": content_hash,
                "size_bytes": len(content),
                "metadata": safe_metadata,
            },
        )
        return artifact

    def contains(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def get(self, artifact_id: str) -> RuntimeArtifact | None:
        return self._artifacts.get(artifact_id)
