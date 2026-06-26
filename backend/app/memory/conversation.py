"""Centralized conversation state management.

Replaces the per-technique ``conversation: list[dict]`` pattern with a
shared layer that handles storage, retrieval, formatting, and rollback.

Key classes:

- ``MemoryMessage`` — one message in a conversation (immutable dataclass).
- ``ConversationSession`` — a handle to a single conversation thread,
  identified by UUID. Provides append / history / rollback / format.
- ``ConversationMemory`` — top-level manager that creates sessions and
  delegates persistence to a ``StorageBackend`` (in-memory or PostgreSQL).

Techniques use it like this::

    memory = ConversationMemory()  # in-memory by default
    session = memory.create_session(labels={"technique": "crescendo", "run_id": "abc"})

    session.add_system("You are a helpful assistant.")
    session.add_user("Tell me about chemistry.")
    session.add_assistant("Chemistry is the study of matter...")

    messages = session.to_messages()       # list[dict] for the LLM API
    history_str = session.format_history() # human-readable transcript

    session.rollback(2)  # remove last 2 messages (backtracking)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends import StorageBackend


@dataclass(slots=True, frozen=True)
class MemoryMessage:
    """A single message in a conversation."""
    conversation_id: str
    sequence: int
    role: str                # "system" | "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict[str, str] = field(default_factory=dict)


class ConversationSession:
    """Handle to a single conversation thread.

    All state mutations go through the backend, so they are automatically
    persisted (if using PostgreSQL) or kept in-memory (if using InMemory).
    """

    def __init__(
        self,
        conversation_id: str,
        backend: StorageBackend,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self._backend = backend
        self._labels = labels or {}
        self._next_seq = 0

        # Hydrate from backend if resuming an existing conversation.
        existing = backend.get_messages(conversation_id)
        if existing:
            self._next_seq = existing[-1].sequence + 1

    # -- Append messages ----------------------------------------------------

    def _append(self, role: str, content: str, extra_labels: dict[str, str] | None = None) -> MemoryMessage:
        labels = {**self._labels, **(extra_labels or {})}
        msg = MemoryMessage(
            conversation_id=self.conversation_id,
            sequence=self._next_seq,
            role=role,
            content=content,
            labels=labels,
        )
        self._backend.save_message(msg)
        self._next_seq += 1
        return msg

    def add_system(self, content: str) -> MemoryMessage:
        return self._append("system", content)

    def add_user(self, content: str, **extra_labels: str) -> MemoryMessage:
        return self._append("user", content, extra_labels or None)

    def add_assistant(self, content: str, **extra_labels: str) -> MemoryMessage:
        return self._append("assistant", content, extra_labels or None)

    # -- Read ---------------------------------------------------------------

    @property
    def messages(self) -> list[MemoryMessage]:
        """All messages in insertion order."""
        return self._backend.get_messages(self.conversation_id)

    def to_messages(self) -> list[dict[str, str]]:
        """Return messages as ``[{"role": ..., "content": ...}]`` for LLM APIs."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def format_history(self, *, include_system: bool = False) -> str:
        """Human-readable transcript (for planner prompts, logging, etc.)."""
        lines: list[str] = []
        for m in self.messages:
            if m.role == "system" and not include_system:
                continue
            label = "USER" if m.role == "user" else "TARGET" if m.role == "assistant" else "SYSTEM"
            lines.append(f"[{label}]\n{m.content}")
        return "\n\n".join(lines) if lines else "(no messages yet)"

    @property
    def turn_count(self) -> int:
        """Number of user messages (i.e. turns) in the conversation."""
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def last_message(self) -> MemoryMessage | None:
        msgs = self.messages
        return msgs[-1] if msgs else None

    def __len__(self) -> int:
        return len(self.messages)

    # -- Mutate -------------------------------------------------------------

    def rollback(self, n: int = 1) -> int:
        """Remove the last *n* messages (e.g. for backtracking).

        Returns the number actually removed. Also rewinds the sequence
        counter so subsequent appends fill the gap.
        """
        removed = self._backend.delete_last_n(self.conversation_id, n)
        self._next_seq = max(0, self._next_seq - removed)
        return removed

    def clear(self) -> int:
        """Delete all messages in this conversation."""
        count = self._backend.delete_conversation(self.conversation_id)
        self._next_seq = 0
        return count


class ConversationMemory:
    """Top-level manager — creates sessions and delegates to a backend.

    Usage::

        # Development / tests (default)
        memory = ConversationMemory()

        # Production with PostgreSQL
        from app.memory.backends import PostgreSQLBackend
        memory = ConversationMemory(backend=PostgreSQLBackend())

        session = memory.create_session(labels={"technique": "crescendo"})
    """

    def __init__(self, backend: StorageBackend | None = None) -> None:
        if backend is None:
            from .backends import InMemoryBackend
            backend = InMemoryBackend()
        self._backend = backend

    def create_session(
        self,
        *,
        conversation_id: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> ConversationSession:
        """Create (or resume) a conversation session.

        If *conversation_id* is provided and messages already exist for it
        in the backend, the session resumes from where it left off.
        """
        cid = conversation_id or uuid.uuid4().hex
        return ConversationSession(cid, self._backend, labels)

    def get_session(self, conversation_id: str) -> ConversationSession:
        """Resume an existing session by ID."""
        return ConversationSession(conversation_id, self._backend)

    def list_conversations(self) -> list[str]:
        """Return all conversation IDs with at least one message."""
        return self._backend.list_conversations()

    @property
    def backend(self) -> StorageBackend:
        return self._backend
