"""Storage backends for conversation memory.

``StorageBackend`` defines the interface. Two implementations:

- ``InMemoryBackend`` — fast, zero-dependency, for dev/tests. State lives
  only in the process; lost on restart.
- ``PostgreSQLBackend`` — production persistence using your existing
  PostgreSQL 16 instance. Requires ``psycopg`` (optional dependency).

Techniques interact only with ``ConversationMemory``, never with backends
directly, so switching from in-memory to Postgres is a one-line config
change.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .conversation import MemoryMessage

log = logging.getLogger("memory.backends")


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class StorageBackend(ABC):
    """Interface for persisting conversation messages."""

    @abstractmethod
    def save_message(self, msg: MemoryMessage) -> None:
        """Persist a single message."""

    @abstractmethod
    def get_messages(self, conversation_id: str) -> list[MemoryMessage]:
        """Retrieve all messages for a conversation, ordered by sequence."""

    @abstractmethod
    def delete_last_n(self, conversation_id: str, n: int) -> int:
        """Delete the last *n* messages from a conversation.

        Returns the number actually deleted (may be < n if conversation
        has fewer messages).
        """

    @abstractmethod
    def list_conversations(self) -> list[str]:
        """Return all conversation IDs that have at least one message."""

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> int:
        """Delete all messages for a conversation. Returns count deleted."""


# ---------------------------------------------------------------------------
# In-memory backend (dev / tests)
# ---------------------------------------------------------------------------

class InMemoryBackend(StorageBackend):
    """Dictionary-backed storage — fast but not persistent."""

    def __init__(self) -> None:
        # conversation_id -> list[MemoryMessage] in insertion order
        self._store: dict[str, list[MemoryMessage]] = {}

    def save_message(self, msg: MemoryMessage) -> None:
        self._store.setdefault(msg.conversation_id, []).append(msg)

    def get_messages(self, conversation_id: str) -> list[MemoryMessage]:
        return list(self._store.get(conversation_id, []))

    def delete_last_n(self, conversation_id: str, n: int) -> int:
        msgs = self._store.get(conversation_id)
        if not msgs:
            return 0
        count = min(n, len(msgs))
        del msgs[-count:]
        return count

    def list_conversations(self) -> list[str]:
        return [cid for cid, msgs in self._store.items() if msgs]

    def delete_conversation(self, conversation_id: str) -> int:
        msgs = self._store.pop(conversation_id, [])
        return len(msgs)


# ---------------------------------------------------------------------------
# PostgreSQL backend (production)
# ---------------------------------------------------------------------------

class PostgreSQLBackend(StorageBackend):
    """PostgreSQL-backed storage using ``psycopg`` (v3).

    Expects a ``DATABASE_URL`` environment variable or an explicit
    ``dsn`` parameter.  The table is auto-created on first use via
    ``ensure_table()``.

    Connection handling: one connection per backend instance, with
    autocommit. For concurrent access (e.g. async runner), instantiate
    one backend per thread or use a connection pool.
    """

    TABLE = "conversation_messages"

    def __init__(self, dsn: str | None = None) -> None:
        try:
            import psycopg  # noqa: F811
        except ImportError:
            raise ImportError(
                "PostgreSQLBackend requires the 'psycopg' package. "
                "Install it with: uv sync --extra postgres"
            ) from None

        self._dsn = dsn or os.environ.get(
            "DATABASE_URL",
            "postgresql://jailbreaker:jailbreaker@localhost:5432/jailbreaker",
        )
        self._conn = psycopg.connect(self._dsn, autocommit=True)
        self.ensure_table()
        log.info("PostgreSQLBackend connected to %s", self._dsn.split("@")[-1])

    # -- Schema management --------------------------------------------------

    def ensure_table(self) -> None:
        """Create the messages table if it doesn't exist."""
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id              BIGSERIAL PRIMARY KEY,
                conversation_id TEXT        NOT NULL,
                sequence        INTEGER     NOT NULL,
                role            TEXT        NOT NULL,
                content         TEXT        NOT NULL DEFAULT '',
                timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
                labels          JSONB       NOT NULL DEFAULT '{{}}'::jsonb,

                UNIQUE (conversation_id, sequence)
            );
        """)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_conv_id
            ON {self.TABLE} (conversation_id);
        """)

    # -- CRUD ---------------------------------------------------------------

    def save_message(self, msg: MemoryMessage) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {self.TABLE}
                (conversation_id, sequence, role, content, timestamp, labels)
            VALUES
                (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (conversation_id, sequence)
            DO UPDATE SET role = EXCLUDED.role,
                          content = EXCLUDED.content,
                          timestamp = EXCLUDED.timestamp,
                          labels = EXCLUDED.labels
            """,
            (
                msg.conversation_id,
                msg.sequence,
                msg.role,
                msg.content,
                msg.timestamp.isoformat(),
                _labels_to_json(msg.labels),
            ),
        )

    def get_messages(self, conversation_id: str) -> list[MemoryMessage]:
        rows = self._conn.execute(
            f"""
            SELECT conversation_id, sequence, role, content, timestamp, labels
            FROM {self.TABLE}
            WHERE conversation_id = %s
            ORDER BY sequence
            """,
            (conversation_id,),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

    def delete_last_n(self, conversation_id: str, n: int) -> int:
        # Find the sequence numbers of the last N messages.
        result = self._conn.execute(
            f"""
            DELETE FROM {self.TABLE}
            WHERE id IN (
                SELECT id FROM {self.TABLE}
                WHERE conversation_id = %s
                ORDER BY sequence DESC
                LIMIT %s
            )
            """,
            (conversation_id, n),
        )
        return result.rowcount

    def list_conversations(self) -> list[str]:
        rows = self._conn.execute(
            f"SELECT DISTINCT conversation_id FROM {self.TABLE}"
        ).fetchall()
        return [row[0] for row in rows]

    def delete_conversation(self, conversation_id: str) -> int:
        result = self._conn.execute(
            f"DELETE FROM {self.TABLE} WHERE conversation_id = %s",
            (conversation_id,),
        )
        return result.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _labels_to_json(labels: dict[str, str]) -> str:
    import json
    return json.dumps(labels)


def _row_to_message(row: tuple) -> MemoryMessage:
    """Convert a database row to a MemoryMessage."""
    conv_id, sequence, role, content, timestamp, labels = row
    # timestamp may already be a datetime or may be a string
    if isinstance(timestamp, str):
        ts = datetime.fromisoformat(timestamp)
    else:
        ts = timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    # labels may be a dict already (psycopg auto-parses JSONB) or a string
    if isinstance(labels, str):
        import json
        labels = json.loads(labels)
    return MemoryMessage(
        conversation_id=conv_id,
        sequence=sequence,
        role=role,
        content=content,
        timestamp=ts,
        labels=labels or {},
    )
