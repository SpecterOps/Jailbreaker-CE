"""Thread-safe cooperative cancellation for technique executions."""

from __future__ import annotations

import threading


class CancellationError(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "execution cancelled") -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason
                self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancellationError(self.reason or "execution cancelled")
