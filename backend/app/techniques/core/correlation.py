"""Context-local correlation ownership for runtime operations."""

from __future__ import annotations

from contextvars import ContextVar, Token
from types import TracebackType

from app.techniques.contracts.execution import CorrelationIds


_CURRENT_CORRELATION: ContextVar[CorrelationIds | None] = ContextVar(
    "technique_correlation",
    default=None,
)


def current_correlation() -> CorrelationIds | None:
    return _CURRENT_CORRELATION.get()


class CorrelationScope:
    def __init__(self, correlation: CorrelationIds) -> None:
        self.correlation = correlation
        self._token: Token[CorrelationIds | None] | None = None

    def __enter__(self) -> CorrelationIds:
        if self._token is not None:
            raise RuntimeError("correlation scope cannot be entered twice")
        self._token = _CURRENT_CORRELATION.set(self.correlation)
        return self.correlation

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is None:
            raise RuntimeError("correlation scope was not entered")
        _CURRENT_CORRELATION.reset(self._token)
        self._token = None
