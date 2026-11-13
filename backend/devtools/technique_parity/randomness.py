"""Serialize synchronous parity transactions that seed Python's global RNG."""

from __future__ import annotations

import random
import threading
from collections.abc import Iterator
from contextlib import contextmanager


_SEEDED_RANDOM_LOCK = threading.RLock()


@contextmanager
def seeded_random(seed: int) -> Iterator[None]:
    """Restore caller state even on failure; cooperating seeded runs share one lock.

    The complete synchronous arm must finish within this context. This does not
    isolate unrelated code using global random, or support concurrent async
    scenarios sharing an event loop.
    """

    with _SEEDED_RANDOM_LOCK:
        state = random.getstate()
        try:
            random.seed(seed)
            yield
        finally:
            random.setstate(state)
