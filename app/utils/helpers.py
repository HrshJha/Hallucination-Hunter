"""
General-purpose helpers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(name: str = "block") -> Generator[None, None, None]:
    """Simple context-manager timer for profiling."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[TIMER] {name}: {elapsed:.3f}s")


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
