"""
Lightweight in-process telemetry counters.

Why this module exists
----------------------
The Diagnostics tab and the issue detector already surface a lot of
runtime information (issue count, agent messages, RL state, SR
schedule, message log). What's been missing is a **typed, queryable
count of each instrumented event**:

  - LLM calls made
  - LLM retries
  - Fallback quizzes used
  - Fallback feedback used
  - Resource Agent cache hits / misses
  - Profile save failures
  - Prompt truncations
  - Bad quiz generations rejected (verifier or schema)
  - Material schema-validation failures

These are the kind of numbers you'd want to look at after a
production run to answer "is the LLM behaving reliably?". The code
already logs / handles most of these conditions; ``Telemetry`` makes
them visible to the Diagnostics tab and to any external observer.

Design notes
------------
- Process-singleton via the module-level ``COUNTERS`` instance — it's
  fine for a single Streamlit process. A test fixture that monkey-
  patches the singleton is sufficient for isolation.
- Thread-safe (``threading.Lock``) so the MessageBus / agent
  callbacks running across worker threads can't lose increments.
- Stores monotonic counts since process start. ``snapshot()`` returns
  a plain dict for the diagnostics renderer; ``reset()`` zeroes
  everything (used by the privacy "reset diagnostics" control and by
  tests).
"""

from __future__ import annotations

import threading
from typing import Iterable


# Canonical counter names. Centralised here so we have one place to
# look when adding a new instrumentation point — a string typo at the
# call site would otherwise create a phantom counter that the
# diagnostics tab never displays.
LLM_CALLS = "llm.calls"
LLM_RETRIES = "llm.retries"
LLM_FALLBACK_USED = "llm.fallback_used"

QUIZ_GENERATED = "quiz.generated"
QUIZ_FALLBACK_USED = "quiz.fallback_used"
QUIZ_SCHEMA_REJECTED = "quiz.schema_rejected"
QUIZ_VERIFIER_REJECTED = "quiz.verifier_rejected"

MATERIAL_GENERATED = "material.generated"
MATERIAL_SCHEMA_REJECTED = "material.schema_rejected"
MATERIAL_FALLBACK_USED = "material.fallback_used"
MATERIAL_CACHE_HIT = "material.cache_hit"
MATERIAL_CACHE_MISS = "material.cache_miss"

PREREQ_LOCAL_HIT = "prereq.local_hit"
PREREQ_LLM_FALLBACK = "prereq.llm_fallback"

PROFILE_SAVE_OK = "profile.save_ok"
PROFILE_SAVE_TIMEOUT = "profile.save_timeout"
PROFILE_SAVE_SERIALIZE_FAIL = "profile.save_serialize_fail"

PROMPT_TRUNCATED = "prompt.truncated"

# Display ordering for the Diagnostics renderer.
COUNTER_DISPLAY_ORDER: tuple[str, ...] = (
    LLM_CALLS,
    LLM_RETRIES,
    LLM_FALLBACK_USED,
    QUIZ_GENERATED,
    QUIZ_FALLBACK_USED,
    QUIZ_SCHEMA_REJECTED,
    QUIZ_VERIFIER_REJECTED,
    MATERIAL_GENERATED,
    MATERIAL_CACHE_HIT,
    MATERIAL_CACHE_MISS,
    MATERIAL_SCHEMA_REJECTED,
    MATERIAL_FALLBACK_USED,
    PREREQ_LOCAL_HIT,
    PREREQ_LLM_FALLBACK,
    PROFILE_SAVE_OK,
    PROFILE_SAVE_TIMEOUT,
    PROFILE_SAVE_SERIALIZE_FAIL,
    PROMPT_TRUNCATED,
)


class Telemetry:
    """Thread-safe monotonic counter store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def incr(self, name: str, by: int = 1) -> None:
        """Increment a counter. Unknown counter names are accepted —
        a typo at a call site creates a new entry rather than raising,
        because we'd rather see the bad data than crash a quiz turn."""
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + int(by)

    def get(self, name: str) -> int:
        with self._lock:
            return self._counts.get(name, 0)

    def snapshot(self, names: Iterable[str] | None = None) -> dict[str, int]:
        """Return a copy of the counters.

        Args:
            names: If provided, restrict the snapshot to these
                   counter names (in order, with zero defaults for
                   never-incremented entries). Useful for the
                   diagnostics renderer which wants a stable display
                   order.
        """
        with self._lock:
            if names is None:
                return dict(self._counts)
            return {n: self._counts.get(n, 0) for n in names}

    def reset(self) -> None:
        """Zero every counter. Used by the privacy "reset diagnostics"
        control and by tests that want a clean slate."""
        with self._lock:
            self._counts.clear()


# Module-level singleton — a single Streamlit process shares one
# Telemetry instance across all tabs / sessions / agents.
COUNTERS = Telemetry()


__all__ = [
    "Telemetry",
    "COUNTERS",
    "COUNTER_DISPLAY_ORDER",
    "LLM_CALLS",
    "LLM_RETRIES",
    "LLM_FALLBACK_USED",
    "QUIZ_GENERATED",
    "QUIZ_FALLBACK_USED",
    "QUIZ_SCHEMA_REJECTED",
    "QUIZ_VERIFIER_REJECTED",
    "MATERIAL_GENERATED",
    "MATERIAL_SCHEMA_REJECTED",
    "MATERIAL_FALLBACK_USED",
    "MATERIAL_CACHE_HIT",
    "MATERIAL_CACHE_MISS",
    "PREREQ_LOCAL_HIT",
    "PREREQ_LLM_FALLBACK",
    "PROFILE_SAVE_OK",
    "PROFILE_SAVE_TIMEOUT",
    "PROFILE_SAVE_SERIALIZE_FAIL",
    "PROMPT_TRUNCATED",
]
