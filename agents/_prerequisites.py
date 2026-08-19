"""
Local prerequisite graph + topic canonicalisation.

Why this module exists
----------------------
Two related problems live here:

1. **Topic identity is fragile.** The whole system keys off topic name
   strings. ``"Recursion"`` and ``"recursion basics"`` and ``"Recursion "``
   are conceptually the same topic but compare unequal. The Resource
   Agent already canonicalises its knowledge-base keys via lowercasing
   (``_kb_key`` in ``resource_agent.py``); ``canonical_topic_id`` here
   is the shared helper so adaptive_engine, spaced_repetition, and
   issue_detector can all match topics consistently if they ever need
   to.

2. **LLM-suggested prerequisites are volatile.** Asking the model
   "what are the prerequisites for Recursion?" returns plausible but
   non-deterministic output — different runs can disagree on whether
   "the call stack" or "function definitions" is the right prereq.
   ``LOCAL_PREREQUISITES`` provides a small hand-curated graph for
   well-known undergraduate-CS / undergraduate-math topics so the
   Resource Agent can prefer stable, course-aware prereqs and fall
   back to LLM suggestions only when no local entry exists.

Adding to LOCAL_PREREQUISITES
-----------------------------
Use the canonical key (lowercased, whitespace-collapsed). Values are
short prereq labels — keep them under 80 chars each so they pass
``_validate_prereqs_list`` cleanly.
"""

from __future__ import annotations

import re

# Canonical key normalisation: lowercase, collapse all whitespace runs
# (including U+2028 / U+2029 line separators) to a single ASCII space,
# strip leading / trailing whitespace.
_TOPIC_WS = re.compile(r"\s+")


def canonical_topic_id(name: object) -> str:
    """Return a canonical key for a topic name.

    Lowercased + whitespace-collapsed + trimmed. Empty input maps to
    the empty string. Used for topic-string matching across modules
    (KB lookup, prereq lookup, weakness-pattern detector).

    >>> canonical_topic_id("Recursion")
    'recursion'
    >>> canonical_topic_id("  RECURSION   basics  ")
    'recursion basics'
    >>> canonical_topic_id("\\u2028Recursion\\u2028")
    'recursion'
    """
    if name is None:
        return ""
    s = _TOPIC_WS.sub(" ", str(name)).strip().lower()
    return s


# ---------------------------------------------------------------------------
# Local prerequisite graph
# ---------------------------------------------------------------------------
# Maps canonical topic id -> list of prerequisite topic NAMES (display
# form, used downstream for both KB lookup and student-facing
# explanations).
#
# This list is intentionally small. The goal is not to cover every
# topic — that's what the LLM is for — but to anchor the most-common
# undergraduate CS / undergraduate math topics so the Resource Agent
# returns stable prereqs for them across runs.

LOCAL_PREREQUISITES: dict[str, list[str]] = {
    # Intro CS
    "recursion": ["functions", "the call stack", "base cases"],
    "recursion basics": ["functions", "the call stack", "base cases"],
    "loops": ["variables and types", "boolean expressions"],
    "control flow": ["variables and types", "boolean expressions"],
    "functions": ["variables and types", "expressions and statements"],
    "data structures": ["arrays and lists", "iteration"],
    "lists": ["variables and types", "iteration"],
    "dictionaries": ["lists", "hashing fundamentals"],
    "stacks and queues": ["arrays and lists", "abstract data types"],
    "trees": ["recursion", "linked structures"],
    "graphs": ["trees", "abstract data types"],
    "sorting": ["loops", "comparisons"],
    "searching": ["loops", "comparisons"],
    "object-oriented programming": ["functions", "data structures"],
    "oop": ["functions", "data structures"],
    "inheritance": ["object-oriented programming", "polymorphism basics"],
    # Algorithms
    "big-o notation": ["loops", "functions"],
    "dynamic programming": ["recursion", "memoisation"],
    "dijkstra's algorithm": ["graphs", "priority queues"],
    "dijkstra": ["graphs", "priority queues"],
    "depth-first search": ["recursion", "graphs"],
    "dfs": ["recursion", "graphs"],
    "breadth-first search": ["queues", "graphs"],
    "bfs": ["queues", "graphs"],
    # Calculus
    "limits": ["functions", "algebra"],
    "derivatives": ["limits", "functions", "algebra"],
    "integration": ["derivatives", "antiderivatives"],
    "integration by parts": ["integration", "the product rule"],
    "series convergence": ["sequences", "limits"],
    "taylor series": ["derivatives", "series convergence"],
    "polar coordinates": ["trigonometry", "the unit circle"],
    # Linear algebra
    "matrices": ["systems of linear equations"],
    "eigenvalues": ["matrices", "determinants"],
    "vector spaces": ["matrices", "linear independence"],
}


def lookup_local_prerequisites(topic: str) -> list[str]:
    """Return the locally-known prerequisites for a topic, or [] if none.

    Lookup is on the canonical id, so capitalisation / extra
    whitespace / line separators don't prevent a hit.
    """
    return list(LOCAL_PREREQUISITES.get(canonical_topic_id(topic), []))


def merge_prerequisites(
    local: list[str], llm_suggested: list[str], max_total: int = 5
) -> list[str]:
    """Combine locally-known and LLM-suggested prerequisites.

    Local entries come FIRST (they're more reliable / stable). Any
    LLM-suggested entries that don't already appear (case-insensitive)
    get appended after. Result capped at ``max_total`` items.

    >>> merge_prerequisites(["functions", "base cases"], ["functions", "tail recursion"])
    ['functions', 'base cases', 'tail recursion']
    """
    combined: list[str] = []
    seen: set[str] = set()
    for source in (local, llm_suggested):
        if not source:
            continue
        for p in source:
            if not isinstance(p, str):
                continue
            stripped = p.strip()
            if not stripped:
                continue
            key = canonical_topic_id(stripped)
            if key in seen:
                continue
            seen.add(key)
            combined.append(stripped)
            if len(combined) >= max_total:
                return combined
    return combined


__all__ = [
    "LOCAL_PREREQUISITES",
    "canonical_topic_id",
    "lookup_local_prerequisites",
    "merge_prerequisites",
]
