"""
Centralised tunable constants for the AI Study Tutor.

Why this module exists
----------------------
Constants used to live as module-private ``_MAX_*`` values scattered
across ``agents/``, ``utils/``, and ``app.py``. Every "what's the cap on
the knowledge base?" question required a grep. Hoisting them here means:

- One place to tune for a different deployment (e.g. raise
  ``MAX_KNOWLEDGE_BASE_SIZE`` for a power user, or drop
  ``LOCK_TIMEOUT_SECONDS`` for a stricter test environment).
- Each constant can be optionally overridden via an environment
  variable, which makes per-environment tuning a matter of editing
  ``.env`` rather than touching code.
- Tests get a single import surface to monkey-patch when they want to
  exercise edge cases (e.g. force the KB eviction path with
  ``MAX_KNOWLEDGE_BASE_SIZE = 3``).

Backward compatibility
----------------------
Each module that previously held a private ``_FOO`` re-exports the same
name pointing at the value here, so legacy imports keep working.
"""

from __future__ import annotations

import os
from typing import Any


def _env_int(name: str, default: int) -> int:
    """Read ``name`` from the environment as an int, falling back to ``default``.

    A malformed value logs a warning via the runtime's normal logger
    (we don't import the logger here to keep ``config.py`` import-cheap)
    and falls back to the default rather than crashing app startup.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# LLM / prompt-budget caps  (consumed by agents/tutor_agent.py)
# ---------------------------------------------------------------------------

# Conservative chars/token estimate. The 4-chars/token heuristic
# undercounts tokens for code/JSON-heavy chat (where the real ratio is
# closer to ~2 chars/token). 3 errs on the side of trimming earlier
# (cheap) while still passing most English chat through intact.
CHARS_PER_TOKEN: int = _env_int("STUDY_TUTOR_CHARS_PER_TOKEN", 3)

# Conversation history token budget — leaves ~150k for system prompt + output.
MAX_HISTORY_TOKENS: int = _env_int("STUDY_TUTOR_MAX_HISTORY_TOKENS", 50_000)

# Hard caps on dynamic prompt sections so a verbose materials blob or
# a large strategy entry can't push the system prompt past the context
# window. Fail loud here instead of as a misleading BadRequestError.
MAX_MATERIALS_CHARS: int = _env_int("STUDY_TUTOR_MAX_MATERIALS_CHARS", 4_000)
MAX_STRATEGY_CHARS: int = _env_int("STUDY_TUTOR_MAX_STRATEGY_CHARS", 2_000)
MAX_SYSTEM_PROMPT_CHARS: int = _env_int("STUDY_TUTOR_MAX_SYSTEM_PROMPT_CHARS", 40_000)
MAX_USER_MESSAGE_CHARS: int = _env_int("STUDY_TUTOR_MAX_USER_MESSAGE_CHARS", 8_000)

# Separate, larger cap for the study-plan user prompt because it
# intentionally embeds several JSON blobs (mastery, due topics, weak
# topics). Without it, a profile with many courses x topics can blow
# the context window; with it, the prompt is bounded.
MAX_STUDY_PLAN_PROMPT_CHARS: int = _env_int("STUDY_TUTOR_MAX_STUDY_PLAN_PROMPT_CHARS", 16_000)


# ---------------------------------------------------------------------------
# Resource Agent caps  (consumed by agents/resource_agent.py)
# ---------------------------------------------------------------------------

# Cap weakness history so it doesn't grow unbounded over a long
# Streamlit session. The pattern detector only looks at recent reports.
MAX_WEAKNESS_HISTORY: int = _env_int("STUDY_TUTOR_MAX_WEAKNESS_HISTORY", 200)

# Cap the knowledge base so the serialised JSON profile doesn't grow
# to several megabytes and slow down the UI. When exceeded, the
# least-requested topics are evicted.
MAX_KNOWLEDGE_BASE_SIZE: int = _env_int("STUDY_TUTOR_MAX_KNOWLEDGE_BASE_SIZE", 50)


# ---------------------------------------------------------------------------
# MessageBus caps  (consumed by utils/agent_comm.py)
# ---------------------------------------------------------------------------

# Cap so a long Streamlit session doesn't accumulate MBs of message
# dicts in RAM. The Diagnostics tab only renders the last 15 anyway.
MESSAGE_LOG_CAP: int = _env_int("STUDY_TUTOR_MESSAGE_LOG_CAP", 500)

# Guardrail against runaway nested dispatch. send -> callback -> send
# is legitimate (request/response), but a callback cycle would blow the
# Python stack. 16 is comfortably above any legitimate depth.
MAX_DISPATCH_DEPTH: int = _env_int("STUDY_TUTOR_MAX_DISPATCH_DEPTH", 16)


# ---------------------------------------------------------------------------
# Persistence caps  (consumed by utils/student_profile.py and issue_detector.py)
# ---------------------------------------------------------------------------

# Filelock acquisition timeout. Long enough that two tabs racing on the
# same profile usually both succeed; short enough that a stuck lockfile
# surfaces an error within ~10 s rather than hanging the UI forever.
LOCK_TIMEOUT_SECONDS: int = _env_int("STUDY_TUTOR_LOCK_TIMEOUT_SECONDS", 10)

# Cap quiz history on disk so save_profile stays cheap even after
# thousands of answers. ``total_quizzes`` remains the lifetime counter.
MAX_QUIZ_HISTORY: int = _env_int("STUDY_TUTOR_MAX_QUIZ_HISTORY", 500)


# ---------------------------------------------------------------------------
# IssueDetector caps  (consumed by utils/issue_detector.py)
# ---------------------------------------------------------------------------

MAX_ISSUES: int = _env_int("STUDY_TUTOR_MAX_ISSUES", 200)
MAX_API_ERRORS: int = _env_int("STUDY_TUTOR_MAX_API_ERRORS", 200)

# Sliding window for stalled-learning and topic-imbalance detection.
STALLED_WINDOW_SIZE: int = _env_int("STUDY_TUTOR_STALLED_WINDOW_SIZE", 20)

# Alert cooldown — same issue type fires at most once per 10 questions
# OR once per 5 minutes, whichever is later.
COOLDOWN_QUESTION_WINDOW: int = _env_int("STUDY_TUTOR_COOLDOWN_QUESTION_WINDOW", 10)
COOLDOWN_TIME_SECONDS: int = _env_int("STUDY_TUTOR_COOLDOWN_TIME_SECONDS", 300)


# ---------------------------------------------------------------------------
# Quiz semantic verifier  (consumed by agents/tutor_agent.py)
# ---------------------------------------------------------------------------

# When enabled, every LLM-generated quiz triggers a second cheap LLM
# pass that fact-checks the answer key (is correct_answer actually
# correct? are distractors plausible but wrong? does explanation
# match?). Schema validation alone (`_validate_quiz_shape`) only
# catches structural bugs — a confident-but-wrong answer key would
# pass structure checks. Failed verification raises JSONDecodeError,
# which `retry_llm_call` re-rolls.
#
# Cost: doubles LLM calls per quiz generation. Defaults ON because the
# alternative is grading the student against a hallucinated answer
# key. Set `STUDY_TUTOR_QUIZ_VERIFIER_ENABLED=0` to disable.
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


QUIZ_VERIFIER_ENABLED: bool = _env_bool("STUDY_TUTOR_QUIZ_VERIFIER_ENABLED", True)

# Live demos should degrade gracefully if the API key, network, or SDK is
# unavailable. When enabled, the Streamlit app skips Anthropic client
# construction and uses local fallback quiz/feedback/plan paths.
OFFLINE_DEMO_MODE: bool = _env_bool("STUDY_TUTOR_OFFLINE_DEMO", False)


# ---------------------------------------------------------------------------
# Resource Agent blocking / latency  (consumed by agents/resource_agent.py)
# ---------------------------------------------------------------------------

# When True, every KB cache hit also fires an extra LLM call to
# generate a freshly-tailored "targeted explanation" for THIS exact
# wrong answer. That makes feedback richer but doubles per-turn LLM
# latency on cache hits.
#
# Default OFF — cache hits should be FAST (that's the whole point of a
# cache). When the cached materials aren't sufficient for the
# specific student/context, the bandit will eventually surface a
# weakness report, which DOES trigger fresh material generation.
RESOURCE_AGENT_EAGER_TARGETED_EXPLANATION: bool = _env_bool(
    "STUDY_TUTOR_RA_EAGER_TARGETED_EXPLANATION", False
)


# ---------------------------------------------------------------------------
# SM-2 spaced-repetition bounds  (consumed by utils/spaced_repetition.py)
# ---------------------------------------------------------------------------

# Cap interval so "next review in 10 years" never displays.
MAX_INTERVAL_DAYS: int = _env_int("STUDY_TUTOR_MAX_INTERVAL_DAYS", 365)

# Easiness factor ceiling (paired with the SM-2 standard 1.3 floor).
MAX_EASINESS_FACTOR: float = _env_float("STUDY_TUTOR_MAX_EASINESS_FACTOR", 3.0)

# Cap on get_due_topics so a student returning after a long absence
# gets a manageable review load (not 200 items at once).
DAILY_REVIEW_CAP: int = _env_int("STUDY_TUTOR_DAILY_REVIEW_CAP", 10)


def snapshot() -> dict[str, Any]:
    """Return a dict of all current config values for diagnostics.

    Useful when filing a bug report — the user can paste the full
    snapshot and the maintainer knows exactly which knobs were tuned.
    """
    return {
        k: v for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }


__all__ = [
    "CHARS_PER_TOKEN",
    "MAX_HISTORY_TOKENS",
    "MAX_MATERIALS_CHARS",
    "MAX_STRATEGY_CHARS",
    "MAX_SYSTEM_PROMPT_CHARS",
    "MAX_USER_MESSAGE_CHARS",
    "MAX_STUDY_PLAN_PROMPT_CHARS",
    "MAX_WEAKNESS_HISTORY",
    "MAX_KNOWLEDGE_BASE_SIZE",
    "MESSAGE_LOG_CAP",
    "MAX_DISPATCH_DEPTH",
    "LOCK_TIMEOUT_SECONDS",
    "MAX_QUIZ_HISTORY",
    "MAX_ISSUES",
    "MAX_API_ERRORS",
    "STALLED_WINDOW_SIZE",
    "COOLDOWN_QUESTION_WINDOW",
    "COOLDOWN_TIME_SECONDS",
    "MAX_INTERVAL_DAYS",
    "MAX_EASINESS_FACTOR",
    "DAILY_REVIEW_CAP",
    "QUIZ_VERIFIER_ENABLED",
    "OFFLINE_DEMO_MODE",
    "RESOURCE_AGENT_EAGER_TARGETED_EXPLANATION",
    "snapshot",
]
