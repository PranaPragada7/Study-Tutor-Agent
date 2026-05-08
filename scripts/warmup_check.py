"""
Sample Profile warm-up.

Loads the ``Sample Student`` profile, instantiates a TutorAgent, and asks
it to generate one quiz question for ``Intro to CS``. The profile is
NOT saved back to disk — quiz_history is left untouched so the live
run starts from a clean slate.

The script reports whether a real Anthropic call succeeded or whether
the fallback path was taken. It NEVER prints the API key.

Usage
-----
    python scripts/warmup_check.py

Exit codes
----------
    0  question generated (real LLM call OR fallback) — app is reachable
    1  unrecoverable error (profile missing, agent failed to construct, etc.)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env so ANTHROPIC_API_KEY is in os.environ before anthropic SDK init.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    # python-dotenv not installed — preflight will catch this; warmup
    # can still proceed if the env var is set in the shell directly.
    pass


def _force_utf8_stdout() -> None:
    """Keep emoji status lines from crashing older Windows consoles."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def warmup() -> int:
    profile_path = REPO_ROOT / "data" / "sample_student.json"
    if not profile_path.exists():
        print("❌ data/sample_student.json not found.")
        print("   fix: python tests/generate_sample_profile.py")
        return 1

    # Imports deferred so a missing dep produces a readable error rather
    # than a top-of-file ImportError.
    try:
        from utils.student_profile import load_profile
        from agents.tutor_agent import TutorAgent
        from utils.telemetry import COUNTERS, LLM_FALLBACK_USED, QUIZ_FALLBACK_USED
    except ImportError as exc:
        print(f"❌ Could not import project modules: {exc}")
        print("   fix: pip install -r requirements.txt and run from the repo root")
        return 1

    profile = load_profile("Sample Student")
    if profile is None:
        print("❌ load_profile('Sample Student') returned None.")
        print("   fix: re-run python tests/generate_sample_profile.py")
        return 1

    # Work on a deep copy so the loaded dict cannot be mutated and
    # accidentally written back by some other code path.
    working = copy.deepcopy(profile)

    try:
        tutor = TutorAgent(working)
    except Exception as exc:  # noqa: BLE001 — we want to surface anything
        print(f"❌ TutorAgent failed to initialise: {exc}")
        return 1

    # Snapshot history length so we can verify we did not append to it.
    history_before = len(working.get("quiz_history", []))

    quiz_fallback_before = COUNTERS.get(QUIZ_FALLBACK_USED)
    llm_fallback_before = COUNTERS.get(LLM_FALLBACK_USED)

    print("Generating one warm-up question for 'Intro to CS' ...")
    try:
        quiz = tutor.generate_quiz_question("Intro to CS")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ generate_quiz_question raised: {exc}")
        return 1

    history_after = len(working.get("quiz_history", []))
    if history_after != history_before:
        print(
            "⚠️  Warm-up appended to quiz_history "
            f"({history_before} → {history_after}). The live app session will "
            "start with that extra entry. Consider re-running "
            "tests/generate_sample_profile.py before presenting."
        )

    fallback = (
        COUNTERS.get(QUIZ_FALLBACK_USED) > quiz_fallback_before
        or COUNTERS.get(LLM_FALLBACK_USED) > llm_fallback_before
    )
    topic = quiz.get("topic", "?")
    difficulty = quiz.get("difficulty", "?")

    if fallback:
        print(f"⚠️  Fallback path used. topic={topic} difficulty={difficulty}")
        print("   This means the LLM call did not succeed; the canned")
        print("   item bank produced the question. Check network and key.")
        # Still exit 0 — the app is reachable, just degraded.
        return 0

    print(f"✅ Real LLM question generated. topic={topic} difficulty={difficulty}")
    print("   Profile was NOT saved; quiz_history untouched.")
    return 0


if __name__ == "__main__":
    _force_utf8_stdout()
    sys.exit(warmup())
