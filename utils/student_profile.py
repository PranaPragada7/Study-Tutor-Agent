"""
Student Profile Manager
Handles saving and loading student data (courses, scores, preferences).
Data is stored as JSON files in the /data directory.
"""

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Callable

from filelock import FileLock, Timeout

from config import LOCK_TIMEOUT_SECONDS as _LOCK_TIMEOUT_SECONDS
from config import MAX_CHAT_HISTORY as _MAX_CHAT_HISTORY
from config import MAX_QUIZ_HISTORY as _MAX_QUIZ_HISTORY
from utils.profile_merge import (
    apply_entry_to_aggregates as _apply_entry_to_aggregates,
    ensure_course as _ensure_course,
    entry_key as _entry_key,
    merge_profiles as _merge_profiles,
    merge_resource_agent_state as _merge_resource_agent_state,
)
from utils.telemetry import (
    COUNTERS,
    PROFILE_SAVE_OK,
    PROFILE_SAVE_SERIALIZE_FAIL,
    PROFILE_SAVE_TIMEOUT,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Module-level alias kept public because tests (and possibly external
# callers) reference ``utils.student_profile.MAX_QUIZ_HISTORY``.
MAX_QUIZ_HISTORY = _MAX_QUIZ_HISTORY
MAX_CHAT_HISTORY = _MAX_CHAT_HISTORY

# ---------------------------------------------------------------------------
# Profile schema versioning
# ---------------------------------------------------------------------------
# CURRENT_SCHEMA_VERSION is the version that ``create_profile`` writes
# today. Any older profile loaded from disk is migrated forward by
# ``migrate_profile`` before being returned to callers.
#
# Version 0 (legacy, pre-versioning): no ``schema_version`` field;
#   may be missing ``spaced_repetition``, ``streak_tracker``, or
#   ``resource_agent_state``; ``quiz_history`` entries may be missing
#   ``id`` (UUID) and have naive (non-tz-aware) ISO timestamps.
# Version 1: explicit ``schema_version``; defaulted top-level fields;
#   ``streak_tracker`` initialised to []; ``resource_agent_state``
#   initialised to {}; entries gain ``id`` lazily on the next
#   ``record_quiz_result``.
# Version 2: ``quiz_sessions`` stores completed five-question session
#   reports so evaluations remain visible after reload.
# Version 3: ``chat_history`` stores durable user/assistant exchanges so the
#   chatbot continues the conversation after a profile is reloaded.
CURRENT_SCHEMA_VERSION = 3


def migrate_profile(profile: dict) -> dict:
    """Upgrade a profile dict from any older schema version to the current one.

    Mutates and returns the input dict. Idempotent: calling on a
    profile already at CURRENT_SCHEMA_VERSION is a no-op.

    The migration sequence is intentionally additive — we only fill in
    missing fields and never delete or rename. That keeps the on-disk
    JSON forward-compatible: an older release reading a newer profile
    just sees fields it doesn't know about and ignores them.
    """
    if not isinstance(profile, dict):
        return profile  # nothing we can safely migrate

    version = int(profile.get("schema_version", 0) or 0)

    if version < 1:
        # v0 -> v1: fill in defaulted top-level fields.
        profile.setdefault("courses", {})
        profile.setdefault("quiz_history", [])
        profile.setdefault("total_quizzes", 0)
        profile.setdefault("session_count", 0)
        profile.setdefault("learning_style", "balanced")
        profile.setdefault("spaced_repetition", {})
        profile.setdefault("streak_tracker", [])
        profile.setdefault("resource_agent_state", {})
        profile["schema_version"] = 1

    if version < 2:
        profile.setdefault("quiz_sessions", [])
        profile["session_count"] = len(profile.get("quiz_sessions", []) or [])
        profile["schema_version"] = 2

    if version < 3:
        profile.setdefault("chat_history", [])
        profile.setdefault("chat_history_reset_at", "")
        profile["schema_version"] = 3

    profile.setdefault("quiz_sessions", [])
    profile.setdefault("session_count", len(profile.get("quiz_sessions", []) or []))
    profile.setdefault("chat_history", [])
    profile.setdefault("chat_history_reset_at", "")

    return profile


def _lock_path_for(profile_path: str) -> str:
    return profile_path + ".lock"


def _profile_path(student_name: str) -> str:
    """Get the file path for a student's profile."""
    # Sanitize: keep only alphanumeric, spaces, hyphens, underscores
    safe_name = re.sub(r'[^a-zA-Z0-9 _-]', '', student_name).strip()
    if not safe_name:
        raise ValueError("Student name must contain at least one alphanumeric character")
    safe_name = safe_name.lower().replace(" ", "_")
    path = os.path.join(DATA_DIR, f"{safe_name}.json")
    # CB-3: Ensure the resolved path stays inside DATA_DIR using commonpath
    # (startswith alone can false-positive on e.g. /data vs /data_backup)
    resolved = os.path.realpath(path)
    base = os.path.realpath(DATA_DIR)
    if os.path.commonpath([resolved, base]) != base:
        raise ValueError("Invalid student name")
    return path


# Note: _entry_key, _ensure_course, _apply_entry_to_aggregates, and
# _merge_resource_agent_state are imported from utils.profile_merge above
# (preserving the underscore-prefixed names for backward-compat with any
# test that grabbed them via the legacy import path). The merge logic
# lives in profile_merge.py with its own focused unit tests; this file
# is just the I/O wrapper around it.


def create_profile(student_name: str, courses: list[dict],
                   learning_style: str = "balanced") -> dict:
    """
    Create a new student profile.

    Args:
        student_name: The student's name
        courses: List of dicts like [{"name": "CS101", "exam_date": "2025-05-01", "difficulty": 3}]
                 difficulty is self-reported 1-5
        learning_style: One of "concise", "detailed", "visual", "balanced"

    Returns:
        The created profile dict
    """
    profile = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "name": student_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "learning_style": learning_style,
        "courses": {},
        "quiz_history": [],
        "quiz_sessions": [],
        "chat_history": [],
        "chat_history_reset_at": "",
        "total_quizzes": 0,
        "session_count": 0,
        "spaced_repetition": {},   # SM-2 scheduler state
        "streak_tracker": [],       # IssueDetector streak window (LA-7)
        "resource_agent_state": {}, # ResourceAgent KB + weakness history
    }

    for course in courses:
        course_name = course["name"]
        profile["courses"][course_name] = {
            "exam_date": course.get("exam_date", ""),
            "self_rated_difficulty": course.get("difficulty", 3),
            "topics": {},  # Will be populated as quizzes happen
            "total_correct": 0,
            "total_attempted": 0,
        }

    save_profile(student_name, profile)
    return profile


def save_profile(student_name: str, profile: dict) -> bool:
    """
    Atomically persist a student profile.

    Uses an advisory cross-process file lock (filelock) plus a tempfile +
    os.replace to provide:
      - no two cooperating writers truncate the same file concurrently
      - a crash mid-write leaves the previous good file intact
      - read-merge of ``quiz_history`` *and* the derived per-course /
        per-topic aggregate counters so two tabs answering questions at
        the same time can't silently lose each other's entries or let
        the aggregate stats drift from the actual quiz log
      - union-merge of ``resource_agent_state`` so one tab's learning
        doesn't clobber another's

    Caveats (cooperative, not absolute):
      - Fields not explicitly merged here (``spaced_repetition``,
        ``streak_tracker``) are last-write-wins. A real DB is the proper
        fix if that becomes a problem.
      - The lock is advisory: any writer that bypasses ``save_profile``
        is not bound by it.

    Returns:
        ``True`` when the write completed. ``False`` when the lock could
        not be acquired within ``_LOCK_TIMEOUT_SECONDS`` — the caller can
        choose to surface this (e.g., warn the user their last answer
        may not have persisted) or retry.
    """
    # Create the data directory BEFORE deriving lock/profile paths so the
    # lock file never lands in a directory that is racing with mkdir.
    os.makedirs(DATA_DIR, exist_ok=True)
    final_path = _profile_path(student_name)
    lock = FileLock(_lock_path_for(final_path), timeout=_LOCK_TIMEOUT_SECONDS)

    try:
        with lock:
            # Re-read the on-disk version and merge so we never clobber
            # another session's appended entries.
            on_disk = None
            if os.path.exists(final_path):
                try:
                    with open(final_path, "r") as f:
                        on_disk = json.load(f)
                except (json.JSONDecodeError, OSError):
                    on_disk = None

            # Delegate the actual merge to utils.profile_merge — same
            # semantics as the previous inline block, plus directly
            # unit-tested in tests/test_profile_merge.py.
            _merge_profiles(profile, on_disk)

            # Stamp the current schema version so an older release that
            # reads this profile back can detect that it should migrate.
            profile["schema_version"] = CURRENT_SCHEMA_VERSION

            # Trim the tail we actually write — keep total_quizzes as the
            # true lifetime counter, not len(quiz_history).
            if len(profile.get("quiz_history", [])) > MAX_QUIZ_HISTORY:
                profile["quiz_history"] = profile["quiz_history"][-MAX_QUIZ_HISTORY:]
            if len(profile.get("chat_history", [])) > MAX_CHAT_HISTORY:
                profile["chat_history"] = profile["chat_history"][-MAX_CHAT_HISTORY:]

            # Serialize to string BEFORE opening the temp file so the JSON
            # encoding cost is not paid inside the fsync window. With 500
            # quiz entries this is a few KB of serialization that no longer
            # blocks a second tab waiting on the lock.
            # Guard against non-serializable data (e.g., a datetime object
            # accidentally stored in resource_agent_state). Without this,
            # a TypeError from json.dumps would propagate out of
            # save_profile unguarded and crash the caller mid-quiz.
            try:
                payload = json.dumps(profile, indent=2)
            except (TypeError, ValueError) as e:
                COUNTERS.incr(PROFILE_SAVE_SERIALIZE_FAIL)
                logger.error(
                    "Failed to serialize profile for %s: %s — dropping save",
                    student_name, e,
                )
                return False

            # Atomic write: tempfile in the same dir, fsync, os.replace.
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(final_path)}.",
                suffix=".tmp",
                dir=os.path.dirname(final_path),
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, final_path)
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                raise
    except Timeout:
        # Don't crash the UI, but the caller needs to KNOW the write was
        # dropped so they can warn the user instead of silently believing
        # the quiz result was saved.
        COUNTERS.incr(PROFILE_SAVE_TIMEOUT)
        logger.error(
            "Could not acquire profile lock for %s within %ss — skipping write",
            student_name, _LOCK_TIMEOUT_SECONDS,
        )
        return False
    COUNTERS.incr(PROFILE_SAVE_OK)
    return True


def load_profile(student_name: str) -> dict | None:
    """Load a student profile from disk, guarded by the same lock.

    On lock timeout we log and return ``None``. We deliberately do NOT
    silently fall through to an unlocked read — the lock is advisory,
    but a stuck lockfile usually means another writer is actually in the
    middle of writing, and an unlocked read during that window can still
    observe a partial ``os.replace`` on some filesystems. Surfacing the
    timeout lets the UI prompt the user to retry instead of loading an
    undetectably-stale snapshot.
    """
    path = _profile_path(student_name)
    if not os.path.exists(path):
        return None
    try:
        with FileLock(_lock_path_for(path), timeout=_LOCK_TIMEOUT_SECONDS):
            with open(path, "r") as f:
                profile = json.load(f)
        # Migrate forward from any earlier schema version. Done outside
        # the lock so the migration's mutation can't deadlock if it
        # ever calls back into save_profile.
        return migrate_profile(profile)
    except Timeout:
        logger.error(
            "Could not acquire profile lock for %s within %ss on load",
            student_name, _LOCK_TIMEOUT_SECONDS,
        )
        return None


def list_saved_profiles() -> list[dict]:
    """Return saved student profiles for the sidebar profile picker.

    The app stores one JSON file per profile. This helper reads just enough
    metadata to make those saved profiles visible in the UI without requiring
    the user to remember the exact name they typed when creating the profile.
    Corrupt or partially-written files are skipped.
    """
    if not os.path.isdir(DATA_DIR):
        return []

    profiles: list[dict] = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue
        try:
            with FileLock(_lock_path_for(path), timeout=_LOCK_TIMEOUT_SECONDS):
                with open(path, "r") as f:
                    profile = migrate_profile(json.load(f))
        except (Timeout, OSError, json.JSONDecodeError, ValueError, TypeError):
            continue

        name = str(profile.get("name", "")).strip()
        if not name:
            continue
        profiles.append({
            "name": name,
            "total_quizzes": int(profile.get("total_quizzes", 0) or 0),
            "courses": len(profile.get("courses", {}) or {}),
            "updated_at": os.path.getmtime(path),
        })

    return sorted(
        profiles,
        key=lambda p: (str(p.get("name", "")).lower(), -int(p.get("updated_at", 0))),
    )


def record_quiz_result(profile: dict, course: str, topic: str,
                       difficulty: int, correct: bool, question: str, answer: str,
                       confidence: int = 3,
                       *,
                       correct_answer: str | None = None,
                       explanation: str | None = None,
                       now_fn: Callable[[], datetime] | None = None) -> dict:
    """
    Record the result of a single quiz question.

    Args:
        profile: The student profile dict
        course: Course name
        topic: Topic within the course
        difficulty: Difficulty level 1-5
        correct: Whether the student answered correctly
        question: The question text
        answer: The student's answer
        confidence: Student's self-rated confidence 1-5
        correct_answer: The answer key shown by the tutor
        explanation: The explanation attached to the generated question
        now_fn: Optional clock callable returning a tz-aware datetime, used
                for stamping the quiz_history entry's ``timestamp``. Defaults
                to ``datetime.now(timezone.utc)``. Tests and the sample
                profile generator inject a fake clock so generated histories can
                trace a deterministic timeline rather than collapsing onto
                the wall clock at script-run time.

    Returns:
        Updated profile
    """
    # Clamp at the persistence boundary. The downstream RL bandit and
    # SM-2 scheduler also clamp internally for defence-in-depth, but
    # clamping here means the persisted JSON history can never carry
    # an out-of-range value that would break a future replay.
    try:
        difficulty = max(1, min(5, int(difficulty)))
    except (TypeError, ValueError):
        difficulty = 2
    try:
        confidence = max(1, min(5, int(confidence)))
    except (TypeError, ValueError):
        confidence = 3

    # CB-2: Ensure the course exists before recording. If the caller
    # passes an unknown course name we initialise it on-the-fly so that
    # quiz_history and total_quizzes never contain orphaned entries.
    course_data = _ensure_course(profile, course)

    # Update course-level stats
    course_data["total_attempted"] += 1
    if correct:
        course_data["total_correct"] += 1

    # Update topic-level stats
    topics = profile["courses"][course]["topics"]
    if topic not in topics:
        topics[topic] = {"correct": 0, "attempted": 0, "current_difficulty": difficulty}
    topics[topic]["attempted"] += 1
    if correct:
        topics[topic]["correct"] += 1
    topics[topic]["current_difficulty"] = difficulty

    # Add to quiz history. C4: UTC-aware timestamp + stable UUID id so
    # the save_profile merge can dedupe reliably even when two tabs
    # answer the same question at the same microsecond.
    now = (now_fn or (lambda: datetime.now(timezone.utc)))()
    history_entry = {
        "id": uuid.uuid4().hex,
        "timestamp": now.isoformat(),
        "course": course,
        "topic": topic,
        "difficulty": difficulty,
        "correct": correct,
        "question": question,
        "student_answer": answer,
        "confidence": confidence,
    }
    if correct_answer is not None:
        history_entry["correct_answer"] = str(correct_answer)[:40]
    if explanation is not None:
        history_entry["explanation"] = str(explanation)[:4000]
    profile["quiz_history"].append(history_entry)

    # CB-2: Sync total_quizzes deterministically from course-level stats
    # so it can never drift out of sync with the actual data.
    profile["total_quizzes"] = sum(
        c["total_attempted"] for c in profile["courses"].values()
    )

    return profile


def record_chat_exchange(
    profile: dict,
    user_message: str,
    assistant_response: str,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict:
    """Append one durable chatbot exchange to the student profile."""
    now = (now_fn or (lambda: datetime.now(timezone.utc)))()
    history = profile.setdefault("chat_history", [])
    history.append({
        "id": uuid.uuid4().hex,
        "timestamp": now.isoformat(),
        "user_message": str(user_message)[:8000],
        "assistant_response": str(assistant_response)[:12000],
    })
    if len(history) > MAX_CHAT_HISTORY:
        del history[:-MAX_CHAT_HISTORY]
    return profile


def clear_chat_history(profile: dict) -> dict:
    """Clear durable chatbot memory without allowing an older tab to restore it."""
    profile["chat_history"] = []
    profile["chat_history_reset_at"] = datetime.now(timezone.utc).isoformat()
    return profile


def delete_profile(student_name: str) -> bool:
    """Permanently remove a student's profile JSON + lockfile from disk.

    Returns True on success (or if no profile existed), False if the
    file existed but couldn't be removed (e.g. open by another process).
    The companion lockfile is also removed if present. The persisted
    issue log under ``data/logs/`` is NOT touched here — call
    ``IssueDetector.delete_log`` separately if the user wants it gone.

    Used by the Streamlit "Delete my profile" privacy control.
    """
    try:
        path = _profile_path(student_name)
    except ValueError:
        return False
    lock_path = _lock_path_for(path)
    ok = True
    for p in (path, lock_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError as e:
                logger.error("Could not remove %s: %s", p, e)
                ok = False
    return ok


def export_profile_json(profile: dict) -> str:
    """Return a pretty-printed JSON string of the profile, suitable for
    download via Streamlit's ``st.download_button``.

    Raises ``TypeError`` (via ``json.dumps``) if the profile contains
    non-serialisable state — but this would also cause ``save_profile``
    to fail, so any in-memory profile that has ever been saved must be
    serialisable here.
    """
    return json.dumps(profile, indent=2, sort_keys=True)


def reset_quiz_history(profile: dict) -> dict:
    """Wipe the student's quiz history + per-course/topic counters.

    Preserves: name, created_at, learning_style, course list (with
    exam_date and self_rated_difficulty), schema_version.
    Resets: quiz_history, quiz_sessions, total_quizzes, session_count,
    spaced_repetition, streak_tracker, resource_agent_state,
    per-course total_attempted/total_correct, per-topic stats.

    Mutates and returns ``profile``. Caller is responsible for calling
    ``save_profile`` afterwards.
    """
    profile["quiz_history"] = []
    profile["quiz_sessions"] = []
    profile["total_quizzes"] = 0
    profile["session_count"] = 0
    profile["spaced_repetition"] = {}
    profile["streak_tracker"] = []
    profile["resource_agent_state"] = {}
    for course in profile.get("courses", {}).values():
        course["total_attempted"] = 0
        course["total_correct"] = 0
        for topic in course.get("topics", {}).values():
            topic["correct"] = 0
            topic["attempted"] = 0
    return profile


def record_quiz_feedback(profile: dict, quiz_id: str, flags: list[str],
                         note: str | None = None) -> bool:
    """Attach student feedback flags to a previously-recorded quiz_history entry.

    ``flags`` is a list drawn from the controlled vocabulary
    {``unclear_question``, ``wrong_answer``, ``unhelpful_explanation``,
    ``too_easy``, ``too_hard``}; unknown flags are dropped silently
    (forward-compat with future UI additions).

    Returns True if the entry was found and updated, False otherwise.
    Mutates ``profile`` in place; the caller is responsible for
    calling ``save_profile`` afterwards.

    Why this lives here, not on TutorAgent: feedback is durable
    student-state metadata that survives Streamlit reruns regardless
    of the agent instance. Putting it on the agent would lose flags
    on session restart.
    """
    valid_flags = {
        "unclear_question",
        "wrong_answer",
        "unhelpful_explanation",
        "too_easy",
        "too_hard",
    }
    cleaned = [f for f in (flags or []) if isinstance(f, str) and f in valid_flags]

    history = profile.get("quiz_history", [])
    for entry in reversed(history):  # newest-first; feedback usually targets the most-recent quiz
        if entry.get("id") == quiz_id:
            existing = entry.get("feedback_flags", [])
            # Union the new flags with any existing ones — students can
            # add follow-up flags without losing earlier ones.
            entry["feedback_flags"] = sorted(set(existing) | set(cleaned))
            if note:
                entry["feedback_note"] = str(note)[:500]
            entry["feedback_at"] = datetime.now(timezone.utc).isoformat()
            return True
    return False


def get_weakest_topics(profile: dict, course: str, top_n: int = 3) -> list[dict]:
    """
    Get the topics where the student struggles most in a given course.

    Returns list of dicts: [{"topic": str, "accuracy": float, "attempted": int}]
    """
    if course not in profile["courses"]:
        return []

    topics = profile["courses"][course]["topics"]
    if not topics:
        return []

    topic_stats = []
    for topic_name, stats in topics.items():
        attempted = int(stats.get("attempted", 0) or 0)
        if attempted <= 0:
            continue
        correct = int(stats.get("correct", 0) or 0)
        accuracy = correct / attempted
        if accuracy >= 0.8:
            continue
        topic_stats.append({
            "topic": topic_name,
            "accuracy": accuracy,
            "attempted": attempted,
        })

    # Sort by accuracy (lowest first), then by fewer attempts
    topic_stats.sort(key=lambda x: (x["accuracy"], x["attempted"]))
    return topic_stats[:top_n]


def get_strongest_topics(profile: dict, course: str, top_n: int = 3) -> list[dict]:
    """Return practiced topics with at least 80% accuracy, strongest first."""
    if course not in profile.get("courses", {}):
        return []

    topic_stats = []
    for topic_name, stats in profile["courses"][course].get("topics", {}).items():
        attempted = int(stats.get("attempted", 0) or 0)
        if attempted <= 0:
            continue
        correct = int(stats.get("correct", 0) or 0)
        accuracy = correct / attempted
        if accuracy < 0.8:
            continue
        topic_stats.append({
            "topic": topic_name,
            "accuracy": accuracy,
            "attempted": attempted,
        })

    topic_stats.sort(
        key=lambda item: (
            -item["accuracy"],
            -item["attempted"],
            item["topic"],
        )
    )
    return topic_stats[:top_n]


def get_performance_summary(profile: dict) -> dict:
    """Get an overall performance summary for the student."""
    summary = {
        "name": profile["name"],
        "total_quizzes": profile["total_quizzes"],
        "courses": {},
    }

    for course_name, course_data in profile["courses"].items():
        attempted = course_data["total_attempted"]
        correct = course_data["total_correct"]
        accuracy = correct / attempted if attempted > 0 else 0.0
        summary["courses"][course_name] = {
            "accuracy": round(accuracy * 100, 1),
            "attempted": attempted,
            "correct": correct,
            "num_topics_covered": sum(
                1
                for stats in course_data["topics"].values()
                if int(stats.get("attempted", 0) or 0) > 0
            ),
            "weak_topics": get_weakest_topics(profile, course_name),
            "strong_topics": get_strongest_topics(profile, course_name),
        }

    return summary
