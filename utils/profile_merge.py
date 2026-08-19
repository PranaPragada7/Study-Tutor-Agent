"""
Profile-merge primitives for multi-tab concurrent writers.

Why this module exists
----------------------
``utils/student_profile.save_profile`` runs an in-process re-read +
union with the on-disk version of the profile so two Streamlit tabs
answering quiz questions at the same time don't clobber each other's
appended history. The actual merge is the single most subtle
correctness-critical piece of the codebase:

  - dedupe ``quiz_history`` rows by stable UUID id (with a fallback to
    ``(timestamp, question)`` for legacy entries written before the id
    was added),
  - replay each NEW-to-us disk entry's contribution into the
    per-course / per-topic aggregate counters, so totals don't drift
    from ``quiz_history``,
  - dedupe ``quiz_sessions`` by stable report id so completed
    five-question reports survive concurrent tabs,
  - union-merge the ResourceAgent's ``knowledge_base`` (highest
    ``times_requested`` wins on collision) and ``weakness_history``
    (concat + dedupe).

Living inside ``student_profile.py`` next to ``create_profile`` and
``load_profile``, the merge logic was hard to reason about in
isolation, and impossible to unit-test without going through the whole
filelock + tempfile + os.replace dance. Pulling it out into this
module gives:

  - a clear contract: "given two profile dicts, return the merged
    profile as if both were valid concurrent writes",
  - room for direct unit tests (``tests/test_profile_merge.py``)
    that exercise edge cases (id collisions, legacy timestamps,
    aggregate drift) without touching disk,
  - a single import surface for any future caller that needs to
    reconcile two profile copies (e.g., a future cloud-sync feature).

``save_profile`` now imports from here and is a thin atomic-write
wrapper around the merge, which is also easier to read.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# quiz_history dedupe
# ---------------------------------------------------------------------------


def entry_key(entry: dict) -> str:
    """
    Stable dedupe key for a quiz_history entry.

    Newer entries carry a UUID ``id`` which is guaranteed unique even if
    two tabs answer at the same microsecond. Legacy entries (pre-fix)
    fall back to ``(timestamp, question)`` which the original code used.
    This makes the merge correct for both old and new rows.
    """
    eid = entry.get("id")
    if eid:
        return f"id:{eid}"
    return f"tq:{entry.get('timestamp')}|{entry.get('question')}"


# ---------------------------------------------------------------------------
# Aggregate replay  (course/topic counters)
# ---------------------------------------------------------------------------


def ensure_course(profile: dict, course: str) -> dict:
    """Initialise a course slot on ``profile`` if missing; return it.

    Used both by ``record_quiz_result`` (when a caller passes an unknown
    course name on the local-write path) and by ``apply_entry_to_aggregates``
    when replaying a disk entry whose course we haven't seen.
    """
    courses = profile.setdefault("courses", {})
    if course not in courses:
        courses[course] = {
            "exam_date": "",
            "self_rated_difficulty": 3,
            "topics": {},
            "total_correct": 0,
            "total_attempted": 0,
        }
    return courses[course]


def apply_entry_to_aggregates(profile: dict, entry: dict) -> None:
    """
    Apply a single ``quiz_history`` entry's contribution to the per-course
    and per-topic counters. Used during the merge for entries we picked up
    from disk that were not in our in-memory ``quiz_history`` — keeping
    aggregate stats in lock-step with the merged log.

    Idempotency note
    ----------------
    The caller MUST pre-filter entries through a ``seen`` set keyed by
    ``entry_key``; this function unconditionally adds 1 to
    ``total_attempted`` for the course/topic. Calling it twice for the
    same entry would double-count.

    Schema-tolerant of malformed entries:
      - missing ``course`` is dropped (we can't attribute the count)
      - missing ``topic`` updates only the course-level totals
      - ``correct`` defaults to ``False`` if absent
      - ``current_difficulty`` is set via ``setdefault`` so we never
        clobber the in-memory value with an older disk entry
    """
    course_name = entry.get("course")
    topic = entry.get("topic")
    correct = bool(entry.get("correct", False))
    if not course_name:
        return
    c = ensure_course(profile, course_name)
    c["total_attempted"] = int(c.get("total_attempted", 0)) + 1
    if correct:
        c["total_correct"] = int(c.get("total_correct", 0)) + 1
    if topic:
        topics = c.setdefault("topics", {})
        t = topics.setdefault(
            topic,
            {"correct": 0, "attempted": 0, "current_difficulty": entry.get("difficulty", 2)},
        )
        t["attempted"] = int(t.get("attempted", 0)) + 1
        if correct:
            t["correct"] = int(t.get("correct", 0)) + 1
        # Don't clobber current_difficulty: the in-memory value already
        # reflects the most-recent local quiz; disk entries can be from
        # before or after it. Only fill if missing.
        t.setdefault("current_difficulty", entry.get("difficulty", 2))


# ---------------------------------------------------------------------------
# ResourceAgent state merge
# ---------------------------------------------------------------------------


def merge_resource_agent_state(in_memory: dict, on_disk: dict) -> dict:
    """
    Union ResourceAgent state across concurrent writers so one tab
    doesn't clobber the other's accumulated learning.

    - ``knowledge_base``: key-level union; on collision keep the entry
      with the higher ``times_requested`` (approx. "most valuable" seen).
      Ties favour the in-memory (fresher materials) side.
    - ``weakness_history``: concatenate and dedupe on
      ``(topic, course, accuracy, attempts)``; the list is already
      tail-capped by ``ResourceAgent``.

    Either side may be ``None``/``{}``; the function gracefully degrades
    to whichever has data.
    """
    if not in_memory and not on_disk:
        return {}
    if not on_disk:
        return in_memory
    if not in_memory:
        return on_disk

    merged_kb: dict[str, dict] = {}
    for source in (on_disk.get("knowledge_base") or {}, in_memory.get("knowledge_base") or {}):
        for topic, data in source.items():
            existing = merged_kb.get(topic)
            if existing is None:
                merged_kb[topic] = dict(data)
            else:
                # Keep whichever side reports more requests; prefer the
                # newer (in_memory) materials when counts tie.
                if data.get("times_requested", 0) >= existing.get("times_requested", 0):
                    merged_kb[topic] = dict(data)

    seen: set[tuple] = set()
    merged_weak: list[dict] = []
    for source in (on_disk.get("weakness_history") or [], in_memory.get("weakness_history") or []):
        for entry in source:
            key = (
                entry.get("topic"),
                entry.get("course"),
                round(float(entry.get("accuracy", 0.0)), 4),
                int(entry.get("attempts", 0)),
            )
            if key in seen:
                continue
            seen.add(key)
            merged_weak.append(entry)

    return {
        "knowledge_base": merged_kb,
        "weakness_history": merged_weak,
    }


def merge_quiz_sessions(in_memory: dict, on_disk: dict, *, limit: int = 20) -> None:
    """Union completed quiz-session reports by stable report id."""
    disk_sessions = on_disk.get("quiz_sessions") or []
    if not disk_sessions:
        return

    memory_sessions = in_memory.setdefault("quiz_sessions", [])
    seen = {
        str(session.get("id"))
        for session in memory_sessions
        if isinstance(session, dict) and session.get("id")
    }
    for session in disk_sessions:
        if not isinstance(session, dict):
            continue
        sid = session.get("id")
        if not sid:
            continue
        key = str(sid)
        if key in seen:
            continue
        seen.add(key)
        memory_sessions.append(session)

    memory_sessions.sort(key=lambda s: str(s.get("created_at", "")))
    if len(memory_sessions) > limit:
        del memory_sessions[:-limit]
    in_memory["session_count"] = len(memory_sessions)


def merge_chat_history(in_memory: dict, on_disk: dict, *, limit: int = 100) -> None:
    """Union durable chatbot exchanges while respecting explicit clears."""
    reset_at = max(
        str(in_memory.get("chat_history_reset_at", "")),
        str(on_disk.get("chat_history_reset_at", "")),
    )
    memory_history = [
        item
        for item in in_memory.get("chat_history", []) or []
        if isinstance(item, dict) and str(item.get("timestamp", "")) > reset_at
    ]
    seen = {
        str(item.get("id") or f"{item.get('timestamp')}|{item.get('user_message')}")
        for item in memory_history
    }
    for item in on_disk.get("chat_history", []) or []:
        if not isinstance(item, dict) or str(item.get("timestamp", "")) <= reset_at:
            continue
        key = str(item.get("id") or f"{item.get('timestamp')}|{item.get('user_message')}")
        if key in seen:
            continue
        seen.add(key)
        memory_history.append(item)

    memory_history.sort(key=lambda item: str(item.get("timestamp", "")))
    in_memory["chat_history"] = memory_history[-limit:]
    in_memory["chat_history_reset_at"] = reset_at


# ---------------------------------------------------------------------------
# Top-level merge
# ---------------------------------------------------------------------------


def merge_profiles(in_memory: dict, on_disk: dict | None) -> dict:
    """
    Merge an in-memory profile with the on-disk version, mutating
    ``in_memory`` in place and returning it.

    The mutation-in-place form is intentional: ``save_profile`` uses
    the in-memory profile as both input AND the dict that gets written
    back to disk, so callers expect the changes to be reflected in the
    object they handed in.

    Merge contract:
      - ``quiz_history``: appended-to with any disk entries we hadn't
        seen, deduped by ``entry_key`` (UUID first, then
        ``(timestamp, question)`` for legacy rows).
      - per-course / per-topic aggregates: each newly-merged disk entry's
        delta replayed via ``apply_entry_to_aggregates``.
      - ``total_quizzes``: recomputed as ``max(sum(course totals),
        on_disk['total_quizzes'])`` so it can never under-count.
      - ``resource_agent_state``: union-merged via
        ``merge_resource_agent_state`` (on-disk side fed in if present).
      - ``quiz_sessions``: union-merged by report id so completed
        five-question evaluations are not lost across tabs.
      - ``chat_history``: union-merged by exchange id, with explicit history
        clears taking precedence over older entries.
      - ``quiz_history`` is then sorted by timestamp (string sort —
        chronological for a homogeneous tz-aware fleet, with the known
        caveat that legacy naive timestamps sort before tz-aware ones
        for the same instant; see student_profile.py M-4).

    Last-write-wins for fields NOT in the merge contract above
    (``spaced_repetition``, ``streak_tracker``, ``learning_style``,
    course config). Those mutate rarely enough that a true 2-tab race
    on them is not a realistic concern — and a real DB is the proper
    fix if it ever becomes one.
    """
    if not on_disk:
        return in_memory

    if "quiz_history" in on_disk:
        memory_history = in_memory.setdefault("quiz_history", [])
        seen = {entry_key(e) for e in memory_history}
        for entry in on_disk["quiz_history"]:
            key = entry_key(entry)
            if key in seen:
                continue
            seen.add(key)
            memory_history.append(entry)
            apply_entry_to_aggregates(in_memory, entry)

        in_memory["quiz_history"].sort(key=lambda e: e.get("timestamp", ""))

        in_memory["total_quizzes"] = max(
            sum(int(c.get("total_attempted", 0)) for c in in_memory.get("courses", {}).values()),
            int(on_disk.get("total_quizzes", 0)),
        )

    merge_quiz_sessions(in_memory, on_disk)
    merge_chat_history(in_memory, on_disk)

    disk_resource_state = on_disk.get("resource_agent_state")
    if disk_resource_state:
        in_memory["resource_agent_state"] = merge_resource_agent_state(
            in_memory.get("resource_agent_state") or {},
            disk_resource_state or {},
        )

    return in_memory


__all__ = [
    "entry_key",
    "ensure_course",
    "apply_entry_to_aggregates",
    "merge_quiz_sessions",
    "merge_chat_history",
    "merge_resource_agent_state",
    "merge_profiles",
]
