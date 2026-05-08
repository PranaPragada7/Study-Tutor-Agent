"""
Tests for utils/profile_merge.py and the concurrent save_profile merge.

Covers the previously-untested correctness-critical merge logic:

  - entry_key dedupe (UUID + legacy fallback)
  - apply_entry_to_aggregates (per-course / per-topic counter replay)
  - merge_resource_agent_state (KB union, weakness dedup)
  - merge_profiles (top-level integration)
  - save_profile under genuine multi-thread contention
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import profile_merge
from utils.profile_merge import (
    apply_entry_to_aggregates,
    ensure_course,
    entry_key,
    merge_profiles,
    merge_resource_agent_state,
)
from utils.student_profile import (
    create_profile,
    load_profile,
    record_quiz_result,
    save_profile,
)


# ═══════════════════════════════════════════════
# entry_key: UUID + legacy fallback
# ═══════════════════════════════════════════════

class TestEntryKey:

    def test_uuid_id_takes_priority(self):
        e = {"id": "abc123", "timestamp": "t", "question": "q"}
        assert entry_key(e) == "id:abc123"

    def test_legacy_fallback_to_timestamp_question(self):
        e = {"timestamp": "2025-01-01T00:00:00", "question": "What is X?"}
        assert entry_key(e) == "tq:2025-01-01T00:00:00|What is X?"

    def test_no_id_no_timestamp_still_keys(self):
        # Worst case: malformed entry. We still want a stable string
        # (None|None) so dedupe at least collapses identical garbage.
        e = {}
        assert entry_key(e) == "tq:None|None"

    def test_id_collisions_unique(self):
        e1 = {"id": uuid.uuid4().hex}
        e2 = {"id": uuid.uuid4().hex}
        assert entry_key(e1) != entry_key(e2)


# ═══════════════════════════════════════════════
# apply_entry_to_aggregates
# ═══════════════════════════════════════════════

class TestApplyEntryToAggregates:

    def _empty_profile(self) -> dict:
        return {"courses": {}}

    def test_creates_missing_course(self):
        profile = self._empty_profile()
        apply_entry_to_aggregates(profile, {
            "course": "CS101", "topic": "Loops",
            "difficulty": 2, "correct": True,
        })
        assert "CS101" in profile["courses"]
        assert profile["courses"]["CS101"]["total_attempted"] == 1
        assert profile["courses"]["CS101"]["total_correct"] == 1

    def test_increments_topic_stats(self):
        profile = self._empty_profile()
        for correct in (True, True, False):
            apply_entry_to_aggregates(profile, {
                "course": "CS101", "topic": "Recursion",
                "difficulty": 3, "correct": correct,
            })
        topic = profile["courses"]["CS101"]["topics"]["Recursion"]
        assert topic["attempted"] == 3
        assert topic["correct"] == 2

    def test_missing_course_field_is_dropped(self):
        profile = self._empty_profile()
        apply_entry_to_aggregates(profile, {"correct": True})
        # No course → silently drop. Don't create a phantom course.
        assert profile["courses"] == {}

    def test_missing_topic_updates_only_course(self):
        profile = self._empty_profile()
        apply_entry_to_aggregates(profile, {
            "course": "CS101", "correct": True, "difficulty": 2,
        })
        assert profile["courses"]["CS101"]["total_attempted"] == 1
        assert profile["courses"]["CS101"]["topics"] == {}

    def test_does_not_clobber_current_difficulty(self):
        """Disk entries can be older than the in-memory state. We must
        not overwrite a freshly-set ``current_difficulty`` with a stale
        one from disk."""
        profile = {"courses": {
            "CS101": {
                "exam_date": "", "self_rated_difficulty": 3,
                "total_attempted": 0, "total_correct": 0,
                "topics": {"Loops": {"correct": 0, "attempted": 0,
                                     "current_difficulty": 5}},
            }
        }}
        # Replay an old-disk entry with difficulty=1
        apply_entry_to_aggregates(profile, {
            "course": "CS101", "topic": "Loops",
            "difficulty": 1, "correct": True,
        })
        # The freshly-set 5 must survive.
        assert profile["courses"]["CS101"]["topics"]["Loops"]["current_difficulty"] == 5


# ═══════════════════════════════════════════════
# ensure_course
# ═══════════════════════════════════════════════

class TestEnsureCourse:

    def test_creates_with_defaults(self):
        profile: dict = {}
        c = ensure_course(profile, "CS101")
        assert c["self_rated_difficulty"] == 3
        assert c["total_attempted"] == 0
        assert c["topics"] == {}

    def test_idempotent(self):
        profile = {"courses": {"CS101": {"total_attempted": 5,
                                          "total_correct": 3,
                                          "topics": {}, "exam_date": "",
                                          "self_rated_difficulty": 4}}}
        c = ensure_course(profile, "CS101")
        # Existing data must survive.
        assert c["total_attempted"] == 5
        assert c["self_rated_difficulty"] == 4


# ═══════════════════════════════════════════════
# merge_resource_agent_state
# ═══════════════════════════════════════════════

class TestMergeResourceAgentState:

    def test_both_empty_returns_empty_dict(self):
        assert merge_resource_agent_state({}, {}) == {}
        assert merge_resource_agent_state(None, None) == {}  # type: ignore[arg-type]

    def test_one_side_empty_returns_other(self):
        kb = {"knowledge_base": {"loops": {"times_requested": 3}}}
        assert merge_resource_agent_state(kb, {}) is kb
        assert merge_resource_agent_state({}, kb) is kb

    def test_kb_collision_keeps_higher_request_count(self):
        in_mem = {"knowledge_base": {"loops": {"times_requested": 5,
                                                "marker": "in_mem"}}}
        on_disk = {"knowledge_base": {"loops": {"times_requested": 2,
                                                 "marker": "on_disk"}}}
        merged = merge_resource_agent_state(in_mem, on_disk)
        assert merged["knowledge_base"]["loops"]["marker"] == "in_mem"

    def test_kb_collision_tie_favors_in_memory(self):
        # Tie on times_requested → in-memory (newer materials) wins.
        in_mem = {"knowledge_base": {"loops": {"times_requested": 3,
                                                "marker": "in_mem"}}}
        on_disk = {"knowledge_base": {"loops": {"times_requested": 3,
                                                 "marker": "on_disk"}}}
        merged = merge_resource_agent_state(in_mem, on_disk)
        assert merged["knowledge_base"]["loops"]["marker"] == "in_mem"

    def test_kb_unions_distinct_topics(self):
        in_mem = {"knowledge_base": {"loops": {"times_requested": 1}}}
        on_disk = {"knowledge_base": {"recursion": {"times_requested": 1}}}
        merged = merge_resource_agent_state(in_mem, on_disk)
        assert set(merged["knowledge_base"].keys()) == {"loops", "recursion"}

    def test_weakness_history_dedupe(self):
        entry = {"topic": "Recursion", "course": "CS101",
                 "accuracy": 0.4, "attempts": 3}
        in_mem = {"weakness_history": [entry]}
        on_disk = {"weakness_history": [dict(entry)]}  # exact duplicate
        merged = merge_resource_agent_state(in_mem, on_disk)
        assert len(merged["weakness_history"]) == 1

    def test_weakness_history_distinct_kept(self):
        e1 = {"topic": "Recursion", "course": "CS101",
              "accuracy": 0.4, "attempts": 3}
        e2 = {"topic": "Loops", "course": "CS101",
              "accuracy": 0.3, "attempts": 5}
        merged = merge_resource_agent_state(
            {"weakness_history": [e1]},
            {"weakness_history": [e2]},
        )
        assert len(merged["weakness_history"]) == 2


# ═══════════════════════════════════════════════
# merge_profiles
# ═══════════════════════════════════════════════

class TestMergeProfiles:

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def test_no_disk_returns_in_memory_unchanged(self):
        profile = create_profile("Solo", [{"name": "CS101", "difficulty": 3}])
        result = merge_profiles(profile, None)
        assert result is profile
        assert result["total_quizzes"] == 0

    def test_merges_disjoint_quiz_histories(self):
        """The canonical concurrent-write scenario: tab A wrote entry X,
        tab B wrote entry Y, both starting from the same base. After the
        merge the in-memory state must contain BOTH entries with
        consistent aggregate counters."""
        profile_a = create_profile("Disjoint", [
            {"name": "CS101", "difficulty": 3},
        ])
        profile_a = record_quiz_result(profile_a, "CS101", "Loops", 2,
                                       True, "Q-A", "A")
        # Disk version simulates Tab B's state: same base, different entry.
        profile_b = create_profile("Disjoint", [
            {"name": "CS101", "difficulty": 3},
        ])
        profile_b = record_quiz_result(profile_b, "CS101", "Recursion", 3,
                                       False, "Q-B", "C")

        # profile_a is "in memory" for the about-to-save tab; profile_b
        # is what was on disk written by the other tab.
        merge_profiles(profile_a, profile_b)

        # Both entries present
        assert len(profile_a["quiz_history"]) == 2
        topics_seen = {e["topic"] for e in profile_a["quiz_history"]}
        assert topics_seen == {"Loops", "Recursion"}

        # Aggregates correct: 2 attempts (1 correct, 1 wrong)
        cs101 = profile_a["courses"]["CS101"]
        assert cs101["total_attempted"] == 2
        assert cs101["total_correct"] == 1
        assert profile_a["total_quizzes"] == 2

    def test_dedupe_does_not_double_count_aggregates(self):
        """A disk entry that already exists in memory (same UUID) must
        NOT be re-counted on aggregate replay."""
        profile = create_profile("Dedupe", [{"name": "CS101", "difficulty": 3}])
        profile = record_quiz_result(profile, "CS101", "Loops", 2,
                                     True, "Q1", "A")
        # Simulate disk state: an exact copy (same UUID id).
        on_disk = {
            "quiz_history": list(profile["quiz_history"]),
        }
        before_attempted = profile["courses"]["CS101"]["total_attempted"]
        merge_profiles(profile, on_disk)
        # Same entry → no re-count.
        assert profile["courses"]["CS101"]["total_attempted"] == before_attempted
        assert profile["total_quizzes"] == 1

    def test_legacy_entries_without_uuid_still_dedupe(self):
        """Entries written before the UUID id was added fall back to
        ``(timestamp, question)`` keying."""
        legacy_entry = {
            "timestamp": "2025-01-01T00:00:00",
            "course": "CS101", "topic": "Loops",
            "difficulty": 2, "correct": True,
            "question": "Legacy Q", "student_answer": "A",
            "confidence": 3,
        }
        # Both sides have the same legacy entry.
        in_memory = {
            "name": "Legacy", "courses": {"CS101": {
                "exam_date": "", "self_rated_difficulty": 3,
                "topics": {}, "total_correct": 0, "total_attempted": 0,
            }},
            "quiz_history": [dict(legacy_entry)], "total_quizzes": 0,
        }
        on_disk = {
            "name": "Legacy", "courses": {},
            "quiz_history": [dict(legacy_entry)], "total_quizzes": 1,
        }
        merge_profiles(in_memory, on_disk)
        # Dedup'd: only one history entry, no aggregate double-count.
        assert len(in_memory["quiz_history"]) == 1

    def test_merges_quiz_session_reports(self):
        in_memory = {
            "name": "Sessions",
            "courses": {},
            "quiz_history": [],
            "quiz_sessions": [
                {"id": "s1", "created_at": "2026-05-07T12:00:00+00:00"},
            ],
            "total_quizzes": 0,
        }
        on_disk = {
            "name": "Sessions",
            "courses": {},
            "quiz_history": [],
            "quiz_sessions": [
                {"id": "s1", "created_at": "2026-05-07T12:00:00+00:00"},
                {"id": "s2", "created_at": "2026-05-07T12:05:00+00:00"},
            ],
            "total_quizzes": 0,
        }

        merge_profiles(in_memory, on_disk)

        assert [s["id"] for s in in_memory["quiz_sessions"]] == ["s1", "s2"]
        assert in_memory["session_count"] == 2


# ═══════════════════════════════════════════════
# Concurrency: save_profile under thread contention
# ═══════════════════════════════════════════════

class TestSaveProfileConcurrency:

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def test_two_threads_disjoint_writes_union(self):
        """
        Two threads each take a fresh copy of the same base profile,
        append their own quiz_history entry, and call save_profile
        concurrently behind a barrier. The final on-disk state must
        contain BOTH entries with totals = 2.

        This is the exact scenario the merge code was designed for and
        was previously untested. A regression in the dedup or aggregate
        replay would surface here as either lost data or doubled totals.
        """
        # Shared base profile on disk (both threads load this).
        base = create_profile("Concurrent", [
            {"name": "CS101", "difficulty": 3},
        ])
        # base is already saved by create_profile.

        n_per_thread = 5
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker(label: str):
            try:
                # Each thread starts from its own load (simulating two
                # browser tabs with their own session_state copies).
                local = load_profile("Concurrent")
                assert local is not None
                for i in range(n_per_thread):
                    local = record_quiz_result(
                        local, "CS101", f"T{label}", 2,
                        i % 2 == 0,
                        f"Q-{label}-{i}", "A",
                    )
                # Both threads block here, then race on save_profile.
                barrier.wait(timeout=10)
                # Each thread's save_profile re-reads disk under the
                # filelock, merges, writes. The second writer should
                # pick up the first writer's entries via the merge.
                save_profile("Concurrent", local)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start(); t2.start()
        t1.join(timeout=15); t2.join(timeout=15)

        assert not errors, f"Worker raised: {errors}"

        # The final on-disk profile must reflect BOTH threads' work.
        final = load_profile("Concurrent")
        assert final is not None
        # 2 threads × n_per_thread entries each, no dedupe loss.
        assert len(final["quiz_history"]) == 2 * n_per_thread, \
            f"expected {2*n_per_thread}, got {len(final['quiz_history'])}"
        # Aggregate totals match
        assert final["courses"]["CS101"]["total_attempted"] == 2 * n_per_thread
        # total_quizzes matches sum of course totals
        assert final["total_quizzes"] == 2 * n_per_thread
        # Both topic labels are present
        topics = set(final["courses"]["CS101"]["topics"].keys())
        assert topics == {"TA", "TB"}

    def test_idempotent_save_does_not_double(self):
        """Calling save_profile twice in a row with the same in-memory
        state must not double-count anything."""
        profile = create_profile("Idem", [{"name": "CS101", "difficulty": 3}])
        profile = record_quiz_result(profile, "CS101", "Loops", 2,
                                     True, "Q1", "A")
        save_profile("Idem", profile)
        save_profile("Idem", profile)  # second save with same state
        loaded = load_profile("Idem")
        assert loaded is not None
        assert len(loaded["quiz_history"]) == 1
        assert loaded["total_quizzes"] == 1
        assert loaded["courses"]["CS101"]["total_attempted"] == 1
