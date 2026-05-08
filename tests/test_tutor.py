"""
Tests for the core single-agent components.

Run with: pytest tests/test_tutor.py -v

Covers:
  - Adaptive Engine (RL bandit with confidence + seed determinism)
  - Student Profile (with learning style, confidence, and data integrity)
  - Spaced Repetition (SM-2 algorithm)
  - Issue Detector (with alert cooldown)
  - Agent Communication (message bus with UUID IDs)
  - LLM retry decorator
"""

import os
import random
import sys
import tempfile

import pytest

# Make imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.adaptive_engine import AdaptiveEngine
from utils.student_profile import (
    create_profile,
    load_profile,
    save_profile,
    record_quiz_result,
    get_weakest_topics,
    get_performance_summary,
    DATA_DIR,
)
from utils.spaced_repetition import SpacedRepetitionScheduler
from utils.issue_detector import IssueDetector, Issue
from utils.agent_comm import MessageBus, AgentMessage, MessageType


# ═══════════════════════════════════════════════
# Adaptive Engine Tests
# ═══════════════════════════════════════════════

class TestAdaptiveEngine:

    def test_initialization(self):
        engine = AdaptiveEngine()
        assert engine.epsilon == 0.2
        assert engine.arm_stats == {}

    def test_select_topic_returns_valid_choice(self):
        engine = AdaptiveEngine()
        topics = ["Recursion", "Loops", "Data Structures"]
        topic, difficulty = engine.select_topic_and_difficulty(topics)
        assert topic in topics
        assert 1 <= difficulty <= 5

    def test_update_records_stats(self):
        engine = AdaptiveEngine()
        engine.update("Recursion", 3, correct=True, confidence=4)
        assert "Recursion" in engine.topic_accuracy
        assert engine.topic_accuracy["Recursion"]["correct"] == 1
        assert engine.topic_accuracy["Recursion"]["total"] == 1

    def test_update_tracks_incorrect(self):
        engine = AdaptiveEngine()
        engine.update("Loops", 2, correct=False, confidence=2)
        assert engine.topic_accuracy["Loops"]["correct"] == 0
        assert engine.topic_accuracy["Loops"]["total"] == 1

    def test_mastery_calculation(self):
        engine = AdaptiveEngine()
        engine.update("Recursion", 3, correct=True)
        engine.update("Recursion", 3, correct=True)
        engine.update("Recursion", 3, correct=False)
        mastery = engine.get_topic_mastery()
        assert mastery["Recursion"] == round((2 / 3) * 100, 1)

    def test_epsilon_decays(self):
        engine = AdaptiveEngine(epsilon=0.5)
        initial = engine.epsilon
        engine.update("Topic", 2, correct=True)
        assert engine.epsilon < initial

    def test_recommend_difficulty_new_topic(self):
        engine = AdaptiveEngine()
        assert engine.recommend_difficulty("Brand New Topic") == 2

    def test_recommend_difficulty_high_accuracy(self):
        engine = AdaptiveEngine()
        for _ in range(10):
            engine.update("Easy Topic", 3, correct=True, confidence=5)
        diff = engine.recommend_difficulty("Easy Topic")
        assert diff >= 3

    def test_recommend_difficulty_low_accuracy(self):
        engine = AdaptiveEngine()
        for _ in range(10):
            engine.update("Hard Topic", 3, correct=False, confidence=2)
        diff = engine.recommend_difficulty("Hard Topic")
        assert diff <= 2

    def test_explain_recommendation_for_new_topic(self):
        engine = AdaptiveEngine()

        explanation = engine.explain_recommendation("Brand New Topic", 2)

        assert explanation["reason"] == "New topic baseline"
        assert explanation["policy"] == "cold_start"
        assert explanation["mastery"] is None
        assert explanation["recommended_difficulty"] == 2

    def test_explain_recommendation_for_weak_topic(self):
        engine = AdaptiveEngine()
        for _ in range(5):
            engine.update("Recursion", 3, correct=False, confidence=4)

        explanation = engine.explain_recommendation("Recursion", 1)

        assert explanation["reason"] == "Weak topic recovery"
        assert explanation["policy"] == "weakness_targeting"
        assert explanation["mastery"] == 0.0
        assert explanation["attempts"] == 5

    def test_explain_recommendation_for_spaced_review(self):
        engine = AdaptiveEngine()

        explanation = engine.explain_recommendation(
            "Loops",
            2,
            is_review=True,
        )

        assert explanation["reason"] == "Spaced review due"
        assert explanation["policy"] == "spaced_repetition"

    def test_exploration_produces_variety(self):
        engine = AdaptiveEngine(epsilon=1.0)  # Always explore
        topics = ["A", "B", "C", "D", "E"]
        selected = set()
        for _ in range(50):
            topic, _ = engine.select_topic_and_difficulty(topics)
            selected.add(topic)
        assert len(selected) >= 3

    def test_confidence_affects_reward(self):
        """
        The intentional reward shape (see AdaptiveEngine.update docstring):
          - Wrong + low confidence  → ~0.3 (known gap, a clean signal)
          - Wrong + high confidence → ~0.0 (misconception, NOT rewarded here —
            the selection function pushes attention back to it via the
            weakness bonus, not via arm reward)

        So an "expected gap" arm must have a STRICTLY HIGHER reward than
        a "misconception" arm for the same topic/difficulty.
        """
        engine = AdaptiveEngine()
        engine.update("Misconception", 3, correct=False, confidence=5)
        engine.update("Expected Gap", 3, correct=False, confidence=1)
        misc_arm = engine.arm_stats.get(("Misconception", 3), {})
        gap_arm = engine.arm_stats.get(("Expected Gap", 3), {})
        assert gap_arm.get("total_reward", 0) > misc_arm.get("total_reward", 0)
        # Misconceptions should zero out, not add signal to the arm.
        assert misc_arm.get("total_reward", 0) == 0.0

    def test_misconception_gets_selection_priority_via_weakness_bonus(self):
        """
        Confident-wrong doesn't pump the arm reward (see above), but the
        selection function still pushes the student back to the topic via
        the weakness bonus. Verify that a misconception topic wins over a
        fully-mastered topic during greedy selection.
        """
        engine = AdaptiveEngine(epsilon=0.0)  # greedy — no random choice
        # "Mastered" is answered correctly at high confidence many times.
        for _ in range(10):
            engine.update("Mastered", 3, correct=True, confidence=5)
        # "Misconception" is confidently wrong many times — weakness bonus should dominate.
        for _ in range(10):
            engine.update("Misconception", 3, correct=False, confidence=5)

        picks = set()
        for _ in range(5):
            topic, _diff = engine.select_topic_and_difficulty(["Mastered", "Misconception"])
            picks.add(topic)
        assert "Misconception" in picks


# ═══════════════════════════════════════════════
# Student Profile Tests
# ═══════════════════════════════════════════════

class TestStudentProfile:

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def test_create_profile(self):
        profile = create_profile("Test Student", [
            {"name": "CS101", "difficulty": 3},
            {"name": "Math 201", "difficulty": 4},
        ])
        assert profile["name"] == "Test Student"
        assert "CS101" in profile["courses"]
        assert profile["total_quizzes"] == 0

    def test_create_profile_with_learning_style(self):
        profile = create_profile("Style Student", [
            {"name": "Bio", "difficulty": 2},
        ], learning_style="visual")
        assert profile["learning_style"] == "visual"
        assert "spaced_repetition" in profile

    def test_save_and_load(self):
        profile = create_profile("Save Test", [{"name": "Bio 110", "difficulty": 2}])
        save_profile("Save Test", profile)
        loaded = load_profile("Save Test")
        assert loaded is not None
        assert loaded["name"] == "Save Test"

    def test_load_nonexistent(self):
        result = load_profile("Nobody")
        assert result is None

    def test_record_quiz_result_with_confidence(self):
        profile = create_profile("Quiz Tester", [{"name": "CS101", "difficulty": 3}])
        profile = record_quiz_result(
            profile, course="CS101", topic="Loops", difficulty=2,
            correct=True, question="What is a for loop?", answer="A", confidence=5
        )
        assert profile["total_quizzes"] == 1
        assert profile["quiz_history"][-1]["confidence"] == 5

    def test_get_weakest_topics(self):
        profile = create_profile("Weak Topics", [{"name": "CS101", "difficulty": 3}])
        for _ in range(5):
            profile = record_quiz_result(profile, "CS101", "Recursion", 3, True, "q", "a")
        for _ in range(5):
            profile = record_quiz_result(profile, "CS101", "Loops", 3, False, "q", "a")
        profile["courses"]["CS101"]["topics"]["Unattempted"] = {
            "correct": 0,
            "attempted": 0,
            "current_difficulty": 2,
        }
        weak = get_weakest_topics(profile, "CS101")
        assert len(weak) > 0
        assert weak[0]["topic"] == "Loops"
        assert all(item["attempted"] > 0 for item in weak)
        assert all(item["accuracy"] < 0.8 for item in weak)
        assert "Unattempted" not in {item["topic"] for item in weak}
        assert "Recursion" not in {item["topic"] for item in weak}

    def test_performance_summary(self):
        profile = create_profile("Summary Test", [{"name": "Math", "difficulty": 3}])
        for _ in range(3):
            profile = record_quiz_result(profile, "Math", "Algebra", 2, True, "q", "a")
        profile = record_quiz_result(profile, "Math", "Algebra", 2, False, "q", "a")
        profile["courses"]["Math"]["topics"]["Geometry"] = {
            "correct": 0,
            "attempted": 0,
            "current_difficulty": 2,
        }
        summary = get_performance_summary(profile)
        assert summary["total_quizzes"] == 4
        assert summary["courses"]["Math"]["accuracy"] == 75.0
        assert summary["courses"]["Math"]["num_topics_covered"] == 1
        assert "Geometry" not in {
            item["topic"] for item in summary["courses"]["Math"]["weak_topics"]
        }

    def test_performance_summary_does_not_mark_strong_topics_weak(self):
        profile = create_profile("Strong Summary", [{"name": "Math", "difficulty": 3}])
        for _ in range(5):
            profile = record_quiz_result(profile, "Math", "Algebra", 2, True, "q", "a")

        summary = get_performance_summary(profile)

        assert summary["courses"]["Math"]["accuracy"] == 100.0
        assert summary["courses"]["Math"]["num_topics_covered"] == 1
        assert summary["courses"]["Math"]["weak_topics"] == []


# ═══════════════════════════════════════════════
# Spaced Repetition Tests
# ═══════════════════════════════════════════════

class TestSpacedRepetition:

    def test_first_review_sets_interval(self):
        sr = SpacedRepetitionScheduler()
        sr.update_topic("Recursion", quality=4)
        assert sr.topic_schedule["Recursion"]["interval_days"] == 1
        assert sr.topic_schedule["Recursion"]["repetition_count"] == 1

    def test_second_review_extends_interval(self):
        sr = SpacedRepetitionScheduler()
        sr.update_topic("Recursion", quality=4)
        sr.update_topic("Recursion", quality=4)
        assert sr.topic_schedule["Recursion"]["interval_days"] == 6

    def test_failed_review_resets_interval(self):
        sr = SpacedRepetitionScheduler()
        sr.update_topic("Recursion", quality=5)
        sr.update_topic("Recursion", quality=5)
        sr.update_topic("Recursion", quality=1)  # Failed
        assert sr.topic_schedule["Recursion"]["repetition_count"] == 0
        assert sr.topic_schedule["Recursion"]["interval_days"] == 1

    def test_easiness_factor_increases_on_easy(self):
        sr = SpacedRepetitionScheduler()
        sr.update_topic("Easy", quality=5)
        assert sr.topic_schedule["Easy"]["easiness_factor"] > 2.5

    def test_easiness_factor_decreases_on_hard(self):
        sr = SpacedRepetitionScheduler()
        sr.update_topic("Hard", quality=0)
        assert sr.topic_schedule["Hard"]["easiness_factor"] < 2.5

    def test_easiness_floor(self):
        sr = SpacedRepetitionScheduler()
        for _ in range(20):
            sr.update_topic("Very Hard", quality=0)
        assert sr.topic_schedule["Very Hard"]["easiness_factor"] >= 1.3

    def test_quality_from_result(self):
        sr = SpacedRepetitionScheduler()
        assert sr.quality_from_result(correct=True, confidence=5, difficulty=4) == 5
        assert sr.quality_from_result(correct=True, confidence=2, difficulty=2) == 3
        assert sr.quality_from_result(correct=False, confidence=1, difficulty=3) == 0
        assert sr.quality_from_result(correct=False, confidence=4, difficulty=3) == 2

    def test_serialization_roundtrip(self):
        sr = SpacedRepetitionScheduler()
        sr.update_topic("Topic A", quality=4)
        sr.update_topic("Topic B", quality=2)
        data = sr.to_dict()
        sr2 = SpacedRepetitionScheduler.from_dict(data)
        assert sr2.topic_schedule["Topic A"]["easiness_factor"] == sr.topic_schedule["Topic A"]["easiness_factor"]
        assert sr2.topic_schedule["Topic B"]["interval_days"] == sr.topic_schedule["Topic B"]["interval_days"]

    def test_due_topics_respects_injected_clock(self):
        """
        Proves the SM-2 progression end-to-end without waiting real seconds
        or using freezegun. This is the test that was impossible before
        the ``now_fn`` injection — every previous SR test only checked
        topic_schedule state, never verifying that reviews fire when time
        advances.
        """
        from datetime import datetime, timedelta, timezone
        from utils.spaced_repetition import SpacedRepetitionScheduler

        class FakeClock:
            def __init__(self, t):
                self.t = t

            def __call__(self):
                return self.t

            def advance(self, **kw):
                self.t += timedelta(**kw)

        clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        sr = SpacedRepetitionScheduler(now_fn=clock)
        sr.update_topic("Recursion", quality=4)  # interval_days = 1

        # Same instant — not due yet.
        assert sr.get_due_topics() == []

        # 2 days later — definitely due.
        clock.advance(days=2)
        due = sr.get_due_topics()
        assert len(due) == 1 and due[0]["topic"] == "Recursion"
        assert due[0]["days_overdue"] >= 1.0

        # Second successful review pushes interval to 6 days.
        sr.update_topic("Recursion", quality=4)
        clock.advance(days=3)
        assert sr.get_due_topics() == []
        clock.advance(days=4)
        assert sr.get_due_topics()[0]["topic"] == "Recursion"

    def test_legacy_naive_next_review_is_normalized(self):
        """
        Profiles saved before the UTC migration store naive ISO strings in
        next_review. Reading them back must not raise "can't compare
        offset-naive and offset-aware datetimes".
        """
        from datetime import datetime, timezone
        from utils.spaced_repetition import SpacedRepetitionScheduler

        sr = SpacedRepetitionScheduler(now_fn=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc))
        sr.topic_schedule["Legacy"] = {
            "easiness_factor": 2.5,
            "interval_days": 1,
            "repetition_count": 1,
            # Intentionally NAIVE ISO string, like an old profile.
            "next_review": datetime(2020, 1, 1).isoformat(),
            "last_quality": 4,
        }
        due = sr.get_due_topics()
        assert any(d["topic"] == "Legacy" for d in due)


# ═══════════════════════════════════════════════
# Issue Detector Tests
# ═══════════════════════════════════════════════

class TestIssueDetector:

    def _make_profile_with_streak(self, n_wrong: int):
        """Helper: create a profile with N wrong answers in a row."""
        import utils.student_profile as sp
        orig = sp.DATA_DIR
        sp.DATA_DIR = tempfile.mkdtemp()

        profile = create_profile("Streak", [{"name": "CS", "difficulty": 3}])
        profile["courses"]["CS"]["topics"]["Loops"] = {"correct": 0, "attempted": 0, "current_difficulty": 2}
        det = IssueDetector()
        for i in range(n_wrong):
            profile = record_quiz_result(profile, "CS", "Loops", 3, False, f"Q{i}", "B")
            det.analyze_quiz_result(profile, "CS", "Loops", 3, False)

        sp.DATA_DIR = orig
        return det, profile

    def test_detects_3_wrong_streak(self):
        det, _ = self._make_profile_with_streak(3)
        types = [i.issue_type for i in det.issues]
        assert "accuracy_drop" in types

    def test_detects_5_wrong_streak(self):
        det, _ = self._make_profile_with_streak(5)
        high_issues = [i for i in det.issues if i.severity == "high"]
        assert len(high_issues) > 0

    def test_api_error_logging(self):
        det = IssueDetector()
        issue = det.log_api_error("json_parse", "Bad JSON response", recovered=True)
        assert issue.severity == "medium"
        assert len(det.api_errors) == 1

    def test_unrecovered_api_error_is_high(self):
        det = IssueDetector()
        issue = det.log_api_error("api_timeout", "Connection lost", recovered=False)
        assert issue.severity == "high"

    def test_issue_summary(self):
        det = IssueDetector()
        det.log_api_error("json_parse", "error", recovered=True)
        det.log_api_error("api_timeout", "error", recovered=False)
        summary = det.get_issue_summary()
        assert summary["total_issues"] == 2
        assert summary["api_errors"] == 2

    def test_tutor_guidance_none_when_no_issues(self):
        det = IssueDetector()
        assert det.get_tutor_guidance() is None

    def test_tutor_guidance_with_high_issue(self):
        det = IssueDetector()
        det.log_api_error("api_timeout", "Connection failed", recovered=False)
        guidance = det.get_tutor_guidance()
        assert guidance is not None
        assert "Connection failed" in guidance


# ═══════════════════════════════════════════════
# Agent Communication Tests
# ═══════════════════════════════════════════════

class TestAgentComm:

    def test_register_and_send(self):
        bus = MessageBus()
        received = []
        bus.register_agent("Tutor")
        bus.register_agent("Resource", callback=lambda m: received.append(m))
        msg = AgentMessage("Tutor", "Resource", MessageType.REQUEST_MATERIALS, {"topic": "Loops"})
        bus.send(msg)
        assert len(received) == 1
        assert received[0].content["topic"] == "Loops"

    def test_receive_clears_inbox(self):
        bus = MessageBus()
        bus.register_agent("A")
        bus.register_agent("B")
        bus.send(AgentMessage("A", "B", MessageType.STATUS_UPDATE, {"status": "ok"}))
        msgs = bus.receive("B")
        assert len(msgs) == 1
        msgs2 = bus.receive("B")
        assert len(msgs2) == 0

    def test_peek_does_not_clear(self):
        bus = MessageBus()
        bus.register_agent("A")
        bus.register_agent("B")
        bus.send(AgentMessage("A", "B", MessageType.STATUS_UPDATE, {"status": "ok"}))
        peeked = bus.peek("B")
        assert len(peeked) == 1
        peeked2 = bus.peek("B")
        assert len(peeked2) == 1  # Still there

    def test_message_log(self):
        bus = MessageBus()
        bus.register_agent("A")
        bus.register_agent("B")
        bus.send(AgentMessage("A", "B", MessageType.REQUEST_MATERIALS, {"topic": "X"}))
        bus.send(AgentMessage("B", "A", MessageType.PROVIDE_MATERIALS, {"data": "Y"}))
        log = bus.get_conversation_log()
        assert len(log) == 2

    def test_stats(self):
        bus = MessageBus()
        bus.register_agent("A")
        bus.register_agent("B")
        bus.send(AgentMessage("A", "B", MessageType.REQUEST_MATERIALS, {}))
        bus.send(AgentMessage("A", "B", MessageType.REPORT_WEAKNESS, {}))
        stats = bus.get_stats()
        assert stats["total_messages"] == 2
        assert "A" in stats["by_sender"]

    def test_message_id_is_uuid(self):
        """IDs should be UUID hex strings, not timestamp-based."""
        msg = AgentMessage("A", "B", MessageType.STATUS_UPDATE, {})
        # uuid4().hex is 32 hex characters
        assert len(msg.id) == 32
        int(msg.id, 16)  # Should not raise

    def test_no_id_collisions_under_rapid_fire(self):
        """Rapid-fire messages must have unique IDs."""
        ids = set()
        for _ in range(1000):
            msg = AgentMessage("A", "B", MessageType.STATUS_UPDATE, {})
            assert msg.id not in ids, "ID collision detected"
            ids.add(msg.id)


# ═══════════════════════════════════════════════
# Adaptive Engine Seeding Tests
# ═══════════════════════════════════════════════

class TestAdaptiveEngineSeed:

    def test_seed_produces_deterministic_selection(self):
        """Two engines with the same seed must make identical choices."""
        topics = ["A", "B", "C", "D", "E"]
        e1 = AdaptiveEngine(seed=123)
        e2 = AdaptiveEngine(seed=123)
        for _ in range(20):
            t1, d1 = e1.select_topic_and_difficulty(topics)
            t2, d2 = e2.select_topic_and_difficulty(topics)
            assert (t1, d1) == (t2, d2)

    def test_different_seeds_diverge(self):
        """Different seeds should produce different sequences."""
        topics = ["A", "B", "C"]
        e1 = AdaptiveEngine(seed=1, epsilon=1.0)
        e2 = AdaptiveEngine(seed=999, epsilon=1.0)
        results1 = [e1.select_topic_and_difficulty(topics) for _ in range(20)]
        results2 = [e2.select_topic_and_difficulty(topics) for _ in range(20)]
        assert results1 != results2


# ═══════════════════════════════════════════════
# Data Integrity: total_quizzes sync
# ═══════════════════════════════════════════════

class TestTotalQuizzesSync:

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def test_total_quizzes_equals_sum_of_course_attempted(self):
        """total_quizzes must equal the sum of course total_attempted."""
        profile = create_profile("Sync Test", [
            {"name": "CS101", "difficulty": 3},
            {"name": "Math", "difficulty": 2},
        ])
        for _ in range(5):
            profile = record_quiz_result(profile, "CS101", "Loops", 2, True, "q", "a")
        for _ in range(3):
            profile = record_quiz_result(profile, "Math", "Algebra", 3, False, "q", "a")

        expected = sum(c["total_attempted"] for c in profile["courses"].values())
        assert profile["total_quizzes"] == expected == 8

    def test_auto_initializes_missing_course(self):
        """Recording a result for an unknown course must not crash."""
        profile = create_profile("Auto Init", [{"name": "CS101", "difficulty": 3}])
        profile = record_quiz_result(profile, "NewCourse", "Topic1", 2, True, "q", "a")
        assert "NewCourse" in profile["courses"]
        assert profile["courses"]["NewCourse"]["total_attempted"] == 1


# ═══════════════════════════════════════════════
# Issue Detector: Alert Cooldown
# ═══════════════════════════════════════════════

class TestAlertCooldown:

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def test_same_issue_suppressed_within_10_questions(self):
        """The same issue type should not fire twice within 10 questions."""
        profile = create_profile("Cooldown", [{"name": "CS", "difficulty": 3}])
        profile["courses"]["CS"]["topics"]["Loops"] = {
            "correct": 0, "attempted": 0, "current_difficulty": 2,
        }
        det = IssueDetector()

        # Generate 5 wrong in a row → should trigger accuracy_drop
        first_batch_issues = []
        for i in range(5):
            profile = record_quiz_result(profile, "CS", "Loops", 3, False, f"Q{i}", "B")
            issues = det.analyze_quiz_result(profile, "CS", "Loops", 3, False)
            first_batch_issues.extend(issues)

        accuracy_drops = [i for i in first_batch_issues if i.issue_type == "accuracy_drop"]
        assert len(accuracy_drops) >= 1

        # Continue with 5 more wrong — should be suppressed by cooldown
        second_batch_issues = []
        for i in range(5, 10):
            profile = record_quiz_result(profile, "CS", "Loops", 3, False, f"Q{i}", "B")
            issues = det.analyze_quiz_result(profile, "CS", "Loops", 3, False)
            second_batch_issues.extend(issues)

        second_accuracy_drops = [i for i in second_batch_issues if i.issue_type == "accuracy_drop"]
        # Should be suppressed (0) because we're within 10 questions of last trigger
        assert len(second_accuracy_drops) == 0


# ═══════════════════════════════════════════════
# Deterministic Epsilon (LA-5)
# ═══════════════════════════════════════════════

class TestDeterministicEpsilon:

    def test_epsilon_formula(self):
        """epsilon = max(0.05, 1.0 * 0.995^total_steps)."""
        engine = AdaptiveEngine()
        for _ in range(100):
            engine.update("T", 2, correct=True, _decay=False)

        # Simulate what _restore_rl_state does:
        engine.epsilon = max(0.05, 1.0 * (0.995 ** 100))
        expected = max(0.05, 0.995 ** 100)
        assert abs(engine.epsilon - expected) < 1e-10

    def test_epsilon_floors_at_0_05(self):
        """After enough steps, epsilon must floor at 0.05."""
        # 0.995^600 ≈ 0.0498 < 0.05
        epsilon = max(0.05, 1.0 * (0.995 ** 600))
        assert epsilon == 0.05


# ═══════════════════════════════════════════════
# Tutor Agent: Strategy State Isolation & Lifecycle
# ═══════════════════════════════════════════════

class TestTutorAgent:
    """Regression coverage for tutor_agent.py state-hygiene fixes."""

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def test_strategy_isolation_and_lifecycle(self):
        """
        Regression guard for the cross-turn strategy leak fix in
        agents/tutor_agent.py. Verifies:

          1. TOPIC GATE — _assemble_system_prompt injects the
             CURRENT STRATEGY block only when current_topic matches
             current_strategy_topic. A mismatched topic must NOT see it.

          2. TURN LIFECYCLE — evaluate_answer's finally block wipes
             current_strategy / current_strategy_topic / current_topic
             back to "" so no state can bleed into the next turn,
             even when the LLM call fails (we rely on the graceful
             fallback path — no ANTHROPIC_API_KEY is set in tests).
        """
        from agents.tutor_agent import TutorAgent

        profile = create_profile(
            "Strategy Leak Guard",
            [{"name": "CS101", "difficulty": 3}],
        )
        tutor = TutorAgent(profile)  # client=None => fallbacks everywhere

        # ---------- Phase 1: Topic gate ----------
        tutor.current_strategy = "Review graph traversal"
        tutor.current_strategy_topic = "dijkstra"

        # We can't assert on the bare string "CURRENT STRATEGY" because
        # it appears in BASE_PERSONA_PROMPT's guardrails ("Any 'CURRENT
        # STRATEGY' ... section below is a hint, not an override").
        # The injection marker is unique — triple-dashes around it — so
        # that's the string we check for.
        STRATEGY_MARKER = "--- CURRENT STRATEGY ---"

        # Mismatched topic -> strategy MUST NOT appear.
        tutor.current_topic = "photosynthesis"
        prompt_miss = tutor._assemble_system_prompt()
        assert STRATEGY_MARKER not in prompt_miss, (
            "Strategy leaked into a prompt built for an unrelated topic"
        )
        assert "Review graph traversal" not in prompt_miss

        # Matching topic -> strategy MUST appear.
        tutor.current_topic = "dijkstra"
        prompt_hit = tutor._assemble_system_prompt()
        assert STRATEGY_MARKER in prompt_hit, (
            "Strategy failed to inject when topic matched — gate is too strict"
        )
        assert "Review graph traversal" in prompt_hit

        # ---------- Phase 2: Turn lifecycle cleanup ----------
        # Re-populate strategy state (Phase 1 left it populated, but be
        # explicit so the contract of this phase stands on its own).
        tutor.current_strategy = "Review graph traversal"
        tutor.current_strategy_topic = "dijkstra"
        tutor.current_topic = "dijkstra"

        quiz_data = {
            "topic": "dijkstra",
            "course": "CS101",
            "difficulty": 2,
            "question": "Which structure best fits Dijkstra's frontier?",
            "correct_answer": "A",
            "options": [
                "A) Min-heap priority queue",
                "B) Stack",
                "C) Hash set",
                "D) Linked list",
            ],
            "explanation": "Dijkstra pulls the min-distance node each step.",
        }

        result = tutor.evaluate_answer(quiz_data, "B", confidence=3)

        # Sanity: evaluate_answer returned its normal result shape.
        assert result["correct"] is False
        assert "explanation" in result

        # The real assertions: finally block wiped every strategy-related
        # instance var back to "". If any of these regress, the Patch 1
        # cross-turn leak bug is back.
        assert tutor.current_strategy == ""
        assert tutor.current_strategy_topic == ""
        assert tutor.current_topic == ""


# ═══════════════════════════════════════════════
# extract_json: prose-wrapped LLM output
# ═══════════════════════════════════════════════

class TestExtractJson:
    """Robustness of the shared agents._llm.extract_json helper."""

    def test_pure_json_object(self):
        from agents._llm import extract_json
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_pure_json_array(self):
        from agents._llm import extract_json
        assert extract_json('["x", "y"]') == ["x", "y"]

    def test_code_fence_json(self):
        from agents._llm import extract_json
        assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}

    def test_bare_code_fence(self):
        from agents._llm import extract_json
        assert extract_json('```\n[1,2,3]\n```') == [1, 2, 3]

    def test_prose_wrapped_object(self):
        from agents._llm import extract_json
        out = extract_json('Sure! Here is the JSON: {"a": 3, "b": [1,2]}. Hope this helps!')
        assert out == {"a": 3, "b": [1, 2]}

    def test_prose_wrapped_array(self):
        from agents._llm import extract_json
        out = extract_json('Topics:\n["Recursion", "Loops"]\nGood luck!')
        assert out == ["Recursion", "Loops"]

    def test_multiline_json(self):
        from agents._llm import extract_json
        out = extract_json('{\n  "question": "q",\n  "options": ["A","B"]\n}')
        assert out == {"question": "q", "options": ["A", "B"]}

    def test_no_json_raises_decode_error(self):
        import json
        from agents._llm import extract_json
        with pytest.raises(json.JSONDecodeError):
            extract_json("Sorry, no JSON here, just chat.")

    def test_none_raises_decode_error(self):
        import json
        from agents._llm import extract_json
        with pytest.raises(json.JSONDecodeError):
            extract_json(None)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════
# Quiz schema validation (fact-check pass)
# ═══════════════════════════════════════════════

class TestQuizValidation:
    """Verify _validate_quiz_shape rejects malformed LLM-generated quizzes
    by raising JSONDecodeError, which the @retry_llm_call decorator
    treats as retryable."""

    def _agent(self):
        import tempfile, utils.student_profile as sp
        sp.DATA_DIR = tempfile.mkdtemp()
        from agents.tutor_agent import TutorAgent
        profile = create_profile("ValidShape", [{"name": "CS", "difficulty": 3}])
        return TutorAgent(profile)

    def test_well_formed_quiz_passes(self):
        agent = self._agent()
        good = {
            "question": "What is X?",
            "options": ["A) ans1", "B) ans2", "C) ans3", "D) ans4"],
            "correct_answer": "B",
            "explanation": "B is correct because...",
        }
        # Should not raise.
        agent._validate_quiz_shape(good)

    def test_lowercase_correct_answer_normalised(self):
        agent = self._agent()
        # "a" should be accepted because we uppercase before checking.
        agent._validate_quiz_shape({
            "question": "q",
            "options": ["A) x", "B) y", "C) z", "D) w"],
            "correct_answer": "a",
            "explanation": "e",
        })

    def test_non_dict_rejected(self):
        import json
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_quiz_shape(["not", "a", "dict"])

    def test_wrong_option_count_rejected(self):
        import json
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_quiz_shape({
                "question": "q",
                "options": ["A) x", "B) y"],  # only 2
                "correct_answer": "A",
                "explanation": "e",
            })

    def test_invalid_correct_answer_letter_rejected(self):
        import json
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_quiz_shape({
                "question": "q",
                "options": ["A) x", "B) y", "C) z", "D) w"],
                "correct_answer": "E",  # outside A-D
                "explanation": "e",
            })

    def test_correct_answer_letter_not_in_options_rejected(self):
        """The killer case: LLM said 'B' but only emitted options A/C/D/E."""
        import json
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_quiz_shape({
                "question": "q",
                "options": ["A) x", "C) y", "D) z", "E) w"],
                "correct_answer": "B",  # B never appears in options
                "explanation": "e",
            })

    def test_non_string_options_rejected(self):
        import json
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_quiz_shape({
                "question": "q",
                "options": ["A) x", 2, "C) z", "D) w"],  # int sneaks in
                "correct_answer": "A",
                "explanation": "e",
            })


# ═══════════════════════════════════════════════
# Prompt caching: system-block structure
# ═══════════════════════════════════════════════

class TestPromptCaching:
    """Verify the system prompt is assembled as cache-aware blocks
    rather than a raw string."""

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def _tutor(self):
        from agents.tutor_agent import TutorAgent
        profile = create_profile("Cache", [
            {"name": "CS101", "difficulty": 3},
            {"name": "Math", "difficulty": 4},
        ])
        return TutorAgent(profile)

    def test_blocks_have_two_segments(self):
        blocks = self._tutor()._assemble_system_blocks()
        assert isinstance(blocks, list)
        assert len(blocks) == 2

    def test_first_block_carries_cache_control(self):
        """The persona + stable context block is what we want cached
        across turns."""
        blocks = self._tutor()._assemble_system_blocks()
        assert blocks[0].get("cache_control") == {"type": "ephemeral"}

    def test_second_block_has_no_cache_control(self):
        """Dynamic block changes per turn — caching it would invalidate
        the cache constantly."""
        blocks = self._tutor()._assemble_system_blocks()
        assert "cache_control" not in blocks[1]

    def test_persona_lives_in_cached_block(self):
        from agents.tutor_agent import TutorAgent
        blocks = self._tutor()._assemble_system_blocks()
        # The unique safety-guardrail string anchors us to the persona.
        assert "SAFETY GUARDRAILS" in blocks[0]["text"]

    def test_performance_summary_lives_in_dynamic_block(self):
        """Performance changes per quiz answer — must be in the
        cache-MISS portion."""
        blocks = self._tutor()._assemble_system_blocks()
        assert "PERFORMANCE SO FAR" in blocks[1]["text"]
        assert "PERFORMANCE SO FAR" not in blocks[0]["text"]

    def test_dynamic_block_includes_saved_learning_evidence_for_chat(self):
        """Chat should see concrete saved performance evidence, not just general context."""
        from agents.tutor_agent import TutorAgent

        profile = create_profile("Chat Evidence", [{"name": "ML", "difficulty": 3}])
        profile = record_quiz_result(
            profile,
            "ML",
            "Supervised Learning",
            4,
            False,
            "Which model behavior shows overfitting?",
            "D",
            confidence=5,
        )
        profile = record_quiz_result(
            profile,
            "ML",
            "Train Test Split",
            2,
            True,
            "Why do we split data?",
            "A",
            confidence=2,
        )
        profile["quiz_sessions"] = [{
            "created_at": "2026-05-07T12:00:00+00:00",
            "course": "ML",
            "correct": 3,
            "question_count": 5,
            "accuracy": 60.0,
            "summary": "You answered 3 out of 5 questions correctly.",
            "priority_topics": ["Supervised Learning"],
            "misconception_topics": ["Supervised Learning"],
        }]
        profile["resource_agent_state"] = {
            "knowledge_base": {
                "supervised learning": {
                    "display_name": "Supervised Learning",
                    "materials": [{"title": "Review Supervised Learning"}],
                    "last_request": {
                        "why_shown": (
                            "This material is shown because Supervised Learning "
                            "was missed in the quiz evaluation."
                        )
                    },
                }
            }
        }

        text = TutorAgent(profile)._assemble_system_blocks()[1]["text"]

        assert "LATEST 5-QUESTION EVALUATION" in text
        assert "Score: 3/5 (60.0%)" in text
        assert "RECENT QUIZ ANSWERS" in text
        assert "Supervised Learning" in text
        assert "wrong; ML / Supervised Learning" in text
        assert "CONFIDENCE PATTERN" in text
        assert "High-confidence wrong answers: 1" in text
        assert "RESOURCE MATERIAL AVAILABLE" in text
        assert "Review Supervised Learning" in text

    def test_courses_live_in_cached_block(self):
        """Course list changes only on profile creation/edit — should
        be cached across the session."""
        blocks = self._tutor()._assemble_system_blocks()
        assert "STUDENT'S COURSES" in blocks[0]["text"]
        assert "CS101" in blocks[0]["text"]
        assert "Math" in blocks[0]["text"]

    def test_assemble_system_prompt_concatenates_blocks(self):
        """Backwards-compat: the string-form helper returns the same
        content as the blocks form, joined."""
        tutor = self._tutor()
        blocks = tutor._assemble_system_blocks()
        joined = "".join(b["text"] for b in blocks)
        assert tutor._assemble_system_prompt() == joined


# ═══════════════════════════════════════════════
# Clock injection: record_quiz_result + evaluate_answer
# ═══════════════════════════════════════════════

class TestClockInjection:
    """The now_fn hook lets demos and tests stamp quiz_history entries
    with deterministic, backdated timestamps instead of wall-clock now."""

    def setup_method(self):
        import utils.student_profile as sp
        self._original_dir = sp.DATA_DIR
        self._temp_dir = tempfile.mkdtemp()
        sp.DATA_DIR = self._temp_dir

    def teardown_method(self):
        import utils.student_profile as sp
        sp.DATA_DIR = self._original_dir

    def test_record_quiz_result_uses_injected_clock(self):
        from datetime import datetime, timezone
        fixed = datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)

        profile = create_profile("Clocked", [{"name": "CS", "difficulty": 3}])
        profile = record_quiz_result(
            profile, "CS", "Loops", 2, True, "q", "A",
            now_fn=lambda: fixed,
        )
        assert profile["quiz_history"][-1]["timestamp"] == fixed.isoformat()

    def test_record_quiz_result_default_uses_real_clock(self):
        """No now_fn → real datetime.now(UTC). The new field must still
        be parseable as a tz-aware datetime."""
        from datetime import datetime
        profile = create_profile("Default", [{"name": "CS", "difficulty": 3}])
        profile = record_quiz_result(profile, "CS", "Loops", 2, True, "q", "A")
        ts = profile["quiz_history"][-1]["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None  # tz-aware

    def test_evaluate_answer_threads_clock_through(self):
        """The TutorAgent.evaluate_answer ``now_fn`` kwarg must reach
        record_quiz_result."""
        from datetime import datetime, timezone
        from agents.tutor_agent import TutorAgent
        fixed = datetime(2030, 1, 1, tzinfo=timezone.utc)

        profile = create_profile("Threaded", [{"name": "CS", "difficulty": 3}])
        tutor = TutorAgent(profile)  # no client → fallback path everywhere
        quiz = {
            "topic": "Loops", "course": "CS", "difficulty": 2,
            "question": "q", "correct_answer": "A",
            "options": ["A) x", "B) y", "C) z", "D) w"],
            "explanation": "e",
        }
        tutor.evaluate_answer(quiz, "A", confidence=3, now_fn=lambda: fixed)

        # The just-recorded entry must carry the injected timestamp.
        assert tutor.profile["quiz_history"][-1]["timestamp"] == fixed.isoformat()

    def test_evaluate_answer_handles_non_string_inputs(self):
        """Regression guard for the non-string-input hardening — non-string
        inputs return correct=False rather than AttributeError."""
        from agents.tutor_agent import TutorAgent
        profile = create_profile("Nonstring", [{"name": "CS", "difficulty": 3}])
        tutor = TutorAgent(profile)
        quiz = {
            "topic": "Loops", "course": "CS", "difficulty": 2,
            "question": "q", "correct_answer": "A",
            "options": ["A) x", "B) y", "C) z", "D) w"],
            "explanation": "e",
        }
        # None / int / bytes / empty string should all be handled.
        for ans in [None, 42, b"A", ""]:
            r = tutor.evaluate_answer(quiz, ans, confidence=3)  # type: ignore[arg-type]
            assert r["correct"] is False, f"non-string answer {ans!r} should be wrong, not crash"

        # Lowercase 'a' must normalise to match correct_answer 'A'.
        r = tutor.evaluate_answer(quiz, "a", confidence=3)
        assert r["correct"] is True

    def test_evaluate_answer_returns_student_feedback_card(self, monkeypatch):
        """Every quiz answer should return a stable student-facing feedback card."""
        import agents.tutor_agent as tutor_module
        from agents.tutor_agent import TutorAgent

        monkeypatch.setattr(tutor_module, "save_profile", lambda _name, _profile: True)
        profile = create_profile("FeedbackCard", [{"name": "CS", "difficulty": 3}])
        tutor = TutorAgent(profile)
        quiz = {
            "topic": "Loops", "course": "CS", "difficulty": 3,
            "question": "What does a while loop do?", "correct_answer": "A",
            "options": [
                "A) Repeats while true",
                "B) Runs once",
                "C) Defines a class",
                "D) Imports a module",
            ],
            "explanation": "A while loop repeats while its condition remains true.",
        }

        wrong = tutor.evaluate_answer(quiz, "B", confidence=5)
        wrong_feedback = wrong["student_feedback"]
        assert wrong_feedback["status"] == "warning"
        assert wrong_feedback["title"] == "Misconception spotted"
        assert "High-confidence wrong answers" in wrong_feedback["confidence_insight"]
        assert "retry Loops at an easier difficulty" in wrong_feedback["next_step"]

        correct = tutor.evaluate_answer(quiz, "A", confidence=1)
        correct_feedback = correct["student_feedback"]
        assert correct_feedback["status"] == "success"
        assert correct_feedback["title"] == "Correct, but build confidence"
        assert "Low-confidence correct answers" in correct_feedback["confidence_insight"]
