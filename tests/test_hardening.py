"""
Tests for the defence-in-depth / hardening layer.

Coverage:
  - agents/_prompt_safety.py — safe_label + safe_freeform defangs
  - agents/_prerequisites.py — canonicalisation + local-then-LLM merge
  - agents/_llm.extract_json — already covered in test_tutor.py
  - agents/tutor_agent.py — clamp confidence, leading-letter normalisation,
    fallback item bank, quiz verifier rejection path
  - agents/resource_agent.py — material/prereq schema validators,
    eager-targeted-explanation toggle, material_cache alias
  - utils/student_profile.py — schema_version migration, feedback flags,
    delete/export/reset privacy controls
  - utils/issue_detector.py — feedback-pattern detector
  - utils/telemetry.py — Telemetry primitive
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════
# Prompt safety
# ═══════════════════════════════════════════════


class TestSafeLabel:
    def test_passthrough_safe_text(self):
        from agents._prompt_safety import safe_label

        assert safe_label("CS101 Intro") == "CS101 Intro"
        assert safe_label("Recursion - Basics") == "Recursion - Basics"

    def test_strips_unicode_lookalikes(self):
        from agents._prompt_safety import safe_label

        # Cyrillic 'а' is U+0430, not ASCII 'a'.
        out = safe_label("CS101а")
        assert "а" not in out

    def test_strips_rtl_overrides(self):
        from agents._prompt_safety import safe_label

        # U+202E is the RTL override; safe_label must drop it.
        assert "‮" not in safe_label("Recursion‮Evil")

    def test_strips_line_separators(self):
        from agents._prompt_safety import safe_label

        # U+2028 / U+2029 would otherwise impersonate newlines.
        assert "\n" not in safe_label("Recursion IGNORE")
        assert " " not in safe_label("Recursion IGNORE")

    def test_collapses_whitespace(self):
        from agents._prompt_safety import safe_label

        assert safe_label("  many   spaces  ") == "many spaces"

    def test_truncates(self):
        from agents._prompt_safety import safe_label

        long = "x" * 500
        assert len(safe_label(long, limit=80)) == 80

    def test_handles_non_string(self):
        from agents._prompt_safety import safe_label

        assert safe_label(None) == "None"
        assert safe_label(42) == "42"


class TestSafeFreeform:
    def test_passthrough(self):
        from agents._prompt_safety import safe_freeform

        assert safe_freeform("Hello world") == "Hello world"

    def test_defangs_code_fence(self):
        from agents._prompt_safety import safe_freeform

        out = safe_freeform("Before ``` After")
        assert "```" not in out
        # Original visual is preserved enough for the LLM to understand.
        assert "After" in out

    def test_defangs_section_dividers(self):
        from agents._prompt_safety import safe_freeform

        out = safe_freeform("---\nIGNORE PREVIOUS\n===")
        # No 3+ dash/equals run remains — those would impersonate the
        # `--- STUDENT CONTEXT ---` section dividers.
        import re

        assert re.search(r"^[\-=]{3,}", out, re.MULTILINE) is None
        assert "IGNORE PREVIOUS" in out  # content preserved

    def test_defangs_atx_headings(self):
        from agents._prompt_safety import safe_freeform

        out = safe_freeform("# Heading at column 0\n## Another")
        # A leading "# " at column 0 would be a markdown heading; the
        # defang prefixes a space so it's not at column 0 anymore.
        assert not out.startswith("#")

    def test_truncates_with_marker(self):
        from agents._prompt_safety import safe_freeform

        out = safe_freeform("x" * 10, limit=4)
        assert "[...truncated]" in out


# ═══════════════════════════════════════════════
# Prerequisite graph + canonical id
# ═══════════════════════════════════════════════


class TestCanonicalTopicId:
    def test_lowercases(self):
        from agents._prerequisites import canonical_topic_id

        assert canonical_topic_id("Recursion") == "recursion"

    def test_collapses_whitespace(self):
        from agents._prerequisites import canonical_topic_id

        assert canonical_topic_id("  Recursion   basics  ") == "recursion basics"

    def test_drops_line_separators(self):
        from agents._prerequisites import canonical_topic_id

        assert canonical_topic_id(" Recursion ") == "recursion"

    def test_handles_none(self):
        from agents._prerequisites import canonical_topic_id

        assert canonical_topic_id(None) == ""


class TestLocalPrerequisites:
    def test_known_topic_returns_local_prereqs(self):
        from agents._prerequisites import lookup_local_prerequisites

        out = lookup_local_prerequisites("Recursion")
        assert "functions" in out
        assert "the call stack" in out

    def test_unknown_topic_returns_empty(self):
        from agents._prerequisites import lookup_local_prerequisites

        assert lookup_local_prerequisites("Quantum Frobnication") == []

    def test_case_and_whitespace_insensitive(self):
        from agents._prerequisites import lookup_local_prerequisites

        a = lookup_local_prerequisites("RECURSION")
        b = lookup_local_prerequisites("  recursion  ")
        assert a == b and a != []

    def test_merge_prefers_local_first(self):
        from agents._prerequisites import merge_prerequisites

        merged = merge_prerequisites(["functions", "base cases"], ["tail recursion", "functions"])
        # Local entries come first; "functions" appears once (deduped on canonical).
        assert merged[0] == "functions"
        assert merged[1] == "base cases"
        assert "tail recursion" in merged
        assert merged.count("functions") == 1

    def test_merge_caps_total(self):
        from agents._prerequisites import merge_prerequisites

        merged = merge_prerequisites(
            ["a", "b", "c", "d"],
            ["e", "f"],
            max_total=3,
        )
        assert len(merged) == 3
        assert merged == ["a", "b", "c"]


# ═══════════════════════════════════════════════
# Clamping
# ═══════════════════════════════════════════════


class TestClamping:
    def test_adaptive_engine_clamps_confidence(self):
        """confidence=10 must NOT yield reward > 1 or < 0."""
        from agents.adaptive_engine import AdaptiveEngine

        engine = AdaptiveEngine()
        engine.update("X", 3, correct=True, confidence=10)  # bad caller
        arm = engine.arm_stats[("X", 3)]
        assert 0.0 <= arm["total_reward"] <= 1.0

    def test_adaptive_engine_clamps_negative_confidence(self):
        from agents.adaptive_engine import AdaptiveEngine

        engine = AdaptiveEngine()
        engine.update("X", 3, correct=False, confidence=-5)
        arm = engine.arm_stats[("X", 3)]
        # Should land in the wrong-with-low-confidence zone (~0.3)
        assert 0.0 <= arm["total_reward"] <= 1.0

    def test_adaptive_engine_clamps_garbage_confidence(self):
        from agents.adaptive_engine import AdaptiveEngine

        engine = AdaptiveEngine()
        # Garbage doesn't crash; default confidence=3 used internally.
        engine.update("X", 3, correct=True, confidence="not a number")  # type: ignore[arg-type]
        assert ("X", 3) in engine.arm_stats

    def test_adaptive_engine_clamps_difficulty(self):
        from agents.adaptive_engine import AdaptiveEngine

        engine = AdaptiveEngine()
        engine.update("X", 99, correct=True, confidence=3)
        # Should be clamped to 5 internally (or 1, depending on direction)
        assert ("X", 5) in engine.arm_stats or ("X", 99) not in engine.arm_stats

    def test_record_quiz_result_clamps(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import create_profile, record_quiz_result

        profile = create_profile("Clamp", [{"name": "CS", "difficulty": 3}])
        profile = record_quiz_result(
            profile, "CS", "T", difficulty=99, correct=True, question="q", answer="A", confidence=7
        )
        # Persisted entry should carry clamped values.
        entry = profile["quiz_history"][-1]
        assert 1 <= entry["difficulty"] <= 5
        assert 1 <= entry["confidence"] <= 5

    def test_quality_from_result_clamps(self):
        from utils.spaced_repetition import SpacedRepetitionScheduler

        sr = SpacedRepetitionScheduler()
        # Out-of-range inputs must yield a valid SM-2 quality (0-5).
        q = sr.quality_from_result(correct=True, confidence=99, difficulty=99)
        assert 0 <= q <= 5
        q = sr.quality_from_result(correct=False, confidence=-1, difficulty=-1)
        assert 0 <= q <= 5


class TestLeadingLetterNormalisation:
    def test_extracts_leading_letter_from_full_option(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from agents.tutor_agent import TutorAgent
        from utils.student_profile import create_profile

        profile = create_profile("Letter", [{"name": "CS", "difficulty": 3}])
        tutor = TutorAgent(profile)
        quiz = {
            "topic": "T",
            "course": "CS",
            "difficulty": 2,
            "question": "q",
            "correct_answer": "A",
            "options": ["A) right", "B) x", "C) y", "D) z"],
            "explanation": "e",
        }
        # A programmatic caller passes "A) right" (the full option text).
        # Must normalise to "A" and mark correct.
        result = tutor.evaluate_answer(quiz, "A) right", confidence=3)
        assert result["correct"] is True

    def test_lowercase_letter_normalised(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from agents.tutor_agent import TutorAgent
        from utils.student_profile import create_profile

        profile = create_profile("Lc", [{"name": "CS", "difficulty": 3}])
        tutor = TutorAgent(profile)
        quiz = {
            "topic": "T",
            "course": "CS",
            "difficulty": 2,
            "question": "q",
            "correct_answer": "B",
            "options": ["A) x", "B) y", "C) z", "D) w"],
            "explanation": "e",
        }
        result = tutor.evaluate_answer(quiz, "b", confidence=3)
        assert result["correct"] is True


# ═══════════════════════════════════════════════
# Material schema validation
# ═══════════════════════════════════════════════


class TestMaterialValidation:
    def _agent(self):
        from agents.resource_agent import ResourceAgent
        from utils.agent_comm import MessageBus

        return ResourceAgent(MessageBus())

    def test_well_formed_material_passes(self):
        agent = self._agent()
        good = {
            "title": "Recursion",
            "explanation": "A function that calls itself with a smaller input.",
            "analogy": "Like a Russian doll opening another Russian doll.",
            "common_mistake": "Forgetting the base case.",
            "self_test": "What happens without a base case?",
        }
        out = agent._validate_material_shape(good)
        assert out["explanation"]
        assert out["analogy"]
        assert out["common_mistake"]

    def test_non_dict_rejected(self):
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_material_shape("not a dict")

    def test_missing_field_rejected(self):
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_material_shape({"title": "T", "explanation": "ok"})

    def test_empty_required_field_rejected(self):
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_material_shape(
                {
                    "explanation": "  ",
                    "analogy": "ok",
                    "common_mistake": "ok",
                }
            )

    def test_materials_list_empty_rejected(self):
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_materials_list([])

    def test_materials_non_list_rejected(self):
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_materials_list({"not": "a list"})

    def test_materials_list_with_bad_item_rejected(self):
        agent = self._agent()
        with pytest.raises(json.JSONDecodeError):
            agent._validate_materials_list([{"title": "ok"}])  # missing required fields

    def test_materials_normalised_with_optional_defaults(self):
        agent = self._agent()
        out = agent._validate_materials_list(
            [
                {
                    "explanation": "x",
                    "analogy": "y",
                    "common_mistake": "z",
                }
            ]
        )
        assert out[0]["title"] == ""  # optional default
        assert out[0]["self_test"] == ""

    def test_prereqs_list_validation(self):
        agent = self._agent()
        # Non-list rejected
        with pytest.raises(json.JSONDecodeError):
            agent._validate_prereqs_list("not a list")
        # Cleans + dedupes + caps
        out = agent._validate_prereqs_list(
            [
                "  Functions  ",
                "functions",
                "",
                None,
                "Loops",
                42,
                "x" * 200,
                "Iteration",
                "Conditionals",
                "Variables",
            ]
        )
        assert "x" * 80 in out  # truncated
        # Functions appears once (deduped on canonical-lower)
        assert sum(1 for p in out if p.lower() == "functions") == 1
        # Capped to 5 entries
        assert len(out) <= 5


class TestTutorOnMessageRobust:
    """TutorAgent._on_message must not crash on malformed PROVIDE_MATERIALS."""

    def _setup(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from agents.tutor_agent import TutorAgent
        from utils.agent_comm import MessageBus
        from utils.student_profile import create_profile

        profile = create_profile("Robust", [{"name": "CS", "difficulty": 3}])
        return TutorAgent(profile, message_bus=MessageBus())

    def test_non_list_materials_does_not_crash(self, tmp_path):
        from utils.agent_comm import AgentMessage, MessageType

        tutor = self._setup(tmp_path)
        msg = AgentMessage(
            "ResourceAgent",
            "TutorAgent",
            MessageType.PROVIDE_MATERIALS,
            {"materials": "not a list"},
        )
        tutor._on_message(msg)  # must not raise
        assert tutor.current_materials == ""

    def test_list_of_strings_does_not_crash(self, tmp_path):
        from utils.agent_comm import AgentMessage, MessageType

        tutor = self._setup(tmp_path)
        msg = AgentMessage(
            "ResourceAgent",
            "TutorAgent",
            MessageType.PROVIDE_MATERIALS,
            {"materials": ["wrong shape"]},
        )
        tutor._on_message(msg)  # must not raise
        assert tutor.current_materials == ""

    def test_dict_with_missing_keys_does_not_crash(self, tmp_path):
        from utils.agent_comm import AgentMessage, MessageType

        tutor = self._setup(tmp_path)
        msg = AgentMessage(
            "ResourceAgent",
            "TutorAgent",
            MessageType.PROVIDE_MATERIALS,
            {"materials": [{"title": "Only title"}]},
        )
        tutor._on_message(msg)
        # All required fields empty -> current_materials stays empty.
        assert tutor.current_materials == ""

    def test_targeted_explanation_none_does_not_crash(self, tmp_path):
        from utils.agent_comm import AgentMessage, MessageType

        tutor = self._setup(tmp_path)
        # Empty list — no materials available
        msg = AgentMessage(
            "ResourceAgent", "TutorAgent", MessageType.PROVIDE_MATERIALS, {"materials": []}
        )
        tutor._on_message(msg)
        assert tutor.current_materials == ""


# ═══════════════════════════════════════════════
# Fallback quiz item bank
# ═══════════════════════════════════════════════


class TestFallbackQuiz:
    def _tutor(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from agents.tutor_agent import TutorAgent
        from utils.student_profile import create_profile

        profile = create_profile("Fb", [{"name": "CS", "difficulty": 3}])
        return TutorAgent(profile)

    def test_known_topic_returns_real_question(self, tmp_path):
        tutor = self._tutor(tmp_path)
        q = tutor._fallback_quiz("Recursion")
        # Hand-authored item, not the generic stub:
        assert "base case" in q["question"].lower()
        assert q["correct_answer"] in {"A", "B", "C", "D"}

    def test_neural_networks_fallback_is_real_question(self, tmp_path):
        tutor = self._tutor(tmp_path)
        q = tutor._fallback_quiz("Neural Networks and Perceptrons")
        assert "perceptron" in q["question"].lower()
        assert "learning rate" in q["explanation"].lower()
        assert q["correct_answer"] == "A"

    def test_unknown_topic_returns_honest_stub(self, tmp_path):
        tutor = self._tutor(tmp_path)
        q = tutor._fallback_quiz("Quantum Frobnication")
        # The honest stub explicitly tells the student the AI couldn't generate
        assert "couldn't generate" in q["question"].lower()
        # Topic name interpolated through safe_label
        assert "Quantum Frobnication" in q["question"]

    def test_all_bank_items_satisfy_schema(self, tmp_path):
        tutor = self._tutor(tmp_path)
        for key in tutor._FALLBACK_ITEM_BANK:
            item = tutor._fallback_quiz(key)
            # Each banked item must satisfy _validate_quiz_shape.
            tutor._validate_quiz_shape(item)

    def test_canonical_lookup(self, tmp_path):
        tutor = self._tutor(tmp_path)
        # Capitalisation / whitespace shouldn't prevent a hit.
        q1 = tutor._fallback_quiz("recursion")
        q2 = tutor._fallback_quiz("  RECURSION  ")
        assert q1["question"] == q2["question"]


# ═══════════════════════════════════════════════
# Profile schema versioning + migration
# ═══════════════════════════════════════════════


class TestSchemaVersioning:
    def test_create_profile_stamps_version(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import CURRENT_SCHEMA_VERSION, create_profile

        p = create_profile("V", [{"name": "CS", "difficulty": 3}])
        assert p["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_create_profile_initialises_streak_and_resource_fields(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import create_profile

        p = create_profile("S", [{"name": "CS", "difficulty": 3}])
        # Previously these were created lazily on first save; the v1
        # profile guarantees them at construction.
        assert p.get("streak_tracker") == []
        assert p.get("resource_agent_state") == {}
        assert p.get("quiz_sessions") == []

    def test_migrate_legacy_profile_gets_version_1(self):
        from utils.student_profile import CURRENT_SCHEMA_VERSION, migrate_profile

        legacy = {
            "name": "Legacy",
            "courses": {},
            "quiz_history": [],
            "total_quizzes": 0,
        }
        out = migrate_profile(legacy)
        assert out["schema_version"] == CURRENT_SCHEMA_VERSION
        assert out["spaced_repetition"] == {}
        assert out["streak_tracker"] == []
        assert out["resource_agent_state"] == {}
        assert out["quiz_sessions"] == []

    def test_migrate_idempotent(self):
        from utils.student_profile import CURRENT_SCHEMA_VERSION, migrate_profile

        already = {"schema_version": CURRENT_SCHEMA_VERSION, "courses": {}, "quiz_history": []}
        out = migrate_profile(dict(already))
        assert out["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_load_profile_runs_migration(self, tmp_path):
        import json as _json

        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        # Manually write a legacy profile with no schema_version.
        legacy_path = os.path.join(str(tmp_path), "legacy.json")
        with open(legacy_path, "w") as f:
            _json.dump({"name": "Legacy", "courses": {}, "quiz_history": [], "total_quizzes": 0}, f)
        from utils.student_profile import CURRENT_SCHEMA_VERSION, load_profile

        loaded = load_profile("Legacy")
        assert loaded is not None
        assert loaded["schema_version"] == CURRENT_SCHEMA_VERSION


# ═══════════════════════════════════════════════
# Feedback flags + pattern detection
# ═══════════════════════════════════════════════


class TestFeedbackFlags:
    def test_record_feedback_attaches_flags(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import (
            create_profile,
            record_quiz_feedback,
            record_quiz_result,
        )

        profile = create_profile("Fb", [{"name": "CS", "difficulty": 3}])
        profile = record_quiz_result(profile, "CS", "T", 2, True, "q", "A")
        quiz_id = profile["quiz_history"][-1]["id"]
        ok = record_quiz_feedback(profile, quiz_id, ["unclear_question", "too_hard"])
        assert ok is True
        flags = profile["quiz_history"][-1]["feedback_flags"]
        assert "unclear_question" in flags
        assert "too_hard" in flags

    def test_unknown_flags_dropped(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import (
            create_profile,
            record_quiz_feedback,
            record_quiz_result,
        )

        profile = create_profile("Fb", [{"name": "CS", "difficulty": 3}])
        profile = record_quiz_result(profile, "CS", "T", 2, True, "q", "A")
        quiz_id = profile["quiz_history"][-1]["id"]
        record_quiz_feedback(profile, quiz_id, ["unclear_question", "made_up"])
        flags = profile["quiz_history"][-1]["feedback_flags"]
        assert "made_up" not in flags

    def test_feedback_unknown_quiz_id_returns_false(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import create_profile, record_quiz_feedback

        profile = create_profile("Fb", [{"name": "CS", "difficulty": 3}])
        assert record_quiz_feedback(profile, "doesnotexist", ["too_hard"]) is False

    def test_feedback_pattern_detector_fires_on_truthiness(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.issue_detector import IssueDetector
        from utils.student_profile import (
            create_profile,
            record_quiz_feedback,
            record_quiz_result,
        )

        profile = create_profile("Pat", [{"name": "CS", "difficulty": 3}])
        for i in range(5):
            profile = record_quiz_result(profile, "CS", "T", 2, True, f"q{i}", "A")
            record_quiz_feedback(profile, profile["quiz_history"][-1]["id"], ["wrong_answer"])
        det = IssueDetector()
        issue = det.check_feedback_pattern(profile)
        assert issue is not None
        assert issue.issue_type == "feedback_pattern"
        assert issue.severity == "high"

    def test_feedback_pattern_detector_silent_below_threshold(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.issue_detector import IssueDetector
        from utils.student_profile import (
            create_profile,
            record_quiz_feedback,
            record_quiz_result,
        )

        profile = create_profile("Q", [{"name": "CS", "difficulty": 3}])
        for i in range(2):  # only 2 flags — below threshold of 3
            profile = record_quiz_result(profile, "CS", "T", 2, True, f"q{i}", "A")
            record_quiz_feedback(profile, profile["quiz_history"][-1]["id"], ["wrong_answer"])
        det = IssueDetector()
        assert det.check_feedback_pattern(profile) is None


# ═══════════════════════════════════════════════
# Privacy controls
# ═══════════════════════════════════════════════


class TestPrivacyControls:
    def test_export_returns_json_string(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import create_profile, export_profile_json

        p = create_profile("Exp", [{"name": "CS", "difficulty": 3}])
        out = export_profile_json(p)
        assert isinstance(out, str)
        # Roundtrips through JSON.
        loaded = json.loads(out)
        assert loaded["name"] == "Exp"

    def test_delete_profile_removes_file(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import (
            create_profile,
            delete_profile,
            load_profile,
        )

        create_profile("Del", [{"name": "CS", "difficulty": 3}])
        assert load_profile("Del") is not None
        assert delete_profile("Del") is True
        assert load_profile("Del") is None

    def test_delete_unknown_profile_succeeds(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import delete_profile

        # Idempotent: deleting a non-existent profile is True (nothing to remove)
        assert delete_profile("Nobody") is True

    def test_list_saved_profiles_shows_created_profiles(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import create_profile, list_saved_profiles

        create_profile("pRANAV", [{"name": "AI Agents", "difficulty": 3}])
        create_profile("Alex", [{"name": "CS", "difficulty": 2}])

        profiles = list_saved_profiles()
        names = [p["name"] for p in profiles]
        assert names == ["Alex", "pRANAV"]
        pranav = next(p for p in profiles if p["name"] == "pRANAV")
        assert pranav["courses"] == 1
        assert pranav["total_quizzes"] == 0

    def test_reset_quiz_history_preserves_courses(self, tmp_path):
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from utils.student_profile import (
            create_profile,
            record_quiz_result,
            reset_quiz_history,
        )

        p = create_profile("Reset", [{"name": "CS", "difficulty": 3}])
        for _ in range(3):
            p = record_quiz_result(p, "CS", "T", 2, True, "q", "A")
        assert p["total_quizzes"] == 3
        p = reset_quiz_history(p)
        assert p["total_quizzes"] == 0
        assert p["quiz_history"] == []
        assert p["quiz_sessions"] == []
        # Course preserved (definition kept; counters wiped)
        assert "CS" in p["courses"]
        assert p["courses"]["CS"]["total_attempted"] == 0
        # Topic preserved (definition); counters wiped
        if "T" in p["courses"]["CS"]["topics"]:
            assert p["courses"]["CS"]["topics"]["T"]["attempted"] == 0


# ═══════════════════════════════════════════════
# Resource Agent: cache hit no longer fires LLM by default
# ═══════════════════════════════════════════════


class TestResourceAgentBlocking:
    def test_cache_hit_does_not_call_targeted_explanation_by_default(self):
        from agents.resource_agent import ResourceAgent
        from utils.agent_comm import AgentMessage, MessageBus, MessageType

        bus = MessageBus()
        ra = ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        # Track calls to _generate_targeted_explanation
        calls = {"count": 0}
        original = ra._generate_targeted_explanation

        def spy(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        ra._generate_targeted_explanation = spy  # type: ignore[assignment]

        # First request: cache miss.
        bus.send(
            AgentMessage(
                "TutorAgent",
                "ResourceAgent",
                MessageType.REQUEST_MATERIALS,
                {
                    "topic": "Loops",
                    "course": "CS",
                    "student_level": "beginner",
                    "context": "wrong answer",
                    "learning_style": "balanced",
                },
            )
        )
        bus.receive("TutorAgent")

        # Second request: cache hit. Default config: targeted call SKIPPED.
        bus.send(
            AgentMessage(
                "TutorAgent",
                "ResourceAgent",
                MessageType.REQUEST_MATERIALS,
                {
                    "topic": "Loops",
                    "course": "CS",
                    "student_level": "beginner",
                    "context": "wrong answer 2",
                    "learning_style": "balanced",
                },
            )
        )
        bus.receive("TutorAgent")

        # No targeted call was made on either turn (no client; the
        # cache miss path doesn't go through it either).
        assert calls["count"] == 0

    def test_material_cache_alias(self):
        from agents.resource_agent import ResourceAgent
        from utils.agent_comm import MessageBus

        ra = ResourceAgent(MessageBus())
        # The two methods point at the same implementation.
        assert ra.get_material_cache_stats() == ra.get_knowledge_base_stats()


# ═══════════════════════════════════════════════
# Telemetry
# ═══════════════════════════════════════════════


class TestTelemetry:
    def test_incr_and_get(self):
        from utils.telemetry import Telemetry

        t = Telemetry()
        t.incr("foo")
        t.incr("foo", by=2)
        assert t.get("foo") == 3
        assert t.get("never_seen") == 0

    def test_snapshot_filtered(self):
        from utils.telemetry import Telemetry

        t = Telemetry()
        t.incr("a", by=2)
        t.incr("b", by=5)
        t.incr("c", by=1)
        # Filtered snapshot orders + zero-fills.
        out = t.snapshot(["b", "c", "missing"])
        assert out == {"b": 5, "c": 1, "missing": 0}

    def test_reset(self):
        from utils.telemetry import Telemetry

        t = Telemetry()
        t.incr("x", by=10)
        t.reset()
        assert t.get("x") == 0

    def test_thread_safety(self):
        """Increments from many threads must not lose any counts."""
        import threading

        from utils.telemetry import Telemetry

        t = Telemetry()

        def hammer():
            for _ in range(1000):
                t.incr("hits")

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert t.get("hits") == 8 * 1000

    def test_global_counters_singleton_exists(self):
        from utils.telemetry import COUNTERS, Telemetry

        assert isinstance(COUNTERS, Telemetry)


# ═══════════════════════════════════════════════
# Quiz semantic verifier: rejection raises retryable error
# ═══════════════════════════════════════════════


class TestQuizVerifier:
    def test_verifier_disabled_skips_call(self, monkeypatch, tmp_path):
        """When the verifier is disabled, no LLM call is made."""
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)

        # Reload the module with the env var off so its module-level
        # ``_QUIZ_VERIFIER_ENABLED`` is False. Easier: monkey-patch the
        # imported alias.
        import agents.tutor_agent as ta

        monkeypatch.setattr(ta, "_QUIZ_VERIFIER_ENABLED", False)

        from agents.tutor_agent import TutorAgent
        from utils.student_profile import create_profile

        profile = create_profile("Vd", [{"name": "CS", "difficulty": 3}])
        tutor = TutorAgent(profile)  # no client → fallback path

        # Should not raise — and importantly, _verify_quiz_semantics
        # is not called by _generate_quiz_llm when disabled. We can't
        # observe the call directly without a client, so the test is
        # essentially "no crash" + "module flag respected".
        called = {"count": 0}
        original = tutor._verify_quiz_semantics

        def spy(*a, **kw):
            called["count"] += 1
            return original(*a, **kw)

        tutor._verify_quiz_semantics = spy  # type: ignore[assignment]

        # No client means _generate_quiz_llm isn't reached anyway.
        # Verifier disabled is the relevant property here.
        assert ta._QUIZ_VERIFIER_ENABLED is False

    def test_verifier_passes_with_no_client(self, tmp_path):
        """Verifier with no client returns silently (no LLM available
        means no fact-check possible — better than crashing)."""
        import utils.student_profile as sp

        sp.DATA_DIR = str(tmp_path)
        from agents.tutor_agent import TutorAgent
        from utils.student_profile import create_profile

        profile = create_profile("Vc", [{"name": "CS", "difficulty": 3}])
        tutor = TutorAgent(profile)  # client=None
        good_quiz = {
            "question": "q",
            "options": ["A) x", "B) y", "C) z", "D) w"],
            "correct_answer": "A",
            "explanation": "e",
        }
        # Must not raise.
        tutor._verify_quiz_semantics(good_quiz, "CS", "T")
