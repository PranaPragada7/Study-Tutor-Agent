"""Smoke-check the student-facing quiz feedback component.

This script avoids the live LLM and patches disk saves in-memory, so it is safe
to run right before presenting. It verifies that answer evaluation returns the
structured `student_feedback` payload used by the Streamlit Quiz tab.

Run:
    python scripts/feedback_check.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.tutor_agent as tutor_module
from agents.tutor_agent import TutorAgent
import utils.student_profile as profile_module
from utils.student_profile import create_profile


def _assert_feedback_shape(feedback: dict) -> None:
    required = {
        "title",
        "status",
        "topic",
        "summary",
        "confidence_insight",
        "next_step",
        "review_note",
        "resource_note",
    }
    missing = sorted(required - set(feedback))
    if missing:
        raise AssertionError(f"student_feedback missing keys: {missing}")
    for key in required:
        if not isinstance(feedback[key], str) or not feedback[key].strip():
            raise AssertionError(f"student_feedback[{key!r}] must be a non-empty string")


def main() -> int:
    original_save_profile = tutor_module.save_profile
    original_data_dir = profile_module.DATA_DIR
    tutor_module.save_profile = lambda _name, _profile: True
    try:
        with tempfile.TemporaryDirectory(prefix="study_tutor_feedback_check_") as tmp:
            profile_module.DATA_DIR = tmp
            profile = create_profile("Feedback Sample Profile", [{"name": "Intro to CS", "difficulty": 3}])
            return _run_feedback_check(profile)
    finally:
        profile_module.DATA_DIR = original_data_dir
        tutor_module.save_profile = original_save_profile


def _run_feedback_check(profile: dict) -> int:
    tutor = TutorAgent(profile, client=None)
    quiz = {
        "topic": "Recursion",
        "course": "Intro to CS",
        "difficulty": 3,
        "question": "What is the purpose of a base case in recursion?",
        "options": [
            "A) It stops recursive calls when a simple condition is met",
            "B) It makes every recursive call larger",
            "C) It stores all variables globally",
            "D) It removes the need for a function",
        ],
        "correct_answer": "A",
        "explanation": "A base case stops recursion so the calls eventually unwind.",
    }

    wrong = tutor.evaluate_answer(quiz, "B", confidence=5)
    correct = tutor.evaluate_answer(quiz, "A", confidence=1)

    _assert_feedback_shape(wrong["student_feedback"])
    _assert_feedback_shape(correct["student_feedback"])

    if wrong["student_feedback"]["title"] != "Misconception spotted":
        raise AssertionError("wrong/high-confidence path did not flag misconception")
    if correct["student_feedback"]["title"] != "Correct, but build confidence":
        raise AssertionError("correct/low-confidence path did not flag confidence-building")

    print("Student feedback check passed.")
    print(f"Wrong-answer feedback: {wrong['student_feedback']['title']}")
    print(f"Correct-answer feedback: {correct['student_feedback']['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
