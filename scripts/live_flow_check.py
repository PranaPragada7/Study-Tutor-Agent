"""All-in-one live-flow readiness check.

This script exercises the exact components that should be visible during the
presentation:

- environment/preflight checks
- in-memory 5-question quiz-session evaluation
- confidence-aware student feedback
- Resource Agent material suggestions for missed topics
- MessageBus request/provide-materials flow

It avoids live LLM calls for the interaction check so the result is stable and
safe to run right before presenting. Run `warmup_check.py` separately to verify
the live Claude call.

Run:
    python scripts/live_flow_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_check import render_report, run_all_checks


def _fail(message: str) -> None:
    raise AssertionError(message)


def _check_5_question_flow() -> list[str]:
    import agents.tutor_agent as tutor_module
    import utils.student_profile as sp
    from agents.resource_agent import ResourceAgent
    from agents.tutor_agent import TutorAgent
    from utils.agent_comm import MessageBus
    from utils.quiz_session import (
        build_quiz_session_report,
        build_quiz_session_summary,
        can_evaluate_session,
    )
    from utils.student_profile import create_profile

    notes: list[str] = []

    original_sp_save = sp.save_profile
    original_tutor_save = tutor_module.save_profile
    sp.save_profile = lambda _name, _profile: True
    tutor_module.save_profile = lambda _name, _profile: True
    try:
        profile = create_profile(
            "Live Flow Check",
            [{"name": "Intro to CS", "difficulty": 3}],
            learning_style="balanced",
        )
        bus = MessageBus()
        tutor = TutorAgent(profile, message_bus=bus, client=None)
        resource = ResourceAgent(bus, client=None)

        choice_explanation = tutor.explain_quiz_choice({
            "topic": "Recursion",
            "difficulty": 2,
            "is_review": False,
        })
        if not choice_explanation.get("reason") or not choice_explanation.get("detail"):
            _fail("adaptive quiz choice explanation was not generated")

        quizzes = [
            {
                "topic": "Loops",
                "course": "Intro to CS",
                "difficulty": 2,
                "question": "What does a while loop do?",
                "options": ["A) Repeats while true", "B) Runs once", "C) Imports code", "D) Defines a class"],
                "correct_answer": "A",
                "explanation": "A while loop repeats while its condition remains true.",
                "answer": "A",
                "confidence": 4,
            },
            {
                "topic": "Recursion",
                "course": "Intro to CS",
                "difficulty": 3,
                "question": "What is a base case?",
                "options": ["A) A stopping condition", "B) A larger recursive call", "C) A global variable", "D) A class"],
                "correct_answer": "A",
                "explanation": "A base case stops recursion so calls can unwind.",
                "answer": "B",
                "confidence": 5,
            },
            {
                "topic": "Functions",
                "course": "Intro to CS",
                "difficulty": 2,
                "question": "What is an argument?",
                "options": ["A) A value passed to a function", "B) A loop", "C) A file", "D) A package"],
                "correct_answer": "A",
                "explanation": "An argument is the value supplied when calling a function.",
                "answer": "A",
                "confidence": 2,
            },
            {
                "topic": "Lists",
                "course": "Intro to CS",
                "difficulty": 2,
                "question": "What does an index identify?",
                "options": ["A) A position in a sequence", "B) A boolean only", "C) A function call", "D) A module"],
                "correct_answer": "A",
                "explanation": "An index identifies a position in a sequence.",
                "answer": "A",
                "confidence": 3,
            },
            {
                "topic": "Dictionaries",
                "course": "Intro to CS",
                "difficulty": 3,
                "question": "How do dictionaries usually retrieve values?",
                "options": ["A) By key", "B) Only by numeric position", "C) By loop count", "D) By file name"],
                "correct_answer": "A",
                "explanation": "Dictionaries retrieve values by key.",
                "answer": "C",
                "confidence": 4,
            },
        ]

        session_results: list[dict] = []
        used_resource_agent = False
        for idx, quiz in enumerate(quizzes, start=1):
            result = tutor.evaluate_answer(quiz, quiz["answer"], confidence=quiz["confidence"])
            if "student_feedback" not in result:
                _fail(f"question {idx} did not return student_feedback")
            if result.get("used_multi_agent"):
                used_resource_agent = True
            session_results.append({
                "quiz": quiz,
                "result": result,
                "confidence": quiz["confidence"],
                "student_answer": quiz["answer"],
            })

            if idx < 5 and can_evaluate_session(session_results):
                _fail("session became evaluable before five questions")

        if not can_evaluate_session(session_results):
            _fail("session was not evaluable after five questions")

        summary = build_quiz_session_summary(session_results)
        if not summary["ready"]:
            _fail("summary is not ready after five questions")
        if summary["answered"] != 5:
            _fail(f"expected 5 answered questions, got {summary['answered']}")
        if summary["correct"] != 3:
            _fail(f"expected 3 correct answers, got {summary['correct']}")
        if "Recursion" not in summary["misconception_topics"]:
            _fail("high-confidence miss on Recursion was not marked as a misconception")
        if not used_resource_agent:
            _fail("wrong answers did not trigger Resource Agent materials")

        report = build_quiz_session_report(session_results)
        if len(report["rows"]) != 5:
            _fail(f"expected 5 report rows, got {len(report['rows'])}")
        if not any(row["signal"] == "High-confidence miss" for row in report["rows"]):
            _fail("session report did not surface the high-confidence miss")

        suggestions = resource.suggest_materials_for_topics(summary["priority_topics"])
        if not suggestions:
            _fail("Resource Agent did not return suggested materials for priority topics")

        stats = bus.get_stats()
        by_type = stats.get("by_type", {})
        if by_type.get("request_materials", 0) < 1 or by_type.get("provide_materials", 0) < 1:
            _fail("MessageBus did not record request/provide material messages")

        notes.append(f"5-question evaluation: {summary['correct']}/{summary['answered']} correct")
        notes.append(f"Adaptive explanation ready: {choice_explanation['reason']}")
        notes.append("Detailed session report produced five question rows")
        notes.append("Student feedback cards returned for all five answers")
        notes.append(f"Resource suggestions available: {len(suggestions)}")
        notes.append(
            "Agent messages: "
            f"request_materials={by_type.get('request_materials', 0)}, "
            f"provide_materials={by_type.get('provide_materials', 0)}"
        )
        return notes
    finally:
        sp.save_profile = original_sp_save
        tutor_module.save_profile = original_tutor_save


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def main() -> int:
    _force_utf8_stdout()
    print("\n─── Live Flow Readiness Check ───")
    preflight = run_all_checks(auto_regenerate=True)
    print(render_report(preflight))
    if not all(result.ok for result in preflight):
        print("❌ Stop here: fix the failed preflight check(s) before presenting.")
        return 1

    try:
        notes = _check_5_question_flow()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Interaction check failed: {exc}")
        return 1

    for note in notes:
        print(f"✅ {note}")
    print("\n✅ Live flow is ready. Run warmup_check.py next if you want to verify live Claude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
