"""Tests for multi-question quiz session evaluation."""

from utils.quiz_session import (
    MIN_QUESTIONS_BEFORE_EVALUATION,
    build_persisted_quiz_session,
    build_quiz_session_feedback,
    build_quiz_session_from_history,
    build_quiz_session_report,
    build_quiz_session_summary,
    can_evaluate_session,
    store_quiz_session_report,
)


def _item(topic: str, correct: bool, confidence: int) -> dict:
    return {
        "quiz": {"topic": topic},
        "result": {
            "correct": correct,
            "student_feedback": {"topic": topic},
            "confidence": confidence,
        },
        "confidence": confidence,
    }


def _rich_item(topic: str, correct: bool, confidence: int, quiz_id: str) -> dict:
    feedback_title = "Strong answer" if correct else "Review needed"
    return {
        "quiz": {
            "course": "CS101",
            "topic": topic,
            "difficulty": 2,
            "question": f"What about {topic}?",
            "correct_answer": "A",
        },
        "result": {
            "correct": correct,
            "correct_answer": "A",
            "explanation": f"Explanation for {topic}",
            "student_feedback": {
                "topic": topic,
                "title": feedback_title,
                "summary": f"Feedback summary for {topic}",
                "confidence_insight": f"Confidence insight for {topic}",
                "resource_note": f"Resource note for {topic}",
                "review_note": f"Review note for {topic}",
            },
            "confidence": confidence,
            "quiz_id": quiz_id,
        },
        "confidence": confidence,
        "student_answer": "A",
    }


def test_quiz_session_waits_for_five_answers_before_evaluation():
    results = [
        _item("Loops", True, 4),
        _item("Functions", False, 2),
        _item("Recursion", True, 1),
        _item("Lists", True, 3),
    ]

    assert can_evaluate_session(results) is False
    summary = build_quiz_session_summary(results)
    assert summary["ready"] is False
    assert summary["remaining"] == 1
    assert summary["required"] == MIN_QUESTIONS_BEFORE_EVALUATION
    assert "Answer 1 more" in summary["summary"]


def test_quiz_session_evaluates_after_five_answers():
    results = [
        _item("Loops", True, 4),
        _item("Functions", False, 2),
        _item("Recursion", False, 5),
        _item("Lists", True, 3),
        _item("Dictionaries", True, 1),
    ]

    assert can_evaluate_session(results) is True
    summary = build_quiz_session_summary(results)
    assert summary["ready"] is True
    assert summary["answered"] == 5
    assert summary["correct"] == 3
    assert summary["accuracy"] == 60.0
    assert "Functions" in summary["priority_topics"]
    assert "Recursion" in summary["misconception_topics"]


def test_quiz_session_report_includes_question_level_signals():
    results = [
        _item("Loops", True, 4),
        _item("Functions", False, 2),
        _item("Recursion", False, 5),
        _item("Lists", True, 3),
        _item("Dictionaries", True, 1),
    ]

    report = build_quiz_session_report(results)

    assert report["ready"] is True
    assert len(report["rows"]) == 5
    assert report["rows"][0]["signal"] == "Strong mastery signal"
    assert report["rows"][1]["signal"] == "Known gap"
    assert report["rows"][2]["signal"] == "High-confidence miss"
    assert report["rows"][4]["signal"] == "Correct but low confidence"
    assert "Recursion" in report["misconception_topics"]


def test_quiz_session_report_keeps_full_explanation_text():
    long_explanation = (
        "This explanation is intentionally long so the report keeps the full "
        "teaching detail instead of cutting it off in the sample profile. "
        "Students and presenters need the complete rationale when reviewing "
        "the five-question session report, especially for misconception "
        "signals and recovery steps."
    )
    results = [
        {
            "quiz": {"topic": "Supervised Learning", "difficulty": 2},
            "result": {
                "correct": True,
                "correct_answer": "A",
                "explanation": long_explanation,
            },
            "confidence": 3,
            "student_answer": "A",
        }
    ]

    report = build_quiz_session_report(results, min_questions=1)

    assert report["rows"][0]["explanation"] == long_explanation
    assert not report["rows"][0]["explanation"].endswith("...")


def test_quiz_session_feedback_summarizes_all_answers():
    results = [
        _item("Loops", True, 4),
        _item("Functions", False, 2),
        _item("Recursion", False, 5),
        _item("Lists", True, 3),
        _item("Dictionaries", True, 1),
    ]

    feedback = build_quiz_session_feedback(results)

    assert feedback["ready"] is True
    assert "Across all 5 answers" in feedback["tutor_read"]
    assert "Loops" in feedback["strengths"]
    assert "Recursion" in feedback["priority_feedback"]
    assert "High confidence" in feedback["confidence_feedback"]
    assert "Resource materials" in feedback["resource_reason"]


def test_persisted_quiz_session_snapshot_can_be_stored_once():
    results = [
        _rich_item("Loops", True, 4, "q1"),
        _rich_item("Functions", False, 2, "q2"),
        _rich_item("Recursion", False, 5, "q3"),
        _rich_item("Lists", True, 3, "q4"),
        _rich_item("Dictionaries", True, 1, "q5"),
    ]
    profile = {"quiz_sessions": [], "session_count": 0}

    resource_materials = [
        {
            "topic": "Functions",
            "title": "Review Functions",
            "explanation": "Function review material",
            "rationale": {
                "why_shown": "This material is shown because Functions was missed.",
                "quiz_evidence": "The student answered a Functions question wrong.",
                "generation_source": "Generated after evaluating the session.",
            },
        }
    ]
    snapshot = build_persisted_quiz_session(
        results,
        course="CS101",
        resource_materials=resource_materials,
    )
    assert snapshot["question_count"] == 5
    assert snapshot["correct"] == 3
    assert snapshot["accuracy"] == 60.0
    assert snapshot["quiz_history_ids"] == ["q1", "q2", "q3", "q4", "q5"]
    assert snapshot["rows"][0]["question"] == "What about Loops?"
    assert snapshot["rows"][0]["feedback_title"] == "Strong answer"
    assert snapshot["rows"][1]["feedback_summary"] == "Feedback summary for Functions"
    assert snapshot["resource_materials"] == resource_materials

    assert store_quiz_session_report(profile, snapshot) is True
    assert store_quiz_session_report(profile, snapshot) is False
    assert len(profile["quiz_sessions"]) == 1
    assert profile["session_count"] == 1


def test_history_reconstruction_restores_latest_five_answer_evaluation():
    profile = {
        "quiz_history": [
            {
                "id": f"old-{i}",
                "timestamp": f"2026-05-07T12:0{i}:00+00:00",
                "course": "CS101",
                "topic": f"Topic {i}",
                "difficulty": 2,
                "correct": i in {0, 2, 4},
                "question": f"Question {i}",
                "student_answer": "A",
                "confidence": 5 if i == 3 else 3,
            }
            for i in range(6)
        ],
    }

    restored = build_quiz_session_from_history(profile)

    assert restored is not None
    assert restored["source"] == "history_reconstruction"
    assert restored["question_count"] == 5
    assert restored["correct"] == 2
    assert restored["accuracy"] == 40.0
    assert restored["rows"][0]["question"] == "Question 1"
    assert restored["rows"][0]["correct_answer"] == "Not saved in older history"
    assert "Older answer history" in restored["note"]
