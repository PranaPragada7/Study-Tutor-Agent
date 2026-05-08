"""Helpers for multi-question quiz sessions."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

MIN_QUESTIONS_BEFORE_EVALUATION = 5
MAX_SAVED_QUIZ_SESSIONS = 20


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _coerce_confidence(value: object) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return 3


def _session_item_parts(item: dict) -> tuple[dict, dict, int, str]:
    result = item.get("result", item) if isinstance(item, dict) else {}
    quiz = item.get("quiz", {}) if isinstance(item, dict) else {}
    confidence = _coerce_confidence(
        item.get("confidence", result.get("confidence", 3))
        if isinstance(item, dict)
        else 3
    )
    topic = (
        quiz.get("topic")
        or result.get("student_feedback", {}).get("topic")
        or result.get("topic")
        or "this topic"
    )
    return quiz, result, confidence, str(topic)


def _session_course(session_results: list[dict], fallback: str | None = None) -> str:
    courses: list[str] = []
    for item in session_results:
        quiz = item.get("quiz", {}) if isinstance(item, dict) else {}
        result = item.get("result", item) if isinstance(item, dict) else {}
        course = quiz.get("course") or result.get("course")
        if course:
            courses.append(str(course))
    unique_courses = _unique(courses)
    if len(unique_courses) == 1:
        return unique_courses[0]
    if len(unique_courses) > 1:
        return "Mixed courses"
    return fallback or "Unknown course"


def _quiz_history_ids(session_results: list[dict]) -> list[str]:
    ids: list[str] = []
    for item in session_results:
        quiz = item.get("quiz", {}) if isinstance(item, dict) else {}
        result = item.get("result", item) if isinstance(item, dict) else {}
        quiz_id = result.get("quiz_id") or quiz.get("quiz_id") or quiz.get("id")
        if quiz_id:
            ids.append(str(quiz_id))
    return ids


def _stable_session_id(prefix: str, ids: list[str]) -> str:
    digest = hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _parse_history_time(value: object):
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def can_evaluate_session(
    session_results: list[dict],
    min_questions: int = MIN_QUESTIONS_BEFORE_EVALUATION,
) -> bool:
    """Return True only after the session has enough answered questions."""
    return len(session_results) >= min_questions


def build_quiz_session_summary(
    session_results: list[dict],
    min_questions: int = MIN_QUESTIONS_BEFORE_EVALUATION,
) -> dict:
    """Build an overall student evaluation for a quiz session.

    Each item may be either a raw evaluate_answer result or a wrapper shaped
    like {"quiz": quiz, "result": result, "confidence": int}. The Streamlit UI
    uses the wrapper so the summary can include topic and confidence signals.
    """
    answered = len(session_results)
    remaining = max(0, min_questions - answered)
    correct = 0
    incorrect_topics: list[str] = []
    misconception_topics: list[str] = []
    low_confidence_correct: list[str] = []

    for item in session_results:
        _quiz, result, confidence, topic = _session_item_parts(item)

        if result.get("correct"):
            correct += 1
            if confidence <= 2:
                low_confidence_correct.append(str(topic))
        else:
            incorrect_topics.append(str(topic))
            if confidence >= 4:
                misconception_topics.append(str(topic))

    accuracy = round((correct / answered) * 100, 1) if answered else 0.0
    ready = answered >= min_questions

    if not ready:
        return {
            "ready": False,
            "answered": answered,
            "required": min_questions,
            "remaining": remaining,
            "correct": correct,
            "accuracy": accuracy,
            "headline": "Keep going",
            "summary": (
                f"Answer {remaining} more question(s) before the tutor gives an "
                "overall quiz evaluation."
            ),
            "next_step": "Continue the quiz session.",
            "priority_topics": [],
            "misconception_topics": [],
            "low_confidence_correct": low_confidence_correct,
        }

    unique_incorrect = list(dict.fromkeys(incorrect_topics))
    unique_misconceptions = list(dict.fromkeys(misconception_topics))

    if accuracy >= 80:
        headline = "Strong first quiz"
        next_step = "Move up one difficulty level or explain the hardest answer from memory."
    elif accuracy >= 60:
        headline = "Good baseline"
        next_step = "Review the missed topics, then take another 5-question session."
    else:
        headline = "Needs focused review"
        next_step = "Review the priority topics first, then retry with easier questions."

    if unique_misconceptions:
        next_step = (
            "Start with the high-confidence misses: "
            f"{', '.join(unique_misconceptions[:3])}. "
            "Those are likely misconceptions."
        )
    elif unique_incorrect:
        next_step = (
            f"Review: {', '.join(unique_incorrect[:3])}. Then take another "
            "short session to check improvement."
        )

    return {
        "ready": True,
        "answered": answered,
        "required": min_questions,
        "remaining": 0,
        "correct": correct,
        "accuracy": accuracy,
        "headline": headline,
        "summary": f"You answered {correct} out of {answered} questions correctly ({accuracy}%).",
        "next_step": next_step,
        "priority_topics": unique_incorrect[:5],
        "misconception_topics": unique_misconceptions[:5],
        "low_confidence_correct": list(dict.fromkeys(low_confidence_correct))[:5],
    }


def build_quiz_session_report(
    session_results: list[dict],
    min_questions: int = MIN_QUESTIONS_BEFORE_EVALUATION,
) -> dict:
    """Build a detailed, display-ready report for a quiz session."""
    summary = build_quiz_session_summary(session_results, min_questions=min_questions)
    rows: list[dict] = []

    for index, item in enumerate(session_results, start=1):
        quiz, result, confidence, topic = _session_item_parts(item)
        correct = bool(result.get("correct"))
        difficulty = quiz.get("difficulty", result.get("difficulty", "?"))
        student_answer = (
            item.get("student_answer")
            if isinstance(item, dict)
            else None
        ) or quiz.get("answer") or result.get("student_answer") or "not shown"
        correct_answer = result.get("correct_answer") or quiz.get("correct_answer") or "?"
        question = quiz.get("question") or result.get("question") or ""
        quiz_id = result.get("quiz_id") or quiz.get("quiz_id") or quiz.get("id") or ""
        feedback = result.get("student_feedback", {})
        if not isinstance(feedback, dict):
            feedback = {}

        if correct and confidence >= 4:
            signal = "Strong mastery signal"
            next_action = "Increase difficulty or explain the answer from memory."
        elif correct and confidence <= 2:
            signal = "Correct but low confidence"
            next_action = "Practice one similar question to build fluency."
        elif (not correct) and confidence >= 4:
            signal = "High-confidence miss"
            next_action = "Review this first; it is likely a misconception."
        elif not correct and confidence <= 2:
            signal = "Known gap"
            next_action = "Revisit the basic definition, then retry slowly."
        elif correct:
            signal = "Solid progress"
            next_action = "Keep this topic in normal review rotation."
        else:
            signal = "Needs review"
            next_action = "Read the explanation and retry at the same difficulty."

        explanation = str(
            result.get("explanation")
            or quiz.get("explanation")
            or ""
        ).strip()

        rows.append({
            "number": index,
            "topic": topic,
            "difficulty": difficulty,
            "correct": correct,
            "result_label": "Correct" if correct else "Review",
            "confidence": confidence,
            "student_answer": str(student_answer),
            "correct_answer": str(correct_answer),
            "question": str(question),
            "quiz_id": str(quiz_id),
            "signal": signal,
            "next_action": next_action,
            "explanation": explanation,
            "feedback_title": str(feedback.get("title", "")).strip(),
            "feedback_summary": str(feedback.get("summary", "")).strip(),
            "confidence_insight": str(feedback.get("confidence_insight", "")).strip(),
            "resource_note": str(feedback.get("resource_note", "")).strip(),
            "review_note": str(feedback.get("review_note", "")).strip(),
        })

    return {
        "ready": summary["ready"],
        "summary": summary,
        "rows": rows,
        "missed_topics": summary.get("priority_topics", []),
        "misconception_topics": summary.get("misconception_topics", []),
        "low_confidence_correct": summary.get("low_confidence_correct", []),
    }


def build_quiz_session_feedback(
    session_results: list[dict],
    min_questions: int = MIN_QUESTIONS_BEFORE_EVALUATION,
) -> dict:
    """Build a concise tutor-style feedback summary for the session.

    Unlike ``build_quiz_session_report``, this is written as student-facing
    synthesis: what the tutor learned from all answers so far, what looks
    strong, and what should be reviewed next.
    """
    summary = build_quiz_session_summary(session_results, min_questions=min_questions)
    report = build_quiz_session_report(session_results, min_questions=min_questions)
    rows = report["rows"]

    strong_topics = _unique(
        [row["topic"] for row in rows if row["correct"] and row["confidence"] >= 3]
    )[:5]
    low_confidence_correct = report.get("low_confidence_correct", [])
    missed_topics = report.get("missed_topics", [])
    misconception_topics = report.get("misconception_topics", [])

    if not rows:
        tutor_read = "No answers have been submitted yet."
    elif summary["ready"]:
        tutor_read = (
            f"Across all {summary['answered']} answers, you got "
            f"{summary['correct']} correct for {summary['accuracy']}% accuracy."
        )
    else:
        tutor_read = (
            f"So far, you have answered {summary['correct']} of "
            f"{summary['answered']} question(s) correctly. The overall "
            f"evaluation unlocks after {summary['remaining']} more question(s)."
        )

    if misconception_topics:
        priority_feedback = (
            "Highest priority: review high-confidence misses in "
            f"{', '.join(misconception_topics[:3])}. These usually mean the "
            "tutor found a misconception, not just a guess."
        )
    elif missed_topics:
        priority_feedback = (
            "Review next: "
            f"{', '.join(missed_topics[:3])}. These are the topics the tutor "
            "will use for resource recommendations."
        )
    elif low_confidence_correct:
        priority_feedback = (
            "You are getting answers right but still building confidence in "
            f"{', '.join(low_confidence_correct[:3])}. Practice one similar "
            "question for fluency."
        )
    else:
        priority_feedback = "No urgent review topic yet. Keep the session going."

    if strong_topics:
        strengths = (
            "Strong signals: " + ", ".join(strong_topics[:3]) + "."
        )
    else:
        strengths = "Strong signals will appear here as correct answers accumulate."

    confidence_notes: list[str] = []
    if misconception_topics:
        confidence_notes.append(
            "High confidence on a wrong answer makes that topic a priority."
        )
    if low_confidence_correct:
        confidence_notes.append(
            "Low confidence on a correct answer means the concept is close but not fluent."
        )
    confidence_feedback = (
        " ".join(confidence_notes)
        if confidence_notes else
        "Your confidence ratings are being used to tune difficulty and review timing."
    )

    return {
        "ready": summary["ready"],
        "answered": summary["answered"],
        "required": summary["required"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "headline": summary["headline"],
        "tutor_read": tutor_read,
        "strengths": strengths,
        "priority_feedback": priority_feedback,
        "confidence_feedback": confidence_feedback,
        "next_step": summary["next_step"],
        "resource_reason": (
            "Resource materials are shown for missed priority topics because "
            "the session assessment found wrong answers on those topics and "
            "matched them to generated or cached review material."
        ),
    }


def build_persisted_quiz_session(
    session_results: list[dict],
    *,
    course: str | None = None,
    now_fn=None,
    source: str = "completed_session",
    resource_materials: list[dict] | None = None,
) -> dict:
    """Build a durable report snapshot for a completed quiz session."""
    summary = build_quiz_session_summary(session_results)
    report = build_quiz_session_report(session_results)
    feedback = build_quiz_session_feedback(session_results)
    quiz_ids = _quiz_history_ids(session_results)
    created_at = (now_fn or (lambda: datetime.now(timezone.utc)))().isoformat()
    session_id = (
        _stable_session_id("session", quiz_ids)
        if quiz_ids
        else f"session-{uuid.uuid4().hex}"
    )

    return {
        "id": session_id,
        "created_at": created_at,
        "source": source,
        "course": course or _session_course(session_results),
        "question_count": summary["answered"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "headline": summary["headline"],
        "summary": summary["summary"],
        "tutor_read": feedback["tutor_read"],
        "strengths": feedback["strengths"],
        "priority_feedback": feedback["priority_feedback"],
        "confidence_feedback": feedback["confidence_feedback"],
        "next_step": summary["next_step"],
        "priority_topics": summary["priority_topics"],
        "misconception_topics": summary["misconception_topics"],
        "low_confidence_correct": summary["low_confidence_correct"],
        "quiz_history_ids": quiz_ids,
        "resource_materials": resource_materials or [],
        "rows": report["rows"],
    }


def store_quiz_session_report(
    profile: dict,
    session_report: dict,
    *,
    limit: int = MAX_SAVED_QUIZ_SESSIONS,
) -> bool:
    """Append a session report to the profile, deduping by report id."""
    if not session_report or not session_report.get("id"):
        return False

    sessions = profile.setdefault("quiz_sessions", [])
    existing_ids = {s.get("id") for s in sessions if isinstance(s, dict)}
    if session_report["id"] in existing_ids:
        return False

    sessions.append(session_report)
    sessions.sort(key=lambda s: str(s.get("created_at", "")))
    if len(sessions) > limit:
        del sessions[:-limit]
    profile["session_count"] = len(sessions)
    return True


def build_quiz_session_from_history(
    profile: dict,
    *,
    min_questions: int = MIN_QUESTIONS_BEFORE_EVALUATION,
) -> dict | None:
    """Reconstruct a latest session summary from saved quiz_history rows.

    Older profiles saved each answer but did not save the composed session
    report. This gives the UI a reload-safe evaluation from the last five
    saved answers while being honest about fields that old history lacks.
    """
    history = profile.get("quiz_history", []) if isinstance(profile, dict) else []
    if len(history) < min_questions:
        return None

    window = history[-min_questions:]
    session_results: list[dict] = []
    for entry in window:
        if not isinstance(entry, dict):
            continue
        confidence = _coerce_confidence(entry.get("confidence", 3))
        correct_answer = entry.get("correct_answer") or "Not saved in older history"
        session_results.append({
            "quiz": {
                "id": entry.get("id", ""),
                "course": entry.get("course", ""),
                "topic": entry.get("topic", "this topic"),
                "difficulty": entry.get("difficulty", "?"),
                "question": entry.get("question", ""),
                "correct_answer": correct_answer,
                "explanation": entry.get("explanation", ""),
            },
            "result": {
                "correct": bool(entry.get("correct", False)),
                "correct_answer": correct_answer,
                "confidence": confidence,
                "explanation": entry.get("explanation", ""),
                "quiz_id": entry.get("id", ""),
            },
            "confidence": confidence,
            "student_answer": entry.get("student_answer", "not shown"),
        })

    if len(session_results) < min_questions:
        return None

    last_timestamp = str(window[-1].get("timestamp", ""))
    report = build_persisted_quiz_session(
        session_results,
        course=_session_course(session_results),
        now_fn=lambda: _parse_history_time(last_timestamp),
        source="history_reconstruction",
    )
    quiz_ids = report.get("quiz_history_ids", [])
    if quiz_ids:
        report["id"] = _stable_session_id("history", quiz_ids)
    report["note"] = (
        "Restored from saved quiz answers. Older answer history did not save "
        "the generated explanation text or exact correct-answer letter."
    )
    return report


__all__ = [
    "MIN_QUESTIONS_BEFORE_EVALUATION",
    "MAX_SAVED_QUIZ_SESSIONS",
    "can_evaluate_session",
    "build_quiz_session_summary",
    "build_quiz_session_report",
    "build_quiz_session_feedback",
    "build_persisted_quiz_session",
    "store_quiz_session_report",
    "build_quiz_session_from_history",
]
