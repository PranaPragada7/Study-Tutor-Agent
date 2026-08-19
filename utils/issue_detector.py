"""
Issue Detector -- self-monitors the tutor's behaviour and surfaces
actionable issues so the agent can self-correct.

Detects:
  1. Accuracy drops (student struggling for 5+ questions in a row)
  2. Stalled learning (no mastery improvement over many questions)
  3. Difficulty mismatch (too easy or too hard for too long)
  4. API/parsing failures (LLM returned bad JSON, API timeout, etc.)
  5. Topic imbalance (some topics never quizzed)

Each issue generates an alert with a suggested fix that the tutor agent
can act on autonomously.

Alert cooldown prevents the same issue from logging twice within a
10-question window or 5-minute timeframe. Stalled-learning and
topic-imbalance detection both operate over a sliding window of the
last 20 events.
"""

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from typing import Any

from filelock import FileLock, Timeout

from config import (
    COOLDOWN_QUESTION_WINDOW as _COOLDOWN_QUESTION_WINDOW,
)
from config import (
    COOLDOWN_TIME_SECONDS as _COOLDOWN_TIME_SECONDS,
)
from config import (
    LOCK_TIMEOUT_SECONDS as _LOCK_TIMEOUT_SECONDS,
)
from config import (
    MAX_API_ERRORS as _MAX_API_ERRORS,
)
from config import (
    MAX_ISSUES as _MAX_ISSUES,
)
from config import (
    STALLED_WINDOW_SIZE as _STALLED_WINDOW_SIZE,
)

logger = logging.getLogger(__name__)

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")


class Issue:
    """Represents a detected issue."""

    def __init__(
        self,
        issue_type: str,
        severity: str,
        message: str,
        suggested_fix: str,
        details: dict | None = None,
    ):
        self.issue_type = (
            issue_type  # accuracy_drop, stalled, difficulty_mismatch, api_error, topic_imbalance
        )
        self.severity = severity  # low, medium, high, critical
        self.message = message
        self.suggested_fix = suggested_fix
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class IssueDetector:
    """
    Monitors the tutor agent and student performance to detect and log issues.
    """

    def __init__(self, streak_data: list[bool] | None = None):
        """
        Args:
            streak_data: Optional persisted streak tracker (LA-7).
                         If provided, the detector resumes from the
                         saved state instead of starting empty.
        """
        self.issues: list[Issue] = []
        self.api_errors: list[dict] = []
        # LA-7: Accept persisted streak data so the tracker survives restarts.
        self._streak_tracker: list[bool] = list(streak_data[-20:]) if streak_data else []
        # Alert cooldown — tracks when each issue type last fired.
        # {issue_type: {"question_number": int, "timestamp": datetime}}
        self._last_triggered: dict[str, dict] = {}
        self._total_questions: int = 0

    # ------------------------------------------------------------------
    # LA-7 public accessor
    # ------------------------------------------------------------------

    def get_streak_data(self) -> list[bool]:
        """Return the current streak tracker for persistence."""
        return list(self._streak_tracker)

    # ------------------------------------------------------------------
    # Alert cooldown logic
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, issue_type: str) -> bool:
        """Check if an issue type is still within the cooldown window.

        Returns True if the same issue was triggered fewer than 10
        questions ago OR fewer than 5 minutes ago.
        """
        if issue_type not in self._last_triggered:
            return False
        last = self._last_triggered[issue_type]
        # Question-count cooldown
        if self._total_questions - last["question_number"] < _COOLDOWN_QUESTION_WINDOW:
            return True
        # Time-based cooldown
        elapsed = datetime.now() - last["timestamp"]
        if elapsed < timedelta(seconds=_COOLDOWN_TIME_SECONDS):
            return True
        return False

    def _mark_triggered(self, issue_type: str) -> None:
        """Record that an issue type was just triggered."""
        self._last_triggered[issue_type] = {
            "question_number": self._total_questions,
            "timestamp": datetime.now(),
        }

    # ------------------------------------------------------------------
    # Core analysis entry point
    # ------------------------------------------------------------------

    def analyze_quiz_result(
        self, profile: dict, course: str, topic: str, difficulty: int, correct: bool
    ) -> list[Issue]:
        """
        Run all detection checks after a quiz answer.
        Returns list of newly detected issues.
        """
        self._total_questions += 1
        new_issues: list[Issue] = []

        # Track streak
        self._streak_tracker.append(correct)
        if len(self._streak_tracker) > 20:
            self._streak_tracker = self._streak_tracker[-20:]

        # Check 1: Accuracy drop (5+ wrong in a row)
        issue = self._check_accuracy_drop()
        if issue:
            new_issues.append(issue)

        # Check 2: Difficulty mismatch
        issue = self._check_difficulty_mismatch(profile, course, topic, difficulty, correct)
        if issue:
            new_issues.append(issue)

        # Check 3: Stalled learning (sliding window)
        issue = self._check_stalled_learning(profile, course)
        if issue:
            new_issues.append(issue)

        # Check 4: Topic imbalance (sliding window)
        issue = self._check_topic_imbalance(profile, course)
        if issue:
            new_issues.append(issue)

        self.issues.extend(new_issues)
        self._trim_lists()
        return new_issues

    def _trim_lists(self) -> None:
        """Cap the in-memory issue/error lists so a long session can't OOM."""
        if len(self.issues) > _MAX_ISSUES:
            self.issues = self.issues[-_MAX_ISSUES:]
        if len(self.api_errors) > _MAX_API_ERRORS:
            self.api_errors = self.api_errors[-_MAX_API_ERRORS:]

    def log_api_error(self, error_type: str, details: str, recovered: bool = False) -> Issue:
        """
        Log an API or parsing error.

        Args:
            error_type: "json_parse", "api_timeout", "api_rate_limit", "unexpected"
            details: Description of what went wrong
            recovered: Whether the system recovered (used fallback)
        """
        error_entry = {
            "type": error_type,
            "details": details,
            "recovered": recovered,
            "timestamp": datetime.now().isoformat(),
        }
        self.api_errors.append(error_entry)

        severity = "medium" if recovered else "high"
        issue = Issue(
            issue_type="api_error",
            severity=severity,
            message=f"API error: {error_type} -- {details}",
            suggested_fix="Retry with exponential backoff"
            if not recovered
            else "Recovered with fallback",
            details=error_entry,
        )
        self.issues.append(issue)
        self._trim_lists()
        return issue

    # ------------------------------------------------------------------
    # Detection checks
    # ------------------------------------------------------------------

    def _check_accuracy_drop(self) -> Issue | None:
        """Detect if student has gotten 3+ or 5+ wrong in a row."""
        # Check for 5 wrong in a row (high severity) first
        if len(self._streak_tracker) >= 5:
            last_5 = self._streak_tracker[-5:]
            if not any(last_5):  # All False = 5 wrong in a row
                # Cooldown check
                if self._is_on_cooldown("accuracy_drop_5"):
                    return None
                self._mark_triggered("accuracy_drop_5")
                return Issue(
                    issue_type="accuracy_drop",
                    severity="high",
                    message="Student has answered 5 questions wrong in a row.",
                    suggested_fix="Reduce difficulty by 2 levels and switch to a different topic. "
                    "Offer encouragement and ask if the student wants to review fundamentals.",
                    details={"wrong_streak": 5},
                )

        # Check for 3 wrong in a row (medium severity)
        if len(self._streak_tracker) >= 3:
            last_3 = self._streak_tracker[-3:]
            if not any(last_3):
                # Cooldown check
                if self._is_on_cooldown("accuracy_drop_3"):
                    return None
                self._mark_triggered("accuracy_drop_3")
                return Issue(
                    issue_type="accuracy_drop",
                    severity="medium",
                    message="Student has answered 3 questions wrong in a row.",
                    suggested_fix="Reduce difficulty by 1 level. Consider offering a hint before the next question.",
                    details={"wrong_streak": 3},
                )

        return None

    def _check_difficulty_mismatch(
        self, profile: dict, course: str, topic: str, difficulty: int, correct: bool
    ) -> Issue | None:
        """Detect if difficulty is too easy or too hard for the student."""
        if course not in profile["courses"]:
            return None

        topics = profile["courses"][course].get("topics", {})
        if topic not in topics:
            return None

        stats = topics[topic]
        if stats["attempted"] < 5:
            return None  # Not enough data

        accuracy = stats["correct"] / stats["attempted"]

        # Too easy: >90% accuracy at current difficulty
        if accuracy > 0.9 and difficulty <= 2:
            # Cooldown check
            if self._is_on_cooldown("difficulty_too_easy"):
                return None
            self._mark_triggered("difficulty_too_easy")
            return Issue(
                issue_type="difficulty_mismatch",
                severity="low",
                message=f"Topic '{topic}' may be too easy ({accuracy * 100:.0f}% accuracy at difficulty {difficulty}).",
                suggested_fix="Increase difficulty to challenge the student and promote deeper learning.",
                details={"accuracy": accuracy, "difficulty": difficulty, "topic": topic},
            )

        # Too hard: <30% accuracy at current difficulty
        if accuracy < 0.3 and difficulty >= 4:
            # Cooldown check
            if self._is_on_cooldown("difficulty_too_hard"):
                return None
            self._mark_triggered("difficulty_too_hard")
            return Issue(
                issue_type="difficulty_mismatch",
                severity="medium",
                message=f"Topic '{topic}' may be too hard ({accuracy * 100:.0f}% accuracy at difficulty {difficulty}).",
                suggested_fix="Decrease difficulty and provide more foundational questions first.",
                details={"accuracy": accuracy, "difficulty": difficulty, "topic": topic},
            )

        return None

    def _check_stalled_learning(self, profile: dict, course: str) -> Issue | None:
        """
        Detect if accuracy hasn't improved over recent questions.

        Uses a sliding window of the last 20 course-specific questions,
        split 10/10, rather than splitting the entire history in half
        (which diluted the signal as history grew).
        """
        history = profile.get("quiz_history", [])
        course_history = [h for h in history if h["course"] == course]

        if len(course_history) < _STALLED_WINDOW_SIZE:
            return None

        # Sliding window: last 20 questions, split 10 / 10
        half = _STALLED_WINDOW_SIZE // 2
        window = course_history[-_STALLED_WINDOW_SIZE:]
        first_half = window[:half]
        second_half = window[half:]

        first_acc = sum(1 for h in first_half if h["correct"]) / len(first_half)
        second_acc = sum(1 for h in second_half if h["correct"]) / len(second_half)

        # If accuracy dropped or stayed flat over the window
        if second_acc <= first_acc:
            # Cooldown check
            if self._is_on_cooldown("stalled_learning"):
                return None
            self._mark_triggered("stalled_learning")
            return Issue(
                issue_type="stalled_learning",
                severity="medium",
                message=f"Learning appears stalled for {course}. "
                f"Last {half} Qs: {second_acc * 100:.0f}%, "
                f"previous {half} Qs: {first_acc * 100:.0f}%.",
                suggested_fix="Change study strategy: try different question formats, "
                "suggest the student explain concepts in their own words, "
                "or focus on prerequisite topics.",
                details={
                    "first_half_accuracy": round(first_acc, 3),
                    "second_half_accuracy": round(second_acc, 3),
                    "window_size": _STALLED_WINDOW_SIZE,
                },
            )

        return None

    def check_feedback_pattern(self, profile: dict) -> Issue | None:
        """Surface a meta-issue when the student's RECENT quiz history
        carries a cluster of negative-feedback flags.

        Triggers:
          - 3+ ``wrong_answer`` or ``unclear_question`` flags in the
            last 20 entries -> high severity (the LLM is producing
            structurally-valid but factually-wrong quizzes; the
            verifier should be catching this and isn't, OR the
            verifier is disabled).
          - 3+ ``too_easy`` or ``too_hard`` in the same window -> low
            severity (difficulty mismatch the bandit hasn't yet
            corrected).

        Cooldown-gated identically to the other detectors so a sustained
        cluster doesn't keep re-firing.
        """
        history = profile.get("quiz_history", [])
        recent = [
            e for e in history[-_STALLED_WINDOW_SIZE:] if isinstance(e.get("feedback_flags"), list)
        ]

        truthiness_flags = {"wrong_answer", "unclear_question"}
        difficulty_flags = {"too_easy", "too_hard"}

        truthiness_count = sum(
            1 for e in recent if any(f in truthiness_flags for f in e.get("feedback_flags", []))
        )
        difficulty_count = sum(
            1 for e in recent if any(f in difficulty_flags for f in e.get("feedback_flags", []))
        )

        if truthiness_count >= 3:
            if self._is_on_cooldown("feedback_truthiness"):
                return None
            self._mark_triggered("feedback_truthiness")
            issue = Issue(
                issue_type="feedback_pattern",
                severity="high",
                message=(
                    f"Student flagged {truthiness_count} recent quizzes as "
                    f"having a wrong answer key or unclear question."
                ),
                suggested_fix=(
                    "The LLM-generated answer keys may be unreliable. "
                    "Verify STUDY_TUTOR_QUIZ_VERIFIER_ENABLED is on; "
                    "consider lowering temperature or switching topics until "
                    "the model regains accuracy."
                ),
                details={"flagged_recent": truthiness_count, "window": _STALLED_WINDOW_SIZE},
            )
            self.issues.append(issue)
            self._trim_lists()
            return issue

        if difficulty_count >= 3:
            if self._is_on_cooldown("feedback_difficulty"):
                return None
            self._mark_triggered("feedback_difficulty")
            issue = Issue(
                issue_type="feedback_pattern",
                severity="low",
                message=(
                    f"Student flagged {difficulty_count} recent quizzes as too easy or too hard."
                ),
                suggested_fix=(
                    "The bandit's recommended difficulty may be lagging the "
                    "student's actual ability. The next few quizzes will "
                    "automatically widen the explore window."
                ),
                details={"flagged_recent": difficulty_count, "window": _STALLED_WINDOW_SIZE},
            )
            self.issues.append(issue)
            self._trim_lists()
            return issue

        return None

    def _check_topic_imbalance(self, profile: dict, course: str) -> Issue | None:
        """Detect if some topics are being ignored.

        Uses a sliding window of the last 20 quiz events for this
        course, so a student's coverage 200 questions ago doesn't affect
        the current imbalance check.
        """
        if course not in profile["courses"]:
            return None

        topics = profile["courses"][course].get("topics", {})
        if len(topics) < 3:
            return None

        # Count topic coverage in the last 20 events only
        history = profile.get("quiz_history", [])
        course_history = [h for h in history if h["course"] == course]
        window = course_history[-_STALLED_WINDOW_SIZE:]

        if len(window) < _STALLED_WINDOW_SIZE:
            return None  # Not enough recent data

        recent_attempts: dict[str, int] = {}
        for entry in window:
            t = entry["topic"]
            recent_attempts[t] = recent_attempts.get(t, 0) + 1

        # Check all known topics — those absent from the window get count 0
        for t in topics:
            if t not in recent_attempts:
                recent_attempts[t] = 0

        max_attempts = max(recent_attempts.values())
        min_attempts = min(recent_attempts.values())

        # If one topic has 5x more questions than another (or a topic has 0)
        if max_attempts > 0 and (min_attempts == 0 or max_attempts / min_attempts >= 5):
            # Cooldown check
            if self._is_on_cooldown("topic_imbalance"):
                return None
            self._mark_triggered("topic_imbalance")

            neglected = [t for t, a in recent_attempts.items() if a == min_attempts]
            return Issue(
                issue_type="topic_imbalance",
                severity="low",
                message=f"Topics are unevenly covered in the last {_STALLED_WINDOW_SIZE} questions. "
                f"'{neglected[0]}' has only {min_attempts} questions vs "
                f"{max_attempts} for the most-quizzed topic.",
                suggested_fix=f"Schedule more questions on: {', '.join(neglected)}",
                details={"topic_attempts": recent_attempts},
            )

        return None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_active_issues(self, severity_min: str = "low") -> list[dict]:
        """Get all issues at or above a severity level."""
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_order.get(severity_min, 0)

        return [i.to_dict() for i in self.issues if severity_order.get(i.severity, 0) >= min_level]

    def get_recent_issues(self, n: int = 5) -> list[dict]:
        """Get the N most recent issues."""
        return [i.to_dict() for i in self.issues[-n:]]

    def get_issue_summary(self) -> dict:
        """Get a summary of all detected issues by type."""
        summary: dict[str, Any] = {
            "total_issues": len(self.issues),
            "api_errors": len(self.api_errors),
            "by_type": {},
            "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
        }

        for issue in self.issues:
            summary["by_type"][issue.issue_type] = summary["by_type"].get(issue.issue_type, 0) + 1
            summary["by_severity"][issue.severity] += 1

        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self, student_name: str) -> None:
        """
        Atomically save the issue log to disk.

        Uses the same locked tempfile + os.replace pattern as
        ``utils.student_profile.save_profile`` so two concurrent sessions
        can't corrupt each other's log file or leave a half-written one
        behind on a crash.
        """
        os.makedirs(LOG_DIR, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9 _-]", "", student_name).strip().lower().replace(" ", "_")
        if not safe_name:
            return  # nothing sensible to write
        log_path = os.path.join(LOG_DIR, f"{safe_name}_issues.json")

        # CB-3: Verify the resolved path stays inside LOG_DIR using
        # commonpath to prevent path-traversal attacks.
        resolved = os.path.realpath(log_path)
        base = os.path.realpath(LOG_DIR)
        if os.path.commonpath([resolved, base]) != base:
            logger.error(
                "Path traversal blocked in save_log for student name %r",
                student_name,
            )
            return

        lock_path = log_path + ".lock"

        log_data = {
            "student": student_name,
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_issue_summary(),
            "issues": [i.to_dict() for i in self.issues],
            "api_errors": self.api_errors,
        }

        try:
            with FileLock(lock_path, timeout=_LOCK_TIMEOUT_SECONDS):
                fd, tmp_path = tempfile.mkstemp(
                    prefix=f".{os.path.basename(log_path)}.",
                    suffix=".tmp",
                    dir=os.path.dirname(log_path),
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(log_data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, log_path)
                except Exception:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    raise
        except Timeout:
            logger.error(
                "Could not acquire issue-log lock for %s within %ss -- skipping write",
                student_name,
                _LOCK_TIMEOUT_SECONDS,
            )

    # ------------------------------------------------------------------
    # Tutor guidance
    # ------------------------------------------------------------------

    def get_tutor_guidance(self) -> str | None:
        """
        Generate guidance text for the tutor agent based on current issues.
        Returns None if no action needed.

        This is injected into the LLM system prompt so the tutor can
        self-correct its behavior.
        """
        high_issues = [i for i in self.issues[-5:] if i.severity in ("high", "critical")]
        if not high_issues:
            return None

        guidance_parts: list[str] = []
        for issue in high_issues:
            guidance_parts.append(
                f"DETECTED ISSUE: {issue.message}\nSUGGESTED ACTION: {issue.suggested_fix}"
            )

        return "\n\n".join(guidance_parts)
