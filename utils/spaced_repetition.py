"""
Spaced Repetition Scheduler (SM-2 Algorithm)

The SM-2 algorithm was developed by Piotr Wozniak for SuperMemo. It schedules
review sessions at increasing intervals based on how well the student knows each topic.

Key concept:
  - Each topic has an "easiness factor" (EF) that starts at 2.5, floored at
    1.3 and capped at 3.0 so a misbehaving caller can't drive EF unboundedly
  - After each review, EF adjusts up (if easy) or down (if hard)
  - The interval between reviews grows: 1 day → 6 days → EF*previous_interval,
    capped at 365 days so the UI doesn't display "next review in 10 years"
  - If the student fails, the interval resets to 1 day

This is a PURE LOCAL algorithm — no LLM needed.

Time handling:
  All datetimes are timezone-aware UTC. A ``now_fn`` can be injected at
  construction time so unit tests don't have to wait real seconds (or use
  ``freezegun``) to verify the SM-2 progression. Legacy naive ISO strings
  in persisted state are normalized to UTC on read so old profiles keep
  working.
"""

import copy
from datetime import datetime, timedelta, timezone
from typing import Callable

from config import (
    DAILY_REVIEW_CAP as _DAILY_REVIEW_CAP,
)
from config import (
    MAX_EASINESS_FACTOR as _MAX_EASINESS_FACTOR,
)
from config import (
    MAX_INTERVAL_DAYS as _MAX_INTERVAL_DAYS,
)


def _default_now() -> datetime:
    """Default clock — timezone-aware UTC."""
    return datetime.now(timezone.utc)


def _parse_iso_utc(value: str) -> datetime:
    """Parse an ISO timestamp; normalize naive values to UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class SpacedRepetitionScheduler:
    """
    SM-2 based scheduler that decides WHEN to review each topic.

    Works alongside the adaptive engine which decides WHAT difficulty to use.
    """

    def __init__(self, now_fn: Callable[[], datetime] | None = None):
        """
        Args:
            now_fn: Optional callable that returns "the current time". Defaults
                    to ``datetime.now(timezone.utc)``. Tests can inject a fake
                    clock to advance time deterministically without sleeping.
        """
        # {topic: {easiness_factor, interval_days, repetition_count, next_review, last_quality}}
        self.topic_schedule: dict[str, dict] = {}
        self._now: Callable[[], datetime] = now_fn or _default_now

    def update_topic(self, topic: str, quality: int) -> dict:
        """
        Update the schedule for a topic after a review.

        Args:
            quality: 0-5 rating of how well the student knew it
                     0-1 = complete blackout, needs immediate re-review
                     2   = wrong but recognized the answer
                     3   = correct but with serious difficulty
                     4   = correct with some hesitation
                     5   = perfect recall, felt easy

        Returns:
            Updated schedule info for this topic
        """
        # Clamp quality to the SM-2 valid range. A bad caller passing
        # quality=10 would yield ef_delta = 0.1 - (-5)*(0.08 + (-5)*0.02)
        # = 0.1 - (-5)*(-0.02) = 0.0 — wrong direction. Clamp first so the
        # EF math below always behaves.
        quality = max(0, min(5, int(quality)))

        now = self._now()

        if topic not in self.topic_schedule:
            # Initialize with a SAFE placeholder (1 day out, not "now") so that
            # if we crash between this dict creation and the SM-2 computation
            # below, the topic isn't silently left as "due immediately".
            self.topic_schedule[topic] = {
                "easiness_factor": 2.5,
                "interval_days": 0,
                "repetition_count": 0,
                "next_review": (now + timedelta(days=1)).isoformat(),
                "last_quality": quality,
            }

        sched = self.topic_schedule[topic]

        # SM-2 algorithm
        if quality >= 3:
            # Correct response — increase interval
            if sched["repetition_count"] == 0:
                sched["interval_days"] = 1
            elif sched["repetition_count"] == 1:
                sched["interval_days"] = 6
            else:
                sched["interval_days"] = max(
                    1,
                    min(
                        _MAX_INTERVAL_DAYS,
                        round(sched["interval_days"] * sched["easiness_factor"]),
                    ),
                )

            sched["repetition_count"] += 1
        else:
            # Incorrect response — reset to beginning
            sched["repetition_count"] = 0
            sched["interval_days"] = 1

        # Update easiness factor
        # EF' = EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))
        ef_delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        sched["easiness_factor"] = max(
            1.3, min(_MAX_EASINESS_FACTOR, sched["easiness_factor"] + ef_delta)
        )

        # Schedule next review
        sched["next_review"] = (now + timedelta(days=sched["interval_days"])).isoformat()
        sched["last_quality"] = quality
        # O7: record the actual review time so get_due_topics can
        # deprioritise topics that were reviewed minutes ago — avoids the
        # "same 10 chronically-missed topics forever" starvation where
        # every miss resets the topic to 'due again immediately'.
        sched["last_reviewed"] = now.isoformat()

        return sched

    def get_due_topics(
        self, available_topics: list[str] | None = None, max_results: int = _DAILY_REVIEW_CAP
    ) -> list[dict]:
        """
        Get topics that are due for review (next_review <= now).

        Args:
            available_topics: If provided, only return topics from this list.
            max_results: Cap on returned entries so a student returning after
                a long absence gets a manageable review load, not 200 items.
                Pass a large number explicitly if all-entries is required.

        Returns:
            List of dicts: [{"topic": str, "days_overdue": float, "easiness": float}]
            Sorted by most-overdue first, breaking ties on lowest easiness
            so the hardest-for-this-student topics surface ahead of equally
            overdue easy ones.
        """
        now = self._now()
        due = []

        for topic, sched in self.topic_schedule.items():
            if available_topics and topic not in available_topics:
                continue

            nr = sched.get("next_review")
            if not nr:
                # Skip malformed entries instead of raising KeyError — a
                # bad on-disk dict shouldn't break the whole sidebar.
                continue
            try:
                next_review = _parse_iso_utc(nr)
            except (ValueError, TypeError):
                continue
            if next_review <= now:
                days_overdue = (now - next_review).total_seconds() / 86400
                # O7: Figure out how long ago the topic was actually
                # reviewed. Chronically-missed topics reset to interval=1
                # on every fail, which under the old sort "days_overdue
                # desc" kept them pinned to the top of the queue forever
                # and newly-learned topics never surfaced. Reviewed-today
                # topics drop below fresh ones in the secondary sort key.
                last_reviewed = sched.get("last_reviewed")
                reviewed_recently = False
                if last_reviewed:
                    try:
                        lr = _parse_iso_utc(last_reviewed)
                        reviewed_recently = (now - lr).total_seconds() < 86400
                    except (ValueError, TypeError):
                        reviewed_recently = False
                due.append(
                    {
                        "topic": topic,
                        "days_overdue": round(days_overdue, 1),
                        "easiness": round(sched.get("easiness_factor", 2.5), 2),
                        "interval_days": sched.get("interval_days", 0),
                        "_reviewed_recently": reviewed_recently,
                    }
                )

        # Primary: topics NOT reviewed in the last 24h come first (False <
        # True under tuple sort, so we flip with an int). Within each
        # group: most-overdue first, then lowest easiness (hardest).
        due.sort(
            key=lambda x: (
                1 if x["_reviewed_recently"] else 0,
                -x["days_overdue"],
                x["easiness"],
            )
        )
        # Strip the internal sort-only field before returning.
        for d in due:
            d.pop("_reviewed_recently", None)
        return due[:max_results]

    def get_upcoming_reviews(self, days_ahead: int = 7) -> list[dict]:
        """Get topics scheduled for review in the next N days.

        Mirrors the malformed-entry tolerance of get_due_topics: a bad
        on-disk dict should skip that row, not crash the whole sidebar.
        """
        now = self._now()
        cutoff = now + timedelta(days=days_ahead)
        upcoming = []

        for topic, sched in self.topic_schedule.items():
            nr = sched.get("next_review")
            if not nr:
                continue
            try:
                next_review = _parse_iso_utc(nr)
            except (ValueError, TypeError):
                continue
            if now < next_review <= cutoff:
                days_until = (next_review - now).total_seconds() / 86400
                upcoming.append(
                    {
                        "topic": topic,
                        "days_until_review": round(days_until, 1),
                        "interval_days": sched.get("interval_days", 0),
                    }
                )

        upcoming.sort(key=lambda x: x["days_until_review"])
        return upcoming

    def quality_from_result(self, correct: bool, confidence: int, difficulty: int) -> int:
        """
        Convert a quiz result + confidence into an SM-2 quality score (0-5).

        O4: Wrong answers are stratified by confidence so SM-2's EF math
        can separate "misconception" (confidently wrong — a disastrous
        signal) from "ordinary miss" (unsure and wrong — a known gap).
        Both reset the interval to 1 day, but the misconception drops EF
        more, which keeps the topic harder-to-master in the scheduler
        and matches the penalty the adaptive engine applies via reward=0.

        Args:
            correct: Whether the answer was correct
            confidence: Student's self-rated confidence (1-5; clamped)
            difficulty: Question difficulty (1-5; clamped)

        Returns:
            Quality score 0-5 for SM-2
        """
        try:
            confidence = max(1, min(5, int(confidence)))
        except (TypeError, ValueError):
            confidence = 3
        try:
            difficulty = max(1, min(5, int(difficulty)))
        except (TypeError, ValueError):
            difficulty = 2

        if not correct:
            if confidence >= 5:
                # Misconception: max confidence AND wrong. Biggest EF hit.
                return 1
            if confidence <= 2:
                return 0  # Complete blackout (known gap, unsure going in)
            # confidence 3 or 4: ordinary miss.
            return 2

        # Correct answer — quality depends on confidence and difficulty
        if confidence >= 4 and difficulty >= 3:
            return 5  # Easy recall on a hard question
        elif confidence >= 3:
            return 4  # Correct with some hesitation
        else:
            return 3  # Correct but it was a struggle

    def to_dict(self) -> dict:
        """Serialize for saving to profile."""
        return copy.deepcopy(self.topic_schedule)

    @classmethod
    def from_dict(cls, data: dict) -> "SpacedRepetitionScheduler":
        """Restore from saved profile data."""
        scheduler = cls()
        scheduler.topic_schedule = data
        return scheduler
