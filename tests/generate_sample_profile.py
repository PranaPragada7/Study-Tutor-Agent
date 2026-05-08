"""
Sample Profile Generator

Creates a pre-loaded student profile with realistic quiz history, useful for
inspecting the Progress and Diagnostics dashboards without starting from a blank
profile.

Run with: python tests/generate_sample_profile.py
"""

import sys
import os
import random
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.student_profile import create_profile, record_quiz_result, save_profile
from utils.spaced_repetition import SpacedRepetitionScheduler


class _MutableClock:
    """Tiny advancing clock so the sample profile quiz_history and SM-2 schedule
    actually reflect 2 simulated weeks instead of all collapsing to 'now'.

    Used as ``now_fn`` for ``SpacedRepetitionScheduler``: each iteration
    of the day loop advances ``self.t`` to the simulated quiz time, and
    the scheduler computes ``next_review`` from that — so a topic
    quizzed on simulated day 0 with interval=1 becomes due on simulated
    day 1, not real-time tomorrow.
    """

    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def generate_sample_profile():
    """Generate a realistic sample profile with learning progression."""

    print("📚 Generating sample profile: 'Sample Student'")

    # Create profile with courses
    profile = create_profile("Sample Student", [
        {"name": "Intro to CS", "difficulty": 3, "exam_date": "2025-05-15"},
        {"name": "Calculus II", "difficulty": 4, "exam_date": "2025-05-20"},
    ], learning_style="balanced")

    # Define topics per course
    cs_topics = {
        "Variables & Types": {"base_accuracy": 0.85, "trend": 0.02},
        "Control Flow": {"base_accuracy": 0.75, "trend": 0.03},
        "Functions": {"base_accuracy": 0.65, "trend": 0.04},
        "Recursion": {"base_accuracy": 0.35, "trend": 0.05},
        "Data Structures": {"base_accuracy": 0.50, "trend": 0.03},
    }

    calc_topics = {
        "Integration by Parts": {"base_accuracy": 0.55, "trend": 0.03},
        "Series Convergence": {"base_accuracy": 0.40, "trend": 0.04},
        "Taylor Series": {"base_accuracy": 0.30, "trend": 0.05},
        "Polar Coordinates": {"base_accuracy": 0.70, "trend": 0.02},
    }

    # Add topics to profile
    for t in cs_topics:
        profile["courses"]["Intro to CS"]["topics"][t] = {"correct": 0, "attempted": 0, "current_difficulty": 2}
    for t in calc_topics:
        profile["courses"]["Calculus II"]["topics"][t] = {"correct": 0, "attempted": 0, "current_difficulty": 2}

    # Mutable clock so both record_quiz_result timestamps AND the
    # spaced-repetition scheduler advance through 2 simulated weeks.
    base_time = datetime.now(timezone.utc) - timedelta(days=14)
    clock = _MutableClock(base_time)
    scheduler = SpacedRepetitionScheduler(now_fn=clock)

    question_num = 0

    sample_questions = {
        "Variables & Types": [
            "What is the difference between an integer and a float?",
            "Which keyword declares a constant in Python?",
            "What type does input() return by default?",
        ],
        "Control Flow": [
            "What does a while loop do when the condition is False?",
            "How does elif differ from a nested if?",
            "What happens when break is inside a nested loop?",
        ],
        "Functions": [
            "What is the difference between parameters and arguments?",
            "What does return None mean in a function?",
            "Can a function call itself?",
        ],
        "Recursion": [
            "What is the base case in a recursive function?",
            "Why does recursion without a base case cause a stack overflow?",
            "How does the call stack work with recursion?",
        ],
        "Data Structures": [
            "What is the time complexity of list.append()?",
            "When would you use a dictionary vs a list?",
            "What is the difference between a stack and a queue?",
        ],
        "Integration by Parts": [
            "What is the formula for integration by parts?",
            "When should you use integration by parts vs substitution?",
            "How do you choose u and dv in integration by parts?",
        ],
        "Series Convergence": [
            "What is the ratio test for convergence?",
            "How do you determine if an alternating series converges?",
            "What is the difference between absolute and conditional convergence?",
        ],
        "Taylor Series": [
            "What is the Taylor series expansion of e^x?",
            "How do you find the radius of convergence?",
            "What is a Maclaurin series?",
        ],
        "Polar Coordinates": [
            "How do you convert from Cartesian to polar coordinates?",
            "What does r = cos(theta) look like?",
            "How do you find area in polar coordinates?",
        ],
    }

    for day in range(14):
        # 3-5 questions per day, spread across the afternoon/evening
        num_today = random.randint(3, 5)
        day_start = base_time + timedelta(days=day, hours=14)

        for q_in_day in range(num_today):
            question_num += 1

            # Spread quizzes across the day (14:00 to ~22:00) so the
            # quiz_history timestamps are monotonic, not bunched.
            timestamp = day_start + timedelta(
                hours=random.randint(0, 8),
                minutes=random.randint(0, 59),
            )
            clock.t = timestamp  # advance the SM-2 scheduler's clock

            # Alternate between courses
            if random.random() < 0.55:
                course = "Intro to CS"
                topics = cs_topics
            else:
                course = "Calculus II"
                topics = calc_topics

            topic = random.choice(list(topics.keys()))
            config = topics[topic]

            # Accuracy improves over time
            progress = day / 14.0
            accuracy = min(0.95, config["base_accuracy"] + progress * config["trend"] * 10)

            # Difficulty scales with accuracy
            if accuracy > 0.8:
                difficulty = random.choice([3, 4, 5])
            elif accuracy > 0.5:
                difficulty = random.choice([2, 3, 4])
            else:
                difficulty = random.choice([1, 2, 3])

            correct = random.random() < accuracy
            confidence = max(1, min(5, int(accuracy * 5) + random.randint(-1, 1)))

            question = random.choice(sample_questions.get(topic, [f"Question about {topic}?"]))

            # Inject the simulated timestamp via record_quiz_result's
            # ``now_fn`` hook so the sample profile quiz_history traces a 2-week
            # study arc instead of collapsing to wall-clock-now.
            profile = record_quiz_result(
                profile, course, topic, difficulty, correct,
                question, "A" if correct else random.choice(["B", "C", "D"]),
                confidence,
                now_fn=lambda t=timestamp: t,
            )

            # Update spaced repetition (uses the advanced clock)
            sm2_quality = scheduler.quality_from_result(correct, confidence, difficulty)
            scheduler.update_topic(topic, sm2_quality)

    # Save spaced repetition state
    profile["spaced_repetition"] = scheduler.to_dict()
    save_profile("Sample Student", profile)

    print(f"  ✅ Created profile with {profile['total_quizzes']} quiz results")
    print(f"  📊 Courses: {', '.join(profile['courses'].keys())}")
    for cname, cdata in profile["courses"].items():
        acc = cdata["total_correct"] / max(cdata["total_attempted"], 1) * 100
        print(f"     {cname}: {acc:.1f}% accuracy over {cdata['total_attempted']} questions")
    print(f"  💾 Saved to: data/sample_student.json")
    print()
    print("  To use: run the app and click Load Sample Student in the sidebar.")

    return profile


if __name__ == "__main__":
    generate_sample_profile()
