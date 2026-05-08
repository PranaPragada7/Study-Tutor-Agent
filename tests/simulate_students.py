"""
Simulated Student Evaluation — Automated Testing & Metrics

Runs automated quiz sessions with different simulated student personas
to evaluate whether the adaptive engine behaves correctly.

Personas:
  1. "Struggling Sam"   — Gets most questions wrong, especially on specific topics
  2. "Improving Iris"   — Starts weak, gradually improves over time
  3. "Random Randy"     — Answers randomly (50/50) — control baseline
  4. "Strong Steve"     — Gets most questions right — tests difficulty scaling up
  5. "Confused Casey"   — High confidence but often wrong — tests misconception detection

Each persona runs N quiz sessions. The script logs:
  - Accuracy over time (windowed)
  - Difficulty progression
  - Whether the RL engine adapted correctly
  - Issue detection events
  - Spaced repetition scheduling

Outputs:
  - CSV file: data/eval/evaluation_metrics.csv
  - Summary report: data/eval/evaluation_report.txt

Run with: python tests/simulate_students.py
      or: pytest tests/simulate_students.py -v
"""

import sys
import os
import csv
import random
from datetime import datetime

# Make imports work from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.adaptive_engine import AdaptiveEngine
from utils.spaced_repetition import SpacedRepetitionScheduler
from utils.issue_detector import IssueDetector
from utils.student_profile import create_profile, record_quiz_result

# ─── Output setup ───
# Eval outputs (CSV + report + persona profiles) live under eval-output/
# rather than data/, so a fresh clone doesn't ship 5 stale "students" as
# real student data. eval-output/ is gitignored.
EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval-output")
PERSONA_PROFILE_DIR = os.path.join(EVAL_DIR, "profiles")

TOPICS = ["Recursion", "Loops", "Data Structures", "Algorithms", "OOP"]
COURSE = "CS101"
NUM_QUESTIONS = 30  # Per persona


# ─── Simulated Student Personas ───

class SimulatedStudent:
    """Base class for simulated students.

    Each student accepts an optional seeded ``random.Random`` instance
    so that ``simulate_students.py`` tests are 100 % reproducible.
    """

    def __init__(self, name: str, description: str,
                 rng: random.Random | None = None):
        self.name = name
        self.description = description
        self.question_number = 0
        self._rng = rng or random.Random()

    def answer(self, topic: str, difficulty: int) -> tuple[bool, int]:
        """
        Simulate answering a question.
        Returns: (correct: bool, confidence: int 1-5)
        """
        raise NotImplementedError


class StrugglingStudent(SimulatedStudent):
    """Gets most questions wrong, especially on specific weak topics."""

    def __init__(self, rng: random.Random | None = None):
        super().__init__("Struggling Sam",
                         "Low accuracy, weak on Recursion and Algorithms",
                         rng=rng)
        self.weak_topics = {"Recursion", "Algorithms"}

    def answer(self, topic: str, difficulty: int) -> tuple[bool, int]:
        self.question_number += 1
        if topic in self.weak_topics:
            correct = self._rng.random() < 0.15
            confidence = self._rng.randint(1, 2)
        else:
            correct = self._rng.random() < 0.45
            confidence = self._rng.randint(2, 3)
        return correct, confidence


class ImprovingStudent(SimulatedStudent):
    """Starts weak but gradually improves."""

    def __init__(self, rng: random.Random | None = None):
        super().__init__("Improving Iris",
                         "Accuracy improves from 30% to 80% over time",
                         rng=rng)

    def answer(self, topic: str, difficulty: int) -> tuple[bool, int]:
        self.question_number += 1
        progress = min(self.question_number / NUM_QUESTIONS, 1.0)
        accuracy = 0.3 + progress * 0.5
        accuracy -= (difficulty - 3) * 0.08
        correct = self._rng.random() < max(0.1, min(0.95, accuracy))
        confidence = min(5, 2 + int(progress * 3))
        return correct, confidence


class RandomStudent(SimulatedStudent):
    """Answers randomly — control baseline."""

    def __init__(self, rng: random.Random | None = None):
        super().__init__("Random Randy",
                         "50/50 chance on every question (control)",
                         rng=rng)

    def answer(self, topic: str, difficulty: int) -> tuple[bool, int]:
        self.question_number += 1
        correct = self._rng.random() < 0.5
        confidence = self._rng.randint(1, 5)
        return correct, confidence


class StrongStudent(SimulatedStudent):
    """Gets most questions right — tests difficulty scaling up."""

    def __init__(self, rng: random.Random | None = None):
        super().__init__("Strong Steve",
                         "High accuracy, should trigger difficulty increase",
                         rng=rng)

    def answer(self, topic: str, difficulty: int) -> tuple[bool, int]:
        self.question_number += 1
        accuracy = 0.90 - (difficulty - 1) * 0.05
        correct = self._rng.random() < accuracy
        confidence = self._rng.randint(4, 5)
        return correct, confidence


class ConfusedStudent(SimulatedStudent):
    """High confidence but often wrong — tests misconception detection."""

    def __init__(self, rng: random.Random | None = None):
        super().__init__("Confused Casey",
                         "Confident but wrong — should trigger misconception alerts",
                         rng=rng)

    def answer(self, topic: str, difficulty: int) -> tuple[bool, int]:
        self.question_number += 1
        correct = self._rng.random() < 0.35
        confidence = self._rng.randint(4, 5)
        return correct, confidence


# ─── Simulation Runner ───

def run_simulation(student: SimulatedStudent,
                   num_questions: int = NUM_QUESTIONS,
                   engine_seed: int | None = None) -> list[dict]:
    """
    Run a full simulation for one student persona.

    Args:
        student: The simulated student persona.
        num_questions: Number of questions to simulate.
        engine_seed: Optional seed for the AdaptiveEngine's RNG so
                     topic/difficulty selection is also deterministic.

    Returns list of per-question metrics.
    """
    engine = AdaptiveEngine(seed=engine_seed)
    scheduler = SpacedRepetitionScheduler()
    detector = IssueDetector()

    # Create a profile for this simulated student
    profile = create_profile(f"sim_{student.name}", [{"name": COURSE, "difficulty": 3}])

    # Add topics to profile
    for t in TOPICS:
        profile["courses"][COURSE]["topics"][t] = {"correct": 0, "attempted": 0, "current_difficulty": 2}

    metrics = []
    window_size = 5
    recent_correct = []

    for q_num in range(1, num_questions + 1):
        # Agent selects topic and difficulty
        topic, difficulty = engine.select_topic_and_difficulty(TOPICS)

        # Student answers
        correct, confidence = student.answer(topic, difficulty)

        # Update all systems
        engine.update(topic, difficulty, correct, confidence)
        sm2_quality = scheduler.quality_from_result(correct, confidence, difficulty)
        scheduler.update_topic(topic, sm2_quality)

        # Record in profile for issue detection
        profile = record_quiz_result(
            profile, COURSE, topic, difficulty, correct,
            f"Q{q_num} about {topic}", "A" if correct else "B", confidence
        )

        # Run issue detection
        new_issues = detector.analyze_quiz_result(profile, COURSE, topic, difficulty, correct)

        # Track windowed accuracy
        recent_correct.append(1 if correct else 0)
        if len(recent_correct) > window_size:
            recent_correct = recent_correct[-window_size:]
        window_accuracy = sum(recent_correct) / len(recent_correct)

        # Record metrics
        metrics.append({
            "student": student.name,
            "question_num": q_num,
            "topic": topic,
            "difficulty": difficulty,
            "correct": correct,
            "confidence": confidence,
            "window_accuracy": round(window_accuracy, 3),
            "epsilon": round(engine.epsilon, 4),
            "sm2_quality": sm2_quality,
            "issues_detected": len(new_issues),
            "issue_types": "|".join([i.issue_type for i in new_issues]) if new_issues else "",
            "recommended_diff": engine.recommend_difficulty(topic),
        })

    return metrics


def run_all_simulations(seed: int = 42) -> list[dict]:
    """Run simulations for all student personas with deterministic seeding."""
    # Each persona gets its own seeded RNG derived from the master seed
    # so results are fully reproducible.
    personas = [
        StrugglingStudent(rng=random.Random(seed + 1)),
        ImprovingStudent(rng=random.Random(seed + 2)),
        RandomStudent(rng=random.Random(seed + 3)),
        StrongStudent(rng=random.Random(seed + 4)),
        ConfusedStudent(rng=random.Random(seed + 5)),
    ]

    all_metrics = []
    for i, persona in enumerate(personas):
        print(f"  Running: {persona.name} ({persona.description})...")
        metrics = run_simulation(persona, engine_seed=seed + 100 + i)
        all_metrics.extend(metrics)
        print(f"    ✅ {len(metrics)} questions simulated")

    return all_metrics


def save_csv(metrics: list[dict], filepath: str) -> None:
    """Save metrics to CSV."""
    if not metrics:
        return

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)


def generate_report(metrics: list[dict]) -> str:
    """Generate a human-readable evaluation report."""
    lines = [
        "=" * 60,
        "AI STUDY TUTOR — EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total simulated questions: {len(metrics)}",
        "=" * 60,
        "",
    ]

    # Group by student
    students = {}
    for m in metrics:
        name = m["student"]
        if name not in students:
            students[name] = []
        students[name].append(m)

    for name, data in students.items():
        total = len(data)
        correct = sum(1 for d in data if d["correct"])
        accuracy = correct / total * 100

        # First half vs second half accuracy (learning check)
        mid = total // 2
        first_half_acc = sum(1 for d in data[:mid] if d["correct"]) / mid * 100
        second_half_acc = sum(1 for d in data[mid:] if d["correct"]) / (total - mid) * 100

        # Average difficulty
        avg_diff = sum(d["difficulty"] for d in data) / total

        # First and last difficulty
        first_diff = sum(d["difficulty"] for d in data[:5]) / 5
        last_diff = sum(d["difficulty"] for d in data[-5:]) / 5

        # Issues detected
        total_issues = sum(d["issues_detected"] for d in data)

        lines.extend([
            f"--- {name} ---",
            f"  Overall accuracy:       {accuracy:.1f}%",
            f"  First half accuracy:    {first_half_acc:.1f}%",
            f"  Second half accuracy:   {second_half_acc:.1f}%",
            f"  Average difficulty:     {avg_diff:.1f}/5",
            f"  Difficulty trend:       {first_diff:.1f} → {last_diff:.1f}",
            f"  Issues detected:        {total_issues}",
            "",
        ])

        # Adaptation analysis
        if name == "Struggling Sam":
            if last_diff < first_diff:
                lines.append(f"  ✅ PASS: Difficulty decreased for struggling student ({first_diff:.1f}→{last_diff:.1f})")
            else:
                lines.append(f"  ⚠️  CHECK: Difficulty did not decrease enough ({first_diff:.1f}→{last_diff:.1f})")

        elif name == "Improving Iris":
            if second_half_acc > first_half_acc:
                lines.append(f"  ✅ PASS: Accuracy improved over time ({first_half_acc:.1f}%→{second_half_acc:.1f}%)")
            else:
                lines.append(f"  ⚠️  CHECK: Accuracy did not improve as expected")

        elif name == "Strong Steve":
            if last_diff > first_diff:
                lines.append(f"  ✅ PASS: Difficulty increased for strong student ({first_diff:.1f}→{last_diff:.1f})")
            else:
                lines.append(f"  ⚠️  CHECK: Difficulty did not increase enough ({first_diff:.1f}→{last_diff:.1f})")

        elif name == "Confused Casey":
            if total_issues > 0:
                lines.append(f"  ✅ PASS: Issues detected for confused student ({total_issues} issues)")
            else:
                lines.append(f"  ⚠️  CHECK: No issues detected for student with misconceptions")

        elif name == "Random Randy":
            lines.append(f"  📊 BASELINE: Control student (random answers)")

        lines.append("")

    # Overall summary
    lines.extend([
        "=" * 60,
        "OVERALL EVALUATION SUMMARY",
        "=" * 60,
        f"  Total personas tested:  {len(students)}",
        f"  Total questions run:    {len(metrics)}",
        f"  Total issues detected:  {sum(d['issues_detected'] for d in metrics)}",
        "",
    ])

    # Count passes
    passes = 0
    total_checks = 4

    struggling = students.get("Struggling Sam", [])
    if struggling:
        s_first = sum(d["difficulty"] for d in struggling[:5]) / 5
        s_last = sum(d["difficulty"] for d in struggling[-5:]) / 5
        if s_last <= s_first:
            passes += 1

    strong = students.get("Strong Steve", [])
    if strong:
        st_first = sum(d["difficulty"] for d in strong[:5]) / 5
        st_last = sum(d["difficulty"] for d in strong[-5:]) / 5
        if st_last >= st_first:
            passes += 1

    improving = students.get("Improving Iris", [])
    if improving:
        mid = len(improving) // 2
        i_first = sum(1 for d in improving[:mid] if d["correct"]) / mid
        i_second = sum(1 for d in improving[mid:] if d["correct"]) / (len(improving) - mid)
        if i_second >= i_first:
            passes += 1

    confused = students.get("Confused Casey", [])
    if confused:
        c_issues = sum(d["issues_detected"] for d in confused)
        if c_issues > 0:
            passes += 1

    lines.append(f"  Adaptation checks passed: {passes}/{total_checks}")
    lines.append("")

    return "\n".join(lines)


# ─── Pytest-compatible test functions ───

def test_simulation_determinism():
    """Running the same seed twice must produce identical metrics."""
    rng1 = random.Random(999)
    student1 = StrugglingStudent(rng=rng1)
    m1 = run_simulation(student1, num_questions=10, engine_seed=999)

    rng2 = random.Random(999)
    student2 = StrugglingStudent(rng=rng2)
    m2 = run_simulation(student2, num_questions=10, engine_seed=999)

    for a, b in zip(m1, m2):
        assert a["correct"] == b["correct"], "Non-deterministic student answers"
        assert a["topic"] == b["topic"], "Non-deterministic topic selection"
        assert a["difficulty"] == b["difficulty"], "Non-deterministic difficulty"


def test_struggling_student_triggers_issues():
    """Struggling Sam should trigger accuracy_drop issues."""
    student = StrugglingStudent(rng=random.Random(42))
    metrics = run_simulation(student, num_questions=30, engine_seed=42)
    issue_types = set()
    for m in metrics:
        if m["issue_types"]:
            for t in m["issue_types"].split("|"):
                issue_types.add(t)
    assert "accuracy_drop" in issue_types


def test_strong_student_difficulty_scales():
    """Strong Steve's recommended difficulty should be >= 3 after enough Qs."""
    student = StrongStudent(rng=random.Random(42))
    metrics = run_simulation(student, num_questions=30, engine_seed=42)
    final_diffs = [m["recommended_diff"] for m in metrics[-5:]]
    assert max(final_diffs) >= 3


# ─── Main ───

if __name__ == "__main__":
    import io, sys as _sys
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(PERSONA_PROFILE_DIR, exist_ok=True)

    # Redirect persona profiles into eval-output/profiles/ so they don't
    # pollute data/ next to real student JSONs. The simulation reuses
    # create_profile() which writes via student_profile.DATA_DIR.
    import utils.student_profile as _sp
    _sp.DATA_DIR = PERSONA_PROFILE_DIR

    # Seed for reproducible evaluation results
    random.seed(42)

    print("Running AI Study Tutor — Simulated Evaluation")
    print(f"   {NUM_QUESTIONS} questions per persona x 5 personas = {NUM_QUESTIONS * 5} total")
    print()

    metrics = run_all_simulations(seed=42)

    # Save CSV
    csv_path = os.path.join(EVAL_DIR, "evaluation_metrics.csv")
    save_csv(metrics, csv_path)
    print(f"\n📊 Metrics saved to: {csv_path}")

    # Generate and save report
    report = generate_report(metrics)
    report_path = os.path.join(EVAL_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📝 Report saved to:  {report_path}")

    # Print report
    print()
    print(report)
