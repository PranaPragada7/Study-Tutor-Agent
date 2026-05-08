"""
Baseline-policy benchmark for the Adaptive Engine.

Why this exists
---------------
``simulate_students.py`` already runs five personas through the
adaptive engine and asserts directional adaptation (Sam's difficulty
trends down, Steve's trends up, etc.). What it does NOT do is prove
that the adaptive engine actually beats simpler policies. Without
that comparison, claims like "the bandit targets ZPD" are
self-reported.

This module runs each persona through six policies on the same seed
sequence and emits a comparison table. The metrics are:

  - Per-window accuracy (last 5 questions) — does the policy keep
    the student in the productive band?
  - Average difficulty — is the policy actually adapting, or is it
    pinned at one level?
  - Issues flagged — are misconceptions surfacing?
  - Distance to ZPD — mean |window_accuracy − 0.8|, smaller is better.

Run with::

    python tests/benchmark_policies.py
    pytest tests/benchmark_policies.py -k determinism

Outputs::

    eval-output/policy_benchmark.csv     -- per-policy/persona row metrics
    eval-output/policy_benchmark.txt     -- human-readable comparison table

Policies
--------
``RandomPolicy``           : uniform topic + uniform difficulty
``FixedDifficultyPolicy``  : random topic + constant difficulty=3
``WeakestTopicPolicy``     : pick lowest-accuracy topic, difficulty=2
``Sm2OnlyPolicy``          : pick a due-for-review topic if any, else random
``AdaptiveNoO9Policy``     : the engine WITHOUT the cold-start dampening
                              tweak (regression target — does O9 actually help?)
``AdaptiveFullPolicy``     : the current engine
"""

from __future__ import annotations

import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.adaptive_engine import AdaptiveEngine
from utils.spaced_repetition import SpacedRepetitionScheduler
from utils.issue_detector import IssueDetector
from utils.student_profile import create_profile, record_quiz_result

from tests.simulate_students import (
    ConfusedStudent,
    ImprovingStudent,
    NUM_QUESTIONS,
    RandomStudent,
    StrongStudent,
    StrugglingStudent,
    TOPICS,
)

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval-output")
COURSE = "CS101"


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

@dataclass
class PolicyContext:
    """State the policies are allowed to read. The bandit-style
    policies mutate ``engine``; the simpler policies ignore it."""
    engine: AdaptiveEngine
    scheduler: SpacedRepetitionScheduler


class Policy:
    name: str = "policy"

    def select(self, ctx: PolicyContext, available_topics: list[str]) -> tuple[str, int]:
        raise NotImplementedError

    def update(self, ctx: PolicyContext, topic: str, difficulty: int,
               correct: bool, confidence: int) -> None:
        # Default: feed the engine so its mastery / topic_accuracy stays
        # current even for non-bandit policies (so the metrics tab can
        # still show "did the student get better?").
        ctx.engine.update(topic, difficulty, correct, confidence)


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def select(self, ctx, available_topics):
        return self._rng.choice(available_topics), self._rng.randint(1, 5)


class FixedDifficultyPolicy(Policy):
    name = "fixed_diff_3"

    def __init__(self, rng: random.Random, difficulty: int = 3):
        self._rng = rng
        self._d = difficulty

    def select(self, ctx, available_topics):
        return self._rng.choice(available_topics), self._d


class WeakestTopicPolicy(Policy):
    name = "weakest_first"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def select(self, ctx, available_topics):
        # Pick the topic with the lowest ``correct/total`` so far. Ties
        # break randomly (so we don't always pick the same topic on a
        # fresh engine where all topics tie at 0/0).
        scores = []
        for t in available_topics:
            stats = ctx.engine.topic_accuracy.get(t, {"correct": 0, "total": 0})
            total = max(stats["total"], 1)
            scores.append((stats["correct"] / total, self._rng.random(), t))
        scores.sort()
        return scores[0][2], 2


class Sm2OnlyPolicy(Policy):
    name = "sm2_only"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def select(self, ctx, available_topics):
        due = ctx.scheduler.get_due_topics(available_topics)
        if due:
            return due[0]["topic"], 2
        return self._rng.choice(available_topics), 2


class AdaptiveNoO9Policy(Policy):
    """The full adaptive engine WITHOUT the O9 cold-start dampening.

    This is the regression target: running the same simulation through
    "AdaptiveFull" vs "AdaptiveNoO9" lets us measure whether O9 (the
    Struggling-Sam fix) actually buys anything.
    """
    name = "adaptive_no_o9"

    def select(self, ctx, available_topics):
        return ctx.engine.select_topic_and_difficulty(available_topics)


class AdaptiveFullPolicy(Policy):
    name = "adaptive_full"

    def select(self, ctx, available_topics):
        return ctx.engine.select_topic_and_difficulty(available_topics)


def _build_engine_no_o9(seed: int) -> AdaptiveEngine:
    """Construct an engine and monkey-patch ``_compute_arm_value`` to
    use the pre-O9 cold-start formula (no weakness-aware dampening of
    cold_exploration). Surgical so we don't have to maintain a fork."""
    import math
    engine = AdaptiveEngine(seed=seed)
    original = engine._compute_arm_value

    def pre_o9(topic, difficulty, log_total=None):  # noqa: ANN001
        # Reproduce the seen-arm branch unchanged; only override the
        # cold-start path.
        key = (topic, difficulty)
        arm = engine.arm_stats.get(key, None)
        if log_total is None:
            total = max(sum(a["count"] for a in engine.arm_stats.values()), 0)
            log_total = math.log(total + 1)
        if arm is None or arm["count"] == 0:
            zpd_score = 1.0 - abs(engine._expected_mastery(topic, difficulty) - 0.8)
            cold_exploration = 0.2 * math.sqrt(2 * log_total / 0.5)  # NO dampening
            return zpd_score + cold_exploration + engine._weakness_bonus(topic)
        # Defer to the (unchanged) seen-arm branch by calling the
        # original. We can't, because the original would re-apply the
        # cold-start branch too. Inline the seen-arm math instead.
        estimated_mastery = arm["total_reward"] / arm["count"]
        zpd_score = 1.0 - abs(estimated_mastery - 0.8)
        exploration_bonus = math.sqrt(2 * log_total / arm["count"])
        return zpd_score + (0.2 * exploration_bonus) + engine._weakness_bonus(topic)

    engine._compute_arm_value = pre_o9  # type: ignore[assignment]
    return engine


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_one(policy: Policy, student, num_questions: int,
             engine_seed: int) -> dict:
    """Run a single (policy, student) pair, return aggregated metrics."""
    if policy.name == "adaptive_no_o9":
        engine = _build_engine_no_o9(engine_seed)
    else:
        engine = AdaptiveEngine(seed=engine_seed)
    scheduler = SpacedRepetitionScheduler()
    detector = IssueDetector()
    profile = create_profile(f"bench_{policy.name}_{student.name}",
                              [{"name": COURSE, "difficulty": 3}])
    for t in TOPICS:
        profile["courses"][COURSE]["topics"][t] = {
            "correct": 0, "attempted": 0, "current_difficulty": 2,
        }

    ctx = PolicyContext(engine=engine, scheduler=scheduler)

    correct_count = 0
    diffs: list[int] = []
    window_acc_history: list[float] = []
    recent: list[int] = []
    total_issues = 0

    for _ in range(num_questions):
        topic, difficulty = policy.select(ctx, TOPICS)
        is_correct, confidence = student.answer(topic, difficulty)
        policy.update(ctx, topic, difficulty, is_correct, confidence)
        sm2 = scheduler.quality_from_result(is_correct, confidence, difficulty)
        scheduler.update_topic(topic, sm2)
        profile = record_quiz_result(profile, COURSE, topic, difficulty,
                                      is_correct, "q", "A", confidence)
        new_issues = detector.analyze_quiz_result(profile, COURSE, topic,
                                                   difficulty, is_correct)
        total_issues += len(new_issues)

        diffs.append(difficulty)
        recent.append(1 if is_correct else 0)
        if len(recent) > 5:
            recent = recent[-5:]
        window_acc_history.append(sum(recent) / len(recent))
        if is_correct:
            correct_count += 1

    avg_window_acc = sum(window_acc_history) / len(window_acc_history)
    zpd_distance = sum(abs(w - 0.8) for w in window_acc_history) / len(window_acc_history)
    return {
        "policy": policy.name,
        "student": student.name,
        "questions": num_questions,
        "overall_accuracy": round(correct_count / num_questions, 3),
        "avg_window_accuracy": round(avg_window_acc, 3),
        "avg_difficulty": round(sum(diffs) / len(diffs), 2),
        "first_5_diff": round(sum(diffs[:5]) / 5, 2),
        "last_5_diff": round(sum(diffs[-5:]) / 5, 2),
        "zpd_distance": round(zpd_distance, 3),
        "issues_detected": total_issues,
    }


def _make_personas(seed: int):
    """Persona factory; each persona gets its own seeded RNG so
    cross-policy comparisons run on identical answer streams."""
    return [
        StrugglingStudent,
        ImprovingStudent,
        RandomStudent,
        StrongStudent,
        ConfusedStudent,
    ]


def _make_policies(seed: int) -> list[Policy]:
    return [
        RandomPolicy(rng=random.Random(seed + 11)),
        FixedDifficultyPolicy(rng=random.Random(seed + 12)),
        WeakestTopicPolicy(rng=random.Random(seed + 13)),
        Sm2OnlyPolicy(rng=random.Random(seed + 14)),
        AdaptiveNoO9Policy(),
        AdaptiveFullPolicy(),
    ]


def run_benchmark(seed: int = 42,
                  num_questions: int = NUM_QUESTIONS) -> list[dict]:
    """Run every persona × every policy. Returns flat list of metric rows."""
    rows: list[dict] = []
    persona_classes = _make_personas(seed)
    policies = _make_policies(seed)

    for persona_idx, persona_cls in enumerate(persona_classes):
        for policy in policies:
            # Re-seed the persona for each policy so they all see the
            # same answer-correctness stream (but selections diverge).
            student = persona_cls(rng=random.Random(seed + persona_idx + 1))
            engine_seed = seed + 100 + persona_idx
            row = _run_one(policy, student, num_questions, engine_seed)
            rows.append(row)
    return rows


def format_report(rows: list[dict]) -> str:
    """Pivot the rows into a per-persona policy-comparison table."""
    by_persona: dict[str, list[dict]] = {}
    for r in rows:
        by_persona.setdefault(r["student"], []).append(r)

    lines: list[str] = [
        "=" * 78,
        "POLICY BENCHMARK — adaptive engine vs. baselines",
        "Lower zpd_distance is better (closer to 80% productive band).",
        "=" * 78,
        "",
    ]
    for persona, persona_rows in by_persona.items():
        # Sort by zpd_distance ascending so the best policy appears first.
        persona_rows.sort(key=lambda r: r["zpd_distance"])
        lines.append(f"--- {persona} ---")
        lines.append(
            f"  {'policy':<18} {'overall_acc':>11}  {'avg_window':>10}  "
            f"{'avg_diff':>8}  {'zpd_dist':>9}  {'issues':>6}"
        )
        for r in persona_rows:
            lines.append(
                f"  {r['policy']:<18} {r['overall_accuracy']:>11.3f}  "
                f"{r['avg_window_accuracy']:>10.3f}  "
                f"{r['avg_difficulty']:>8.2f}  "
                f"{r['zpd_distance']:>9.3f}  {r['issues_detected']:>6}"
            )
        lines.append("")
    return "\n".join(lines)


def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Pytest hook: at minimum, prove the benchmark runs deterministically.
# ---------------------------------------------------------------------------

def test_benchmark_determinism():
    """Same seed → identical metrics. Catches accidental
    nondeterminism from a stray ``random.random()`` slipping into a
    new policy."""
    a = run_benchmark(seed=999, num_questions=10)
    b = run_benchmark(seed=999, num_questions=10)
    assert a == b


def test_adaptive_full_beats_random_on_zpd_for_strong_student():
    """Sanity check: adaptive_full should keep Strong Steve closer to
    ZPD than random selection. If this fails, the bandit is broken."""
    rows = run_benchmark(seed=42, num_questions=30)
    by_key = {(r["policy"], r["student"]): r for r in rows}
    full = by_key[("adaptive_full", "Strong Steve")]
    rnd = by_key[("random", "Strong Steve")]
    assert full["zpd_distance"] <= rnd["zpd_distance"], (
        f"adaptive_full ({full['zpd_distance']}) should beat random "
        f"({rnd['zpd_distance']}) on Strong Steve"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    os.makedirs(EVAL_DIR, exist_ok=True)
    print("Running policy benchmark — 6 policies × 5 personas × 30 questions...\n")
    rows = run_benchmark(seed=42)
    csv_path = os.path.join(EVAL_DIR, "policy_benchmark.csv")
    save_csv(rows, csv_path)
    report = format_report(rows)
    txt_path = os.path.join(EVAL_DIR, "policy_benchmark.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nMetrics saved to: {csv_path}")
    print(f"Report saved to:  {txt_path}")
