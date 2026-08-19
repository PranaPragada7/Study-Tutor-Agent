"""
Adaptive Learning Engine (Reinforcement Learning Component)

Uses an epsilon-greedy multi-armed bandit approach to decide:
  1. Which TOPIC to quiz the student on next
  2. What DIFFICULTY level to use

The "reward" signal comes from the student's quiz performance:
  - If the student gets it right at a hard level -> high reward (mastery!)
  - If the student gets it wrong at an easy level -> we need more practice here
  - The agent balances EXPLORATION (trying new topics/difficulties)
    vs EXPLOITATION (focusing on what needs work)

This is the LOCAL RL piece -- no LLM needed here.
"""

import math
import random


class AdaptiveEngine:
    """
    Multi-armed bandit for adaptive topic and difficulty selection.

    Each "arm" is a (topic, difficulty) pair.
    We track estimated reward for each arm and use epsilon-greedy
    to balance exploration vs exploitation.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        min_epsilon: float = 0.05,
        decay: float = 0.995,
        rng: random.Random | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            epsilon: Starting exploration rate (0-1). Higher = more random exploration.
            min_epsilon: Minimum exploration rate after decay.
            decay: How fast epsilon decreases each step.
            rng: Optional seeded ``random.Random`` instance for
                 deterministic behaviour in tests. When *None* the
                 module-level ``random`` functions are used (backward
                 compatible).
            seed: Convenience shortcut — if provided and *rng* is None,
                  creates ``random.Random(seed)`` automatically.  Makes
                  ``simulate_students.py`` tests 100% reproducible.
        """
        self.epsilon: float = epsilon
        self.min_epsilon: float = min_epsilon
        self.decay: float = decay

        # Deterministic RL: if a seeded RNG or seed is supplied, every
        # random selection in this engine is reproducible.
        if seed is not None and rng is None:
            rng = random.Random(seed)
        self._rng: random.Random | None = rng

        # Tracks: {(topic, difficulty): {"total_reward": float, "count": int}}
        self.arm_stats: dict[tuple[str, int], dict] = {}

        # Track overall topic performance for smarter decisions
        self.topic_accuracy: dict[str, dict] = {}  # {topic: {"correct": int, "total": int}}

    # ------------------------------------------------------------------
    # Internal RNG helpers (use seeded instance when provided)
    # ------------------------------------------------------------------

    def _rand(self) -> float:
        return self._rng.random() if self._rng else random.random()

    def _choice(self, seq: list) -> object:
        return self._rng.choice(seq) if self._rng else random.choice(seq)

    def _randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b) if self._rng else random.randint(a, b)

    # ------------------------------------------------------------------

    def _get_or_create_arm(self, topic: str, difficulty: int) -> dict:
        """Get stats for an arm, creating if needed."""
        key = (topic, difficulty)
        if key not in self.arm_stats:
            self.arm_stats[key] = {"total_reward": 0.0, "count": 0}
        return self.arm_stats[key]

    def select_topic_and_difficulty(
        self,
        available_topics: list[str],
        difficulty_range: tuple[int, int] = (1, 5),
    ) -> tuple[str, int]:
        """
        Select the next topic and difficulty level to quiz on.

        Args:
            available_topics: List of topic names to choose from
            difficulty_range: (min_difficulty, max_difficulty) inclusive

        Returns:
            (topic, difficulty) tuple
        """
        if not available_topics:
            raise ValueError("Must provide at least one available topic")

        min_diff, max_diff = difficulty_range

        # Epsilon-greedy: explore randomly with probability epsilon.
        # O3: stratify exploration by weakness bonus and by how "unseen"
        # each (topic, difficulty) arm is, so a single unlucky pick at
        # difficulty=5 on a fresh topic is less likely to tank the
        # student. A uniform choice over topic × difficulty gave every
        # struggling topic the same 1/N shot as a newly-learned one.
        if self._rand() < self.epsilon:
            return self._stratified_explore(available_topics, min_diff, max_diff)

        # Hoist the total-pull count (and its log) out of the inner loop.
        # Previously _compute_arm_value re-summed every arm's count on every
        # single call -- O(arms^2) per question. The value is constant across
        # this whole selection, so compute once.
        # O6: log(total_pulls + 1) so a single-pull session doesn't
        # collapse the exploration bonus to zero for every seen arm.
        total_pulls = max(sum(a["count"] for a in self.arm_stats.values()), 0)
        log_total = math.log(total_pulls + 1)

        # Exploit: pick the arm with the best "learning value"
        # We WANT to pick topics the student is weak on (low accuracy)
        # at an appropriate difficulty (not too easy, not too hard)
        best_score = -float("inf")
        best_choice: tuple[str, int] = (self._choice(available_topics), 2)

        for topic in available_topics:
            for diff in range(min_diff, max_diff + 1):
                score = self._compute_arm_value(topic, diff, log_total=log_total)
                if score > best_score:
                    best_score = score
                    best_choice = (topic, diff)

        return best_choice

    def _compute_arm_value(
        self, topic: str, difficulty: int, log_total: float | None = None
    ) -> float:
        """
        Compute the "learning value" of quizzing this topic at this difficulty.

        We target the Zone of Proximal Development (ZPD): arms where the
        student's estimated mastery is closest to 80%. This is what learning
        science says maximizes retention -- not too easy, not too hard.

        Args:
            log_total: Optional precomputed ``log(total_pulls)`` from the
                       caller. When called inside
                       :meth:`select_topic_and_difficulty`, it is hoisted
                       out of the (topic x difficulty) loop; otherwise we
                       recompute it here for callers that only need a
                       single arm value.
        """
        key = (topic, difficulty)
        arm = self.arm_stats.get(key, None)

        # O6: never take log(0); a single-pull session used to collapse
        # the UCB bonus to zero for every seen arm on the second question.
        if log_total is None:
            total = max(sum(a["count"] for a in self.arm_stats.values()), 0)
            log_total = math.log(total + 1)

        # O1/O2: cold-start arms used to win with a flat +0.1 tie-breaker.
        # After a few hundred pulls, the UCB bonus for seen arms
        # (0.2 * sqrt(2 * log T / count)) dwarfs +0.1, so unseen arms get
        # permanently starved — ~55% of the (topic, difficulty) space was
        # never sampled over 600 quizzes. Treat unseen arms as if they
        # had a fractional pseudo-count (0.5) so their exploration bonus
        # grows with T just like seen arms', keeping them competitive
        # for selection while still grounded in the ZPD estimate.
        #
        # Cold-start dampening: dampen exploration of harder-than-
        # recommended difficulties on topics the student is already
        # known-weak on.
        # Pre-fix, the cold-start bonus on an unseen (weak_topic, diff=5)
        # arm could outscore the exploited (weak_topic, diff=1) arm —
        # the student would do badly at diff=1 a few times, cement that
        # arm's avg reward at ~0.18, and then the bandit would route
        # them to diff=5 unseen "for exploration value", making things
        # harder for a struggling student. This is what surfaced as the
        # "Struggling Sam: difficulty did not decrease" eval miss.
        #
        # Scope of the dampening: only arms with difficulty STRICTLY
        # above the recommended level are dampened. Easier-than-
        # recommended unseen arms (the "review fundamentals" zone) keep
        # their full exploration bonus. This preserves the
        # misconception_gets_selection_priority guarantee — at
        # weakness=1.0, the recommended difficulty is 1, so the (diff=1)
        # unseen arm still gets the full +0.7 cold bonus and beats
        # mastered topics' unseen arms.
        #
        # Scaling factor — (1 - 0.5 * weakness_bonus) when above
        # recommended, 1.0 otherwise:
        #  - weakness=0.00 (mastered):       cold *= 1.00
        #  - weakness=0.55 (45 % accuracy):  cold *= 0.725 above recommend
        #  - weakness=0.85 (15 % accuracy):  cold *= 0.575 above recommend
        #  - weakness=1.00 (0 % accuracy):   cold *= 0.50  above recommend
        if arm is None or arm["count"] == 0:
            weakness_bonus = self._weakness_bonus(topic)
            zpd_score = 1.0 - abs(self._expected_mastery(topic, difficulty) - 0.8)
            recommended = self.recommend_difficulty(topic)
            if difficulty > recommended:
                cold_multiplier = 1.0 - 0.5 * weakness_bonus
            else:
                cold_multiplier = 1.0
            cold_exploration = 0.2 * math.sqrt(2 * log_total / 0.5) * cold_multiplier
            return zpd_score + cold_exploration + weakness_bonus

        # With the new reward shape, avg reward in [0, 1] approximates the
        # student's observed mastery on this (topic, difficulty) arm.
        estimated_mastery = arm["total_reward"] / arm["count"]

        # ZPD score peaks at mastery == 0.8 (80% success rate target).
        zpd_score = 1.0 - abs(estimated_mastery - 0.8)

        # UCB-like exploration bonus: try less-visited arms more. Weight
        # is 0.2 (not 0.5) so it nudges rather than dominates.
        exploration_bonus = math.sqrt(2 * log_total / arm["count"])

        return zpd_score + (0.2 * exploration_bonus) + self._weakness_bonus(topic)

    def _expected_mastery(self, topic: str, difficulty: int) -> float:
        """
        Estimate mastery for a (topic, difficulty) arm we've never pulled.

        Anchor on the student's observed accuracy for this topic (when
        we have any), then shift down for harder difficulties and up for
        easier ones. Neutral 0.5 prior when there is no topic data yet.

        O5: slope is 0.2/step so the ZPD delta between the
        ``recommend_difficulty`` target and the next difficulty up/down
        is large enough to dominate the +0.1 cold-start nudge and the
        weakness bonus. With 0.15/step the engine was only mildly
        preferring the recommended difficulty, and small weakness
        bonuses could flip the choice to an unexpectedly hard arm.
        """
        stats = self.topic_accuracy.get(topic)
        if stats and stats["total"] > 0:
            baseline = stats["correct"] / stats["total"]
        else:
            baseline = 0.5
        expected = baseline - 0.2 * (difficulty - 3)
        return max(0.0, min(1.0, expected))

    def _stratified_explore(
        self,
        available_topics: list[str],
        min_diff: int,
        max_diff: int,
    ) -> tuple[str, int]:
        """
        Epsilon-random branch with two refinements over uniform sampling:

        1. Topic pick is weighted by ``1 + weakness_bonus`` so a
           struggling topic is explored more often than an on-track one.
        2. Difficulty pick is biased toward the student's current
           ability band via ``recommend_difficulty`` — uniform ±1 around
           it. This stops a totally random difficulty=5 landing on a
           topic the student has never seen, which the old code did
           with probability 1/(max_diff - min_diff + 1).

        Falls back to uniform sampling when we have no data — a fresh
        session still gets genuine exploration.
        """
        # Weight by 1 + weakness_bonus so weak topics get extra mass but
        # no topic goes to zero weight (everyone keeps a chance).
        weights = [1.0 + self._weakness_bonus(t) for t in available_topics]
        total_weight = sum(weights)
        if total_weight <= 0:
            topic = self._choice(available_topics)
        else:
            r = self._rand() * total_weight
            acc = 0.0
            topic = available_topics[-1]
            for cand, w in zip(available_topics, weights):
                acc += w
                if r <= acc:
                    topic = cand
                    break

        # Bias difficulty toward the recommended level for this topic.
        recommended = self.recommend_difficulty(topic)
        lo = max(min_diff, recommended - 1)
        hi = min(max_diff, recommended + 1)
        if lo > hi:
            lo, hi = min_diff, max_diff
        difficulty = self._randint(lo, hi)
        return topic, difficulty

    def _weakness_bonus(self, topic: str) -> float:
        """Nudge selection toward topics the student struggles on."""
        stats = self.topic_accuracy.get(topic)
        if not stats or stats["total"] == 0:
            return 0.0
        accuracy = stats["correct"] / stats["total"]
        return 1.0 - accuracy

    def update(
        self,
        topic: str,
        difficulty: int,
        correct: bool,
        confidence: int = 3,
        *,
        _decay: bool = True,
    ) -> None:
        """
        Update the engine after a quiz question is answered.

        Args:
            topic: The topic that was quizzed
            difficulty: The difficulty level used
            correct: Whether the student answered correctly
            confidence: Student's self-rated confidence (1-5)
            _decay: Internal flag -- set to ``False`` during history
                    replay so epsilon is not decayed once per entry
                    (see LA-5).

        The reward signal approximates "observed mastery" in [0, 1]:
          - Correct + high confidence  -> ~1.0 (true mastery)
          - Correct + low confidence   -> ~0.6 (shaky but right)
          - Wrong   + low confidence   -> ~0.3 (known gap)
          - Wrong   + high confidence  -> ~0.0 (misconception -- NOT rewarded;
            the selection function handles this via the weakness bonus, not
            by pumping this arm's reward)

        ``confidence`` and ``difficulty`` are CLAMPED to [1, 5] before any
        math runs. The Streamlit UI bounds them with sliders, but the
        method is also called by tests, simulations, the sample profile generator,
        and any future programmatic caller. A bad input (``confidence=10``)
        would have generated rewards outside [0, 1] and silently broken
        the bandit's ZPD math; clamping at the boundary is the cheapest
        defence.
        """
        # Coerce + clamp at the public boundary. ``int(...)`` raises
        # TypeError on truly garbage input — that's preferable to
        # silently casting None/'foo' to 0.
        try:
            confidence = max(1, min(5, int(confidence)))
        except (TypeError, ValueError):
            confidence = 3
        try:
            difficulty = max(1, min(5, int(difficulty)))
        except (TypeError, ValueError):
            difficulty = 2

        confidence_factor = confidence / 5.0  # Normalize to 0-1

        if correct:
            # 0.6 (low conf) -> 1.0 (high conf)
            reward = 0.6 + (0.4 * confidence_factor)
        else:
            # 0.3 (low conf) -> 0.0 (high conf, misconception)
            reward = 0.3 * (1.0 - confidence_factor)

        # Update arm stats
        arm = self._get_or_create_arm(topic, difficulty)
        arm["total_reward"] += reward
        arm["count"] += 1

        # Update topic accuracy tracker
        if topic not in self.topic_accuracy:
            self.topic_accuracy[topic] = {"correct": 0, "total": 0}
        self.topic_accuracy[topic]["total"] += 1
        if correct:
            self.topic_accuracy[topic]["correct"] += 1

        # Decay epsilon (less exploration over time)
        if _decay:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def get_topic_mastery(self) -> dict[str, float]:
        """
        Get estimated mastery level (0-100%) for each topic.
        Useful for displaying progress to the student.
        """
        mastery: dict[str, float] = {}
        for topic, stats in self.topic_accuracy.items():
            if stats["total"] > 0:
                mastery[topic] = round((stats["correct"] / stats["total"]) * 100, 1)
            else:
                mastery[topic] = 0.0
        return mastery

    def recommend_difficulty(self, topic: str) -> int:
        """
        Recommend an appropriate difficulty for a given topic
        based on the student's history.
        """
        if topic not in self.topic_accuracy:
            return 2  # Default: medium-easy for new topics

        stats = self.topic_accuracy[topic]
        if stats["total"] == 0:
            return 2

        accuracy = stats["correct"] / stats["total"]

        # Scale difficulty based on accuracy
        if accuracy >= 0.8:
            return 4  # Student is doing well -> increase
        elif accuracy >= 0.6:
            return 3  # Solid -> stay moderate
        elif accuracy >= 0.4:
            return 2  # Struggling -> keep easier
        else:
            return 1  # Really struggling -> go easy

    def explain_recommendation(
        self,
        topic: str,
        difficulty: int,
        *,
        is_review: bool = False,
    ) -> dict:
        """Return a student-facing explanation for a topic/difficulty choice.

        This is intentionally read-only: it does not change the bandit's
        selection policy, reward state, or exploration rate. The Streamlit UI
        uses it to make the adaptive choice visible in the UI.
        """
        topic = str(topic or "this topic")
        try:
            difficulty = max(1, min(5, int(difficulty)))
        except (TypeError, ValueError):
            difficulty = 2

        stats = self.topic_accuracy.get(topic, {}) or {}
        total = int(stats.get("total", 0) or 0)
        correct = int(stats.get("correct", 0) or 0)
        mastery = round((correct / total) * 100, 1) if total else None
        recommended = self.recommend_difficulty(topic)
        arm = self.arm_stats.get((topic, difficulty), {}) or {}
        arm_count = int(arm.get("count", 0) or 0)

        if is_review:
            reason = "Spaced review due"
            detail = (
                "This topic is scheduled for review, so the tutor is mixing "
                "retention practice into the adaptive quiz."
            )
            policy = "spaced_repetition"
        elif total == 0:
            reason = "New topic baseline"
            detail = (
                "There is no prior quiz history for this topic yet, so the "
                "tutor starts at a medium-easy difficulty to gather signal."
            )
            policy = "cold_start"
        elif mastery is not None and mastery < 40:
            reason = "Weak topic recovery"
            detail = (
                "Recent accuracy is low, so the tutor keeps the difficulty "
                "easier and gives the student a chance to rebuild the basics."
            )
            policy = "weakness_targeting"
        elif mastery is not None and mastery >= 80 and difficulty >= recommended:
            reason = "Mastery challenge"
            detail = (
                "Accuracy is strong, so the tutor can safely raise or maintain "
                "challenge instead of repeating easy items."
            )
            policy = "mastery_scaling"
        elif difficulty > recommended:
            reason = "Exploration check"
            detail = (
                "The bandit is sampling a harder arm to learn whether the "
                "student is ready to move up."
            )
            policy = "exploration"
        elif difficulty < recommended:
            reason = "Confidence rebuild"
            detail = (
                "The tutor is using a slightly easier question to confirm the "
                "foundation before increasing difficulty."
            )
            policy = "confidence_rebuild"
        else:
            reason = "ZPD practice"
            detail = (
                "The difficulty matches the current estimate for productive "
                "practice: not too easy, not too hard."
            )
            policy = "zpd_targeting"

        return {
            "topic": topic,
            "difficulty": difficulty,
            "reason": reason,
            "detail": detail,
            "policy": policy,
            "mastery": mastery,
            "attempts": total,
            "correct": correct,
            "recommended_difficulty": recommended,
            "arm_attempts": arm_count,
        }
