"""
Resource Agent -- companion agent in the multi-agent system.

This agent works alongside the Tutor Agent to improve decision-making.
The two agents communicate through the MessageBus:

FLOW 1: Enriched explanations
  1. Student answers a quiz question wrong
  2. Tutor Agent sends REQUEST_MATERIALS to Resource Agent
  3. Resource Agent generates targeted materials (explanations, analogies, examples)
  4. Resource Agent sends PROVIDE_MATERIALS back
  5. Tutor Agent incorporates materials into its feedback

FLOW 2: Strategy negotiation
  1. Tutor Agent sends REPORT_WEAKNESS when a student is struggling
  2. Resource Agent analyzes the weakness pattern and identifies prerequisites
  3. Resource Agent sends SUGGEST_STRATEGY with a teaching plan
  4. Tutor Agent modifies its behavior (difficulty, topic selection, question format)

FLOW 3: Knowledge base building
  - Resource Agent accumulates a local knowledge base of topic summaries
  - When the same topic is requested again, it returns cached + fresh materials
  - This simulates "learning" -- the agent gets faster over time
"""

import itertools
import json
import logging

logger = logging.getLogger(__name__)

# An explicit ``None`` means offline mode. Omitting the argument keeps the
# convenient default-client behavior for scripts and direct agent use.
_DEFAULT_CLIENT = object()


def _kb_key(topic: str) -> str:
    """Canonicalise knowledge-base keys via the shared helper.

    Delegates to ``agents._prerequisites.canonical_topic_id`` so the KB,
    the local prerequisite graph, and the weakness-pattern detector all
    agree on topic identity (lowercased, whitespace-collapsed, trimmed).
    """
    return canonical_topic_id(topic)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Shared LLM config + SDK import (with graceful fallback stubs when
# anthropic isn't installed). The MODEL constant is the single source
# of truth for the Claude model ID used by every call site below.
from agents._llm import (
    MODEL,
    Anthropic,
    APIError,
    APIConnectionError,
    APITimeoutError,
    extract_json,
    get_default_client,
    llm_error_handler,
    retry_llm_call,
)
from agents._prerequisites import (
    canonical_topic_id,
    lookup_local_prerequisites,
    merge_prerequisites,
)
from agents._prompt_safety import safe_label, safe_freeform
from config import RESOURCE_AGENT_EAGER_TARGETED_EXPLANATION as _EAGER_TARGETED
from utils.telemetry import (
    COUNTERS,
    LLM_CALLS,
    MATERIAL_CACHE_HIT,
    MATERIAL_CACHE_MISS,
    MATERIAL_FALLBACK_USED,
    MATERIAL_GENERATED,
    MATERIAL_SCHEMA_REJECTED,
    PREREQ_LLM_FALLBACK,
    PREREQ_LOCAL_HIT,
)

from utils.agent_comm import MessageBus, AgentMessage, MessageType
from config import MAX_KNOWLEDGE_BASE_SIZE, MAX_WEAKNESS_HISTORY


class ResourceAgent:
    """
    The Resource Agent finds and provides study materials to help
    the Tutor Agent give better explanations.

    It maintains a growing knowledge base and can detect prerequisite gaps.
    """

    AGENT_NAME = "ResourceAgent"

    # Cap the in-memory weakness history so it doesn't grow unbounded
    # over a long Streamlit session. The pattern detector only looks at
    # recent reports anyway. Backed by config.MAX_WEAKNESS_HISTORY so a
    # test can monkey-patch the class attribute (existing tests already
    # reach into ResourceAgent._MAX_KNOWLEDGE_BASE_SIZE this way).
    _MAX_WEAKNESS_HISTORY = MAX_WEAKNESS_HISTORY

    # Cap the knowledge base so the serialised JSON profile doesn't grow
    # to several megabytes and slow down the UI. When exceeded, the
    # least-requested topics are evicted.
    _MAX_KNOWLEDGE_BASE_SIZE = MAX_KNOWLEDGE_BASE_SIZE

    def __init__(
        self,
        message_bus: MessageBus,
        client: "Anthropic | None | object" = _DEFAULT_CLIENT,
    ):
        """
        Args:
            message_bus: Shared MessageBus instance.
            client: Optional pre-built Anthropic client to share with
                    the Tutor Agent. If omitted, the agent constructs its
                    own. Pass ``None`` explicitly to force local fallbacks.
        """
        self.bus = message_bus
        self.client = (
            get_default_client() if client is _DEFAULT_CLIENT else client
        )

        # Knowledge base: grows as topics are researched.
        # Keyed by `_kb_key(topic)` (lowercased) so case variants collapse
        # to one row. Each value carries the display-cased topic under
        # "display_name" for UI/diagnostics.
        # {key: {"display_name": str, "materials": [...], "prerequisites": [...],
        #        "times_requested": int, "insertion_order": int,
        #        "last_request": {...}}}
        self.knowledge_base: dict[str, dict] = {}

        # Monotonic counter used as a tiebreaker when evicting by
        # times_requested — lowest insertion_order loses first, so the
        # oldest entry of the least-requested pool is evicted, not a
        # shiny-new one that happens to share the same request count.
        self._insertion_counter = itertools.count()

        # Track weakness reports to detect patterns across topics
        self.weakness_history: list[dict] = []

        # Register on the message bus (also stores `self` so other agents
        # can look us up by name to read serializable state for persistence).
        self.bus.register_agent(self.AGENT_NAME, callback=self._on_message, instance=self)

    def _on_message(self, message: AgentMessage) -> None:
        """Handle incoming messages from the Tutor Agent."""
        if message.msg_type == MessageType.REQUEST_MATERIALS:
            self._handle_material_request(message)
        elif message.msg_type == MessageType.REPORT_WEAKNESS:
            self._handle_weakness_report(message)

    # -----------------------------------------------------------------
    # Flow 1: Material requests
    # -----------------------------------------------------------------

    @staticmethod
    def _material_request_rationale(
        topic: str,
        course: str,
        student_level: str,
        context: str,
        learning_style: str,
        from_cache: bool,
    ) -> dict:
        """Build student-facing metadata explaining why material exists."""
        safe_topic = safe_label(topic, limit=80) or "this topic"
        safe_course = safe_label(course, limit=80) or "this course"
        safe_level = safe_label(student_level, limit=30) or "current"
        safe_style = safe_label(learning_style, limit=30) or "balanced"
        safe_context = safe_freeform(context, limit=240).strip()
        if not safe_context:
            safe_context = (
                f"A quiz answer on {safe_topic} was marked incorrect, so the "
                "student needs targeted review material."
            )

        storage = (
            "reused from the ResourceAgent material cache"
            if from_cache else
            "generated by the ResourceAgent and saved in its material cache"
        )
        why_shown = (
            f"This material is shown because the latest quiz answer on "
            f"{safe_topic} was wrong, making it a review priority in "
            f"{safe_course}."
        )
        generation_source = (
            f"{storage} after evaluating the quiz answer; tailored for a "
            f"{safe_level} student with a {safe_style} learning preference."
        )
        return {
            "requested_by": "TutorAgent",
            "produced_by": "ResourceAgent",
            "trigger": "wrong quiz answer",
            "why_shown": why_shown,
            # Backwards-compatible key used by older UI/tests.
            "why_tutor_requested": why_shown,
            "quiz_evidence": safe_context,
            "student_need": safe_context,
            "generation_source": generation_source,
            "source": generation_source,
        }

    def _handle_material_request(self, message: AgentMessage) -> None:
        """
        Respond to a material request from the Tutor Agent.

        If the topic is in the knowledge base, return cached materials
        plus a fresh explanation tailored to the specific context.
        Otherwise, generate everything from scratch and cache it.
        """
        topic = message.content.get("topic", "")
        course = message.content.get("course", "")
        student_level = message.content.get("student_level", "beginner")
        context = message.content.get("context", "")
        learning_style = message.content.get("learning_style", "balanced")

        # Guard: a malformed REQUEST_MATERIALS with no topic would collide
        # under an empty-string KB key, bucketing unrelated requests into
        # the same cached materials row. Drop the request loudly instead.
        if not str(topic).strip():
            logger.warning(
                "REQUEST_MATERIALS from %s has empty topic — ignoring",
                message.sender,
            )
            return

        # Use the canonical lowercased key for cache lookup AND write
        # so "Recursion" and "recursion" collide on the same row.
        key = _kb_key(topic)
        cached = self.knowledge_base.get(key, None)
        from_cache = cached is not None
        rationale = self._material_request_rationale(
            topic, course, student_level, context, learning_style, from_cache
        )

        if cached:
            # We've seen this topic before -- use cached base. Default
            # behaviour: SKIP the per-turn fresh-tailored-material LLM
            # call (saves 1 LLM call per cache hit). The eager-targeted
            # path stays available behind
            # ``config.RESOURCE_AGENT_EAGER_TARGETED_EXPLANATION``.
            COUNTERS.incr(MATERIAL_CACHE_HIT)
            cached["times_requested"] += 1
            # Refresh insertion_order so eviction treats this as recently-used.
            cached["insertion_order"] = next(self._insertion_counter)
            cached["last_request"] = rationale
            materials = cached["materials"]

            if _EAGER_TARGETED and context:
                # Generate a FRESH explanation tailored to the specific context.
                fresh = self._generate_targeted_explanation(
                    topic, course, student_level, context, learning_style
                )
                if fresh:
                    materials = [fresh] + materials[:2]  # Fresh first, then cached
        else:
            # New topic -- generate full materials and cache.
            # ``identify_prerequisites`` is the LAYERED public API:
            # local-graph first (deterministic), LLM fill-in only for
            # topics the local graph doesn't cover.
            COUNTERS.incr(MATERIAL_CACHE_MISS)
            materials = self._generate_materials(topic, course, student_level, learning_style)
            COUNTERS.incr(MATERIAL_GENERATED)
            prerequisites = self.identify_prerequisites(topic, course)

            self.knowledge_base[key] = {
                "display_name": topic,
                "materials": materials,
                "prerequisites": prerequisites,
                "times_requested": 1,
                "insertion_order": next(self._insertion_counter),
                "last_request": rationale,
            }
            # Evict least-requested entries if the KB exceeds the cap.
            self._evict_knowledge_base()

        # Send materials back to the Tutor Agent
        response = AgentMessage(
            sender=self.AGENT_NAME,
            receiver=message.sender,
            msg_type=MessageType.PROVIDE_MATERIALS,
            content={
                "topic": topic,
                "materials": materials,
                "prerequisites": self.knowledge_base.get(key, {}).get("prerequisites", []),
                "from_cache": from_cache,
                "rationale": rationale,
                "request_id": message.id,
            },
            priority=message.priority,
        )
        self.bus.send(response)

    # ------------------------------------------------------------------
    # LLM output schema validation
    # ------------------------------------------------------------------
    # Materials are injected into TutorAgent.feedback_prompt as system
    # context. A malformed material — e.g. a list of strings, a dict
    # missing ``explanation`` — would either crash ``_on_message`` (which
    # does ``materials[0].get(...)``) or land empty fields in the prompt.
    # We mirror ``TutorAgent._validate_quiz_shape``: schema failures
    # raise ``json.JSONDecodeError``, which is in the retry decorator's
    # retryable set, so a bad sample re-rolls instead of being canonised.

    _MATERIAL_REQUIRED_FIELDS = ("explanation", "analogy", "common_mistake")
    _MATERIAL_OPTIONAL_FIELDS = ("title", "self_test")

    @staticmethod
    def _validate_material_shape(material: object) -> dict:
        """Validate one material item; return a defaulted dict on success."""
        if not isinstance(material, dict):
            raise json.JSONDecodeError(
                f"Material must be a dict, got {type(material).__name__}",
                str(material)[:200], 0,
            )
        missing = [f for f in ResourceAgent._MATERIAL_REQUIRED_FIELDS
                   if not isinstance(material.get(f), str) or not material.get(f, "").strip()]
        if missing:
            raise json.JSONDecodeError(
                f"Material missing/empty required fields: {missing}",
                str(material)[:200], 0,
            )
        # Normalise: provide default empty strings for optional fields so
        # downstream `.get()` callers always see a string.
        normalised = {f: str(material.get(f, "")).strip()
                      for f in (*ResourceAgent._MATERIAL_REQUIRED_FIELDS,
                                *ResourceAgent._MATERIAL_OPTIONAL_FIELDS)}
        return normalised

    @staticmethod
    def _validate_materials_list(materials: object) -> list[dict]:
        """Validate the LLM-emitted list of materials. Reject empty lists
        and any item that fails ``_validate_material_shape``."""
        if not isinstance(materials, list):
            COUNTERS.incr(MATERIAL_SCHEMA_REJECTED)
            raise json.JSONDecodeError(
                f"Materials must be a list, got {type(materials).__name__}",
                str(materials)[:200], 0,
            )
        if len(materials) == 0:
            COUNTERS.incr(MATERIAL_SCHEMA_REJECTED)
            raise json.JSONDecodeError(
                "Materials list is empty", str(materials)[:200], 0,
            )
        return [ResourceAgent._validate_material_shape(m) for m in materials]

    @staticmethod
    def _validate_prereqs_list(prereqs: object) -> list[str]:
        """Validate the LLM-emitted prerequisites list — must be a list
        of non-empty strings. Return a cleaned, deduped, length-limited
        list. Schema failure raises JSONDecodeError so retry kicks in."""
        if not isinstance(prereqs, list):
            raise json.JSONDecodeError(
                f"Prerequisites must be a list, got {type(prereqs).__name__}",
                str(prereqs)[:200], 0,
            )
        cleaned: list[str] = []
        seen: set[str] = set()
        for p in prereqs:
            if not isinstance(p, str):
                continue
            stripped = p.strip()
            if not stripped:
                continue
            key = stripped.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(stripped[:80])  # also bound length per item
            if len(cleaned) >= 5:
                break
        return cleaned

    # -- Decorated LLM helpers for Flow 1 --

    @staticmethod
    def _fallback_materials(
        self: "ResourceAgent", topic: str, course: str,
        student_level: str, learning_style: str,
    ) -> list[dict]:
        return [self._fallback_material(topic, course)]

    @llm_error_handler(
        fallback=_fallback_materials.__func__,  # type: ignore[attr-defined]
        error_label="generate_materials",
    )
    @retry_llm_call()
    def _generate_materials(self, topic: str, course: str,
                            student_level: str, learning_style: str) -> list[dict]:
        """Generate study materials using LLM. Cached for reuse."""
        if not self.client:
            return [self._fallback_material(topic, course)]

        style_instructions = {
            "concise": "Keep each explanation under 3 sentences. Be direct.",
            "detailed": "Give thorough step-by-step explanations with worked examples.",
            "visual": "Use vivid analogies, metaphors, and visual descriptions throughout.",
            "balanced": "Mix concise definitions with one illustrative example each.",
        }
        safe_style_key = safe_label(learning_style, limit=20)
        style_guide = style_instructions.get(safe_style_key, style_instructions["balanced"])

        # Sanitise every field interpolated into the LLM prompt. Topics and
        # courses can be user-typed; student_level is derived but still
        # passes through this boundary defensively.
        safe_topic = safe_label(topic, limit=80)
        safe_course = safe_label(course, limit=80)
        safe_student_level = safe_label(student_level, limit=20)

        prompt = f"""Generate 3 study materials for a {safe_student_level}-level student
studying "{safe_topic}" in the course "{safe_course}".

Style: {style_guide}

For each material, provide:
1. A clear explanation appropriate for the student's level
2. A memorable analogy or real-world example
3. A common mistake students make on this topic
4. A self-test question to check understanding

Respond in this JSON format ONLY:
[
  {{
    "title": "Short descriptive title",
    "explanation": "Clear explanation...",
    "analogy": "Think of it like...",
    "common_mistake": "Students often confuse...",
    "self_test": "Can you explain why..."
  }}
]"""

        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1200,
            system="You are a study material generator. Respond ONLY with valid JSON, no other text.",
            messages=[{"role": "user", "content": prompt}],
        )
        materials = extract_json(response.content[0].text)
        # Schema validation raises JSONDecodeError on malformed output,
        # which the @retry_llm_call decorator will retry. After
        # validation, every material is guaranteed to have non-empty
        # ``explanation`` / ``analogy`` / ``common_mistake`` strings.
        return self._validate_materials_list(materials)

    @staticmethod
    def _fallback_targeted(
        self: "ResourceAgent", topic: str, course: str,
        student_level: str, context: str, learning_style: str,
    ) -> None:
        return None

    @llm_error_handler(
        fallback=_fallback_targeted.__func__,  # type: ignore[attr-defined]
        error_label="generate_targeted_explanation",
    )
    @retry_llm_call()
    def _generate_targeted_explanation(self, topic: str, course: str,
                                       student_level: str, context: str,
                                       learning_style: str) -> dict | None:
        """Generate a context-specific explanation for a particular quiz miss.

        ``context`` is the most dangerous interpolation in the system —
        it contains the student's exact wrong answer and the original
        question text, which are produced by the LLM upstream. Run it
        through ``safe_freeform`` so a poisoned question can't smuggle
        a section divider or code-fence escape across the boundary.
        """
        if not context or not self.client:
            return None

        safe_topic = safe_label(topic, limit=80)
        safe_course = safe_label(course, limit=80)
        safe_student_level = safe_label(student_level, limit=20)
        safe_style = safe_label(learning_style, limit=20)
        safe_context = safe_freeform(context, limit=1500)

        prompt = f"""A student just got a question wrong about "{safe_topic}" in {safe_course}.

Context: {safe_context}

The student's level is {safe_student_level} and they prefer {safe_style} explanations.

Generate ONE targeted explanation that directly addresses what they got wrong.
Include an analogy and a tip for remembering.

Respond in JSON:
{{
    "title": "Targeted: [specific concept]",
    "explanation": "The key thing to understand is...",
    "analogy": "Think of it like...",
    "common_mistake": "You might have confused...",
    "self_test": "Now try this: ..."
}}"""

        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=500,
            system="You are a study material generator. Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        targeted = extract_json(response.content[0].text)
        # Single-material validation. Returns a fully-normalised dict
        # so TutorAgent._on_message's ``mat.get(...)`` calls always
        # land on a string field.
        return self._validate_material_shape(targeted)

    @staticmethod
    def _fallback_prereqs(self: "ResourceAgent", topic: str, course: str) -> list[str]:
        # Even when the LLM call fails entirely, return the locally-known
        # prereqs (if any) — they're hand-curated and reliable.
        return lookup_local_prerequisites(topic)

    def identify_prerequisites(self, topic: str, course: str) -> list[str]:
        """Public API: layered prerequisite resolution.

          1. Look up locally-curated prereqs from
             ``agents._prerequisites.LOCAL_PREREQUISITES``. Stable
             across runs, no LLM cost.
          2. If no local entry exists OR we want to enrich the local
             list with course-specific suggestions, ask the LLM via
             ``_identify_prerequisites``.
          3. Merge the two lists (local first, LLM dedupes against
             local) and cap to 5 entries.

        This layering is the answer to "LLM-suggested prereqs are
        volatile" — well-known topics get deterministic answers; novel
        topics still get LLM suggestions; the merge step preserves
        locally-known reliability when both sources fire.
        """
        local = lookup_local_prerequisites(topic)
        # Skip the LLM call entirely when we have a full local entry —
        # deterministic + zero cost. The 3-prereq threshold matches the
        # local graph's typical entry size.
        if len(local) >= 3:
            COUNTERS.incr(PREREQ_LOCAL_HIT)
            return local
        try:
            llm_suggested = self._identify_prerequisites(topic, course)
            if llm_suggested:
                COUNTERS.incr(PREREQ_LLM_FALLBACK)
        except Exception:  # noqa: BLE001
            llm_suggested = []
        return merge_prerequisites(local, llm_suggested or [])

    @llm_error_handler(
        fallback=_fallback_prereqs.__func__,  # type: ignore[attr-defined]
        error_label="identify_prerequisites",
    )
    @retry_llm_call()
    def _identify_prerequisites(self, topic: str, course: str) -> list[str]:
        """Use LLM to identify prerequisite topics the student should know first.

        Internal helper — call ``identify_prerequisites`` (no leading
        underscore) for the layered local-then-LLM resolution.
        """
        if not self.client:
            return []

        safe_topic = safe_label(topic, limit=80)
        safe_course = safe_label(course, limit=80)
        prompt = f"""What are the 2-3 prerequisite topics a student must understand
BEFORE they can learn "{safe_topic}" in the course "{safe_course}"?

Respond with a JSON array of strings ONLY:
["prerequisite 1", "prerequisite 2"]"""

        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=150,
            system="Respond only with a JSON array of strings.",
            messages=[{"role": "user", "content": prompt}],
        )
        prereqs = extract_json(response.content[0].text)
        # Schema validation: list of non-empty strings, deduped on
        # canonical-lower, capped to 5 entries × 80 chars each. Bad
        # samples re-roll via the retry decorator's JSONDecodeError path.
        return self._validate_prereqs_list(prereqs)

    # -----------------------------------------------------------------
    # Flow 2: Weakness reports & strategy negotiation
    # -----------------------------------------------------------------

    def _handle_weakness_report(self, message: AgentMessage) -> None:
        """
        Analyze a weakness report and suggest a teaching strategy.
        Considers the full history of weakness reports to detect patterns.
        """
        topic = message.content.get("topic", "")
        course = message.content.get("course", "")
        accuracy = message.content.get("accuracy", 0)
        attempts = message.content.get("attempts", 0)

        # Track this report (with a tail cap so it can't grow unbounded)
        self.weakness_history.append({
            "topic": topic,
            "course": course,
            "accuracy": accuracy,
            "attempts": attempts,
        })
        if len(self.weakness_history) > self._MAX_WEAKNESS_HISTORY:
            self.weakness_history = self.weakness_history[-self._MAX_WEAKNESS_HISTORY:]

        # Check for prerequisite gaps — use the canonical KB key.
        prereqs = self.knowledge_base.get(_kb_key(topic), {}).get("prerequisites", [])
        weak_prereqs = self._find_weak_prerequisites(prereqs, course)

        # Determine strategy
        strategy, suggestion = self._compute_strategy(
            topic, accuracy, attempts, weak_prereqs
        )

        response = AgentMessage(
            sender=self.AGENT_NAME,
            receiver=message.sender,
            msg_type=MessageType.SUGGEST_STRATEGY,
            content={
                "topic": topic,
                "strategy": strategy,
                "suggestion": suggestion,
                "prerequisites": prereqs,
                "weak_prerequisites": weak_prereqs,
                "based_on": {"accuracy": accuracy, "attempts": attempts},
                "pattern_detected": self._detect_weakness_pattern(),
            },
            priority=3,
        )
        self.bus.send(response)

    def _compute_strategy(self, topic: str, accuracy: float,
                          attempts: int, weak_prereqs: list[str]) -> tuple[str, str]:
        """Determine the best teaching strategy based on the student's situation."""

        # Priority 1: Prerequisite gaps detected
        if weak_prereqs:
            return "prerequisite_review", (
                f"The student appears to be missing prerequisite knowledge. "
                f"Before continuing with '{topic}', review these foundational topics first: "
                f"{', '.join(weak_prereqs)}. Once those are solid, '{topic}' will make more sense."
            )

        # Priority 2: Severe struggle -- complete blackout
        if accuracy < 0.15 and attempts >= 5:
            return "restart_from_basics", (
                f"The student is really struggling with '{topic}' ({accuracy*100:.0f}% over "
                f"{attempts} attempts). Start over with the most basic definition. "
                f"Use a concrete, real-world analogy. Ask them to explain it back in their own words "
                f"before attempting any quiz questions."
            )

        # Priority 3: Moderate struggle
        if accuracy < 0.4:
            return "simplify_and_scaffold", (
                f"Break '{topic}' into smaller sub-concepts and quiz on each separately. "
                f"Start with the simplest component. Use scaffolded questions: "
                f"first fill-in-the-blank, then multiple choice, then open-ended. "
                f"Provide hints before each question."
            )

        # Priority 4: Plateau -- not improving
        if attempts >= 8 and accuracy < 0.6:
            return "change_approach", (
                f"The student has plateaued on '{topic}' at {accuracy*100:.0f}%. "
                f"Try a completely different approach: instead of multiple-choice, "
                f"ask them to explain the concept, compare/contrast with related topics, "
                f"or apply it to a specific scenario."
            )

        # Priority 5: Mild struggle -- just needs practice
        return "varied_practice", (
            f"The student is making progress on '{topic}' but needs more practice. "
            f"Mix different question types and gradually increase difficulty. "
            f"Emphasize the 'why' in explanations, not just the 'what'."
        )

    def _find_weak_prerequisites(self, prereqs: list[str], course: str) -> list[str]:
        """Check if any prerequisite topics have low accuracy in the knowledge base."""
        weak: list[str] = []
        for prereq in prereqs:
            # If we've seen weakness reports for this prerequisite
            for report in self.weakness_history:
                if report["topic"].lower() == prereq.lower() and report["accuracy"] < 0.5:
                    weak.append(prereq)
                    break
        return weak

    def _detect_weakness_pattern(self) -> str | None:
        """Look for patterns across weakness reports."""
        if len(self.weakness_history) < 3:
            return None

        # Check if many DISTINCT topics in the same course are weak.
        # Counting raw reports double-counts a single repeatedly-weak
        # topic as "multiple weak topics", which is factually wrong.
        # Use a set of unique topic names per course instead.
        course_weakness: dict[str, set[str]] = {}
        for report in self.weakness_history:
            c = report.get("course", "unknown")
            t = str(report.get("topic", "")).strip().lower()
            if not t:
                continue
            course_weakness.setdefault(c, set()).add(t)

        for course, topics in course_weakness.items():
            if len(topics) >= 3:
                return (f"Multiple weak topics detected in {course} ({len(topics)} topics). "
                        f"The student may need to revisit foundational material for this course.")

        # Check if accuracy is consistently very low
        recent = self.weakness_history[-5:]
        avg_accuracy = sum(r["accuracy"] for r in recent) / len(recent)
        if avg_accuracy < 0.25:
            return ("Very low accuracy across recent topics. Consider reducing overall "
                    "difficulty and focusing on building confidence.")

        return None

    def _evict_knowledge_base(self) -> None:
        """Evict least-requested entries when the KB exceeds the size cap.

        O8: tie-break on ``insertion_order`` so that among equally-rarely
        requested entries, the OLDEST is evicted — which is LRU-ish
        semantics rather than "whichever happened to sort first". Under
        the old code, insertion order was the tiebreaker implicitly via
        sort stability, which meant a brand-new (just-inserted) topic
        could be evicted in favor of an equally-unused topic inserted 5
        minutes earlier — the reverse of what you want.
        """
        if len(self.knowledge_base) <= self._MAX_KNOWLEDGE_BASE_SIZE:
            return
        sorted_topics = sorted(
            self.knowledge_base.items(),
            key=lambda kv: (
                kv[1].get("times_requested", 0),
                kv[1].get("insertion_order", 0),
            ),
        )
        to_remove = len(self.knowledge_base) - self._MAX_KNOWLEDGE_BASE_SIZE
        for topic_key, _ in sorted_topics[:to_remove]:
            del self.knowledge_base[topic_key]

    def _fallback_material(self, topic: str, course: str) -> dict:
        """Fallback material when LLM fails. Must satisfy
        ``_validate_material_shape`` — every required field is a
        non-empty string."""
        safe_topic = safe_label(topic, limit=80) or "this topic"
        safe_course = safe_label(course, limit=80) or "this course"
        return {
            "title": f"Overview of {safe_topic}",
            "explanation": (
                f"Key concepts in {safe_topic} for {safe_course}. "
                "(This is a fallback explanation: the AI service is "
                "temporarily unavailable; please retry for a richer one.)"
            ),
            "analogy": "Think about the fundamental principles involved and how they connect to what you already know.",
            "common_mistake": "Make sure you understand the basic definition first before tackling harder examples.",
            "self_test": f"Can you explain {safe_topic} in your own words?",
        }

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize the knowledge base + weakness history so the agent's
        "learning" can ride along inside the student profile and survive
        Streamlit restarts.
        """
        return {
            "knowledge_base": self.knowledge_base,
            "weakness_history": self.weakness_history[-self._MAX_WEAKNESS_HISTORY:],
        }

    def load_state(self, data: dict) -> None:
        """Restore from a previously-persisted dict (no-op if data is empty).

        Legacy profiles may store keys in mixed case (before L2 fix) and
        without insertion_order. Migrate on load so the in-memory shape
        is consistent.
        """
        if not data:
            return
        raw_kb = data.get("knowledge_base", {}) or {}
        migrated: dict[str, dict] = {}
        for raw_key, entry in raw_kb.items():
            key = _kb_key(raw_key)
            if not isinstance(entry, dict):
                continue
            # If a case-variant is already in migrated, merge counts so
            # legacy duplicates don't double-count or lose requests.
            if key in migrated:
                existing = migrated[key]
                existing["times_requested"] = (
                    existing.get("times_requested", 0)
                    + int(entry.get("times_requested", 0))
                )
                continue
            migrated[key] = {
                "display_name": entry.get("display_name", raw_key),
                "materials": entry.get("materials", []),
                "prerequisites": entry.get("prerequisites", []),
                "times_requested": int(entry.get("times_requested", 0)),
                "insertion_order": int(
                    entry.get("insertion_order", next(self._insertion_counter))
                ),
                "last_request": entry.get("last_request", {}) or {},
            }
        self.knowledge_base = migrated
        self.weakness_history = data.get("weakness_history", []) or []
        if len(self.weakness_history) > self._MAX_WEAKNESS_HISTORY:
            self.weakness_history = self.weakness_history[-self._MAX_WEAKNESS_HISTORY:]

    # -----------------------------------------------------------------
    # Public API for diagnostics
    # -----------------------------------------------------------------

    def get_material_cache_stats(self) -> dict:
        """Get stats about the cached LLM material for diagnostics.

        Note: previously named ``get_knowledge_base_stats``. The
        rename reflects what the data IS — cached LLM output, not
        verified knowledge — without changing the on-disk
        ``resource_agent_state.knowledge_base`` key (which would
        break old profiles). The legacy method name is preserved as
        an alias below so existing callers don't break.
        """
        return {
            "topics_researched": len(self.knowledge_base),
            "total_requests": sum(
                v.get("times_requested", 0) for v in self.knowledge_base.values()
            ),
            "topics_with_prerequisites": sum(
                1 for v in self.knowledge_base.values() if v.get("prerequisites")
            ),
            "weakness_reports_received": len(self.weakness_history),
            "topics": {
                # Prefer the display_name captured when the entry was
                # inserted so users don't see lowercased topic names in
                # Diagnostics.
                data.get("display_name", key): {
                    "times_requested": data.get("times_requested", 0),
                    "prerequisites": data.get("prerequisites", []),
                    "materials_count": len(data.get("materials", [])),
                    "last_request": data.get("last_request", {}) or {},
                }
                for key, data in self.knowledge_base.items()
            },
        }

    def suggest_materials_for_topics(
        self,
        topics: list[str],
        *,
        limit_per_topic: int = 1,
    ) -> list[dict]:
        """Return cached study material suggestions for the requested topics.

        The Resource Agent creates/caches materials when the Tutor Agent sends
        REQUEST_MATERIALS after a wrong answer. This method turns that cache
        into a small student-facing recommendation list for the quiz-session
        evaluation screen.
        """
        suggestions: list[dict] = []
        seen_topics: set[str] = set()

        for topic in topics:
            key = _kb_key(topic)
            if not key or key in seen_topics:
                continue
            seen_topics.add(key)

            entry = self.knowledge_base.get(key)
            if not isinstance(entry, dict):
                continue
            materials = entry.get("materials", [])
            if not isinstance(materials, list) or not materials:
                continue

            added = 0
            for material in materials:
                if not isinstance(material, dict):
                    continue
                rationale = dict(entry.get("last_request", {}) or {})
                safe_topic = safe_label(entry.get("display_name", topic), limit=80) or "this topic"
                if not rationale:
                    why_shown = (
                        f"This material is shown because {safe_topic} appeared "
                        "as a missed priority topic in the quiz evaluation."
                    )
                    quiz_evidence = (
                        "The saved quiz report marked this topic for review "
                        "after one or more wrong answers."
                    )
                    generation_source = (
                        "ResourceAgent material cache; this entry was created "
                        "before detailed quiz-evidence metadata was recorded."
                    )
                    rationale = {
                        "requested_by": "TutorAgent",
                        "produced_by": "ResourceAgent",
                        "trigger": "wrong quiz answer",
                        "why_shown": why_shown,
                        "why_tutor_requested": why_shown,
                        "quiz_evidence": quiz_evidence,
                        "student_need": quiz_evidence,
                        "generation_source": generation_source,
                        "source": generation_source,
                    }
                if not rationale.get("why_shown"):
                    rationale["why_shown"] = (
                        f"This material is shown because {safe_topic} appeared "
                        "as a missed priority topic in the quiz evaluation."
                    )
                    rationale["why_tutor_requested"] = rationale["why_shown"]
                if not rationale.get("quiz_evidence"):
                    rationale["quiz_evidence"] = (
                        rationale.get("student_need")
                        or "The saved quiz report marked this topic for review after one or more wrong answers."
                    )
                if not rationale.get("generation_source"):
                    rationale["generation_source"] = (
                        rationale.get("source")
                        or "ResourceAgent generated or reused cached material after evaluating quiz answers."
                    )
                suggestions.append({
                    "topic": entry.get("display_name", topic),
                    "title": str(material.get("title", f"Review {topic}")).strip(),
                    "explanation": str(material.get("explanation", "")).strip(),
                    "analogy": str(material.get("analogy", "")).strip(),
                    "common_mistake": str(material.get("common_mistake", "")).strip(),
                    "self_test": str(material.get("self_test", "")).strip(),
                    "prerequisites": list(entry.get("prerequisites", []) or []),
                    "rationale": rationale,
                })
                added += 1
                if added >= max(1, limit_per_topic):
                    break

        return suggestions

    # Backwards-compat alias. Old callers (existing app.py paths and
    # external eval/benchmark scripts) keep working unchanged.
    get_knowledge_base_stats = get_material_cache_stats
