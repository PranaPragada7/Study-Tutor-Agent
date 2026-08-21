"""
Tutor Agent -- the primary tutoring orchestrator.

Combines:
  - LLM (Anthropic Claude API) for natural conversation, quiz generation, and explanations
  - Adaptive Engine (RL bandit) for choosing what to study and at what difficulty
  - Spaced Repetition (SM-2) for scheduling when to revisit topics
  - Issue Detector for self-monitoring and self-correction
  - Learning Style adaptation for personalized explanations
  - Message Bus integration for multi-agent communication with the Resource Agent

This agent can:
  1. Chat with the student about their courses
  2. Generate a personalized study plan
  3. Run adaptive quizzes that get harder/easier based on performance
  4. Explain wrong answers in the student's preferred style
  5. Detect and self-correct when something isn't working
  6. Schedule spaced reviews so students retain what they learn
"""

import json
import logging
import random
import re
from datetime import datetime
from typing import Callable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Shared LLM config + SDK import (with graceful fallback stubs when
# anthropic isn't installed). The MODEL constant is the single source
# of truth for the Claude model ID used by every call site below.
import config
from agents._llm import (
    MODEL,
    Anthropic,
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    extract_json,
    get_default_client,
    llm_error_handler,
    retry_llm_call,
)
from agents._prompt_safety import safe_freeform, safe_label

# Import our components
from agents.adaptive_engine import AdaptiveEngine
from agents.tutor_content import (
    FALLBACK_ITEM_BANK as _FALLBACK_ITEM_BANK_CONTENT,
)
from agents.tutor_content import (
    TUTOR_BASE_PERSONA_PROMPT,
)
from utils.agent_comm import AgentMessage, MessageBus, MessageType
from utils.issue_detector import IssueDetector
from utils.quiz_session import build_quiz_session_from_history
from utils.spaced_repetition import SpacedRepetitionScheduler
from utils.student_profile import (
    get_performance_summary,
    record_chat_exchange,
    record_quiz_feedback,
    record_quiz_result,
    save_profile,
)
from utils.telemetry import (
    COUNTERS,
    LLM_CALLS,
    PROMPT_TRUNCATED,
    QUIZ_FALLBACK_USED,
    QUIZ_GENERATED,
    QUIZ_SCHEMA_REJECTED,
    QUIZ_VERIFIER_REJECTED,
)

if load_dotenv is not None:
    load_dotenv()

logger = logging.getLogger(__name__)

# Distinguish "no client supplied" from an explicit ``None``. Tests and
# deterministic evaluation scripts pass ``None`` to avoid external calls.
_DEFAULT_CLIENT = object()


# Keep underscore-prefixed aliases for backward compatibility with existing
# diagnostics and tests while config.py remains the source of truth.
_CHARS_PER_TOKEN = config.CHARS_PER_TOKEN
_MAX_HISTORY_TOKENS = config.MAX_HISTORY_TOKENS
_MAX_MATERIALS_CHARS = config.MAX_MATERIALS_CHARS
_MAX_STRATEGY_CHARS = config.MAX_STRATEGY_CHARS
_MAX_STUDY_PLAN_PROMPT_CHARS = config.MAX_STUDY_PLAN_PROMPT_CHARS
_MAX_SYSTEM_PROMPT_CHARS = config.MAX_SYSTEM_PROMPT_CHARS
_MAX_USER_MESSAGE_CHARS = config.MAX_USER_MESSAGE_CHARS
_QUIZ_VERIFIER_ENABLED = config.QUIZ_VERIFIER_ENABLED


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...truncated for token budget]"


# Backwards-compat alias: the original ``_safe_course_field`` lives in
# ``agents/_prompt_safety.safe_label`` now and is shared with the
# Resource Agent so both ends of the bus apply the same sanitisation.
_safe_course_field = safe_label


class TutorAgent:
    """
    The main Tutor Agent that a student interacts with.
    """

    AGENT_NAME = "TutorAgent"

    # Immutable persona + safety guardrails. This block is ALWAYS present in
    # every LLM call and is never overwritten by strategy suggestions from
    # the Resource Agent -- that is how we prevent "instruction drift" where
    # a dynamic strategy nudge accidentally nukes the core role or guardrails.
    BASE_PERSONA_PROMPT: str = TUTOR_BASE_PERSONA_PROMPT

    def __init__(
        self,
        student_profile: dict,
        message_bus: MessageBus | None = None,
        client: "Anthropic | None | object" = _DEFAULT_CLIENT,
    ):
        """
        Args:
            student_profile: The student's profile dict (from student_profile.py)
            message_bus: Optional message bus for multi-agent communication
            client: Optional pre-built Anthropic client. If omitted, the
                    agent constructs its own. Tests may pass ``None`` to
                    disable external requests deterministically.
        """
        self.profile = student_profile
        self.client = get_default_client() if client is _DEFAULT_CLIENT else client
        self.adaptive_engine = AdaptiveEngine()

        # LA-7: Restore streak tracker from profile so it survives restarts.
        streak_data: list[bool] | None = self.profile.get("streak_tracker")
        self.issue_detector = IssueDetector(streak_data=streak_data)

        # Modular prompt state -- these are *injected* into the system prompt
        # on each LLM call, not used to rewrite it. Either may be empty.
        self.current_strategy: str = ""
        self.current_strategy_topic: str = ""  # topic the strategy was computed for
        self.current_materials: str = ""
        self.current_topic: str = ""  # topic in the active turn (quiz/feedback)

        # Spaced repetition scheduler
        sr_data = self.profile.get("spaced_repetition", {})
        if sr_data:
            self.scheduler = SpacedRepetitionScheduler.from_dict(sr_data)
        else:
            self.scheduler = SpacedRepetitionScheduler()

        # Multi-agent message bus
        self.bus = message_bus
        if self.bus:
            self.bus.register_agent(self.AGENT_NAME, callback=self._on_message)

        # Rebuild RL state from quiz history so the agent "remembers"
        self._restore_rl_state()

        # Restore durable multi-turn chat memory from the student profile.
        self.conversation_history: list[dict] = []
        for exchange in self.profile.get("chat_history", [])[-25:]:
            if not isinstance(exchange, dict):
                continue
            user_message = str(exchange.get("user_message", "")).strip()
            assistant_response = str(exchange.get("assistant_response", "")).strip()
            if user_message:
                self.conversation_history.append({"role": "user", "content": user_message})
            if assistant_response:
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_response,
                    }
                )

    def _on_message(self, message: AgentMessage) -> None:
        """
        Handle incoming messages from other agents on the bus.

        NOTE: these handlers only update local state variables. They never
        touch a cached system prompt. The system parameter is reassembled
        fresh on every LLM call via `_assemble_system_prompt()`, so new
        strategy / materials take effect automatically without any rewrite.

        Defensive against malformed payloads: even though
        ``ResourceAgent._validate_material_shape`` should have rejected
        bad shapes upstream, this handler treats every field as
        optional and falls back to an empty string rather than letting
        a `.get()` on a non-dict raise.
        """
        if message.msg_type == MessageType.PROVIDE_MATERIALS:
            materials = (
                message.content.get("materials") if isinstance(message.content, dict) else None
            )
            if not isinstance(materials, list) or len(materials) == 0:
                self.current_materials = ""
                return
            mat = materials[0]
            if not isinstance(mat, dict):
                # Upstream validation should have caught this; clear
                # rather than crash if it didn't.
                self.current_materials = ""
                return
            explanation = str(mat.get("explanation", "")).strip()
            analogy = str(mat.get("analogy", "")).strip()
            common_mistake = str(mat.get("common_mistake", "")).strip()
            if not (explanation or analogy or common_mistake):
                self.current_materials = ""
                return
            raw = (
                f"- Explanation: {explanation}\n"
                f"- Analogy: {analogy}\n"
                f"- Common mistake: {common_mistake}"
            )
            self.current_materials = _truncate(raw, _MAX_MATERIALS_CHARS)
        elif message.msg_type == MessageType.SUGGEST_STRATEGY:
            content = message.content if isinstance(message.content, dict) else {}
            self.current_strategy = str(content.get("suggestion", ""))
            self.current_strategy_topic = str(content.get("topic", ""))

    def _restore_rl_state(self) -> None:
        """
        Rebuild the adaptive engine's knowledge from past quiz history.

        LA-5: Replay history entries with ``_decay=False`` so epsilon is
        NOT decayed once per entry.  Epsilon is a deterministic function
        of the total training steps recorded in history:
            epsilon = max(min_epsilon, init_epsilon * 0.995^total_steps)
        This ensures identical epsilon regardless of session boundaries.

        C5: For a brand-new student (quiz_history empty) we do NOT reset
        epsilon. Streamlit reconstructs the agent on every rerun, so
        forcing epsilon back to 1.0 on each reconstruction would spike
        exploration to 100% on every tab switch. Leaving the engine's
        constructor default in place means a fresh student starts at
        the declared init (0.2 by default) and decays from there in
        ``update()`` — smooth in-session behavior rather than
        boundary-driven whiplash.
        """
        quiz_history: list[dict] = self.profile.get("quiz_history", [])

        for entry in quiz_history:
            self.adaptive_engine.update(
                topic=entry["topic"],
                difficulty=entry["difficulty"],
                correct=entry["correct"],
                confidence=entry.get("confidence", 3),
                _decay=False,
            )

        count = len(quiz_history)
        if count > 0:
            # Decay from the engine's *constructor* epsilon (not a
            # hard-coded 1.0), so tests and callers that pass a custom
            # starting epsilon still get a consistent curve. This also
            # avoids the old bug where a returning student with 20
            # quizzes jumped to ε≈0.905 — far from both their fresh
            # starting ε and their intended steady-state.
            base = getattr(self.adaptive_engine, "epsilon", 1.0)
            self.adaptive_engine.epsilon = max(
                self.adaptive_engine.min_epsilon,
                base * (0.995**count),
            )

    def _build_stable_student_context(self) -> str:
        """
        Stable portion of the student context — name, course list,
        learning-style preference. These don't change per quiz answer
        within a session, so they're concatenated to BASE_PERSONA_PROMPT
        in the cacheable system-prompt block.

        Every interpolated value is run through ``safe_label`` so the
        student name, course names, exam dates, and learning style
        cannot smuggle Unicode lookalikes, RTL overrides, line
        separators, or pseudo-section headers into the prompt.
        """
        student_name = safe_label(self.profile.get("name", ""), limit=80)
        courses_info = ""
        for name, data in self.profile["courses"].items():
            safe_name = safe_label(name, limit=80)
            try:
                rating = int(data.get("self_rated_difficulty", 3))
            except (TypeError, ValueError):
                rating = 3
            rating = max(1, min(5, rating))  # clamp self-rated difficulty
            courses_info += f"\n  - {safe_name} (self-rated difficulty: {rating}/5"
            if data.get("exam_date"):
                courses_info += f", exam: {safe_label(data['exam_date'], limit=20)}"
            courses_info += ")"

        style = safe_label(self.profile.get("learning_style", "balanced"), limit=20)
        style_instructions = {
            "concise": "Keep explanations SHORT and to the point. Use bullet points. No fluff.",
            "detailed": "Give thorough, step-by-step explanations with examples. Be comprehensive.",
            "visual": "Use analogies, metaphors, and describe things visually. Help them 'see' concepts.",
            "balanced": "Mix concise summaries with occasional deeper explanations when needed.",
        }
        style_guide = style_instructions.get(style, style_instructions["balanced"])

        return f"""

--- STUDENT CONTEXT ---
You are currently tutoring: {student_name}

STUDENT'S COURSES:{courses_info}

LEARNING STYLE PREFERENCE: {style}
{style_guide}"""

    def _build_dynamic_student_context(self) -> str:
        """
        Per-turn-volatile portion of the student context — performance
        summary, detected issues, spaced-repetition reminders. Lives on
        the cache-MISS side of the system prompt because every quiz
        answer mutates these.
        """
        performance = get_performance_summary(self.profile)
        perf_info = ""
        for cname, cdata in performance["courses"].items():
            safe_course = safe_label(cname, limit=80)
            perf_info += f"\n  - {safe_course}: {cdata['accuracy']}% accuracy over {cdata['attempted']} questions"
            if cdata.get("strong_topics"):
                strong = ", ".join(safe_label(t["topic"], limit=80) for t in cdata["strong_topics"])
                perf_info += f" (strong areas: {strong})"
            if cdata["weak_topics"]:
                weak = ", ".join(safe_label(t["topic"], limit=80) for t in cdata["weak_topics"])
                perf_info += f" (weak areas: {weak})"

        recent_context = self._build_recent_learning_context()

        issue_guidance = self.issue_detector.get_tutor_guidance()
        issue_section = ""
        if issue_guidance:
            issue_section = f"""

SELF-CORRECTION (act on these):
{issue_guidance}"""

        due_topics = self.scheduler.get_due_topics()
        sr_section = ""
        if due_topics:
            topics_due = ", ".join([d["topic"] for d in due_topics[:3]])
            sr_section = f"""

SPACED REPETITION REMINDER:
These topics are due for review: {topics_due}
When appropriate, steer the conversation toward reviewing these."""

        return f"""

PERFORMANCE SO FAR:{perf_info if perf_info else " No quizzes taken yet."}{recent_context}{issue_section}{sr_section}"""

    def _build_recent_learning_context(self) -> str:
        """Summarize saved quiz/session evidence for chat and planning turns."""
        sections: list[str] = []

        sessions = [s for s in self.profile.get("quiz_sessions", []) or [] if isinstance(s, dict)]
        if not sessions:
            restored = build_quiz_session_from_history(self.profile)
            if restored:
                sessions = [restored]
        if sessions:
            latest = sorted(
                sessions,
                key=lambda s: str(s.get("created_at", "")),
                reverse=True,
            )[0]
            course = safe_label(latest.get("course", "Unknown course"), limit=80)
            summary = safe_freeform(latest.get("summary", ""), limit=240)
            priority = (
                ", ".join(safe_label(t, limit=80) for t in latest.get("priority_topics", [])[:3])
                or "none"
            )
            misconceptions = (
                ", ".join(
                    safe_label(t, limit=80) for t in latest.get("misconception_topics", [])[:3]
                )
                or "none"
            )
            sections.append(
                "\n\nLATEST 5-QUESTION EVALUATION:"
                f"\n  - Course: {course}"
                f"\n  - Score: {latest.get('correct', 0)}/{latest.get('question_count', 0)} "
                f"({latest.get('accuracy', 0)}%)"
                f"\n  - Summary: {summary or 'No summary saved.'}"
                f"\n  - Priority topics: {priority}"
                f"\n  - High-confidence misses: {misconceptions}"
            )

        history = [h for h in self.profile.get("quiz_history", []) or [] if isinstance(h, dict)]
        if history:
            recent_lines: list[str] = []
            for entry in history[-5:]:
                topic = safe_label(entry.get("topic", "this topic"), limit=80)
                course = safe_label(entry.get("course", ""), limit=80)
                status = "correct" if entry.get("correct") else "wrong"
                confidence = entry.get("confidence", "?")
                difficulty = entry.get("difficulty", "?")
                question = safe_freeform(entry.get("question", ""), limit=140)
                answer = safe_label(entry.get("student_answer", "?"), limit=8)
                recent_lines.append(
                    f"\n  - {status}; {course} / {topic}; "
                    f"difficulty {difficulty}/5; confidence {confidence}/5; "
                    f"answer {answer}; question: {question}"
                )
            sections.append("\n\nRECENT QUIZ ANSWERS:" + "".join(recent_lines))

            confident_wrong = sum(
                1 for h in history if h.get("confidence", 3) >= 4 and not h.get("correct")
            )
            unsure_right = sum(
                1 for h in history if h.get("confidence", 3) <= 2 and h.get("correct")
            )
            sections.append(
                "\n\nCONFIDENCE PATTERN:"
                f"\n  - High-confidence wrong answers: {confident_wrong}"
                f"\n  - Low-confidence correct answers: {unsure_right}"
            )

        resource_state = self.profile.get("resource_agent_state", {}) or {}
        knowledge_base = resource_state.get("knowledge_base", {}) or {}
        if isinstance(knowledge_base, dict) and knowledge_base:
            material_lines: list[str] = []
            for key, data in list(knowledge_base.items())[:5]:
                if not isinstance(data, dict):
                    continue
                topic = safe_label(data.get("display_name", key), limit=80)
                materials = data.get("materials", []) or []
                titles = (
                    ", ".join(
                        safe_label(m.get("title", ""), limit=80)
                        for m in materials[:2]
                        if isinstance(m, dict) and m.get("title")
                    )
                    or f"{len(materials)} material(s)"
                )
                last_request = data.get("last_request", {}) or {}
                why = safe_freeform(
                    last_request.get("why_shown")
                    or last_request.get("quiz_evidence")
                    or last_request.get("student_need")
                    or "",
                    limit=180,
                )
                line = f"\n  - {topic}: {titles}"
                if why:
                    line += f"; quiz evidence: {why}"
                material_lines.append(line)
            if material_lines:
                sections.append("\n\nRESOURCE MATERIAL AVAILABLE:" + "".join(material_lines))

        if not sections:
            return ""
        return _truncate("".join(sections), 4000)

    # Backwards-compat shim: tests and diagnostics reach for the full string.
    def _build_student_context(self) -> str:
        return self._build_stable_student_context() + self._build_dynamic_student_context()

    def _assemble_system_blocks(self) -> list[dict]:
        """
        Assemble the system parameter as a list of content blocks with a
        prompt-cache breakpoint between the stable and dynamic halves.

        Block layout:

          [0]  BASE_PERSONA_PROMPT + stable student context
               -- carries ``cache_control: {"type": "ephemeral"}`` so
               Anthropic caches the prefix and replays it across turns
               without re-billing the input tokens.

          [1]  Dynamic student context (performance, issues, spaced-rep)
               + topic-gated CURRENT STRATEGY + PROVIDED MATERIALS
               -- no cache marker; this varies per turn.

        Cache effectiveness: Anthropic's prompt cache requires the
        cached segment to be ≥ 1024 tokens (Sonnet 4.x minimum). For
        profiles with several courses + topics, BASE_PERSONA + stable
        context easily clears that bar. Below the threshold, the
        cache_control marker is a no-op — calls still succeed, just
        without the cache hit.

        Reassembled fresh on every call so updates to
        ``current_strategy`` / ``current_materials`` / mid-session
        course additions take effect without invalidating the cached
        prefix (the prefix only depends on
        BASE_PERSONA_PROMPT + courses + learning_style).
        """
        stable = self.BASE_PERSONA_PROMPT + self._build_stable_student_context()
        dynamic_parts: list[str] = [self._build_dynamic_student_context()]

        # Only inject the strategy when the active turn's topic matches
        # the topic the strategy was computed for. Otherwise a strategy
        # prescribed for Topic A ("Before continuing with 'A'...") would
        # leak into a chat / study-plan / Topic-B quiz and mis-steer the
        # tutor.
        if (
            self.current_strategy
            and self.current_strategy_topic
            and self.current_topic == self.current_strategy_topic
        ):
            dynamic_parts.append(
                f"\n\n--- CURRENT STRATEGY ---\n"
                f"{_truncate(self.current_strategy, _MAX_STRATEGY_CHARS)}"
            )

        if self.current_materials:
            dynamic_parts.append(f"\n\n--- PROVIDED MATERIALS ---\n{self.current_materials}")

        dynamic = "".join(dynamic_parts)

        # Final safety net: per-section caps are not enough — a profile
        # with many courses can still push the COMBINED prompt past the
        # 200 k context window. Trim from the dynamic side first; the
        # persona is sacred and trimming it would void the safety
        # guardrails.
        combined_len = len(stable) + len(dynamic)
        if combined_len > _MAX_SYSTEM_PROMPT_CHARS:
            dropped = combined_len - _MAX_SYSTEM_PROMPT_CHARS
            keep = max(0, len(dynamic) - dropped)
            dynamic = dynamic[:keep] + "\n[...truncated for token budget]"
            COUNTERS.incr(PROMPT_TRUNCATED)
            logger.warning(
                "System prompt exceeded %d chars — truncating %d trailing chars from dynamic block",
                _MAX_SYSTEM_PROMPT_CHARS,
                dropped,
            )
            try:
                self.issue_detector.log_api_error(
                    "prompt_truncated",
                    f"System prompt was {combined_len} chars; "
                    f"truncated {dropped} trailing chars (materials may be cut).",
                    recovered=True,
                )
            except Exception:
                # Issue detector shouldn't be able to break prompt assembly.
                pass

        return [
            {
                "type": "text",
                "text": stable,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": dynamic,
            },
        ]

    def _assemble_system_prompt(self) -> str:
        """
        Backwards-compat helper: return the assembled prompt as a single
        string, useful for tests and diagnostics that want to inspect
        the full prompt text. The actual LLM call sites use
        ``_assemble_system_blocks()`` so they benefit from prompt
        caching; both methods produce equivalent prompt content.
        """
        blocks = self._assemble_system_blocks()
        return "".join(b["text"] for b in blocks)

    def _trim_history_by_tokens(self) -> None:
        """
        Drop the oldest turns until the conversation fits inside the
        Claude context budget. Trimming by *estimated tokens* -- not by
        message count -- is the only way a single 10k-token paste won't
        silently overflow the 200k window and surface as a misleading
        "connection issue".
        """

        def est_tokens(msgs: list[dict]) -> int:
            return sum(len(m.get("content", "")) for m in msgs) // _CHARS_PER_TOKEN

        while (
            self.conversation_history
            and est_tokens(self.conversation_history) > _MAX_HISTORY_TOKENS
        ):
            self.conversation_history.pop(0)

        # Anthropic requires the first message to be from the user.
        while self.conversation_history and self.conversation_history[0]["role"] != "user":
            self.conversation_history.pop(0)

    @retry_llm_call()
    def _chat_llm(self) -> str:
        """Make the LLM call for chat. Retries on transient errors
        (timeouts, rate limits, 5xx including 529 "Overloaded")."""
        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=self._assemble_system_blocks(),
            messages=self.conversation_history,
        )
        return response.content[0].text

    def chat(self, user_message: str) -> str:
        """General chat with the student. For questions, study tips, etc."""
        # Validate + bound the input at the boundary. An empty message would
        # make Anthropic reject the request; a giant paste (logs, essays)
        # would dominate the context budget and could carry injection text
        # past the guardrails via sheer volume.
        if not isinstance(user_message, str):
            return "I can only handle text messages."
        user_message = user_message.strip()
        if not user_message:
            return "Could you type a question for me?"
        if len(user_message) > _MAX_USER_MESSAGE_CHARS:
            user_message = user_message[:_MAX_USER_MESSAGE_CHARS] + "\n[...truncated]"

        self.conversation_history.append({"role": "user", "content": user_message})
        self._trim_history_by_tokens()  # trim BEFORE the API call, not after

        if not self.client:
            courses = ", ".join(self.profile.get("courses", {}).keys()) or "your courses"
            summary = get_performance_summary(self.profile)
            weak_topics: list[str] = []
            for course_name in self.profile.get("courses", {}):
                weak_topics.extend(
                    item.get("topic", "")
                    for item in summary["courses"].get(course_name, {}).get("weak_topics", [])
                )
            focus = ", ".join(t for t in weak_topics[:3] if t) or "the next quiz topic"
            assistant_message = (
                f"Based on your saved progress in {courses}, start by reviewing {focus}, "
                "then answer one adaptive quiz question and check the Progress and Diagnostics tabs."
            )
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            self._persist_chat_exchange(user_message, assistant_message)
            return assistant_message

        try:
            assistant_message = self._chat_llm()
        except (APITimeoutError, APIConnectionError) as e:
            self.issue_detector.log_api_error("api_timeout", str(e), recovered=True)
            assistant_message = (
                "I'm having a bit of trouble connecting right now. "
                "Could you try asking that again in a moment?"
            )
        except InternalServerError as e:
            # 5xx including 529 "Overloaded". retry_llm_call already
            # exhausted its attempts; give the user a clear transient hint.
            logger.exception("InternalServerError from Anthropic in chat()")
            self.issue_detector.log_api_error("server_error", str(e), recovered=True)
            assistant_message = (
                "The AI service is temporarily overloaded. Please try again in a few seconds."
            )
        except BadRequestError as e:
            logger.exception("BadRequestError from Anthropic in chat()")
            self.issue_detector.log_api_error("bad_request", str(e), recovered=True)
            assistant_message = (
                "That message is a bit too long for me to process with our chat so far. "
                "Could you either paste a shorter excerpt or start a fresh chat?"
            )
        except AuthenticationError as e:
            logger.error("Anthropic authentication failed in chat(): %s", e)
            self.issue_detector.log_api_error("authentication_error", str(e), recovered=False)
            assistant_message = (
                "Claude authentication failed. Update ANTHROPIC_API_KEY with a valid key, "
                "restart Study Tutor, and try again."
            )
        except APIError as e:
            logger.exception("Anthropic APIError in chat()")
            self.issue_detector.log_api_error("api_error", str(e), recovered=True)
            assistant_message = "I hit an API error. Please try again in a moment."
        except Exception:
            logger.exception("Unexpected error in chat()")
            self.issue_detector.log_api_error("unknown", "see logs", recovered=True)
            assistant_message = "Something unexpected happened. Please try again."

        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        self._trim_history_by_tokens()
        self._persist_chat_exchange(user_message, assistant_message)
        return assistant_message

    def _persist_chat_exchange(self, user_message: str, assistant_message: str) -> None:
        """Save a completed chat turn so a reloaded profile remembers it."""
        self.profile = record_chat_exchange(
            self.profile,
            user_message,
            assistant_message,
        )
        if not save_profile(self.profile["name"], self.profile):
            self.issue_detector.log_api_error(
                "profile_save_timeout",
                f"Could not persist chat history for {self.profile['name']}",
                recovered=False,
            )

    # ------------------------------------------------------------------
    # Quiz generation (decorator applied to inner LLM call)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_quiz_for_decorator(
        self: "TutorAgent", course: str, topic: str, prompt: str
    ) -> dict:
        """Fallback factory used by the ``@llm_error_handler`` on ``_generate_quiz_llm``."""
        return self._fallback_quiz(topic)

    @llm_error_handler(
        fallback=_fallback_quiz_for_decorator.__func__,  # type: ignore[attr-defined]
        error_label="generate_quiz_question",
    )
    @retry_llm_call()
    def _generate_quiz_llm(self, course: str, topic: str, prompt: str) -> dict:
        """Make the LLM call for quiz generation, parse and structurally
        validate the JSON, then optionally do a second LLM pass to
        verify the answer key is *semantically* correct (not just
        well-formed)."""
        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=512,
            system="You are a quiz question generator. Respond ONLY with valid JSON, no other text.",
            messages=[{"role": "user", "content": prompt}],
        )
        quiz = extract_json(response.content[0].text)
        # Layer 1: structural validation. Schema failures raise
        # JSONDecodeError, which @retry_llm_call retries — a single bad
        # sample shouldn't drop straight to the canned fallback when the
        # next sample would likely fix it.
        self._validate_quiz_shape(quiz)
        # Layer 2: semantic fact-check. Schema-only validation lets a
        # confident-but-wrong answer key through; the verifier asks the
        # model whether the proposed correct_answer is actually correct.
        # Failed verification also raises JSONDecodeError to trigger a
        # retry. Configurable via STUDY_TUTOR_QUIZ_VERIFIER_ENABLED so
        # tests / cost-sensitive deployments can opt out.
        if _QUIZ_VERIFIER_ENABLED:
            self._verify_quiz_semantics(quiz, course, topic)
        return quiz

    @retry_llm_call(max_retries=1, base_delay=0.5)
    def _verify_quiz_semantics(self, quiz: dict, course: str, topic: str) -> None:
        """Cheap second LLM pass: ask the model whether the proposed
        ``correct_answer`` is actually correct, the distractors are
        plausible-but-wrong, and the explanation matches.

        On failed verification, raises ``json.JSONDecodeError`` so the
        outer ``_generate_quiz_llm``'s ``@retry_llm_call`` re-samples
        a fresh quiz. Inner retry budget is conservative (1 retry for
        the verification call itself) — we don't want to multiply LLM
        costs on a verifier blip; we want the OUTER retry to roll a
        fresh quiz that hopefully verifies cleanly.

        On *verifier* failure (LLM error, parse error, anything other
        than the verifier explicitly returning verified=False), raise
        a benign warning and pass — better to skip verification than
        kill the quiz entirely on a verifier hiccup.
        """
        if not self.client:
            return  # No LLM available; no fact-check is possible. Pass.

        safe_course = safe_label(course, limit=80)
        safe_topic = safe_label(topic, limit=80)
        verifier_prompt = f"""You are fact-checking a quiz question generated for the course "{safe_course}", topic "{safe_topic}".

QUESTION: {safe_freeform(quiz.get("question", ""), limit=1000)}

OPTIONS:
{chr(10).join(safe_freeform(o, limit=300) for o in quiz.get("options", []))}

PROPOSED CORRECT ANSWER: {safe_label(quiz.get("correct_answer", ""), limit=4)}

PROPOSED EXPLANATION: {safe_freeform(quiz.get("explanation", ""), limit=1000)}

Decide whether the PROPOSED CORRECT ANSWER is actually correct. The
distractors should be plausible but wrong. The explanation should match
the answer.

Respond with valid JSON ONLY, in EXACTLY this shape:
{{"verified": true|false, "issues": ["short reason 1", ...]}}

If everything looks correct, return {{"verified": true, "issues": []}}.
If anything is wrong (wrong answer key, mismatched explanation, multiple
correct options, off-topic question), return {{"verified": false, "issues": [...]}}.
Be conservative — if you are uncertain, mark verified=true."""

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=200,
                system="You are a strict but conservative fact-checker. Respond with valid JSON only.",
                messages=[{"role": "user", "content": verifier_prompt}],
            )
            result = extract_json(response.content[0].text)
        except Exception as exc:  # noqa: BLE001
            # The verifier itself blew up. Log and pass — better to ship
            # an unverified quiz than to error out on a verifier hiccup.
            logger.warning("Quiz verifier failed; skipping fact-check: %s", exc)
            return

        if not isinstance(result, dict):
            logger.warning("Quiz verifier returned non-dict; skipping: %r", result)
            return
        if result.get("verified", True) is False:
            issues = result.get("issues", [])
            issues_str = (
                "; ".join(str(i) for i in issues) if isinstance(issues, list) else str(issues)
            )
            # Surface the failure through the issue detector so the
            # Diagnostics tab shows that a quiz was rejected pre-display.
            try:
                self.issue_detector.log_api_error(
                    "quiz_verifier_rejected",
                    f"verifier flagged: {issues_str[:200]}",
                    recovered=True,  # outer retry will recover with a fresh quiz
                )
            except Exception:  # noqa: BLE001
                pass
            COUNTERS.incr(QUIZ_VERIFIER_REJECTED)
            # Raise JSONDecodeError so @retry_llm_call on _generate_quiz_llm
            # treats this as a bad sample and re-rolls.
            raise json.JSONDecodeError(
                f"Quiz verifier rejected the question: {issues_str}",
                str(quiz)[:200],
                0,
            )

    @staticmethod
    def _validate_quiz_shape(quiz: object) -> None:
        """Sanity-check an LLM-generated quiz dict.

        We assert that:
          - the response is a dict with the expected keys
          - ``options`` is a 4-element list of strings
          - ``correct_answer`` is exactly one of "A"/"B"/"C"/"D"
          - the option whose label starts with that letter exists

        On failure raises ``json.JSONDecodeError`` so the retry decorator
        re-samples the LLM rather than dropping straight to the canned
        fallback. A bad answer key from the LLM would silently grade the
        student wrong forever; this turns it into a retry-able transient.
        """
        if not isinstance(quiz, dict):
            COUNTERS.incr(QUIZ_SCHEMA_REJECTED)
            raise json.JSONDecodeError(
                f"Quiz response must be a dict, got {type(quiz).__name__}",
                str(quiz)[:200],
                0,
            )
        options = quiz.get("options")
        if not isinstance(options, list) or len(options) != 4:
            COUNTERS.incr(QUIZ_SCHEMA_REJECTED)
            raise json.JSONDecodeError(
                f"Quiz must have exactly 4 options, got {options!r}",
                str(quiz)[:200],
                0,
            )
        if not all(isinstance(o, str) for o in options):
            COUNTERS.incr(QUIZ_SCHEMA_REJECTED)
            raise json.JSONDecodeError(
                "All quiz options must be strings",
                str(quiz)[:200],
                0,
            )
        correct = str(quiz.get("correct_answer", "")).strip().upper()
        if correct not in {"A", "B", "C", "D"}:
            COUNTERS.incr(QUIZ_SCHEMA_REJECTED)
            raise json.JSONDecodeError(
                f"correct_answer must be one of A/B/C/D, got {correct!r}",
                str(quiz)[:200],
                0,
            )
        # Verify the labelled option actually exists. ``str.upper().startswith``
        # is permissive about whitespace and trailing punctuation in the
        # option prefix (e.g. "A) ...", "A. ...", "A: ...").
        prefix = correct
        if not any(o.strip().upper().startswith(prefix) for o in options):
            COUNTERS.incr(QUIZ_SCHEMA_REJECTED)
            raise json.JSONDecodeError(
                f"correct_answer={correct!r} but no option starts with that letter",
                str(quiz)[:200],
                0,
            )

    def generate_quiz_question(self, course: str, topic: str | None = None) -> dict:
        """
        Generate a single quiz question using LLM + RL difficulty selection.
        Now also considers spaced repetition schedule.
        """
        available_topics = list(self.profile["courses"].get(course, {}).get("topics", {}).keys())

        if not available_topics:
            available_topics = self._suggest_topics(course)

        # Check if any topics are due for spaced repetition review
        due_topics = self.scheduler.get_due_topics(available_topics)
        if due_topics and not topic:
            # 60% chance to pick a due topic (balance review with new learning)
            if random.random() < 0.6:
                topic = due_topics[0]["topic"]

        # Let the RL engine pick topic and difficulty
        if topic:
            difficulty = self.adaptive_engine.recommend_difficulty(topic)
        else:
            topic, difficulty = self.adaptive_engine.select_topic_and_difficulty(available_topics)

        # Style-aware question generation. Every value interpolated into
        # the prompt is sanitised — course / topic / style come from the
        # profile (which may contain user input) and could otherwise
        # smuggle pseudo-instructions through the prompt boundary.
        style = safe_label(self.profile.get("learning_style", "balanced"), limit=20)
        style_hint = ""
        if style == "visual":
            style_hint = "\nMake the question scenario-based or use a real-world analogy."
        elif style == "concise":
            style_hint = "\nKeep the question and options brief and direct."

        safe_course = safe_label(course, limit=80)
        safe_topic_name = safe_label(topic, limit=80)
        difficulty = max(1, min(5, int(difficulty)))

        prompt = f"""Generate a quiz question for the course "{safe_course}" on the topic "{safe_topic_name}".

Difficulty level: {difficulty}/5 (1=very easy, 5=very hard){style_hint}

Quality requirements:
- Make exactly one option correct, and make correct_answer point to that option.
- If the question includes arithmetic, recompute it carefully. The explanation
  must reach the same value shown in the correct option; do not use "closest"
  unless the question explicitly asks for the closest value.
- Avoid numeric questions when a conceptual question would test the topic just
  as well.
- If using an analogy, word the question around one precise role. For neural
  networks and perceptrons, distinguish weights (learned parameters that get
  adjusted), learning rate (step size for updates), activation function
  (threshold/nonlinearity), and loss/error (feedback signal).
- The explanation must directly support the proposed correct answer and must
  not contradict the options.

Respond in EXACTLY this JSON format and nothing else:
{{
    "question": "The quiz question text",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "correct_answer": "A",
    "explanation": "Brief explanation of why the correct answer is right"
}}"""

        if not self.client:
            quiz_data = self._fallback_quiz(topic)
            COUNTERS.incr(QUIZ_FALLBACK_USED)
        else:
            quiz_data = self._generate_quiz_llm(course, topic, prompt)
            COUNTERS.incr(QUIZ_GENERATED)

        quiz_data["topic"] = topic
        quiz_data["difficulty"] = difficulty
        quiz_data["course"] = course
        quiz_data["is_review"] = topic in [d["topic"] for d in due_topics] if due_topics else False
        quiz_data["selection_explanation"] = self.explain_quiz_choice(quiz_data)

        return quiz_data

    def explain_quiz_choice(self, quiz_data: dict) -> dict:
        """Explain why the adaptive engine selected this quiz item."""
        return self.adaptive_engine.explain_recommendation(
            quiz_data.get("topic", "this topic"),
            quiz_data.get("difficulty", 2),
            is_review=bool(quiz_data.get("is_review", False)),
        )

    # ------------------------------------------------------------------
    # Answer evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_feedback_for_decorator(
        self: "TutorAgent",
        quiz_data: dict,
        correct: bool,
        feedback_prompt: str,
    ) -> str:
        """Fallback factory used by ``@llm_error_handler`` on ``_generate_feedback_llm``."""
        if correct:
            return "Great job getting that right! Keep up the good work."
        return (
            f"Not quite -- the correct answer was {quiz_data['correct_answer']}. "
            f"{quiz_data.get('explanation', 'Review this topic and try again!')}"
        )

    @llm_error_handler(
        fallback=_fallback_feedback_for_decorator.__func__,  # type: ignore[attr-defined]
        error_label="evaluate_answer_feedback",
    )
    @retry_llm_call()
    def _generate_feedback_llm(
        self,
        quiz_data: dict,
        correct: bool,
        feedback_prompt: str,
    ) -> str:
        """LLM call for personalised feedback after a quiz answer."""
        if not self.client:
            return self._fallback_feedback_for_decorator(self, quiz_data, correct, feedback_prompt)
        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=self._assemble_system_blocks(),
            messages=[{"role": "user", "content": feedback_prompt}],
        )
        return response.content[0].text

    def _build_student_feedback(
        self,
        quiz_data: dict,
        correct: bool,
        confidence: int,
        sm2_quality: int,
        used_multi_agent: bool,
    ) -> dict:
        """Build a deterministic, student-facing feedback card.

        The LLM explanation is still the rich natural-language teaching
        response. This card is a stable UI summary that can be tested and
        shown even when a live model request fails.
        """
        topic = safe_label(quiz_data.get("topic", ""), limit=80) or "this topic"
        try:
            difficulty = max(1, min(5, int(quiz_data.get("difficulty", 2))))
        except (TypeError, ValueError):
            difficulty = 2

        if correct:
            status = "success"
            if confidence <= 2:
                title = "Correct, but build confidence"
                summary = (
                    f"You answered {topic} correctly, but your confidence was low. "
                    "That means the idea is close, but it is not automatic yet."
                )
            elif confidence >= 4:
                title = "Strong answer"
                summary = (
                    f"You answered {topic} correctly with high confidence. "
                    "This is a good sign of usable mastery."
                )
            else:
                title = "Good progress"
                summary = (
                    f"You answered {topic} correctly. Keep practicing until the "
                    "reasoning feels quick and repeatable."
                )

            if difficulty < 5:
                next_step = (
                    f"Try one more {topic} question at a slightly higher difficulty, "
                    "or explain the answer in your own words."
                )
            else:
                next_step = f"Teach {topic} back from memory or apply it to a fresh example."
        else:
            status = "warning"
            if confidence >= 4:
                title = "Misconception spotted"
                summary = (
                    f"You were confident but missed {topic}. Treat this as a "
                    "misconception, not just a careless mistake."
                )
            elif confidence <= 2:
                title = "Good self-awareness"
                summary = (
                    f"You were unsure about {topic}, and the answer confirmed that "
                    "this is a good topic to review next."
                )
            else:
                title = "Review needed"
                summary = (
                    f"{topic} needs another pass. Focus on why the correct option "
                    "beats the tempting wrong options."
                )

            if difficulty > 1:
                next_step = (
                    f"Review the explanation, then retry {topic} at an easier "
                    "difficulty before moving back up."
                )
            else:
                next_step = (
                    f"Review the basic definition of {topic}, then do one worked example slowly."
                )

        if confidence >= 4 and not correct:
            confidence_insight = (
                "High-confidence wrong answers are the highest-priority feedback "
                "signal because they reveal a hidden misconception."
            )
        elif confidence <= 2 and correct:
            confidence_insight = (
                "Low-confidence correct answers are a cue to practice for fluency, "
                "not to start over."
            )
        elif confidence >= 4 and correct:
            confidence_insight = "Your confidence matched your result."
        elif confidence <= 2 and not correct:
            confidence_insight = "Your confidence matched your uncertainty."
        else:
            confidence_insight = (
                "Your confidence gives the tutor extra context for the next question."
            )

        if sm2_quality >= 4:
            review_note = "Spaced repetition will schedule this farther out."
        elif sm2_quality <= 2:
            review_note = "Spaced repetition will bring this topic back soon."
        else:
            review_note = "Spaced repetition will keep this topic in regular rotation."

        resource_note = (
            "Resource Agent added extra support material for this explanation."
            if used_multi_agent
            else "No extra Resource Agent material was needed for this turn."
        )

        return {
            "title": title,
            "status": status,
            "topic": topic,
            "summary": summary,
            "confidence_insight": confidence_insight,
            "next_step": next_step,
            "review_note": review_note,
            "resource_note": resource_note,
        }

    def evaluate_answer(
        self,
        quiz_data: dict,
        student_answer: str,
        confidence: int = 3,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> dict:
        """
        Evaluate a student's quiz answer and update all systems.

        Args:
            quiz_data: The quiz dict returned by ``generate_quiz_question``.
            student_answer: The student's selected option (a single letter).
            confidence: Student's self-rated confidence 1-5.
            now_fn: Optional clock for stamping the persisted quiz_history
                    entry. Threaded through to ``record_quiz_result``.
                    Default is ``datetime.now(timezone.utc)``. Tests and
                    the sample profile generator backdates via this hook.
        """
        # Clamp confidence at the public boundary. Streamlit's slider
        # already bounds it 1-5, but tests, simulations, and any
        # programmatic caller passing through evaluate_answer should be
        # defended too — see AdaptiveEngine.update for the same
        # defence-in-depth at the inner layer.
        try:
            confidence = max(1, min(5, int(confidence)))
        except (TypeError, ValueError):
            confidence = 3

        # Mark the active-turn topic so _assemble_system_prompt can gate
        # the CURRENT STRATEGY block (only injected when it matches).
        self.current_topic = quiz_data["topic"]

        # Coerce to string before comparing. Streamlit UI always passes
        # a one-char letter from a button, but the README documents
        # programmatic use, where a caller might pass None / int / bytes.
        # Additionally, normalise inputs like "A) my answer" (which a
        # programmatic caller could plausibly pass) to just "A" by
        # taking the leading letter when it's A-D.
        student_answer_norm = (
            str(student_answer if student_answer is not None else "").upper().strip()
        )
        leading_letter = re.match(r"[A-D]", student_answer_norm)
        if leading_letter:
            student_answer_norm = leading_letter.group(0)
        correct_answer_norm = str(quiz_data.get("correct_answer", "")).upper().strip()
        correct = bool(correct_answer_norm) and student_answer_norm == correct_answer_norm
        student_answer = student_answer_norm  # keep downstream record_quiz_result happy

        # 1. Update the RL adaptive engine (with confidence)
        self.adaptive_engine.update(
            topic=quiz_data["topic"],
            difficulty=quiz_data["difficulty"],
            correct=correct,
            confidence=confidence,
        )

        # 2. Update spaced repetition schedule
        sm2_quality = self.scheduler.quality_from_result(
            correct, confidence, quiz_data["difficulty"]
        )
        self.scheduler.update_topic(quiz_data["topic"], sm2_quality)

        # 3. Record in student profile
        self.profile = record_quiz_result(
            profile=self.profile,
            course=quiz_data["course"],
            topic=quiz_data["topic"],
            difficulty=quiz_data["difficulty"],
            correct=correct,
            question=quiz_data["question"],
            answer=student_answer,
            confidence=confidence,
            correct_answer=quiz_data.get("correct_answer"),
            explanation=quiz_data.get("explanation"),
            now_fn=now_fn,
        )

        # 4. Run issue detection (updates streak tracker state).
        new_issues = self.issue_detector.analyze_quiz_result(
            self.profile,
            quiz_data["course"],
            quiz_data["topic"],
            quiz_data["difficulty"],
            correct,
        )

        # 5. MULTI-AGENT COMMUNICATION.
        # Done BEFORE save_profile so that resource_agent_state captures
        # the KB rows and weakness reports generated by THIS turn (not
        # lagged by one turn as before).
        used_multi_agent = False

        if self.bus:
            topic_stats = (
                self.profile["courses"]
                .get(quiz_data["course"], {})
                .get("topics", {})
                .get(quiz_data["topic"], {})
            )
            topic_accuracy = topic_stats.get("correct", 0) / max(topic_stats.get("attempted", 1), 1)

            if not correct:
                material_request = AgentMessage(
                    sender=self.AGENT_NAME,
                    receiver="ResourceAgent",
                    msg_type=MessageType.REQUEST_MATERIALS,
                    content={
                        "topic": quiz_data["topic"],
                        "course": quiz_data["course"],
                        "student_level": "beginner" if topic_accuracy < 0.4 else "intermediate",
                        "context": (
                            f"Student answered '{student_answer}' but correct was "
                            f"'{quiz_data['correct_answer']}'. "
                            f"Question: {quiz_data['question']}"
                        ),
                        "learning_style": self.profile.get("learning_style", "balanced"),
                    },
                    priority=3,
                )
                self.bus.send(material_request)
                if self.current_materials:
                    used_multi_agent = True

            # REPORT WEAKNESS if accuracy is low (triggers strategy negotiation)
            if topic_stats.get("attempted", 0) >= 3 and topic_accuracy < 0.5:
                weakness_msg = AgentMessage(
                    sender=self.AGENT_NAME,
                    receiver="ResourceAgent",
                    msg_type=MessageType.REPORT_WEAKNESS,
                    content={
                        "topic": quiz_data["topic"],
                        "course": quiz_data["course"],
                        "accuracy": topic_accuracy,
                        "attempts": topic_stats.get("attempted", 0),
                    },
                    priority=3,
                )
                self.bus.send(weakness_msg)

        # 6. Persist profile AFTER bus dispatch so resource_agent_state
        # captures this turn's KB / weakness updates (closes the one-turn
        # lag). Spaced repetition, streak tracker, and resource state are
        # all snapshotted here.
        self.profile["spaced_repetition"] = self.scheduler.to_dict()
        self.profile["streak_tracker"] = self.issue_detector.get_streak_data()
        if self.bus:
            resource = self.bus.get_agent("ResourceAgent")
            if resource is not None and hasattr(resource, "to_dict"):
                self.profile["resource_agent_state"] = resource.to_dict()
        persisted = save_profile(self.profile["name"], self.profile)
        if not persisted:
            self.issue_detector.log_api_error(
                "profile_save_timeout",
                f"Could not acquire profile lock for {self.profile['name']} "
                "— last quiz result not persisted; it will retry on the next save.",
                recovered=False,
            )

        # 7. If issues detected, persist the log.
        if new_issues:
            self.issue_detector.save_log(self.profile["name"])

        # 8. Generate personalized feedback (enriched with Resource Agent materials).
        # Every interpolated field goes through the prompt-safety helpers:
        #   - safe_freeform on long natural-language fragments (question text,
        #     explanation, suggested fix) so a poisoned LLM-generated
        #     question can't impersonate a section divider.
        #   - safe_label on the short letter-and-topic fields.
        issue_context = ""
        if new_issues:
            issue_context = f"\n\nNOTE: {safe_freeform(new_issues[0].suggested_fix, limit=500)}"

        safe_question = safe_freeform(quiz_data.get("question", ""), limit=2000)
        safe_topic_name = safe_label(quiz_data.get("topic", ""), limit=80)
        safe_student_letter = safe_label(student_answer, limit=4)
        safe_correct_letter = safe_label(quiz_data.get("correct_answer", ""), limit=4)
        safe_explanation = safe_freeform(quiz_data.get("explanation", "N/A"), limit=1500)

        if correct:
            feedback_prompt = f"""The student answered correctly!
Question: {safe_question}
Their answer: {safe_student_letter}
Difficulty: {quiz_data["difficulty"]}/5
Their confidence: {confidence}/5{issue_context}

Give a brief (1-2 sentence) encouraging response and one bonus fun fact related to the topic "{safe_topic_name}"."""
        else:
            confidence_note = ""
            if confidence >= 4:
                confidence_note = "\nNote: this student had HIGH confidence but was wrong -- gently address the misconception."

            feedback_prompt = f"""The student answered incorrectly.
Question: {safe_question}
Their answer: {safe_student_letter}
Correct answer: {safe_correct_letter}
Explanation: {safe_explanation}
Difficulty: {quiz_data["difficulty"]}/5
Their confidence: {confidence}/5{issue_context}{confidence_note}

Give an encouraging response (don't make them feel bad), then explain the correct answer.
If a "PROVIDED MATERIALS" section is in your system context, use those materials to
enrich your explanation. Keep it to 3-4 sentences. End with a helpful tip or analogy
for remembering this concept."""

        try:
            explanation = self._generate_feedback_llm(quiz_data, correct, feedback_prompt)
        finally:
            # Materials are one-shot for this feedback turn. Strategy is
            # now topic-gated in _assemble_system_prompt, but we still
            # clear it (plus its topic, plus current_topic) so a later
            # evaluate_answer on the SAME topic doesn't resurrect stale
            # advice, and chat / study-plan never see it.
            self.current_materials = ""
            self.current_strategy = ""
            self.current_strategy_topic = ""
            self.current_topic = ""

        # The just-recorded quiz_history entry — the UUID is what the
        # Streamlit feedback widget keys against, and it's what
        # ``record_quiz_feedback`` looks up when the student flags an
        # issue. ``record_quiz_result`` always appends, so the last
        # entry IS this turn's record.
        last_entry = self.profile["quiz_history"][-1] if self.profile.get("quiz_history") else {}
        quiz_id = last_entry.get("id")
        student_feedback = self._build_student_feedback(
            quiz_data, correct, confidence, sm2_quality, used_multi_agent
        )
        if last_entry:
            last_entry["tutor_response"] = explanation
            last_entry["student_feedback"] = student_feedback
            persisted = save_profile(self.profile["name"], self.profile) and persisted

        return {
            "correct": correct,
            "correct_answer": quiz_data["correct_answer"],
            "confidence": confidence,
            "explanation": explanation,
            "student_feedback": student_feedback,
            "new_mastery": self.adaptive_engine.get_topic_mastery(),
            "issues": [i.to_dict() for i in new_issues],
            "is_review": quiz_data.get("is_review", False),
            "sm2_quality": sm2_quality,
            "used_multi_agent": used_multi_agent,
            "persisted": persisted,
            "quiz_id": quiz_id,
        }

    # ------------------------------------------------------------------
    # Study plan (decorator applied)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_study_plan(self: "TutorAgent") -> str:
        summary = get_performance_summary(self.profile)
        due_topics = self.scheduler.get_due_topics()
        upcoming = self.scheduler.get_upcoming_reviews(days=7)

        weak: list[str] = []
        for course_name, data in summary.get("courses", {}).items():
            for item in data.get("weak_topics", [])[:2]:
                weak.append(f"{item['topic']} ({course_name})")

        due = ", ".join(d["topic"] for d in due_topics[:3]) or "no topics due today"
        weak_text = ", ".join(weak[:4]) or "start with a balanced review across your courses"
        upcoming_text = (
            ", ".join(f"{u['topic']} in {u['days_until_review']:.0f} day(s)" for u in upcoming[:3])
            or "no scheduled reviews this week"
        )

        return (
            "### 7-Day Study Plan\n\n"
            f"**Today:** Review spaced-repetition items first: {due}. Then do one adaptive quiz.\n\n"
            f"**Next focus:** Spend 25 minutes each on: {weak_text}.\n\n"
            f"**This week:** Watch for upcoming reviews: {upcoming_text}.\n\n"
            "**Local tutor plan:** This plan uses your saved profile, weak topics, "
            "and spaced-review schedule."
        )

    @llm_error_handler(
        fallback=_fallback_study_plan.__func__,  # type: ignore[attr-defined]
        error_label="generate_study_plan",
    )
    @retry_llm_call()
    def generate_study_plan(self) -> str:
        """Generate a personalized study plan with spaced repetition integration."""
        if not self.client:
            return self._fallback_study_plan(self)

        mastery = self.adaptive_engine.get_topic_mastery()
        summary = get_performance_summary(self.profile)
        due_topics = self.scheduler.get_due_topics()
        upcoming = self.scheduler.get_upcoming_reviews(days_ahead=7)

        # Student name + learning style are short-label fields; the JSON
        # blobs are bounded by _MAX_STUDY_PLAN_PROMPT_CHARS via _truncate
        # below — and they're already produced by code we control, not
        # the LLM, so they don't need the freeform defang.
        safe_name = safe_label(self.profile.get("name", ""), limit=80)
        safe_style = safe_label(self.profile.get("learning_style", "balanced"), limit=20)
        prompt = f"""Based on this student's data, create a personalized weekly study plan.

STUDENT: {safe_name}
LEARNING STYLE: {safe_style}

COURSES AND PERFORMANCE:
{json.dumps(summary["courses"], indent=2)}

TOPIC MASTERY LEVELS:
{json.dumps(mastery, indent=2) if mastery else "No data yet -- suggest a balanced plan."}

TOPICS DUE FOR REVIEW (spaced repetition):
{json.dumps(due_topics, indent=2) if due_topics else "None due yet."}

UPCOMING REVIEWS (next 7 days):
{json.dumps(upcoming, indent=2) if upcoming else "None scheduled."}

Create a specific, actionable 7-day study plan that:
1. Prioritizes topics due for spaced repetition review FIRST
2. Then focuses on weak topics and upcoming exams
3. Spaces out study sessions (no cramming)
4. Includes specific time estimates per topic
5. Mixes review of weak areas with practice on stronger ones
6. Is encouraging and realistic

Format it clearly with days and bullet points."""

        # L7: the user prompt embeds full JSON dumps of per-course weak
        # topics and mastery. For a long-running user with many courses
        # this could silently overshoot the 200k token window. Truncate
        # explicitly so overflow fails in one bounded place.
        prompt = _truncate(prompt, _MAX_STUDY_PLAN_PROMPT_CHARS)

        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=self._assemble_system_blocks(),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Topic suggestion (decorator applied)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_topics(self: "TutorAgent", course: str) -> list[str]:
        return ["Introduction", "Core Concepts", "Applications", "Problem Solving", "Review"]

    @llm_error_handler(
        fallback=_fallback_topics.__func__,  # type: ignore[attr-defined]
        error_label="suggest_topics",
    )
    @retry_llm_call()
    def _suggest_topics(self, course: str) -> list[str]:
        """Use LLM to suggest common topics for a course."""
        safe_course = safe_label(course, limit=80)
        prompt = f"""List 5 common study topics for a course called "{safe_course}".
Respond ONLY with a JSON array of strings, nothing else.
Example: ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]"""

        COUNTERS.incr(LLM_CALLS)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=200,
            system="Respond only with a JSON array of strings. No other text.",
            messages=[{"role": "user", "content": prompt}],
        )
        topics = extract_json(response.content[0].text)
        if isinstance(topics, list) and len(topics) > 0:
            if course in self.profile["courses"]:
                for t in topics:
                    if t not in self.profile["courses"][course]["topics"]:
                        self.profile["courses"][course]["topics"][t] = {
                            "correct": 0,
                            "attempted": 0,
                            "current_difficulty": 2,
                        }
                save_profile(self.profile["name"], self.profile)
            return topics
        raise ValueError("LLM returned empty or invalid topic list")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Hand-authored fallback item bank, keyed by canonical topic id.
    # Used when ``_generate_quiz_llm`` exhausts its retry budget — gives
    # the student something pedagogically real to engage with rather
    # than the previous "What is an important concept in X?" stub.
    # Adding to this bank: pick a topic that's in
    # ``agents/_prerequisites.LOCAL_PREREQUISITES``, write a single
    # canonical question, mark the correct option's letter, write a
    # one-sentence explanation. ALL fallback items must satisfy
    # ``_validate_quiz_shape``: 4 options, correct_answer ∈ A-D,
    # the labelled letter must appear in options.
    _FALLBACK_ITEM_BANK: dict[str, dict] = _FALLBACK_ITEM_BANK_CONTENT

    def _fallback_quiz(self, topic: str) -> dict:
        """Fallback quiz when LLM generation fails (retries exhausted).

        First tries the hand-authored ``_FALLBACK_ITEM_BANK`` (keyed by
        canonical topic id) for a real, pedagogically-useful question.
        If the topic isn't in the bank, returns a clearly-labelled
        generic stub that fits the schema but tells the student
        honestly that a general review item is being used.
        """
        from agents._prerequisites import canonical_topic_id  # avoid import cycle at module top

        key = canonical_topic_id(topic)
        item = self._FALLBACK_ITEM_BANK.get(key)
        if item is not None:
            # Make a shallow copy so the caller can mutate ``topic`` /
            # ``difficulty`` / ``course`` on the result without
            # corrupting the bank.
            return dict(item)

        safe = safe_label(topic, limit=80) or "this topic"
        return {
            "question": (
                f"You are beginning a review cycle for '{safe}'. "
                "Which first step builds the strongest foundation?"
            ),
            "options": [
                f"A) Review the definition of {safe} and a worked example",
                f"B) Solve a practice problem from the textbook on {safe}",
                f"C) Write a short summary of {safe} in your own words",
                "D) Move on to a different topic and revisit this one tomorrow",
            ],
            "correct_answer": "A",
            "explanation": (
                "A definition-plus-example review rebuilds the foundation before "
                "you attempt more difficult recall and application questions."
            ),
        }

    def record_feedback(self, quiz_id: str, flags: list[str], note: str | None = None) -> bool:
        """Persist student feedback flags against a recent quiz history entry.

        Returns True on success (the entry was found and updated and
        the profile was saved), False if the entry could not be located
        or the save was dropped on a lock timeout. The Streamlit UI
        uses the return value to surface a "couldn't save your
        feedback" warning rather than silently swallowing it.

        Side effects: appends/unions feedback flags to the quiz_history
        entry, persists the profile, and surfaces a pattern-issue
        through the IssueDetector when 3+ recent entries carry
        ``wrong_answer`` or ``unclear_question`` flags — those are the
        signals that the LLM is producing structurally-valid but
        factually-wrong quizzes, the exact failure mode the verifier
        also defends against. Capturing it from BOTH the verifier and
        live student feedback makes the detection more robust.
        """
        updated = record_quiz_feedback(self.profile, quiz_id, flags, note=note)
        if not updated:
            return False
        # Run the pattern detector after each new flag so consecutive
        # flag-storms surface promptly in Diagnostics.
        try:
            self.issue_detector.check_feedback_pattern(self.profile)
        except Exception:  # noqa: BLE001
            # The pattern detector is best-effort; never let it block
            # a feedback save.
            pass
        return save_profile(self.profile["name"], self.profile)

    def get_mastery_data(self) -> dict:
        """Get current mastery levels for display."""
        return self.adaptive_engine.get_topic_mastery()

    def get_due_reviews(self) -> list[dict]:
        """Get topics due for spaced repetition review."""
        return self.scheduler.get_due_topics()

    def get_upcoming_reviews(self, days: int = 7) -> list[dict]:
        """Get upcoming review schedule."""
        return self.scheduler.get_upcoming_reviews(days)

    def get_issue_summary(self) -> dict:
        """Get a summary of detected issues."""
        return self.issue_detector.get_issue_summary()

    def get_recent_issues(self, n: int = 10) -> list[dict]:
        """Get recent issues for the diagnostics tab."""
        return self.issue_detector.get_recent_issues(n)

    def get_message_log(self) -> list[dict]:
        """Get full inter-agent communication log."""
        if self.bus:
            return self.bus.get_conversation_log()
        return []

    def get_comm_stats(self) -> dict:
        """Get communication statistics between agents."""
        if self.bus:
            return self.bus.get_stats()
        return {"total_messages": 0, "registered_agents": [], "by_type": {}, "by_sender": {}}
