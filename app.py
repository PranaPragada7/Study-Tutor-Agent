"""
AI Study Tutor — Streamlit App

Run with: streamlit run app.py

Features:
  1. Chat — Talk to the tutor about anything study-related
  2. Quiz — Adaptive quizzes with confidence self-rating + multi-agent explanations
  3. Study Plan — Personalized weekly plan with spaced repetition
  4. Progress — Mastery levels, quiz history, and review schedule
  5. Diagnostics — Issue detection, agent communication log, RL state
"""

import streamlit as st
import sys
import os

# Make imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils.student_profile import (
    create_profile,
    delete_profile,
    export_profile_json,
    get_performance_summary,
    list_saved_profiles,
    load_profile,
    reset_quiz_history,
    save_profile,
)
from utils.agent_comm import MessageBus
from utils.quiz_session import (
    MIN_QUESTIONS_BEFORE_EVALUATION,
    build_quiz_session_feedback,
    build_persisted_quiz_session,
    build_quiz_session_from_history,
    build_quiz_session_report,
    build_quiz_session_summary,
    can_evaluate_session,
    store_quiz_session_report,
)
from utils.telemetry import COUNTERS, COUNTER_DISPLAY_ORDER
from agents.tutor_agent import TutorAgent
from agents.resource_agent import ResourceAgent
from config import OFFLINE_MODE

try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Tutor",
    page_icon="📘",
    layout="wide",
)


def inject_app_styles() -> None:
    """Apply a compact, professional visual system on top of Streamlit."""
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --surface-soft: #f3f6f8;
            --border: #d9e1e7;
            --text: #111827;
            --muted: #607080;
            --accent: #0f766e;
            --accent-soft: #e6f4f1;
        }

        .stApp {
            background: #f7f8fb;
            color: var(--text);
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1360px;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        .app-header {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: center;
        }

        .app-title {
            margin: 0;
            color: var(--text);
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 760;
            letter-spacing: 0;
        }

        .app-subtitle {
            margin: 6px 0 0 0;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .status-pill {
            border: 1px solid var(--border);
            background: var(--surface-soft);
            border-radius: 999px;
            padding: 7px 11px;
            color: #25313d;
            font-size: 0.83rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .status-pill.accent {
            border-color: #9bd3ca;
            background: var(--accent-soft);
            color: #0b5f59;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
        }

        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 760;
        }

        .stButton > button {
            border-radius: 7px;
            border: 1px solid var(--border);
            font-weight: 700;
            min-height: 2.55rem;
            transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
        }

        .stButton > button:hover {
            border-color: var(--accent);
            box-shadow: 0 2px 10px rgba(15, 118, 110, 0.12);
            transform: translateY(-1px);
        }

        div[data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        div[data-testid="stTabs"] button {
            font-weight: 700;
            color: #44515f;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 7px;
        }

        hr {
            border-color: #e5eaf0;
            margin: 1.15rem 0;
        }

        @media (max-width: 760px) {
            .app-header {
                align-items: flex-start;
                flex-direction: column;
            }
            .status-row {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    mode_label = "Offline ready" if OFFLINE_MODE else "Live API"
    mode_class = "" if OFFLINE_MODE else "accent"
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1 class="app-title">AI Study Tutor</h1>
                <p class="app-subtitle">Adaptive study workspace</p>
            </div>
            <div class="status-row">
                <span class="status-pill accent">5-question sessions</span>
                <span class="status-pill">Resource Agent</span>
                <span class="status-pill {mode_class}">{mode_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_app_styles()
render_app_header()


# ──────────────────────────────────────────────
# Helper: Shared Anthropic client (one HTTPX pool per Streamlit process,
# not one per agent per session)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_perf_summary_for_key(cache_key: str, _profile: dict) -> dict:
    """Compute the performance summary once per (student, quiz_count, topic_count) tuple.

    Streamlit reruns every tab switch; on a profile with many courses this
    was O(courses * topics) of pure Python work per rerun. Keying by
    (name, total_quizzes, topic_count) invalidates on every answered question
    AND on topic-list changes from `_suggest_topics` (which adds topics
    without bumping total_quizzes), so the Progress tab never shows stale
    `num_topics_covered` / `weak_topics`.
    """
    return get_performance_summary(_profile)


def cached_performance(profile: dict) -> dict:
    topic_count = sum(
        len(course.get("topics", {}))
        for course in profile.get("courses", {}).values()
    )
    cache_key = (
        f"{profile.get('name', '?')}"
        f"::{profile.get('total_quizzes', 0)}"
        f"::{topic_count}"
    )
    return _cached_perf_summary_for_key(cache_key, profile)


@st.cache_resource
def get_anthropic_client() -> "Anthropic | None":
    """One shared Anthropic client per Streamlit server process.

    The API key is read explicitly from the environment — no
    hardcoded keys. If the SDK or key is unavailable, return ``None``
    so the agents can use their local fallback paths and the app
    still has a usable backup flow.

    Returns ``None`` (not raises) when the SDK itself is not installed
    so callers can degrade gracefully to fallback responses.
    """
    if OFFLINE_MODE:
        st.sidebar.info("Offline mode is on. Live Claude calls are skipped.")
        return None
    if not _HAS_ANTHROPIC:
        st.sidebar.warning("Anthropic SDK is not installed. Using local fallback mode.")
        return None
    api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.sidebar.warning("ANTHROPIC_API_KEY is not set. Using local fallback mode.")
        return None
    try:
        return Anthropic(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not initialise Anthropic client: {exc}. Using fallback mode.")
        return None


# ──────────────────────────────────────────────
# Helper: Initialize multi-agent system
# ──────────────────────────────────────────────
def init_agents(profile: dict) -> tuple[TutorAgent, ResourceAgent, MessageBus]:
    """Create MessageBus, TutorAgent, and ResourceAgent wired together."""
    bus = MessageBus()
    client = get_anthropic_client()
    tutor = TutorAgent(profile, message_bus=bus, client=client)
    resource = ResourceAgent(bus, client=client)
    # Restore the Resource Agent's accumulated knowledge base + weakness
    # history so its "learning" actually persists across server restarts.
    resource.load_state(profile.get("resource_agent_state", {}))
    return tutor, resource, bus


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "profile" not in st.session_state:
    st.session_state.profile = None
if "tutor" not in st.session_state:
    st.session_state.tutor = None
if "resource_agent" not in st.session_state:
    st.session_state.resource_agent = None
if "message_bus" not in st.session_state:
    st.session_state.message_bus = None
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False
if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "confidence" not in st.session_state:
    st.session_state.confidence = 3
if "quiz_session_results" not in st.session_state:
    st.session_state.quiz_session_results = []
if "quiz_session_active" not in st.session_state:
    st.session_state.quiz_session_active = False
if "quiz_session_complete" not in st.session_state:
    st.session_state.quiz_session_complete = False
if "quiz_session_course" not in st.session_state:
    st.session_state.quiz_session_course = None


def reset_quiz_ui_state() -> None:
    """Clear in-memory quiz/session widgets when switching or resetting profiles."""
    st.session_state.current_quiz = None
    st.session_state.quiz_answered = False
    st.session_state.quiz_result = None
    st.session_state.confidence = 3
    st.session_state.quiz_session_results = []
    st.session_state.quiz_session_active = False
    st.session_state.quiz_session_complete = False
    st.session_state.quiz_session_course = None


def load_profile_into_session(profile: dict) -> None:
    """Wire a loaded profile into Streamlit session state."""
    st.session_state.profile = profile
    tutor, resource, bus = init_agents(profile)
    st.session_state.tutor = tutor
    st.session_state.resource_agent = resource
    st.session_state.message_bus = bus
    reset_quiz_ui_state()


def get_visible_quiz_sessions(profile: dict) -> list[dict]:
    """Return saved reports, or a reconstructed latest report for old profiles."""
    saved = [
        s for s in profile.get("quiz_sessions", []) or []
        if isinstance(s, dict)
    ]
    if saved:
        return sorted(saved, key=lambda s: str(s.get("created_at", "")), reverse=True)

    restored = build_quiz_session_from_history(profile)
    return [restored] if restored else []


def render_resource_material_suggestions(
    topics: list[str],
    *,
    suggestions: list[dict] | None = None,
    caption: str | None = None,
    empty_message: str | None = None,
) -> bool:
    """Render ResourceAgent materials for the supplied assessed topics."""
    if suggestions is None:
        resource = st.session_state.get("resource_agent")
        if not resource or not topics:
            return False

        suggestions = resource.suggest_materials_for_topics(
            topics,
            limit_per_topic=1,
        )
    if not suggestions:
        if empty_message:
            st.info(empty_message)
        return False

    st.markdown("### Suggested Resource Materials")
    if caption:
        st.caption(caption)
    for suggestion in suggestions:
        rationale = suggestion.get("rationale", {}) or {}
        with st.expander(
            f"{suggestion['topic']}: {suggestion['title']}",
            expanded=True,
        ):
            st.write("**Why this material is shown**")
            st.write(
                rationale.get(
                    "why_shown",
                    "This material is shown because a quiz answer on this topic was wrong.",
                )
            )
            st.write("**Quiz evidence used**")
            st.write(
                rationale.get(
                    "quiz_evidence",
                    rationale.get(
                        "student_need",
                        "This topic appeared in your missed-question pattern.",
                    ),
                )
            )
            st.write("**How the material was generated**")
            st.write(
                rationale.get(
                    "generation_source",
                    rationale.get(
                        "source",
                        "ResourceAgent generated or reused this material after evaluating the quiz answers.",
                    ),
                )
            )
            st.divider()
            if suggestion.get("explanation"):
                st.write(suggestion["explanation"])
            if suggestion.get("analogy"):
                st.write(f"**Analogy:** {suggestion['analogy']}")
            if suggestion.get("common_mistake"):
                st.write(f"**Common mistake:** {suggestion['common_mistake']}")
            if suggestion.get("self_test"):
                st.write(f"**Self-test:** {suggestion['self_test']}")
            if suggestion.get("prerequisites"):
                st.caption(
                    "Prerequisites: "
                    + ", ".join(suggestion["prerequisites"])
                )
    return True


def render_quiz_session_snapshot(session: dict, *, expanded_rows: bool = False) -> None:
    """Render a saved or reconstructed five-question session report."""
    course = session.get("course", "Unknown course")
    created_at = session.get("created_at", "")
    source = session.get("source", "")
    st.caption(f"{course} | {created_at}")
    if source == "history_reconstruction":
        st.info(session.get(
            "note",
            "Restored from saved quiz answers. Some old report details were not stored.",
        ))

    cols = st.columns(3)
    cols[0].metric("Questions", session.get("question_count", 0))
    cols[1].metric("Correct", session.get("correct", 0))
    cols[2].metric("Accuracy", f"{session.get('accuracy', 0)}%")

    st.write(f"**Summary:** {session.get('summary', 'No summary saved.')}")
    if session.get("strengths"):
        st.write(f"**Strengths:** {session['strengths']}")
    if session.get("priority_feedback"):
        st.write(f"**Review priority:** {session['priority_feedback']}")
    if session.get("confidence_feedback"):
        st.write(f"**Confidence pattern:** {session['confidence_feedback']}")
    if session.get("next_step"):
        st.write(f"**Next step:** {session['next_step']}")

    rows = session.get("rows", []) or []
    if rows:
        st.markdown("#### Question-by-Question Report")
        for row in rows:
            status_text = "Correct" if row.get("correct") else "Review"
            label = (
                f"Question {row.get('number', '?')}: "
                f"{row.get('topic', 'this topic')} - {status_text}"
            )
            with st.expander(label, expanded=expanded_rows):
                row_cols = st.columns(3)
                row_cols[0].metric("Difficulty", f"{row.get('difficulty', '?')}/5")
                row_cols[1].metric("Confidence", f"{row.get('confidence', '?')}/5")
                row_cols[2].write("**Signal**")
                row_cols[2].write(row.get("signal", "Not available"))
                if row.get("question"):
                    st.write(f"**Question:** {row['question']}")
                st.write(f"**Your answer:** {row.get('student_answer', 'not shown')}")
                st.write(f"**Correct answer:** {row.get('correct_answer', '?')}")
                if row.get("explanation"):
                    st.write(f"**Explanation:** {row['explanation']}")
                if row.get("feedback_summary"):
                    st.write(f"**Feedback:** {row['feedback_summary']}")
                if row.get("confidence_insight"):
                    st.write(f"**Confidence signal:** {row['confidence_insight']}")
                if row.get("resource_note"):
                    st.write(f"**Tutor action:** {row['resource_note']}")
                if row.get("review_note"):
                    st.caption(row["review_note"])
                st.write(f"**Next action:** {row.get('next_action', 'Review this item.')}")

    priority_topics = session.get("priority_topics", []) or []
    if priority_topics:
        st.divider()
        saved_materials = session.get("resource_materials", []) or []
        render_resource_material_suggestions(
            priority_topics,
            suggestions=saved_materials if saved_materials else None,
            caption=(
                "Shown because this session's saved quiz answers missed these "
                "topics and the session assessment marked them for review."
            ),
        )


# ──────────────────────────────────────────────
# Sidebar: Student Profile Setup
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Student Profile")

    st.markdown("#### Sample Profile")
    sample_cols = st.columns(2)
    with sample_cols[0]:
        if st.button("Load Sample Student", use_container_width=True, type="primary"):
            sample_profile = load_profile("Sample Student")
            if sample_profile:
                load_profile_into_session(sample_profile)
                st.session_state.chat_messages = []
                COUNTERS.reset()
                st.success("Sample Student loaded.")
                st.rerun()
            else:
                st.warning(
                    "Sample profile is missing. Run python scripts\\preflight_check.py "
                    "to regenerate it."
                )
    with sample_cols[1]:
        if st.button("Reset Sample Profile", use_container_width=True):
            sample_profile = load_profile("Sample Student")
            if sample_profile:
                load_profile_into_session(sample_profile)
                st.session_state.chat_messages = []
                COUNTERS.reset()
                st.success("Sample profile reset.")
                st.rerun()
            else:
                reset_quiz_ui_state()
                st.warning("Sample profile could not be loaded.")

    st.divider()

    saved_profiles = list_saved_profiles()
    if saved_profiles:
        st.markdown("#### Existing Profiles")

        def _profile_label(profile_info: dict) -> str:
            quiz_word = "quiz" if profile_info["total_quizzes"] == 1 else "quizzes"
            course_word = "course" if profile_info["courses"] == 1 else "courses"
            return (
                f"{profile_info['name']} - "
                f"{profile_info['total_quizzes']} {quiz_word}, "
                f"{profile_info['courses']} {course_word}"
            )

        selected_profile = st.selectbox(
            "Saved profiles",
            saved_profiles,
            format_func=_profile_label,
            label_visibility="collapsed",
        )
        if st.button("Load Selected Profile", use_container_width=True):
            loaded_profile = load_profile(selected_profile["name"])
            if loaded_profile:
                load_profile_into_session(loaded_profile)
                st.session_state.chat_messages = []
                st.success(f"{loaded_profile['name']} loaded.")
                st.rerun()
            else:
                st.warning("Could not load that profile. It may have been deleted or locked.")

        st.divider()

    # Try to load existing profile
    student_name = st.text_input("Your Name", placeholder="e.g. Alex")

    if student_name:
        existing = load_profile(student_name)
        if existing:
            st.success(f"Welcome back, {student_name}!")
            if st.button("Load Profile", type="primary"):
                load_profile_into_session(existing)
                st.rerun()
        else:
            st.info("New student! Set up your courses below.")

    st.divider()

    # New profile creation
    with st.expander("Add Profile / Courses", expanded=st.session_state.profile is None):
        st.write("Enter your courses (one per line):")
        st.caption(
            "Use enough context for the tutor to understand the course: university, "
            "course number, and course title if you know them. A bare course number "
            "may mean different things at different schools."
        )
        courses_text = st.text_area(
            "Courses",
            placeholder=(
                "Georgia Tech CS 7641 - Machine Learning\n"
                "University of Illinois MATH 241 - Calculus III\n"
                "Biology 110 - Cell Biology"
            ),
            label_visibility="collapsed",
        )

        # Normalise newlines once so Windows (\r\n), classic Mac (\r) and *nix
        # (\n) all parse the same. Splitting on "\n" alone left a trailing "\r"
        # on each course name, which silently broke the slider->dict lookup
        # below ("diff_CS101\r" vs lookup "CS101") and dropped the user's
        # difficulty back to the default 3.
        course_lines = [
            line.strip()
            for line in courses_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]
        num_courses = len(course_lines)

        # Learning style preference
        learning_style = st.selectbox(
            "How do you prefer explanations?",
            options=["balanced", "concise", "detailed", "visual"],
            format_func=lambda x: {
                "balanced": "Balanced — mix of concise and detailed",
                "concise": "Concise — short and to the point",
                "detailed": "Detailed — step-by-step with examples",
                "visual": "Visual — analogies and metaphors",
            }[x],
        )

        # Difficulty ratings
        difficulties = {}
        if num_courses > 0:
            st.write("Rate difficulty (1=easy, 5=hard):")
            for course_name in course_lines:
                difficulties[course_name] = st.slider(
                    course_name, 1, 5, 3, key=f"diff_{course_name}"
                )

        if st.button("Create Profile", disabled=not student_name or num_courses == 0, type="primary"):
            courses = [
                {
                    "name": name,
                    "difficulty": difficulties.get(name, 3),
                    "exam_date": "",
                }
                for name in course_lines
            ]

            profile = create_profile(student_name, courses, learning_style=learning_style)
            load_profile_into_session(profile)
            st.session_state.chat_messages = []
            st.success("Profile created!")
            st.rerun()

    # Show current profile info
    if st.session_state.profile:
        st.divider()
        p = st.session_state.profile
        st.write(f"**Student:** {p['name']}")
        st.write(f"**Style:** {p.get('learning_style', 'balanced')}")
        st.write(f"**Courses:** {len(p['courses'])}")
        st.write(f"**Quizzes taken:** {p['total_quizzes']}")

        # Show due reviews
        if st.session_state.tutor:
            due = st.session_state.tutor.get_due_reviews()
            if due:
                st.divider()
                st.write(f"**{len(due)} topic(s) due for review**")
                for d in due[:3]:
                    st.write(f"- {d['topic']} ({d['days_overdue']:.0f}d overdue)")

        # ── Privacy controls ──
        # Even for a single-user local app, "I want to take my data out"
        # and "I want to start over" are reasonable expectations. None
        # of these touch the data of OTHER profiles.
        st.divider()
        with st.expander("🔒 Privacy & data controls"):
            # Export — download the profile as JSON.
            try:
                profile_json = export_profile_json(p)
                st.download_button(
                    label="Export profile (JSON)",
                    data=profile_json,
                    file_name=f"{p['name']}_profile.json",
                    mime="application/json",
                    use_container_width=True,
                )
            except Exception as e:  # noqa: BLE001
                st.warning(f"Export unavailable: {e}")

            # Clear chat history (in-memory only — not persisted to disk).
            if st.button("Clear chat history", use_container_width=True):
                st.session_state.chat_messages = []
                if st.session_state.tutor:
                    st.session_state.tutor.conversation_history = []
                st.success("Chat history cleared.")

            # Reset adaptive + spaced-repetition state, keep courses.
            confirm_reset = st.checkbox("I understand this wipes quiz progress (kept: courses + name)",
                                         key="confirm_reset")
            if st.button("Reset learning state", use_container_width=True,
                         disabled=not confirm_reset):
                st.session_state.profile = reset_quiz_history(p)
                save_profile(p["name"], st.session_state.profile)
                # Re-init agents on the cleaned profile.
                tutor, resource, bus = init_agents(st.session_state.profile)
                st.session_state.tutor = tutor
                st.session_state.resource_agent = resource
                st.session_state.message_bus = bus
                # Reset diagnostic counters too — a fresh start should
                # show a fresh telemetry surface.
                reset_quiz_ui_state()
                COUNTERS.reset()
                st.success("Learning state reset. Course list preserved.")
                st.rerun()

            # Permanent delete.
            confirm_delete = st.checkbox("I understand this PERMANENTLY deletes my profile",
                                          key="confirm_delete")
            if st.button("Delete my profile", use_container_width=True,
                         disabled=not confirm_delete, type="primary"):
                if delete_profile(p["name"]):
                    st.session_state.profile = None
                    st.session_state.tutor = None
                    st.session_state.resource_agent = None
                    st.session_state.message_bus = None
                    reset_quiz_ui_state()
                    st.session_state.chat_messages = []
                    COUNTERS.reset()
                    st.success("Profile deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete profile (file may be in use).")


# ──────────────────────────────────────────────
# Main Content (only if profile is loaded)
# ──────────────────────────────────────────────
if st.session_state.profile is None or st.session_state.tutor is None:
    st.info("Enter your name and set up your courses in the sidebar to get started.")
    st.stop()

tutor: TutorAgent = st.session_state.tutor
profile_summary = cached_performance(st.session_state.profile)
active_due_reviews = tutor.get_due_reviews()
active_comm_stats = tutor.get_comm_stats()

overview_cols = st.columns(4)
overview_cols[0].metric("Student", st.session_state.profile.get("name", "Unknown"))
overview_cols[1].metric("Courses", len(st.session_state.profile.get("courses", {})))
overview_cols[2].metric("Questions", profile_summary.get("total_quizzes", 0))
overview_cols[3].metric("Agent Messages", active_comm_stats.get("total_messages", 0))

if active_due_reviews:
    st.info(
        f"{len(active_due_reviews)} topic(s) due for review. "
        f"Next: {', '.join([d['topic'] for d in active_due_reviews[:3]])}"
    )

# Tabs for different features
tab_chat, tab_quiz, tab_plan, tab_progress, tab_diag = st.tabs(
    ["Chat", "Quiz", "Study Plan", "Progress", "Diagnostics"]
)

# ──────────────────────────────────────────────
# Tab 1: Chat with the Tutor
# ──────────────────────────────────────────────
with tab_chat:
    st.subheader("Chat with your tutor")
    st.caption("Ask a course question or request a study recommendation.")

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask your tutor anything..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = tutor.chat(user_input)
            st.write(response)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

# ──────────────────────────────────────────────
# Tab 2: Adaptive Quiz with Confidence Rating
# ──────────────────────────────────────────────
with tab_quiz:
    st.subheader("Adaptive Quiz Session")
    st.caption("Complete five questions to unlock the overall evaluation.")

    # Course selector
    course_names = list(st.session_state.profile["courses"].keys())
    selected_course = st.selectbox("Choose a course:", course_names)

    # Show if there are review topics
    due = tutor.get_due_reviews()
    if due:
        review_topics = [d["topic"] for d in due if d["topic"] in
                        st.session_state.profile["courses"].get(selected_course, {}).get("topics", {})]
        if review_topics:
            st.info(f"Topics due for review: {', '.join(review_topics[:3])}")

    def load_next_session_question(course: str) -> None:
        with st.spinner("Generating question..."):
            quiz = tutor.generate_quiz_question(course)
            st.session_state.current_quiz = quiz
            st.session_state.quiz_answered = False
            st.session_state.quiz_result = None
            st.session_state.confidence = 3

    session_count = len(st.session_state.quiz_session_results)
    session_target = MIN_QUESTIONS_BEFORE_EVALUATION
    if st.session_state.quiz_session_active or st.session_state.quiz_session_complete:
        st.progress(
            min(session_count, session_target) / session_target,
            text=f"{session_count}/{session_target} questions answered before overall evaluation",
        )
        if st.session_state.quiz_session_course:
            st.caption(f"Current quiz session: {st.session_state.quiz_session_course}")
    else:
        st.info("Start a 5-question quiz session. The tutor gives the overall evaluation after question 5.")

    control_cols = st.columns([1, 1, 3])

    with control_cols[0]:
        start_label = (
            "Start New 5-Question Quiz"
            if st.session_state.quiz_session_complete
            else "Start 5-Question Quiz"
        )
        can_start = (
            not st.session_state.quiz_session_active
            and not st.session_state.current_quiz
        ) or st.session_state.quiz_session_complete
        if st.button(start_label, use_container_width=True, disabled=not can_start, type="primary"):
            reset_quiz_ui_state()
            st.session_state.quiz_session_active = True
            st.session_state.quiz_session_course = selected_course
            load_next_session_question(selected_course)
            st.rerun()

    with control_cols[1]:
        can_reset = bool(
            st.session_state.quiz_session_results
            or st.session_state.current_quiz
            or st.session_state.quiz_session_active
        )
        if st.button("Reset Session", use_container_width=True, disabled=not can_reset):
            reset_quiz_ui_state()
            st.rerun()

    if st.session_state.quiz_session_active and st.session_state.current_quiz is None:
        if st.button("Continue Quiz", use_container_width=True):
            load_next_session_question(st.session_state.quiz_session_course or selected_course)
            st.rerun()

    # Display current question
    if st.session_state.current_quiz:
        quiz = st.session_state.current_quiz

        st.divider()

        # Show metadata
        meta_cols = st.columns(3)
        meta_cols[0].write(f"**Topic:** {quiz['topic']}")
        meta_cols[1].write(f"**Difficulty:** {quiz['difficulty']}/5")
        if quiz.get("is_review"):
            meta_cols[2].write("**Spaced Review**")

        selection_explanation = quiz.get("selection_explanation") or tutor.explain_quiz_choice(quiz)
        with st.expander("Why this question?", expanded=True):
            reason_cols = st.columns(3)
            reason_cols[0].write("**Reason**")
            reason_cols[0].write(selection_explanation.get("reason", "Adaptive practice"))
            reason_cols[1].write("**Policy**")
            reason_cols[1].write(selection_explanation.get("policy", "adaptive"))
            mastery = selection_explanation.get("mastery")
            reason_cols[2].write("**Current mastery**")
            reason_cols[2].write("No history yet" if mastery is None else f"{mastery}%")
            st.caption(selection_explanation.get("detail", "The tutor selected this from your current profile state."))
            st.write(
                f"Recommended difficulty now: "
                f"{selection_explanation.get('recommended_difficulty', '?')}/5"
            )

        st.markdown(f"### {quiz['question']}")

        if not st.session_state.quiz_answered:
            # Confidence self-rating BEFORE answering
            st.session_state.confidence = st.slider(
                "How confident are you? (1=guessing, 5=certain)",
                min_value=1, max_value=5, value=3, key="conf_slider"
            )

            # Show answer options as buttons
            for option in quiz.get("options", []):
                letter = option[0]
                if st.button(option, key=f"opt_{letter}", use_container_width=True):
                    with st.spinner("Evaluating..."):
                        result = tutor.evaluate_answer(
                            quiz, letter, confidence=st.session_state.confidence
                        )
                    st.session_state.quiz_result = result
                    st.session_state.quiz_answered = True
                    st.session_state.quiz_session_results.append({
                        "quiz": quiz,
                        "result": result,
                        "confidence": st.session_state.confidence,
                        "student_answer": letter,
                    })
                    if can_evaluate_session(st.session_state.quiz_session_results):
                        st.session_state.quiz_session_complete = True
                        st.session_state.quiz_session_active = False
                    # NOTE: no st.session_state.profile = tutor.profile here.
                    # They already point at the same dict (init_agents passes
                    # the profile by reference into TutorAgent), so the
                    # reassignment was a no-op that obscured ownership.
                    st.rerun()
        else:
            # Show result
            result = st.session_state.quiz_result

            if result.get("persisted") is False:
                st.warning(
                    "Could not save your answer to disk "
                    "(another tab may be writing). The tutor will retry on the next save."
                )

            if result["correct"]:
                st.success("Correct")
            else:
                st.error(f"The correct answer was: {result['correct_answer']}")

            st.markdown(result["explanation"])

            feedback = result.get("student_feedback")
            if feedback:
                st.markdown("### Feedback for this question")
                st.caption(
                    "Shown after every answer. The tutor uses correctness, confidence, "
                    "difficulty, spaced repetition, and ResourceAgent support."
                )
                if feedback.get("status") == "success":
                    st.success(feedback.get("summary", "Nice work."))
                elif feedback.get("status") == "warning":
                    st.warning(feedback.get("summary", "Review this topic before moving on."))
                else:
                    st.info(feedback.get("summary", "Keep practicing this topic."))

                fb_cols = st.columns(2)
                with fb_cols[0]:
                    st.write("**Confidence signal**")
                    st.write(feedback.get("confidence_insight", "Confidence helps tune the next question."))
                with fb_cols[1]:
                    st.write("**Next step**")
                    st.write(feedback.get("next_step", "Try another question on this topic."))
                st.write("**Tutor action**")
                st.write(feedback.get("resource_note", "The tutor updated your learning state for this turn."))
                st.caption(feedback.get("review_note", "Spaced repetition will update after this answer."))

            # Show if issues were detected
            if result.get("issues"):
                for issue in result["issues"]:
                    if issue["severity"] in ("high", "critical"):
                        st.warning(f"Agent noticed: {issue['message']}")

            # Show multi-agent collaboration indicator
            if result.get("used_multi_agent"):
                st.info("Resource Agent provided supplementary materials for this explanation.")

            # Show spaced repetition info
            if result.get("is_review"):
                st.info("This was a spaced repetition review question.")

            # ── Student feedback on the question ──
            # Captures structurally-valid-but-pedagogically-poor quizzes
            # that the verifier missed. Flags feed into the IssueDetector
            # pattern check (3+ recent truthiness flags -> high-severity
            # alert) so the Diagnostics tab surfaces a quality regression.
            quiz_id = result.get("quiz_id")
            if quiz_id and not st.session_state.get(f"feedback_done_{quiz_id}"):
                with st.expander("Send feedback about this question (optional)", expanded=False):
                    fb_cols = st.columns(2)
                    with fb_cols[0]:
                        f_unclear = st.checkbox("Question was unclear", key=f"fb_uc_{quiz_id}")
                        f_wrong = st.checkbox("Answer key seems wrong", key=f"fb_wa_{quiz_id}")
                        f_unhelpful = st.checkbox("Explanation didn't help", key=f"fb_ue_{quiz_id}")
                    with fb_cols[1]:
                        f_easy = st.checkbox("Too easy", key=f"fb_te_{quiz_id}")
                        f_hard = st.checkbox("Too hard", key=f"fb_th_{quiz_id}")
                    if st.button("Submit feedback", key=f"fb_submit_{quiz_id}"):
                        flags = []
                        if f_unclear:    flags.append("unclear_question")
                        if f_wrong:      flags.append("wrong_answer")
                        if f_unhelpful:  flags.append("unhelpful_explanation")
                        if f_easy:       flags.append("too_easy")
                        if f_hard:       flags.append("too_hard")
                        if flags:
                            saved = tutor.record_feedback(quiz_id, flags)
                            if saved:
                                st.success("Thanks — feedback recorded.")
                                st.session_state[f"feedback_done_{quiz_id}"] = True
                            else:
                                st.warning("Couldn't save feedback (another tab may be writing). Try again.")
                        else:
                            st.info("Pick at least one flag, or skip.")

            # Show updated mastery
            if result.get("new_mastery"):
                st.divider()
                st.write("**Topic Mastery Updates:**")
                for topic, score in result["new_mastery"].items():
                    st.progress(score / 100, text=f"{topic}: {score}%")

            session_summary = build_quiz_session_summary(st.session_state.quiz_session_results)
            session_feedback = build_quiz_session_feedback(st.session_state.quiz_session_results)
            st.divider()
            if session_summary["ready"]:
                resource_material_suggestions = []
                resource = st.session_state.get("resource_agent")
                if resource and session_summary["priority_topics"]:
                    resource_material_suggestions = resource.suggest_materials_for_topics(
                        session_summary["priority_topics"],
                        limit_per_topic=1,
                    )

                st.markdown("### Overall Feedback Across All 5 Questions")
                metric_cols = st.columns(3)
                metric_cols[0].metric("Questions", session_summary["answered"])
                metric_cols[1].metric("Correct", session_summary["correct"])
                metric_cols[2].metric("Accuracy", f"{session_summary['accuracy']}%")

                if session_summary["accuracy"] >= 80:
                    st.success(session_summary["summary"])
                elif session_summary["accuracy"] >= 60:
                    st.warning(session_summary["summary"])
                else:
                    st.error(session_summary["summary"])

                st.write(f"**Tutor read:** {session_feedback['tutor_read']}")
                st.write(f"**Strengths:** {session_feedback['strengths']}")
                st.write(f"**Review priority:** {session_feedback['priority_feedback']}")
                st.write(f"**Confidence pattern:** {session_feedback['confidence_feedback']}")
                st.write(f"**Next step:** {session_feedback['next_step']}")
                if session_summary["priority_topics"]:
                    st.write("**Priority topics:** " + ", ".join(session_summary["priority_topics"]))
                if session_summary["misconception_topics"]:
                    st.write(
                        "**High-confidence misses:** "
                        + ", ".join(session_summary["misconception_topics"])
                    )

                session_report = build_quiz_session_report(
                    st.session_state.quiz_session_results
                )
                persisted_session_report = build_persisted_quiz_session(
                    st.session_state.quiz_session_results,
                    course=st.session_state.quiz_session_course or selected_course,
                    resource_materials=resource_material_suggestions,
                )
                if store_quiz_session_report(
                    st.session_state.profile,
                    persisted_session_report,
                ):
                    if not save_profile(
                        st.session_state.profile["name"],
                        st.session_state.profile,
                    ):
                        st.warning(
                            "The 5-question report is visible now, but it could not "
                            "be saved to disk. Try again after closing other tabs."
                        )
                st.markdown("### Question-by-Question Report")
                for row in session_report["rows"]:
                    status_text = "Correct" if row["correct"] else "Review"
                    with st.expander(
                        f"Question {row['number']}: {row['topic']} - {status_text}",
                        expanded=False,
                    ):
                        row_cols = st.columns(3)
                        row_cols[0].metric("Difficulty", f"{row['difficulty']}/5")
                        row_cols[1].metric("Confidence", f"{row['confidence']}/5")
                        row_cols[2].write("**Signal**")
                        row_cols[2].write(row["signal"])
                        st.write(f"**Your answer:** {row['student_answer']}")
                        st.write(f"**Correct answer:** {row['correct_answer']}")
                        if row.get("explanation"):
                            st.write(f"**Explanation:** {row['explanation']}")
                        st.write(f"**Next action:** {row['next_action']}")

                if session_summary["priority_topics"]:
                    render_resource_material_suggestions(
                        session_summary["priority_topics"],
                        suggestions=resource_material_suggestions,
                        caption=session_feedback["resource_reason"],
                        empty_message=(
                            "No cached Resource Agent materials yet for these priority topics. "
                            "Wrong answers during the session will create them automatically."
                        ),
                    )

                if st.button("Start Another 5-Question Quiz", use_container_width=True):
                    reset_quiz_ui_state()
                    st.session_state.quiz_session_active = True
                    st.session_state.quiz_session_course = selected_course
                    load_next_session_question(selected_course)
                    st.rerun()
            else:
                st.markdown("### Session feedback so far")
                st.info(session_feedback["tutor_read"])
                st.write(f"**Current review signal:** {session_feedback['priority_feedback']}")
                st.write(f"**Confidence pattern:** {session_feedback['confidence_feedback']}")
                if st.button("Next Question", use_container_width=True):
                    load_next_session_question(st.session_state.quiz_session_course or selected_course)
                    st.rerun()

# ──────────────────────────────────────────────
# Tab 3: Study Plan
# ──────────────────────────────────────────────
with tab_plan:
    st.subheader("Personalized Study Plan")
    st.caption("Generate a weekly plan from weak topics, mastery, and scheduled reviews.")

    # Show upcoming reviews
    upcoming = tutor.get_upcoming_reviews(days=7)
    if upcoming:
        st.write("**Upcoming spaced reviews this week:**")
        for item in upcoming:
            st.write(f"- {item['topic']} — in {item['days_until_review']:.0f} day(s)")
        st.divider()

    if st.button("Generate Study Plan", use_container_width=True, type="primary"):
        with st.spinner("Creating your personalized plan..."):
            plan = tutor.generate_study_plan()
        st.markdown(plan)

# ──────────────────────────────────────────────
# Tab 4: Progress Dashboard
# ──────────────────────────────────────────────
with tab_progress:
    st.subheader("Your Progress")

    summary = profile_summary

    if summary["total_quizzes"] == 0:
        st.info("No quiz results yet. Start a quiz session to build the progress view.")
    else:
        st.metric("Total Questions Answered", summary["total_quizzes"])

        for course_name, data in summary["courses"].items():
            st.divider()
            st.markdown(f"### {course_name}")

            col1, col2, col3 = st.columns(3)
            accuracy_label = f"{data['accuracy']}%" if data["attempted"] else "No data"
            col1.metric("Accuracy", accuracy_label)
            col2.metric("Questions", data["attempted"])
            col3.metric("Topics Covered", data["num_topics_covered"])

            if data["weak_topics"]:
                st.write("**Areas to improve:**")
                for wt in data["weak_topics"]:
                    acc_pct = round(wt["accuracy"] * 100, 1)
                    st.progress(wt["accuracy"], text=f"{wt['topic']}: {acc_pct}% ({wt['attempted']} questions)")

        # Mastery chart
        mastery = tutor.get_mastery_data()
        if mastery:
            st.divider()
            st.markdown("### Topic Mastery Overview")
            for topic, score in sorted(mastery.items(), key=lambda x: x[1]):
                st.progress(score / 100, text=f"{topic}: {score}%")

        # Confidence analysis
        st.divider()
        st.markdown("### Confidence Analysis")
        history = st.session_state.profile.get("quiz_history", [])
        if history:
            confident_wrong = sum(1 for h in history if h.get("confidence", 3) >= 4 and not h["correct"])
            unsure_right = sum(1 for h in history if h.get("confidence", 3) <= 2 and h["correct"])
            st.write(f"**Misconceptions detected** (confident but wrong): {confident_wrong}")
            st.write(f"**Lucky guesses** (unsure but right): {unsure_right}")
            if confident_wrong > 3:
                st.warning("Several misconceptions are visible. The tutor will prioritize clarification.")

        # Saved 5-question session reports
        visible_sessions = get_visible_quiz_sessions(st.session_state.profile)
        if visible_sessions:
            st.divider()
            st.markdown("### Latest 5-Question Evaluation")
            with st.expander("Open latest session report", expanded=True):
                render_quiz_session_snapshot(visible_sessions[0])

        # Recent quiz history
        st.divider()
        st.markdown("### Recent Quiz History")
        history = st.session_state.profile.get("quiz_history", [])[-10:]
        for entry in reversed(history):
            status = "Correct" if entry["correct"] else "Review"
            conf = entry.get("confidence", "?")
            st.write(f"**{status}** · **{entry['course']}** — {entry['topic']} "
                     f"(Diff: {entry['difficulty']}/5, Conf: {conf}/5) — "
                     f"{entry['question']}")

# ──────────────────────────────────────────────
# Tab 5: Diagnostics (for evaluation)
# ──────────────────────────────────────────────
with tab_diag:
    st.subheader("Agent Diagnostics")
    st.caption("Runtime state for agent communication, telemetry, and learning controls.")

    # Top-level metrics
    issue_summary = tutor.get_issue_summary()
    comm_stats = tutor.get_comm_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Issues Detected", issue_summary["total_issues"])
    col2.metric("API Errors", issue_summary["api_errors"])
    col3.metric("Agent Messages", comm_stats["total_messages"])
    col4.metric("Agents Active", len(comm_stats["registered_agents"]))

    visible_sessions = get_visible_quiz_sessions(st.session_state.profile)
    st.divider()
    st.markdown("### Quiz Session Reports")
    if visible_sessions:
        latest = visible_sessions[0]
        s_cols = st.columns(3)
        s_cols[0].metric("Reports Available", len(visible_sessions))
        s_cols[1].metric("Latest Accuracy", f"{latest.get('accuracy', 0)}%")
        s_cols[2].metric("Latest Questions", latest.get("question_count", 0))
        with st.expander("Inspect latest 5-question evaluation", expanded=False):
            render_quiz_session_snapshot(latest)
    else:
        st.info("No completed 5-question session report yet.")

    # ─── Multi-Agent Communication Log ───
    st.divider()
    st.markdown("### Agent Communication Log")
    st.caption("Messages exchanged between the Tutor Agent and Resource Agent.")

    msg_log = tutor.get_message_log()
    if msg_log:
        # Message type stats
        type_counts = comm_stats.get("by_type", {})
        if type_counts:
            type_cols = st.columns(len(type_counts))
            for i, (mtype, count) in enumerate(type_counts.items()):
                type_icons = {
                    "request_materials": "REQ",
                    "provide_materials": "MAT",
                    "report_weakness": "WEAK",
                    "suggest_strategy": "PLAN",
                    "status_update": "STAT",
                }
                type_cols[i].metric(f"{type_icons.get(mtype, 'MSG')} {mtype}", count)

        # Show recent messages
        st.write("")
        for msg in reversed(msg_log[-15:]):
            if "error" in msg:
                st.error(msg["error"])
                continue
            direction = "→" if msg.get("sender") == "TutorAgent" else "←"
            icon = {"request_materials": "REQ", "provide_materials": "MAT",
                    "report_weakness": "WEAK", "suggest_strategy": "PLAN"
                    }.get(msg.get("type", ""), "MSG")

            sender = msg.get("sender", "?")
            receiver = msg.get("receiver", "?")
            content = msg.get("content", {})

            with st.expander(f"{icon} {sender} {direction} {receiver} — {msg.get('type', '?')}", expanded=False):
                # Show key content fields (not the full blob)
                if msg.get("type") == "request_materials":
                    st.write(f"**Topic:** {content.get('topic', '?')}")
                    st.write(f"**Context:** {content.get('context', 'N/A')[:120]}...")
                elif msg.get("type") == "provide_materials":
                    materials = content.get("materials", [])
                    rationale = content.get("rationale", {}) or {}
                    st.write(f"**Materials sent:** {len(materials)}")
                    st.write(f"**From cache:** {content.get('from_cache', False)}")
                    if rationale:
                        st.write(
                            "**Why shown:** "
                            + rationale.get(
                                "why_shown",
                                "A wrong quiz answer made this topic a review priority.",
                            )
                        )
                        st.write(
                            "**Quiz evidence:** "
                            + rationale.get(
                                "quiz_evidence",
                                rationale.get(
                                    "student_need",
                                    "The quiz result showed this topic needs review.",
                                ),
                            )
                        )
                        st.write(
                            "**Generation:** "
                            + rationale.get(
                                "generation_source",
                                rationale.get(
                                    "source",
                                    "ResourceAgent generated or reused cached material.",
                                ),
                            )
                        )
                    if materials:
                        st.write(f"**Top material:** {materials[0].get('title', '?')}")
                elif msg.get("type") == "report_weakness":
                    st.write(f"**Topic:** {content.get('topic', '?')}")
                    st.write(f"**Accuracy:** {content.get('accuracy', 0)*100:.0f}%")
                    st.write(f"**Attempts:** {content.get('attempts', 0)}")
                elif msg.get("type") == "suggest_strategy":
                    st.write(f"**Strategy:** {content.get('strategy', '?')}")
                    st.write(f"**Suggestion:** {content.get('suggestion', '')}")
                    pattern = content.get("pattern_detected")
                    if pattern:
                        st.warning(f"**Pattern detected:** {pattern}")
                st.caption(msg.get("timestamp", ""))
    else:
        st.info("No agent messages yet. When you answer quiz questions wrong, the Tutor Agent will consult the Resource Agent for better explanations.")

    # ─── Resource Agent Knowledge Base ───
    st.divider()
    st.markdown("### Resource Agent Knowledge Base")
    resource = st.session_state.resource_agent
    if resource:
        kb_stats = resource.get_knowledge_base_stats()
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Topics Researched", kb_stats["topics_researched"])
        rcol2.metric("Material Uses", kb_stats["total_requests"])
        rcol3.metric("Weakness Reports", kb_stats["weakness_reports_received"])

        weakness_history = getattr(resource, "weakness_history", []) or []
        if weakness_history:
            with st.expander("Weakness Report History", expanded=False):
                for idx, report in enumerate(reversed(weakness_history[-10:]), start=1):
                    topic = report.get("topic", "?")
                    course = report.get("course", "?")
                    accuracy = report.get("accuracy", 0)
                    attempts = report.get("attempts", 0)
                    try:
                        accuracy_label = f"{float(accuracy) * 100:.0f}%"
                    except (TypeError, ValueError):
                        accuracy_label = str(accuracy)
                    st.write(
                        f"**{idx}. {topic}** ({course}) - "
                        f"accuracy {accuracy_label}, attempts {attempts}"
                    )

        if kb_stats["topics"]:
            for topic, data in kb_stats["topics"].items():
                prereqs = ", ".join(data["prerequisites"]) if data["prerequisites"] else "none identified"
                st.write(f"**{topic}**: {data['times_requested']}x used, "
                         f"{data['materials_count']} materials, prerequisites: {prereqs}")
                last_request = data.get("last_request", {}) or {}
                if last_request:
                    st.caption(
                        "Last quiz reason: "
                        + last_request.get(
                            "why_shown",
                            "A wrong quiz answer made this topic a review priority.",
                        )
                    )
                render_resource_material_suggestions(
                    [topic],
                )
    else:
        st.info("Resource Agent not initialized.")

    # ─── Issue Severity Breakdown ───
    if issue_summary["total_issues"] > 0:
        st.divider()
        st.markdown("### Issues by Severity")
        for sev, count in issue_summary["by_severity"].items():
            if count > 0:
                color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                st.write(f"{color.get(sev, '⚪')} **{sev.capitalize()}**: {count}")

    # ─── Recent Issues Log ───
    st.divider()
    st.markdown("### Recent Issue Log")
    recent = tutor.get_recent_issues(10)
    if recent:
        for issue in reversed(recent):
            severity_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
            icon = severity_icon.get(issue["severity"], "⚪")
            st.write(f"{icon} **[{issue['type']}]** {issue['message']}")
            st.write(f"   → Fix: {issue['suggested_fix']}")
            st.caption(issue['timestamp'])
            st.write("")
    else:
        st.info("No issues detected yet.")

    # ─── Observability Counters ───
    # Process-singleton counters from utils.telemetry. Read in
    # COUNTER_DISPLAY_ORDER so the table layout is stable across reruns.
    st.divider()
    st.markdown("### 📈 Observability Counters")
    st.caption("Monotonic counts since process start — useful for spotting LLM flakiness, cache effectiveness, and persistence failures.")
    counters = COUNTERS.snapshot(COUNTER_DISPLAY_ORDER)
    if any(v > 0 for v in counters.values()):
        # Render in a 3-wide grid so the diagnostics tab doesn't sprawl.
        items = list(counters.items())
        rows = (len(items) + 2) // 3
        for r in range(rows):
            cols = st.columns(3)
            for c, col in enumerate(cols):
                idx = r * 3 + c
                if idx >= len(items):
                    break
                name, value = items[idx]
                # Pretty-print the dotted counter name.
                label = name.replace(".", " · ").replace("_", " ")
                col.metric(label, value)
    else:
        st.info("No instrumented events yet — counters appear after the first quiz / chat turn.")

    # ─── RL Engine State ───
    st.divider()
    st.markdown("### RL Engine State")
    st.write(f"**Exploration rate (epsilon):** {tutor.adaptive_engine.epsilon:.4f}")
    st.write(f"**Arms tracked:** {len(tutor.adaptive_engine.arm_stats)}")
    st.write(f"**Topics tracked:** {len(tutor.adaptive_engine.topic_accuracy)}")

    # ─── Spaced Repetition State ───
    st.divider()
    st.markdown("### Spaced Repetition Schedule")
    sr_data = tutor.scheduler.topic_schedule
    if sr_data:
        for topic, sched in sr_data.items():
            ef = sched.get("easiness_factor", 2.5)
            interval = sched.get("interval_days", 0)
            reps = sched.get("repetition_count", 0)
            st.write(f"**{topic}**: EF={ef:.2f}, interval={interval}d, reps={reps}")
    else:
        st.info("No spaced repetition data yet.")
