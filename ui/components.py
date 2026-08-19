"""Reusable report and resource components for the Streamlit interface."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st


def render_resource_material_suggestions(
    topics: list[str],
    *,
    suggestions: list[dict] | None = None,
    caption: str | None = None,
    empty_message: str | None = None,
    suggestion_loader: Callable[[list[str]], list[dict]] | None = None,
) -> bool:
    """Render Resource Agent materials for the supplied assessed topics."""
    if suggestions is None:
        if not suggestion_loader or not topics:
            return False
        suggestions = suggestion_loader(topics)
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
                        "Resource Agent generated or reused this material after evaluating the quiz answers.",
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
                st.caption("Prerequisites: " + ", ".join(suggestion["prerequisites"]))
    return True


def render_quiz_session_snapshot(
    session: dict,
    *,
    expanded_rows: bool = False,
    suggestion_loader: Callable[[list[str]], list[dict]] | None = None,
) -> None:
    """Render a saved or reconstructed five-question session report."""
    course = session.get("course", "Unknown course")
    created_at = session.get("created_at", "")
    source = session.get("source", "")
    st.caption(f"{course} | {created_at}")
    if source == "history_reconstruction":
        st.info(
            session.get(
                "note",
                "Restored from saved quiz answers. Some old report details were not stored.",
            )
        )

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
            suggestion_loader=suggestion_loader,
        )
