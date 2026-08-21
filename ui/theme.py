"""Professional visual system and application shell for Study Tutor."""

from __future__ import annotations

from html import escape

import streamlit as st

from services.runtime import RuntimeConfig

APP_CSS = """
<style>
:root {
    --canvas: #f4f7fb;
    --surface: #ffffff;
    --surface-soft: #f8faff;
    --ink: #15213b;
    --muted: #667085;
    --line: #e2e8f2;
    --brand: #3b5ccc;
    --brand-deep: #1d2b5b;
    --teal: #0f766e;
    --teal-soft: #e8f7f4;
    --shadow: 0 16px 44px rgba(24, 39, 75, 0.08);
}

.stApp {
    background:
        radial-gradient(circle at 92% 3%, rgba(59, 92, 204, 0.10), transparent 24rem),
        var(--canvas);
    color: var(--ink);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1380px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17234a 0%, #111a35 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] * {
    color: #eef2ff;
}

section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    color: var(--ink) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.96);
    color: var(--brand-deep);
    border-color: rgba(255, 255, 255, 0.2);
}

section[data-testid="stSidebar"] .stButton > button * {
    color: var(--brand-deep) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.12);
}

.app-header {
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 2rem;
    min-height: 190px;
    padding: 34px 38px;
    margin-bottom: 14px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    background: linear-gradient(118deg, rgba(29, 43, 91, 0.98), rgba(47, 74, 166, 0.94));
    box-shadow: 0 22px 60px rgba(29, 43, 91, 0.18);
}

.app-header::before,
.app-header::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.app-header::before {
    width: 270px;
    height: 270px;
    right: -70px;
    top: -125px;
}

.app-header::after {
    width: 155px;
    height: 155px;
    right: 55px;
    bottom: -115px;
}

.app-header > * {
    position: relative;
    z-index: 1;
}

.app-kicker {
    color: #a9d6ff;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.app-title {
    margin: 0;
    color: #ffffff !important;
    font-size: clamp(2.35rem, 4vw, 3.6rem);
    line-height: 1;
    font-weight: 780;
    letter-spacing: -0.045em;
}

.app-subtitle {
    max-width: 660px;
    margin: 14px 0 0;
    color: #dbe5ff;
    font-size: 1.04rem;
    line-height: 1.55;
}

.status-row {
    display: flex;
    justify-content: flex-end;
    gap: 9px;
    flex-wrap: wrap;
    max-width: 390px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 12px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.09);
    color: #f8fbff;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
    backdrop-filter: blur(8px);
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: #5eead4;
    box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.14);
}

.runtime-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 22px;
    padding: 12px 15px;
    border: 1px solid #cde9e3;
    border-radius: 13px;
    background: var(--teal-soft);
    color: #155e58;
    font-size: 0.9rem;
}

.runtime-banner strong {
    color: #115e59;
    white-space: nowrap;
}

div[data-testid="stMetric"] {
    min-height: 112px;
    padding: 17px 18px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: var(--surface);
    box-shadow: 0 9px 28px rgba(24, 39, 75, 0.055);
}

div[data-testid="stMetric"] label {
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 720;
    letter-spacing: 0.035em;
    text-transform: uppercase;
}

div[data-testid="stMetricValue"] {
    color: var(--ink);
    font-weight: 760;
}

.stButton > button {
    min-height: 2.7rem;
    border: 1px solid #d8dfec;
    border-radius: 11px;
    font-weight: 720;
    transition: border-color 130ms ease, box-shadow 130ms ease, transform 130ms ease;
}

.stButton > button:hover {
    border-color: var(--brand);
    box-shadow: 0 7px 20px rgba(59, 92, 204, 0.15);
    transform: translateY(-1px);
}

div[data-testid="stExpander"],
div[data-testid="stAlert"],
div[data-testid="stChatMessage"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: var(--surface);
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px;
    padding: 6px;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.88);
    box-shadow: 0 8px 24px rgba(24, 39, 75, 0.04);
}

div[data-testid="stTabs"] button {
    padding: 9px 16px;
    border-radius: 9px;
    color: #526078;
    font-weight: 720;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff;
    background: var(--brand-deep);
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 10px;
}

hr {
    border-color: var(--line);
    margin: 1.2rem 0;
}

.section-intro {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: flex-end;
    margin: 4px 0 18px;
}

.section-eyebrow {
    color: var(--brand);
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.section-intro h2 {
    margin: 5px 0 0;
    color: var(--ink);
    font-size: 1.65rem;
    letter-spacing: -0.025em;
}

.section-intro p {
    max-width: 560px;
    margin: 0;
    color: var(--muted);
    line-height: 1.55;
}

.onboarding-card,
.insight-card {
    height: 100%;
    padding: 24px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.93);
    box-shadow: var(--shadow);
}

.onboarding-number {
    display: inline-grid;
    place-items: center;
    width: 38px;
    height: 38px;
    margin-bottom: 18px;
    border-radius: 11px;
    background: #e9eeff;
    color: var(--brand);
    font-weight: 820;
}

.onboarding-card h3,
.insight-card h3 {
    margin: 0 0 8px;
    color: var(--ink);
    font-size: 1.04rem;
}

.onboarding-card p,
.insight-card p {
    margin: 0;
    color: var(--muted);
    line-height: 1.58;
}

.insight-card {
    box-shadow: 0 9px 28px rgba(24, 39, 75, 0.055);
}

.insight-label {
    margin-bottom: 9px;
    color: var(--teal);
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

@media (max-width: 820px) {
    .app-header,
    .section-intro {
        align-items: flex-start;
        flex-direction: column;
    }
    .app-header {
        padding: 28px 25px;
    }
    .status-row {
        justify-content: flex-start;
    }
    .runtime-banner {
        align-items: flex-start;
        flex-wrap: wrap;
    }
}
</style>
"""


def render_app_shell(runtime: RuntimeConfig) -> None:
    """Render global styling, runtime status, and the product header."""

    mode_label = escape(runtime.label)
    storage_label = escape(runtime.storage_label)
    description = escape(runtime.description)
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-kicker">Personal learning command center</div>
                <h1 class="app-title">Study Tutor</h1>
                <p class="app-subtitle">
                    Turn practice into a focused plan with adaptive quizzes,
                    spaced review, and progress you can explain.
                </p>
            </div>
            <div class="status-row">
                <span class="status-pill"><span class="status-dot"></span>{mode_label}</span>
                <span class="status-pill">{storage_label}</span>
                <span class="status-pill">Private by default</span>
            </div>
        </div>
        <div class="runtime-banner">
            <span class="status-dot"></span>
            <strong>{mode_label}</strong>
            <span>{description}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["APP_CSS", "render_app_shell"]
