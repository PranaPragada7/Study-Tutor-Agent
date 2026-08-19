"""Study Tutor's paper-and-ink visual system and application header."""

from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
:root {
    --paper: #fffaf0;
    --paper-deep: #f7f2e8;
    --ink: #17223b;
    --muted: #667085;
    --rule: #d8cdbb;
    --coral: #e4573d;
    --coral-soft: #fde8df;
    --gold: #f2b84b;
    --navy: #17223b;
}

.stApp {
    background-color: var(--paper-deep);
    background-image:
        linear-gradient(rgba(23, 34, 59, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23, 34, 59, 0.035) 1px, transparent 1px);
    background-size: 28px 28px;
    color: var(--ink);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

section[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 4px solid var(--gold);
}

section[data-testid="stSidebar"] * {
    color: #f8fafc;
}

section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    color: var(--ink) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: #fffaf0;
    color: var(--ink);
    border-color: rgba(255, 255, 255, 0.22);
}

section[data-testid="stSidebar"] .stButton > button * {
    color: var(--ink) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.16);
}

.app-header {
    background: var(--navy);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 22px 22px 22px 6px;
    padding: 28px 30px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: center;
    box-shadow: 0 14px 34px rgba(23, 34, 59, 0.16);
    position: relative;
    overflow: hidden;
}

.app-header::after {
    content: "";
    position: absolute;
    width: 170px;
    height: 170px;
    right: -55px;
    top: -75px;
    border: 28px solid rgba(242, 184, 75, 0.18);
    border-radius: 50%;
}

.app-header > * {
    position: relative;
    z-index: 1;
}

.app-kicker {
    color: var(--gold);
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.app-title {
    margin: 0;
    color: #fffaf0 !important;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.15rem, 4vw, 3.35rem);
    line-height: 1;
    font-weight: 700;
    letter-spacing: -0.035em;
}

.app-subtitle {
    margin: 11px 0 0 0;
    color: #cbd5e1;
    font-size: 1rem;
}

.status-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.status-pill {
    border: 1px solid rgba(255, 255, 255, 0.18);
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    padding: 7px 11px;
    color: #f8fafc;
    font-size: 0.83rem;
    font-weight: 650;
    white-space: nowrap;
    backdrop-filter: blur(8px);
}

.status-pill.accent {
    border-color: rgba(242, 184, 75, 0.55);
    background: rgba(242, 184, 75, 0.16);
    color: #ffe6a6;
}

div[data-testid="stMetric"] {
    background: var(--paper);
    border: 1px solid var(--rule);
    border-top: 4px solid var(--coral);
    border-radius: 4px 14px 14px 14px;
    padding: 14px 16px;
    box-shadow: 0 8px 18px rgba(23, 34, 59, 0.06);
}

div[data-testid="stMetric"] label {
    color: var(--muted);
    font-weight: 650;
}

div[data-testid="stMetricValue"] {
    color: var(--ink);
    font-weight: 760;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--rule);
    font-weight: 700;
    min-height: 2.55rem;
    transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}

.stButton > button:hover {
    border-color: var(--coral);
    box-shadow: 0 5px 14px rgba(228, 87, 61, 0.15);
    transform: translateY(-1px);
}

div[data-testid="stExpander"] {
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--rule);
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255, 250, 240, 0.82);
    border: 1px solid var(--rule);
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
}

div[data-testid="stTabs"] button {
    font-weight: 700;
    color: #4c566a;
    border-radius: 9px;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff;
    background: var(--navy);
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 10px;
}

hr {
    border-color: var(--rule);
    margin: 1.15rem 0;
}

.onboarding-card {
    background: rgba(255, 250, 240, 0.92);
    border: 1px solid var(--rule);
    border-radius: 4px 16px 16px 16px;
    padding: 24px;
    min-height: 168px;
    box-shadow: 0 10px 24px rgba(23, 34, 59, 0.06);
}

.onboarding-number {
    display: inline-grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--coral-soft);
    color: #a93624;
    font-weight: 800;
    margin-bottom: 16px;
}

.onboarding-card h3 {
    color: var(--ink);
    margin: 0 0 8px;
    font-family: Georgia, "Times New Roman", serif;
}

.onboarding-card p {
    color: var(--muted);
    margin: 0;
    line-height: 1.55;
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
"""

APP_HEADER = """
<div class="app-header">
    <div>
        <div class="app-kicker">Adaptive learning workspace</div>
        <h1 class="app-title">Study Tutor</h1>
        <p class="app-subtitle">Plan with purpose. Practice what matters. See your progress.</p>
    </div>
    <div class="status-row">
        <span class="status-pill accent">Adaptive practice</span>
        <span class="status-pill">Spaced review</span>
        <span class="status-pill accent">Claude assistant</span>
    </div>
</div>
"""


def render_app_shell() -> None:
    """Render the global styles and branded header once per Streamlit run."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(APP_HEADER, unsafe_allow_html=True)
