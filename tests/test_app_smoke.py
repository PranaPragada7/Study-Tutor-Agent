"""
Streamlit AppTest smoke tests for app.py.

These are intentionally minimal — the goal is to catch UI-layer
regressions that today only surface when a human runs the app:

  - app.py imports without raising
  - the sidebar renders the Student Profile section
  - missing ``ANTHROPIC_API_KEY`` activates the complete local demo runtime
  - the create-profile flow wires through to a usable session state
    (profile populated, tutor instantiated, no exceptions)

Full offline quiz, chat, study-plan, and persistence workflows are exercised
through AppTest in tests/test_app_workflows.py without live LLM calls.

Run with: pytest tests/test_app_smoke.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture
def temp_data_dir(monkeypatch):
    """Redirect DATA_DIR to a tempdir for each test so we don't touch
    the real student profiles in data/."""
    import utils.student_profile as sp

    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(sp, "DATA_DIR", tmp)
    monkeypatch.setattr(sp, "_LOCK_TIMEOUT_SECONDS", 1)
    yield tmp


@pytest.fixture
def fake_api_key(monkeypatch):
    """Set a fake API key so get_anthropic_client constructs a client
    without hitting the network. The Anthropic SDK constructor does NOT
    make a network call, so this is safe — but the client must never
    be USED in these tests (we don't trigger any messages.create call)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key-for-testing")


# ═══════════════════════════════════════════════
# 1. Bare-bones load
# ═══════════════════════════════════════════════


def test_app_loads_without_exception(temp_data_dir, fake_api_key):
    """The app must run end-to-end without raising before any user input."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    assert not at.exception, f"App raised on initial load: {at.exception}"

    # Sidebar should render the profile header.
    headers = [h.value for h in at.sidebar.header]
    assert any("Student Profile" in h for h in headers)

    # Main content should be the onboarding workspace, not the learning tabs.
    markdown = [item.value for item in at.markdown]
    assert any("Your learning workspace is ready" in value for value in markdown)
    assert not at.tabs


# ═══════════════════════════════════════════════
# 2. Missing API key surfaces friendly error
# ═══════════════════════════════════════════════


def test_missing_api_key_uses_local_demo_without_traceback(temp_data_dir, monkeypatch):
    """
    Without ANTHROPIC_API_KEY, the app should remain fully usable and explain
    that the local deterministic tutor is active.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    assert not at.exception, f"App load should not raise without API key, got: {at.exception}"
    assert not at.error
    markdown = [item.value for item in at.markdown]
    assert any("Local demo" in value for value in markdown)


# ═══════════════════════════════════════════════
# 3. Create-profile flow lights up the main tabs
# ═══════════════════════════════════════════════


def test_create_profile_flow(temp_data_dir, fake_api_key):
    """
    Walk the new-profile path:
      1. Type a name in the sidebar.
      2. Type one course in the textarea.
      3. Click Create Profile.
      4. Verify session_state.profile is populated and the main info
         hint is gone (i.e. the tabs are now showing).
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)
    assert not at.exception

    # Sidebar inputs are ordered: text_input[0]=name, text_area[0]=courses
    name_inputs = [t for t in at.sidebar.text_input if "Name" in (t.label or "")]
    assert name_inputs, "Couldn't find the Name text_input in the sidebar"
    name_inputs[0].set_value("SmokeTest").run(timeout=10)
    assert not at.exception

    course_areas = at.sidebar.text_area
    assert course_areas, "Couldn't find the Courses text_area in the sidebar"
    course_areas[0].set_value("CS101").run(timeout=10)
    assert not at.exception

    # The create-profile button is on the sidebar; click it.
    create_buttons = [b for b in at.sidebar.button if "Create Profile" in (b.label or "")]
    assert create_buttons, "Couldn't find the Create Profile button"
    create_buttons[0].click().run(timeout=10)
    assert not at.exception, f"Create Profile flow raised: {at.exception}"

    # Profile must be populated in session_state. Streamlit's
    # session_state is a Mapping that doesn't support .get(); use
    # __contains__ + __getitem__.
    assert "profile" in at.session_state
    profile = at.session_state["profile"]
    assert profile is not None
    assert profile["name"] == "SmokeTest"
    assert "CS101" in profile["courses"]
    assert profile["total_quizzes"] == 0

    # Tutor agent must be wired up.
    assert "tutor" in at.session_state
    tutor = at.session_state["tutor"]
    assert tutor is not None
    # Tutor should have the same profile dict (by reference).
    assert tutor.profile is profile

    # Main-content "enter your name" hint should be gone.
    info_messages = [i.value for i in at.info]
    assert not any("Enter your name" in m for m in info_messages), (
        "Expected the create-profile flow to dismiss the name prompt"
    )


def test_sidebar_lists_saved_profiles(temp_data_dir, fake_api_key):
    """Saved profiles should be visible without typing the exact name."""
    from utils.student_profile import create_profile

    create_profile(
        "Visible Student",
        [{"name": "Georgia Tech CS 7641 - Machine Learning", "difficulty": 3}],
    )

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)
    assert not at.exception

    markdown = [m.value for m in at.sidebar.markdown]
    assert any("Existing Profiles" in m for m in markdown)

    saved_selectboxes = [s for s in at.sidebar.selectbox if "Saved profiles" in (s.label or "")]
    assert saved_selectboxes, "Saved profile selector was not rendered"
    assert "Visible Student - 0 quizzes, 1 course" in saved_selectboxes[0].options

    load_buttons = [b for b in at.sidebar.button if "Load Selected Profile" in (b.label or "")]
    assert load_buttons, "Saved profile load button was not rendered"


def test_course_input_explains_ambiguous_course_numbers(temp_data_dir, fake_api_key):
    """Course setup should tell students to include school/title context."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)
    assert not at.exception

    captions = [c.value for c in at.sidebar.caption]
    assert any("university, course number, and course title" in c for c in captions)

    course_areas = at.sidebar.text_area
    assert course_areas, "Couldn't find the Courses text_area in the sidebar"
    placeholder = course_areas[0].placeholder or ""
    assert "Georgia Tech CS 7641 - Machine Learning" in placeholder


def test_sample_profile_button_loads_existing_sample(temp_data_dir, fake_api_key):
    """The sidebar sample shortcut should load Sample Student without typing."""
    from utils.student_profile import create_profile

    create_profile(
        "Sample Student",
        [{"name": "Intro to CS", "difficulty": 3, "exam_date": ""}],
        learning_style="balanced",
    )

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)
    assert not at.exception

    sample_buttons = [b for b in at.sidebar.button if "Load Sample Student" in (b.label or "")]
    assert sample_buttons, "Couldn't find the Load Sample Student button"

    sample_buttons[0].click().run(timeout=10)
    assert not at.exception, f"Sample profile load raised: {at.exception}"

    assert "profile" in at.session_state
    profile = at.session_state["profile"]
    assert profile is not None
    assert profile["name"] == "Sample Student"
    assert "Intro to CS" in profile["courses"]

    assert "tutor" in at.session_state
    assert at.session_state["tutor"] is not None
