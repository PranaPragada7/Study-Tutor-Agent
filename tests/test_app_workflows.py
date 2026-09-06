"""Exercise the real offline tutor through its Streamlit user workflows."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.student_profile import create_profile, load_profile

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def click(app, label):
    next(button for button in app.button if button.label == label).click().run(timeout=15)
    assert not app.exception


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    import utils.student_profile as profiles

    monkeypatch.setattr(profiles, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STUDY_TUTOR_MODE", "demo")
    monkeypatch.setenv("STUDY_TUTOR_STORAGE", "sqlite")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    create_profile("UI Student", [{"name": "Computer Science", "difficulty": 3}])
    app = AppTest.from_file(APP_PATH).run(timeout=15)
    click(app, "Load Selected Profile")
    return app


@pytest.mark.parametrize("correct_count", [5, 2])
def test_five_question_session_persists_results_and_feedback(workspace, correct_count):
    app = workspace
    click(app, "Start 5-Question Quiz")
    for index in range(5):
        quiz = app.session_state["current_quiz"]
        correct = quiz["correct_answer"]
        option = next(
            option
            for option in quiz["options"]
            if (option[0] == correct) == (index < correct_count)
        )
        click(app, option)
        assert len(app.session_state["quiz_session_results"]) == index + 1
        assert app.session_state["quiz_result"]["correct"] == (index < correct_count)
        if index == 0:
            next(box for box in app.checkbox if box.label == "Question was unclear").check()
            click(app, "Submit feedback")
        if index < 4:
            click(app, "Next Question")

    assert app.session_state["quiz_session_complete"] is True
    assert any("Overall Feedback Across All 5 Questions" in item.value for item in app.markdown)
    persisted = load_profile("UI Student")
    assert persisted["total_quizzes"] == 5
    assert len(persisted["quiz_sessions"]) == 1
    assert sum(entry["correct"] for entry in persisted["quiz_history"]) == correct_count
    assert any(
        "unclear_question" in entry.get("feedback_flags", []) for entry in persisted["quiz_history"]
    )

    # A rerun must not append the same completed report again.
    app.run(timeout=15)
    assert len(load_profile("UI Student")["quiz_sessions"]) == 1
    click(app, "Generate Study Plan")
    assert not app.exception
    click(app, "Start Another 5-Question Quiz")
    assert app.session_state["quiz_session_complete"] is False
    assert app.session_state["current_quiz"] is not None
    click(app, "Reset Session")
    assert app.session_state["current_quiz"] is None
    assert load_profile("UI Student")["total_quizzes"] == 5


def test_chat_survives_reload_and_can_be_cleared(workspace):
    app = workspace
    app.chat_input[0].set_value("Explain recursion").run(timeout=15)
    assert not app.exception
    assert len(app.session_state["chat_messages"]) == 2
    assert load_profile("UI Student")["chat_history"]

    reloaded = AppTest.from_file(APP_PATH).run(timeout=15)
    click(reloaded, "Load Selected Profile")
    assert len(reloaded.session_state["chat_messages"]) == 2
    click(reloaded, "Clear chat history")
    assert load_profile("UI Student")["chat_history"] == []
