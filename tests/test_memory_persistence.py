"""Durable chatbot and learning-memory tests."""

from __future__ import annotations

from types import SimpleNamespace

from agents.tutor_agent import TutorAgent
from utils.student_profile import (
    clear_chat_history,
    create_profile,
    load_profile,
    record_quiz_result,
    save_profile,
)


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text="Review Recursion, then practice one quiz.")]
        )


class _FakeClaude:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def _profile(monkeypatch, tmp_path) -> dict:
    import utils.student_profile as profile_module

    monkeypatch.setattr(profile_module, "DATA_DIR", str(tmp_path))
    return create_profile(
        "Memory Student",
        [{"name": "Computer Science", "difficulty": 3}],
    )


def test_chat_exchanges_survive_profile_reload(monkeypatch, tmp_path):
    profile = _profile(monkeypatch, tmp_path)
    client = _FakeClaude()
    tutor = TutorAgent(profile, client=client)

    response = tutor.chat("What should I revise next?")
    loaded = load_profile("Memory Student")

    assert loaded is not None
    assert loaded["chat_history"][-1]["user_message"] == "What should I revise next?"
    assert loaded["chat_history"][-1]["assistant_response"] == response
    assert client.messages.calls

    restored_tutor = TutorAgent(loaded, client=None)
    assert restored_tutor.conversation_history[-2:] == [
        {"role": "user", "content": "What should I revise next?"},
        {"role": "assistant", "content": response},
    ]


def test_strengths_and_weaknesses_are_in_chatbot_context(monkeypatch, tmp_path):
    profile = _profile(monkeypatch, tmp_path)
    for index in range(4):
        record_quiz_result(
            profile,
            "Computer Science",
            "Loops",
            3,
            True,
            f"Loops question {index}",
            "A",
            confidence=4,
        )
    for index in range(3):
        record_quiz_result(
            profile,
            "Computer Science",
            "Recursion",
            3,
            False,
            f"Recursion question {index}",
            "B",
            confidence=4,
        )
    assert save_profile(profile["name"], profile)

    restored = load_profile("Memory Student")
    assert restored is not None
    context = TutorAgent(restored, client=None)._build_dynamic_student_context()

    assert "strong areas: Loops" in context
    assert "weak areas: Recursion" in context
    assert "RECENT QUIZ ANSWERS" in context
    assert "High-confidence wrong answers: 3" in context


def test_quiz_answer_and_tutor_response_are_persisted(monkeypatch, tmp_path):
    profile = _profile(monkeypatch, tmp_path)
    tutor = TutorAgent(profile, client=None)
    quiz = {
        "course": "Computer Science",
        "topic": "Recursion",
        "difficulty": 3,
        "question": "What prevents infinite recursion?",
        "options": ["A) A base case", "B) A loop", "C) A class", "D) A file"],
        "correct_answer": "A",
        "explanation": "A base case stops the recursive calls.",
    }

    result = tutor.evaluate_answer(quiz, "B", confidence=5)
    loaded = load_profile("Memory Student")

    assert loaded is not None
    entry = loaded["quiz_history"][-1]
    assert entry["student_answer"] == "B"
    assert entry["correct_answer"] == "A"
    assert entry["explanation"] == quiz["explanation"]
    assert entry["tutor_response"] == result["explanation"]
    assert entry["student_feedback"] == result["student_feedback"]


def test_clearing_chat_history_is_durable(monkeypatch, tmp_path):
    profile = _profile(monkeypatch, tmp_path)
    TutorAgent(profile, client=None).chat("Remember this exchange")

    loaded = load_profile("Memory Student")
    assert loaded is not None and loaded["chat_history"]
    clear_chat_history(loaded)
    assert save_profile(loaded["name"], loaded)

    reloaded = load_profile("Memory Student")
    assert reloaded is not None
    assert reloaded["chat_history"] == []
