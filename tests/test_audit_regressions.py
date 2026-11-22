"""Regression checks for reset persistence, runtime safety, and SQLite cleanup."""

import sqlite3
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from services.runtime import get_runtime_config
from utils.profile_store import SQLiteProfileStore
from utils.student_profile import (
    create_profile,
    load_profile,
    record_quiz_result,
    reset_quiz_history,
    save_profile,
)


@pytest.mark.parametrize("mode", ["unsupported", "", "lvei"])
def test_invalid_runtime_mode_never_enables_live_calls(monkeypatch, mode):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 80)
    monkeypatch.setenv("STUDY_TUTOR_MODE", mode)
    assert get_runtime_config().llm_enabled is False


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("stale_write", [False, True])
def test_learning_reset_survives_reload_and_stale_tab_save(
    monkeypatch, tmp_path, backend, stale_write
):
    import utils.student_profile as profiles

    monkeypatch.setenv("STUDY_TUTOR_STORAGE", backend)
    monkeypatch.setattr(profiles, "DATA_DIR", str(tmp_path))
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    reset_at = started + timedelta(days=1)
    profile = create_profile("Reset Student", [{"name": "CS", "difficulty": 3}])
    record_quiz_result(profile, "CS", "Loops", 3, True, "old", "A", now_fn=lambda: started)
    profile["quiz_sessions"] = [{"id": "old", "created_at": started.isoformat()}]
    profile["resource_agent_state"] = {"knowledge_base": {"Loops": {"times_requested": 1}}}
    profile["spaced_repetition"] = {"Loops": {"interval": 10}}
    assert save_profile(profile["name"], profile)
    stale = deepcopy(profile)

    reset_quiz_history(profile, now_fn=lambda: reset_at)
    assert save_profile(profile["name"], profile)
    if stale_write:
        record_quiz_result(
            stale,
            "CS",
            "Loops",
            3,
            False,
            "new",
            "B",
            now_fn=lambda: reset_at + timedelta(minutes=1),
        )
        assert save_profile(stale["name"], stale)

    restored = load_profile(profile["name"])
    assert restored["total_quizzes"] == int(stale_write)
    assert [entry["question"] for entry in restored["quiz_history"]] == (
        ["new"] if stale_write else []
    )
    assert restored["courses"]["CS"]["total_correct"] == 0
    assert restored["courses"]["CS"]["topics"]["Loops"]["attempted"] == int(stale_write)
    assert restored["quiz_sessions"] == []
    assert restored["resource_agent_state"] == {}
    assert restored["spaced_repetition"] == {}


def test_sqlite_operations_close_connections_and_commit_delete(monkeypatch, tmp_path):
    store = SQLiteProfileStore(str(tmp_path))
    with store.write_transaction() as connection:
        store.upsert_in_transaction(connection, "student", "Student", '{"courses": {}}')

    connections = []
    connect = store._connect

    def track_connection():
        connection = connect()
        connections.append(connection)
        return connection

    monkeypatch.setattr(store, "_connect", track_connection)
    assert store.read("student") == {"courses": {}}
    assert len(store.list_profiles()) == 1
    assert store.delete("student")
    assert store.read("student") is None
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
