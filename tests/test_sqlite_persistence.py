"""Transactional SQLite persistence tests."""

from utils.student_profile import (
    create_profile,
    delete_profile,
    list_saved_profiles,
    load_profile,
    record_quiz_result,
    save_profile,
)


def _use_sqlite(monkeypatch, tmp_path) -> None:
    import utils.student_profile as profile_module

    monkeypatch.setenv("STUDY_TUTOR_STORAGE", "sqlite")
    monkeypatch.setattr(profile_module, "DATA_DIR", str(tmp_path))


def test_sqlite_profile_lifecycle(monkeypatch, tmp_path):
    _use_sqlite(monkeypatch, tmp_path)
    profile = create_profile(
        "Database Student",
        [{"name": "Distributed Systems", "difficulty": 4}],
    )
    record_quiz_result(
        profile,
        "Distributed Systems",
        "Replication",
        4,
        True,
        "What improves availability?",
        "Replication",
        confidence=4,
    )
    assert save_profile(profile["name"], profile)

    restored = load_profile("Database Student")
    assert restored is not None
    assert restored["total_quizzes"] == 1
    assert restored["courses"]["Distributed Systems"]["total_correct"] == 1
    assert (tmp_path / "study_tutor.db").exists()

    listed = list_saved_profiles()
    assert listed == [
        {
            "name": "Database Student",
            "total_quizzes": 1,
            "courses": 1,
            "updated_at": listed[0]["updated_at"],
        }
    ]

    assert delete_profile("Database Student")
    assert load_profile("Database Student") is None


def test_sqlite_imports_legacy_json_once(monkeypatch, tmp_path):
    import utils.student_profile as profile_module

    monkeypatch.setattr(profile_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STUDY_TUTOR_STORAGE", "json")
    create_profile("Legacy Student", [{"name": "Algorithms", "difficulty": 3}])

    monkeypatch.setenv("STUDY_TUTOR_STORAGE", "sqlite")
    restored = load_profile("Legacy Student")

    assert restored is not None
    assert restored["name"] == "Legacy Student"
    assert (tmp_path / "study_tutor.db").exists()
