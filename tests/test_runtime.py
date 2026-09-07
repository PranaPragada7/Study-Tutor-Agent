"""Runtime selection tests for free local and optional live modes."""

from services.runtime import get_runtime_config


def test_auto_mode_defaults_to_local_demo(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("STUDY_TUTOR_MODE", "auto")
    monkeypatch.setenv("STUDY_TUTOR_STORAGE", "sqlite")

    runtime = get_runtime_config()

    assert runtime.mode == "demo"
    assert runtime.llm_enabled is False
    assert runtime.storage_label == "SQLite"


def test_default_mode_stays_free_even_when_a_key_exists(monkeypatch):
    monkeypatch.delenv("STUDY_TUTOR_MODE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 80)

    runtime = get_runtime_config()

    assert runtime.mode == "demo"
    assert runtime.llm_enabled is False


def test_live_mode_uses_plausible_local_key_without_exposing_it(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 80)
    monkeypatch.setenv("STUDY_TUTOR_MODE", "live")

    runtime = get_runtime_config()

    assert runtime.mode == "live"
    assert runtime.llm_enabled is True
    assert "x" * 20 not in runtime.description


def test_requested_live_mode_degrades_safely(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
    monkeypatch.setenv("STUDY_TUTOR_MODE", "live")

    runtime = get_runtime_config()

    assert runtime.mode == "demo"
    assert "safely switched" in runtime.description


def test_invalid_runtime_settings_fall_back_to_supported_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("STUDY_TUTOR_MODE", "unsupported")
    monkeypatch.setenv("STUDY_TUTOR_STORAGE", "unsupported")
    runtime = get_runtime_config()
    assert runtime.mode == "demo"
    assert runtime.storage_backend == "sqlite"
