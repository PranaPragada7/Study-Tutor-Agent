"""
Tests for scripts/preflight_check.py.

These tests must NOT make real network calls and must NOT make real
Anthropic API calls. The preflight checks themselves do not call
Anthropic — they only inspect the local environment — so this is
naturally satisfied; the tests below additionally use monkeypatch to
isolate the filesystem and environment so each check can be exercised
independently.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preflight_check as pf  # noqa: E402


# ---------------------------------------------------------------------------
# Python-version check
# ---------------------------------------------------------------------------
def test_python_version_passes_on_supported_interpreter():
    # We assume the test runner is on Python ≥ 3.10 (the project floor).
    result = pf.check_python_version()
    assert result.ok, result.detail
    assert "Python" in result.name


# ---------------------------------------------------------------------------
# Imports check
# ---------------------------------------------------------------------------
def test_check_imports_passes_when_all_present(monkeypatch):
    # Use a tuple of modules guaranteed to be in the standard library
    # so the test does not depend on the dev environment having streamlit.
    result = pf.check_imports(("os", "sys", "json"))
    assert result.ok
    assert "importable" in result.detail


def test_check_imports_reports_missing(monkeypatch):
    result = pf.check_imports(("os", "sys", "definitely_not_a_real_module_xyz"))
    assert not result.ok
    assert "definitely_not_a_real_module_xyz" in result.detail
    assert "pip install" in result.fix


# ---------------------------------------------------------------------------
# .env existence
# ---------------------------------------------------------------------------
def test_env_file_missing_reports_failure(tmp_path):
    missing = tmp_path / ".env"
    result = pf.check_env_file_exists(missing)
    assert not result.ok
    assert "not found" in result.detail
    assert ".env.example" in result.fix


def test_env_file_present_reports_pass(tmp_path):
    p = tmp_path / ".env"
    p.write_text("ANTHROPIC_API_KEY=does-not-matter\n", encoding="utf-8")
    result = pf.check_env_file_exists(p)
    assert result.ok


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
def test_api_key_missing(tmp_path, monkeypatch):
    # Ensure no key in env, no .env file.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = pf.check_api_key(env_path=tmp_path / "nope.env")
    assert not result.ok
    assert "not set" in result.detail.lower()


def test_api_key_placeholder_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", pf.PLACEHOLDER_KEY)
    # Use a non-existent .env path so load_dotenv is a no-op.
    result = pf.check_api_key(env_path=tmp_path / "nope.env")
    assert not result.ok
    assert "placeholder" in result.detail.lower()
    # The placeholder text must NOT appear in fix without context — fix
    # should refer to editing the file, not echo the key.
    assert pf.PLACEHOLDER_KEY not in result.fix


def test_api_key_short_string_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-too-short")
    result = pf.check_api_key(env_path=tmp_path / "nope.env")
    assert not result.ok
    assert "anthropic key" in result.detail.lower()


def test_api_key_well_formed_passes(tmp_path, monkeypatch):
    # 40+ chars and starts with sk-: shape check passes.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 80)
    result = pf.check_api_key(env_path=tmp_path / "nope.env")
    assert result.ok
    # Key value must not appear in either field.
    assert "x" * 80 not in result.detail
    assert "x" * 80 not in (result.fix or "")


def test_api_key_loaded_from_dotenv_when_env_unset(tmp_path, monkeypatch):
    """If the env var isn't set but .env contains a real key, the check
    should pick it up via load_dotenv."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_key = "sk-ant-api03-" + "z" * 80
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={fake_key}\n", encoding="utf-8")
    result = pf.check_api_key(env_path=env_file)
    assert result.ok
    assert "z" * 80 not in result.detail


def test_real_shell_key_wins_over_placeholder_dotenv(tmp_path, monkeypatch):
    """A valid exported key should not be overwritten by a stale template .env."""
    fake_key = "sk-ant-api03-" + "x" * 80
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key)
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"ANTHROPIC_API_KEY={pf.PLACEHOLDER_KEY}\n",
        encoding="utf-8",
    )
    result = pf.check_api_key(env_path=env_file)
    assert result.ok
    assert "environment" in result.detail
    assert "x" * 80 not in result.detail


# ---------------------------------------------------------------------------
# Sample profile check
# ---------------------------------------------------------------------------
def test_sample_profile_missing(tmp_path):
    result = pf.check_sample_profile(tmp_path / "sample_student.json")
    assert not result.ok
    assert "missing" in result.detail
    assert "generate_sample_profile.py" in result.fix


def test_sample_profile_present(tmp_path):
    p = tmp_path / "sample_student.json"
    p.write_text(json.dumps({"name": "Sample Student"}), encoding="utf-8")
    result = pf.check_sample_profile(p)
    assert result.ok


# ---------------------------------------------------------------------------
# Regenerate via subprocess — guard against accidentally launching
# something. We use a no-op script the test writes itself.
# ---------------------------------------------------------------------------
def test_regenerate_sample_profile_handles_missing_script(tmp_path):
    # Pass a path that does not exist; expect a clean error result, not
    # a raised exception.
    result = pf.regenerate_sample_profile(tmp_path / "no_such_script.py")
    assert not result.ok
    assert "not found" in result.detail


def test_regenerate_sample_profile_runs_a_dummy_generator(tmp_path):
    # Write a tiny no-op generator the preflight can launch. This proves
    # the subprocess wiring works without invoking the real generator.
    fake_gen = tmp_path / "fake_generator.py"
    fake_gen.write_text("print('ok')\n", encoding="utf-8")
    result = pf.regenerate_sample_profile(fake_gen, python_executable=sys.executable)
    assert result.ok
    assert "created" in result.detail.lower()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def test_run_all_checks_does_not_call_network(monkeypatch, tmp_path):
    """run_all_checks() must complete without any real network calls.

    We monkeypatch socket.getaddrinfo / socket.create_connection to
    prove the preflight doesn't try to reach the internet — if it did,
    we'd raise.
    """
    import socket

    def _no_net(*args, **kwargs):
        raise AssertionError(f"unexpected network call: args={args}")

    monkeypatch.setattr(socket, "getaddrinfo", _no_net)
    monkeypatch.setattr(socket, "create_connection", _no_net)

    # Point checks at temp paths so we don't read the real repo state.
    monkeypatch.setattr(pf, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(pf, "SAMPLE_PROFILE_PATH", tmp_path / "sample_student.json")
    # Use a fake generator that should not actually be invoked since we
    # turn auto_regenerate off.
    monkeypatch.setattr(pf, "GENERATE_SAMPLE_SCRIPT", tmp_path / "no_gen.py")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    results = pf.run_all_checks(auto_regenerate=False)
    # Should produce: python, imports, env, key, sample profile = 5 entries.
    assert len(results) == 5
    names = [r.name for r in results]
    assert "Python version" in names
    assert "Required imports" in names
    assert ".env file" in names
    assert "ANTHROPIC_API_KEY" in names
    assert "Sample profile" in names


def test_shell_api_key_does_not_require_env_file(monkeypatch, tmp_path):
    """A securely exported key should make a separate .env file optional."""
    sample = tmp_path / "sample_student.json"
    sample.write_text(json.dumps({"name": "Sample Student"}), encoding="utf-8")

    monkeypatch.setattr(pf, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(pf, "SAMPLE_PROFILE_PATH", sample)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "k" * 80)

    results = pf.run_all_checks(auto_regenerate=False)
    assert all(result.ok for result in results), pf.render_report(results)
    env_result = next(result for result in results if result.name == ".env file")
    assert "environment" in env_result.detail


def test_render_report_marks_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(pf, "SAMPLE_PROFILE_PATH", tmp_path / "sample_student.json")
    monkeypatch.setattr(pf, "GENERATE_SAMPLE_SCRIPT", tmp_path / "no_gen.py")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    results = pf.run_all_checks(auto_regenerate=False)
    text = pf.render_report(results)
    assert "Preflight Check" in text
    assert "❌" in text  # at least one failure (env, key, profile all missing)


def test_no_real_anthropic_call_during_preflight(monkeypatch, tmp_path):
    """The preflight script must not import the anthropic SDK in a way
    that triggers a network call. We patch any plausible client method
    to raise on use; if preflight calls it, the test fails.
    """
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("anthropic SDK not installed in this environment")

    # If anything tries to instantiate Anthropic() during preflight, blow up.
    def _no_client(*args, **kwargs):
        raise AssertionError("preflight attempted to construct an Anthropic client")

    monkeypatch.setattr(anthropic, "Anthropic", _no_client)

    # Reload preflight_check to pick up the patched module — but in fact
    # preflight_check does NOT import anthropic at all; this assertion
    # is the real check.
    importlib.reload(pf)
    monkeypatch.setattr(pf, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(pf, "SAMPLE_PROFILE_PATH", tmp_path / "sample_student.json")
    monkeypatch.setattr(pf, "GENERATE_SAMPLE_SCRIPT", tmp_path / "no_gen.py")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    pf.run_all_checks(auto_regenerate=False)
    # Reaching this line means no Anthropic() was constructed.
