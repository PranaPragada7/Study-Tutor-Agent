"""Pytest fixtures shared across the local test suite."""

from __future__ import annotations

import tempfile
import uuid
import os
from pathlib import Path

import pytest

# Keep the unit test suite hermetic even when a developer has a real key
# in .env. Individual tests can still monkeypatch a fake key when needed.
os.environ["ANTHROPIC_API_KEY"] = ""

_WORKSPACE_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "pytest-runtime"
_WORKSPACE_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _workspace_mkdtemp(suffix=None, prefix=None, dir=None):
    suffix = suffix or ""
    prefix = prefix or "tmp"
    base = Path(dir) if dir else _WORKSPACE_TEMP_ROOT
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        candidate = base / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            candidate.mkdir()
            return str(candidate)
        except FileExistsError:
            continue
    raise FileExistsError("Could not create a unique workspace temp directory")


# Apply at import time so xunit-style setup_method hooks also use it.
tempfile.tempdir = str(_WORKSPACE_TEMP_ROOT)
tempfile.mkdtemp = _workspace_mkdtemp


@pytest.fixture
def tmp_path():
    """Provide a workspace-local temp path without pytest's numbered temp root."""
    return Path(_workspace_mkdtemp(prefix="tmp_path_"))


@pytest.fixture(autouse=True)
def _workspace_tempdir_env(monkeypatch):
    """Keep test temp files inside the repo's ignored .tmp folder.

    Some Windows sandbox environments deny access to AppData temp folders
    after a Streamlit/AppTest run. The project already ignores .tmp/, so tests
    can safely create transient profile data there without touching real
    student files or relying on OS-global temp cleanup.
    """
    monkeypatch.setenv("TMP", str(_WORKSPACE_TEMP_ROOT))
    monkeypatch.setenv("TEMP", str(_WORKSPACE_TEMP_ROOT))
