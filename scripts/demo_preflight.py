"""
Demo preflight checklist.

Verifies that the local environment is ready for a live demo of the
AI Study Tutor app. Each check is an independent function returning a
``CheckResult`` so the same checks can be exercised by unit tests
without running the full CLI.

Usage
-----
    python scripts/demo_preflight.py

Exit codes
----------
    0  all checks passed
    1  one or more checks failed

The script never prints the API key. When a check involves the key,
it only reports presence / placeholder status.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
DEMO_PROFILE_PATH = REPO_ROOT / "data" / "demo_student.json"
GENERATE_DEMO_SCRIPT = REPO_ROOT / "tests" / "generate_demo.py"

# Marker that the .env / .env.example template uses for an unset key.
PLACEHOLDER_KEY = "your_anthropic_api_key_here"
LEGACY_PLACEHOLDER_KEY = "sk-" + "ant-api03-REPLACE_WITH_YOUR_KEY"
PLACEHOLDER_KEYS = {
    PLACEHOLDER_KEY,
    LEGACY_PLACEHOLDER_KEY,
}

# Required Python interpreter floor.
MIN_PYTHON = (3, 10)

# Modules that must import for the app + agents to run.
REQUIRED_MODULES = ("streamlit", "anthropic", "dotenv", "filelock")
OFFLINE_REQUIRED_MODULES = ("streamlit", "dotenv", "filelock")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""

    def render(self) -> str:
        icon = "✅" if self.ok else "❌"
        line = f"{icon} {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        if not self.ok and self.fix:
            line += f"\n     fix: {self.fix}"
        return line


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_python_version() -> CheckResult:
    have = sys.version_info[:3]
    if have < MIN_PYTHON:
        return CheckResult(
            name="Python version",
            ok=False,
            detail=f"found {have[0]}.{have[1]}.{have[2]}, need ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            fix=f"install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer and rebuild the venv",
        )
    return CheckResult(
        name="Python version",
        ok=True,
        detail=f"{have[0]}.{have[1]}.{have[2]}",
    )


def check_imports(modules: tuple[str, ...] = REQUIRED_MODULES) -> CheckResult:
    missing: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return CheckResult(
            name="Required imports",
            ok=False,
            detail=f"missing: {', '.join(missing)}",
            fix="pip install -r requirements.txt",
        )
    return CheckResult(
        name="Required imports",
        ok=True,
        detail=f"{', '.join(modules)} all importable",
    )


def _offline_demo_enabled(env_path: Path = ENV_PATH) -> bool:
    raw = os.environ.get("STUDY_TUTOR_OFFLINE_DEMO", "").strip()
    if raw:
        return raw.lower() in {"1", "true", "yes", "on"}
    try:
        from dotenv import dotenv_values  # type: ignore[import-not-found]
        if env_path.exists():
            raw = str(dotenv_values(env_path).get("STUDY_TUTOR_OFFLINE_DEMO", "") or "").strip()
    except ImportError:
        pass
    return raw.lower() in {"1", "true", "yes", "on"}


def check_env_file_exists(path: Path = ENV_PATH) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name=".env file",
            ok=False,
            detail="not found at repo root",
            fix=f"cp {ENV_EXAMPLE_PATH.name} {path.name} and fill in your real key",
        )
    return CheckResult(name=".env file", ok=True, detail=f"present at {path.name}")


def check_api_key(env_path: Path = ENV_PATH) -> CheckResult:
    """Check that ANTHROPIC_API_KEY is set and not the placeholder.

    Reads from os.environ; calls ``load_dotenv()`` first if available so
    that a fresh shell that has not yet sourced ``.env`` still gets
    accurate results.

    NEVER prints the key value itself.
    """
    if _offline_demo_enabled(env_path):
        return CheckResult(
            name="ANTHROPIC_API_KEY",
            ok=True,
            detail="skipped because STUDY_TUTOR_OFFLINE_DEMO=1",
        )

    # Prefer a real shell-provided key over .env. This matches app
    # startup (`load_dotenv(..., override=False)`) and avoids a common
    # demo trap: .env still contains the template placeholder, but the
    # presenter correctly exported a real key in the terminal.
    raw = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if raw and raw not in PLACEHOLDER_KEYS and raw.startswith("sk-") and len(raw) > 30:
        return CheckResult(
            name="ANTHROPIC_API_KEY",
            ok=True,
            detail="set in environment (value not printed)",
        )

    try:
        from dotenv import dotenv_values  # type: ignore[import-not-found]
        if env_path.exists():
            env_raw = str(dotenv_values(env_path).get("ANTHROPIC_API_KEY", "") or "").strip()
            if env_raw and (not raw or raw in PLACEHOLDER_KEYS):
                raw = env_raw
    except ImportError:
        # python-dotenv missing is already reported by check_imports.
        pass

    if not raw:
        return CheckResult(
            name="ANTHROPIC_API_KEY",
            ok=False,
            detail="not set in environment",
            fix=f"edit {env_path.name} and set ANTHROPIC_API_KEY=<your real key>",
        )
    if raw in PLACEHOLDER_KEYS:
        return CheckResult(
            name="ANTHROPIC_API_KEY",
            ok=False,
            detail="still set to the .env.example placeholder",
            fix=f"edit {env_path.name} and replace the placeholder with your real key",
        )
    # Light shape check; do not expose the value.
    looks_like_key = raw.startswith("sk-") and len(raw) > 30
    if not looks_like_key:
        return CheckResult(
            name="ANTHROPIC_API_KEY",
            ok=False,
            detail="set, but does not look like an Anthropic key",
            fix="confirm the key starts with 'sk-' and is the full string copied from console.anthropic.com",
        )
    return CheckResult(
        name="ANTHROPIC_API_KEY",
        ok=True,
        detail="set (value not printed)",
    )


def _display_path(path: Path) -> str:
    """Render ``path`` relative to the repo root when possible, else absolute.

    Tests may pass paths inside ``tmp_path``; ``Path.relative_to`` raises
    ``ValueError`` for those, so fall back to a plain string.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check_demo_profile(profile_path: Path = DEMO_PROFILE_PATH) -> CheckResult:
    if not profile_path.exists():
        return CheckResult(
            name="Demo profile",
            ok=False,
            detail=f"missing {_display_path(profile_path)}",
            fix="python tests/generate_demo.py",
        )
    return CheckResult(
        name="Demo profile",
        ok=True,
        detail=f"present at {_display_path(profile_path)}",
    )


def regenerate_demo_profile(
    script_path: Path = GENERATE_DEMO_SCRIPT,
    python_executable: str | None = None,
) -> CheckResult:
    """Reuse tests/generate_demo.py to recreate the sample profile.

    Returns a check result describing whether regeneration succeeded.
    """
    if not script_path.exists():
        return CheckResult(
            name="Regenerate demo profile",
            ok=False,
            detail=f"{script_path} not found",
            fix="restore tests/generate_demo.py from version control",
        )
    py = python_executable or sys.executable
    try:
        result = subprocess.run(
            [py, str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Regenerate demo profile",
            ok=False,
            detail="generator timed out after 60s",
            fix="run python tests/generate_demo.py manually and check the error",
        )
    except OSError as exc:
        return CheckResult(
            name="Regenerate demo profile",
            ok=False,
            detail=f"could not launch generator: {exc}",
            fix="check that python is on PATH and the venv is activated",
        )
    if result.returncode != 0:
        # stderr can leak detail but not the key (the generator does not see the key).
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        return CheckResult(
            name="Regenerate demo profile",
            ok=False,
            detail="generator exited non-zero: " + " | ".join(tail),
            fix="run python tests/generate_demo.py manually for the full error",
        )
    return CheckResult(
        name="Regenerate demo profile",
        ok=True,
        detail="created data/demo_student.json",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_all_checks(auto_regenerate: bool = True) -> list[CheckResult]:
    """Run every preflight check in order. Returns the result list."""
    results: list[CheckResult] = []
    results.append(check_python_version())
    offline = _offline_demo_enabled(ENV_PATH)
    results.append(check_imports(OFFLINE_REQUIRED_MODULES if offline else REQUIRED_MODULES))
    results.append(check_env_file_exists(ENV_PATH))
    results.append(check_api_key(ENV_PATH))
    profile = check_demo_profile(DEMO_PROFILE_PATH)
    results.append(profile)
    if not profile.ok and auto_regenerate:
        regen = regenerate_demo_profile(GENERATE_DEMO_SCRIPT)
        results.append(regen)
        if regen.ok:
            # Re-check after regeneration so the final report reflects reality.
            results.append(check_demo_profile(DEMO_PROFILE_PATH))
    return results


def render_report(results: list[CheckResult]) -> str:
    lines = ["", "─── Demo Preflight ───"]
    for r in results:
        lines.append(r.render())
    lines.append("")
    failed = [r for r in results if not r.ok]
    if failed:
        lines.append(f"❌ {len(failed)} check(s) failed.")
    else:
        lines.append("✅ All checks passed — ready to launch.")
    lines.append("")
    return "\n".join(lines)


def _force_utf8_stdout() -> None:
    """Reconfigure stdout to UTF-8 so the emoji checklist renders on
    Windows consoles whose default encoding is cp1252. Best-effort —
    silently no-op on platforms where ``reconfigure`` is unavailable.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def main() -> int:
    _force_utf8_stdout()
    results = run_all_checks()
    print(render_report(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
