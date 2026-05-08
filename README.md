# AI Study Tutor Agent

AI Study Tutor is a local Streamlit app for adaptive study practice. It stores a student profile on disk, chooses quiz topics and difficulty with local learning algorithms, uses spaced repetition for review timing, and can call Anthropic Claude for chat, quiz generation, explanations, resource material, and study plans.

The app can also run without Claude in offline mode. In that mode it uses local fallback quiz questions, feedback, and a study plan so the classroom review still works if the API key, Wi-Fi, or provider is unavailable.

## Current Features

- One-click **Load Sample Student** button in the sidebar.
- Five-tab UI: Chat, Quiz, Study Plan, Progress, Diagnostics.
- Chat is profile-aware: it receives saved course performance, latest five-question evaluation, recent quiz answers, confidence patterns, due reviews, and ResourceAgent material summaries.
- Five-question quiz sessions before the overall session evaluation appears.
- Per-question feedback with confidence-aware signals.
- **Why this question?** expander showing the adaptive reason, policy, mastery, and recommended difficulty.
- Five-question session report with accuracy, priority topics, high-confidence misses, and question-by-question rows; completed reports are saved, and older profiles can reconstruct a latest report from saved answer history.
- Resource Agent suggestions for missed/priority topics when cached materials exist.
- Local adaptive engine, SM-2 spaced repetition, issue detection, telemetry counters, and JSON profile persistence.
- Offline mode through `STUDY_TUTOR_OFFLINE_MODE=1`.

## Runtime Notes

- Python 3.10+ is required by the code. The Windows setup below uses Python 3.11 because it was verified in this workspace.
- The current Claude model ID is in `agents/_llm.py`: `claude-sonnet-4-20250514`.
- Profiles are stored as JSON under `data/`.
- `.env` may contain a real API key locally, but it must never be committed or included in a submitted zip.

## Windows PowerShell Setup

From PowerShell:

```powershell
cd "c:\Users\prana\Downloads\study-tutor-agent\study-tutor-agent"

py -3.11 -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -r requirements-dev.txt
```

If `python` is not recognized or points to the wrong version:

```powershell
py --version
py -3.11 --version
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

In this workspace, unactivated `python` is Python 3.9.13, which is too old for this project. Use the activated venv or `.\venv\Scripts\python.exe`.

## Professor / Fresh GitHub Clone Setup

From a fresh clone, run:

```powershell
git clone <repo-url>
cd study-tutor-agent

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

To test the frontend and backend with live Claude calls, create `.env` from
the template and add a real Anthropic API key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Then run:

```powershell
python -m streamlit run app.py
```

Open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

If no API key is available, the app still runs in fallback mode. For a fully
offline check:

```powershell
$env:STUDY_TUTOR_OFFLINE_MODE="1"
python -m streamlit run app.py
```

## Environment Setup

Copy the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and use only this placeholder format until you add your real key:

```text
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

`ANTHROPIC_API_KEY` is used for live Claude calls. The app can start without it and will fall back locally, but live chat, live quiz generation, richer explanations, Resource Agent generation, and live study-plan generation need a valid key.

Never commit `.env`, `.env.txt`, `.streamlit/secrets.toml`, or real keys. `.gitignore` already ignores those files.

## Run Commands

Normal app, after activating the venv:

```powershell
python -m streamlit run app.py
```

Offline mode:

```powershell
$env:STUDY_TUTOR_OFFLINE_MODE="1"
python -m streamlit run app.py
```

Using venv Python directly:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Offline mode with venv Python directly:

```powershell
$env:STUDY_TUTOR_OFFLINE_MODE="1"
.\venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Test Commands

After activating the venv:

```powershell
python -m pytest tests/ -q
python scripts/preflight_check.py
python scripts/live_flow_check.py
python scripts/feedback_check.py
python scripts/warmup_check.py
```

Using venv Python directly:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q
.\venv\Scripts\python.exe scripts\preflight_check.py
.\venv\Scripts\python.exe scripts\live_flow_check.py
.\venv\Scripts\python.exe scripts\feedback_check.py
.\venv\Scripts\python.exe scripts\warmup_check.py
```

Verified in this workspace on 2026-05-07 with Python 3.11.9:

- `.\venv\Scripts\python.exe -m pytest tests/ -q`: `222 passed in 3.21s`
- `.\venv\Scripts\python.exe scripts\preflight_check.py`: all checks passed
- `.\venv\Scripts\python.exe scripts\live_flow_check.py`: live flow ready
- `.\venv\Scripts\python.exe scripts\feedback_check.py`: feedback check passed
- `.\venv\Scripts\python.exe scripts\warmup_check.py`: real LLM question generated

The pytest suite is configured to avoid live Anthropic calls even if a real `.env` key exists. The warm-up script is the check that intentionally verifies one real Claude quiz-generation call.

## Suggested Review Flow

1. Start the app.
2. Click **Load Sample Student** in the sidebar.
3. Show profile/progress in the Progress tab.
4. Open the Quiz tab and start a five-question quiz.
5. Show the **Why this question?** explanation.
6. Answer five questions. For one question, answer wrong with confidence 5 to show the misconception signal.
7. Show the five-question session report, then show that Progress/Diagnostics can still display the latest report after loading a saved profile.
8. Open Study Plan and generate the plan.
9. Open Diagnostics and show quiz session reports, agent messages, telemetry, Resource Agent cache, RL state, and spaced repetition state.
10. Explain that offline fallback keeps the app working if Claude is unavailable.

## Project Layout

```text
app.py                      Streamlit app
config.py                   Environment-driven constants
requirements.txt            Runtime dependencies
requirements-dev.txt        Runtime dependencies plus pytest
.env.example                Safe local environment template
.streamlit/config.toml      Streamlit theme and browser config
.streamlit/secrets.toml.example
agents/                     Tutor, Resource, adaptive engine, LLM helpers
utils/                      Profile storage, quiz sessions, telemetry, scheduling
scripts/                    Preflight, live-flow, feedback, warm-up, and offline launch checks
tests/                      Unit, integration, hardening, and Streamlit smoke tests
docs/                       Presentation and live walkthrough notes
data/sample_student.json      Sample profile used by the sidebar loader
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Run `py --version`, then use `py -3.11 -m venv venv`. |
| `py` is not recognized | Install Python 3.11 from python.org and check "Add python.exe to PATH", then reopen PowerShell. |
| Venv activation is blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen PowerShell, then run `.\venv\Scripts\Activate.ps1`. |
| `streamlit` is not recognized | Use `python -m streamlit run app.py` or `.\venv\Scripts\python.exe -m streamlit run app.py`. |
| `ModuleNotFoundError` | Install dependencies with `python -m pip install -r requirements-dev.txt` inside the venv. |
| Missing API key warning | Add `ANTHROPIC_API_KEY=your_real_key` to `.env`, or run offline mode. |
| Offline mode not active | In the same PowerShell window, run `$env:STUDY_TUTOR_OFFLINE_MODE="1"` before starting Streamlit. |
| Port already in use | Run `python -m streamlit run app.py --server.port 8502`. |
| Tests fail with Python 3.9 type errors | Use `py -3.11` or `.\venv\Scripts\python.exe`; Python 3.9 is too old. |
| Sample profile does not load | Run `.\venv\Scripts\python.exe scripts\preflight_check.py`; it checks `data/sample_student.json` and can regenerate it if missing. |
| Live Claude is slow/unavailable | Use offline mode or `powershell -ExecutionPolicy Bypass -File scripts\run_offline_app.ps1`. |

## Security

- `.env.example` and `.streamlit/secrets.toml.example` must contain placeholders only.
- `.env`, `.env.txt`, and `.streamlit/secrets.toml` must stay ignored.
- If a real key was ever shared, committed, emailed, or zipped, rotate/revoke it in the Anthropic console.
