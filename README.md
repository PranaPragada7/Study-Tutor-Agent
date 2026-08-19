# Study Tutor Agent

[![CI](https://github.com/PranaPragada7/Study-Tutor-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/PranaPragada7/Study-Tutor-Agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-17223b?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-e4573d?logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-powered-f2b84b)

Study Tutor is a local adaptive-learning workspace built with Streamlit and Claude. It combines personalized tutoring, five-question practice sessions, spaced repetition, progress tracking, and a collaborating Resource Agent in one application.

The app also includes a fully offline fallback, so the interface and core learning flow remain demonstrable without an API key or network connection.

## What it does

- Keeps separate local profiles for students, courses, learning preferences, and quiz history.
- Selects topics and difficulty with an epsilon-greedy adaptive engine.
- Runs focused five-question sessions with confidence-aware feedback.
- Schedules review using the SM-2 spaced-repetition algorithm.
- Generates profile-aware explanations, quizzes, resources, and study plans with Claude.
- Coordinates a Tutor Agent and Resource Agent through an internal message bus.
- Surfaces mastery, weak topics, review dates, session reports, and diagnostic telemetry.
- Protects local state with atomic writes, file locks, schema migration, and bounded history.

## How it works

```mermaid
flowchart LR
    UI[Streamlit workspace] --> TA[Tutor Agent]
    TA --> AE[Adaptive engine]
    TA --> SR[Spaced repetition]
    TA <--> MB[Message bus]
    MB <--> RA[Resource Agent]
    TA --> LLM[Claude API]
    RA --> LLM
    TA <--> P[(Local JSON profiles)]
```

The Tutor Agent owns the student-facing flow. It reads the profile, chooses an appropriate practice target, records the result, and updates the review schedule. When a student needs more support, it asks the Resource Agent for targeted material through the message bus. In offline mode, deterministic local fallbacks replace the external model calls.

## Run locally

Requirements: Python 3.10 or newer.

```powershell
git clone https://github.com/PranaPragada7/Study-Tutor-Agent.git
cd Study-Tutor-Agent

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### Offline demo

No secret file is required.

```powershell
$env:STUDY_TUTOR_OFFLINE_MODE="1"
python -m streamlit run app.py
```

### Live Claude mode

Create a local environment file and add an Anthropic API key:

```powershell
Copy-Item .env.example .env
notepad .env
python -m streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501). Use **Load Sample Student** in the sidebar for a prepared walkthrough.

## Validate the project

```powershell
$env:STUDY_TUTOR_OFFLINE_MODE="1"
python scripts/preflight_check.py
python -m pytest -q
```

Additional checks are available for the feedback loop, live flow, and a real Claude warm-up call:

```powershell
python scripts/feedback_check.py
python scripts/live_flow_check.py
python scripts/warmup_check.py
```

The pytest suite does not make live Anthropic requests. GitHub Actions runs the offline preflight and complete test suite for every pull request.

## Project structure

```text
app.py                 Streamlit interface and session workflow
config.py              Environment-driven runtime settings
agents/
  tutor_agent.py       Tutoring, quizzes, feedback, and study plans
  resource_agent.py    Learning-material generation and caching
  adaptive_engine.py   Topic and difficulty selection
utils/
  student_profile.py   Local profile persistence and migration
  spaced_repetition.py SM-2 review scheduling
  quiz_session.py      Five-question session reports
  agent_comm.py        Inter-agent message bus
  telemetry.py         Runtime counters and diagnostics
scripts/               Preflight and presentation checks
tests/                 Unit, integration, hardening, and UI smoke tests
data/                  Sample profile; personal profiles stay untracked
```

## Data and security

Student profiles are stored locally as JSON files in `data/`. Personal profiles, `.env`, Streamlit secrets, virtual environments, caches, and lock files are excluded from Git.

Only placeholder values belong in `.env.example` and `.streamlit/secrets.toml.example`. If a real API key is ever shared or committed, revoke it in the Anthropic Console and create a new one.
