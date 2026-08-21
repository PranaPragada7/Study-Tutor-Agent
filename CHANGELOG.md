# Changelog

## Unreleased — Local platform and UI upgrade

- Redesigned the Streamlit experience as a responsive learning dashboard with
  clear onboarding, recommendations, runtime visibility, and privacy status.
- Added a complete no-key local demo runtime with deterministic tutor fallbacks.
- Added transactional SQLite profile storage and automatic legacy JSON import.
- Added a versioned FastAPI service with Pydantic validation and OpenAPI docs.
- Added a non-root Docker image and loopback-only Compose stack for the UI/API.
- Expanded the suite to 241 tests covering runtime selection, SQLite, API contracts,
  the application shell, adaptive learning, persistence, and multi-agent behavior.

## v1.0.0 — Presentation-ready release

- Added a live Claude tutoring experience with durable conversation memory.
- Persisted quiz answers, answer keys, explanations, confidence, tutor feedback, and five-question session reports.
- Added adaptive topic and difficulty selection, SM-2 spaced review, and explicit strengths and weaknesses.
- Added Tutor Agent and Resource Agent collaboration with inspectable diagnostics.
- Added atomic local profile persistence, schema migration, privacy controls, and bounded history.
- Added a distinct Streamlit visual system and a prepared sample-student walkthrough.
- Added Python 3.10–3.13 CI, Ruff linting and formatting, branch coverage enforcement, and dependency updates.
- Added recruiter-facing screenshots, architecture documentation, and clear deployment boundaries.
