"""Resolve the local AI and storage runtime without making network calls."""

from __future__ import annotations

import os
from dataclasses import dataclass

_PLACEHOLDER_KEYS = {
    "your_anthropic_api_key_here",
    "sk-ant-api03-REPLACE_WITH_YOUR_KEY",
}


@dataclass(frozen=True)
class RuntimeConfig:
    """User-facing runtime state for the UI, API, and diagnostics."""

    mode: str
    label: str
    description: str
    llm_enabled: bool
    storage_backend: str

    @property
    def storage_label(self) -> str:
        return "SQLite" if self.storage_backend == "sqlite" else "JSON files"


def _has_usable_api_key() -> bool:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return bool(
        api_key
        and api_key not in _PLACEHOLDER_KEYS
        and api_key.startswith("sk-")
        and len(api_key) > 30
    )


def get_runtime_config() -> RuntimeConfig:
    """Return a deterministic runtime configuration.

    ``auto`` uses Claude only when a plausible key is configured. ``demo``
    always uses the built-in deterministic tutor content. ``live`` requests
    Claude but degrades to demo mode when the key is unavailable so the local
    application never becomes a blank setup screen.
    """

    requested_mode = os.getenv("STUDY_TUTOR_MODE", "demo").strip().lower()
    if requested_mode not in {"auto", "demo", "live"}:
        requested_mode = "auto"

    storage = os.getenv("STUDY_TUTOR_STORAGE", "sqlite").strip().lower()
    if storage not in {"sqlite", "json"}:
        storage = "sqlite"

    has_key = _has_usable_api_key()
    use_live = requested_mode != "demo" and has_key
    if use_live:
        return RuntimeConfig(
            mode="live",
            label="Claude live",
            description="Claude responses are enabled with your local API key.",
            llm_enabled=True,
            storage_backend=storage,
        )

    description = "Built-in tutor content is active; no API key or paid service is required."
    if requested_mode == "live":
        description = (
            "Live mode was requested, but no usable API key was found. "
            "The app safely switched to its local tutor."
        )
    return RuntimeConfig(
        mode="demo",
        label="Local demo",
        description=description,
        llm_enabled=False,
        storage_backend=storage,
    )


__all__ = ["RuntimeConfig", "get_runtime_config"]
