"""
Shared LLM configuration + client factory for the agents package.

Why this module exists
----------------------
- **Single source of truth for the model ID.** Previously the literal
  ``"claude-sonnet-4-20250514"`` was repeated at eight call sites across
  ``tutor_agent.py`` and ``resource_agent.py``. A partial edit during a
  model upgrade would leave the codebase in a split state. Now every
  agent reads ``MODEL`` from this module -- one line to upgrade.

- **Single place to try-import the anthropic SDK.** Both agents used to
  carry duplicated ``try/except ImportError`` blocks with stub exception
  classes so that ``except`` clauses still parsed when the SDK was not
  installed (CI without the dep, or local dev offline). Those are
  consolidated here and re-exported.

- **Default client factory for non-Streamlit callers.** ``app.py`` already
  owns the shared client for the Streamlit process (cached via
  ``@st.cache_resource``) and passes it into both agents. This module
  provides a plain default for tests, CLI scripts, and notebooks that
  instantiate an agent directly without a host-supplied client.

- **Central LLM error-handling decorator.** ``llm_error_handler`` wraps
  agent methods that make Anthropic API calls with a uniform
  try/except for Timeout, Connection, BadRequest, and generic API
  errors -- eliminating the duplicated four-clause blocks that were
  copy-pasted across both agents.

- **Retry decorator.** ``retry_llm_call`` wraps LLM-calling functions
  with exponential backoff on transient errors (timeout, rate-limit,
  connection), so callers don't have to implement retry loops themselves.

This module is intentionally underscore-prefixed: it is package-private.
Call sites outside ``agents/`` should not import from it.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from typing import Any, Callable, TypeVar

from utils.telemetry import (
    COUNTERS,
    LLM_FALLBACK_USED,
    LLM_RETRIES,
)

# ---------------------------------------------------------------------------
# Model ID -- the ONE place to change the Claude model used by the app.
# ---------------------------------------------------------------------------
# To upgrade the model, edit this constant and nothing else. Every LLM call
# site in the agents package reads ``MODEL`` from here.
MODEL: str = "claude-sonnet-4-20250514"

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# SDK import with graceful fallback stubs
# ---------------------------------------------------------------------------
try:
    from anthropic import (  # type: ignore[import-not-found]
        Anthropic,
        APIError,
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        BadRequestError,
        InternalServerError,
        RateLimitError,
    )
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

    # Fallback stubs so ``except`` blocks in the agent modules still parse
    # when the anthropic SDK isn't installed. The agents will simply fall
    # back to their canned non-LLM responses in that case.
    class APIError(Exception):  # type: ignore[no-redef]
        pass

    class APIConnectionError(APIError):  # type: ignore[no-redef]
        pass

    class APIStatusError(APIError):  # type: ignore[no-redef]
        status_code: int = 0

    class APITimeoutError(APIError):  # type: ignore[no-redef]
        pass

    class BadRequestError(APIError):  # type: ignore[no-redef]
        pass

    class InternalServerError(APIError):  # type: ignore[no-redef]
        pass

    class RateLimitError(APIError):  # type: ignore[no-redef]
        pass

    Anthropic = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Zero-trust API key validation
# ---------------------------------------------------------------------------

def require_api_key() -> str:
    """Return the ANTHROPIC_API_KEY or raise a descriptive RuntimeError.

    Call this at application startup when LLM features are mandatory
    (as opposed to optional/degradable).
    """
    api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it in your shell or in a .env file before starting the app. "
            "Get a key at https://console.anthropic.com/settings/keys"
        )
    return api_key


# ---------------------------------------------------------------------------
# Default client factory  (explicit os.getenv, no hardcoded keys)
# ---------------------------------------------------------------------------

def get_default_client() -> "Anthropic | None":
    """
    Construct a default Anthropic client for callers that don't want to
    manage one themselves.

    ``app.py`` caches a shared client for the Streamlit process and passes
    it into both agents; this helper is the fallback for tests, CLI
    scripts, and notebooks that create an agent directly without a
    host-supplied client.

    Returns ``None`` when the ``anthropic`` SDK is not installed **or**
    when ``ANTHROPIC_API_KEY`` is not set, which lets the agents degrade
    gracefully to their fallback responses.
    """
    if not _HAS_ANTHROPIC:
        return None
    api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY environment variable is not set "
            "-- LLM features will be unavailable"
        )
        return None
    return Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Retry decorator for transient LLM errors (exponential backoff)
# ---------------------------------------------------------------------------

_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    # 5xx server errors -- explicitly including 529 "Overloaded", which
    # the SDK surfaces as ``InternalServerError``. Without this clause a
    # single transient overload drops straight to the fallback response.
    InternalServerError,
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors worth another attempt.

    ``json.JSONDecodeError`` is retried too: the decoded text comes from
    an LLM sampling step, and a second attempt often produces clean JSON
    on prompts that hit a mid-text code fence the first time around.
    """
    if isinstance(exc, _RETRYABLE_ERRORS):
        return True
    if isinstance(exc, json.JSONDecodeError):
        return True
    # APIStatusError with a 5xx / 429 status should also be retried, even
    # if the SDK didn't map it to one of the typed subclasses above.
    status = getattr(exc, "status_code", None)
    if isinstance(exc, APIStatusError) and isinstance(status, int):
        if status == 429 or 500 <= status < 600:
            return True
    return False


def retry_llm_call(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable[[F], F]:
    """Decorator that retries on transient Anthropic API errors.

    Handles timeouts, rate-limits, connection errors, 5xx server errors
    (including 529 "Overloaded"), and JSON parse failures with
    exponential backoff. If all retries are exhausted the last exception
    is re-raised so the outer ``llm_error_handler`` can catch it and
    return its fallback value.

    Args:
        max_retries:  Maximum number of retry attempts after the initial call.
        base_delay:   Base delay in seconds; doubles on each retry.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if not _is_retryable(exc):
                        raise
                    last_error = exc
                    if attempt < max_retries:
                        # Telemetry: every retry attempt counts. The
                        # Diagnostics tab can compare LLM_RETRIES against
                        # LLM_CALLS to spot a flaky deployment.
                        COUNTERS.incr(LLM_RETRIES)
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Retryable LLM error in %s (attempt %d/%d), "
                            "retrying in %.1fs: %s",
                            func.__name__, attempt + 1, max_retries,
                            delay, exc,
                        )
                        time.sleep(delay)
            raise last_error  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Central LLM error-handling decorator
# ---------------------------------------------------------------------------

def _handle_llm_error(
    instance: Any,
    error_type: str,
    label: str,
    error: Any,
) -> None:
    """Route an LLM error to *instance.issue_detector* or the module logger."""
    detail = f"{label}: {error}"
    detector = getattr(instance, "issue_detector", None)
    if detector is not None and hasattr(detector, "log_api_error"):
        detector.log_api_error(error_type, detail, recovered=True)
    else:
        logger.warning("LLM error [%s] %s", error_type, detail)


def llm_error_handler(
    fallback: Callable[..., Any],
    *,
    error_label: str = "",
) -> Callable[[F], F]:
    """
    Decorator that wraps LLM-calling methods with standard Anthropic API
    error handling.

    On any API / network / JSON-parse error the decorator:

    1. Logs the error via ``self.issue_detector`` (if the instance has one)
       or via the module-level logger.
    2. Returns ``fallback(self, *args, **kwargs)`` so the caller always
       gets a usable value.

    Args:
        fallback:    Callable with the **same** ``(self, *args, **kwargs)``
                     signature as the decorated method.  Called to produce
                     the return value when the LLM call fails.
        error_label: Human-readable label for log messages.  Defaults to
                     the decorated function's ``__name__``.
    """

    def decorator(func: F) -> F:
        label = error_label or func.__name__

        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except (APITimeoutError, APIConnectionError) as e:
                _handle_llm_error(self, "api_timeout", label, e)
            except RateLimitError as e:
                _handle_llm_error(self, "rate_limit", label, e)
            except InternalServerError as e:
                # Covers 5xx including 529 "Overloaded" after retries exhausted.
                logger.exception("InternalServerError in %s", label)
                _handle_llm_error(self, "server_error", label, e)
            except BadRequestError as e:
                logger.exception("BadRequestError in %s", label)
                _handle_llm_error(self, "bad_request", label, e)
            except json.JSONDecodeError as e:
                _handle_llm_error(self, "json_parse", label, e)
            except APIError as e:
                logger.exception("APIError in %s", label)
                _handle_llm_error(self, "api_error", label, e)
            except Exception:
                logger.exception("Unexpected error in %s", label)
                _handle_llm_error(self, "unknown", label, "see logs")
            # Telemetry: fallback path was taken — i.e., retries were
            # exhausted (or the error was non-retryable). The
            # Diagnostics tab uses this to surface "the LLM is failing
            # often enough that the canned fallback is being shipped".
            COUNTERS.incr(LLM_FALLBACK_USED)
            return fallback(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Robust JSON extraction from LLM output
# ---------------------------------------------------------------------------
# Match the OUTERMOST balanced JSON object or array. ``re.DOTALL`` lets ``.``
# cross newlines so multi-line JSON parses; the lazy-greedy alternation picks
# the largest enclosing ``{...}`` or ``[...]`` in the response — which beats
# the previous fence-stripping when the LLM emits prose like
# "Sure! Here's your JSON: { ... }".
_JSON_OBJECT_OR_ARRAY = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse a JSON object or array from possibly prose-wrapped LLM output.

    Tolerant to:
      - leading/trailing whitespace
      - ```json or plain ``` code fences
      - prose around the JSON (e.g. "Sure! Here is the JSON: { ... } Hope this helps!")

    Raises ``json.JSONDecodeError`` (preserving the
    ``retry_llm_call`` retry semantics — JSONDecodeError is in
    ``_RETRYABLE_ERRORS`` so a malformed sample re-rolls the LLM).
    """
    if text is None:
        raise json.JSONDecodeError("LLM response was None", "", 0)

    cleaned = text.strip()
    # Strip both ```json and bare ``` fences first; this handles the common case
    # without paying the regex cost.
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Fast path: well-formed JSON after fence-stripping.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the largest balanced object or array. This handles
    # "Here is your quiz: {...}." and similar wrappers.
    match = _JSON_OBJECT_OR_ARRAY.search(cleaned)
    if match is None:
        raise json.JSONDecodeError(
            "No JSON object or array found in LLM response",
            cleaned,
            0,
        )
    return json.loads(match.group(0))


__all__ = [
    "MODEL",
    "Anthropic",
    "APIError",
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "BadRequestError",
    "InternalServerError",
    "RateLimitError",
    "_HAS_ANTHROPIC",
    "get_default_client",
    "require_api_key",
    "retry_llm_call",
    "llm_error_handler",
    "extract_json",
]
