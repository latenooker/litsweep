"""Label-backend registry: pick which LLM provider drives `label_with_stanford.py`.

V1 ships Stanford gateway (existing behavior, default) and local Ollama
(`/api/chat` with `format="json"`). Additional providers (OpenAI,
Anthropic, Voyage, etc.) slot in as one new dataclass + one entry in
`BACKENDS`.

The driver in `scripts/label_with_stanford.py` keeps the project's
`SYSTEM_PROMPT` and `_error_label` (both tied to per-project schemas).
Backends only know how to call their provider and return a parsed dict
on success, or raise `BackendError` on permanent failure.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger("label_backends")


class BackendError(Exception):
    """Permanent failure from a label backend (retries exhausted, bad config).

    The driver converts ``reason`` into a project-shaped error label via
    its local ``_error_label`` so the schema dependency stays in the
    driver, not the backend.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3
MAX_429_RETRIES = 8


def _strip_code_fence(text: str) -> str:
    """Drop ```json ... ``` fences if a model added them despite instructions."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


@dataclass
class StanfordBackend:
    """Stanford AI API gateway (OpenAI-compatible chat-completions).

    Carries over from the pre-registry ``call_stanford`` verbatim:
    bearer-token auth, 429-aware retry loop with bounded backoff,
    optional client-side rate limiter via ``min_interval_s``.

    Attributes:
        api_key: Required. Pass ``STANFORD_API_KEY`` here.
        api_url: Base URL; default Stanford prod.
        model: Model id; default ``gemini-2.0-flash-lite-001``.
        temperature: Sampling temperature forwarded to the endpoint.
        max_tokens: Max completion tokens forwarded to the endpoint.
        min_interval_s: Minimum wall-clock seconds between calls (>0 to
            pre-throttle below the gateway's quota).
    """

    api_key: str
    api_url: str = "https://aiapi-prod.stanford.edu/v1"
    model: str = "gemini-2.0-flash-lite-001"
    temperature: float = 0.0
    max_tokens: int = 400
    min_interval_s: float = 0.0
    _last_request_t: float = field(default=0.0, init=False, repr=False)

    def check(self) -> None:
        """Probe the gateway before a run.

        Raises:
            SystemExit: If ``api_key`` is empty. A failure to list
                models is logged but non-fatal (best-effort probe).
        """
        if not self.api_key:
            raise SystemExit(
                "Stanford API key not set. Pass --api-key or STANFORD_API_KEY."
            )
        url = f"{self.api_url.rstrip('/')}/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
            models = [m.get("id") for m in body.get("data", []) if m.get("id")]
            logger.info("Stanford reachable. %d models available.", len(models))
            if self.model not in models:
                logger.warning(
                    "Model %r not in list; will attempt anyway. Sample: %s",
                    self.model, models[:5],
                )
        except Exception as exc:  # pragma: no cover - probe is best-effort
            logger.warning("Could not list Stanford models (probe only): %s", exc)

    def call(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        """Call the gateway and return parsed JSON, or raise BackendError.

        Args:
            prompt: The user-role content (title + abstract block).
            system_prompt: The project SYSTEM_PROMPT (schema instructions).

        Returns:
            The parsed JSON object the model returned.

        Raises:
            BackendError: On permanent failure (HTTP error after retries,
                429 budget exhausted, unparseable response).
        """
        if self.min_interval_s > 0:
            elapsed = time.monotonic() - self._last_request_t
            if elapsed < self.min_interval_s:
                time.sleep(self.min_interval_s - elapsed)
        self._last_request_t = time.monotonic()

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")

        url = f"{self.api_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        attempt = 0
        rate_limit_attempts = 0
        while attempt < MAX_RETRIES:
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                    body = json.loads(resp.read())
                content = body["choices"][0]["message"]["content"]
                return json.loads(_strip_code_fence(content))
            except urllib.error.HTTPError as exc:
                err_body = (
                    exc.read().decode("utf-8", errors="replace")
                    if hasattr(exc, "read") else ""
                )
                if exc.code == 429:
                    rate_limit_attempts += 1
                    if rate_limit_attempts >= MAX_429_RETRIES:
                        raise BackendError(
                            f"HTTP 429 after {MAX_429_RETRIES} retries"
                        )
                    wait = min(2 ** (rate_limit_attempts + 1), 60)
                    logger.warning("429 rate limit (%d/%d), sleeping %ds",
                                   rate_limit_attempts, MAX_429_RETRIES, wait)
                    time.sleep(wait)
                    continue
                elif attempt < MAX_RETRIES - 1:
                    logger.warning("HTTP %d (attempt %d): %s",
                                   exc.code, attempt + 1, err_body[:200])
                    time.sleep(2 ** attempt)
                else:
                    raise BackendError(f"HTTP {exc.code}: {err_body[:200]}")
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                if attempt < MAX_RETRIES - 1:
                    logger.warning("parse error attempt %d: %s", attempt + 1, exc)
                    time.sleep(1)
                else:
                    raise BackendError(f"parse: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                if attempt < MAX_RETRIES - 1:
                    logger.warning("attempt %d failed: %s", attempt + 1, exc)
                    time.sleep(2 ** attempt)
                else:
                    raise BackendError(str(exc))
            attempt += 1
        raise BackendError("max retries exhausted")


@dataclass
class OllamaLabelBackend:
    """Local Ollama chat backend with ``format="json"`` for structured labels.

    Reuses the same daemon the embedder already needs. Smaller open-weights
    models are less reliable at strict-JSON than gateway gemini/gpt — set
    ``model`` to something instruction-tuned (llama3.1, qwen2.5, mistral-nemo)
    and verify with a 50-record pilot before a full run.

    Attributes:
        host: Ollama base URL.
        model: Ollama model tag (must be pulled: ``ollama pull <model>``).
        temperature: Forwarded; 0.0 for label determinism.
        num_predict: Max tokens; Ollama's analogue of max_tokens.
        timeout_s: Per-request timeout.
        max_retries: Retries on RequestException or unparseable JSON.
    """

    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.0
    num_predict: int = 400
    timeout_s: int = 300
    max_retries: int = 3

    def check(self) -> None:
        """Probe the Ollama daemon.

        Raises:
            SystemExit: If the daemon is unreachable or the model is
                not pulled.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(
                f"Cannot reach Ollama at {self.host}: {exc}\n"
                f"Start it with: ollama serve   "
                f"(and ensure: ollama pull {self.model})"
            )
        tags = [m["name"] for m in resp.json().get("models", [])]
        base = self.model.split(":")[0]
        if not any(t == self.model or t.startswith(base + ":") for t in tags):
            raise SystemExit(
                f"Model '{self.model}' not pulled. Run: ollama pull {self.model}"
            )

    def call(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        """Call local Ollama chat and return parsed JSON, or raise BackendError.

        Args:
            prompt: User-role content (title + abstract block).
            system_prompt: The project SYSTEM_PROMPT (schema instructions).

        Returns:
            The parsed JSON object the model returned.

        Raises:
            BackendError: On permanent failure (transport error or
                unparseable response after ``max_retries``).
        """
        url = f"{self.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                return json.loads(_strip_code_fence(content))
            except (requests.RequestException, ValueError, KeyError) as err:
                last_err = err
                logger.warning(
                    "Ollama label attempt %d/%d failed: %s",
                    attempt, self.max_retries, err,
                )
                time.sleep(1.5 * attempt)
        raise BackendError(
            f"Ollama label failed after {self.max_retries} retries: {last_err}"
        )


BACKENDS: dict[str, type] = {
    "stanford": StanfordBackend,
    "ollama": OllamaLabelBackend,
}


def make_backend(name: str, **kwargs: Any) -> Any:
    """Build a label backend instance by name.

    Args:
        name: Registry key (must be present in ``BACKENDS``).
        **kwargs: Candidate constructor arguments; only those matching
            the backend dataclass's fields are forwarded, so callers can
            pass the union of all backends' options (e.g. an argparse
            namespace) without error.

    Returns:
        An instantiated backend (one of the ``BACKENDS`` values).

    Raises:
        SystemExit: If ``name`` is not a registered backend, or the
            registered class is not a dataclass.
    """
    if name not in BACKENDS:
        raise SystemExit(
            f"Unknown label backend: {name!r}. "
            f"Valid: {sorted(BACKENDS)}"
        )
    cls = BACKENDS[name]
    if not hasattr(cls, "__dataclass_fields__"):
        raise SystemExit(f"Backend {name!r} ({cls!r}) must be a @dataclass.")
    accepted = {k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__}
    return cls(**accepted)
