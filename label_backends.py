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

from typing import Any


class BackendError(Exception):
    """Permanent failure from a label backend (retries exhausted, bad config).

    The driver converts ``reason`` into a project-shaped error label via
    its local ``_error_label`` so the schema dependency stays in the
    driver, not the backend.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


BACKENDS: dict[str, type] = {}


def make_backend(name: str, **kwargs: Any):
    """Build a backend by name; raise SystemExit on unknown name."""
    if name not in BACKENDS:
        raise SystemExit(
            f"Unknown label backend: {name!r}. "
            f"Valid: {sorted(BACKENDS) or ['stanford', 'ollama']}"
        )
    cls = BACKENDS[name]
    accepted = {k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__}
    return cls(**accepted)
