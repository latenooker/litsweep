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
            f"Valid: {sorted(BACKENDS) or ['stanford', 'ollama']}"
        )
    cls = BACKENDS[name]
    if not hasattr(cls, "__dataclass_fields__"):
        raise SystemExit(f"Backend {name!r} ({cls!r}) must be a @dataclass.")
    accepted = {k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__}
    return cls(**accepted)
