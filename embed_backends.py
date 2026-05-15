"""Embed-backend registry: pluggable embedding providers.

V1 ships local Ollama (the only existing path). The shape leaves room
for Voyage / Jina / Cohere / OpenAI-compatible additions as one new
dataclass + one entry in ``BACKENDS``.

Backends expose a single ``embed(texts) -> np.ndarray`` returning a
``(N, D)`` float32 matrix, L2-normed by default (see ``normalize``),
so dot products are cosines.
``check()`` raises ``SystemExit`` with a friendly message if the
backend can't be reached.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger("embed_backends")


@dataclass
class OllamaEmbedBackend:
    """Local Ollama embeddings via ``/api/embed``.

    Attributes:
        model: Ollama embedding model tag. ``bge-m3`` is multilingual
            (100+ langs) and is the canonical default for the
            multilingual lit-search use case.
        host: Ollama base URL.
        batch_size: Texts per request.
        timeout_s: Per-batch timeout.
        max_retries: Retries on RequestException or bad shape.
        normalize: L2-norm output so dot product equals cosine.
    """

    model: str = "bge-m3"
    host: str = "http://localhost:11434"
    batch_size: int = 32
    timeout_s: int = 300
    max_retries: int = 3
    normalize: bool = True

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
        if not any(t.startswith(self.model) for t in tags):
            raise SystemExit(
                f"Model '{self.model}' not pulled. Run: ollama pull {self.model}"
            )

    def _post_batch(self, texts: Sequence[str]) -> np.ndarray:
        url = f"{self.host.rstrip('/')}/api/embed"
        payload = {"model": self.model, "input": list(texts)}
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                embeds = data.get("embeddings")
                if embeds is None and "embedding" in data:
                    embeds = [data["embedding"]]
                if not embeds:
                    raise RuntimeError(f"empty embeddings: {data!r}")
                arr = np.asarray(embeds, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] != len(texts):
                    raise RuntimeError(f"shape {arr.shape} != batch {len(texts)}")
                return arr
            except (requests.RequestException, ValueError, KeyError, RuntimeError) as err:
                last_err = err
                logger.warning(
                    "Ollama embed attempt %d/%d failed: %s",
                    attempt, self.max_retries, err,
                )
                time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"Ollama embedding failed after {self.max_retries} retries: {last_err}"
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of texts; returns ``(N, D)`` float32, L2-normed.

        Args:
            texts: Strings to encode.

        Returns:
            ``(len(texts), D)`` float32 matrix; rows L2-normed when
            ``normalize`` is true. Empty input returns ``(0, 0)``.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out: list[np.ndarray] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(texts), self.batch_size),
                      total=n_batches, desc=f"embed ({self.model})"):
            batch = texts[i : i + self.batch_size]
            out.append(self._post_batch(batch))
        matrix = np.vstack(out)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


BACKENDS: dict[str, type] = {
    "ollama": OllamaEmbedBackend,
}


def make_backend(name: str, **kwargs: Any) -> Any:
    """Build an embed backend instance by name.

    Args:
        name: Registry key (must be present in ``BACKENDS``).
        **kwargs: Candidate constructor arguments; only those matching
            the backend dataclass's fields are forwarded, so callers can
            pass the union of all backends' options (e.g. an argparse
            namespace) without error.

    Returns:
        An instantiated embed backend.

    Raises:
        SystemExit: If ``name`` is not a registered backend, or the
            registered class is not a dataclass.
    """
    if name not in BACKENDS:
        raise SystemExit(
            f"Unknown embed backend: {name!r}. Valid: {sorted(BACKENDS)}"
        )
    cls = BACKENDS[name]
    if not hasattr(cls, "__dataclass_fields__"):
        raise SystemExit(f"Backend {name!r} ({cls!r}) must be a @dataclass.")
    accepted = {k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__}
    return cls(**accepted)
