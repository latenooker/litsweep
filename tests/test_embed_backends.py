"""Tests for embed_backends registry dispatch and OllamaEmbedBackend."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embed_backends


def _mock_post(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_make_backend_unknown_name_exits():
    with pytest.raises(SystemExit) as exc:
        embed_backends.make_backend("foo")
    assert "ollama" in str(exc.value)


def test_ollama_embed_backend_returns_normalized_matrix():
    backend = embed_backends.make_backend("ollama")
    texts = ["a", "b"]
    payload = {"embeddings": [[3.0, 4.0], [0.0, 5.0]]}
    with patch("embed_backends.requests.post",
               return_value=_mock_post(payload)):
        mat = backend.embed(texts)
    assert mat.shape == (2, 2)
    norms = np.linalg.norm(mat, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
