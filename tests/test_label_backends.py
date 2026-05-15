"""Tests for label_backends registry dispatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import label_backends


def test_make_backend_unknown_name_exits():
    with pytest.raises(SystemExit) as exc:
        label_backends.make_backend("does_not_exist", api_key="x")
    assert "does_not_exist" in str(exc.value)
    assert "stanford" in str(exc.value)  # lists valid options


def test_backend_error_carries_reason():
    err = label_backends.BackendError("HTTP 500")
    assert err.reason == "HTTP 500"


def _mock_urlopen_ok(payload: dict) -> MagicMock:
    """Build a context-manager mock that returns ``payload`` JSON."""
    cm = MagicMock()
    cm.__enter__.return_value = MagicMock(
        read=MagicMock(return_value=json.dumps(payload).encode("utf-8"))
    )
    cm.__exit__.return_value = False
    return cm


def test_stanford_backend_parses_json_response():
    backend = label_backends.make_backend("stanford", api_key="sk-x")
    payload = {
        "choices": [
            {"message": {"content": '{"relevance": "core", "rationale": "ok"}'}}
        ]
    }
    with patch("label_backends.urllib.request.urlopen",
               return_value=_mock_urlopen_ok(payload)):
        result = backend.call("prompt", "system")
    assert result == {"relevance": "core", "rationale": "ok"}


def test_stanford_backend_strips_code_fences():
    backend = label_backends.make_backend("stanford", api_key="sk-x")
    payload = {
        "choices": [{"message": {"content": '```json\n{"relevance": "core"}\n```'}}]
    }
    with patch("label_backends.urllib.request.urlopen",
               return_value=_mock_urlopen_ok(payload)):
        result = backend.call("prompt", "system")
    assert result == {"relevance": "core"}


def _mock_requests_post(json_payload: dict, status: int = 200) -> MagicMock:
    """Mock for requests.post returning ``json_payload``."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_payload
    resp.raise_for_status.side_effect = (
        None if status < 400 else Exception(f"HTTP {status}")
    )
    return resp


def test_ollama_label_backend_parses_chat_response():
    backend = label_backends.make_backend("ollama")
    payload = {
        "message": {"content": '{"relevance": "adjacent", "rationale": "x"}'}
    }
    with patch("label_backends.requests.post",
               return_value=_mock_requests_post(payload)):
        result = backend.call("prompt", "system")
    assert result == {"relevance": "adjacent", "rationale": "x"}


def test_ollama_label_backend_raises_on_unparseable():
    backend = label_backends.make_backend("ollama")
    payload = {"message": {"content": "not json at all"}}
    with patch("label_backends.requests.post",
               return_value=_mock_requests_post(payload)):
        with patch("label_backends.time.sleep"):
            with pytest.raises(label_backends.BackendError):
                backend.call("prompt", "system")
