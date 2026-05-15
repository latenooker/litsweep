"""Tests for label_backends registry dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

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
