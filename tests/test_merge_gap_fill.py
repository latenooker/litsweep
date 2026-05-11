"""Unit tests for merge_gap_fill's source-coverage guardrail."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from merge_gap_fill import _check_source_coverage  # noqa: E402


def _frame(coverage: float, n: int = 100) -> pd.DataFrame:
    """Build a frame with N rows where ``coverage`` fraction have a
    non-empty ``source_databases``."""
    filled = int(round(n * coverage))
    sources = ["openalex"] * filled + [""] * (n - filled)
    return pd.DataFrame({
        "doi": [f"10.x/{i}" for i in range(n)],
        "title": [f"Title {i}" for i in range(n)],
        "source_databases": sources,
    })


def test_no_warning_when_main_lacks_source_databases():
    main = pd.DataFrame({"doi": ["a"], "title": ["t"]})
    merged = pd.DataFrame({"doi": ["a"], "title": ["t"], "source_databases": [""]})
    assert _check_source_coverage(main, merged) is None


def test_no_warning_when_main_already_empty():
    """If the bug already happened upstream, the merge isn't the cause."""
    main = _frame(coverage=0.0)
    merged = _frame(coverage=0.0)
    assert _check_source_coverage(main, merged) is None


def test_no_warning_when_coverage_preserved():
    main = _frame(coverage=0.95)
    merged = _frame(coverage=0.92)  # tiny drop from dedup overlap is normal
    assert _check_source_coverage(main, merged) is None


def test_warning_when_full_collapse():
    """The worm-tea-lit scenario: 100% main coverage → 0% merged coverage."""
    main = _frame(coverage=1.0)
    merged = _frame(coverage=0.0)
    msg = _check_source_coverage(main, merged)
    assert msg is not None
    assert "780588c" in msg or "dedup.py" in msg


def test_warning_when_severe_drop():
    """80% → 10% is suspicious enough to flag."""
    main = _frame(coverage=0.80)
    merged = _frame(coverage=0.10)
    msg = _check_source_coverage(main, merged)
    assert msg is not None


def test_no_warning_on_mild_drop():
    """50% → 35% can happen if the gap source is sparse — don't false-positive."""
    main = _frame(coverage=0.50)
    merged = _frame(coverage=0.35)
    assert _check_source_coverage(main, merged) is None
