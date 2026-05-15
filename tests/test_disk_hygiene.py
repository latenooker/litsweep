"""Tests for scripts/disk_hygiene.py — parquet-archive results/raw/ then delete."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import disk_hygiene


def _make_raw(results: Path, n: int = 3) -> None:
    raw = results / "raw"
    raw.mkdir(parents=True)
    for i in range(n):
        (raw / f"openalex__query_{i}.json").write_text(f'{{"i": {i}}}')


def test_archive_then_delete(tmp_path: Path):
    results = tmp_path / "results"
    _make_raw(results, 3)
    out = disk_hygiene.archive_raw(results, delete=True)
    assert out == results / "archive" / "raw_archive.parquet"
    assert out.exists()
    assert not (results / "raw").exists()                # deleted on success
    df = pd.read_parquet(out)
    assert len(df) == 3
    assert set(df.columns) == {"source", "query", "payload_json"}
    assert (df["source"] == "openalex").all()


def test_no_delete_keeps_raw(tmp_path: Path):
    results = tmp_path / "results"
    _make_raw(results, 2)
    out = disk_hygiene.archive_raw(results, delete=False)
    assert out.exists()
    assert (results / "raw").exists()                    # kept
    assert len(list((results / "raw").glob("*.json"))) == 2


def test_missing_raw_is_noop(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    assert disk_hygiene.archive_raw(results, delete=True) is None


def test_empty_raw_is_noop(tmp_path: Path):
    results = tmp_path / "results"
    (results / "raw").mkdir(parents=True)
    assert disk_hygiene.archive_raw(results, delete=True) is None
    assert (results / "raw").exists()                    # nothing archived, not deleted


def test_cli_main_no_delete(tmp_path: Path):
    results = tmp_path / "results"
    _make_raw(results, 1)
    rc = disk_hygiene.main(["--results", str(results), "--no-delete"])
    assert rc == 0
    assert (results / "archive" / "raw_archive.parquet").exists()
    assert (results / "raw").exists()
