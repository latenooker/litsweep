"""End-to-end smoke test: scaffold succeeds and creates new layout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_scaffold_creates_new_subdirs(tmp_path: Path):
    """Scaffold creates the cleanup-layout results/ subdirs and data/."""
    target = tmp_path / "foo_lit"
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/scaffold_new_search.py"),
            str(target), "--name", "foo_lit", "--no-git",
        ],
        check=True, capture_output=True, text=True,
    )
    for sub in ("gapfills", "pilots", "analysis", "archive", "logs"):
        assert (target / "results" / sub).is_dir(), f"missing results/{sub}/"
    assert (target / "results/raw").is_dir()
    assert (target / "data").is_dir()
    assert (target / "results/gapfills/.gitkeep").is_file()
    assert not (target / "scripts/.gitkeep").exists()


def test_scaffolded_orchestrator_dry_run_succeeds(tmp_path: Path):
    """A freshly scaffolded orchestrator passes --dry-run."""
    target = tmp_path / "bar_lit"
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/scaffold_new_search.py"),
            str(target), "--name", "bar_lit", "--no-git",
        ],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [sys.executable, str(target / "bar_lit_search.py"), "--dry-run"],
        cwd=target, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
