"""Tests for scripts/migrate_layout.py — moves cruft into new subdirs.

Builds a fixture results/ mirroring the messes catalogued in the
2026-05-14 sister-repo survey. Confirms canonical pipeline files are
NEVER moved and that cruft sorts into the right subdirectories.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/migrate_layout.py"


def _build_messy_results(root: Path) -> None:
    """Create a results/ tree with the cruft patterns the survey found."""
    r = root / "results"
    r.mkdir(parents=True)
    (r / "foo_lit_bibliography.csv").write_text("doi,title\n")
    (r / "foo_lit_bibliography.bib").write_text("@article{}\n")
    (r / "foo_lit_bibliography_embedded.csv").write_text("id\n")
    (r / "foo_lit_bibliography_embedded.embeddings.npy").write_bytes(b"\x93NUMPY")
    (r / "foo_lit_bibliography_embedded.embeddings.ids.txt").write_text("1\n")
    (r / "foo_lit_labeled_corpus.csv").write_text("id\n")
    (r / "foo_lit_labeled_corpus.checkpoints").mkdir()
    (r / "foo_lit_labeled_corpus.checkpoints/chunk_00001.csv").write_text("id\n")
    (r / "harvest.log").write_text("x")
    (r / "label.log").write_text("x")
    (r / "label_v2.log").write_text("x")
    (r / "errors.log").write_text("x")
    (r / "raw_archive.parquet").write_bytes(b"PAR1")
    (r / "foo_lit_bibliography.bib.bak").write_text("x")
    (r / "foo_lit_labeled_corpus.pre_source_recovery.bak").write_text("x")
    (r / "wos_gap_records.csv").write_text("id\n")
    (r / "wos_gap_records_embedded.csv").write_text("id\n")
    (r / "wos_gap_records_labeled.csv").write_text("id\n")
    (r / "pilot50_labeled.csv").write_text("id\n")
    (r / "gap_matrix_cells.csv").write_text("id\n")
    (r / "cross_project_bridges_2026-04-01.csv").write_text("id\n")
    (r / "management_levers.png").write_bytes(b"\x89PNG")
    (r / "management_levers.pdf").write_bytes(b"%PDF-1.4")


def test_dry_run_lists_moves_without_changing_anything(tmp_path: Path):
    _build_messy_results(tmp_path)
    before = sorted(p.relative_to(tmp_path).as_posix()
                    for p in (tmp_path / "results").rglob("*"))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    after = sorted(p.relative_to(tmp_path).as_posix()
                   for p in (tmp_path / "results").rglob("*"))
    assert before == after, "dry-run modified files"
    assert "would move" in result.stdout.lower()
    assert "harvest.log" in result.stdout


def test_apply_moves_cruft_preserves_canonical(tmp_path: Path):
    _build_messy_results(tmp_path)
    subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--apply"],
        check=True, capture_output=True, text=True,
    )
    r = tmp_path / "results"
    assert (r / "foo_lit_bibliography.csv").exists()
    assert (r / "foo_lit_bibliography_embedded.embeddings.npy").exists()
    assert (r / "foo_lit_labeled_corpus.csv").exists()
    assert (r / "foo_lit_labeled_corpus.checkpoints/chunk_00001.csv").exists()
    assert (r / "logs/harvest.log").exists()
    assert (r / "logs/label.log").exists()
    assert (r / "logs/label_v2.log").exists()
    assert (r / "logs/errors.log").exists()
    assert not (r / "harvest.log").exists()
    assert (r / "archive/raw_archive.parquet").exists()
    assert (r / "archive/foo_lit_bibliography.bib.bak").exists()
    assert (r / "archive/foo_lit_labeled_corpus.pre_source_recovery.bak").exists()
    assert (r / "gapfills/wos_gap/records.csv").exists() or \
           (r / "gapfills/wos_gap/wos_gap_records.csv").exists()
    assert (r / "pilots/pilot50_labeled.csv").exists()
    assert (r / "analysis/gap_matrix_cells.csv").exists()
    assert (r / "analysis/cross_project_bridges_2026-04-01.csv").exists()
    assert (r / "analysis/management_levers.png").exists()
    assert (r / "analysis/management_levers.pdf").exists()


def test_apply_is_idempotent(tmp_path: Path):
    _build_messy_results(tmp_path)
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--apply"],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--apply"],
                            capture_output=True, text=True, check=True)
    assert "0 files moved" in result.stdout.lower() or "nothing to move" in result.stdout.lower()
