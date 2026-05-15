"""Migrate an existing litsweep project's `results/` to the new layout.

Moves accumulated cruft (logs, .bak files, gap-fills, pilots, analysis
artifacts, parquet archives) into the additive subdirectories
introduced in the 2026-05-14 layout cleanup. Canonical pipeline files
(`<slug>_bibliography*.csv`, `<slug>_labeled_corpus*`, embedding
sidecars, checkpoints) are NEVER touched, so cross-project tools that
hardcode those paths keep working.

Idempotent: a second run on a tidy tree is a no-op.

Usage::

    python scripts/migrate_layout.py /path/to/project           # dry-run
    python scripts/migrate_layout.py /path/to/project --apply   # do it
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# (glob, destination subdir). Order matters: more specific first.
RULES: list[tuple[str, str]] = [
    ("*.log", "logs"),
    ("raw_archive*.parquet", "archive"),
    ("*.bak", "archive"),
    ("*.bak.*", "archive"),
    ("pilot*_labeled*.csv", "pilots"),
    ("pilot*.csv", "pilots"),
    ("gap_matrix_*.csv", "analysis"),
    ("cross_project_*.csv", "analysis"),
    ("coverage_matrix_*.csv", "analysis"),
    ("knn_anchor_*.csv", "analysis"),
    ("*_reading_list.csv", "analysis"),
    ("*.png", "analysis"),
    ("*.pdf", "analysis"),
]

# Gap-fill CSVs: the slug must END in `_gap` (e.g. wos_gap_records.csv),
# NOT start with `gap_` (gap_matrix_*.csv stays an analysis artifact).
# Routed to gapfills/<slug>/<stage>.csv; stage defaults to "records".
_GAP_RE = re.compile(
    r"^(?P<name>[a-z][a-z0-9_]*_gap)(?:_(?P<stage>.+))?\.csv$"
)


def _canonical_protected(name: str) -> bool:
    """True if a filename is canonical pipeline output and must not move.

    A ``.bak`` (or ``.bak.*``) copy of a canonical file is cruft, not
    the canonical file itself, so it is explicitly *not* protected even
    though it contains ``_bibliography`` / ``_labeled_corpus``.

    Args:
        name: The bare filename (no directory component).

    Returns:
        ``True`` if the file is a live canonical pipeline artifact that
        must never be moved; ``False`` otherwise.
    """
    if name.endswith(".bak") or ".bak." in name:
        return False
    if "_bibliography" in name:
        return True
    if "_labeled_corpus" in name:
        return True
    if name.endswith(".embeddings.npy") or name.endswith(".embeddings.ids.txt"):
        return True
    return False


def _plan_gapfill(path: Path) -> tuple[Path, Path] | None:
    """Map a gap-fill CSV to gapfills/<name>/<stage>.csv, or None.

    Args:
        path: A file inside ``results/``.

    Returns:
        A ``(src, dst)`` tuple if ``path`` is a gap-fill CSV, else
        ``None``.
    """
    m = _GAP_RE.match(path.name)
    if not m:
        return None
    name = m.group("name")
    stage = m.group("stage") or "records"
    return (path, path.parent / "gapfills" / name / f"{stage}.csv")


def _plan_moves(results: Path) -> list[tuple[Path, Path]]:
    """Compute the list of ``(src, dst)`` moves for a ``results/`` dir.

    Args:
        results: The project's ``results/`` directory.

    Returns:
        Ordered list of source/destination path pairs to move.
    """
    moves: list[tuple[Path, Path]] = []
    for entry in sorted(results.iterdir()):
        if not entry.is_file():
            continue
        if _canonical_protected(entry.name):
            continue
        gap = _plan_gapfill(entry)
        if gap is not None:
            moves.append(gap)
            continue
        for pattern, sub in RULES:
            if entry.match(pattern):
                moves.append((entry, results / sub / entry.name))
                break
    return moves


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("project", type=Path, help="Project root (contains results/).")
    p.add_argument("--apply", action="store_true",
                   help="Without this flag, prints the plan and exits.")
    args = p.parse_args(argv)

    project: Path = args.project.resolve()
    results = project / "results"
    if not results.is_dir():
        raise SystemExit(f"No results/ directory at {results}")

    moves = _plan_moves(results)
    if not moves:
        print("nothing to move (0 files moved).")
        return 0

    from collections import Counter
    dst_counts = Counter(dst for _, dst in moves)
    collisions = sorted(
        str(d.relative_to(project)) for d, n in dst_counts.items() if n > 1
    )
    if collisions:
        raise SystemExit(
            "ERROR: multiple source files map to the same destination "
            f"(refusing to move anything): {collisions}. Resolve the "
            "duplicates manually, then re-run."
        )

    for src, dst in moves:
        rel_src = src.relative_to(project)
        rel_dst = dst.relative_to(project)
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise SystemExit(
                    f"ERROR: destination already exists, refusing to "
                    f"overwrite: {dst.relative_to(project)} (from "
                    f"{src.relative_to(project)}). Resolve manually."
                )
            shutil.move(str(src), str(dst))
            print(f"moved {rel_src} -> {rel_dst}")
        else:
            print(f"would move {rel_src} -> {rel_dst}")

    if not args.apply:
        print(f"\n(dry-run) would move {len(moves)} files. Re-run with --apply.")
    else:
        print(f"\nmoved {len(moves)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
