"""Merge a gap-fill harvest into the main bibliography.

Workflow this script supports:

1. Run a gap-fill harvest (e.g. WoS Expanded with the API key newly
   set) into a sibling output directory: ``results_gap_fill/``.
2. Run this script to dedup-merge the gap-fill bibliography into the
   main ``results/<slug>_bibliography.csv``, copy the gap-fill's
   ``raw/`` JSONs into the main ``raw/`` directory (so
   ``backfill_abstracts.py`` can find them), and overwrite the main
   bibliography in place.
3. Re-run ``scripts/embed_filter.py`` against the merged CSV; the
   embedding cache skips already-encoded rows so only the newly
   merged records pay the encoding cost.

The dedup uses the project's shared ``dedup.dedup()`` (DOI + Jaccard
title similarity), so the final row count reflects records that are
genuinely new vs. the main bibliography.

Usage::

    # Default: auto-detect both CSVs (single match in each dir)
    python scripts/merge_gap_fill.py --gap-dir results_wos_expanded

    # Explicit input
    python scripts/merge_gap_fill.py \\
        --main-csv results/my_lit_bibliography.csv \\
        --gap-csv results_wos_expanded/my_lit_bibliography.csv \\
        --gap-raw-dir results_wos_expanded/raw \\
        --main-raw-dir results/raw
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("merge_gap_fill")


def _check_source_coverage(
    main: pd.DataFrame, merged: pd.DataFrame
) -> str | None:
    """Return a guardrail error message if source_databases collapsed.

    Catches the byte-copy-divergence scenario diagnosed in worm-tea-lit:
    if a project's local ``dedup.py`` predates upstream commit
    ``780588c`` ("dedup: read existing source_databases column when
    available"), the local dedup reads only ``source_database``
    (singular) from inputs. Already-deduped frames don't have that
    column — they have ``source_databases`` (plural) — so every output
    row gets ``source_databases = ""`` and provenance is silently
    wiped.

    The check fires only when ``main`` had non-trivial coverage to
    begin with (i.e. the bug-state isn't already-present upstream) and
    the merged frame's coverage drops far enough that legitimate
    dedup-overlap can't account for it.

    Args:
        main: Pre-merge main bibliography.
        merged: Post-dedup merged bibliography.

    Returns:
        Error message if collapse detected; ``None`` otherwise.
    """
    if "source_databases" not in main.columns:
        return None
    main_filled = (
        main["source_databases"].fillna("").astype(str).str.len() > 0
    ).sum()
    if main_filled == 0:
        return None
    merged_filled = (
        merged["source_databases"].fillna("").astype(str).str.len() > 0
    ).sum()
    main_cov = main_filled / len(main)
    merged_cov = merged_filled / len(merged) if len(merged) else 0.0
    if main_cov >= 0.5 and merged_cov < 0.2:
        return (
            f"source_databases coverage collapsed during merge: "
            f"{main_cov:.0%} of main rows had a source ({main_filled}/{len(main)}) "
            f"but only {merged_cov:.0%} of merged rows do "
            f"({merged_filled}/{len(merged)}). "
            "This usually means this project's local dedup.py predates "
            "upstream commit 780588c (\"dedup: read existing "
            "source_databases column when available\"). To fix, byte-copy "
            "the upstream dedup.py and re-run; see "
            "docs/BACKPORTING_NEW_SOURCES.md (\"Critical: dedup idempotence\")."
        )
    return None


def _autodetect_csv(directory: Path) -> Path:
    candidates = sorted(directory.glob("*_bibliography.csv"))
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(
        f"Could not auto-detect bibliography in {directory} "
        f"(found {len(candidates)} candidates: {candidates}). "
        "Pass an explicit path."
    )


def merge(
    main_csv: Path,
    gap_csv: Path,
    main_raw_dir: Path | None,
    gap_raw_dir: Path | None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Dedup-merge the gap-fill into the main bibliography.

    Args:
        main_csv: Path to the main bibliography CSV (will be overwritten).
        gap_csv: Path to the gap-fill bibliography CSV.
        main_raw_dir: Path to ``results/raw`` (gap files copied here).
        gap_raw_dir: Path to the gap-fill harvest's ``raw/``.
        dry_run: If True, log the planned changes without writing.

    Returns:
        ``(net_new_records, raw_files_copied)``.
    """
    main = pd.read_csv(main_csv, low_memory=False)
    gap = pd.read_csv(gap_csv, low_memory=False)
    logger.info("main: %d rows", len(main))
    logger.info("gap : %d rows", len(gap))

    only_main = set(main.columns) - set(gap.columns)
    only_gap = set(gap.columns) - set(main.columns)
    if only_main:
        logger.info("columns only in main (will be NaN for gap rows): %s",
                    sorted(only_main))
    if only_gap:
        logger.info("columns only in gap (will be appended): %s",
                    sorted(only_gap))

    union = pd.concat([main, gap], ignore_index=True)
    logger.info("union before dedup: %d", len(union))

    # Use the project's shared dedup (DOI + Jaccard title similarity).
    sys.path.insert(0, str(Path.cwd()))
    from dedup import dedup as dedup_df  # type: ignore[import-not-found]

    merged = dedup_df(union)
    logger.info("after dedup:        %d", len(merged))

    # Restore main column order with extras appended.
    ordered = [c for c in main.columns if c in merged.columns]
    ordered += [c for c in merged.columns if c not in ordered]
    merged = merged[ordered]

    net_new = len(merged) - len(main)
    logger.info("net new records:    %d", net_new)

    coverage_err = _check_source_coverage(main, merged)
    if coverage_err is not None:
        raise SystemExit(
            f"Aborting before overwriting {main_csv}: {coverage_err}"
        )

    if not dry_run:
        merged.to_csv(main_csv, index=False)
        logger.info("wrote %s (%d rows)", main_csv, len(merged))

    raw_copied = 0
    if gap_raw_dir is not None and main_raw_dir is not None and gap_raw_dir.exists():
        main_raw_dir.mkdir(parents=True, exist_ok=True)
        for f in gap_raw_dir.iterdir():
            if not f.is_file():
                continue
            dest = main_raw_dir / f.name
            if dest.exists():
                logger.debug("raw collision (keeping main): %s", f.name)
                continue
            if not dry_run:
                shutil.copy2(f, dest)
            raw_copied += 1
        logger.info("merged %d raw files from %s -> %s",
                    raw_copied, gap_raw_dir, main_raw_dir)

    return net_new, raw_copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gap-dir", type=Path, default=None,
        help="Gap-fill output directory (auto-detects gap CSV and "
             "<dir>/raw inside). Mutually exclusive with --gap-csv.",
    )
    parser.add_argument(
        "--main-csv", type=Path, default=None,
        help="Main bibliography CSV. Default: auto-detect "
             "results/*_bibliography.csv.",
    )
    parser.add_argument(
        "--gap-csv", type=Path, default=None,
        help="Gap-fill bibliography CSV. Required if --gap-dir is unset "
             "and the gap CSV cannot be inferred.",
    )
    parser.add_argument(
        "--main-raw-dir", type=Path, default=Path("results/raw"),
        help="Main raw-cache directory. Default: results/raw.",
    )
    parser.add_argument(
        "--gap-raw-dir", type=Path, default=None,
        help="Gap-fill raw-cache directory. Default: <gap-dir>/raw or "
             "(parent of gap-csv)/raw.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log the merge plan without writing anything.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.main_csv is None:
        args.main_csv = _autodetect_csv(Path("results"))
        logger.info("auto-detected main: %s", args.main_csv)

    if args.gap_csv is None:
        if args.gap_dir is None:
            raise SystemExit("Pass either --gap-dir or --gap-csv.")
        args.gap_csv = _autodetect_csv(args.gap_dir)
        logger.info("auto-detected gap : %s", args.gap_csv)

    if args.gap_raw_dir is None:
        args.gap_raw_dir = (args.gap_dir or args.gap_csv.parent) / "raw"

    net_new, raw_copied = merge(
        args.main_csv, args.gap_csv,
        args.main_raw_dir, args.gap_raw_dir,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        logger.info("DRY RUN: no files written. net_new=%d, raw_copied=%d",
                    net_new, raw_copied)
    else:
        logger.info(
            "Merge complete. Re-run scripts/embed_filter.py against %s "
            "to encode the new rows (cache will skip the rest).",
            args.main_csv,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
