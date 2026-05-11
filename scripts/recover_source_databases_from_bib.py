"""Recover the ``source_databases`` column from the project's .bib file.

When `merge_gap_fill.py` runs against a project whose local ``dedup.py``
predates upstream commit ``780588c``, the buggy dedup zeroes out the
plural ``source_databases`` column in every CSV it writes (see
``docs/BACKPORTING_NEW_SOURCES.md``: "Critical: dedup idempotence").

The .bib write happens after dedup but uses the same in-memory frame,
so freshly-written .bib files still carry per-entry source provenance
in their ``note = {…}`` field. **However**, in the worm-tea-lit
incident the .bib was *not* rewritten by the bad merge — the original
pre-merge .bib remained on disk — so this script's recovery is bounded
by whichever .bib was most recently written.

This script:

1. Backs up each target CSV to ``<name>.pre_source_recovery.bak`` (only
   if no backup exists yet — re-running is safe).
2. Parses the .bib for ``doi`` and ``note`` per entry.
3. For each row with an empty ``source_databases``, fills it from the
   normalized-DOI lookup. Rows missing from the .bib stay empty and
   are reported.

Usage::

    # Auto-detect: results/<slug>_bibliography.bib +
    # results/<slug>_bibliography{,_embedded,_labeled_corpus}.csv
    python scripts/recover_source_databases_from_bib.py

    # Explicit
    python scripts/recover_source_databases_from_bib.py \\
        --bib results/foo_bibliography.bib \\
        --csv results/foo_bibliography.csv \\
        --csv results/foo_bibliography_embedded.csv

    # Preview only
    python scripts/recover_source_databases_from_bib.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("recover_source_databases")

_ENTRY_HEAD = re.compile(r"^@\w+\{([^,]+),")
_FIELD = re.compile(r"^\s*(\w+)\s*=\s*\{(.*)\}\s*,?\s*$")


def _normalize_doi(doi: str | None) -> str | None:
    if not isinstance(doi, str) or not doi.strip():
        return None
    s = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s or None


def parse_bib(path: Path) -> dict[str, str]:
    """Parse a litsweep-style .bib into a {normalized_doi: note} map."""
    out: dict[str, str] = {}
    cur: dict[str, str] = {}
    total = skipped = 0

    def _flush() -> None:
        nonlocal total, skipped
        if not cur:
            return
        total += 1
        doi = _normalize_doi(cur.get("doi"))
        note = cur.get("note", "")
        if doi and note:
            out[doi] = note
        else:
            skipped += 1

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if _ENTRY_HEAD.match(line):
                _flush()
                cur = {}
                continue
            m = _FIELD.match(line)
            if m:
                cur[m.group(1)] = m.group(2)
        _flush()

    logger.info(
        "parsed %s: %d entries, %d usable (DOI+note), %d skipped",
        path.name, total, len(out), skipped,
    )
    return out


def update_csv(
    csv: Path,
    doi_to_src: dict[str, str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Fill ``source_databases`` from the .bib map. Returns (filled, still_empty)."""
    df = pd.read_csv(csv, low_memory=False)
    n = len(df)
    if "source_databases" not in df.columns:
        logger.warning("%s: no source_databases column; skipping", csv.name)
        return 0, 0

    cur_empty = df["source_databases"].isna() | (df["source_databases"] == "")
    norm = df["doi"].map(_normalize_doi)
    new_src = norm.map(doi_to_src)
    fill_mask = cur_empty & new_src.notna()
    filled = int(fill_mask.sum())

    if not dry_run and filled:
        backup = csv.with_suffix(csv.suffix + ".pre_source_recovery.bak")
        if not backup.exists():
            shutil.copy2(csv, backup)
            logger.info("backed up %s -> %s", csv.name, backup.name)
        # Cast column to object to silence pandas dtype-promotion warning.
        df["source_databases"] = df["source_databases"].astype("object")
        df.loc[fill_mask, "source_databases"] = new_src[fill_mask]
        df.to_csv(csv, index=False)

    after_empty = df["source_databases"].isna() | (df["source_databases"] == "")
    still_empty = int(after_empty.sum())
    logger.info(
        "%s: rows=%d, filled=%d, still_empty=%d%s",
        csv.name, n, filled, still_empty, " (dry-run)" if dry_run else "",
    )
    return filled, still_empty


def _autodetect(results_dir: Path) -> tuple[Path, list[Path]]:
    bibs = sorted(results_dir.glob("*_bibliography.bib"))
    if len(bibs) != 1:
        raise SystemExit(
            f"Could not auto-detect .bib in {results_dir} "
            f"(found {len(bibs)}). Pass --bib explicitly."
        )
    bib = bibs[0]
    slug = bib.stem.removesuffix("_bibliography")
    candidates = [
        results_dir / f"{slug}_bibliography.csv",
        results_dir / f"{slug}_bibliography_embedded.csv",
        results_dir / f"{slug}_labeled_corpus.csv",
    ]
    csvs = [c for c in candidates if c.exists()]
    if not csvs:
        raise SystemExit(
            f"No CSVs found matching {slug}_*.csv in {results_dir}"
        )
    return bib, csvs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", type=Path, default=None,
                        help="Path to .bib (default: auto-detect in results/).")
    parser.add_argument("--csv", type=Path, action="append", default=None,
                        help="CSV to update; pass multiple times. "
                             "Default: auto-detect bibliography/embedded/"
                             "labeled_corpus.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Auto-detect root. Default: results/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.bib is None or args.csv is None:
        auto_bib, auto_csvs = _autodetect(args.results_dir)
        bib = args.bib or auto_bib
        csvs = args.csv or auto_csvs
    else:
        bib, csvs = args.bib, args.csv

    if not bib.exists():
        raise SystemExit(f"missing {bib}")

    doi_to_src = parse_bib(bib)
    if not doi_to_src:
        raise SystemExit(f"no usable entries in {bib}")

    total_filled = 0
    for csv in csvs:
        if not csv.exists():
            logger.warning("skip missing CSV: %s", csv)
            continue
        filled, _ = update_csv(csv, doi_to_src, dry_run=args.dry_run)
        total_filled += filled

    logger.info("total cells filled: %d%s",
                total_filled, " (dry-run)" if args.dry_run else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
