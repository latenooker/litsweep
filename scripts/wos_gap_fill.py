"""WoS Expanded gap-fill harvest (one-shot, 2026-04-28).

Runs `queries.WOS_EXPANDED_GAP` against the WoS Expanded API and writes
records to `results/wos_gap_records_<DATE>.csv`. Reports how many are
new (DOI not in the existing bibliography) and how many overlap.

This is a side-channel run: it does not touch the main pipeline outputs
(bibliography.bib / labeled_corpus.csv). Records that look promising can
later be merged via the standard dedup flow.

Usage:
    WOS_EXPANDED_API_KEY=... python scripts/wos_gap_fill.py [--cap 200] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import api_clients  # noqa: E402
import queries as Q  # noqa: E402
from dedup import normalize_doi  # noqa: E402


def _existing_dois(bib_csv: Path) -> set[str]:
    """Collect normalised DOIs already in the project bibliography."""
    if not bib_csv.exists():
        return set()
    dois: set[str] = set()
    df = pd.read_csv(bib_csv, usecols=["doi"], dtype=str, low_memory=False)
    for d in df["doi"].dropna():
        n = normalize_doi(d)
        if n:
            dois.add(n)
    return dois


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=200,
                    help="Per-query record cap (default 200).")
    ap.add_argument("--bib", type=Path,
                    default=Path("results/native_sand_labeled_corpus.csv"),
                    help="Existing labeled-corpus CSV used to detect novelty.")
    ap.add_argument("--out", type=Path,
                    default=Path("results/wos_gap_records_2026-04-28.csv"))
    ap.add_argument("--raw-dir", type=Path,
                    default=Path("results/raw_wos_gap_2026-04-28"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print first 3 queries and quotas, do not harvest.")
    args = ap.parse_args()

    key = os.environ.get("WOS_EXPANDED_API_KEY")
    if not key:
        print("ERROR: WOS_EXPANDED_API_KEY not set", file=sys.stderr)
        return 2

    queries = Q.WOS_EXPANDED_GAP
    print(f"WOS_EXPANDED_GAP: {len(queries)} queries, per_query_cap={args.cap}")
    print(f"  bib: {args.bib}")
    print(f"  out: {args.out}")
    print(f"  raw: {args.raw_dir}")

    if args.dry_run:
        for i, q in enumerate(queries[:3], 1):
            print(f"  [{i}] {q[:140]}")
        return 0

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg = api_clients.ClientConfig(
        wos_expanded_key=key,
        per_query_cap=args.cap,
        raw_dir=args.raw_dir,
    )

    print(f"Loading existing bibliography for novelty check...")
    known = _existing_dois(args.bib)
    print(f"  {len(known):,} normalised DOIs already in corpus")

    records = api_clients.search_wos_expanded(queries, cfg)
    print(f"\nHarvested {len(records):,} raw records (post-pagination, pre-dedup).")

    # Stable per-WoS-UID dedup within this batch
    seen_ids: set[str] = set()
    rows: list[dict] = []
    new_doi_count = 0
    overlap_doi_count = 0
    no_doi_count = 0
    no_abstract_count = 0
    for r in records:
        uid = r.get("id") or ""
        if uid in seen_ids:
            continue
        seen_ids.add(uid)
        doi = normalize_doi(r.get("doi") or "")
        if not doi:
            no_doi_count += 1
            novelty = "no_doi"
        elif doi in known:
            overlap_doi_count += 1
            novelty = "in_corpus"
        else:
            new_doi_count += 1
            novelty = "new"
        if not (r.get("abstract") or "").strip():
            no_abstract_count += 1
        rows.append({
            "novelty": novelty,
            "id": uid,
            "doi": doi or "",
            "publication_year": r.get("publication_year"),
            "cited_by_count": r.get("cited_by_count"),
            "title": r.get("title") or "",
            "abstract": r.get("abstract") or "",
            "authors": r.get("authors") or "",
            "venue": r.get("source_journal_or_publisher") or "",
            "language": r.get("language") or "",
            "doctype": r.get("type") or "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nWrote {len(rows):,} unique-by-UID records to {args.out}")
    print(f"  new (DOI not in corpus):     {new_doi_count:,}")
    print(f"  overlapping (DOI in corpus): {overlap_doi_count:,}")
    print(f"  no DOI:                      {no_doi_count:,}")
    print(f"  missing abstract:            {no_abstract_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
