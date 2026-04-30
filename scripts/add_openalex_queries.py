"""Run an additional set of OpenAlex queries and merge into the bibliography.

Surgical alternative to re-running the full pipeline. Reads the
existing ``results/native_sand_bibliography.csv``, runs only the
queries supplied via ``--query-group`` (a constant name from
``queries.py``), filters and augments the new records, drops anything
already in the bibliography by OpenAlex id or DOI, and concatenates
the residual onto the existing CSV. Re-running with the same group is
idempotent.

Usage::

    python scripts/add_openalex_queries.py \
        --query-group OPENALEX_GROUP_C_NON_ENGLISH \
        --queries-include "microtextura granos minerales MEB" \
        --queries-include "exoscopía granos arena minerales pesados" \
        --queries-include "morfoscopía granos detríticos arenas" \
        --queries-include "textura superficial granos feldespato circón granate"

The ``--queries-include`` flag may be passed multiple times; only
queries matching one of those strings (exact match) are run. If
omitted, the full named group is run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_clients  # noqa: E402
import dedup as dedup_mod  # noqa: E402
import native_sand_search as M  # noqa: E402
import queries as Q  # noqa: E402

logger = logging.getLogger("add_openalex_queries")

DEFAULT_BIB = Path("results/native_sand_bibliography.csv")
DEFAULT_BIB_BIB = Path("results/native_sand_bibliography.bib")


def _augment_new(records: list[dict]) -> pd.DataFrame:
    """Apply the main-pipeline augmentation to a fresh records list."""
    if not records:
        return pd.DataFrame()
    # Run dedup among the new records themselves first so source_databases
    # is populated correctly.
    df = dedup_mod.dedup_iter(records)
    return M._augment(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-group", default="OPENALEX_GROUP_C_NON_ENGLISH",
        help="Name of a list constant in queries.py whose entries to run.",
    )
    parser.add_argument(
        "--queries-include", action="append", default=[],
        help="Restrict to specific query strings (exact match). Repeatable.",
    )
    parser.add_argument("--bib-csv", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--bib-bib", type=Path, default=DEFAULT_BIB_BIB)
    parser.add_argument("--email", default="ntlooker@gmail.com")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not hasattr(Q, args.query_group):
        raise SystemExit(f"queries.py has no list named {args.query_group!r}")
    full_group: list[str] = getattr(Q, args.query_group)
    if args.queries_include:
        wanted = set(args.queries_include)
        run_queries = [q for q in full_group if q in wanted]
        missing = wanted - set(full_group)
        if missing:
            raise SystemExit(
                f"--queries-include strings not in {args.query_group}: {missing}"
            )
    else:
        run_queries = list(full_group)
    if not run_queries:
        raise SystemExit("no queries selected")

    logger.info("running %d query/queries:", len(run_queries))
    for q in run_queries:
        logger.info("  - %s", q)

    if not args.bib_csv.exists():
        raise SystemExit(f"existing bibliography not found: {args.bib_csv}")
    existing = pd.read_csv(args.bib_csv)
    logger.info("existing bibliography: %d rows", len(existing))

    cfg = api_clients.ClientConfig(
        email=args.email,
        raw_dir=args.bib_csv.parent / "raw",
        error_log=args.bib_csv.parent / "errors.log",
    )
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    new_raw = api_clients.search_openalex(run_queries, cfg)
    logger.info("new raw records: %d", len(new_raw))

    new_filtered = M._filter_records(new_raw)
    logger.info("after spec filters: %d", len(new_filtered))

    new_df = _augment_new(new_filtered)
    if new_df.empty:
        logger.info("no new records after augment — nothing to merge")
        return 0

    # Drop new records already present in the existing CSV by id or DOI.
    existing_ids = set(existing["id"].astype(str)) if "id" in existing.columns else set()
    existing_dois = {
        dedup_mod.normalize_doi(d)
        for d in existing.get("doi", pd.Series(dtype="object")).tolist()
        if dedup_mod.normalize_doi(d)
    }

    def _is_truly_new(row: pd.Series) -> bool:
        rid = str(row.get("id"))
        if rid in existing_ids:
            return False
        rdoi = dedup_mod.normalize_doi(row.get("doi"))
        if rdoi and rdoi in existing_dois:
            return False
        return True

    truly_new = new_df[new_df.apply(_is_truly_new, axis=1)].copy()
    logger.info("truly new (not already in bibliography): %d", len(truly_new))

    if truly_new.empty:
        logger.info("nothing new to add — bibliography already complete for this group")
        return 0

    # Align columns and concat.
    for col in existing.columns:
        if col not in truly_new.columns:
            truly_new[col] = pd.NA
    for col in truly_new.columns:
        if col not in existing.columns:
            existing[col] = pd.NA
    truly_new = truly_new[existing.columns]

    combined = pd.concat([existing, truly_new], ignore_index=True)
    combined = combined.sort_values("priority_score", ascending=False, na_position="last")

    args.bib_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.bib_csv, index=False)
    logger.info("wrote %s (%d rows, +%d new)",
                args.bib_csv, len(combined), len(truly_new))

    # Rebuild BibTeX from the combined CSV.
    M.write_bibtex(combined, args.bib_bib)
    logger.info("rewrote %s", args.bib_bib)

    if "language" in truly_new.columns:
        logger.info("\nnew records by language:\n%s",
                    truly_new["language"].fillna("(none)")
                    .value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
