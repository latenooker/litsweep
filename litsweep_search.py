"""CLI entry point — orchestrate searches, dedup, and write CSV + BibTeX.

Topic: particle-size / comminution / sand fraction in soils developed in
situ on igneous or metamorphic bedrock, saprolite, or glacial till
derived from crystalline sources. See README.md for full scope.

Usage::

    python litsweep_search.py --email nate@stanford.edu --output results/
    python litsweep_search.py --email nate@stanford.edu --sources openalex,hal
    python litsweep_search.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

import api_clients
import dedup as dedup_mod
import parent_lithologies as minerals_mod  # alias kept for code reuse
import queries as Q

logger = logging.getLogger("litsweep_search")

SOURCE_QUERIES: dict[str, list[str]] = {
    "openalex": Q.OPENALEX_ALL,
    "semantic_scholar": Q.SEMANTIC_SCHOLAR,
    "wos": Q.WOS_STARTER,
    "wos_expanded": Q.WOS_EXPANDED,
    "hal": Q.HAL,
    "theses_fr": Q.THESES_FR,
    "base": Q.BASE,
    "bdtd": Q.BDTD,
    "scielo": getattr(Q, "SCIELO", []),
    "europepmc": getattr(Q, "EUROPEPMC", []),
    "eartharxiv": getattr(Q, "EARTHARXIV", []),
    "crossref": getattr(Q, "CROSSREF", []),
    "core": getattr(Q, "CORE", []),
}

# Defaults exclude the Starter-tier WoS endpoint (wos) and BASE because both
# require API keys, and projects with WOS_EXPANDED_API_KEY get redundant
# results from `wos_expanded`. Keep them in SOURCE_QUERIES so projects can
# still opt in via --sources wos,base if desired.
DEFAULT_SOURCES: tuple[str, ...] = (
    "openalex", "semantic_scholar", "wos_expanded",
    "hal", "theses_fr", "bdtd", "scielo",
    "europepmc", "eartharxiv", "crossref", "core",
)


# ---------------------------------------------------------------------------
# Filtering, scoring, augmentation
# ---------------------------------------------------------------------------


_ENGLISH_LANG_TOKENS = {"en", "eng", "english"}


def _is_non_english(language: str | None) -> bool:
    if language is None:
        return False
    if isinstance(language, float) and pd.isna(language):
        return False
    return str(language).lower() not in _ENGLISH_LANG_TOKENS


_EXCLUDED_TYPES = {"paratext", "erratum", "editorial", "letter"}


def _passes_type_filter(rec_type: str | None) -> bool:
    if not rec_type:
        return True
    return rec_type.lower() not in _EXCLUDED_TYPES


def _has_blocked_substring(title: str | None) -> bool:
    if not title:
        return False
    low = title.lower()
    return any(sub in low for sub in minerals_mod.TITLE_EXCLUDE_SUBSTRINGS)


def _abstract_snippet(abstract: str | None, n: int = 400) -> str:
    if abstract is None:
        return ""
    if isinstance(abstract, float) and pd.isna(abstract):
        return ""
    s = str(abstract)
    if not s:
        return ""
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


# TODO: define a project-specific regex if you want a topic-relevance
# bump in priority_score (e.g. canonical terms like "saprolite" in a
# regolith-weathering search). Leave as None for a generic project.
_TITLE_TOPIC_RE: "re.Pattern[str] | None" = None


def _coerce_str(value: object) -> str:
    """Coerce a pandas-row cell (may be None or NaN-float) to a clean str."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _priority_score(row: dict) -> int:
    """Order records for citation_chase / chunked labeling.

    Topic-agnostic by default: scores +1 per VOCAB_AXES column that has
    at least one tag, +2 for theses/dissertations, +3 for non-English,
    +1 for an open_access_url, +1 for a low-citation record (under-cited
    work that surfaces here is often more interesting), and +1 if the
    optional ``_TITLE_TOPIC_RE`` matches the title.

    Customize this per project: weight whichever axes matter most.
    """
    score = 0
    if row.get("non_english"):
        score += 3
    rec_type = _coerce_str(row.get("type")).lower()
    if "thesis" in rec_type or "dissertation" in rec_type:
        score += 2
    # +1 for each vocab axis that fired (registry-driven).
    axes = getattr(minerals_mod, "VOCAB_AXES", None)
    if axes is not None:
        for axis in axes:
            tags = [
                t for t in _coerce_str(row.get(axis.column)).split("|") if t
            ]
            if tags:
                score += 1
    if _TITLE_TOPIC_RE is not None and _TITLE_TOPIC_RE.search(
        _coerce_str(row.get("title"))
    ):
        score += 1
    cited = row.get("cited_by_count")
    if isinstance(cited, (int, float)) and not pd.isna(cited) and cited < 10:
        score += 1
    if row.get("open_access_url"):
        score += 1
    return score


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-axis tag columns, non_english, abstract_snippet, priority_score.

    The tag columns are derived from ``vocab.VOCAB_AXES`` (one column per
    declared axis). The optional ``vocab.TOPIC_PRESENCE`` adds one
    boolean column for use as a hard ``--require-column`` gate before
    LLM labeling.
    """
    if df.empty:
        base_cols = [
            "title_english_translation",
            "non_english", "abstract_snippet", "priority_score",
            "year", "source_journal_or_publisher", "notes",
        ]
        # Vocab-axis columns (registry-driven).
        axes = getattr(minerals_mod, "VOCAB_AXES", None)
        if axes is not None:
            base_cols.extend(a.column for a in axes)
        presence = getattr(minerals_mod, "TOPIC_PRESENCE", None)
        if presence is not None:
            base_cols.append(presence[0])
        for col in base_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return df

    df = df.copy()
    df["non_english"] = df["language"].map(_is_non_english)

    def _haystack(row: pd.Series) -> str:
        return f"{row.get('title') or ''} {row.get('abstract') or ''}"

    # Vocab axis registry — one column per axis declared in vocab.py.
    # Adding a new axis is one line in vocab.py; nothing to edit here.
    axes = getattr(minerals_mod, "VOCAB_AXES", None)
    if axes is None:
        raise SystemExit(
            "vocab.py must declare VOCAB_AXES (a list of VocabAxis). "
            "See docs/DIRECTORY_STRUCTURE.md for the canonical layout."
        )
    for axis in axes:
        df[axis.column] = [
            "|".join(axis.find(_haystack(row)))
            for _, row in df.iterrows()
        ]

    # Optional topic-presence boolean (e.g. mentions_earthworm). The gate
    # column name and predicate live in the vocab module so adding/removing
    # the gate is a single-line change there.
    presence = getattr(minerals_mod, "TOPIC_PRESENCE", None)
    if presence is not None:
        col, fn = presence
        df[col] = [fn(_haystack(row)) for _, row in df.iterrows()]

    df["abstract_snippet"] = df["abstract"].map(_abstract_snippet)
    df["title_english_translation"] = ""
    df["notes"] = ""
    df["year"] = df["publication_year"]
    df["priority_score"] = [
        _priority_score(row.to_dict()) for _, row in df.iterrows()
    ]
    return df


def _filter_records(records: Iterable[dict]) -> list[dict]:
    """Apply spec post-filters: type, blocked-substring, quartz-only title."""
    kept: list[dict] = []
    for r in records:
        if not _passes_type_filter(r.get("type")):
            continue
        if _has_blocked_substring(r.get("title")):
            continue
        kept.append(r)
    return kept


# ---------------------------------------------------------------------------
# BibTeX export (no external bibtexparser dep needed for write)
# ---------------------------------------------------------------------------


def _bibtex_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", r"\{").replace("}", r"\}")


def _str_or_empty(value: object) -> str:
    """Coerce CSV cells (including pandas NaN floats) to a clean string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


def _bibtex_key(row: dict) -> str:
    authors_str = _str_or_empty(row.get("authors"))
    authors = authors_str.split(";")
    first_author = (authors[0] or "anon").strip().split(",")[0]
    first_author = re.sub(r"[^A-Za-z]+", "", first_author) or "anon"
    year = _str_or_empty(row.get("year")) or _str_or_empty(row.get("publication_year")) or "n.d."
    title = _str_or_empty(row.get("title"))
    first_word = re.sub(r"[^A-Za-z0-9]+", "", title.split(" ")[0])[:20] or "untitled"
    return f"{first_author}{year}{first_word}".lower()


def _format_bibtex_authors(s: str) -> str:
    """Convert our semicolon-delimited author string to BibTeX form.

    Upstream APIs are normalized to ``"Last, First; Last, First; ..."``
    (see ``api_clients.py``). BibTeX/BibLaTeX — and therefore Zotero —
    require authors to be joined by the literal token `` and ``; a
    semicolon is treated as part of a name, which is why Zotero ingests
    the whole list as a single mangled author.
    """
    parts = [p.strip() for p in s.split(";")]
    parts = [p for p in parts if p]
    return " and ".join(parts)


def _bibtex_entry(row: dict) -> str:
    rec_type = _str_or_empty(row.get("type")).lower()
    if "thesis" in rec_type or "dissertation" in rec_type:
        entry_type = "phdthesis"
    elif "book" in rec_type or "chapter" in rec_type:
        entry_type = "incollection"
    else:
        entry_type = "article"
    fields: list[tuple[str, str]] = []
    for key, src_key in (
        ("title", "title"),
        ("author", "authors"),
        ("year", "year"),
        ("journal", "source_journal_or_publisher"),
        ("doi", "doi"),
        ("url", "open_access_url"),
        ("language", "language"),
        ("note", "source_databases"),
    ):
        sval = _str_or_empty(row.get(src_key))
        if not sval:
            continue
        if key == "author":
            sval = _format_bibtex_authors(sval)
            if not sval:
                continue
        fields.append((key, _bibtex_escape(sval)))
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields)
    return f"@{entry_type}{{{_bibtex_key(row)},\n  {body}\n}}\n"


def write_bibtex(df: pd.DataFrame, path: Path) -> None:
    used: set[str] = set()
    with path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            entry = _bibtex_entry(row.to_dict())
            # Disambiguate keys.
            head_match = re.match(r"@\w+\{([^,]+),", entry)
            if head_match:
                key = head_match.group(1)
                base_key = key
                suffix = ord("a")
                while key in used:
                    key = f"{base_key}{chr(suffix)}"
                    suffix += 1
                used.add(key)
                entry = entry.replace(f"{{{base_key},", f"{{{key},", 1)
            fh.write(entry + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# Base columns that every project persists. The vocab axis columns
# (e.g. minerals_mentioned) and the optional topic-presence column
# are appended automatically at write time from VOCAB_AXES /
# TOPIC_PRESENCE in vocab.py — no need to edit this list when
# adding a vocab axis.
CSV_COLUMNS = [
    "id", "doi", "title", "title_english_translation", "authors", "year",
    "language", "source_journal_or_publisher", "type",
    "abstract", "abstract_snippet",
    "cited_by_count",
    "non_english", "open_access_url",
    "source_databases", "priority_score", "notes",
]


def _parse_sources(arg: str | None) -> list[str]:
    """Resolve --sources arg to a list of source keys.

    When no --sources is passed, falls back to DEFAULT_SOURCES. Logs a
    WARNING if SOURCE_QUERIES defines source keys that DEFAULT_SOURCES
    omits — without this warning, adding a new source to SOURCE_QUERIES
    silently has no effect on default harvests until the user
    remembers to update DEFAULT_SOURCES too.
    """
    missing_from_default = [
        k for k in SOURCE_QUERIES if k not in DEFAULT_SOURCES
    ]
    if missing_from_default and not arg:
        logger.warning(
            "SOURCE_QUERIES defines %s but DEFAULT_SOURCES omits them; "
            "they will be skipped this run. Add them to DEFAULT_SOURCES "
            "or pass --sources %s explicitly.",
            missing_from_default,
            ",".join(missing_from_default),
        )
    if not arg:
        return list(DEFAULT_SOURCES)
    sources = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCE_QUERIES]
    if unknown:
        raise SystemExit(f"Unknown source(s): {unknown}. "
                         f"Valid: {sorted(SOURCE_QUERIES)}")
    return sources


def _build_config(args: argparse.Namespace) -> api_clients.ClientConfig:
    output = Path(args.output)
    return api_clients.ClientConfig(
        email=args.email,
        raw_dir=output / "raw",
        error_log=output / "errors.log",
        semantic_scholar_key=os.environ.get("SEMANTIC_SCHOLAR_KEY"),
        wos_key=os.environ.get("WOS_API_KEY"),
        wos_expanded_key=os.environ.get("WOS_EXPANDED_API_KEY"),
        base_key=os.environ.get("BASE_API_KEY"),
        core_key=os.environ.get("CORE_API_KEY"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multilingual literature search for particle-size / "
                    "comminution studies in soils and regolith on igneous, "
                    "metamorphic, or glacial-till parent material."
    )
    parser.add_argument("--email", default="anonymous@example.com",
                        help="Contact email for OpenAlex polite pool.")
    parser.add_argument("--output", default="results",
                        help="Output directory (CSV, BibTeX, raw/).")
    parser.add_argument("--sources",
                        help="Comma-separated subset of sources. "
                             f"Default: {','.join(DEFAULT_SOURCES)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queries that would run, then exit.")
    parser.add_argument(
        "--doi-exclude", default="data/doi_exclude.txt",
        help="Path to a newline-delimited DOI list to drop after dedup. "
             "Records with a matching DOI are filtered out so the rest "
             "of the pipeline (embed + label) only sees novel content. "
             "Default: data/doi_exclude.txt — written by "
             "scripts/scaffold_new_search.py --from-existing-corpus, "
             "or hand-managed. Set to '' to disable.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sources = _parse_sources(args.sources)

    if args.dry_run:
        for src in sources:
            qs = SOURCE_QUERIES[src]
            print(f"\n=== {src} ({len(qs)} queries) ===")
            for q in qs:
                print(f"  {q}")
        return 0

    cfg = _build_config(args)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.error_log.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for src in sources:
        fn = api_clients.CLIENTS[src]
        qs = SOURCE_QUERIES[src]
        logger.info("Searching %s with %d queries…", src, len(qs))
        try:
            records = fn(qs, cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Source %s failed wholesale: %s", src, exc)
            cfg.log_error(src, "<wholesale>", str(exc))
            continue
        logger.info("  %s returned %d records", src, len(records))
        all_records.extend(records)

    logger.info("Total records before filtering: %d", len(all_records))
    kept = _filter_records(all_records)
    logger.info("After spec post-filters: %d", len(kept))

    deduped = dedup_mod.dedup_iter(kept)
    logger.info("After dedup: %d", len(deduped))

    # Cross-project DOI exclusion. Loads a newline-delimited file of
    # lower-cased DOIs already harvested in a sibling project (default
    # data/doi_exclude.txt, populated by
    # scripts/scaffold_new_search.py --from-existing-corpus or
    # hand-managed). Records with matching DOIs are dropped here so
    # downstream embed + label only see novel content.
    if args.doi_exclude:
        excl_path = Path(args.doi_exclude)
        if excl_path.exists():
            excl = {ln.strip().lower() for ln in excl_path.read_text().splitlines()
                    if ln.strip()}
            before = len(deduped)
            mask = ~deduped["doi"].fillna("").astype(str).str.strip().str.lower().isin(excl)
            deduped = deduped[mask].reset_index(drop=True)
            logger.info(
                "DOI exclude (%s): dropped %d records → %d remaining",
                excl_path, before - len(deduped), len(deduped),
            )
        else:
            logger.warning(
                "--doi-exclude %s not found; skipping cross-project filter",
                excl_path,
            )

    augmented = _augment(deduped)
    augmented = augmented.sort_values("priority_score", ascending=False)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "litsweep_bibliography.csv"
    bib_path = output / "litsweep_bibliography.bib"

    # Build the output column list. Start from the project's CSV_COLUMNS,
    # then auto-extend with any columns the vocab registry declared (so
    # adding a vocab axis doesn't require editing two places) and the
    # optional topic-presence column. Anything else in the DataFrame
    # but not in the final list gets logged as a drop warning so silent
    # column drift surfaces.
    extras: list[str] = []
    axes = getattr(minerals_mod, "VOCAB_AXES", None)
    if axes is not None:
        for axis in axes:
            if axis.column not in CSV_COLUMNS and axis.column not in extras:
                extras.append(axis.column)
    presence = getattr(minerals_mod, "TOPIC_PRESENCE", None)
    if presence is not None:
        gate_col = presence[0]
        if gate_col not in CSV_COLUMNS and gate_col not in extras:
            extras.append(gate_col)
    final_cols = [c for c in CSV_COLUMNS if c in augmented.columns]
    final_cols += [c for c in extras if c in augmented.columns and c not in final_cols]
    dropped = [c for c in augmented.columns if c not in final_cols]
    # Tolerate a small ignore list of well-known intermediates we
    # intentionally don't persist.
    _IGNORE_DROP = {"abstract", "publication_year", "raw"}
    surprising = [c for c in dropped if c not in _IGNORE_DROP]
    if surprising:
        logger.warning(
            "CSV write dropping %d columns not in CSV_COLUMNS or VOCAB_AXES: %s. "
            "Add to CSV_COLUMNS if you want them persisted.",
            len(surprising), surprising,
        )
    augmented[final_cols].to_csv(csv_path, index=False)
    write_bibtex(augmented, bib_path)
    logger.info("Wrote %s (%d rows) and %s", csv_path, len(augmented), bib_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
