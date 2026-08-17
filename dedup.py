"""Deduplication: DOI exact match + Jaccard title similarity."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import pandas as pd

_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have if in into is it of on or
    that the to was were will with within without via using based study
    studies analysis investigation review report paper article note
    le la les des du de du d et un une et e o de da das dos
    el los las y o en al der die das und des den von zu im
    o a os as e do
    """.split()
)


def normalize_doi(doi: str | None) -> str | None:
    """Lowercase, strip, remove leading 'https://doi.org/' or 'doi:' prefix.

    Args:
        doi: A raw DOI string or None. NaN floats from pandas are
            treated as None.

    Returns:
        Normalized DOI, or None if input is empty / None / NaN.
    """
    if doi is None:
        return None
    if isinstance(doi, float):  # pandas NaN
        return None
    if not isinstance(doi, str):
        doi = str(doi)
    if not doi:
        return None
    s = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s or None


_TOKEN_RE = re.compile(r"[a-z0-9À-ɏЀ-ӿ]+")


def _title_tokens(title: str | None) -> frozenset[str]:
    """Lowercase, strip stopwords, return token set for Jaccard comparison."""
    if title is None or isinstance(title, float):  # None or pandas NaN
        return frozenset()
    if not isinstance(title, str):
        title = str(title)
    if not title:
        return frozenset()
    tokens = _TOKEN_RE.findall(title.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 1)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def dedup(
    df: pd.DataFrame,
    title_threshold: float = 0.85,
    min_title_tokens: int = 6,
) -> pd.DataFrame:
    """Deduplicate a results DataFrame on DOI then title similarity.

    Three passes, in order, against each incoming record:

    1. **DOI exact match** (with prefix-stripping normalization).
    2. **Cross-DOI title near-match** — preprint ↔ journal version,
       versioned DOIs (``...v1`` vs ``...v2``), dual-publisher
       registrations of the same work. Bounded by ``min_title_tokens``
       and language bucketing to suppress false positives on generic
       short titles ("Glossary", "Reply on RC2", "Summary for
       Policymakers").
    3. **No-DOI title near-match** — same Jaccard pass, applied to
       records that lack a DOI, with the same min-token gate.

    The input DataFrame must contain at least: ``doi``, ``title``,
    ``source_database`` columns. Other columns are preserved by keeping
    the first record encountered (which is whichever source was queried
    first); ``source_databases`` (plural) is added with the union of
    sources for that work.

    Args:
        df: Combined results from all sources.
        title_threshold: Jaccard similarity above which two records are
            considered duplicates. Defaults to 0.85 per spec.
        min_title_tokens: Minimum stopword-stripped token count for a
            title to participate in title-similarity matching. Records
            below this are kept as-is even if their titles match
            verbatim — protects against generic-title false positives.
            Defaults to 6.

    Returns:
        A new DataFrame with duplicates merged.
    """
    if df.empty:
        out = df.copy()
        out["source_databases"] = []
        return out

    df = df.copy()
    # Use a list comprehension, NOT Series.map: .map skips NaN (na_action),
    # leaving the np.nan singleton in place. Because np.nan is truthy AND
    # `np.nan in {np.nan: ...}` matches by identity, every DOI-less record
    # (doi=None coerced to NaN once real DOIs share the column) would collapse
    # into a single "phantom-DOI" bucket. Calling normalize_doi on every value
    # coerces None/NaN/"" to a falsy None so the DOI-exact pass skips them.
    df["_norm_doi"] = [normalize_doi(x) for x in df["doi"].tolist()]

    # Track which output index each original row maps to.
    out_rows: list[dict] = []
    sources_per_row: list[set[str]] = []

    doi_index: dict[str, int] = {}
    # Bucket no-DOI titles by language to avoid cross-language Jaccard noise;
    # the spec calls cross-language matching tricky and asks us to flag rather
    # than auto-merge. Records with no language fall in the "" bucket.
    title_buckets: dict[str, list[tuple[frozenset[str], int]]] = defaultdict(list)

    def _row_sources(row: pd.Series) -> set[str]:
        """Source set for one row.

        Accepts either the per-source ``source_database`` column written
        by the API clients OR the union ``source_databases`` column
        produced by a previous ``dedup`` pass. This makes ``dedup``
        idempotent — re-running it on an already-merged frame (e.g. as
        part of ``merge_gap_fill``) preserves the prior union instead
        of wiping it.
        """
        plural = row.get("source_databases")
        if isinstance(plural, str) and plural:
            return {s for s in plural.split("|") if s}
        singular = row.get("source_database")
        if isinstance(singular, str) and singular:
            return {singular}
        return set()

    def _looks_corrupted_authors(s: object) -> bool:
        """Detect the CORE / OpenAlex bibliography-as-authors corruption.

        Trigger: > 25 semicolon-separated entries. Rare false positives
        on real consortium papers (~30+ authors) get re-promoted from a
        sibling source if one exists, otherwise stay as-is.
        """
        if not isinstance(s, str) or not s.strip():
            return True  # treat empty as "open to replacement"
        return len([a for a in s.split(";") if a.strip()]) > 25

    def _merge_into(kept_idx: int, row: pd.Series, sources: set[str]) -> None:
        """Fold ``row`` into the existing record at ``kept_idx``."""
        sources_per_row[kept_idx].update(sources)
        # Prefer the new row's authors if the kept row's
        # authors look corrupted but the new row's don't.
        kept_auth = out_rows[kept_idx].get("authors")
        new_auth = row.get("authors")
        if (_looks_corrupted_authors(kept_auth)
                and isinstance(new_auth, str) and new_auth.strip()
                and not _looks_corrupted_authors(new_auth)):
            out_rows[kept_idx]["authors"] = new_auth

    for _, row in df.iterrows():
        norm_doi = row["_norm_doi"]
        # iterrows re-coerces a stored None back to NaN. A bare NaN is truthy
        # AND `np.nan in {np.nan: ...}` matches by identity, so without this
        # guard every DOI-less record collapses into one phantom-DOI bucket.
        if pd.isna(norm_doi):
            norm_doi = None
        sources = _row_sources(row)

        # Pass 1: DOI exact match.
        if norm_doi and norm_doi in doi_index:
            _merge_into(doi_index[norm_doi], row, sources)
            continue

        # Pass 2 / 3: title-similarity match within a language bucket.
        # Runs whether or not the row has a DOI — catches preprint vs
        # journal-version pairs, versioned DOIs, and dual-publisher
        # dual-registrations of the same work that would otherwise slip
        # past the DOI-exact pass.
        lang_raw = row.get("language")
        if lang_raw is None or (isinstance(lang_raw, float) and pd.isna(lang_raw)):
            lang = ""
        else:
            lang = str(lang_raw).lower()
        toks = _title_tokens(row.get("title"))
        match_idx: int | None = None
        if len(toks) >= min_title_tokens:
            for cand_toks, cand_idx in title_buckets[lang]:
                if _jaccard(toks, cand_toks) >= title_threshold:
                    match_idx = cand_idx
                    break
        if match_idx is not None:
            _merge_into(match_idx, row, sources)
            # Register this row's DOI against the existing record so a
            # third record sharing this DOI also collapses correctly
            # (covers preprint + journal + crossref triples).
            if norm_doi and norm_doi not in doi_index:
                doi_index[norm_doi] = match_idx
            continue

        # New record: register in DOI index and the title bucket.
        idx = len(out_rows)
        out_rows.append(row.to_dict())
        sources_per_row.append(set(sources))
        if norm_doi:
            doi_index[norm_doi] = idx
        if len(toks) >= min_title_tokens:
            title_buckets[lang].append((toks, idx))

    out = pd.DataFrame(out_rows).drop(columns=["_norm_doi"], errors="ignore")
    out["source_databases"] = ["|".join(sorted(s)) for s in sources_per_row]
    if "source_database" in out.columns:
        out = out.drop(columns=["source_database"])
    return out


def dedup_iter(records: Iterable[dict], title_threshold: float = 0.85) -> pd.DataFrame:
    """Convenience wrapper: build a DataFrame from records and dedup it."""
    return dedup(pd.DataFrame(list(records)), title_threshold=title_threshold)
