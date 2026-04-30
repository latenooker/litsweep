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


def dedup(df: pd.DataFrame, title_threshold: float = 0.85) -> pd.DataFrame:
    """Deduplicate a results DataFrame on DOI then title similarity.

    The input DataFrame must contain at least: ``doi``, ``title``,
    ``source_database`` columns. Other columns are preserved by keeping
    the first record encountered (which is whichever source was queried
    first); ``source_databases`` (plural) is added with the union of
    sources for that work.

    Args:
        df: Combined results from all sources.
        title_threshold: Jaccard similarity above which two no-DOI records
            are considered duplicates. Defaults to 0.85 per spec.

    Returns:
        A new DataFrame with duplicates merged.
    """
    if df.empty:
        out = df.copy()
        out["source_databases"] = []
        return out

    df = df.copy()
    df["_norm_doi"] = df["doi"].map(normalize_doi)

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

    for _, row in df.iterrows():
        norm_doi = row["_norm_doi"]
        sources = _row_sources(row)
        if norm_doi:
            if norm_doi in doi_index:
                sources_per_row[doi_index[norm_doi]].update(sources)
                continue
            doi_index[norm_doi] = len(out_rows)
            out_rows.append(row.to_dict())
            sources_per_row.append(set(sources))
            continue

        # No DOI — try title-similarity match within language bucket.
        lang_raw = row.get("language")
        if lang_raw is None or (isinstance(lang_raw, float) and pd.isna(lang_raw)):
            lang = ""
        else:
            lang = str(lang_raw).lower()
        toks = _title_tokens(row.get("title"))
        match_idx: int | None = None
        if toks:
            for cand_toks, cand_idx in title_buckets[lang]:
                if _jaccard(toks, cand_toks) >= title_threshold:
                    match_idx = cand_idx
                    break
        if match_idx is not None:
            sources_per_row[match_idx].update(sources)
            continue

        idx = len(out_rows)
        out_rows.append(row.to_dict())
        sources_per_row.append(set(sources))
        if toks:
            title_buckets[lang].append((toks, idx))

    out = pd.DataFrame(out_rows).drop(columns=["_norm_doi"], errors="ignore")
    out["source_databases"] = ["|".join(sorted(s)) for s in sources_per_row]
    if "source_database" in out.columns:
        out = out.drop(columns=["source_database"])
    return out


def dedup_iter(records: Iterable[dict], title_threshold: float = 0.85) -> pd.DataFrame:
    """Convenience wrapper: build a DataFrame from records and dedup it."""
    return dedup(pd.DataFrame(list(records)), title_threshold=title_threshold)
