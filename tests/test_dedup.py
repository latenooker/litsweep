"""Unit tests for ``dedup.dedup``.

Covers the three passes in priority order:

1. DOI exact match (with normalization).
2. Cross-DOI title near-match — preprint vs journal version, versioned
   DOIs, dual-publisher registrations.
3. No-DOI title near-match (the original behavior).

Plus the safeguards: minimum-token gate to suppress false positives on
generic titles ("Glossary", "Reply on RC2"), language bucketing, and
``source_databases`` union.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dedup


def _df(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the canonical columns dedup expects."""
    cols = ["doi", "title", "authors", "language", "source_database"]
    rows = []
    for r in records:
        row = {c: r.get(c) for c in cols}
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def test_doi_exact_match_collapses_and_unions_sources():
    df = _df([
        {"doi": "10.1/abc", "title": "Foo", "source_database": "openalex"},
        {"doi": "10.1/abc", "title": "Foo", "source_database": "crossref"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 1
    assert set(out.iloc[0]["source_databases"].split("|")) == {"openalex", "crossref"}


def test_doi_normalization_strips_url_prefix():
    df = _df([
        {"doi": "https://doi.org/10.1/abc", "title": "Foo", "source_database": "a"},
        {"doi": "10.1/ABC", "title": "Foo", "source_database": "b"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 1


def test_cross_doi_title_near_match_collapses_preprint_and_journal():
    """Preprint DOI + journal DOI for the same paper should collapse.

    This is the bug the fix addresses: previously the second record
    bypassed the title pass because it had a DOI.
    """
    long_title = (
        "Reviews and syntheses Carbon vs cation based MRV of "
        "Enhanced Rock Weathering and the issue of soil organic carbon"
    )
    df = _df([
        {"doi": "10.5194/egusphere-2025-2740", "title": long_title,
         "language": "en", "source_database": "eartharxiv"},
        {"doi": "10.5194/bg-23-53-2026", "title": long_title,
         "language": "en", "source_database": "openalex"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 1, f"expected merged, got {len(out)} rows: {out['doi'].tolist()}"
    assert set(out.iloc[0]["source_databases"].split("|")) == {"eartharxiv", "openalex"}


def test_cross_doi_versioned_doi_collapses():
    """Same DOI stem with /v1, /v2 suffixes still produces near-twins."""
    long_title = (
        "Potential of temperate agroforestry systems to deliver "
        "ecosystem services an evidence map"
    )
    df = _df([
        {"doi": "10.5194/egusphere-2025-4619", "title": long_title,
         "language": "en", "source_database": "openalex"},
        {"doi": "10.5194/egusphere-2025-4619-v1", "title": long_title,
         "language": "en", "source_database": "eartharxiv"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 1


def test_short_generic_title_with_distinct_dois_does_not_merge():
    """Min-token gate prevents false positives like 'Glossary' or 'Reply on RC2'."""
    df = _df([
        {"doi": "10.1/x", "title": "Glossary",
         "language": "en", "source_database": "a"},
        {"doi": "10.2/y", "title": "Glossary",
         "language": "en", "source_database": "b"},
        {"doi": "10.3/z", "title": "Reply on RC2",
         "language": "en", "source_database": "a"},
        {"doi": "10.4/w", "title": "Reply on RC2",
         "language": "en", "source_database": "b"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 4, "short generic titles must not collapse"


def test_different_languages_do_not_match_on_title():
    """Language bucketing prevents cross-language title matches."""
    long_title = (
        "Microbial diversity of vermicompost bacteria that exhibit "
        "useful agricultural traits and waste management potential"
    )
    df = _df([
        {"doi": "10.1/en", "title": long_title, "language": "en",
         "source_database": "a"},
        {"doi": "10.2/es", "title": long_title, "language": "es",
         "source_database": "b"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 2


def test_no_doi_title_match_still_works():
    """Original no-DOI title-Jaccard behavior must be preserved."""
    long_title = (
        "Reviews and syntheses Carbon vs cation based MRV of "
        "Enhanced Rock Weathering and the issue of soil organic carbon"
    )
    df = _df([
        {"doi": None, "title": long_title, "language": "en",
         "source_database": "a"},
        {"doi": None, "title": long_title, "language": "en",
         "source_database": "b"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 1


def test_three_way_collapse_doi_then_title():
    """Three records: A and B share a DOI; C has a different DOI but same title.
    All three must collapse to one row with all three sources unioned.
    """
    long_title = (
        "Limited effect of organic matter addition on stabilised "
        "organic carbon in four tropical arable soils"
    )
    df = _df([
        {"doi": "10.5194/soil-12-187-2026", "title": long_title,
         "language": "en", "source_database": "openalex"},
        {"doi": "10.5194/soil-12-187-2026", "title": long_title,
         "language": "en", "source_database": "crossref"},
        {"doi": "10.5194/egusphere-2025-2287", "title": long_title,
         "language": "en", "source_database": "eartharxiv"},
    ])
    out = dedup.dedup(df)
    assert len(out) == 1
    sources = set(out.iloc[0]["source_databases"].split("|"))
    assert sources == {"openalex", "crossref", "eartharxiv"}


def test_idempotent_when_run_on_already_deduped_frame():
    """Re-running dedup on its own output (e.g. via merge_gap_fill) must
    preserve the prior source_databases union.
    """
    df = _df([
        {"doi": "10.1/abc", "title": "Foo", "source_database": "openalex"},
        {"doi": "10.1/abc", "title": "Foo", "source_database": "crossref"},
    ])
    once = dedup.dedup(df)
    twice = dedup.dedup(once)
    assert len(twice) == 1
    assert set(twice.iloc[0]["source_databases"].split("|")) == {"openalex", "crossref"}
