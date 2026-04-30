"""Parse a Web of Science plain-text export, screen, and score it.

WoS Tagged format: 2-char tag at column 0, value follows after a space;
continuation lines start with 3 spaces. Records are delimited by ``ER``.
The schema this script expects is whatever was checked in the export
dialog; we coerce known tags and ignore the rest.

Usage::

    python scripts/screen_wos_export.py docs/microtexture_wos.txt \
        --out results/microtexture_wos_screened.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Allow imports from the project root regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import native_sand_search as M  # noqa: E402
import parent_lithologies as minerals_mod  # noqa: E402

# ---------------------------------------------------------------------------
# WoS tagged-format parser
# ---------------------------------------------------------------------------

# Multi-valued tags (one entry per continuation line).
_LIST_TAGS: frozenset[str] = frozenset({"AU", "AF", "C1", "RP", "EM", "OI", "RI"})

# Tags we keep. Anything else is dropped to keep the in-memory record small.
_KEPT_TAGS: frozenset[str] = frozenset({
    "PT", "AU", "AF", "TI", "SO", "AB", "DI", "PY", "PD",
    "TC", "UT", "VL", "IS", "BP", "EP", "AR", "DT", "LA",
})


def parse_wos(path: Path) -> list[dict[str, object]]:
    """Parse a WoS tagged-text file into a list of dict records.

    Args:
        path: Path to the .txt export.

    Returns:
        List of records keyed by 2-char WoS tags. Multi-valued tags
        (authors, addresses) become ``list[str]``; single-valued tags
        become ``str``. Unknown tags are dropped.
    """
    records: list[dict[str, object]] = []
    current: dict[str, object] = {}
    last_tag: str | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("ER"):
                if current:
                    records.append(current)
                current = {}
                last_tag = None
                continue
            # Skip the file header tags FN/VR/EF that aren't record-level.
            if line.startswith("EF"):
                continue
            tag = line[:2]
            if line[2:3] == " " and tag.isalnum() and tag.isupper():
                value = line[3:]
                last_tag = tag
                if tag not in _KEPT_TAGS:
                    last_tag = None
                    continue
                if tag in _LIST_TAGS:
                    current[tag] = [value]
                else:
                    current[tag] = value
            elif line.startswith("   ") and last_tag is not None:
                value = line[3:]
                if last_tag in _LIST_TAGS:
                    cur = current.get(last_tag)
                    if isinstance(cur, list):
                        cur.append(value)
                    else:
                        current[last_tag] = [str(cur or ""), value]
                else:
                    current[last_tag] = f"{current.get(last_tag, '')} {value}".strip()
            # else: header lines (FN/VR) or malformed; ignore.

    if current:
        records.append(current)
    return records


# ---------------------------------------------------------------------------
# Language inference from title characters (no LA tag in this export)
# ---------------------------------------------------------------------------

# Cyrillic, CJK only. Greek-letter detection is dropped because π/η/μ
# appear constantly in English chemistry/mineralogy titles.
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_CJK_RE = re.compile(r"[一-鿿ぁ-ヿ]")

# Distinctive function-word vocabularies (no overlap with English content
# words like "grains" or single-letter ambiguities like "à").
_FRENCH_DISTINCT = re.compile(
    r"\b(les|une|du|des|dans|sur|entre|avec|sont|est|étude|étude)\b",
    re.IGNORECASE,
)
_GERMAN_DISTINCT = re.compile(
    r"\b(der|die|das|und|für|bei|von|zur|eines|einer|durch|über)\b",
    re.IGNORECASE,
)
_PORTUGUESE_DISTINCT = re.compile(
    r"\b(da|do|dos|das|para|com|análise|estudo|grãos?)\b", re.IGNORECASE
)


def infer_language(title: str | None) -> str | None:
    """Best-effort language guess from a title.

    Returns "en" by default — WoS Core Collection is overwhelmingly
    English. Only flips to a non-English code when there is *strong*
    evidence: non-Latin script, or ≥2 distinct function words from a
    single non-English vocabulary.
    """
    if not title:
        return None
    if _CYRILLIC_RE.search(title):
        return "ru"
    if _CJK_RE.search(title):
        return "zh"
    fr_hits = len(set(m.group(0).lower() for m in _FRENCH_DISTINCT.finditer(title)))
    de_hits = len(set(m.group(0).lower() for m in _GERMAN_DISTINCT.finditer(title)))
    pt_hits = len(set(m.group(0).lower() for m in _PORTUGUESE_DISTINCT.finditer(title)))
    if de_hits >= 2:
        return "de"
    if pt_hits >= 2:
        return "pt"
    if fr_hits >= 2:
        return "fr"
    return "en"


# ---------------------------------------------------------------------------
# Normalize WoS records into the common pipeline schema
# ---------------------------------------------------------------------------


def _join(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def to_common_schema(rec: dict[str, object]) -> dict[str, object]:
    """Convert a parsed WoS record into the api_clients-style dict."""
    title = _join(rec.get("TI")) or None
    py = rec.get("PY")
    try:
        year = int(str(py)) if py else None
    except ValueError:
        year = None
    tc = rec.get("TC")
    try:
        cited = int(str(tc)) if tc not in (None, "") else None
    except ValueError:
        cited = None
    pt = (rec.get("PT") or "").strip() if isinstance(rec.get("PT"), str) else ""
    type_ = {"J": "article", "B": "book", "S": "series", "C": "proceedings"}.get(pt, pt or None)
    language = (rec.get("LA") if isinstance(rec.get("LA"), str) else None) or infer_language(title)
    return {
        "id": f"wos:{rec.get('UT')}",
        "doi": rec.get("DI"),
        "title": title,
        "publication_year": year,
        "language": language,
        "type": type_,
        "abstract": rec.get("AB"),
        "authors": _join(rec.get("AF") or rec.get("AU")),
        "source_journal_or_publisher": rec.get("SO"),
        "cited_by_count": cited,
        "open_access_url": None,
        "raw": {k: rec.get(k) for k in rec},
        "source_database": "wos_export",
    }


# ---------------------------------------------------------------------------
# Mineral relevance — for screening triage
# ---------------------------------------------------------------------------


def _relevance(row: dict) -> str:
    """Coarse relevance bucket. Drives the triage column.

    - "blocked"  → matches the title blocklist (concrete/cement/etc.)
    - "off_topic" → no minerals from the controlled vocabulary mentioned
      in title or abstract.
    - "quartz_only" → only quartz mentioned.
    - "non_quartz" → at least one non-quartz mineral mentioned.
    """
    title = row.get("title") or ""
    if any(sub in title.lower() for sub in minerals_mod.TITLE_EXCLUDE_SUBSTRINGS):
        return "blocked"
    haystack = f"{title} {row.get('abstract') or ''}"
    found = minerals_mod.find_minerals(haystack)
    if not found:
        return "off_topic"
    if found == ["quartz"]:
        return "quartz_only"
    return "non_quartz"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="WoS plain-text export.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/microtexture_wos_screened.csv"),
        help="Output CSV (default: results/microtexture_wos_screened.csv)",
    )
    args = parser.parse_args(argv)

    raw_records = parse_wos(args.path)
    print(f"parsed {len(raw_records)} WoS records from {args.path}")

    common = [to_common_schema(r) for r in raw_records]

    # Apply the same blocklist + type filter the main pipeline uses.
    kept = M._filter_records(common)
    print(f"after blocklist/type filter: {len(kept)}")

    df = pd.DataFrame(kept)
    df["relevance"] = [_relevance(r) for r in kept]
    aug = M._augment(df)
    aug = aug.sort_values("priority_score", ascending=False)

    out_cols = [
        "id", "doi", "title", "authors", "year", "language", "type",
        "source_journal_or_publisher", "abstract_snippet", "cited_by_count",
        "minerals_mentioned", "non_english", "relevance", "priority_score",
    ]
    out_cols = [c for c in out_cols if c in aug.columns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    aug[out_cols].to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(aug)} rows)")

    print("\nrelevance breakdown:")
    print(aug["relevance"].value_counts().to_string())
    print("\nlanguage breakdown:")
    print(aug["language"].fillna("unknown").value_counts().head(10).to_string())
    print("\ntop-15 by priority_score:")
    cols = ["priority_score", "year", "language", "minerals_mentioned", "title"]
    cols = [c for c in cols if c in aug.columns]
    print(aug.head(15)[cols].to_string(index=False, max_colwidth=80))

    return 0


if __name__ == "__main__":
    sys.exit(main())
