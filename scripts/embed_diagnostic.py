"""Per-anchor coverage diagnostic for an embedded bibliography.

Reads ``results/<slug>_bibliography_embedded.csv`` (the output of
``scripts/embed_filter.py``), groups records by ``embed_top_anchor``,
and prints a coverage table:

- count of records where this anchor is the argmax
- median, p10, p90, max ``embed_score`` among those records
- top-3 record titles (max ``embed_score``) per anchor

This is the diagnostic to run after ``embed_filter.py`` finishes
and before paying for LLM labeling. If a theme that should account
for ~5-15% of cores is showing 0-1% of records, that's a signal to
split or rewrite the corresponding anchor in
``scripts/embed_filter.py``. Document each iteration in
``docs/ANCHOR_REVISIONS.md`` (template lives there).

Usage::

    # Default: read results/*_bibliography_embedded.csv (auto-detect),
    # use min-score 0.45, print to stdout
    python scripts/embed_diagnostic.py

    # Explicit input + write a markdown copy alongside the table
    python scripts/embed_diagnostic.py \\
        --csv results/my_lit_bibliography_embedded.csv \\
        --min-score 0.50 \\
        --top-k 5 \\
        --markdown docs/anchor_coverage_2026-04-29.md

The script auto-imports ``ANCHORS`` from ``scripts.embed_filter`` so
anchor labels stay in sync with the runtime descriptions used during
embedding. Per-anchor labels (one short gloss each) can be passed
via ``--labels-json`` for a more readable table; default is the
first ~60 chars of the anchor prose.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _autodetect_csv() -> Path:
    candidates = sorted(Path("results").glob("*_bibliography_embedded.csv"))
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(
        f"Could not auto-detect embedded bibliography in results/ "
        f"(found {len(candidates)} candidates: {candidates}). "
        "Pass --csv explicitly."
    )


def _load_anchors(scripts_dir: Path) -> list[str]:
    """Import ANCHORS from scripts/embed_filter.py without running its main()."""
    spec_path = scripts_dir / "embed_filter.py"
    spec = importlib.util.spec_from_file_location("embed_filter", spec_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {spec_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass can resolve cls.__module__
    # via sys.modules during evaluation (required on Python 3.13+).
    sys.modules["embed_filter"] = module
    spec.loader.exec_module(module)
    return list(module.ANCHORS)


def _short_label(anchor_prose: str, max_chars: int = 60) -> str:
    """First sentence (or first ~60 chars) of an anchor's prose."""
    first = anchor_prose.split(". ", 1)[0]
    if len(first) > max_chars:
        return first[: max_chars - 1] + "…"
    return first


def render_table(
    df: pd.DataFrame,
    anchors: list[str],
    min_score: float,
    top_k: int,
    labels: dict[int, str] | None = None,
) -> str:
    """Format the per-anchor coverage table as a Markdown string.

    Args:
        df: Embedded-bibliography DataFrame; must contain ``embed_score``,
            ``embed_top_anchor``, and ``title``.
        anchors: List of anchor prose strings (only ``len()`` is used; the
            actual prose is consumed only when ``labels`` is None).
        min_score: Threshold for the count column.
        top_k: How many top records to print per anchor.
        labels: Optional override for anchor labels keyed by anchor index.

    Returns:
        Multi-section Markdown report.
    """
    df = df.copy()
    df = df[df["embed_score"].notna()]
    df["embed_top_anchor"] = df["embed_top_anchor"].astype(int)
    above = df[df["embed_score"] >= min_score]

    if labels is None:
        labels = {i: _short_label(a) for i, a in enumerate(anchors)}

    out_lines: list[str] = []
    out_lines.append(f"# Anchor coverage diagnostic")
    out_lines.append(
        f"\nTotal records: **{len(df)}** "
        f"(embed_score range {df['embed_score'].min():.3f}-{df['embed_score'].max():.3f}, "
        f"median {df['embed_score'].median():.3f})"
    )
    out_lines.append(
        f"\nRecords ≥ min_score={min_score}: **{len(above)} / {len(df)} "
        f"({100*len(above)/max(len(df),1):.1f}%)**\n"
    )

    # Table header
    out_lines.append("| # | Anchor | N≥thresh | median | p10 | p90 | max |")
    out_lines.append("|---|---|--:|--:|--:|--:|--:|")
    for k in sorted(labels):
        sub = above[above["embed_top_anchor"] == k]["embed_score"]
        if len(sub) == 0:
            out_lines.append(f"| {k} | {labels[k]} | 0 | — | — | — | — |")
            continue
        out_lines.append(
            f"| {k} | {labels[k]} | {len(sub)} | "
            f"{sub.median():.3f} | {sub.quantile(0.1):.3f} | "
            f"{sub.quantile(0.9):.3f} | {sub.max():.3f} |"
        )

    # Per-anchor top-k
    out_lines.append(f"\n## Top {top_k} records per anchor (by max embed_score)")
    for k in sorted(labels):
        sub = above[above["embed_top_anchor"] == k].nlargest(top_k, "embed_score")
        out_lines.append(f"\n### [{k}] {labels[k]}")
        if sub.empty:
            out_lines.append("\n_(no records above threshold)_")
            continue
        for _, r in sub.iterrows():
            year = ""
            yr_val = r.get("year")
            if pd.notna(yr_val):
                try:
                    year = f" {int(yr_val)}"
                except (ValueError, TypeError):
                    year = ""
            title = str(r.get("title") or "").strip()[:90]
            out_lines.append(f"- `{r['embed_score']:.3f}`{year} — {title}")

    return "\n".join(out_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Embedded bibliography CSV. Default: auto-detect "
             "results/*_bibliography_embedded.csv.",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.45,
        help="Coverage threshold for the count column (default: 0.45).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Top-K records to print per anchor (default: 3).",
    )
    parser.add_argument(
        "--scripts-dir", type=Path, default=Path("scripts"),
        help="Directory containing embed_filter.py (default: scripts/).",
    )
    parser.add_argument(
        "--labels-json", type=Path, default=None,
        help="Optional JSON file mapping anchor index (string) to short "
             "label. Overrides the default first-sentence summary.",
    )
    parser.add_argument(
        "--markdown", type=Path, default=None,
        help="If set, also write the report as Markdown to this path.",
    )
    args = parser.parse_args(argv)

    if args.csv is None:
        args.csv = _autodetect_csv()
        print(f"# auto-detected: {args.csv}", file=sys.stderr)

    df = pd.read_csv(args.csv, low_memory=False)
    if "embed_score" not in df.columns or "embed_top_anchor" not in df.columns:
        raise SystemExit(
            "Input CSV missing embed_score / embed_top_anchor columns. "
            "Run scripts/embed_filter.py first."
        )

    anchors = _load_anchors(args.scripts_dir)

    labels: dict[int, str] | None = None
    if args.labels_json is not None:
        raw = json.loads(args.labels_json.read_text(encoding="utf-8"))
        labels = {int(k): str(v) for k, v in raw.items()}

    report = render_table(df, anchors, args.min_score, args.top_k, labels)
    print(report)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report + "\n", encoding="utf-8")
        print(f"\n# wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
