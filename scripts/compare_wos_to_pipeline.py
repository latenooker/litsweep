"""Compare the WoS screened export against the pipeline's CSV output.

Joins on normalized DOI; for records without a DOI on either side,
falls back to Jaccard title similarity (≥0.85). Produces three
artifacts in ``--out-dir``:

- ``overlap.csv``         — records present in both
- ``wos_only.csv``        — in the WoS export, missed by the pipeline
- ``pipeline_only.csv``   — found by the pipeline, not in the WoS export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dedup import _jaccard, _title_tokens, normalize_doi  # noqa: E402


def _load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_norm_doi"] = df.get("doi", pd.Series(dtype="object")).map(normalize_doi)
    df["_source"] = label
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wos",
        type=Path,
        default=Path("results/microtexture_wos_screened.csv"),
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("results/native_sand_bibliography.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/comparison"))
    parser.add_argument(
        "--title-threshold",
        type=float,
        default=0.85,
        help="Jaccard threshold for cross-matching title-only records.",
    )
    args = parser.parse_args(argv)

    if not args.wos.exists():
        raise SystemExit(f"WoS CSV not found: {args.wos}")
    if not args.pipeline.exists():
        raise SystemExit(f"Pipeline CSV not found: {args.pipeline}")

    wos = _load(args.wos, "wos")
    pipe = _load(args.pipeline, "pipeline")

    print(f"WoS rows: {len(wos)}")
    print(f"Pipeline rows: {len(pipe)}")

    # 1) DOI matches.
    wos_dois = {d for d in wos["_norm_doi"] if d}
    pipe_dois = {d for d in pipe["_norm_doi"] if d}
    doi_overlap = wos_dois & pipe_dois

    wos_no_doi = wos[wos["_norm_doi"].isna() | (wos["_norm_doi"] == "")]
    pipe_no_doi = pipe[pipe["_norm_doi"].isna() | (pipe["_norm_doi"] == "")]
    wos_only_by_doi = wos[
        ~wos["_norm_doi"].isin(doi_overlap) & wos["_norm_doi"].notna()
    ]
    pipe_only_by_doi = pipe[
        ~pipe["_norm_doi"].isin(doi_overlap) & pipe["_norm_doi"].notna()
    ]

    # 2) Title-similarity cross-match for the no-DOI residual + the
    #    "missing" sets (catches DOI-on-one-side-only cases).
    title_match_pairs: list[tuple[int, int, float]] = []
    wos_residual = pd.concat([wos_only_by_doi, wos_no_doi], ignore_index=False)
    pipe_residual = pd.concat([pipe_only_by_doi, pipe_no_doi], ignore_index=False)
    wos_tokens = [
        (idx, _title_tokens(t))
        for idx, t in wos_residual["title"].items()
    ]
    pipe_tokens = [
        (idx, _title_tokens(t))
        for idx, t in pipe_residual["title"].items()
    ]
    matched_wos: set = set()
    matched_pipe: set = set()
    for wi, wt in wos_tokens:
        if not wt:
            continue
        best: tuple[float, object] = (0.0, None)
        for pi, pt in pipe_tokens:
            if not pt or pi in matched_pipe:
                continue
            j = _jaccard(wt, pt)
            if j > best[0]:
                best = (j, pi)
        if best[0] >= args.title_threshold and best[1] is not None:
            title_match_pairs.append((wi, best[1], best[0]))
            matched_wos.add(wi)
            matched_pipe.add(best[1])

    # 3) Build output frames.
    wos_overlap_idx = wos.index[wos["_norm_doi"].isin(doi_overlap)].union(
        pd.Index(list(matched_wos))
    )
    pipe_overlap_idx = pipe.index[pipe["_norm_doi"].isin(doi_overlap)].union(
        pd.Index(list(matched_pipe))
    )
    overlap = pd.concat(
        [wos.loc[wos_overlap_idx], pipe.loc[pipe_overlap_idx]], ignore_index=True
    )
    wos_only = wos.loc[wos.index.difference(wos_overlap_idx)]
    pipe_only = pipe.loc[pipe.index.difference(pipe_overlap_idx)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlap.drop(columns=["_norm_doi"]).to_csv(args.out_dir / "overlap.csv", index=False)
    wos_only.drop(columns=["_norm_doi"]).to_csv(args.out_dir / "wos_only.csv", index=False)
    pipe_only.drop(columns=["_norm_doi"]).to_csv(args.out_dir / "pipeline_only.csv", index=False)

    print()
    print(f"DOI overlap: {len(doi_overlap)}")
    print(f"Title-similarity matches (no DOI on one side): {len(title_match_pairs)}")
    print(f"In both (union): {len(wos_overlap_idx)} WoS / {len(pipe_overlap_idx)} pipeline")
    print(f"WoS only:      {len(wos_only)}")
    print(f"Pipeline only: {len(pipe_only)}")

    if "relevance" in wos_only.columns:
        print("\nWoS-only relevance breakdown:")
        print(wos_only["relevance"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
