"""Embed pipeline records with local Ollama BGE-M3 and score by anchor similarity.

First-pass relevance filter for the literature corpus. Encodes
``title + abstract_snippet`` for each record, then scores against a
small set of anchor descriptions covering the relevant topical
neighborhoods (SEM microtexture, exoscopy / morphoscopy, heavy-mineral
provenance, mineral dissolution weathering, ML on mineral SEM).
Writes the input CSV back out with two added columns:

- ``embed_score``      — max cosine similarity over all anchors (∈ [-1, 1])
- ``embed_top_anchor`` — index of the anchor that produced the max

Embeddings are cached to a ``.npy`` keyed by row id so re-runs over the
same CSV skip the encoding step.

Usage::

    # Encode + score the pipeline output
    python scripts/embed_filter.py \
        --csv results/native_sand_bibliography.csv \
        --out results/native_sand_bibliography_embedded.csv

    # Encode + score the WoS export
    python scripts/embed_filter.py \
        --csv results/microtexture_wos_screened.csv \
        --out results/microtexture_wos_screened_embedded.csv

Prerequisites: ``ollama serve`` running locally and ``ollama pull bge-m3``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make embed_backends importable when running this script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embed_backends  # noqa: E402

logger = logging.getLogger("embed_filter")

# ---------------------------------------------------------------------------
# Anchor descriptions (multilingual-friendly: BGE-M3 is the model)
# ---------------------------------------------------------------------------

ANCHORS: list[str] = [
    # 0 — Saprolite / weathering profile particle size
    "Saprolite and weathering profile particle-size distributions on granite, "
    "gneiss, schist, or other crystalline parent material; in-situ regolith "
    "production rates and grain-size evolution down a profile.",

    # 1 — Granitic arena / grus / arenization (Romance + German vocab)
    "Arène granitique, grus, arenization and arénisation: granular "
    "disintegration of granite to coarse sand by isovolumetric weathering, "
    "and the resulting residual soils on crystalline bedrock.",

    # 2 — Spheroidal weathering / corestones / rock-to-soil transition
    "Spheroidal weathering, corestone formation, exfoliation, weathering rinds, "
    "and the saprock-to-saprolite transition in granite, gneiss, and other "
    "crystalline rocks.",

    # 3 — Critical-zone processes and regolith production
    "Critical-zone weathering, regolith production rate, cosmogenic nuclide "
    "denudation budgets, fragmentation models, and the conversion of "
    "crystalline bedrock to soil cover.",

    # 4 — Volcanic ash / Andisols (igneous-source pyroclastic regolith)
    "Volcanic ash and tephra weathering into Andisols and Andosols; "
    "particle-size distributions and pedogenic comminution of pyroclastic "
    "parent materials.",

    # 5 — Glacial till from crystalline source
    "Glacial till and ground moraine derived from granite, gneiss, or other "
    "crystalline source rock; particle-size distributions and weathering "
    "of clast-rich glacial deposits.",

    # 6 — Mineral release: feldspar, mica, biotite weathering in regolith
    "Feldspar dissolution, biotite expansion, mica weathering, and primary "
    "mineral release from weathered crystalline rock; sand-fraction "
    "evolution as a function of weathering intensity.",

    # 7 — Multilingual umbrella: Slavic + German + Romance regolith terms
    "Saprolit wietrzenie skały krystalicznej; сапролит выветривание гранита; "
    "Saprolith Verwitterung Granit Korngröße; saprólito intemperismo granito "
    "tamanho partícula; saprolita meteorización granito tamaño partícula.",
]


# ---------------------------------------------------------------------------
# CSV plumbing
# ---------------------------------------------------------------------------


def _row_text(row: pd.Series) -> str:
    """Join title + abstract_snippet (or abstract). Empty if both missing."""
    parts: list[str] = []
    for col in ("title", "abstract_snippet", "abstract"):
        v = row.get(col)
        if isinstance(v, str) and v and v != "nan":
            parts.append(v)
            if col != "title":  # only need one of abstract/abstract_snippet
                break
    return " ".join(parts).strip()


def _cache_paths(out_csv: Path) -> tuple[Path, Path]:
    """Sibling paths for the embedding matrix and its row-id index."""
    return (
        out_csv.with_suffix(".embeddings.npy"),
        out_csv.with_suffix(".embeddings.ids.txt"),
    )


def _load_cache(cache_npy: Path, cache_ids: Path) -> tuple[dict[str, np.ndarray], int]:
    if not (cache_npy.exists() and cache_ids.exists()):
        return {}, 0
    matrix = np.load(cache_npy)
    ids = cache_ids.read_text(encoding="utf-8").splitlines()
    if matrix.shape[0] != len(ids):
        logger.warning("cache shape mismatch — discarding")
        return {}, 0
    return {rid: matrix[i] for i, rid in enumerate(ids)}, matrix.shape[1]


def _save_cache(cache_npy: Path, cache_ids: Path,
                ids: list[str], matrix: np.ndarray) -> None:
    cache_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, matrix)
    cache_ids.write_text("\n".join(ids), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True,
                        help="Input CSV (must have an `id` column and `title`).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV with embed_score + embed_top_anchor.")
    parser.add_argument(
        "--embed-backend", choices=sorted(embed_backends.BACKENDS),
        default="ollama",
        help="Embedding backend. Only 'ollama' today; flag exists so "
             "future Voyage/Jina/OpenAI-compatible backends slot in "
             "without breaking the CLI.",
    )
    parser.add_argument("--model", default="bge-m3", help="Ollama model tag.")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-encode all rows even if a cache exists.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    backend = embed_backends.make_backend(
        args.embed_backend,
        model=args.model, host=args.host, batch_size=args.batch_size,
    )
    backend.check()

    df = pd.read_csv(args.csv)
    if "id" not in df.columns:
        raise SystemExit(f"CSV missing required `id` column: {args.csv}")
    logger.info("loaded %d rows from %s", len(df), args.csv)

    df["_row_text"] = df.apply(_row_text, axis=1)
    has_text = df["_row_text"].str.len() > 0
    logger.info("rows with text: %d / %d", int(has_text.sum()), len(df))

    cache_npy, cache_ids_path = _cache_paths(args.out)
    cache: dict[str, np.ndarray] = {}
    if not args.no_cache:
        cache, dim = _load_cache(cache_npy, cache_ids_path)
        if cache:
            logger.info("loaded %d cached embeddings (dim=%d)", len(cache), dim)

    # Figure out which rows need encoding.
    to_encode_idx: list[int] = []
    to_encode_texts: list[str] = []
    for i, row in df.iterrows():
        rid = str(row["id"])
        text = row["_row_text"]
        if not text:
            continue
        if rid in cache:
            continue
        to_encode_idx.append(i)
        to_encode_texts.append(text)

    if to_encode_texts:
        logger.info("encoding %d new rows", len(to_encode_texts))
        # Encode in chunks and checkpoint after each so a mid-run kill
        # loses at most CHECKPOINT_EVERY rows of work; resume picks up
        # cached embeddings by id on the next invocation.
        CHECKPOINT_EVERY = 1000
        n = len(to_encode_texts)
        for chunk_start in range(0, n, CHECKPOINT_EVERY):
            chunk_end = min(chunk_start + CHECKPOINT_EVERY, n)
            chunk_mat = backend.embed(
                to_encode_texts[chunk_start:chunk_end]
            )
            for offset, df_idx in enumerate(
                to_encode_idx[chunk_start:chunk_end]
            ):
                cache[str(df.at[df_idx, "id"])] = chunk_mat[offset]
            partial_ids = list(cache.keys())
            partial_mat = np.vstack([cache[rid] for rid in partial_ids])
            _save_cache(cache_npy, cache_ids_path, partial_ids, partial_mat)
            logger.info(
                "checkpoint %d/%d → %s", chunk_end, n, cache_npy
            )
    else:
        logger.info("all rows already cached — skipping encode")

    # Encode anchors fresh every run (cheap, and content may change).
    logger.info("encoding %d anchors", len(ANCHORS))
    anchor_mat = backend.embed(ANCHORS)

    # Score: max cosine over anchors. Both already L2-normed → dot is cosine.
    scores = np.full(len(df), np.nan, dtype=np.float32)
    top_anchor = np.full(len(df), -1, dtype=np.int32)
    for i, row in df.iterrows():
        rid = str(row["id"])
        emb = cache.get(rid)
        if emb is None:
            continue
        sims = anchor_mat @ emb
        j = int(np.argmax(sims))
        scores[i] = float(sims[j])
        top_anchor[i] = j

    df["embed_score"] = scores
    df["embed_top_anchor"] = top_anchor
    df = df.drop(columns=["_row_text"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Persist cache aligned with the order of df.id (so subsequent runs that
    # add rows can extend without reordering existing entries).
    ordered_ids = [str(rid) for rid in df["id"].tolist() if str(rid) in cache]
    if ordered_ids:
        ordered_mat = np.vstack([cache[rid] for rid in ordered_ids])
        _save_cache(cache_npy, cache_ids_path, ordered_ids, ordered_mat)
        logger.info("cached %d embeddings to %s", len(ordered_ids), cache_npy)

    finite = ~np.isnan(scores)
    if finite.any():
        s = scores[finite]
        logger.info(
            "score stats: n=%d  min=%.3f  median=%.3f  p90=%.3f  max=%.3f",
            int(finite.sum()), float(s.min()), float(np.median(s)),
            float(np.quantile(s, 0.9)), float(s.max()),
        )
        # Suggest a reasonable cutoff: tag rows above the 75th percentile.
        cutoff = float(np.quantile(s, 0.75))
        keep = int((s >= cutoff).sum())
        logger.info("p75 cutoff = %.3f → %d rows ≥ cutoff", cutoff, keep)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
