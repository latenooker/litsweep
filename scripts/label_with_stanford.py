"""Label abstracts via a pluggable LLM backend (Stanford gateway or local Ollama).

Reads a CSV of records (typically the high-`embed_score` survivors from
``embed_filter.py``), sends ``title + abstract`` to a chat-completion
endpoint with a strict-JSON system prompt, and writes per-row labels.

Output columns appended to the input CSV (all suffixed ``_llm`` to
avoid colliding with screener-derived columns of the same name)::

    relevance_llm      "core" | "adjacent" | "off_topic"
    method_focus_llm   "microtexture_method" | "provenance" | "weathering" |
                       "diagenesis" | "ml_classification" | "other"
    minerals_llm       pipe-separated canonical mineral names
    transport_env_llm  "fluvial" | "aeolian" | "glacial" | "marine" |
                       "mixed" | "n/a"
    is_thesis_llm      bool
    is_review_llm      bool
    label_rationale    one short sentence

Checkpoints to ``--checkpoint-dir`` every ``--chunk-size`` rows so a
Ctrl-C costs at most one chunk. Re-running the same command resumes.

Backend: selected with ``--label-backend`` (default ``stanford``).
The ``stanford`` backend needs ``STANFORD_API_KEY`` and defaults to
model ``gemini-2.0-flash-lite-001`` at ``https://aiapi-prod.stanford.edu/v1``;
the ``ollama`` backend needs a local Ollama daemon with the chat model
pulled (e.g. ``ollama pull llama3.1``).

Usage::

    export STANFORD_API_KEY=sk-...
    python scripts/label_with_stanford.py \
        --csv results/native_sand_bibliography_embedded.csv \
        --out results/native_sand_bibliography_labeled.csv \
        --min-score 0.55

    # Test on 25 rows:
    python scripts/label_with_stanford.py \
        --csv results/native_sand_bibliography_embedded.csv \
        --out results/labels_smoke.csv --limit 25
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Make label_backends importable when running this script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import label_backends  # noqa: E402

logger = logging.getLogger("label_with_stanford")

STANFORD_DEFAULT_URL = "https://aiapi-prod.stanford.edu/v1"
STANFORD_DEFAULT_MODEL = "gemini-2.0-flash-lite-001"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

SYSTEM_PROMPT = """\
You are a soil-science / regolith-geology research librarian. Given the \
title and abstract of a scientific paper or thesis, classify it for a \
bibliography on PARTICLE-SIZE / COMMINUTION / SAND-FRACTION studies in \
soils and regolith developed in situ on igneous or metamorphic bedrock, \
saprolite, or glacial till derived from crystalline source rock.

In scope: saprolite / saprock / regolith on granite, gneiss, schist, \
amphibolite, gabbro, basalt, andesite, etc.; volcanic ash / Andisols; \
glacial till from crystalline source; weathering profiles, critical-zone \
science, fragmentation models, in-situ rock breakdown, pedogenic vs. \
lithogenic sand.

Out of scope: sedimentary-rock parent material, marine sediments, \
clastic sedimentary basins, aeolian or alluvial deposits (even of \
crystalline origin once they are reworked sediments).

Return ONLY a JSON object — no prose, no code fences. Schema:
{
  "relevance": "core" | "adjacent" | "off_topic",
      // "core"      = the work IS about particle size / comminution /
      //               sand-fraction processes in soils or regolith on
      //               igneous, metamorphic, or crystalline-derived
      //               glacial-till parent material.
      // "adjacent"  = mentions parent-material weathering or particle
      //               size in passing while focused on something else.
      // "off_topic" = unrelated (sedimentary-basin work, marine
      //               sediments, mineral processing, materials science,
      //               etc.)
  "parent_lithology": "granite" | "granitoid" | "gneiss" | "schist" |
                      "amphibolite" | "gabbro_diorite" |
                      "volcanic_extrusive" | "volcanic_pyroclastic" |
                      "ultramafic" | "quartzite" | "marble" |
                      "glacial_till" | "mixed_crystalline" |
                      "sedimentary" | "other" | "not_specified",
      // Use "sedimentary" when the paper IS about sedimentary parent
      // material — that signals an off_topic record, not an in-scope one.
  "process_focus": "chemical_weathering" | "physical_comminution" |
                   "biological_bioturbation" | "frost_periglacial" |
                   "mixed" | "other",
  "regolith_zone": "soil_profile" | "saprolite" | "saprock" |
                   "weathering_rind" | "till_or_drift" | "fresh_rock" |
                   "mixed" | "not_applicable",
  "methods": [<from: laser_diffraction, sieve, pipette, SEM, TEM, AFM,
               tomography, cosmogenic, geochem_major, geochem_REE,
               XRD, isotope, modeling, field_description, other>],
  "minerals": [<feldspar, plagioclase, k-feldspar, biotite, muscovite,
               amphibole, pyroxene, olivine, quartz, garnet, zircon,
               magnetite, ilmenite, rutile, kaolinite, clay>],
  "climate_zone": "tropical" | "temperate" | "boreal" | "polar_periglacial" |
                  "arid" | "mixed" | "not_applicable",
  "is_thesis": true | false,
  "is_review": true | false,
  "rationale": "<one short sentence (≤25 words) explaining the labels>"
}

Rules:
- If the abstract is missing/empty, label from the title alone and set
  rationale to "title-only".
- For glacial till, set parent_lithology to "glacial_till" and
  regolith_zone to "till_or_drift". The user explicitly INCLUDES till
  derived from crystalline source as relevant.
- For pure sedimentary-rock-parent papers, label relevance "off_topic"
  and parent_lithology "sedimentary" — do NOT label them "core".
- Lists may be empty if nothing is mentioned.
- Be strict about "core" vs "adjacent": only label "core" if particle
  size / comminution / sand-fraction on crystalline (or till) parent
  material is the PRIMARY topic, not just background.
"""


# ---------------------------------------------------------------------------
# Project-schema-tied error label
# ---------------------------------------------------------------------------


def _error_label(reason: str) -> dict[str, Any]:
    return {
        "relevance": "error",
        "parent_lithology": "error",
        "process_focus": "error",
        "regolith_zone": "error",
        "methods": [],
        "minerals": [],
        "climate_zone": "not_applicable",
        "is_thesis": False,
        "is_review": False,
        "rationale": reason[:200],
    }


# ---------------------------------------------------------------------------
# Row formatting + checkpoint resume
# ---------------------------------------------------------------------------


def _row_prompt(row: pd.Series) -> str:
    """Build the labeler prompt for one record.

    Prefers the full ``abstract`` column over ``abstract_snippet`` (the
    snippet is capped at 400 chars by the orchestrator). When both are
    populated, the snippet wins under the original logic and the LLM
    sees a truncated abstract — the back half of the abstract (methods,
    results, numbers) was being silently dropped from the labeler's
    context.
    """
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    if abstract in ("nan", ""):
        abstract = str(row.get("abstract_snippet") or "").strip()
    if abstract == "nan":
        abstract = ""
    minerals = str(row.get("minerals_mentioned") or "").strip()
    lang = str(row.get("language") or "").strip()
    parts = [f"TITLE: {title}"]
    if lang:
        parts.append(f"LANGUAGE: {lang}")
    if minerals:
        parts.append(f"MINERALS_DETECTED_BY_REGEX: {minerals}")
    parts.append(f"ABSTRACT: {abstract or '(none)'}")
    return "\n".join(parts)


def _load_checkpoints(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    if not checkpoint_dir.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for f in sorted(checkpoint_dir.glob("chunk_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df = df.rename(columns=_LEGACY_RENAME)
        for _, row in df.iterrows():
            by_id[str(row["id"])] = row.to_dict()
    return by_id


def _save_checkpoint(checkpoint_dir: Path, chunk_idx: int,
                     rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoint_dir / f"chunk_{chunk_idx:05d}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


LABEL_COLUMNS = (
    "relevance_llm", "parent_lithology_llm", "process_focus_llm",
    "regolith_zone_llm", "methods_llm", "minerals_llm",
    "climate_zone_llm", "is_thesis_llm", "is_review_llm",
    "label_rationale",
)

# All LLM outputs use the `_llm` suffix to avoid colliding with screener-
# derived columns like `relevance` or `lithologies_mentioned`.
_LEGACY_RENAME: dict[str, str] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True,
                        help="Input CSV (typically *_embedded.csv).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV with label columns appended.")
    parser.add_argument(
        "--min-score", type=float, default=0.45,
        help="Only label rows with embed_score >= this value. "
             "Default 0.45 (project policy): the 0.45–0.55 band recovered "
             "97 cores in microtexture and 595 in native-sand, so labeling "
             "from 0.45 is worth the spend. Pass --min-score 0 to label "
             "everything; pass higher for cheaper / coarser passes.",
    )
    parser.add_argument(
        "--max-score", type=float, default=None,
        help="If set, only label rows with embed_score < this value. "
             "Combine with --min-score for band-limited labeling.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows labeled (smoke-test).")
    parser.add_argument(
        "--label-backend", choices=sorted(label_backends.BACKENDS),
        default="stanford",
        help="Which label backend. stanford (default) needs "
             "STANFORD_API_KEY; ollama needs a local Ollama daemon "
             "with the chat model pulled.",
    )
    parser.add_argument("--api-key", default=os.environ.get("STANFORD_API_KEY", ""),
                        help="Stanford API key (stanford backend only).")
    parser.add_argument("--api-url", default=STANFORD_DEFAULT_URL,
                        help="Stanford base URL (stanford backend only).")
    parser.add_argument("--model", default=None,
                        help="Model id. Default per backend: "
                             f"stanford={STANFORD_DEFAULT_MODEL}, "
                             f"ollama={OLLAMA_DEFAULT_MODEL}.")
    parser.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST,
                        help="Ollama base URL (ollama backend only).")
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=None,
        help="Default: <out>.checkpoints/",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--min-interval-s", type=float, default=0.0,
        help="Pre-throttle: min seconds between successive gateway "
             "requests. Default 0 (reactive only). stanford backend only.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    backend_kwargs = {
        "api_key": args.api_key,
        "api_url": args.api_url,
        "host": args.ollama_host,
        "min_interval_s": args.min_interval_s,
    }
    if args.model:
        backend_kwargs["model"] = args.model
    elif args.label_backend == "ollama":
        backend_kwargs["model"] = OLLAMA_DEFAULT_MODEL
    # stanford's dataclass default already equals STANFORD_DEFAULT_MODEL.

    backend = label_backends.make_backend(args.label_backend, **backend_kwargs)
    backend.check()

    df = pd.read_csv(args.csv)
    logger.info("loaded %d rows from %s", len(df), args.csv)
    if args.min_score is not None and "embed_score" in df.columns:
        before = len(df)
        df = df[df["embed_score"] >= args.min_score].reset_index(drop=True)
        logger.info("min_score=%.3f → %d / %d rows", args.min_score, len(df), before)
    if args.max_score is not None and "embed_score" in df.columns:
        before = len(df)
        df = df[df["embed_score"] < args.max_score].reset_index(drop=True)
        logger.info("max_score=%.3f → %d / %d rows", args.max_score, len(df), before)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
        logger.info("limit=%d", len(df))

    if "id" not in df.columns:
        raise SystemExit("Input CSV must have an `id` column.")

    checkpoint_dir = args.checkpoint_dir or args.out.with_suffix(".checkpoints")
    cached = _load_checkpoints(checkpoint_dir)
    if cached:
        logger.info("resumed %d previously-labeled rows from %s",
                    len(cached), checkpoint_dir)

    chunk_idx = 1 + max(
        (int(f.stem.split("_")[1]) for f in checkpoint_dir.glob("chunk_*.csv")),
        default=0,
    )
    pending: list[dict[str, Any]] = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="label")
    for _, row in pbar:
        rid = str(row["id"])
        if rid in cached:
            continue
        prompt = _row_prompt(row)
        try:
            result = backend.call(prompt, SYSTEM_PROMPT)
        except label_backends.BackendError as exc:
            result = _error_label(exc.reason)
        record = {
            "id": rid,
            "relevance_llm": result.get("relevance"),
            "parent_lithology_llm": result.get("parent_lithology"),
            "process_focus_llm": result.get("process_focus"),
            "regolith_zone_llm": result.get("regolith_zone"),
            "methods_llm": "|".join(result.get("methods") or []),
            "minerals_llm": "|".join(result.get("minerals") or []),
            "climate_zone_llm": result.get("climate_zone"),
            "is_thesis_llm": bool(result.get("is_thesis")),
            "is_review_llm": bool(result.get("is_review")),
            "label_rationale": result.get("rationale"),
        }
        cached[rid] = record
        pending.append(record)
        if len(pending) >= args.chunk_size:
            _save_checkpoint(checkpoint_dir, chunk_idx, pending)
            chunk_idx += 1
            pending = []

    if pending:
        _save_checkpoint(checkpoint_dir, chunk_idx, pending)

    # Merge labels back onto the input frame.
    labels_df = pd.DataFrame(cached.values())
    if "id" not in labels_df.columns and not labels_df.empty:
        labels_df["id"] = list(cached.keys())
    merged = df.merge(labels_df, on="id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows, %d labeled)",
                args.out, len(merged),
                int(merged["relevance_llm"].notna().sum())
                if "relevance_llm" in merged.columns else 0)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        logger.info("removed checkpoints %s (corpus write succeeded)",
                    checkpoint_dir)

    if "relevance_llm" in merged.columns:
        logger.info("\nrelevance_llm breakdown:\n%s",
                    merged["relevance_llm"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
