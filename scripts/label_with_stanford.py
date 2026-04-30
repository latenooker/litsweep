"""Label abstracts via the Stanford AI API gateway (OpenAI-compatible).

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

Backend: Stanford AI API gateway. Default base URL
``https://aiapi-prod.stanford.edu/v1``; default model
``gemini-2.0-flash-lite-001``. Set ``STANFORD_API_KEY`` env var.

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
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("label_with_stanford")

STANFORD_DEFAULT_URL = "https://aiapi-prod.stanford.edu/v1"
STANFORD_DEFAULT_MODEL = "gemini-2.0-flash-lite-001"
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3

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
# Stanford API call
# ---------------------------------------------------------------------------


@dataclass
class StanfordConfig:
    """Stanford gateway configuration."""

    api_key: str
    api_url: str = STANFORD_DEFAULT_URL
    model: str = STANFORD_DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 400


def check_stanford(cfg: StanfordConfig) -> None:
    """Probe the gateway. Logs available models; non-fatal if list fails."""
    if not cfg.api_key:
        raise SystemExit(
            "Stanford API key not set. Pass --api-key or set STANFORD_API_KEY."
        )
    url = f"{cfg.api_url.rstrip('/')}/models"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {cfg.api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        models = [m.get("id") for m in body.get("data", []) if m.get("id")]
        logger.info("Stanford reachable. %d models available.", len(models))
        if cfg.model not in models:
            logger.warning(
                "Model %r not in list; will attempt anyway. Sample: %s",
                cfg.model, models[:5],
            )
    except Exception as exc:  # pragma: no cover - probe is best-effort
        logger.warning("Could not list Stanford models (probe only): %s", exc)


def _strip_code_fence(text: str) -> str:
    """Drop ```json ... ``` fences if the model added them despite instructions."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


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


def call_stanford(prompt: str, cfg: StanfordConfig) -> dict[str, Any]:
    """Call Stanford gateway; return parsed JSON or an error label."""
    payload = json.dumps({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }).encode("utf-8")

    url = f"{cfg.api_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
    )

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                body = json.loads(resp.read())
            content = body["choices"][0]["message"]["content"]
            return json.loads(_strip_code_fence(content))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            if exc.code == 429:
                wait = min(2 ** (attempt + 2), 60)
                logger.warning("429 rate limit, sleeping %ds", wait)
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                logger.warning("HTTP %d (attempt %d): %s",
                               exc.code, attempt + 1, err_body[:200])
                time.sleep(2 ** attempt)
            else:
                return _error_label(f"HTTP {exc.code}: {err_body[:200]}")
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            if attempt < MAX_RETRIES - 1:
                logger.warning("parse error attempt %d: %s", attempt + 1, exc)
                time.sleep(1)
            else:
                return _error_label(f"parse: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            if attempt < MAX_RETRIES - 1:
                logger.warning("attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
            else:
                return _error_label(str(exc))
    return _error_label("max retries exhausted")


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
    parser.add_argument("--api-key", default=os.environ.get("STANFORD_API_KEY", ""))
    parser.add_argument("--api-url", default=STANFORD_DEFAULT_URL)
    parser.add_argument("--model", default=STANFORD_DEFAULT_MODEL)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=None,
        help="Default: <out>.checkpoints/",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = StanfordConfig(api_key=args.api_key, api_url=args.api_url, model=args.model)
    check_stanford(cfg)

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
        result = call_stanford(prompt, cfg)
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

    if "relevance_llm" in merged.columns:
        logger.info("\nrelevance_llm breakdown:\n%s",
                    merged["relevance_llm"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
