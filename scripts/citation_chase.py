"""Citation-graph chase via OpenAlex; merge new records into the bibliography.

For each seed paper, fetches:
- **Forward chase** — every paper that cites the seed (``filter=cites:W…``).
- **Backward chase** — every paper the seed references (from
  ``referenced_works`` on the seed's record, batch-fetched by id).

Seeds are the union of:
- 3 foundational works pinned by title search:
    * Krinsley & Doornkamp 1973  *Atlas of Quartz Sand Surface Textures*
    * Cailleux 1942 (morphoscopie pioneer)
    * Mahaney 2002  *Atlas of Sand Grain Surface Textures and Applications*
- the top-N highest-`embed_score` records labeled ``relevance_llm == "core"``
  by the Stanford pass.

Usage::

    python scripts/citation_chase.py \
        --labeled results/litsweep_bibliography_labeled.csv \
        --bib-csv results/litsweep_bibliography.csv \
        --top-n 20

Idempotent: dedups by id and DOI against the existing bibliography.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_clients  # noqa: E402
import dedup as dedup_mod  # noqa: E402
import litsweep_search as M  # noqa: E402

logger = logging.getLogger("citation_chase")

OPENALEX_BASE = "https://api.openalex.org/works"

# Foundational works for native-sand: saprolite / regolith / particle-size /
# crystalline-bedrock-weathering. The microtexture atlases were swapped out
# for the canonical regolith-and-saprolite literature.
FOUNDATIONAL_SEEDS: list[dict[str, str]] = [
    {
        "label": "Brimhall & Dietrich 1987 mass balance",
        "search": "Brimhall Dietrich constitutive mass balance chemical "
                  "composition volume porosity strain",
        "year_hint": "1987",
    },
    {
        "label": "Goldich 1938 weathering sequence",
        "search": "Goldich weathering granite minerals sequence",
        "year_hint": "1938",
    },
    {
        "label": "Anand Paine 2002 Yilgarn regolith atlas",
        "search": "Anand Paine regolith Yilgarn Craton Western Australia atlas",
        "year_hint": "2002",
    },
    {
        "label": "Riebe Kirchner Finkel cosmogenic regolith production",
        "search": "Riebe cosmogenic regolith production rate denudation granite",
        "year_hint": "2003",
    },
    {
        "label": "Velbel etch-pit weathering kinetics",
        "search": "Velbel etch pit weathering pyroxene amphibole feldspar",
        "year_hint": "1989",
    },
    {
        "label": "Heimsath Dietrich Nishiizumi soil production function",
        "search": "Heimsath Dietrich Nishiizumi soil production function "
                  "cosmogenic granite",
        "year_hint": "1997",
    },
    {
        "label": "Wilson 2004 weathering of primary rock-forming minerals",
        "search": "Wilson weathering primary rock-forming minerals processes "
                  "products rates",
        "year_hint": "2004",
    },
    {
        "label": "Berner Schott 1982 dissolution of pyroxenes amphiboles",
        "search": "Berner Schott dissolution pyroxenes amphiboles weathering",
        "year_hint": "1982",
    },
]


# ---------------------------------------------------------------------------
# Foundational seed lookup
# ---------------------------------------------------------------------------


def find_foundational_seed(spec: dict[str, str], cfg: api_clients.ClientConfig) -> dict | None:
    """Search OpenAlex for a foundational work; return the best match record."""
    params = {
        "search": spec["search"],
        "per_page": 25,
        "mailto": cfg.email,
    }
    resp = api_clients._request_with_retry("GET", OPENALEX_BASE, params=params)
    if resp is None or not resp.ok:
        logger.warning("foundational search failed: %s", spec["label"])
        return None
    results = resp.json().get("results") or []
    if not results:
        logger.warning("no results for: %s", spec["label"])
        return None
    # Prefer matches near the year hint, otherwise highest cited_by_count.
    year_hint = int(spec["year_hint"]) if spec.get("year_hint") else None

    def _score(r: dict) -> tuple[int, int]:
        year = r.get("publication_year") or 0
        year_proximity = -abs(year - year_hint) if year_hint else 0
        return (year_proximity, r.get("cited_by_count") or 0)

    best = max(results, key=_score)
    logger.info(
        "  %s → %s (%s, %d citations) %s",
        spec["label"], best.get("display_name", "")[:80],
        best.get("publication_year"),
        best.get("cited_by_count") or 0,
        best.get("id"),
    )
    return best


# ---------------------------------------------------------------------------
# Forward chase: papers that cite the seed
# ---------------------------------------------------------------------------


def _strip_id(work_id: str | None) -> str | None:
    if not work_id:
        return None
    return work_id.rsplit("/", 1)[-1]


def forward_chase(
    seed_ids: list[str], cfg: api_clients.ClientConfig, cap_per_seed: int = 500
) -> list[dict]:
    """For each seed, fetch papers citing it, paginated. Returns parsed records."""
    out: list[dict] = []
    seen_work_ids: set[str] = set()
    for sid in seed_ids:
        sid_short = _strip_id(sid)
        if not sid_short:
            continue
        cursor = "*"
        collected = 0
        while True:
            params = {
                "filter": f"cites:{sid_short}",
                "per_page": 200,
                "cursor": cursor,
                "mailto": cfg.email,
            }
            resp = api_clients._request_with_retry("GET", OPENALEX_BASE, params=params)
            if resp is None or not resp.ok:
                cfg.log_error("citation_chase_forward", sid_short,
                              f"status={getattr(resp, 'status_code', 'NA')}")
                break
            payload = resp.json()
            results = payload.get("results", []) or []
            for w in results:
                wid = w.get("id")
                if wid and wid not in seen_work_ids:
                    seen_work_ids.add(wid)
                    out.append(api_clients._openalex_record(w))
            collected += len(results)
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor or not results or collected >= cap_per_seed:
                break
            time.sleep(0.15)
        logger.info("  forward(%s): collected %d", sid_short, collected)
    return out


# ---------------------------------------------------------------------------
# Backward chase: works each seed cites
# ---------------------------------------------------------------------------


def backward_chase(
    seed_ids: list[str], cfg: api_clients.ClientConfig
) -> list[dict]:
    """Extract referenced_works from each seed and batch-fetch those records."""
    # First, fetch each seed's full record to read referenced_works.
    referenced_ids: set[str] = set()
    for sid in seed_ids:
        sid_short = _strip_id(sid)
        if not sid_short:
            continue
        url = f"{OPENALEX_BASE}/{sid_short}"
        resp = api_clients._request_with_retry(
            "GET", url, params={"mailto": cfg.email}
        )
        if resp is None or not resp.ok:
            cfg.log_error("citation_chase_backward_seed", sid_short,
                          f"status={getattr(resp, 'status_code', 'NA')}")
            continue
        payload = resp.json()
        refs = payload.get("referenced_works") or []
        for r in refs:
            short = _strip_id(r)
            if short:
                referenced_ids.add(short)
        logger.info("  backward(%s): %d references", sid_short, len(refs))
        time.sleep(0.15)

    if not referenced_ids:
        return []
    logger.info("  fetching %d unique referenced works", len(referenced_ids))

    # OpenAlex allows ids.openalex filter with |-separated IDs (up to ~100).
    out: list[dict] = []
    seen: set[str] = set()
    ref_list = list(referenced_ids)
    BATCH = 50
    for i in range(0, len(ref_list), BATCH):
        batch = ref_list[i : i + BATCH]
        params = {
            "filter": "openalex:" + "|".join(batch),
            "per_page": BATCH,
            "mailto": cfg.email,
        }
        resp = api_clients._request_with_retry("GET", OPENALEX_BASE, params=params)
        if resp is None or not resp.ok:
            cfg.log_error("citation_chase_backward_batch", str(i),
                          f"status={getattr(resp, 'status_code', 'NA')}")
            continue
        results = (resp.json() or {}).get("results", []) or []
        for w in results:
            wid = w.get("id")
            if wid and wid not in seen:
                seen.add(wid)
                out.append(api_clients._openalex_record(w))
        time.sleep(0.15)
    return out


# ---------------------------------------------------------------------------
# Auto seeds
# ---------------------------------------------------------------------------


def auto_seeds_from_labeled(labeled_csv: Path, top_n: int) -> list[dict]:
    """Pick the top-N highest-`embed_score` core records as seed metadata."""
    df = pd.read_csv(labeled_csv)
    if "relevance_llm" not in df.columns or "embed_score" not in df.columns:
        raise SystemExit(
            f"{labeled_csv} missing relevance_llm/embed_score; run labeling first."
        )
    core = df[df["relevance_llm"] == "core"]
    seeds = (core.sort_values("embed_score", ascending=False)
                 .head(top_n)
                 .to_dict("records"))
    logger.info("auto-selected %d core seeds", len(seeds))
    return seeds


# ---------------------------------------------------------------------------
# Merge into bibliography
# ---------------------------------------------------------------------------


def merge_into_bibliography(
    new_records: list[dict],
    bib_csv: Path,
    bib_bib: Path,
    source_tag: str,
) -> int:
    """Filter, dedup, augment, dedup-against-existing, append, write CSV+bib."""
    if not new_records:
        return 0
    filtered = M._filter_records(new_records)
    new_df = dedup_mod.dedup_iter(filtered)
    new_df = M._augment(new_df)
    if new_df.empty:
        return 0

    existing = pd.read_csv(bib_csv)
    existing_ids = set(existing["id"].astype(str))
    existing_dois = {
        dedup_mod.normalize_doi(d)
        for d in existing.get("doi", pd.Series(dtype="object")).tolist()
        if dedup_mod.normalize_doi(d)
    }

    def _truly_new(row: pd.Series) -> bool:
        rid = str(row.get("id"))
        if rid in existing_ids:
            return False
        rdoi = dedup_mod.normalize_doi(row.get("doi"))
        if rdoi and rdoi in existing_dois:
            return False
        return True

    truly_new = new_df[new_df.apply(_truly_new, axis=1)].copy()
    if truly_new.empty:
        return 0
    # Tag source
    truly_new["source_databases"] = source_tag

    # Align columns and concat
    for col in existing.columns:
        if col not in truly_new.columns:
            truly_new[col] = pd.NA
    for col in truly_new.columns:
        if col not in existing.columns:
            existing[col] = pd.NA
    truly_new = truly_new[existing.columns]

    combined = pd.concat([existing, truly_new], ignore_index=True)
    combined = combined.sort_values("priority_score", ascending=False, na_position="last")
    combined.to_csv(bib_csv, index=False)
    M.write_bibtex(combined, bib_bib)
    return len(truly_new)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path,
                        default=Path("results/litsweep_bibliography_labeled.csv"))
    parser.add_argument("--bib-csv", type=Path,
                        default=Path("results/litsweep_bibliography.csv"))
    parser.add_argument("--bib-bib", type=Path,
                        default=Path("results/litsweep_bibliography.bib"))
    parser.add_argument("--top-n", type=int, default=20,
                        help="Auto-select N highest-score core records as seeds.")
    parser.add_argument("--cap-per-seed", type=int, default=500,
                        help="Max forward-chase results per seed.")
    parser.add_argument("--email", default="ntlooker@gmail.com")
    parser.add_argument("--skip-foundational", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = api_clients.ClientConfig(
        email=args.email,
        raw_dir=args.bib_csv.parent / "raw",
        error_log=args.bib_csv.parent / "errors.log",
    )
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) Foundational seeds
    foundational_records: list[dict] = []
    if not args.skip_foundational:
        logger.info("looking up foundational seeds…")
        for spec in FOUNDATIONAL_SEEDS:
            rec = find_foundational_seed(spec, cfg)
            if rec is not None:
                foundational_records.append(rec)

    foundational_ids = [r["id"] for r in foundational_records if r.get("id")]

    # 2) Auto seeds from labeled core
    auto_records: list[dict] = []
    if args.top_n > 0:
        auto_records = auto_seeds_from_labeled(args.labeled, args.top_n)
    auto_ids = [str(r["id"]) for r in auto_records if r.get("id")]

    seed_ids = list({*foundational_ids, *auto_ids})
    logger.info("seed total: %d unique (%d foundational + %d auto, deduped)",
                len(seed_ids), len(foundational_ids), len(auto_ids))

    # 3) Forward chase
    forward_records: list[dict] = []
    if not args.skip_forward:
        logger.info("forward chase: papers citing each seed…")
        forward_records = forward_chase(seed_ids, cfg, cap_per_seed=args.cap_per_seed)
        logger.info("forward chase total: %d records", len(forward_records))

    # 4) Backward chase
    backward_records: list[dict] = []
    if not args.skip_backward:
        logger.info("backward chase: works each seed cites…")
        backward_records = backward_chase(seed_ids, cfg)
        logger.info("backward chase total: %d records", len(backward_records))

    # 5) Merge
    n_forward = merge_into_bibliography(
        forward_records, args.bib_csv, args.bib_bib, "openalex|citation_chase_forward"
    )
    logger.info("appended %d forward-chase records", n_forward)
    n_backward = merge_into_bibliography(
        backward_records, args.bib_csv, args.bib_bib, "openalex|citation_chase_backward"
    )
    logger.info("appended %d backward-chase records", n_backward)

    return 0


if __name__ == "__main__":
    sys.exit(main())
