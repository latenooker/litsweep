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
import os
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
    """Search OpenAlex for a foundational work; return the best match record.

    With a ``"title"`` key, tries an exact ``title.search`` filter first:
    the keyword ``search`` + year-proximity fallback below can silently
    pick an unrelated paper from the same year (it did for 5/10 seeds in
    humus-forms-social).
    """
    if spec.get("title"):
        q = "".join(c if c.isalnum() or c.isspace() or c == "-" else " "
                    for c in spec["title"])
        resp = api_clients._request_with_retry(
            "GET", OPENALEX_BASE,
            params={"filter": f"title.search:{q}", "per_page": 5,
                    "mailto": cfg.email},
        )
        hits = (resp.json().get("results") or []) if resp is not None and resp.ok else []
        if hits:
            best = max(hits, key=lambda r: r.get("cited_by_count") or 0)
            logger.info("  %s → %s (%s) %s [title match]", spec["label"],
                        best.get("display_name", "")[:80],
                        best.get("publication_year"), best.get("id"))
            return best
        logger.warning("  no exact-title hit for %s; falling back to keyword search",
                       spec["label"])
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


def seeds_from_csv(seeds_csv: Path, cfg: api_clients.ClientConfig) -> list[dict]:
    """Resolve seed works listed in a CSV (e.g. a Zotero collection export).

    Rows with a ``doi`` are fetched directly from OpenAlex; rows without
    one fall back to a title search pinned by ``year``.

    Args:
        seeds_csv: CSV with at least ``title``; optional ``doi``, ``year``.
        cfg: API client config (email used for the polite pool).

    Returns:
        OpenAlex work records for every seed that resolved.
    """
    df = pd.read_csv(seeds_csv, dtype=str).fillna("")
    out: list[dict] = []
    for _, row in df.iterrows():
        doi = row.get("doi", "").strip()
        rec = None
        if doi:
            resp = api_clients._request_with_retry(
                "GET", f"{OPENALEX_BASE}/doi:{doi}", params={"mailto": cfg.email}
            )
            if resp is not None and resp.ok:
                rec = resp.json()
        if rec is None and row.get("title"):
            rec = find_foundational_seed(
                {"label": row["title"][:60], "search": row["title"],
                 "year_hint": row.get("year", "")}, cfg,
            )
        if rec is None:
            logger.warning("  seed unresolved: %s", row.get("title", "")[:80])
            continue
        out.append(rec)
        time.sleep(0.1)
    logger.info("resolved %d / %d seeds from %s", len(out), len(df), seeds_csv)
    return out


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
# WoS Expanded backend (forward chase only)
# ---------------------------------------------------------------------------

WOS_BASE = "https://wos-api.clarivate.com/api/wos"


def _wos_headers(cfg: api_clients.ClientConfig) -> dict[str, str]:
    return {"X-ApiKey": cfg.wos_expanded_key, "Accept": "application/json"}


def _wos_recs(payload: dict) -> list[dict]:
    recs = api_clients._wos_exp_path(payload, "Data", "Records", "records", "REC") or []
    if isinstance(recs, dict):
        recs = [recs]
    return recs if isinstance(recs, list) else []


def wos_resolve_uid(
    cfg: api_clients.ClientConfig,
    doi: str = "",
    title: str = "",
    search: str = "",
    year: str = "",
) -> str | None:
    """Find a seed's WoS UID by DOI, exact title, or keyword search.

    Args:
        cfg: Client config carrying ``wos_expanded_key``.
        doi: DOI (tried first).
        title: Exact title (``TI=``), tried if the DOI misses.
        search: Keyword string (``TS=``, terms ANDed), last resort.
        year: Optional publication year to pin title/keyword lookups.

    Returns:
        The best-matching UID (most cited), or None.
    """
    clean = lambda t: " ".join(  # noqa: E731 - WoS query-safe tokens
        w for w in "".join(c if c.isalnum() or c.isspace() else " " for c in t).split()
    )
    queries: list[str] = []
    if doi:
        queries.append(f'DO=("{doi}")')
    py = f" AND PY={year}" if year and year.isdigit() else ""
    if title:
        queries.append(f'TI=("{clean(title)}"){py}')
        if py:  # year metadata often differs by one (online vs. print)
            queries.append(f'TI=("{clean(title)}")')
    if search:
        queries.append("TS=(" + " AND ".join(clean(search).split()) + f"){py}")
    for q in queries:
        resp = api_clients._request_with_retry(
            "GET", WOS_BASE, headers=_wos_headers(cfg),
            params={"databaseId": "WOS", "usrQuery": q, "count": 5, "firstRecord": 1},
        )
        time.sleep(1.1)
        if resp is None or not resp.ok:
            continue
        recs = _wos_recs(resp.json())
        if recs:
            best = max(recs, key=lambda r: api_clients._wos_exp_citations(r) or 0)
            return best.get("UID")
    return None


def wos_forward_chase(
    uids: list[str], cfg: api_clients.ClientConfig, cap_per_seed: int = 500
) -> list[dict]:
    """Fetch records citing each WoS UID via the Expanded ``/citing`` endpoint.

    Args:
        uids: Seed WoS UIDs.
        cfg: Client config carrying ``wos_expanded_key``.
        cap_per_seed: Max citing records fetched per seed.

    Returns:
        Parsed records (``source_database == "wos_expanded"``), deduped by UID.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for uid in uids:
        first, fetched, total = 1, 0, None
        while fetched < cap_per_seed:
            want = min(100, cap_per_seed - fetched)
            resp = api_clients._request_with_retry(
                "GET", f"{WOS_BASE}/citing", headers=_wos_headers(cfg),
                params={"databaseId": "WOS", "uniqueId": uid,
                        "count": want, "firstRecord": first},
            )
            time.sleep(1.1)  # 1 req/sec throttle
            if resp is None or not resp.ok:
                cfg.log_error("citation_chase_wos", uid,
                              f"status={getattr(resp, 'status_code', 'NA')}")
                break
            payload = resp.json()
            if total is None:
                total = (payload.get("QueryResult") or {}).get("RecordsFound")
            recs = _wos_recs(payload)
            for rec in recs:
                if rec.get("UID") not in seen:
                    seen.add(rec.get("UID"))
                    out.append(api_clients._wos_expanded_record(rec))
            fetched += len(recs)
            if len(recs) < want or (total is not None and fetched >= total):
                break
            first += len(recs)
        logger.info("  wos forward(%s): collected %d of %s", uid, fetched, total)
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


def _main_wos(args: argparse.Namespace, cfg: api_clients.ClientConfig) -> int:
    """Forward chase through WoS Expanded (no OpenAlex calls)."""
    if not cfg.wos_expanded_key:
        raise SystemExit("--backend wos needs WOS_EXPANDED_API_KEY")
    uids: list[str] = []
    if not args.skip_foundational:
        for spec in FOUNDATIONAL_SEEDS:
            uid = wos_resolve_uid(cfg, title=spec.get("title", ""), search=spec["search"],
                                  year=spec.get("year_hint", ""))
            logger.info("  %s -> %s", spec["label"], uid)
            if uid:
                uids.append(uid)
    for seeds_csv in args.seeds_csv:
        df = pd.read_csv(seeds_csv, dtype=str).fillna("")
        n0 = len(uids)
        for _, row in df.iterrows():
            uid = wos_resolve_uid(cfg, doi=row.get("doi", ""),
                                  title=row.get("title", ""),
                                  year=row.get("year", ""))
            if uid:
                uids.append(uid)
            else:
                logger.warning("  seed not in WoS: %s", row.get("title", "")[:80])
        logger.info("resolved %d / %d seeds from %s", len(uids) - n0, len(df), seeds_csv)
    uids = list(dict.fromkeys(uids))
    logger.info("wos seed total: %d unique", len(uids))
    records = wos_forward_chase(uids, cfg, cap_per_seed=args.cap_per_seed)
    logger.info("wos forward chase total: %d records", len(records))
    n = merge_into_bibliography(
        records, args.bib_csv, args.bib_bib, "wos_expanded|citation_chase_forward"
    )
    logger.info("appended %d wos forward-chase records", n)
    return 0


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
    parser.add_argument(
        "--backend", choices=("openalex", "wos"), default="openalex",
        help="Citation index for the chase. 'wos' uses WoS Expanded /citing "
             "(needs WOS_EXPANDED_API_KEY; forward chase only, no auto seeds).",
    )
    parser.add_argument("--skip-foundational", action="store_true")
    parser.add_argument("--seeds-csv", type=Path, action="append", default=[],
                        help="CSV of extra seed works (title, doi, year), e.g. "
                             "data/seeds/zotero_*.csv. Repeatable.")
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
        wos_expanded_key=os.environ.get("WOS_EXPANDED_API_KEY"),
    )
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "wos":
        return _main_wos(args, cfg)

    # 1) Foundational seeds
    foundational_records: list[dict] = []
    if not args.skip_foundational:
        logger.info("looking up foundational seeds…")
        for spec in FOUNDATIONAL_SEEDS:
            rec = find_foundational_seed(spec, cfg)
            if rec is not None:
                foundational_records.append(rec)

    for seeds_csv in args.seeds_csv:
        foundational_records.extend(seeds_from_csv(seeds_csv, cfg))

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
