"""Backfill the `abstract` column in the bibliography from raw API caches.

The orchestrator's first pass dropped the `abstract` column on CSV write
(only `abstract_snippet`, capped at 400 chars, was preserved). This
script rebuilds an id-keyed map of full abstracts from
`results/raw/*.json` and patches the bibliography CSV in place.

Sources handled:

- **OpenAlex**: payload is a JSON list of Work objects; abstract is
  reconstructed from `abstract_inverted_index` via api_clients helper.
- **WoS Expanded**: payload is the Clarivate envelope; abstract is at
  `Data.Records.records.REC[*].static_data.fullrecord_metadata.abstracts.abstract`.
- **Semantic Scholar**: payload is `{data: [...]}`; abstract is in
  `r["abstract"]`.
- **HAL / TEL / theses.fr**: Solr-style; abstract is `r["abstract_s"]`
  / `r["resume"]` / `r["description_s"]`.
- **BDTD**: vufind HTML-scraped; abstract is `r["abstract"]` if present.
- **BASE**: not currently dumped (the BASE client returns less metadata).

For records still missing an abstract after the cache pass, optionally
fall back to CrossRef API by DOI.

Usage::

    python scripts/backfill_abstracts.py
    python scripts/backfill_abstracts.py --no-crossref   # skip the slow online fallback
    python scripts/backfill_abstracts.py --crossref-cap 200   # rate-limit fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from html import unescape
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

# Allow imports from project root regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_clients  # noqa: E402

logger = logging.getLogger("backfill_abstracts")


def _norm_id(value: object) -> str:
    """Coerce id to a string, stripping any URL prefix."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.startswith("https://openalex.org/"):
        return s.rsplit("/", 1)[-1]
    return s


def _wos_abstract(rec: dict) -> str | None:
    """Pull and clean a WoS Expanded record's abstract field."""
    abs_block = api_clients._wos_exp_path(
        rec, "static_data", "fullrecord_metadata", "abstracts", "abstract"
    )
    abs_one = api_clients._wos_exp_first(abs_block)
    if not isinstance(abs_one, dict):
        return None
    paras = api_clients._wos_exp_path(abs_one, "abstract_text", "p")
    if isinstance(paras, str):
        text = paras
    elif isinstance(paras, list):
        text = " ".join(p for p in paras if isinstance(p, str))
    else:
        return None
    return unescape(text).strip() or None


def _wos_id(rec: dict) -> str | None:
    """Reconstruct the WOS:NNN id used by the orchestrator."""
    uid = api_clients._wos_exp_path(rec, "UID")
    if isinstance(uid, str) and uid:
        return uid
    return None


def _hal_abstract(rec: dict) -> str | None:
    for key in ("abstract_s", "abstract", "description_s", "description"):
        v = rec.get(key)
        if isinstance(v, list) and v:
            return str(v[0]).strip() or None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _bdtd_abstract(rec: dict) -> str | None:
    for key in ("abstract", "description", "summary"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _ss_abstract(rec: dict) -> str | None:
    v = rec.get("abstract")
    return v.strip() if isinstance(v, str) and v.strip() else None


def _ss_id(rec: dict) -> str | None:
    return rec.get("paperId") or rec.get("externalIds", {}).get("CorpusId")


def _hal_id(rec: dict) -> str | None:
    """Match the orchestrator's hal id ('hal:<halId>')."""
    halid = rec.get("halId_s") or rec.get("halId")
    if halid:
        return f"hal:{halid}"
    return None


def _bdtd_id(rec: dict) -> str | None:
    return rec.get("id") or rec.get("url")


def build_id_to_abstract(raw_dir: Path) -> dict[str, str]:
    """Walk every raw JSON and return {id_norm: full_abstract}."""
    out: dict[str, str] = {}
    files = sorted(raw_dir.glob("*.json"))
    logger.info("scanning %d raw files…", len(files))
    for f in files:
        fname = f.name
        src = fname.split("__", 1)[0]
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("could not parse %s: %s", fname, exc)
            continue

        records: Iterable[dict] = ()
        if src == "openalex" and isinstance(payload, list):
            records = (r for r in payload if isinstance(r, dict))
        elif src == "wos_expanded":
            try:
                recs = (
                    payload.get("Data", {})
                          .get("Records", {})
                          .get("records", {})
                          .get("REC", [])
                )
                if isinstance(recs, dict):
                    recs = [recs]
                records = (r for r in recs if isinstance(r, dict))
            except Exception:
                continue
        elif src == "semantic_scholar":
            if isinstance(payload, dict):
                records = (r for r in payload.get("data", []) if isinstance(r, dict))
        elif src in ("hal", "theses_fr"):
            if isinstance(payload, dict):
                docs = payload.get("response", {}).get("docs", payload.get("docs", []))
                records = (r for r in docs if isinstance(r, dict))
        elif src == "bdtd":
            if isinstance(payload, list):
                records = (r for r in payload if isinstance(r, dict))
            elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
                records = (r for r in payload["records"] if isinstance(r, dict))

        for r in records:
            if src == "openalex":
                rid = _norm_id(r.get("id"))
                abstract = api_clients._reconstruct_abstract(
                    r.get("abstract_inverted_index")
                )
            elif src == "wos_expanded":
                rid = _wos_id(r) or ""
                abstract = _wos_abstract(r)
            elif src == "semantic_scholar":
                rid = _ss_id(r) or ""
                abstract = _ss_abstract(r)
            elif src == "hal":
                rid = _hal_id(r) or ""
                abstract = _hal_abstract(r)
            elif src == "theses_fr":
                # theses.fr uses 'id' or 'these.id'; orchestrator key TBD
                rid = r.get("id") or ""
                abstract = _hal_abstract(r)
            elif src == "bdtd":
                rid = _bdtd_id(r) or ""
                abstract = _bdtd_abstract(r)
            else:
                continue
            rid = str(rid).strip()
            if rid and abstract and rid not in out:
                out[rid] = abstract
    logger.info("collected abstracts for %d unique records", len(out))
    return out


def _candidate_keys(row_id: str, doi: str | None) -> list[str]:
    """Return all id forms we might match against the raw-cache map.

    The orchestrator stores id values like
    ``https://openalex.org/W4404320389``, ``WOS:000401383400017``,
    ``https://api.semanticscholar.org/graph/v1/paper/abc``, etc.
    The raw-cache map is keyed by the source-native id. This helper
    returns every plausible normalization.
    """
    keys: list[str] = []
    s = str(row_id or "").strip()
    if s:
        keys.append(s)
        if "/" in s:
            keys.append(s.rsplit("/", 1)[-1])
    if doi:
        d = str(doi).strip()
        if d.startswith("https://doi.org/"):
            d = d[len("https://doi.org/"):]
        if d.lower().startswith("doi:"):
            d = d[4:]
        if d:
            keys.append(d.lower())
            keys.append(f"doi:{d.lower()}")
    return keys


def _norm_doi(doi: str) -> str:
    s = (doi or "").strip()
    if s.startswith("https://doi.org/"):
        s = s[len("https://doi.org/"):]
    if s.lower().startswith("doi:"):
        s = s[4:]
    return s


def crossref_lookup(doi: str, session: requests.Session) -> str | None:
    """Pull abstract from CrossRef by DOI. Returns plain text or None."""
    norm = _norm_doi(doi)
    if not norm:
        return None
    url = f"https://api.crossref.org/works/{norm}"
    try:
        resp = session.get(
            url, timeout=10,
            headers={"User-Agent": "litsearch-pipeline/1.0 (mailto:noreply@example.com)"},
        )
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    msg = (resp.json() or {}).get("message") or {}
    abs_xml = msg.get("abstract")
    if not isinstance(abs_xml, str) or not abs_xml.strip():
        return None
    # CrossRef abstracts are JATS-ish XML — strip tags crudely.
    import re
    text = re.sub(r"<[^>]+>", " ", abs_xml)
    text = re.sub(r"\s+", " ", text).strip()
    return unescape(text) or None


def openalex_lookup(doi: str, session: requests.Session, email: str) -> str | None:
    """Pull abstract from OpenAlex by DOI (reconstructs inverted index)."""
    norm = _norm_doi(doi)
    if not norm:
        return None
    url = f"https://api.openalex.org/works/doi:{norm.lower()}"
    try:
        resp = session.get(url, params={"mailto": email}, timeout=10)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    payload = resp.json() or {}
    inverted = payload.get("abstract_inverted_index")
    abs_text = api_clients._reconstruct_abstract(inverted)
    return abs_text or None


def semantic_scholar_lookup(
    doi: str, session: requests.Session, api_key: str | None = None
) -> str | None:
    """Pull abstract from Semantic Scholar Graph API by DOI."""
    norm = _norm_doi(doi)
    if not norm:
        return None
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{norm}"
        f"?fields=abstract"
    )
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = session.get(url, timeout=10, headers=headers)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    body = resp.json() or {}
    ab = body.get("abstract")
    return ab.strip() if isinstance(ab, str) and ab.strip() else None


def europepmc_lookup(doi: str, session: requests.Session) -> str | None:
    """Pull abstract from Europe PMC by DOI."""
    norm = _norm_doi(doi)
    if not norm:
        return None
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": f"DOI:{norm}",
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    }
    try:
        resp = session.get(url, params=params, timeout=10)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    body = resp.json() or {}
    results = (body.get("resultList") or {}).get("result") or []
    if not results:
        return None
    ab = results[0].get("abstractText")
    return ab.strip() if isinstance(ab, str) and ab.strip() else None


def wos_uid_lookup_batch(
    uids: list[str], session: requests.Session, api_key: str
) -> dict[str, str]:
    """Fetch full WoS Expanded records by UID via the usrQuery=UT=() filter.

    Returns ``{uid: abstract}`` for the UIDs whose response contained an
    abstract. Up to 50 UIDs per call (WoS allows long OR'd queries).
    Rate-limited at 1 req/s.
    """
    if not uids or not api_key:
        return {}
    out: dict[str, str] = {}
    base = "https://wos-api.clarivate.com/api/wos"
    headers = {"X-ApiKey": api_key, "Accept": "application/json"}
    BATCH = 50
    for i in range(0, len(uids), BATCH):
        batch = uids[i : i + BATCH]
        usr_query = "UT=(" + " OR ".join(batch) + ")"
        params = {
            "databaseId": "WOS",
            "usrQuery": usr_query,
            "count": BATCH,
            "firstRecord": 1,
        }
        try:
            resp = session.get(base, params=params, headers=headers, timeout=30)
        except requests.RequestException:
            continue
        if not resp.ok:
            continue
        try:
            payload = resp.json()
        except Exception:
            continue
        recs = api_clients._wos_exp_path(
            payload, "Data", "Records", "records", "REC"
        ) or []
        if isinstance(recs, dict):
            recs = [recs]
        if not isinstance(recs, list):
            recs = []
        for rec in recs:
            uid = api_clients._wos_exp_path(rec, "UID")
            if not uid:
                continue
            ab = _wos_abstract(rec)
            if ab:
                out[str(uid)] = ab
        time.sleep(1.1)
    return out


def _autodetect_bib_csv() -> Path:
    """Find the project's bibliography CSV in ``results/``.

    Looks for ``results/*_bibliography.csv``; if exactly one exists,
    return it. Otherwise raise so the user passes ``--bib-csv``
    explicitly. This keeps the script slug-agnostic so it can be
    scaffolded into any project without name substitution.
    """
    candidates = sorted(Path("results").glob("*_bibliography.csv"))
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(
        f"Could not auto-detect bibliography CSV in results/ "
        f"(found {len(candidates)} candidates: {candidates}). "
        "Pass --bib-csv explicitly."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bib-csv", type=Path, default=None,
        help="Path to the bibliography CSV. If omitted, auto-detect "
             "the single results/*_bibliography.csv.",
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("results/raw"),
    )
    parser.add_argument(
        "--no-crossref", action="store_true",
        help="Skip the CrossRef DOI fallback.",
    )
    parser.add_argument(
        "--no-openalex", action="store_true",
        help="Skip the OpenAlex /works/doi: fallback.",
    )
    parser.add_argument(
        "--no-semantic-scholar", action="store_true",
        help="Skip the Semantic Scholar /paper/DOI: fallback.",
    )
    parser.add_argument(
        "--no-wos-uid", action="store_true",
        help="Skip the WoS Expanded UID batched fallback "
             "(targets WOS:* ids missed during the original harvest "
             "due to the dump_raw filename truncation bug).",
    )
    parser.add_argument(
        "--no-europepmc", action="store_true",
        help="Skip the Europe PMC fallback.",
    )
    parser.add_argument(
        "--europepmc-cap", type=int, default=2000,
        help="Max Europe PMC calls per run.",
    )
    parser.add_argument(
        "--crossref-cap", type=int, default=6000,
        help="Max CrossRef calls per run.",
    )
    parser.add_argument(
        "--openalex-cap", type=int, default=6000,
        help="Max OpenAlex DOI lookups per run.",
    )
    parser.add_argument(
        "--ss-cap", type=int, default=3000,
        help="Max Semantic Scholar DOI lookups per run.",
    )
    parser.add_argument(
        "--email", default="looker@stanford.edu",
        help="Contact email for OpenAlex polite pool.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.bib_csv is None:
        args.bib_csv = _autodetect_bib_csv()
        logger.info("auto-detected bibliography CSV: %s", args.bib_csv)
    df = pd.read_csv(args.bib_csv)
    logger.info("loaded %d rows from %s", len(df), args.bib_csv)
    if "abstract" not in df.columns:
        df["abstract"] = pd.NA

    abs_present = df["abstract"].notna() & df["abstract"].astype(str).str.strip().ne("") & df["abstract"].astype(str).str.lower().ne("nan")
    n_have_before = abs_present.sum()
    logger.info("rows already with abstract: %d / %d (%.1f%%)",
                n_have_before, len(df), 100 * n_have_before / len(df))

    id_to_abs = build_id_to_abstract(args.raw_dir)

    n_filled_cache = 0
    for i, row in df.iterrows():
        if abs_present.iat[i]:
            continue
        for key in _candidate_keys(row.get("id"), row.get("doi")):
            ab = id_to_abs.get(key)
            if ab:
                df.at[i, "abstract"] = ab
                n_filled_cache += 1
                break
    logger.info("filled %d abstracts from raw cache", n_filled_cache)

    # Also recompute abstract_snippet for rows where we now have a fuller
    # abstract (snippet was 400 chars; full is up to ~5 KB).
    if "abstract_snippet" in df.columns:
        def _snip(s: object) -> str:
            if not isinstance(s, str):
                return ""
            s = " ".join(s.split())
            return s[:400] + ("…" if len(s) > 400 else "")
        df["abstract_snippet"] = df["abstract"].map(_snip).where(
            df["abstract"].notna(), df["abstract_snippet"]
        )

    def _missing_with_doi() -> "pd.Series[bool]":
        miss = (
            df["abstract"].isna()
            | (df["abstract"].astype(str).str.strip() == "")
            | (df["abstract"].astype(str).str.lower() == "nan")
        )
        return miss & df["doi"].notna()

    def _online_pass(name: str, lookup_fn, cap: int, sleep_s: float) -> int:
        miss = _missing_with_doi()
        n_miss = int(miss.sum())
        cap = min(cap, n_miss)
        logger.info(
            "%s fallback: %d rows missing & have DOI; capped at %d",
            name, n_miss, cap,
        )
        if not cap:
            return 0
        session = requests.Session()
        target_idx = df.index[miss][:cap].tolist()
        n_filled = 0
        for k, idx in enumerate(target_idx, 1):
            ab = lookup_fn(str(df.at[idx, "doi"]), session)
            if ab:
                df.at[idx, "abstract"] = ab
                n_filled += 1
            if k % 200 == 0:
                logger.info("  %s %d/%d → +%d", name, k, cap, n_filled)
            time.sleep(sleep_s)
        logger.info("%s filled %d additional abstracts", name, n_filled)
        return n_filled

    # OpenAlex DOI lookup runs first because its abstract recovery is
    # the highest-yielding for our corpus (records missing from raw
    # cache due to the WoS-Expanded pagination-filename bug).
    if not args.no_openalex:
        _online_pass(
            "openalex_doi",
            lambda d, s: openalex_lookup(d, s, args.email),
            args.openalex_cap,
            sleep_s=0.10,
        )

    if not args.no_crossref:
        _online_pass(
            "crossref",
            crossref_lookup,
            args.crossref_cap,
            sleep_s=0.05,
        )

    if not args.no_semantic_scholar:
        # Semantic Scholar free tier is 1 req/s; respect that.
        _online_pass(
            "semantic_scholar",
            lambda d, s: semantic_scholar_lookup(d, s),
            args.ss_cap,
            sleep_s=1.05,
        )

    if not args.no_europepmc:
        _online_pass(
            "europepmc",
            europepmc_lookup,
            args.europepmc_cap,
            sleep_s=0.10,
        )

    # WoS-UID batched fallback for rows whose id starts with "WOS:" and
    # is still missing an abstract. These are records whose paginated
    # raw dumps got overwritten by the dump_raw filename-truncation bug.
    import os
    wos_key = os.environ.get("WOS_EXPANDED_API_KEY")
    if not args.no_wos_uid and wos_key:
        miss = (
            df["abstract"].isna()
            | (df["abstract"].astype(str).str.strip() == "")
            | (df["abstract"].astype(str).str.lower() == "nan")
        )
        wos_mask = miss & df["id"].astype(str).str.startswith("WOS:")
        wos_uids = df.loc[wos_mask, "id"].astype(str).tolist()
        logger.info("wos_uid fallback: %d WOS:* rows missing abstract",
                    len(wos_uids))
        if wos_uids:
            session = requests.Session()
            uid_to_abs = wos_uid_lookup_batch(wos_uids, session, wos_key)
            n_filled = 0
            for idx in df.index[wos_mask].tolist():
                uid = str(df.at[idx, "id"])
                ab = uid_to_abs.get(uid)
                if ab:
                    df.at[idx, "abstract"] = ab
                    n_filled += 1
            logger.info("wos_uid filled %d additional abstracts", n_filled)
    elif not args.no_wos_uid:
        logger.info("wos_uid fallback skipped: WOS_EXPANDED_API_KEY not set")

    abs_present_after = df["abstract"].notna() & df["abstract"].astype(str).str.strip().ne("") & df["abstract"].astype(str).str.lower().ne("nan")
    n_have_after = abs_present_after.sum()
    logger.info("rows with abstract: %d → %d (+%d)",
                n_have_before, n_have_after, n_have_after - n_have_before)

    # Recompute snippet uniformly so it matches the (possibly newly-filled)
    # full abstract.
    def _snip2(s: object) -> str:
        if not isinstance(s, str):
            return ""
        s = " ".join(s.split())
        return s[:400] + ("…" if len(s) > 400 else "")
    df["abstract_snippet"] = df["abstract"].map(_snip2).where(
        df["abstract"].notna() & df["abstract"].astype(str).str.lower().ne("nan"),
        df.get("abstract_snippet", pd.NA),
    )

    df.to_csv(args.bib_csv, index=False)
    logger.info("wrote %s (%d rows)", args.bib_csv, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
