"""API clients for each bibliographic source.

Every client returns a list of normalized records with a common shape:

    {
        "id": str,                        # source-specific stable id
        "doi": str | None,
        "title": str,
        "publication_year": int | None,
        "language": str | None,           # lowercase 2-letter code or None
        "type": str | None,               # article, thesis, dissertation, ...
        "abstract": str | None,
        "authors": str,                   # "Last, First; Last, First; ..."
        "source_journal_or_publisher": str | None,
        "cited_by_count": int | None,
        "open_access_url": str | None,
        "raw": dict,                      # original record for archival
        "source_database": str,           # which client produced this
    }

Clients log errors but never raise to the caller — the pipeline tolerates
partial failures.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _request_with_retry(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> requests.Response | None:
    """Issue an HTTP request with exponential backoff.

    Returns the Response on success (any non-retryable status), or None if
    all retries fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(
                method, url, params=params, headers=headers, timeout=timeout
            )
            if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                wait = backoff_base ** attempt
                logger.warning(
                    "%s %s -> %d, retrying in %.1fs",
                    method, url, resp.status_code, wait,
                )
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_base ** attempt
                logger.warning("%s %s failed (%s), retrying in %.1fs",
                               method, url, exc, wait)
                time.sleep(wait)
                continue
    logger.error("%s %s failed after %d retries: %s",
                 method, url, max_retries, last_exc)
    return None


# ---------------------------------------------------------------------------
# Client base
# ---------------------------------------------------------------------------


@dataclass
class ClientConfig:
    """Runtime configuration shared by all clients."""

    email: str = "anonymous@example.com"
    raw_dir: Path = field(default_factory=lambda: Path("results/raw"))
    error_log: Path = field(default_factory=lambda: Path("results/errors.log"))
    semantic_scholar_key: str | None = None
    wos_key: str | None = None
    wos_expanded_key: str | None = None
    base_key: str | None = None
    core_key: str | None = None  # CORE.ac.uk; anonymous tier works for moderate use
    per_query_cap: int = 500

    def log_error(self, source: str, query: str, msg: str) -> None:
        self.error_log.parent.mkdir(parents=True, exist_ok=True)
        with self.error_log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{source}]\t{query}\t{msg}\n")

    def dump_raw(self, source: str, query: str, payload: Any) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        # Long queries got their `__pN` page suffix truncated under the
        # old 80-char clamp, so paginated responses overwrote each other.
        # Hash the body so each unique query+page combo gets its own file.
        import hashlib
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in query)
        prefix = safe[:60]
        suffix = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
        path = self.raw_dir / f"{source}__{prefix}__{suffix}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def _reconstruct_abstract(inverted: dict | None) -> str | None:
    """Rebuild abstract text from OpenAlex's inverted index."""
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions) or None


def _openalex_record(work: dict) -> dict[str, Any]:
    primary_loc = (work.get("primary_location") or {}) or {}
    source = (primary_loc.get("source") or {}) or {}
    authors = "; ".join(
        a.get("author", {}).get("display_name", "")
        for a in (work.get("authorships") or [])
        if a.get("author", {}).get("display_name")
    )
    oa = (work.get("open_access") or {}) or {}
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("display_name") or work.get("title"),
        "publication_year": work.get("publication_year"),
        "language": work.get("language"),
        "type": work.get("type"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "authors": authors,
        "source_journal_or_publisher": source.get("display_name"),
        "cited_by_count": work.get("cited_by_count"),
        "open_access_url": oa.get("oa_url"),
        "raw": work,
        "source_database": "openalex",
    }


def search_openalex(
    queries: Iterable[str], cfg: ClientConfig
) -> list[dict[str, Any]]:
    """Search OpenAlex /works for each query, paginating until cap reached."""
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    base = "https://api.openalex.org/works"

    for q in queries:
        collected: list[dict] = []
        cursor = "*"
        while True:
            params = {
                "search": q,
                "per_page": 200,
                "cursor": cursor,
                "mailto": cfg.email,
            }
            resp = _request_with_retry("GET", base, params=params)
            if resp is None or not resp.ok:
                cfg.log_error(
                    "openalex", q,
                    f"status={getattr(resp, 'status_code', 'NA')}",
                )
                break
            payload = resp.json()
            results = payload.get("results", []) or []
            collected.extend(results)
            meta = payload.get("meta") or {}
            cursor = meta.get("next_cursor")
            if not cursor or not results or len(collected) >= cfg.per_query_cap:
                break
            time.sleep(0.15)  # ~6 req/s, well under polite-pool 10/s

        cfg.dump_raw("openalex", q, collected)
        for w in collected:
            wid = w.get("id")
            if wid and wid in seen_ids:
                continue
            if wid:
                seen_ids.add(wid)
            out.append(_openalex_record(w))
    return out


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def _ss_record(p: dict) -> dict[str, Any]:
    ext = p.get("externalIds") or {}
    authors = "; ".join(a.get("name", "") for a in (p.get("authors") or []) if a.get("name"))
    pdf = (p.get("openAccessPdf") or {}) or {}
    return {
        "id": p.get("paperId"),
        "doi": ext.get("DOI"),
        "title": p.get("title"),
        "publication_year": p.get("year"),
        "language": None,  # S2 does not report language
        "type": None,
        "abstract": p.get("abstract"),
        "authors": authors,
        "source_journal_or_publisher": p.get("venue"),
        "cited_by_count": p.get("citationCount"),
        "open_access_url": pdf.get("url"),
        "raw": p,
        "source_database": "semantic_scholar",
    }


def search_semantic_scholar(
    queries: Iterable[str], cfg: ClientConfig
) -> list[dict[str, Any]]:
    """Search Semantic Scholar paper search endpoint."""
    out: list[dict[str, Any]] = []
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = (
        "paperId,externalIds,title,year,abstract,authors,venue,"
        "citationCount,isOpenAccess,openAccessPdf,fieldsOfStudy"
    )
    headers = {}
    if cfg.semantic_scholar_key:
        headers["x-api-key"] = cfg.semantic_scholar_key
        per_call_sleep = 1.0
    else:
        per_call_sleep = 3.5  # 100/5min ≈ one per 3s; pad a bit

    for q in queries:
        params = {"query": q, "limit": 100, "fields": fields}
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error(
                "semantic_scholar", q,
                f"status={getattr(resp, 'status_code', 'NA')}",
            )
            time.sleep(per_call_sleep)
            continue
        payload = resp.json()
        cfg.dump_raw("semantic_scholar", q, payload)
        for p in payload.get("data", []) or []:
            out.append(_ss_record(p))
        time.sleep(per_call_sleep)
    return out


# ---------------------------------------------------------------------------
# WoS Starter
# ---------------------------------------------------------------------------


def _wos_record(d: dict) -> dict[str, Any]:
    ids = d.get("identifiers") or {}
    src = d.get("source") or {}
    names = d.get("names") or {}
    authors_list = (names.get("authors") or []) if isinstance(names, dict) else []
    authors = "; ".join(
        a.get("displayName") or a.get("wosStandard", "")
        for a in authors_list
        if isinstance(a, dict)
    )
    return {
        "id": d.get("uid"),
        "doi": ids.get("doi"),
        "title": (d.get("title") or {}).get("value")
        if isinstance(d.get("title"), dict)
        else d.get("title"),
        "publication_year": (src.get("publishYear")
                             or (src.get("publishedBiblioYear"))),
        "language": None,
        "type": d.get("documentType"),
        "abstract": None,  # not in starter response
        "authors": authors,
        "source_journal_or_publisher": src.get("sourceTitle"),
        "cited_by_count": d.get("citations", [{}])[0].get("count")
        if d.get("citations") else None,
        "open_access_url": None,
        "raw": d,
        "source_database": "wos",
    }


def search_wos_starter(
    queries: Iterable[str], cfg: ClientConfig
) -> list[dict[str, Any]]:
    """Search the WoS Starter API. Skips silently if no key configured."""
    if not cfg.wos_key:
        logger.info("WoS API key not set; skipping WoS")
        return []
    out: list[dict[str, Any]] = []
    base = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    headers = {"X-ApiKey": cfg.wos_key, "accept": "application/json"}

    for q in queries:
        params = {"q": q, "limit": 50, "page": 1, "db": "WOS"}
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error(
                "wos", q, f"status={getattr(resp, 'status_code', 'NA')}",
            )
            time.sleep(1.0)
            continue
        payload = resp.json()
        cfg.dump_raw("wos", q, payload)
        for d in payload.get("hits", []) or []:
            out.append(_wos_record(d))
        time.sleep(1.0)  # WoS Starter: 50/day, no need to hammer
    return out


# ---------------------------------------------------------------------------
# WoS Expanded
#
# Different product from Starter: deeper coverage, returns abstracts, much
# higher quota (~1M+ records/year vs Starter's 50/day), 1 req/sec throttle.
# Endpoint, request shape, and response shape all differ from Starter, so
# this is a separate client rather than a switch in search_wos_starter.
# ---------------------------------------------------------------------------


def _wos_exp_first(x: Any) -> Any:
    """Coerce WoS Expanded's dict-or-list-of-dicts polymorphism to one dict.

    Many fields are wrapped as ``{"count": N, "<field>": <thing>}`` where
    ``<thing>`` is a single dict when count==1 and a list of dicts when
    count>1. This helper returns the first dict regardless.
    """
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _wos_exp_path(d: Any, *keys: str) -> Any:
    """Walk a dict path; return None if any segment is missing or non-dict.

    WoS Expanded responses occasionally substitute a string or scalar
    for what is normally a nested dict (e.g. ``identifiers`` arriving as
    an empty string when a record has no identifiers). Walking with
    ``.get()`` chains then crashes on the next ``.get`` call.
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _wos_exp_title(rec: dict) -> str | None:
    titles = _wos_exp_path(rec, "static_data", "summary", "titles", "title")
    if isinstance(titles, dict):
        titles = [titles]
    for t in titles or []:
        if isinstance(t, dict) and t.get("type") == "item":
            return t.get("content")
    return None


def _wos_exp_authors(rec: dict) -> str:
    names = _wos_exp_path(rec, "static_data", "summary", "names", "name")
    if isinstance(names, dict):
        names = [names]
    out: list[str] = []
    for n in names or []:
        if not isinstance(n, dict):
            continue
        if n.get("role") and n["role"] != "author":
            continue
        out.append(
            n.get("display_name") or n.get("full_name") or n.get("wos_standard") or ""
        )
    return "; ".join(s for s in out if s)


def _wos_exp_doi(rec: dict) -> str | None:
    ids = _wos_exp_path(
        rec, "dynamic_data", "cluster_related", "identifiers", "identifier"
    )
    if isinstance(ids, dict):
        ids = [ids]
    for i in ids or []:
        if isinstance(i, dict) and (i.get("type") or "").lower() == "doi":
            return i.get("value")
    return None


def _wos_exp_abstract(rec: dict) -> str | None:
    abs_block = _wos_exp_path(
        rec, "static_data", "fullrecord_metadata", "abstracts", "abstract"
    )
    abs_one = _wos_exp_first(abs_block)
    if not isinstance(abs_one, dict):
        return None
    p = (abs_one.get("abstract_text") or {}).get("p")
    if isinstance(p, list):
        return " ".join(s for s in p if isinstance(s, str)) or None
    if isinstance(p, str):
        return p or None
    return None


def _wos_exp_year(rec: dict) -> int | None:
    pi = _wos_exp_path(rec, "static_data", "summary", "pub_info") or {}
    if not isinstance(pi, dict):
        return None
    y = pi.get("@pubyear") or pi.get("pubyear")
    try:
        return int(y) if y is not None else None
    except (TypeError, ValueError):
        return None


def _wos_exp_doctype(rec: dict) -> str | None:
    dt = _wos_exp_path(rec, "static_data", "summary", "doctypes", "doctype")
    if isinstance(dt, list):
        return dt[0] if dt else None
    return dt


def _wos_exp_journal(rec: dict) -> str | None:
    titles = _wos_exp_path(rec, "static_data", "summary", "titles", "title")
    if isinstance(titles, dict):
        titles = [titles]
    for t in titles or []:
        if isinstance(t, dict) and t.get("type") in ("source", "source_abbrev"):
            return t.get("content")
    return None


def _wos_exp_language(rec: dict) -> str | None:
    langs = _wos_exp_path(
        rec, "static_data", "fullrecord_metadata", "languages", "language"
    )
    one = _wos_exp_first(langs)
    if isinstance(one, dict):
        return one.get("content")
    return None


def _wos_exp_citations(rec: dict) -> int | None:
    silo = _wos_exp_path(
        rec, "dynamic_data", "citation_related", "tc_list", "silo_tc"
    )
    silo_one = _wos_exp_first(silo)
    if not isinstance(silo_one, dict):
        return None
    v = silo_one.get("@local_count") or silo_one.get("local_count")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _wos_expanded_record(rec: dict) -> dict[str, Any]:
    return {
        "id": rec.get("UID"),
        "doi": _wos_exp_doi(rec),
        "title": _wos_exp_title(rec),
        "publication_year": _wos_exp_year(rec),
        "language": _wos_exp_language(rec),
        "type": _wos_exp_doctype(rec),
        "abstract": _wos_exp_abstract(rec),
        "authors": _wos_exp_authors(rec),
        "source_journal_or_publisher": _wos_exp_journal(rec),
        "cited_by_count": _wos_exp_citations(rec),
        "open_access_url": None,
        "raw": rec,
        "source_database": "wos_expanded",
    }


def search_wos_expanded(
    queries: Iterable[str], cfg: ClientConfig
) -> list[dict[str, Any]]:
    """Search the WoS Expanded API with pagination.

    Skips silently if no key is configured. Paginates each query up to
    ``cfg.per_query_cap`` records (Expanded returns up to 100 per request).
    """
    if not cfg.wos_expanded_key:
        logger.info("WoS Expanded API key not set; skipping wos_expanded")
        return []
    out: list[dict[str, Any]] = []
    base = "https://wos-api.clarivate.com/api/wos"
    headers = {"X-ApiKey": cfg.wos_expanded_key, "Accept": "application/json"}
    page_size = 100

    for q in queries:
        first = 1
        fetched = 0
        total: int | None = None
        while fetched < cfg.per_query_cap:
            want = min(page_size, cfg.per_query_cap - fetched)
            params = {
                "databaseId": "WOS",
                "usrQuery": q,
                "count": want,
                "firstRecord": first,
            }
            resp = _request_with_retry("GET", base, params=params, headers=headers)
            if resp is None or not resp.ok:
                cfg.log_error(
                    "wos_expanded", q,
                    f"status={getattr(resp, 'status_code', 'NA')} first={first}",
                )
                break
            payload = resp.json()
            cfg.dump_raw("wos_expanded", f"{q}__p{first}", payload)
            qr = payload.get("QueryResult", {}) or {}
            if total is None:
                total = qr.get("RecordsFound")
            recs = _wos_exp_path(payload, "Data", "Records", "records", "REC") or []
            if isinstance(recs, dict):
                recs = [recs]
            if not isinstance(recs, list):
                recs = []
            for rec in recs:
                out.append(_wos_expanded_record(rec))
            got = len(recs)
            fetched += got
            if got < want or (total is not None and fetched >= total):
                break
            first += got
            time.sleep(1.1)  # 1 req/sec throttle
        time.sleep(1.1)
    return out


# ---------------------------------------------------------------------------
# HAL / TEL
# ---------------------------------------------------------------------------


def _hal_record(d: dict) -> dict[str, Any]:
    authors = d.get("authFullName_s") or []
    if isinstance(authors, list):
        authors = "; ".join(authors)
    abstract = d.get("abstract_s")
    if isinstance(abstract, list):
        abstract = abstract[0] if abstract else None
    title = d.get("title_s")
    if isinstance(title, list):
        title = title[0] if title else None
    year = None
    produced = d.get("producedDate_s")
    if isinstance(produced, str) and len(produced) >= 4 and produced[:4].isdigit():
        year = int(produced[:4])
    return {
        "id": f"hal:{d.get('docid')}",
        "doi": d.get("doiId_s"),
        "title": title,
        "publication_year": year,
        "language": d.get("language_s")[0]
        if isinstance(d.get("language_s"), list) and d.get("language_s")
        else d.get("language_s"),
        "type": "thesis",
        "abstract": abstract,
        "authors": authors,
        "source_journal_or_publisher": "HAL/TEL",
        "cited_by_count": None,
        "open_access_url": d.get("uri_s"),
        "raw": d,
        "source_database": "hal",
    }


def search_hal(queries: Iterable[str], cfg: ClientConfig) -> list[dict[str, Any]]:
    """Search HAL/TEL (theses portion) via Solr-style query."""
    out: list[dict[str, Any]] = []
    base = "https://api.archives-ouvertes.fr/search/tel"
    fl = (
        "docid,title_s,authFullName_s,producedDate_s,uri_s,abstract_s,"
        "language_s,doiId_s"
    )
    for q in queries:
        params = {"q": q, "fl": fl, "rows": 100, "wt": "json"}
        resp = _request_with_retry("GET", base, params=params)
        if resp is None or not resp.ok:
            cfg.log_error("hal", q, f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(0.5)
            continue
        payload = resp.json()
        cfg.dump_raw("hal", q, payload)
        for d in (payload.get("response") or {}).get("docs", []) or []:
            out.append(_hal_record(d))
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------
# theses.fr
# ---------------------------------------------------------------------------


def _theses_fr_record(d: dict) -> dict[str, Any]:
    authors_field = d.get("auteurs") or d.get("authors") or []
    if isinstance(authors_field, list):
        authors = "; ".join(
            a.get("nom_complet") or a.get("nom", "")
            for a in authors_field
            if isinstance(a, dict)
        )
    else:
        authors = str(authors_field) if authors_field else ""
    year = d.get("dateSoutenance") or d.get("annee")
    if isinstance(year, str) and len(year) >= 4 and year[:4].isdigit():
        year = int(year[:4])
    return {
        "id": f"theses_fr:{d.get('id') or d.get('nnt')}",
        "doi": None,
        "title": d.get("titrePrincipal") or d.get("titre"),
        "publication_year": year if isinstance(year, int) else None,
        "language": d.get("langue"),
        "type": "thesis",
        "abstract": d.get("resume"),
        "authors": authors,
        "source_journal_or_publisher": d.get("etablissementSoutenance")
        or d.get("etablissement"),
        "cited_by_count": None,
        "open_access_url": d.get("accessibleEnLigne"),
        "raw": d,
        "source_database": "theses_fr",
    }


def search_theses_fr(
    queries: Iterable[str], cfg: ClientConfig
) -> list[dict[str, Any]]:
    """Search theses.fr metadata API."""
    out: list[dict[str, Any]] = []
    base = "https://theses.fr/api/v1/theses/recherche"
    for q in queries:
        params = {"q": q, "nombre": 100, "debut": 0}
        resp = _request_with_retry("GET", base, params=params)
        if resp is None or not resp.ok:
            cfg.log_error(
                "theses_fr", q,
                f"status={getattr(resp, 'status_code', 'NA')}",
            )
            time.sleep(0.5)
            continue
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            cfg.log_error("theses_fr", q, f"json_decode={exc}")
            continue
        cfg.dump_raw("theses_fr", q, payload)
        results = (
            payload.get("theses")
            or payload.get("results")
            or payload.get("items")
            or []
        )
        for d in results:
            if isinstance(d, dict):
                out.append(_theses_fr_record(d))
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------
# BASE — XML over HTTP
# ---------------------------------------------------------------------------


def search_base(queries: Iterable[str], cfg: ClientConfig) -> list[dict[str, Any]]:
    """Search BASE (Bielefeld Academic Search Engine). Requires API key."""
    if not cfg.base_key:
        logger.info("BASE API key not set; skipping BASE")
        return []
    out: list[dict[str, Any]] = []
    base = "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
    headers = {"Authorization": cfg.base_key}

    for q in queries:
        params = {"func": "PerformSearch", "query": q, "hits": 100, "format": "xml"}
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error("base", q, f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(0.5)
            continue
        cfg.dump_raw("base", q, {"xml": resp.text})
        try:
            soup = BeautifulSoup(resp.text, "xml")
        except Exception as exc:  # pragma: no cover - defensive
            cfg.log_error("base", q, f"xml_parse={exc}")
            continue
        for doc in soup.find_all("doc"):
            title = (doc.find("dctitle") or {}).get_text(strip=True) if doc.find("dctitle") else None
            authors_tags = doc.find_all("dccreator")
            authors = "; ".join(t.get_text(strip=True) for t in authors_tags)
            year_tag = doc.find("dcyear") or doc.find("dcdate")
            year = None
            if year_tag and year_tag.get_text(strip=True)[:4].isdigit():
                year = int(year_tag.get_text(strip=True)[:4])
            type_tag = doc.find("dctype")
            doctype = type_tag.get_text(strip=True) if type_tag else None
            link_tag = doc.find("dclink") or doc.find("dcidentifier")
            url = link_tag.get_text(strip=True) if link_tag else None
            lang_tag = doc.find("dclang") or doc.find("dclanguage")
            lang = lang_tag.get_text(strip=True) if lang_tag else None
            internal_id = doc.find("internal_id")
            out.append({
                "id": f"base:{internal_id.get_text(strip=True) if internal_id else url}",
                "doi": None,
                "title": title,
                "publication_year": year,
                "language": lang,
                "type": doctype,
                "abstract": None,
                "authors": authors,
                "source_journal_or_publisher": "BASE",
                "cited_by_count": None,
                "open_access_url": url,
                "raw": str(doc),
                "source_database": "base",
            })
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------
# BDTD — VuFind scraper
# ---------------------------------------------------------------------------


def search_bdtd(queries: Iterable[str], cfg: ClientConfig) -> list[dict[str, Any]]:
    """Scrape BDTD VuFind result pages. Best-effort; tolerant of markup drift."""
    out: list[dict[str, Any]] = []
    base = "https://bdtd.ibict.br/vufind/Search/Results"
    headers = {
        "User-Agent": (
            "microtexture-lit-search/0.1 "
            f"(mailto:{cfg.email}; research literature search)"
        ),
    }
    for q in queries:
        params = {"lookfor": q, "type": "AllFields", "limit": 50}
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error("bdtd", q, f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(1.0)
            continue
        cfg.dump_raw("bdtd", q, {"html": resp.text[:200_000]})
        soup = BeautifulSoup(resp.text, "html.parser")
        records = soup.select("div.result, li.result, .record")
        for rec in records:
            title_tag = rec.select_one("a.title, .title a, h3 a")
            title = title_tag.get_text(strip=True) if title_tag else None
            href = title_tag.get("href") if title_tag else None
            url = (
                f"https://bdtd.ibict.br{href}"
                if href and href.startswith("/")
                else href
            )
            author_tag = rec.select_one(".author, .result-author")
            author = author_tag.get_text(" ", strip=True) if author_tag else ""
            year = None
            year_tag = rec.select_one(".publishDate, .date, .year")
            if year_tag:
                txt = year_tag.get_text(strip=True)
                for tok in txt.split():
                    if tok.isdigit() and len(tok) == 4:
                        year = int(tok)
                        break
            inst_tag = rec.select_one(".institution, .publisher")
            inst = inst_tag.get_text(strip=True) if inst_tag else "BDTD"
            if not title:
                continue
            out.append({
                "id": f"bdtd:{url or title[:80]}",
                "doi": None,
                "title": title,
                "publication_year": year,
                "language": "pt",
                "type": "thesis",
                "abstract": None,
                "authors": author,
                "source_journal_or_publisher": inst,
                "cited_by_count": None,
                "open_access_url": url,
                "raw": str(rec)[:5000],
                "source_database": "bdtd",
            })
        time.sleep(1.0)
    return out


# ---------------------------------------------------------------------------
# SciELO — public-search HTML scraper (no API key)
# ---------------------------------------------------------------------------

# Country / collection suffix on the SciELO record id → ISO 639-1 language.
# Most collections publish in their national language; SciELO Brasil = pt,
# the Spanish-American collections = es. The labelers downstream will
# refine if a record turns out to be English.
_SCIELO_COLLECTION_LANG: dict[str, str] = {
    "bra": "pt", "scl": "pt",      # Brasil + SciELO core (mostly Brazilian)
    "prt": "pt", "cap": "pt",      # Portugal, Cabo Verde
    "mex": "es", "arg": "es", "chl": "es", "col": "es", "ven": "es",
    "cri": "es", "per": "es", "cub": "es", "ury": "es", "bol": "es",
    "ecu": "es", "pry": "es",
    "esp": "es",                   # Spain
    "sza": "en",                   # South Africa
}


def search_scielo(queries: Iterable[str], cfg: ClientConfig) -> list[dict[str, Any]]:
    """Search SciELO via the public search interface.

    SciELO's JSON API was retired; the supported public path is the
    HTML search at ``https://search.scielo.org/?q=...&output=site``.
    A real-browser User-Agent is required (server returns 403 to
    default Python/curl UAs).

    Each result ``div.item`` carries the SciELO PID in its ``id``
    attribute (e.g. ``S0016-71692024000401225-mex``); the trailing
    three-letter token identifies the regional collection and is
    used to seed the ``language`` field. Title, authors, journal,
    year, and the article HTML URL are scraped per-item.
    """
    out: list[dict[str, Any]] = []
    base = "https://search.scielo.org/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en,es;q=0.8,pt;q=0.7",
    }
    for q in queries:
        # `lang=all` returns HTTP 500; SciELO defaults to all-language
        # search when the parameter is omitted, so leave it out.
        params = {"q": q, "output": "site", "count": 100}
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error("scielo", q, f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(1.0)
            continue
        cfg.dump_raw("scielo", q, {"html": resp.text[:300_000]})
        soup = BeautifulSoup(resp.text, "html.parser")
        for it in soup.select("div.item"):
            pid = (it.get("id") or "").strip()
            if not pid:
                continue
            collection = pid.rsplit("-", 1)[-1] if "-" in pid else ""
            lang = _SCIELO_COLLECTION_LANG.get(collection)
            title_tag = it.select_one("strong.title")
            title = title_tag.get_text(strip=True) if title_tag else None
            url_tag = it.select_one('a[href*="sci_arttext"]')
            url = url_tag.get("href") if url_tag else None
            authors_tag = it.select_one(".authors")
            authors = authors_tag.get_text(" ", strip=True) if authors_tag else ""
            # Journal + year live in subsequent .line blocks; pull the
            # one whose first non-empty token matches a 4-digit year.
            journal = None
            year: int | None = None
            for line in it.select(".line"):
                txt = line.get_text(" ", strip=True)
                # 'Geofísica internacional ... Dic 2024, Volumen 63 Nº 4'
                if not journal and "," in txt and len(txt) < 200:
                    # First reasonably-short .line that isn't the title
                    # or the authors usually has the journal name first.
                    parts = [p.strip() for p in txt.split() if p.strip()]
                    if parts and parts[0].isalpha():
                        journal = txt.split(" Métricas")[0].split(" Sobre")[0]
                for tok in txt.split():
                    if tok.isdigit() and len(tok) == 4 and 1900 <= int(tok) <= 2100:
                        year = int(tok)
                        break
                if year:
                    break
            if not title:
                continue
            out.append({
                "id": f"scielo:{pid}",
                "doi": None,
                "title": title,
                "publication_year": year,
                "language": lang,
                "type": "article",
                "abstract": None,  # rendered separately on each article page
                "authors": authors,
                "source_journal_or_publisher": journal or "SciELO",
                "cited_by_count": None,
                "open_access_url": url,
                "raw": str(it)[:5000],
                "source_database": "scielo",
            })
        time.sleep(1.0)
    return out


# ---------------------------------------------------------------------------
# Europe PMC — covers MEDLINE, PMC, and (filtered) preprint repos
# (bioRxiv, medRxiv, Research Square, SSRN, Authorea/ESSOAr, etc.).
# Public REST API, no key. Default in DEFAULT_SOURCES uses the
# PPR-only filter because PubMed/PMC overlap with OpenAlex; projects
# that want a fuller Europe PMC pull can call this client directly.
# ---------------------------------------------------------------------------


def _strip_html(s: str) -> str:
    """Strip simple inline HTML wrappers (<title>, <p>, etc.) from an
    abstract string. Europe PMC preprint abstracts arrive as JATS-ish
    fragments like '<title>Abstract</title> <p>The ...</p>'."""
    if not s:
        return ""
    # Cheap: drop anything between <...>
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())


def search_europepmc(
    queries: Iterable[str],
    cfg: ClientConfig,
    *,
    only_preprints: bool = True,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Search Europe PMC.

    By default narrows to preprints (``SRC:PPR``) since PubMed/PMC
    coverage already overlaps OpenAlex. Set ``only_preprints=False`` to
    pull the full index (useful for biomedical projects).
    """
    out: list[dict[str, Any]] = []
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    for q in queries:
        full = f"({q}) AND SRC:PPR" if only_preprints else q
        params = {
            "query": full,
            "format": "json",
            "pageSize": page_size,
            "resultType": "core",
        }
        resp = _request_with_retry("GET", base, params=params, headers=None)
        if resp is None or not resp.ok:
            cfg.log_error("europepmc", q,
                          f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(0.5)
            continue
        try:
            data = resp.json()
        except Exception as exc:
            cfg.log_error("europepmc", q, f"json={exc}")
            continue
        cfg.dump_raw("europepmc", q, data)
        for r in (data.get("resultList") or {}).get("result") or []:
            doi = r.get("doi")
            full_text_urls: list[str] = []
            for u in (r.get("fullTextUrlList") or {}).get("fullTextUrl") or []:
                if u.get("url"):
                    full_text_urls.append(u["url"])
            url = full_text_urls[0] if full_text_urls else (
                f"https://doi.org/{doi}" if doi else None
            )
            year = None
            if r.get("pubYear"):
                try:
                    year = int(r["pubYear"])
                except (TypeError, ValueError):
                    year = None
            abstract = _strip_html(r.get("abstractText") or "")
            out.append({
                "id": f"epmc:{r.get('source','PPR')}:{r.get('id') or doi}",
                "doi": doi,
                "title": r.get("title"),
                "publication_year": year,
                "language": None,  # EPMC doesn't reliably return language
                "type": (r.get("pubTypeList") or {}).get("pubType", [""])[0]
                        if isinstance((r.get("pubTypeList") or {}).get("pubType"), list)
                        else None,
                "abstract": abstract or None,
                "authors": r.get("authorString"),
                "source_journal_or_publisher":
                    r.get("journalTitle")
                    or (r.get("bookOrReportDetails") or {}).get("publisher")
                    or "Europe PMC",
                "cited_by_count": r.get("citedByCount"),
                "open_access_url": url,
                "raw": json.dumps(r)[:5000],
                "source_database": "europepmc",
            })
        time.sleep(0.3)
    return out


# ---------------------------------------------------------------------------
# EarthArXiv — Earth-science preprint repo. Their site search ignores
# query parameters, but the DOI prefix 10.31223 is registered with
# Crossref so we use the Crossref Works API filtered to that prefix.
# ---------------------------------------------------------------------------


def search_eartharxiv(
    queries: Iterable[str],
    cfg: ClientConfig,
    *,
    rows: int = 50,
) -> list[dict[str, Any]]:
    """Search EarthArXiv via Crossref's prefix filter.

    EarthArXiv's own /repository/search endpoint ignores query
    parameters (verified on 2026-04-30), so we hit the Crossref Works
    API with ``filter=prefix:10.31223,type:posted-content`` instead.
    Crossref's response includes title, DOI, and (sometimes) abstract.
    """
    out: list[dict[str, Any]] = []
    base = "https://api.crossref.org/works"
    headers = {
        "User-Agent": (
            f"litsweep/0.1 (mailto:{cfg.email or 'noreply@example.com'}; "
            "research literature search)"
        ),
    }
    for q in queries:
        params = {
            "query": q,
            "filter": "prefix:10.31223,type:posted-content",
            "rows": rows,
            # Note: Crossref's /works `select` rejects `language` and `type`
            # with HTTP 400. Both fields are still returned at the row level
            # under the default selection, so just trim the select list.
            "select": "DOI,title,abstract,author,published,URL,publisher,container-title",
        }
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error("eartharxiv", q,
                          f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(0.5)
            continue
        try:
            data = resp.json()
        except Exception as exc:
            cfg.log_error("eartharxiv", q, f"json={exc}")
            continue
        cfg.dump_raw("eartharxiv", q, data)
        for r in (data.get("message") or {}).get("items") or []:
            doi = r.get("DOI")
            title_list = r.get("title") or []
            title = title_list[0] if title_list else None
            abstract = _strip_html(r.get("abstract") or "")
            authors = "; ".join(
                f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                for a in (r.get("author") or [])
            )
            published = r.get("published") or {}
            year = None
            dp = published.get("date-parts")
            if isinstance(dp, list) and dp and isinstance(dp[0], list) and dp[0]:
                year = dp[0][0]
            url = (r.get("URL") or
                   (f"https://doi.org/{doi}" if doi else None))
            out.append({
                "id": f"eartharxiv:{doi}" if doi else f"eartharxiv:{title[:80] if title else 'unknown'}",
                "doi": doi,
                "title": title,
                "publication_year": year,
                "language": r.get("language"),
                "type": "preprint",
                "abstract": abstract or None,
                "authors": authors,
                "source_journal_or_publisher": r.get("publisher") or "EarthArXiv",
                "cited_by_count": None,
                "open_access_url": url,
                "raw": json.dumps(r)[:5000],
                "source_database": "eartharxiv",
            })
        time.sleep(0.3)
    return out


# ---------------------------------------------------------------------------
# Crossref — broad DOI registry. Higher recall than OpenAlex on
# JSTOR/Cambridge journal-hosted Latin American / Spanish-language
# archaeology and similar regional venues. No key required.
# ---------------------------------------------------------------------------


def search_crossref(
    queries: Iterable[str],
    cfg: ClientConfig,
    *,
    rows: int = 50,
) -> list[dict[str, Any]]:
    """Broad Crossref Works search (no prefix filter).

    Heavy overlap with OpenAlex/Semantic Scholar is expected — dedup by
    DOI handles it. The added value is regional-repository-hosted DOIs
    (institutional repositories, JSTOR, Cambridge journals, etc.) that
    OpenAlex sometimes under-samples.
    """
    out: list[dict[str, Any]] = []
    base = "https://api.crossref.org/works"
    headers = {
        "User-Agent": (
            f"litsweep/0.1 (mailto:{cfg.email or 'noreply@example.com'}; "
            "research literature search)"
        ),
    }
    for q in queries:
        params = {
            "query": q,
            "rows": rows,
            # Crossref's /works `select` rejects `language` and `type`.
            "select": "DOI,title,abstract,author,published,URL,publisher,container-title",
        }
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error("crossref", q,
                          f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(0.5)
            continue
        try:
            data = resp.json()
        except Exception as exc:
            cfg.log_error("crossref", q, f"json={exc}")
            continue
        cfg.dump_raw("crossref", q, data)
        for r in (data.get("message") or {}).get("items") or []:
            doi = r.get("DOI")
            title_list = r.get("title") or []
            title = title_list[0] if title_list else None
            abstract = _strip_html(r.get("abstract") or "")
            authors = "; ".join(
                f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                for a in (r.get("author") or [])
            )
            published = r.get("published") or {}
            year = None
            dp = published.get("date-parts")
            if isinstance(dp, list) and dp and isinstance(dp[0], list) and dp[0]:
                year = dp[0][0]
            url = r.get("URL") or (f"https://doi.org/{doi}" if doi else None)
            container = (r.get("container-title") or [None])[0]
            out.append({
                "id": f"crossref:{doi}" if doi else f"crossref:{title[:80] if title else 'unknown'}",
                "doi": doi,
                "title": title,
                "publication_year": year,
                "language": r.get("language"),
                "type": r.get("type"),
                "abstract": abstract or None,
                "authors": authors,
                "source_journal_or_publisher": container or r.get("publisher") or "Crossref",
                "cited_by_count": None,
                "open_access_url": url,
                "raw": json.dumps(r)[:5000],
                "source_database": "crossref",
            })
        time.sleep(0.3)
    return out


# ---------------------------------------------------------------------------
# CORE.ac.uk — full-text aggregator that explicitly indexes university
# institutional repositories. Catches the academia.edu / dspace /
# JSTOR-hosted gray literature that OpenAlex routinely misses. The
# anonymous tier works for moderate use; pass a CORE_API_KEY env var
# in cfg.core_key for higher-volume access.
# ---------------------------------------------------------------------------


def search_core(
    queries: Iterable[str],
    cfg: ClientConfig,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search CORE.ac.uk Works API.

    Note the trailing slash on the search path — without it the server
    301-redirects through HTML (request.get follows redirects, but if
    the caller's HTTP layer doesn't, you'll get HTML back).
    """
    out: list[dict[str, Any]] = []
    base = "https://api.core.ac.uk/v3/search/works/"
    headers = {}
    core_key = getattr(cfg, "core_key", None)
    if core_key:
        headers["Authorization"] = f"Bearer {core_key}"
    for q in queries:
        params = {"q": q, "limit": limit}
        resp = _request_with_retry("GET", base, params=params, headers=headers)
        if resp is None or not resp.ok:
            cfg.log_error("core", q,
                          f"status={getattr(resp, 'status_code', 'NA')}")
            time.sleep(0.5)
            continue
        try:
            data = resp.json()
        except Exception as exc:
            cfg.log_error("core", q, f"json={exc}")
            continue
        cfg.dump_raw("core", q, data)
        for r in data.get("results") or []:
            doi = r.get("doi")
            title = r.get("title")
            abstract = r.get("abstract")
            authors_list = r.get("authors") or []
            authors = "; ".join(
                a.get("name") for a in authors_list if isinstance(a, dict) and a.get("name")
            )
            year = r.get("yearPublished")
            urls = r.get("sourceFulltextUrls") or []
            url = urls[0] if urls else (f"https://doi.org/{doi}" if doi else None)
            lang_obj = r.get("language") or {}
            lang = lang_obj.get("code") if isinstance(lang_obj, dict) else None
            publisher = r.get("publisher") or "CORE"
            doctype = r.get("documentType")
            cid = r.get("id") or doi or (title[:80] if title else "unknown")
            if not title:
                continue
            out.append({
                "id": f"core:{cid}",
                "doi": doi,
                "title": title,
                "publication_year": year,
                "language": lang,
                "type": doctype,
                "abstract": abstract or None,
                "authors": authors,
                "source_journal_or_publisher": publisher,
                "cited_by_count": r.get("citationCount"),
                "open_access_url": url,
                "raw": json.dumps(r)[:5000],
                "source_database": "core",
            })
        time.sleep(0.4)
    return out


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

SearchFn = Callable[[Iterable[str], ClientConfig], list[dict[str, Any]]]

CLIENTS: dict[str, SearchFn] = {
    "openalex": search_openalex,
    "semantic_scholar": search_semantic_scholar,
    "wos": search_wos_starter,
    "wos_expanded": search_wos_expanded,
    "hal": search_hal,
    "theses_fr": search_theses_fr,
    "base": search_base,
    "bdtd": search_bdtd,
    "scielo": search_scielo,
    "europepmc": search_europepmc,
    "eartharxiv": search_eartharxiv,
    "crossref": search_crossref,
    "core": search_core,
}
