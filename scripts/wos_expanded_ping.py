"""Smoke-test the Web of Science Expanded API.

Reads the key from ``WOS_EXPANDED_API_KEY`` and issues one search against
``https://wos-api.clarivate.com/api/wos``. Prints status, rate-limit
response headers (``X-REQ-ReqPerSec-Remaining``,
``X-REC-AmtPerYear-Remaining``), ``RecordsFound``, and the first few
record titles.

Usage:

    export WOS_EXPANDED_API_KEY="..."
    python scripts/wos_expanded_ping.py
    python scripts/wos_expanded_ping.py --query 'TS=(my topic terms)' --count 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

# Generic high-recall query: should match millions of records on any
# WoS database. Used only to verify credentials, network path, and
# parser. Override via --query for a topic-specific smoke test once
# the project's queries.py is filled in.
DEFAULT_QUERY = 'TS=(soil AND organic AND carbon)'
BASE_URL = "https://wos-api.clarivate.com/api/wos"


def _extract_titles(payload: dict[str, Any]) -> list[str]:
    """Pull a few record titles out of the Expanded JSON response.

    The Expanded JSON shape nests records as
    ``Data.Records.records.REC[*].static_data.summary.titles.title``,
    where ``title`` is a list of dicts with ``type`` (``item`` is the
    article title). Tolerant of variations: returns whatever titles it
    can find.
    """
    titles: list[str] = []
    recs = (
        payload.get("Data", {})
        .get("Records", {})
        .get("records", {})
        .get("REC", [])
    )
    if isinstance(recs, dict):
        recs = [recs]
    for rec in recs or []:
        try:
            ts = (
                rec.get("static_data", {})
                .get("summary", {})
                .get("titles", {})
                .get("title", [])
            )
            if isinstance(ts, dict):
                ts = [ts]
            picked = next(
                (
                    t.get("content")
                    for t in ts
                    if isinstance(t, dict) and t.get("type") == "item"
                ),
                None,
            )
            if picked:
                titles.append(picked)
        except (AttributeError, TypeError):
            continue
    return titles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY,
                        help="WoS advanced query (default: a generic high-recall TS query)")
    parser.add_argument("--database", default="WOS",
                        help="databaseId (WOS = core; WOK = all licensed)")
    parser.add_argument("--count", type=int, default=3,
                        help="Records per request (max 100)")
    parser.add_argument("--first", type=int, default=1,
                        help="firstRecord (1-indexed)")
    parser.add_argument("--dump", metavar="PATH",
                        help="Write raw JSON response to this path")
    args = parser.parse_args(argv)

    key = os.environ.get("WOS_EXPANDED_API_KEY")
    if not key:
        print("ERROR: WOS_EXPANDED_API_KEY is not set in the environment.",
              file=sys.stderr)
        print("       export WOS_EXPANDED_API_KEY='...' and re-run.",
              file=sys.stderr)
        return 2

    headers = {"X-ApiKey": key, "Accept": "application/json"}
    params = {
        "databaseId": args.database,
        "usrQuery": args.query,
        "count": args.count,
        "firstRecord": args.first,
    }

    print(f"GET  {BASE_URL}")
    print(f"     databaseId={args.database!r}  usrQuery={args.query!r}")
    print(f"     count={args.count}  firstRecord={args.first}")
    try:
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    except requests.RequestException as exc:
        print(f"REQUEST FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\nstatus: {resp.status_code} {resp.reason}")
    quota_keys = [
        "X-REQ-ReqPerSec-Remaining",
        "X-REC-AmtPerYear-Remaining",
        "X-REC-AmtPerYear-Limit",
        "X-REQ-ReqPerSec",
    ]
    print("rate-limit headers:")
    for k in quota_keys:
        if k in resp.headers:
            print(f"  {k}: {resp.headers[k]}")
    if not any(k in resp.headers for k in quota_keys):
        print("  (none of the standard X-REQ/X-REC headers were returned)")

    if not resp.ok:
        print("\nresponse body (first 1000 chars):")
        print(resp.text[:1000])
        return 1

    try:
        payload = resp.json()
    except ValueError:
        print("\nresponse was not JSON; first 500 chars:")
        print(resp.text[:500])
        return 1

    qr = payload.get("QueryResult", {}) or {}
    print(f"\nRecordsFound:    {qr.get('RecordsFound')}")
    print(f"RecordsSearched: {qr.get('RecordsSearched')}")
    print(f"QueryID:         {qr.get('QueryID')}")

    titles = _extract_titles(payload)
    if titles:
        print("\nfirst titles:")
        for i, t in enumerate(titles, start=1):
            print(f"  {i}. {t}")
    else:
        print("\n(no titles parsed from response — dump with --dump to inspect)")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote raw response → {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
