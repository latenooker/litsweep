"""Deduplicate a labeled corpus directly, without the embedding artifacts.

Post-label cleanup for a ``*_labeled_corpus.csv`` deliverable: drop
journal comments / replies, then collapse near-duplicate records
(preprint ↔ published, DOI-format variants, acronym/subtitle edits) via
DOI + title-Jaccard clustering, keeping one canonical row per group.

The decisions use only ``title``, ``doi``, ``type`` and
``relevance_llm`` — no embeddings — so this runs on the labeled
deliverable alone, including projects whose ``*_bibliography_embedded.*``
intermediates have been archived/removed by ``disk_hygiene``. It reuses
``normalize_doi`` / ``_title_tokens`` from the shared ``dedup.py`` and
inlines the post-label clustering / canonical-pick helpers so it needs no
other module.

Canonical pick per duplicate group, in order of preference: labeled over
unlabeled, then ``core`` > ``adjacent`` > ``off_topic``, then has-DOI,
then longest title. The output keeps every original column; the
``<stem>_dedup.csv`` it writes is what ``build_filter_html.py`` prefers.

Usage::

    # dry run (prints stats only)
    python scripts/dedup_labeled.py --labeled results/<slug>_labeled_corpus.csv

    # persist <stem>_dedup.csv + a manifest next to the source
    python scripts/dedup_labeled.py \\
        --labeled results/<slug>_labeled_corpus.csv --write
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Reuse the shared primitives; inline the post-label helpers below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dedup import _title_tokens, normalize_doi  # noqa: E402

logger = logging.getLogger("dedup_labeled")

_REL_RANK = {"core": 3, "adjacent": 2, "off_topic": 1}

# Series / multi-chapter volume boilerplate (USGS/book records embed the
# volume title via "A section in <volume>" / "Chapter X in <volume>",
# often in <i>…</i>). Distinct chapters otherwise share the long volume
# title and falsely merge on containment.
_SERIES_BOILER = re.compile(
    r"[:;,.\s]*(?:a\s+section\s+in|chapter\s+\w+\s+in)\b.*", re.IGNORECASE
)
_ITALIC_MARKUP = re.compile(r"&lt;/?i&gt;|</?i>", re.IGNORECASE)

# Journal correspondence (comments / replies / referee responses). OpenAlex
# types them ``peer-review``; the regex catches the rest. The ``comment``
# clause requires a following on/to/: so e.g. French "Comment expliquer…"
# is NOT matched.
_PEER_REVIEW_TYPES = {"peer-review"}
_COMMENT_REPLY = re.compile(
    r"^\s*comments?\s+(?:on|to|regarding|by)\b"
    r"|^\s*comments?\s*[:\"“']"
    r"|^\s*repl(?:y|ies)\b"
    r"|^\s*author\s+repl(?:y|ies)\b"
    r"|^\s*response\s+to\s+(?:rc|sc|cc|ac|reviewer|referee|the\s+comment|comment)\b"
    r"|^\s*(?:ac|rc|sc)\s+to\s+(?:all\s+)?reviewer"
    r"|^\s*peer\s+review\s+#?\d"
    r"|\breply\s+to:?\s+comment\b"
    r"|\bcomment\s+on\b\s+[a-z]{1,8}-?\d{2,4}",
    re.IGNORECASE,
)


class _UnionFind:
    """Minimal union-find over hashable keys with path compression."""

    def __init__(self) -> None:
        self._parent: dict = {}

    def add(self, x) -> None:
        self._parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict:
        out: dict = defaultdict(list)
        for x in self._parent:
            out[self.find(x)].append(x)
        return out


def _clean_title(title: object) -> str:
    """Strip series/volume boilerplate and italic markup from a title."""

    if not isinstance(title, str):
        return ""
    t = _ITALIC_MARKUP.sub(" ", title)
    return _SERIES_BOILER.sub("", t)


def _is_comment_reply(title: object, rec_type: object) -> bool:
    """True if a record is a comment / reply / referee-response artifact."""

    if isinstance(rec_type, str) and rec_type.strip().lower() in _PEER_REVIEW_TYPES:
        return True
    return isinstance(title, str) and bool(_COMMENT_REPLY.search(title))


def _title_match(a: frozenset, b: frozenset, jacc_thr: float, cont_thr: float,
                 min_len: int, min_ratio: float) -> bool:
    """True if two title token sets are near-duplicates.

    Matches on Jaccard >= ``jacc_thr`` OR (containment >= ``cont_thr`` AND
    the smaller set has >= ``min_len`` tokens AND is >= ``min_ratio`` of the
    larger). The guards keep the containment path from merging a short
    generic title into a longer different one while still admitting
    acronym-expansion / subtitle cases.
    """

    if not a or not b:
        return False
    inter = len(a & b)
    if inter == 0:
        return False
    if inter / len(a | b) >= jacc_thr:
        return True
    smaller, larger = min(len(a), len(b)), max(len(a), len(b))
    return (smaller >= min_len
            and smaller / larger >= min_ratio
            and inter / smaller >= cont_thr)


def _cluster(node_tokens: dict, node_doi: dict, jacc_thr: float, cont_thr: float,
             min_len: int, min_ratio: float, block_n: int,
             posting_cap: int) -> _UnionFind:
    """Union-find clustering over all nodes via DOI + blocked title edges."""

    uf = _UnionFind()
    for node in node_tokens:
        uf.add(node)

    # DOI edges.
    by_doi: dict[str, list] = defaultdict(list)
    for node, doi in node_doi.items():
        if isinstance(doi, str) and doi:
            by_doi[doi].append(node)
    for nodes in by_doi.values():
        for other in nodes[1:]:
            uf.union(nodes[0], other)

    # Title edges, blocked by rarest tokens (rare tokens have short postings;
    # common ones over the cap are skipped to bound the candidate set).
    df_count: dict[str, int] = defaultdict(int)
    for toks in node_tokens.values():
        for t in toks:
            df_count[t] += 1
    inverted: dict[str, list] = defaultdict(list)
    for node, toks in node_tokens.items():
        if not toks:
            continue
        for t in sorted(toks, key=lambda t: df_count[t])[:block_n]:
            inverted[t].append(node)

    tested: set = set()
    n_edges = 0
    for nodes in inverted.values():
        if len(nodes) > posting_cap:
            continue
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if uf.find(a) == uf.find(b):
                    continue
                pair = (a, b) if a < b else (b, a)
                if pair in tested:
                    continue
                tested.add(pair)
                if _title_match(node_tokens[a], node_tokens[b], jacc_thr,
                                cont_thr, min_len, min_ratio):
                    uf.union(a, b)
                    n_edges += 1
    logger.info("clustering: %d title edges from %d candidate pairs",
                n_edges, len(tested))
    return uf


def _pick_canonical(ids: list[str], rel: dict, doi: dict, title: dict) -> str:
    """Choose the surviving id for a duplicate group.

    Preference: labeled > unlabeled, then core > adjacent > off_topic, then
    has-DOI, then longest title.
    """

    def key(rid: str) -> tuple:
        r = rel.get(rid)
        return (r is not None, _REL_RANK.get(r, 0),
                bool(doi.get(rid)), len(title.get(rid) or ""))

    return max(ids, key=key)


def main(argv: list[str] | None = None) -> int:
    """Drop comments/replies and collapse duplicates in a labeled corpus."""

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labeled", type=Path, required=True,
                   help="Path to the <slug>_labeled_corpus.csv deliverable.")
    p.add_argument("--name", default=None, help="Short label for logs.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default <labeled_stem>_dedup.csv).")
    p.add_argument("--title-jaccard", type=float, default=0.82)
    p.add_argument("--title-containment", type=float, default=0.92)
    p.add_argument("--min-title-tokens", type=int, default=6)
    p.add_argument("--min-len-ratio", type=float, default=0.6)
    p.add_argument("--block-n", type=int, default=6)
    p.add_argument("--posting-cap", type=int, default=500)
    p.add_argument("--write", action="store_true",
                   help="Persist the deduplicated CSV + manifest.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    name = args.name or args.labeled.stem
    df = pd.read_csv(args.labeled, dtype=str, keep_default_na=False,
                     low_memory=False)
    if "id" not in df.columns:
        sys.exit("labeled CSV has no 'id' column")
    df["id"] = df["id"].astype(str)

    # Per-id metadata (first row wins on id collisions). Rows without an id
    # cannot be clustered and pass through untouched.
    has_id = df["id"].str.strip() != ""
    title: dict[str, str] = {}
    doi: dict[str, str | None] = {}
    rel: dict[str, str | None] = {}
    rec_type: dict[str, str] = {}
    for _, r in df[has_id].drop_duplicates("id").iterrows():
        rid = r["id"]
        title[rid] = r.get("title", "")
        raw_doi = r.get("doi", "")
        doi[rid] = normalize_doi(raw_doi) if raw_doi else None
        rel[rid] = r.get("relevance_llm") or None
        rec_type[rid] = r.get("type", "")
    ids = list(title)

    # Step 0 — drop comments / replies.
    cr_ids = {rid for rid in ids if _is_comment_reply(title[rid], rec_type[rid])}
    keep_ids = [rid for rid in ids if rid not in cr_ids]

    # Step 1 — DOI + title-Jaccard clustering, then collapse each group.
    node_tokens = {rid: _title_tokens(_clean_title(title[rid])) for rid in keep_ids}
    node_doi = {rid: doi[rid] for rid in keep_ids}
    uf = _cluster(node_tokens, node_doi, args.title_jaccard,
                  args.title_containment, args.min_title_tokens,
                  args.min_len_ratio, args.block_n, args.posting_cap)

    drop: set[str] = set()
    dup_groups = 0
    conflicts = 0
    for grp in uf.groups().values():
        if len(grp) < 2:
            continue
        dup_groups += 1
        rels = {rel.get(i) for i in grp if rel.get(i) is not None}
        if len(rels) > 1:
            conflicts += 1
        keeper = _pick_canonical(grp, rel, doi, title)
        drop.update(i for i in grp if i != keeper)

    survivors = {rid for rid in keep_ids if rid not in drop}
    keep_mask = ((~has_id) | df["id"].isin(survivors)) & ~df["id"].isin(cr_ids)
    out_df = df[keep_mask]

    stats = {
        "rows_before": len(df),
        "comment_reply_dropped": len(cr_ids),
        "dup_groups": dup_groups,
        "duplicates_collapsed": len(drop),
        "rows_after": len(out_df),
        "label_conflicts": conflicts,
        "cores_before": sum(1 for v in rel.values() if v == "core"),
        "cores_after": sum(1 for rid in survivors if rel.get(rid) == "core"),
    }
    logger.info(
        "%s: %d -> %d rows | dropped %d comment/reply | collapsed %d dups "
        "in %d groups | cores %d -> %d | %d label-conflicts",
        name, stats["rows_before"], stats["rows_after"],
        stats["comment_reply_dropped"], stats["duplicates_collapsed"],
        stats["dup_groups"], stats["cores_before"], stats["cores_after"],
        stats["label_conflicts"],
    )

    out = args.out or args.labeled.with_name(args.labeled.stem + "_dedup.csv")
    if args.write:
        out_df.to_csv(out, index=False)
        manifest = out.with_name("dedup_manifest_labeled.json")
        manifest.write_text(
            json.dumps({"source": str(args.labeled), "output": str(out),
                        "method": "dedup_labeled.py (comment/reply drop + "
                                  "DOI/title-Jaccard collapse)", **stats},
                       indent=2),
            encoding="utf-8",
        )
        logger.info("wrote %s (%d rows) and %s", out, len(out_df), manifest.name)
    else:
        logger.info("dry-run; pass --write to persist %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
