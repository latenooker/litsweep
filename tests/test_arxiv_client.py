"""Unit tests for the arXiv client (offline; HTTP is stubbed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_clients  # noqa: E402


def _feed(total: int, entries: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>{total}</opensearch:totalResults>
  {entries}
</feed>"""


_ENTRY_PUBLISHED = """
<entry>
  <id>http://arxiv.org/abs/2401.01234v3</id>
  <published>2024-01-02T00:00:00Z</published>
  <updated>2024-03-01T00:00:00Z</updated>
  <title>Self-supervised
     soil profile segmentation</title>
  <summary>  We pretrain a ViT on
    field photos.  </summary>
  <author><name>Ada Lovelace</name></author>
  <author><name>Plato</name></author>
  <arxiv:doi>10.1016/j.geoderma.2024.000001</arxiv:doi>
  <arxiv:journal_ref>Geoderma 441 (2024)</arxiv:journal_ref>
  <link title="pdf" href="http://arxiv.org/pdf/2401.01234v3"/>
  <arxiv:primary_category term="cs.CV"/>
</entry>"""

_ENTRY_PREPRINT = """
<entry>
  <id>http://arxiv.org/abs/soil-bio/9901001v1</id>
  <published>1999-01-05T00:00:00Z</published>
  <title>An old-style identifier</title>
  <summary>Abstract.</summary>
  <author><name>Grace Brewster Hopper</name></author>
</entry>"""


def test_parse_feed_normalizes_fields() -> None:
    total, recs = api_clients._parse_arxiv_feed(_feed(2, _ENTRY_PUBLISHED + _ENTRY_PREPRINT))
    assert total == 2
    a, b = recs
    assert a["id"] == "arxiv:2401.01234"
    assert a["doi"] == "10.1016/j.geoderma.2024.000001"  # journal DOI wins
    assert a["title"] == "Self-supervised soil profile segmentation"
    assert a["abstract"] == "We pretrain a ViT on field photos."
    assert a["authors"] == "Lovelace, Ada; Plato"
    assert a["source_journal_or_publisher"] == "Geoderma 441 (2024)"
    assert a["open_access_url"] == "http://arxiv.org/pdf/2401.01234v3"
    assert a["publication_year"] == 2024
    assert a["source_database"] == "arxiv"
    # Old-style id keeps its slash; DataCite DOI fallback; no pdf link.
    assert b["id"] == "arxiv:soil-bio/9901001"
    assert b["doi"] == "10.48550/arXiv.soil-bio/9901001"
    assert b["source_journal_or_publisher"] == "arXiv"
    assert b["open_access_url"] == "http://arxiv.org/abs/soil-bio/9901001v1"
    assert b["authors"] == "Hopper, Grace Brewster"


def test_parse_feed_skips_error_entries() -> None:
    err = "<entry><id>http://arxiv.org/api/errors#bad_query</id><title>Error</title></entry>"
    total, recs = api_clients._parse_arxiv_feed(_feed(1, err))
    assert recs == []


class _Resp:
    def __init__(self, text: str) -> None:
        self.text, self.ok, self.status_code = text, True, 200


def test_search_paginates_and_retries_empty_page(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    pages = [
        _feed(2, _ENTRY_PUBLISHED),   # page 0
        _feed(2, ""),                 # transient empty page -> retried once
        _feed(2, _ENTRY_PREPRINT),    # page 1
    ]
    calls: list[dict] = []

    def fake_request(method, url, *, params=None, headers=None, timeout=30.0, **kw):
        calls.append(dict(params))
        return _Resp(pages[len(calls) - 1])

    monkeypatch.setattr(api_clients, "_request_with_retry", fake_request)
    monkeypatch.setattr(api_clients.time, "sleep", lambda s: None)
    cfg = api_clients.ClientConfig(raw_dir=tmp_path / "raw", error_log=tmp_path / "err.log")
    recs = api_clients.search_arxiv(["abs:soil"], cfg, page_size=1)
    assert [r["id"] for r in recs] == ["arxiv:2401.01234", "arxiv:soil-bio/9901001"]
    assert [c["start"] for c in calls] == [0, 1, 1]
    # One raw file per page; the retried page overwrites the empty one.
    assert len(list((tmp_path / "raw").glob("arxiv__*.json"))) == 2


def test_arxiv_registered() -> None:
    assert api_clients.CLIENTS["arxiv"] is api_clients.search_arxiv
