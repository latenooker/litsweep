"""Unit tests for the LA Referencia client (offline; HTTP is stubbed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_clients  # noqa: E402

_REC = {
    "id": "CO_0a1b2c3d",
    "title": "Polvo de roca andesítica como enmienda en suelos ácidos",
    "authors": {"primary": {"Pérez, Ana": []}, "secondary": [{"name": "Gómez, Luis"}]},
    "summary": ["Se evaluó la liberación de K y Ca.", "Ensayo en invernadero."],
    "publicationDates": ["2019"],
    "formats": ["masterThesis"],
    "languages": ["spa"],
    "urls": [{"url": "https://doi.org/10.1234/abc.5678?x=1"}, {"url": "https://repo.example/handle/1"}],
}


def test_record_normalises_fields() -> None:
    r = api_clients._lareferencia_record(_REC)
    assert r["id"] == "lareferencia:CO_0a1b2c3d"
    assert r["doi"] == "10.1234/abc.5678"
    assert r["authors"] == "Pérez, Ana; Gómez, Luis"
    assert r["abstract"] == "Se evaluó la liberación de K y Ca. Ensayo en invernadero."
    assert r["publication_year"] == 2019
    assert r["language"] == "es"
    assert r["type"] == "thesis (masterThesis)"
    assert r["source_journal_or_publisher"] == "LA Referencia (Colombia)"
    assert r["open_access_url"] == "https://doi.org/10.1234/abc.5678?x=1"
    assert r["source_database"] == "lareferencia"


def test_record_without_title_is_dropped() -> None:
    assert api_clients._lareferencia_record({"id": "AR_1", "title": ""}) is None


class _Resp:
    ok = True
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._p = payload

    def json(self) -> dict:
        return self._p


def test_search_paginates_and_stops(monkeypatch, tmp_path) -> None:
    pages = []

    def fake(method, url, *, params=None, **kw):
        page = int(dict(params)["page"])
        pages.append(page)
        recs = [{**_REC, "id": f"BR_{page}_{i}"} for i in range(100 if page == 1 else 20)]
        return _Resp({"resultCount": 120, "records": recs})

    monkeypatch.setattr(api_clients, "_request_with_retry", fake)
    monkeypatch.setattr(api_clients.time, "sleep", lambda s: None)
    cfg = api_clients.ClientConfig(raw_dir=tmp_path)
    out = api_clients.search_lareferencia(["rochagem"], cfg, max_pages=5)
    assert pages == [1, 2]
    assert len(out) == 120
    assert out[0]["source_journal_or_publisher"] == "LA Referencia (Brasil)"
