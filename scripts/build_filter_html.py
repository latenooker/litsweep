"""Build a self-contained faceted-filter HTML over a labeled corpus.

Reads one (or several) ``*_labeled_corpus.csv`` deliverables, trims each
record to display + facet fields (no abstract), and emits a single static
HTML file with the records embedded as JSON and a generic client-side
faceted search. Choosing facet values returns a list of links (title ->
DOI / open-access URL).

The facet set is **derived from the data**, so the same script works for
any litsweep deployment regardless of its label schema: every column
ending in ``_llm`` (except ``label_rationale``) becomes a facet, plus the
shared ``language`` / ``type`` metadata. Each facet's kind is detected
from its values — pipe-separated cells become multi-select *list* facets,
``True`` / ``False`` columns become *boolean* facets, everything else is a
single-select facet. Point it at several corpora at once (e.g. sibling
projects sharing one embedding space) and a ``corpus`` facet appears
automatically.

Usage::

    # Single deployment — auto-detects results/<slug>_labeled_corpus.csv
    python scripts/build_filter_html.py

    # Explicit input(s) and output
    python scripts/build_filter_html.py \
        --csv results/foo_labeled_corpus.csv \
        --out results/analysis/articles_filter.html

    # Several corpora in one interface (adds a "corpus" facet)
    python scripts/build_filter_html.py \
        --csv ../a/results/a_labeled_corpus.csv \
              ../b/results/b_labeled_corpus.csv \
        --out results/analysis/articles_filter.html

    # Keep all records (default keeps relevance_llm in core,adjacent)
    python scripts/build_filter_html.py --relevance all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10**9)

# Display columns pulled straight from each row (never facets in themselves,
# beyond what is added below). Abstract is deliberately excluded.
_AUTHORS_COL = "authors"
_YEAR_COL = "year"
_JOURNAL_COL = "source_journal_or_publisher"
_TITLE_COL = "title"
_DOI_COL = "doi"
_OA_COL = "open_access_url"
_RELEVANCE_COL = "relevance_llm"

# `_llm` columns that are not facets.
_FACET_LLM_EXCLUDE = {"label_rationale"}

# Shared (non-`_llm`) metadata facets, with case/synonym normalization.
_SHARED_META = ("language", "type")

_DROP_VALUES = {"", "nan", "none", "not_applicable", "not specified", "n/a"}

# language values arrive as a mix of ISO 639-1 codes and English names, in
# inconsistent case ("en", "English"); canonicalise to one English name.
_LANG_MAP = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "it": "Italian", "pl": "Polish", "ru": "Russian",
    "zh": "Chinese", "ko": "Korean", "ja": "Japanese", "gl": "Galician",
    "el": "Greek", "hu": "Hungarian", "sv": "Swedish", "uk": "Ukrainian",
    "bg": "Bulgarian", "sr": "Serbian", "cs": "Czech", "lv": "Latvian",
    "tr": "Turkish", "eo": "Esperanto", "hr": "Croatian", "id": "Indonesian",
    "ms": "Malay", "fi": "Finnish", "nl": "Dutch", "ca": "Catalan",
    "ro": "Romanian", "no": "Norwegian", "da": "Danish", "sk": "Slovak",
    "sl": "Slovenian", "et": "Estonian", "lt": "Lithuanian", "uk-ua": "Ukrainian",
}

# document types arrive in inconsistent case with a few synonyms; fold them.
_TYPE_MAP = {
    "proceedings paper": "proceedings",
    "dissertation": "thesis",
    "editorial material": "editorial",
    "report-component": "report",
    "reference-entry": "reference",
}


def _normalize_language(raw: str) -> str | None:
    """Canonicalise a language code/name to a single English name."""

    raw = (raw or "").strip()
    if not raw:
        return None
    return _LANG_MAP.get(raw.lower(), raw[:1].upper() + raw[1:])


def _normalize_type(raw: str) -> str | None:
    """Lowercase a document type and fold known synonyms."""

    raw = (raw or "").strip().lower()
    if not raw:
        return None
    return _TYPE_MAP.get(raw, raw)


_META_NORMALIZERS = {"language": _normalize_language, "type": _normalize_type}

# Pretty labels for the shared scaffold; everything else is derived from the
# column name (strip `_llm`/`is_`, spaces, capitalise).
_SHARED_LABELS = {
    "corpus": "Corpus",
    "relevance_llm": "Relevance",
    "is_thesis_llm": "Thesis",
    "is_review_llm": "Review",
    "language": "Language",
    "type": "Type",
    "journal": "Journal",
}
# These shared keys lead the "General" group, in this order.
_GENERAL_ORDER = (
    "corpus",
    "relevance_llm",
    "language",
    "type",
    "is_thesis_llm",
    "is_review_llm",
)


def _label_for(col: str) -> str:
    """Derive a human facet label from a column name."""

    if col in _SHARED_LABELS:
        return _SHARED_LABELS[col]
    name = col[:-4] if col.endswith("_llm") else col
    if name.startswith("is_"):
        name = name[3:]
    name = name.replace("_", " ").replace("-", " ").strip()
    return name[:1].upper() + name[1:] if name else col


def _slug_from(path: Path) -> str:
    """Short corpus name from a `<slug>_labeled_corpus*.csv` filename."""

    stem = path.stem
    for marker in ("_labeled_corpus", "_labeled", "_corpus"):
        if marker in stem:
            stem = stem.split(marker)[0]
            break
    return stem.replace("_", "-") or path.parent.name


def _discover_inputs() -> list[Path]:
    """Find labeled-corpus CSVs under ./results when --csv is omitted."""

    results = Path("results")
    if not results.is_dir():
        return []
    # prefer the deduplicated deliverable, then the canonical labeled
    # corpus, then any labeled CSV as a last resort.
    for pattern in ("*_labeled_corpus_dedup.csv", "*_labeled_corpus.csv",
                    "*_labeled*.csv"):
        found = sorted(results.glob(pattern))
        if found:
            return found
    return []


def _split_list(raw: str) -> list[str]:
    """Split a pipe- (or semicolon-) separated list cell."""

    sep = "|" if "|" in raw else ";"
    return [v.strip() for v in raw.split(sep) if v.strip()]


def _first_author(raw: str) -> str:
    """Return the first author from a `;`-separated authors string."""

    return (raw or "").split(";")[0].strip()


def _year(raw: str) -> int | None:
    """Parse a possibly-float year string to an int, or None."""

    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _link_and_doi(doi: str, oa: str) -> tuple[str, str]:
    """Return (link_url, bare_doi) preferring DOI then open-access URL."""

    doi = (doi or "").strip()
    bare = doi
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if bare.lower().startswith(prefix):
            bare = bare[len(prefix):]
            break
    if doi:
        link = doi if doi.lower().startswith("http") else f"https://doi.org/{doi}"
        return link, bare
    return (oa or "").strip(), ""


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
  :root{
    --bg:#11151a; --panel:#1a2027; --panel2:#222b34; --line:#2e3a45;
    --ink:#e7edf3; --mut:#8fa3b3; --accent:#5fb3d4; --accent2:#9bd17a;
    --chip:#2b3742;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:var(--bg); color:var(--ink);
    font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  }
  a{color:var(--accent); text-decoration:none}
  a:hover{text-decoration:underline}
  header{
    padding:14px 20px; border-bottom:1px solid var(--line);
    display:flex; gap:16px; align-items:center; flex-wrap:wrap;
    background:linear-gradient(180deg,#161c22,#11151a);
  }
  header h1{font-size:16px; margin:0; font-weight:650; letter-spacing:.2px}
  header .sub{color:var(--mut); font-size:12.5px}
  #q{
    flex:1; min-width:240px; max-width:520px; background:var(--panel);
    border:1px solid var(--line); color:var(--ink); border-radius:8px;
    padding:9px 12px; font-size:14px;
  }
  #q::placeholder{color:var(--mut)}
  .count{color:var(--accent2); font-variant-numeric:tabular-nums; font-weight:600}
  button.reset{
    background:var(--chip); border:1px solid var(--line); color:var(--ink);
    border-radius:7px; padding:8px 12px; cursor:pointer; font-size:13px;
  }
  button.reset:hover{border-color:var(--accent)}
  .layout{display:flex; height:calc(100% - 59px)}
  aside{
    width:330px; flex:none; overflow-y:auto; border-right:1px solid var(--line);
    padding:8px 4px 40px 0;
  }
  main{flex:1; overflow-y:auto; padding:6px 18px 60px}
  .group{margin:6px 10px 14px}
  .group h2{
    font-size:11px; text-transform:uppercase; letter-spacing:.7px;
    color:var(--mut); margin:14px 6px 6px; font-weight:700;
  }
  details.facet{
    background:var(--panel); border:1px solid var(--line); border-radius:8px;
    margin:6px 0; overflow:hidden;
  }
  details.facet>summary{
    list-style:none; cursor:pointer; padding:8px 10px; display:flex;
    justify-content:space-between; align-items:center; gap:8px; user-select:none;
  }
  details.facet>summary::-webkit-details-marker{display:none}
  details.facet>summary:hover{background:var(--panel2)}
  .facet .flabel{font-weight:600; font-size:13px}
  .facet .fmeta{color:var(--mut); font-size:11.5px; font-variant-numeric:tabular-nums}
  .facet.active{border-color:var(--accent)}
  .facet.active .flabel{color:var(--accent)}
  .vals{padding:4px 8px 10px; max-height:280px; overflow-y:auto}
  .valsearch{
    width:100%; margin:2px 0 6px; background:var(--bg); border:1px solid var(--line);
    color:var(--ink); border-radius:6px; padding:5px 8px; font-size:12.5px;
  }
  label.opt{
    display:flex; align-items:center; gap:7px; padding:3px 4px; border-radius:5px;
    cursor:pointer; font-size:13px;
  }
  label.opt:hover{background:var(--panel2)}
  label.opt input{accent-color:var(--accent); margin:0}
  label.opt .v{flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  label.opt .c{color:var(--mut); font-size:11.5px; font-variant-numeric:tabular-nums}
  label.opt.sel .v{color:var(--accent2); font-weight:600}
  .yearrow{display:flex; gap:6px; align-items:center; padding:4px 4px 8px}
  .yearrow input{
    width:78px; background:var(--bg); border:1px solid var(--line); color:var(--ink);
    border-radius:6px; padding:5px 7px; font-size:12.5px;
  }
  .yearrow span{color:var(--mut)}
  ul.results{list-style:none; margin:0; padding:0}
  li.rec{
    padding:10px 6px; border-bottom:1px solid var(--line);
  }
  li.rec .title{font-size:14.5px; font-weight:600}
  li.rec .title.nolink{color:var(--ink)}
  li.rec .meta{color:var(--mut); font-size:12.5px; margin-top:3px}
  li.rec .meta .au{color:var(--ink)}
  .badge{
    display:inline-block; font-size:11px; padding:1px 7px; border-radius:20px;
    background:var(--chip); border:1px solid var(--line); margin-right:7px;
    color:var(--mut); vertical-align:1px;
  }
  .badge.core{color:var(--accent2); border-color:#3f5a33}
  .doi{color:var(--mut)}
  .more{padding:16px 6px; color:var(--mut)}
  .empty{padding:40px 6px; color:var(--mut)}
</style>
</head>
<body>
<header>
  <h1>__PAGE_TITLE__</h1>
  <input id="q" placeholder="Search title, author, journal, DOI…">
  <span class="sub"><span id="count" class="count">0</span> of <span id="total">0</span> articles</span>
  <button class="reset" id="reset">Reset filters</button>
</header>
<div class="layout">
  <aside id="facets"></aside>
  <main>
    <ul class="results" id="results"></ul>
    <div id="more" class="more"></div>
  </main>
</div>
<script id="data" type="application/json">/*DATA*/</script>
<script>
"use strict";
const PAYLOAD = JSON.parse(document.getElementById("data").textContent);
const FACETS = PAYLOAD.facets;          // [{key,label,group}]
const RECS = PAYLOAD.records;           // [{co,au,yr,jo,ti,ln,doi,fx}]
const CAP = 1000;                       // max results rendered at once

document.getElementById("total").textContent = RECS.length.toLocaleString();

// active selections: facetKey -> Set(values); plus year range + text query
const sel = new Map();
let yearMin = null, yearMax = null, query = "";
const valSearch = new Map();            // facetKey -> in-facet search text

function facetVal(rec, key){
  if(key === "corpus") return rec.co;
  if(key === "journal") return rec.jo;
  return rec.fx[key];                   // string | array | undefined
}

// does a record match a single facet's selected set?
function matchFacet(rec, key, chosen){
  const v = facetVal(rec, key);
  if(v === undefined) return false;
  if(Array.isArray(v)) return v.some(x => chosen.has(x));
  return chosen.has(v);
}

function matchYear(rec){
  if(yearMin === null && yearMax === null) return true;
  if(rec.yr === null || rec.yr === undefined) return false;
  if(yearMin !== null && rec.yr < yearMin) return false;
  if(yearMax !== null && rec.yr > yearMax) return false;
  return true;
}

function matchQuery(rec){
  if(!query) return true;
  const hay = (rec.ti + " " + rec.au + " " + rec.jo + " " + rec.doi).toLowerCase();
  return query.split(/\s+/).every(t => hay.includes(t));
}

// records passing all facets EXCEPT `exceptKey` (for counting that facet),
// and always passing year + query.
function passing(exceptKey){
  return RECS.filter(rec => {
    if(!matchYear(rec) || !matchQuery(rec)) return false;
    for(const [key, chosen] of sel){
      if(key === exceptKey) continue;
      if(chosen.size && !matchFacet(rec, key, chosen)) return false;
    }
    return true;
  });
}

function finalMatches(){ return passing(null); }

// ---- rendering ----
const facetsEl = document.getElementById("facets");

function buildFacets(){
  const byGroup = new Map();
  for(const f of FACETS){
    if(!byGroup.has(f.group)) byGroup.set(f.group, []);
    byGroup.get(f.group).push(f);
  }
  facetsEl.innerHTML = "";
  for(const [group, list] of byGroup){
    const g = document.createElement("div");
    g.className = "group";
    g.innerHTML = `<h2>${group}</h2>`;
    if(group === "General"){
      g.appendChild(yearFacetEl());
    }
    for(const f of list) g.appendChild(facetEl(f));
    facetsEl.appendChild(g);
  }
}

function yearFacetEl(){
  const d = document.createElement("details");
  d.className = "facet";
  if(yearMin !== null || yearMax !== null) d.classList.add("active");
  d.open = (yearMin !== null || yearMax !== null);
  d.innerHTML = `<summary><span class="flabel">Year</span>
    <span class="fmeta" data-yrmeta></span></summary>`;
  const row = document.createElement("div");
  row.className = "yearrow";
  row.innerHTML = `<input type="number" id="ymin" placeholder="from"
      value="${yearMin ?? ""}"><span>–</span>
      <input type="number" id="ymax" placeholder="to" value="${yearMax ?? ""}">`;
  d.appendChild(row);
  row.querySelector("#ymin").addEventListener("input", e => {
    yearMin = e.target.value === "" ? null : parseInt(e.target.value, 10);
    update();
  });
  row.querySelector("#ymax").addEventListener("input", e => {
    yearMax = e.target.value === "" ? null : parseInt(e.target.value, 10);
    update();
  });
  return d;
}

function facetEl(f){
  const chosen = sel.get(f.key) || new Set();
  const avail = passing(f.key);
  // count values among available records
  const counts = new Map();
  for(const rec of avail){
    const v = facetVal(rec, f.key);
    if(v === undefined) continue;
    if(Array.isArray(v)) for(const x of v) counts.set(x, (counts.get(x)||0)+1);
    else counts.set(v, (counts.get(v)||0)+1);
  }
  // include selected-but-now-zero values so they stay unselectable-visible
  for(const v of chosen) if(!counts.has(v)) counts.set(v, 0);

  const d = document.createElement("details");
  d.className = "facet" + (chosen.size ? " active" : "");
  // hide facets with no values under the current filtering (corpus-aware)
  if(counts.size === 0){ d.style.display = "none"; return d; }
  d.open = chosen.size > 0;

  d.innerHTML = `<summary>
      <span class="flabel">${f.label}</span>
      <span class="fmeta">${chosen.size ? chosen.size + " sel · " : ""}${counts.size}</span>
    </summary>`;

  const vals = document.createElement("div");
  vals.className = "vals";

  const many = counts.size > 12;
  let needle = (valSearch.get(f.key) || "").toLowerCase();
  if(many){
    const s = document.createElement("input");
    s.className = "valsearch";
    s.placeholder = "filter values…";
    s.value = valSearch.get(f.key) || "";
    s.addEventListener("input", e => {
      valSearch.set(f.key, e.target.value);
      const nd = facetEl(f);
      nd.open = true;
      d.replaceWith(nd);
    });
    vals.appendChild(s);
  }

  const entries = [...counts.entries()]
    .filter(([v]) => !needle || v.toLowerCase().includes(needle))
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));

  for(const [v, c] of entries){
    const lab = document.createElement("label");
    lab.className = "opt" + (chosen.has(v) ? " sel" : "");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = chosen.has(v);
    cb.addEventListener("change", () => toggle(f.key, v));
    const vv = document.createElement("span");
    vv.className = "v"; vv.textContent = v; vv.title = v;
    const cc = document.createElement("span");
    cc.className = "c"; cc.textContent = c.toLocaleString();
    lab.append(cb, vv, cc);
    vals.appendChild(lab);
  }
  d.appendChild(vals);
  return d;
}

function toggle(key, v){
  let s = sel.get(key);
  if(!s){ s = new Set(); sel.set(key, s); }
  if(s.has(v)) s.delete(v); else s.add(v);
  if(!s.size) sel.delete(key);
  update();
}

const resultsEl = document.getElementById("results");
const moreEl = document.getElementById("more");

function renderResults(matches){
  document.getElementById("count").textContent = matches.length.toLocaleString();
  resultsEl.innerHTML = "";
  if(matches.length === 0){
    resultsEl.innerHTML = `<div class="empty">No articles match these facets.</div>`;
    moreEl.textContent = "";
    return;
  }
  const sorted = matches.slice().sort((a, b) => (b.yr||0) - (a.yr||0));
  const shown = sorted.slice(0, CAP);
  const frag = document.createDocumentFragment();
  for(const r of shown){
    const li = document.createElement("li");
    li.className = "rec";
    const rel = r.fx.relevance;
    const badge = `<span class="badge ${rel==='core'?'core':''}">${r.co}${rel?' · '+rel:''}</span>`;
    const titleHtml = r.ln
      ? `<a class="title" href="${r.ln}" target="_blank" rel="noopener">${esc(r.ti)||'(untitled)'}</a>`
      : `<span class="title nolink">${esc(r.ti)||'(untitled)'}</span>`;
    const doiHtml = r.doi ? ` · <span class="doi">${esc(r.doi)}</span>` : "";
    const jo = r.jo ? ` · ${esc(r.jo)}` : "";
    li.innerHTML = `${badge}${titleHtml}
      <div class="meta"><span class="au">${esc(r.au)||'—'}</span> · ${r.yr ?? 'n.d.'}${jo}${doiHtml}</div>`;
    frag.appendChild(li);
  }
  resultsEl.appendChild(frag);
  moreEl.textContent = matches.length > CAP
    ? `Showing first ${CAP.toLocaleString()} of ${matches.length.toLocaleString()} — narrow the facets to see the rest.`
    : "";
}

function esc(s){
  return (s||"").replace(/[&<>"]/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
}

function update(){
  buildFacets();
  renderResults(finalMatches());
}

document.getElementById("q").addEventListener("input", e => {
  query = e.target.value.trim().toLowerCase();
  update();
});
document.getElementById("reset").addEventListener("click", () => {
  sel.clear(); valSearch.clear();
  yearMin = yearMax = null; query = "";
  document.getElementById("q").value = "";
  update();
});

update();
</script>
</body>
</html>
"""

def _facet_columns(headers: set[str]) -> list[str]:
    """Facetable columns: shared meta + every `_llm` column we keep."""

    cols: list[str] = [c for c in _SHARED_META if c in headers]
    cols += sorted(
        c for c in headers
        if c.endswith("_llm") and c not in _FACET_LLM_EXCLUDE
    )
    return cols


def _detect_kinds(
    rows_seen: dict[str, set[str]],
) -> dict[str, str]:
    """Classify each facet column as 'list', 'bool', or 'single'.

    Args:
        rows_seen: facet column -> set of non-empty raw cell values observed.

    Returns:
        Facet column -> kind.
    """

    kinds: dict[str, str] = {}
    for col, vals in rows_seen.items():
        if any("|" in v for v in vals):
            kinds[col] = "list"
        elif vals and all(v in ("True", "False") for v in vals):
            kinds[col] = "bool"
        else:
            kinds[col] = "single"
    return kinds


def _facet_cell(kind: str, raw: str, col: str):
    """Normalise one facet cell to a value, list, or None to drop it."""

    raw = (raw or "").strip()
    if col in _META_NORMALIZERS:
        return _META_NORMALIZERS[col](raw)
    if kind == "bool":
        return "yes" if raw == "True" else "no" if raw == "False" else None
    if kind == "list":
        return _split_list(raw) or None
    return None if raw.lower() in _DROP_VALUES else raw


def _build(paths: list[Path], keep: set[str] | None) -> tuple[list[dict], list[str], dict[str, str]]:
    """Read corpora and return (records, facet columns, kinds).

    Args:
        paths: labeled-corpus CSVs to read.
        keep: allowed ``relevance_llm`` values, or None to keep all.
    """

    multi = len(paths) > 1
    raw_records: list[dict] = []
    seen: dict[str, set[str]] = defaultdict(set)
    facet_cols: list[str] = []
    facet_set: set[str] = set()

    for path in paths:
        if not path.exists():
            sys.exit(f"missing corpus CSV: {path}")
        corpus = _slug_from(path)
        kept = 0
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = set(reader.fieldnames or [])
            for col in _facet_columns(headers):
                if col not in facet_set:
                    facet_set.add(col)
                    facet_cols.append(col)
            for row in reader:
                if keep is not None and row.get(_RELEVANCE_COL) not in keep:
                    continue
                link, bare_doi = _link_and_doi(row.get(_DOI_COL, ""), row.get(_OA_COL, ""))
                facets_raw = {}
                for col in _facet_columns(headers):
                    cell = (row.get(col, "") or "").strip()
                    if cell:
                        seen[col].add(cell)
                        facets_raw[col] = cell
                raw_records.append(
                    {
                        "co": corpus,
                        "au": _first_author(row.get(_AUTHORS_COL, "")),
                        "yr": _year(row.get(_YEAR_COL, "")),
                        "jo": (row.get(_JOURNAL_COL, "") or "").strip(),
                        "ti": (row.get(_TITLE_COL, "") or "").strip(),
                        "ln": link,
                        "doi": bare_doi,
                        "_raw": facets_raw,
                    }
                )
                kept += 1
        print(f"  {corpus:20s} {kept:>6d} records")

    kinds = _detect_kinds(seen)
    records: list[dict] = []
    for rr in raw_records:
        fx: dict[str, object] = {}
        for col, cell in rr.pop("_raw").items():
            v = _facet_cell(kinds[col], cell, col)
            if v is not None:
                fx[col] = v
        rr["fx"] = fx
        records.append(rr)

    if multi:
        facet_cols = ["corpus"] + facet_cols
    return records, facet_cols, kinds


def _facet_config(facet_cols: list[str], records: list[dict]) -> list[dict]:
    """Order facets into sidebar groups; drop facets with no values."""

    # which facet columns actually carry a value somewhere
    nonempty = {"corpus"} if any(r["co"] for r in records) else set()
    for r in records:
        nonempty.update(r["fx"].keys())

    def entry(col: str, group: str) -> dict:
        return {"key": "corpus" if col == "corpus" else col,
                "label": _label_for(col), "group": group}

    general = [c for c in _GENERAL_ORDER if c in facet_cols and c in nonempty]
    rest = [c for c in facet_cols if c not in _GENERAL_ORDER and c in nonempty
            and c not in ("journal",)]

    def _is_bool(col: str) -> bool:
        vals = {r["fx"][col] for r in records
                if col in r["fx"] and not isinstance(r["fx"][col], list)}
        return bool(vals) and vals <= {"yes", "no"}

    bool_flags = [c for c in rest if _is_bool(c)]
    lists = [c for c in rest if c not in bool_flags
             and any(isinstance(r["fx"].get(c), list) for r in records)]
    singles = [c for c in rest if c not in bool_flags and c not in lists]

    config: list[dict] = []
    config += [entry(c, "General") for c in general]
    config += [entry(c, "Facets") for c in singles]
    config += [entry(c, "Flags") for c in bool_flags]
    config += [entry(c, "Lists") for c in lists]
    if any(r["jo"] for r in records):
        config.append({"key": "journal", "label": "Journal", "group": "Journal"})
    return config


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", nargs="+", type=Path, default=None,
                   help="labeled-corpus CSV(s); default auto-detects results/*_labeled_corpus.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="output HTML path (default results/analysis/articles_filter.html)")
    p.add_argument("--relevance", default="core,adjacent",
                   help="comma list of relevance_llm values to keep, or 'all' (default: core,adjacent)")
    p.add_argument("--title", default="Lit-search corpus filter",
                   help="page title shown in the header")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build records, render the HTML, and write it out."""

    args = _parse_args(argv)
    paths = args.csv or _discover_inputs()
    if not paths:
        sys.exit("no labeled-corpus CSV found; pass --csv explicitly")
    keep = None if args.relevance.strip().lower() == "all" else {
        v.strip() for v in args.relevance.split(",") if v.strip()
    }

    print(f"Reading {len(paths)} corpus file(s)"
          + ("" if keep is None else f" (relevance in {sorted(keep)}):"))
    records, facet_cols, _ = _build(paths, keep)
    print(f"Total records: {len(records)}")

    config = _facet_config(facet_cols, records)
    payload = {"facets": config, "records": records}
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # escape '<' so the data can never close the embedding <script> tag;
    # '<' is still valid JSON and parses back to '<'.
    data_json = data_json.replace("<", "\\u003c")

    out = args.out or (Path("results") / "analysis" / "articles_filter.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (_TEMPLATE
            .replace("__PAGE_TITLE__", _esc_title(args.title))
            .replace("/*DATA*/", data_json))
    out.write_text(html, encoding="utf-8")
    print(f"Facets: {len(config)}   Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def _esc_title(title: str) -> str:
    """Escape a title for safe insertion into HTML text/attributes."""

    return (title.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))



if __name__ == "__main__":
    main()
