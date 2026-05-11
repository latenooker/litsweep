# Back-porting new sources to existing projects

litsweep is a **byte-copy scaffold**: each project's `api_clients.py`,
`dedup.py`, and the scripts under `scripts/` were copied at scaffold time
and don't auto-update. When litsweep adds a new source (or fixes a bug
in a shared file), existing projects keep running on their original
copy until someone deliberately back-ports the change.

This page is the practical checklist for back-porting recent additions
to the four pre-litsweep siblings:

- `native-sand`
- `reworming-lit`
- `microtexture-lit-search`
- `worm-tea-lit`

…and to any newer projects (currently `char14c`) that were scaffolded
before a particular fix landed.

## Critical: dedup idempotence (back-port even if you back-port nothing else)

If your project was scaffolded **before litsweep commit `780588c`**
("dedup: read existing source_databases column when available"), its
local `dedup.py` has a silent data-loss bug:

- `dedup()` reads `source_database` (singular) from input rows.
- An already-deduped CSV — like one that comes back through
  `merge_gap_fill.py` — only has `source_databases` (plural).
- So every output row gets `source_databases = ""`, and the column
  reads back as NaN from CSV.

**Symptom:** after running `scripts/merge_gap_fill.py`, the
`source_databases` column in `results/<slug>_bibliography.csv` (and
the downstream embedded + labeled corpora) is empty for every row.
The .bib file is unaffected (it's not rewritten by the merge).

**Affected projects:** all four pre-litsweep siblings (`native-sand`,
`reworming-lit`, `microtexture-lit-search`, `worm-tea-lit`) and
anything else scaffolded before 2026-04-30. `worm-tea-lit` is the
confirmed case — see "Recovery" below.

**Fix going forward:** copy `dedup.py` and `scripts/merge_gap_fill.py`
from upstream litsweep. The current `merge_gap_fill.py` ships with a
guardrail (`_check_source_coverage`) that aborts with a clear error
before overwriting the main CSV if it detects this exact collapse, so
the next person to gap-fill won't lose data silently.

```bash
cp /path/to/litsweep/dedup.py                       ./dedup.py
cp /path/to/litsweep/scripts/merge_gap_fill.py      ./scripts/
cp /path/to/litsweep/scripts/recover_source_databases_from_bib.py ./scripts/
```

### Recovery for already-affected projects

Run the recovery script. It parses the .bib's `note = {…}` field
(written before the bug zeroed out the CSV) and back-fills the
`source_databases` column on every CSV that still has a matching DOI:

```bash
cd /path/to/<sibling-project>
python scripts/recover_source_databases_from_bib.py
# or preview first:
python scripts/recover_source_databases_from_bib.py --dry-run
```

Auto-detects `results/<slug>_bibliography.bib` and updates
`<slug>_bibliography.csv`, `<slug>_bibliography_embedded.csv`, and
`<slug>_labeled_corpus.csv` if present. Backs each up to
`*.pre_source_recovery.bak` once before writing.

**Limit:** only rows whose DOIs appear in the .bib are recovered. Any
rows added to the CSV *after* the .bib was last written (e.g. a later
gap-fill harvest) stay empty and are reported. In `worm-tea-lit` this
left ~3,300 of ~34,600 rows unrecovered. To recover those, you'd need
the bibliography from the gap-fill `--output` directory (often deleted
per `DISK_HYGIENE.md`); if it's gone, accept the partial recovery.

## What's available to back-port (as of 2026-04-30)

| Change | Where | Why it matters |
|---|---|---|
| Scaffold `minerals_mod` -> `vocab_mod` rewrite | `scripts/scaffold_new_search.py` | Without this, fresh projects crash in `_augment` on first run |
| Embed-diagnostic Py3.13 importlib fix | `scripts/embed_diagnostic.py` | Lets the diagnostic script run on Python 3.13 |
| Label 429 backoff | `scripts/label_with_stanford.py` | Sustained Stanford-gateway throttling no longer writes error rows |
| `dedup()` idempotence | `dedup.py` | `merge_gap_fill` no longer wipes `source_databases` on re-dedup |
| **SciELO source** | `api_clients.py` :: `search_scielo` | Latin-American + Iberian es/pt papers (no key) |
| **Europe PMC source** | `api_clients.py` :: `search_europepmc` | bioRxiv / medRxiv / Research Square / SSRN preprints (`SRC:PPR`) |
| **EarthArXiv source** | `api_clients.py` :: `search_eartharxiv` | Earth-science preprints (via Crossref `prefix:10.31223`) |
| **Crossref source (broad)** | `api_clients.py` :: `search_crossref` | Journal-hosted papers with DOIs that OpenAlex undersamples (JSTOR, Cambridge, regional repositories) |
| **CORE.ac.uk source** | `api_clients.py` :: `search_core` | University institutional repositories — academia.edu / dspace / regional grey literature |
| Defaults rebalanced | `litsweep_search.py` | Drop `wos` (Starter) and `base` from `DEFAULT_SOURCES`; add scielo / europepmc / eartharxiv / crossref / core |

## Per-project decision: do you back-port?

Three reasonable answers:

1. **No** — the project finished its sweep, the corpus is frozen, you
   don't plan to iterate. Don't touch it. Future projects will pick up
   the new sources automatically; the snapshot you have is fine.
2. **Just the bug fixes** — copy `dedup.py` and the patched scripts so
   the next iteration of an existing search doesn't trip the known
   bugs, but keep the source list as-is.
3. **Full back-port + gap-fill** — copy everything; harvest the new
   sources into a separate `--output` directory; dedup-merge; re-embed
   (cache hits prior rows); label only new survivors above the embed-
   score floor. This is what `char14c` did for SciELO.

Use option (3) only when you have a concrete reason to think a new
source covers a gap relevant to that project's topic — see the per-
source notes below.

## Per-source: who benefits?

| Project | scielo | europepmc | eartharxiv | crossref | core |
|---|:-:|:-:|:-:|:-:|:-:|
| `native-sand` (mineral grain microtexture, sand provenance) | low | low | **medium** (sediment-transport preprints) | medium | medium |
| `reworming-lit` (earthworm restoration meta-analysis) | low | **high** (lots of bioRxiv ecology) | low | medium | medium |
| `microtexture-lit-search` (SEM / quartz / heavy minerals) | low | low | low | medium | medium |
| `worm-tea-lit` (vermicompost extracts; cost / MRV) | low | medium | low | medium | medium |
| `char14c` (radiocarbon dating of charcoal) | **high** (Latin-American archaeology) | medium (some paleoecology preprints) | medium | **high** (institutional-repo gray literature) | **high** (same) |

"High" = expect ≥10% of new core records to come from this source on
gap-fill. "Medium" = expect 1–5%. "Low" = the source's coverage doesn't
materially overlap the project topic; back-port only if you want
completeness.

## Per-project mechanical steps (when option 3)

```bash
# 1. Sync shared infra from the upstream litsweep checkout
cd /path/to/<sibling-project>
cp /path/to/litsweep/api_clients.py ./api_clients.py
cp /path/to/litsweep/dedup.py       ./dedup.py
cp /path/to/litsweep/scripts/scaffold_new_search.py ./scripts/  # optional
cp /path/to/litsweep/scripts/embed_diagnostic.py    ./scripts/
cp /path/to/litsweep/scripts/merge_gap_fill.py      ./scripts/
cp /path/to/litsweep/scripts/label_with_stanford.py ./scripts/  # PRESERVE your project's SYSTEM_PROMPT and _error_label first; merge by hand

# 2. In <slug>_search.py, register new sources in SOURCE_QUERIES and
#    add the ones you want to DEFAULT_SOURCES. Use getattr(Q, "X", [])
#    so the orchestrator gracefully no-ops if the project hasn't
#    written queries for that source yet.

# 3. In queries.py, add per-source query blocks. Patterns:
#    - SCIELO   — 2-3 token strings; AND-of-tokens; es/pt/en
#    - EUROPEPMC — boolean term1 OR term2; defaults to PPR-only
#    - EARTHARXIV — short English keywords; routed via Crossref prefix
#    - CROSSREF — short English / multilingual; broad recall, dedup handles overlap
#    - CORE     — short English; volume-heavy, dedup handles overlap

# 4. Smoke-test
python <slug>_search.py --dry-run --sources scielo,europepmc,eartharxiv,crossref,core

# 5. Gap-fill harvest into an isolated --output
python <slug>_search.py \
    --sources scielo,europepmc,eartharxiv,crossref,core \
    --email you@example.com \
    --output results_newsources

# 6. Dedup-merge — the patched dedup() is idempotent so the pre-existing
#    source_databases values survive
python scripts/merge_gap_fill.py --gap-dir results_newsources

# 7. Re-embed; the cache skips already-encoded rows
python scripts/embed_filter.py \
    --csv results/<slug>_bibliography.csv \
    --out results/<slug>_bibliography_embedded.csv

# 8. Label only the rows above your embed-score floor; the labeler
#    resumes from existing checkpoints, so it'll only call the LLM on
#    the genuinely new survivors
python scripts/label_with_stanford.py \
    --csv results/<slug>_bibliography_embedded.csv \
    --out results/<slug>_labeled_corpus.csv \
    --min-score 0.45
```

## Watch-outs by source

### SciELO
- The article landing pages don't ship abstracts in the search-results
  HTML, so freshly-harvested records arrive title-only.
- The topic-presence regex (the hard `topic_mentioned` gate) over-
  rejects title-only records. Either drop the gate for SciELO rows or
  scrape the article-page abstracts (see `char14c`'s approach in
  `scripts/scrape_scielo_abstracts.py` if it exists).
- The site's search uses AND-of-tokens; **2–3 token queries** work,
  4+ token compound queries return zero. Expand vocabulary by writing
  more 2-token combinations, not by stacking tokens.

### Europe PMC
- Default mode (`only_preprints=True`) filters to `SRC:PPR` to avoid
  re-pulling MEDLINE that OpenAlex already has. For biomedical
  projects you probably want `only_preprints=False`.
- Returns abstracts as JATS-fragment XML; the `_strip_html()` helper
  in `api_clients.py` cleans this up.

### EarthArXiv
- EarthArXiv's own `/repository/search` endpoint **ignores the query
  parameter** (verified 2026-04-30); we route through Crossref filtered
  to `prefix:10.31223,type:posted-content`.
- Yields are small (typically 5–30 records per query). Worth running
  for any Earth-science project; the unique value is high-quality
  preprints not yet in journals.

### Crossref (broad)
- Heavy recall — 60K+ results for a typical broad query; dedup handles
  overlap with OpenAlex/SemScho.
- The `select=` parameter trims response size; do not omit it.
- Returns abstracts only when the publisher provides them — typically
  ~30–50% of records.

### CORE.ac.uk
- The search endpoint URL **must end with a trailing slash**:
  `https://api.core.ac.uk/v3/search/works/`. Without the slash you get
  a 301 to an HTML redirect page that some HTTP layers won't follow.
- Anonymous tier works for moderate use (~50 queries/day). Set
  `CORE_API_KEY` in your shell for a higher quota.
- High recall, lower precision than the publisher-curated sources;
  the LLM scope check carries the load.

## After back-porting: housekeeping

- Update the project's `README.md` source-list section.
- If the project keeps a `docs/PROJECT_STATE.md` (reworming-lit pattern),
  log the back-port date and the net new core count.
- Push the updated `results/<slug>_labeled_corpus.csv` to your rclone
  remote (per `DISK_HYGIENE.md`).
- The 4 already-deployed sibling projects don't have to back-port in
  lockstep — pick them off when each one is next iterated. The
  byte-copy model is exactly so they can drift safely.
