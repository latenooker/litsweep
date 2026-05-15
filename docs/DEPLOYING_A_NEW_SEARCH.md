# Deploying a new lit-search

A practical checklist for spinning up a new topic-specific
bibliographic search using **litsweep** as the canonical scaffold.

> **Read first**: [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)
> for the canonical project layout and
> [DISK_HYGIENE.md](DISK_HYGIENE.md) for the per-stage compression
> and rclone-push norms. These two docs define the discipline a
> new project should follow from day one.

## What you're getting

A pipeline that, end-to-end:

1. Queries up to 8 bibliographic databases (OpenAlex, Semantic Scholar,
   WoS Starter, WoS Expanded, HAL/TEL, theses.fr, BASE, BDTD) with
   topic-specific search strings.
2. Deduplicates by DOI + Jaccard title similarity across sources.
3. Embeds title + abstract with local Ollama BGE-M3 and scores by
   cosine against your topic anchors.
4. Filters by anchor cosine and labels survivors with the Stanford
   AI gateway against a strict-JSON schema you define.
5. Writes a single labeled corpus CSV + BibTeX export.
6. Optional: runs a cross-project bridge against a sibling project's
   embedding matrix.

What you're **not** getting from this scaffold: query design, anchor
descriptions, or the LLM label schema. Those are the actual research
work and depend on your topic.

## Prerequisites (one-time, you already have these)

- **Python 3.11+** with `pip`, ideally in a per-project venv.
- **Ollama** running locally with `bge-m3` pulled
  (`ollama pull bge-m3`).
- **A labeling backend.** `scripts/label_with_stanford.py` takes
  `--label-backend {stanford,ollama}` (default `stanford`):
  - `stanford` — the Stanford AI gateway; needs `STANFORD_API_KEY`
    in the environment. The cheaper/faster default path.
  - `ollama` — local Ollama with a chat model pulled (e.g.
    `ollama pull llama3.1`); **no Stanford key required**. Slower
    but fully offline / zero API cost.
  Pick one; you don't need `STANFORD_API_KEY` if you label with
  `--label-backend ollama`.
- **API keys in your shell environment** (export from `~/.zshrc`):
  - `STANFORD_API_KEY` — gateway for label_with_stanford
    (only needed for the default `--label-backend stanford`).
  - `WOS_EXPANDED_API_KEY` — Clarivate WoS Expanded (paid).
  - `WOS_API_KEY` — WoS Starter (free tier, 50/day).
  - `BASE_API_KEY` — global thesis aggregator (free).
  - `SEMANTIC_SCHOLAR_API_KEY` — optional (free, raises rate limit).
- **`rclone`** with a remote configured for backup
  (`rclone listremotes` should show your destination, e.g. `su-drive:`).
- **Disk headroom** of ~2 GB per project for the embedding matrix
  + intermediate CSVs (use the parquet-archive workflow if tight).

CC needs a relaunch after editing `~/.zshrc` so its tool subshells
inherit the new env vars; an export inside an active CC session
won't propagate.

## Step 1 — scaffold (mechanical, ~30 seconds)

Always scaffold from **litsweep** — it's the canonical source of
truth for the shared infrastructure. Sibling projects (native-sand,
reworming-lit, etc.) drift from litsweep over time as their
topic-specific edits accumulate; do not scaffold from them.

```bash
python /path/to/litsweep/scripts/scaffold_new_search.py \
    /path/to/new-project --name my_topic
```

What it does:

- Creates the directory tree per
  [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)
  (`results/raw/`, `docs/`, `scripts/`).
- Byte-copies the shared infrastructure (`api_clients.py`, `dedup.py`,
  `requirements.txt`, `.gitignore`).
- Copies the orchestrator template + every shared script (
  `embed_filter`, `label_with_stanford`, `embed_diagnostic`,
  `merge_gap_fill`, `backfill_abstracts`, `citation_chase`,
  `wos_gap_fill`, `wos_expanded_ping`, etc.) with project-name
  substitution.
- Writes placeholder `queries.py`, `vocab.py`, anchors, and
  system prompt — every one is marked `TODO` so you can grep them.
  `vocab.py` ships with the `VOCAB_AXES` registry pattern so adding
  axes is a one-line change.
- Writes a minimal `README.md` and `CLAUDE.md`.
- `git init -b main`, stages everything, makes the initial commit
  (skip the entire git workflow with `--no-git`).
- **Creates a private GitHub repo via `gh repo create` and pushes
  the initial commit.** Repo name is the slug with underscores
  rewritten to dashes (e.g. `--name worm_tea_lit` → `worm-tea-lit`).
  The owner defaults to your `gh` auth'd user; pass
  `--remote-owner <org>` for an org. Skip with `--no-remote`; use
  `--public` if the search shouldn't be private.

If `gh` isn't installed, isn't auth'd, or the repo already exists,
the scaffold prints the manual `gh repo create` command and
continues — the local repo is still complete.

The `--from` flag exists for backwards compatibility (scaffold from
a sibling project) but is discouraged. Prefer the canonical litsweep
flow above.

### Avoiding redundant harvest with `--from-existing-corpus`

If your new project's scope overlaps a sibling project's corpus, pass
the sibling's labeled-corpus CSV (or Parquet) once at scaffold time
and the harvest will skip every DOI it contains:

```bash
python /path/to/litsweep/scripts/scaffold_new_search.py \
    /path/to/new-project --name my_topic \
    --from-existing-corpus /path/to/sibling/results/<sibling>_labeled_corpus.csv
```

The flag is repeatable (pass it twice for two siblings). It writes
the union of DOIs to `data/doi_exclude.txt` in the new project; the
orchestrator's `--doi-exclude data/doi_exclude.txt` flag (on by
default) reads that file right after the dedup step and drops every
matching record before embed + label spend.

The exclude is "every DOI from the sibling," not just the
in-scope-for-the-new-topic subset — the assumption is that if the
sibling already harvested it, you don't want to re-fetch and
re-label it; you want to *carry the existing label over* if the
record is also in scope here. For the carryover step itself
(scoring sibling records against your new anchors and pulling in
the relevant ones), see the per-project filter pattern in
`pparadox/run_native_sand_pparadox_filter.py` — this stays out of
litsweep because it depends on whichever sibling project you're
filtering from.

## Step 2 — fill in the four topic-specific files

This is the work the scaffold can't do for you. Each file has TODO
markers; grep `TODO` after scaffolding.

### 2a. `queries.py` — search strings

Six to eight constants the orchestrator dispatches by source. Conventions:

- `OPENALEX_GROUP_A_CORE_EN: list[str]` — core English prose queries
  (10–25 strings). Each is whitespace-tokenized; OpenAlex matches
  loosely against title + abstract.
- `OPENALEX_GROUP_B_PROCESS: list[str]` — process / kinetics / dynamics
  queries (5–10).
- (Optional) `OPENALEX_GROUP_C_*` — non-English variants. BGE-M3
  handles 100+ languages so multilingual queries pay off.
- `WOS_STARTER` / `WOS_EXPANDED` — TS=(...) field-tagged syntax.
  Identical between Starter and Expanded; alias the lists.
- `HAL` / `THESES_FR` — Solr / keyword grammars.
- `BASE` / `BDTD` — keyword strings.

Time investment: half a day to a couple of days, dominated by
domain-vocabulary research. Look at top-cited papers in your topic
and harvest their keyword choices; iterate after the first dry run.

Smoke-test after editing:

```bash
python <slug>_search.py --dry-run
```

### 2b. `vocab.py` — vocabulary axes via the registry

The scaffold's `vocab.py` stub uses the `VOCAB_AXES` registry
pattern. Adding a new axis is a four-step pattern:

1. Define a tag dict (e.g. `MANAGEMENT: dict[str, list[str]]`).
2. Compile its patterns via `_compile()`.
3. Define a `find_<axis>(text) -> list[str]` helper.
4. Register a `VocabAxis(name, column, find=find_<axis>)` in
   `VOCAB_AXES`.

The orchestrator iterates `VOCAB_AXES` to derive one column per axis
on every record, and the CSV write step picks up registered columns
automatically — no edits to the orchestrator or `CSV_COLUMNS`. Stop
here if your topic doesn't need a presence gate.

Optional: `TOPIC_PRESENCE = (column_name, predicate)` declares a
single-axis boolean gate (e.g. "any earthworm taxon mentioned?")
for use as a hard `--require-column` filter before LLM labeling.
Set to `None` for projects that don't have a single-regex topic
(e.g. multi-theme corpora; see [the multi-theme guidance below
](#multi-theme-corpora-do-not-gate-on-a-single-regex-column)).

Plus `TITLE_EXCLUDE_SUBSTRINGS: tuple[str, ...]` for early off-topic
filtering (reserve for high-confidence excludes — these drop records
pre-dedup).

### 2c. `scripts/embed_filter.py` :: `ANCHORS`

A list of 6–10 prose descriptions, each 1–3 sentences, covering one
facet of your topic:

```python
ANCHORS: list[str] = [
    # 0 — primary topic (the paper's main framing)
    "...one to three sentences describing this facet...",
    # 1 — secondary topic / process angle
    "...",
    # 2 — methodological angle (instrumental signature in abstracts)
    "...",
    # ...
    # n-1 — multilingual umbrella (a single anchor in 4-6 languages)
    "...one sentence each in French, German, Polish, Russian, "
    "Portuguese, Spanish — BGE-M3 handles all of them...",
]
```

Each record's `embed_score` is the **max** cosine across all anchors,
with `embed_top_anchor` recording which anchor was the max. Anchor
diversity matters more than anchor count — overlapping anchors give
identical max scores; non-overlapping anchors expand coverage.

Cost-effective starting point: write 4–6 anchors, run on a sample of
1k records, manually inspect the score distribution and the top-
anchor breakdown, add or split anchors where coverage is thin.

For **multi-theme** corpora (one anchor per theme), expect a lower
median cosine ceiling because each record can only be near one
anchor's prose, not "near a broad topic." Score distributions with
median ~0.50 and p90 ~0.62 are healthy in that regime; the per-
anchor breakdown from `scripts/embed_diagnostic.py` is the right
diagnostic. Maintain `docs/ANCHOR_REVISIONS.md` (one section per
edit) so anchor iterations are diff-able.

### 2d. `scripts/label_with_stanford.py` :: `SYSTEM_PROMPT`

Strict-JSON system prompt with a fixed schema. Structure:

1. Identity sentence ("You are a … research librarian").
2. One paragraph defining IN SCOPE (be specific — your search will
   surface adjacent literatures and the LLM needs the boundary).
3. One paragraph defining OUT OF SCOPE (with concrete examples the
   labeler will see).
4. Hard sentence: `"Return ONLY a JSON object — no prose, no code fences."`
5. The schema. Every closed-set field gets enum values; lists are
   bounded vocabularies. At minimum:
   ```
   "relevance": "core" | "adjacent" | "off_topic"
   "rationale": "<one short sentence (≤25 words)>"
   ```
   plus 2–8 axis labels (lithology, process, climate, methods,
   minerals, theme binaries, etc.) appropriate to your topic.
6. Rules section covering edge cases:
   - Missing-abstract handling.
   - The "core" vs "adjacent" boundary (the most-mistaken cell —
     err strict).
   - Topic-specific traps you've hit during dry runs.

The schema you choose is **load-bearing** — every downstream
analysis (`gap_matrix.py`, `cross_project_bridge.py` facets,
post-corpus sweeps) reads from these fields. Adding a field later is
a re-labeling of the entire corpus. Get this right before running at
scale.

The default model `gemini-2.0-flash-lite-001` is cheap and fast at
~$0.04 / 1k calls; bump to a stronger model only if dry-run accuracy
is unacceptable.

## Step 3 — run

```bash
# Install deps in a fresh venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify env (STANFORD_API_KEY only needed for --label-backend stanford)
env | grep -E "STANFORD_API_KEY|WOS_|BASE_API_KEY"

# Stage 1: harvest + dedup (writes results/<slug>_bibliography.csv + .bib)
python <slug>_search.py --email you@example.com

# Stage 2: embed (writes results/<slug>_bibliography_embedded.csv + .npy + .ids.txt)
# --embed-backend defaults to ollama (only backend today; flag reserved
# for future backends).
python scripts/embed_filter.py \
    --csv results/<slug>_bibliography.csv \
    --out results/<slug>_bibliography_embedded.csv

# Stage 2.5: anchor coverage diagnostic (no API cost; run before labeling)
python scripts/embed_diagnostic.py \
    --markdown docs/anchor_coverage_$(date +%F).md
# Reads results/*_bibliography_embedded.csv and prints a per-anchor
# coverage table (count, median, p10/p90, top-3 record titles per
# anchor). If a theme that should be ~5-15% of cores is showing 0-1%
# of records, iterate the corresponding anchor in
# scripts/embed_filter.py before paying for labeling. Document each
# anchor edit in docs/ANCHOR_REVISIONS.md (template in that file).

# Stage 3: label (writes results/<slug>_labeled_corpus.csv;
# checkpoints every 50 rows, resumes on rerun)
python scripts/label_with_stanford.py \
    --csv results/<slug>_bibliography_embedded.csv \
    --out results/<slug>_labeled_corpus.csv \
    --min-score 0.45

# Alternative: label with local Ollama instead of the Stanford gateway
python scripts/label_with_stanford.py \
    --csv results/<slug>_bibliography_embedded.csv \
    --out results/<slug>_labeled_corpus.csv \
    --label-backend ollama \
    --ollama-host http://localhost:11434 \
    --model llama3.1:8b-instruct-q4_K_M \
    --min-score 0.45
```

Wallclock for a typical 10k–20k record corpus: ~1 minute harvest per
source (rate-limited), ~5 min embedding, ~1–2 hours labeling at the
default 0.45 cutoff. Embed is local (Ollama) so free; label is
network + API cost.

### Step 3a — pilot 50 records before the full run

**Always do a 50-record pilot before launching the full label.**
Bugs in `_row_prompt`, the SYSTEM_PROMPT schema, or the column-write
logic surface reliably at 50 records and are 1000× cheaper to fix
here than after a multi-thousand-record full run.

```bash
python scripts/label_with_stanford.py \
    --csv results/<slug>_bibliography_embedded.csv \
    --out results/<slug>_pilot50_labeled.csv \
    --limit 50 \
    --min-score 0.45
```

Spot-check the pilot output:

- All 13 schema fields populate (no constant `none` columns).
- `relevance_llm` distribution looks roughly: 15-25% core, 50-70%
  adjacent, 5-25% off-topic.
- The `label_rationale` cites signals from the back half of
  abstracts (numbers, results sentences) — confirming the LLM is
  using the full abstract, not just the title.
- Off-topic rejections are genuinely off-topic (sanity-check 5).
- Schema enums are honored (no invented values).

If anything looks off, fix it before the full run; the checkpointing
means a Ctrl-C costs at most one chunk, but a wrong schema costs
the entire spend.

### Multi-theme corpora: do not gate on a single regex column

For single-theme searches (where the topic regex *is* the scope —
all native-sand records mention sand grains, all reworming-lit
records mention earthworms), `--require-column <topic>_mentioned`
is a useful pre-filter that cuts label spend by 5-10×.

For multi-theme corpora that legitimately span literatures (e.g.
worm-tea-lit needs cost / MRV / registry-methodology papers that
do not contain "earthworm" in the abstract), the hard gate is
wrong. Use `--min-score` alone and let the labeler's IN-SCOPE /
OUT-OF-SCOPE rules in SYSTEM_PROMPT do the topic exclusion.

The `embed_top_anchor` distribution will be noisy whenever two
anchors overlap (priming/permanence, TEA/cost). That's expected
and fine for filtering — it's a signal aid, not a ground-truth
theme assignment. The LLM's `facet_llm` is authoritative.

### Gap-fill harvests

If a source skipped during the first harvest (missing API key,
quota exhausted), use a separate `--output` directory then merge:

```bash
# Re-harvest just the missing source into a sibling dir
python <slug>_search.py --email you@example.com \
    --sources wos_expanded --output results_wos_expanded

# Dedup-merge into the main bibliography (writes results/<slug>_bibliography.csv
# in place, copies raw/ JSONs across)
python scripts/merge_gap_fill.py --gap-dir results_wos_expanded

# Re-embed (cache skips already-encoded rows)
python scripts/embed_filter.py \
    --csv results/<slug>_bibliography.csv \
    --out results/<slug>_bibliography_embedded.csv
```

## Step 4 — iterate

After the first labeled-corpus pass:

1. **Audit.** Read `relevance_llm` and `parent_lithology_llm` (or your
   topic's equivalent) value counts. Check the size of each
   sub-cluster against your expectation.
2. **Identify gaps.** If a sub-topic you expect to be ~5 % of cores
   is showing 0–1 %, draft a Group F-style gap-fill query batch in
   `queries.py` (see `WOS_EXPANDED_GAP` in this project for the
   pattern).
3. **Re-run only the new queries.** `scripts/wos_gap_fill.py` is the
   reusable harness; it fetches, embeds, labels, and writes a side-
   channel CSV without touching the main corpus.
4. **Merge when satisfied.** Append the gap-fill rows to
   `<slug>_labeled_corpus.csv` with DOI dedup; re-run embed_filter on
   the merged corpus to extend the embedding matrix; re-run any
   downstream analyses.

## Step 5 — back up

The pipeline produces large derivative artifacts (~200 MB embedded
CSV, ~80 MB .npy matrix, ~200 MB labeled corpus on a 20k-row run).
None of these belong in git. Push to your rclone remote after every
major step:

```bash
rclone copy results/<slug>_labeled_corpus.csv su-drive:<slug>/$(date +%F)/
rclone copy results/<slug>_bibliography_embedded.csv su-drive:<slug>/$(date +%F)/
rclone copy results/<slug>_bibliography_embedded.embeddings.npy su-drive:<slug>/$(date +%F)/
rclone hashsum md5 su-drive:<slug>/$(date +%F)/<slug>_labeled_corpus.csv  # verify
```

The per-query JSON cache in `results/raw/` is **archived
automatically**: `<slug>_search.py` runs `scripts/disk_hygiene.py`
at the end of every successful run (`--cleanup`, on by default),
which parquet-archives `results/raw/` to
`results/archive/raw_archive.parquet` (zstd-9, md5-verified) and
then deletes `results/raw/`. Pass `--no-cleanup` to keep
`results/raw/` (e.g. before a backfill or gap-fill that re-reads
the JSON cache). To archive manually or after a `--no-cleanup`
run:

```bash
python scripts/disk_hygiene.py --results results            # archive + delete raw/
python scripts/disk_hygiene.py --results results --no-delete # archive but keep raw/
rclone copy results/archive/raw_archive.parquet su-drive:<slug>/$(date +%F)/
```

The shared `.gitignore` template already excludes `results/raw_*/`
and `results/raw_*.parquet` so you don't accidentally check
archives in. See [DISK_HYGIENE.md](DISK_HYGIENE.md) for the full
norms.

> **Migrating an older project's layout.** Projects scaffolded
> before the subdir layout (`results/logs/`, `archive/`,
> `gapfills/`, `pilots/`, `analysis/`) can adopt it with
> `python scripts/migrate_layout.py <project_root>` (dry-run,
> prints the planned moves) then re-run with `--apply`. It only
> moves cruft (logs, `.bak`, `raw_archive*.parquet`,
> pilot/analysis/gap-fill CSVs) and never touches the canonical
> `*_bibliography*` / `*_labeled_corpus*` / embeddings /
> checkpoint files.

## Step 6 — (optional) cross-project bridge

If you have two related projects, both encoded with the same Ollama
BGE-M3, you can compute a semantic bridge between their core sets.
See `docs/cross_project_bridging_methods.md` for the full
methodology. The script:

```bash
python scripts/cross_project_bridge.py \
    --ns-root /path/to/project-A \
    --mt-root /path/to/project-B \
    --k 5 \
    --out results/cross_project_bridges_$(date +%F).csv
```

Output: one row per (source core, top-k destination neighbor) pair
in both directions, with cosine, relevance labels, and per-project
facet tag strings.

## Maintenance

- The shared infrastructure is **byte-copied at scaffold time** from
  litsweep, not symlinked or imported. Defensive fixes that come out
  of using a project (e.g. `_wos_exp_path` defensive helpers,
  `_row_prompt` snippet-vs-abstract preference) should land **in
  litsweep first** as the canonical source of truth, then propagate
  to existing projects only when those projects do an iteration that
  needs the fix.
- The four already-deployed projects (native-sand, reworming-lit,
  microtexture-lit-search, worm-tea-lit) are pre-litsweep and have
  copy-pasted infra. They keep running as-is until a maintainer
  needs a fix; then either cherry-pick the fix from litsweep or
  re-scaffold and migrate the topic-specific files. Don't bulk-
  migrate them.
- A future V2 of litsweep should make the shared code a real
  `pip install -e litsweep` package so projects import
  `from litsweep.api_clients import …` rather than byte-copying.
  That's a real refactor (touches all existing projects) and is
  out of scope for V1; tracked as a roadmap item in
  litsweep's README.