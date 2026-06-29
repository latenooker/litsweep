# Directory structure (canonical project layout)

The norm a litsweep-scaffolded search project follows. Three levels:

1. **Top-level files** — minimal; what the user edits per project.
2. **`docs/`** — the curated artifacts that live in git.
3. **`results/`** — the harvest output, fully gitignored, periodically
   compressed and pushed to `rclone` per
   [DISK_HYGIENE.md](DISK_HYGIENE.md).

```
my-search-lit/                       # repo root; one project = one repo
├── README.md                         # what this corpus covers (link to scope, anchors)
├── CLAUDE.md                         # project-specific guidance for CC
├── .gitignore                        # byte-copied from litsweep
├── requirements.txt                  # pinned deps (byte-copied from litsweep)
├── api_clients.py                    # shared infra (byte-copied)
├── dedup.py                          # shared infra (byte-copied)
├── label_backends.py                 # shared infra: label backend registry (stanford, ollama)
├── embed_backends.py                 # shared infra: embed backend registry (ollama)
├── <slug>_search.py                  # orchestrator: queries -> dedup -> augment
├── queries.py                        # ★ topic-specific: search strings per source
├── vocab.py                          # ★ topic-specific: VOCAB_AXES + TOPIC_PRESENCE
├── scripts/                          # all templated shared scripts (per litsweep)
│   ├── embed_filter.py               # ★ topic-specific: ANCHORS list inside
│   ├── label_with_stanford.py        # ★ topic-specific: SYSTEM_PROMPT inside
│   ├── embed_diagnostic.py
│   ├── build_filter_html.py          # labeled corpus -> self-contained faceted-filter HTML
│   ├── dedup_labeled.py              # post-label cleanup: drop comments/replies + DOI/title collapse
│   ├── merge_gap_fill.py
│   ├── backfill_abstracts.py
│   ├── disk_hygiene.py              # parquet-archive results/raw/ -> archive/
│   ├── migrate_layout.py            # opt-in: tidy an old project's results/
│   ├── citation_chase.py
│   ├── wos_gap_fill.py
│   ├── wos_expanded_ping.py
│   ├── screen_wos_export.py
│   ├── compare_wos_to_pipeline.py
│   └── add_openalex_queries.py
├── docs/
│   ├── SCOPE.md                      # what's in scope vs. out of scope
│   ├── ANCHORS.md                    # seed list of meta-analyses + DOIs
│   ├── ANCHOR_REVISIONS.md           # diff log for embed_filter ANCHORS edits
│   ├── REGISTRY_METHODOLOGIES.md     # (optional) curated side-channel docs
│   ├── REPORTS.md                    # (optional) IPCC / NASEM / IEA chapters
│   └── session_logs/
│       └── claude_session_log_YYYY-MM-DD.md
├── results/                          # all gitignored; see DISK_HYGIENE.md
│   ├── <slug>_bibliography.csv               # canonical pipeline (never moves)
│   ├── <slug>_bibliography.bib               # BibTeX export
│   ├── <slug>_bibliography_embedded.csv      # + embed_score, embed_top_anchor
│   ├── <slug>_bibliography_embedded.embeddings.npy   # 1024-d float32 matrix
│   ├── <slug>_bibliography_embedded.embeddings.ids.txt
│   ├── <slug>_labeled_corpus.csv             # the deliverable
│   ├── <slug>_labeled_corpus.checkpoints/    # per-50-row chunks (auto-deleted on successful write)
│   ├── raw/                                  # per-query JSON cache (parquet-archived + deleted at end of run unless --no-cleanup)
│   ├── gapfills/<name>/                       # gap-fill chains (bibliography.csv, bibliography_embedded.csv, labeled.csv)
│   ├── pilots/                                # smoke-test outputs (e.g. pilot50_labeled.csv)
│   ├── analysis/                              # derived artifacts (gap_matrix, cross-project bridges, *.png/*.pdf)
│   ├── archive/                               # raw_archive.parquet, *.bak files
│   └── logs/                                  # harvest/embed/label logs, errors.log
└── tests/                                    # (optional) project-specific tests
```

The top-level canonical filenames inside `results/`
(`<slug>_bibliography*`, `<slug>_labeled_corpus*`, the
`*.embeddings.*` matrices, the `*.checkpoints/` dir) **never
change** — cross-project tools hardcode them. The
`gapfills/`, `pilots/`, `analysis/`, `archive/`, and `logs/`
subdirs are *additive*: new projects get them from the
scaffold; existing pre-layout projects adopt them by running
`python scripts/migrate_layout.py <project_root>` (dry-run,
prints the planned moves) and then re-running with `--apply`.
The migration only moves cruft (`*.log`, `*.bak`,
`raw_archive*.parquet`, pilot/analysis/gap-fill CSVs) into the
new subdirs; it never touches the canonical filenames above.

★ = the four files the user actually edits per project. Everything
else is litsweep infrastructure.

## What goes in git vs. rclone vs. nowhere

| Artifact | git | rclone (`su-drive:<slug>/<date>/`) | Note |
|---|:---:|:---:|---|
| `README.md`, `CLAUDE.md`, `requirements.txt` | ✓ | | text, small |
| `queries.py`, `vocab.py`, `<slug>_search.py` | ✓ | | the topic-specific edits |
| `api_clients.py`, `dedup.py`, `scripts/*.py` | ✓ | | byte-copied infra |
| `docs/SCOPE.md`, `ANCHORS.md`, `ANCHOR_REVISIONS.md` | ✓ | | curated, the user reads these |
| `docs/session_logs/*.md` | ✓ | | per-session summary |
| `results/<slug>_bibliography.csv` | | ✓ | the main artifact, often >50 MB |
| `results/<slug>_bibliography_embedded.csv` | | ✓ | larger, +embedded scores |
| `results/<slug>_labeled_corpus.csv` | | ✓ | the deliverable |
| `results/*.embeddings.npy` | | ✓ | expensive to recompute (Ollama) |
| `results/archive/raw_archive.parquet` | | ✓ | compressed raw cache |
| `results/raw/*.json` | | | auto-archived to parquet + deleted at end of run |
| `results/logs/*.log` | | | keep until next major rerun |
| `results/*.checkpoints/` | | | auto-deleted after final labeled CSV is written |

## Why this shape

- **One project, one repo.** Search scopes don't share code; copy-on-
  scaffold is simpler than a multi-project monorepo.
- **Topic-specific files at the root.** When you open the repo, the
  four files that need editing per project are the first thing you
  see. Everything in `scripts/` is shared infra; everything in
  `docs/` is curated context; everything in `results/` is generated.
- **`results/` is gitignored entirely.** Generated artifacts
  shouldn't pollute git history. Treat the `*.parquet` and
  `.embeddings.npy` as the durable outputs (rclone), and the `*.csv`
  as the human-readable deliverables (rclone, optionally also
  versioned in git for small ones).
- **`docs/session_logs/`** captures per-session decisions in markdown.
  PAWPAWS-aligned: a YYYY-MM-DD log per active day. Better than
  burying decisions in commit messages.
- **`docs/ANCHOR_REVISIONS.md`** is *the* most important durable
  artifact for a multi-theme search. Every embed-anchor edit goes
  there with a before/after distribution snapshot. Without it,
  anchor iterations are non-reproducible.

## When to deviate

- Single-theme projects (native-sand, microtexture-lit-search) can
  skip `ANCHOR_REVISIONS.md` and the multi-theme docs.
- Projects with custom post-corpus analyses (e.g. `gap_matrix.py`,
  cross-project bridges) put those scripts in `scripts/` next to
  the litsweep-shared ones. If they accumulate, group under
  `scripts/analysis/`.
- Projects that want PDF ingestion (registry methodologies, IPCC
  chapters) get a `pdfs/` directory at the root, gitignored, with a
  `docs/PDFS.md` manifest of the URLs and last-checked dates.
