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
├── <slug>_search.py                  # orchestrator: queries -> dedup -> augment
├── queries.py                        # ★ topic-specific: search strings per source
├── vocab.py                          # ★ topic-specific: VOCAB_AXES + TOPIC_PRESENCE
├── scripts/                          # all templated shared scripts (per litsweep)
│   ├── embed_filter.py               # ★ topic-specific: ANCHORS list inside
│   ├── label_with_stanford.py        # ★ topic-specific: SYSTEM_PROMPT inside
│   ├── embed_diagnostic.py
│   ├── merge_gap_fill.py
│   ├── backfill_abstracts.py
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
│   ├── <slug>_bibliography.csv               # main corpus (committed via rclone, not git)
│   ├── <slug>_bibliography.bib               # BibTeX export
│   ├── <slug>_bibliography_embedded.csv      # + embed_score, embed_top_anchor
│   ├── <slug>_bibliography_embedded.embeddings.npy   # 1024-d float32 matrix
│   ├── <slug>_bibliography_embedded.embeddings.ids.txt
│   ├── <slug>_labeled_corpus.csv             # + 13 *_llm columns
│   ├── <slug>_labeled_corpus.checkpoints/    # per-50-row chunks
│   ├── raw/                                  # per-query JSON cache (often >1 GB)
│   ├── raw_archive.parquet                   # zstd-compressed cache (10x smaller)
│   ├── *.log                                 # harvest, embed, label logs
│   └── errors.log                            # API failures
└── tests/                                    # (optional) project-specific tests
```

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
| `results/raw_archive.parquet` | | ✓ | compressed raw cache |
| `results/raw/*.json` | | | delete after archiving to parquet |
| `results/*.log` | | | keep until next major rerun |
| `results/*.checkpoints/` | | | delete after final labeled CSV is written |

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
