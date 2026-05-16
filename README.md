# litsweep

Scaffold for multilingual literature-search projects:
harvest from up to 13 bibliographic databases → deduplication → local embedding + filtering around anchor papers → structured labeling (Stanford gateway *or* local Ollama). 

Example use cases:
- Structured identification of data coverage (e.g., publications per cell in covariate space) 
- Simultaneous query of multiple disciplines' treatment of a common phenomenon
- Triage of papers for full download
- Identify bridge themes among multiple corpora (e.g., run litsweep twice and compare pubs in embedding space)

This repo is meant to be a template that is copied for individual lit-search projects. Projects byte-copy the infra at scaffold time — don't
edit them in-place if you want consistency across the family.

## Quick start

```bash
python /path/to/litsweep/scripts/scaffold_new_search.py \
    /path/to/new-project --name my_topic
```

That single command:

1. Creates the project directory tree per
   [docs/DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md).
2. Byte-copies shared infra; templates the orchestrator and scripts
   with slug substitution; stubs the four topic-specific files.
3. Runs `git init` + the initial commit.
4. **Creates a private GitHub repo via `gh repo create` and pushes**
   the commit. Repo name = slug with underscores → dashes.

Flags:

- `--no-remote` — skip the `gh repo create` step (offline scaffold).
- `--public` — create the repo as public (default is private).
- `--remote-owner <org>` — owner override (default: your gh user).
- `--no-git` — skip the entire git workflow.

Then fill in the four topic-specific files (queries, vocab, ANCHORS,
SYSTEM_PROMPT) per [docs/DEPLOYING_A_NEW_SEARCH.md](docs/DEPLOYING_A_NEW_SEARCH.md).

## Read these three docs first

1. [docs/DEPLOYING_A_NEW_SEARCH.md](docs/DEPLOYING_A_NEW_SEARCH.md)
   — end-to-end checklist for a new project.
2. [docs/DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md)
   — canonical project layout: which files live at the root, what
   goes in `docs/`, what goes in `results/`.
3. [docs/DISK_HYGIENE.md](docs/DISK_HYGIENE.md)
   — per-stage compression, rclone push norms, what to delete.
   Disciplined cleanup keeps active projects under 500 MB.

## Skills for Claude Code and Codex

litsweep ships installable skills so a colleague can ask their agent
to deploy a new lit search end to end:

```bash
bash scripts/install_skills.sh
```

Symlinks `skills/claude/litsweep-deploy/` into `~/.claude/skills/` and
`skills/codex/litsweep-deploy/` into `~/.codex/skills/`. Idempotent —
re-run after `git pull` to pick up updates, or if you move the
litsweep checkout. An empty placeholder directory at the target is removed and replaced with the symlink; a non-empty real directory is left untouched with an actionable message and that tool is skipped. No directory containing files is ever deleted.

## What's in this repo

```
litsweep/
├── api_clients.py             # 13-source bibliographic API wrappers (shared)
├── dedup.py                   # DOI + Jaccard title-similarity dedup (shared)
├── label_backends.py          # Stanford + Ollama label backends (registry)
├── embed_backends.py          # Ollama embed backend (Voyage/Jina slot in later)
├── litsweep_search.py         # orchestrator template (renamed at scaffold)
├── requirements.txt           # pinned deps
├── scripts/
│   ├── scaffold_new_search.py # → produces a new project from this repo
│   ├── embed_filter.py        # local Ollama BGE-M3 + anchor cosine
│   ├── embed_diagnostic.py    # post-embed coverage table per anchor
│   ├── label_with_stanford.py # Stanford-gateway labeler with strict JSON
│   ├── backfill_abstracts.py  # rebuild abstract column from raw/ JSONs
│   ├── merge_gap_fill.py      # dedup-merge a sibling --output dir
│   ├── disk_hygiene.py        # parquet-archive results/raw/ → archive/ (auto on --cleanup)
│   ├── migrate_layout.py      # opt-in: tidy an existing project's results/ layout
│   ├── install_skills.sh      # symlink the skills into ~/.claude & ~/.codex
│   ├── citation_chase.py
│   ├── wos_gap_fill.py
│   ├── wos_expanded_ping.py
│   ├── screen_wos_export.py
│   ├── compare_wos_to_pipeline.py
│   └── add_openalex_queries.py
├── skills/
│   ├── claude/litsweep-deploy/SKILL.md
│   └── codex/litsweep-deploy/SKILL.md
└── docs/
    ├── DEPLOYING_A_NEW_SEARCH.md
    ├── DIRECTORY_STRUCTURE.md
    ├── DISK_HYGIENE.md
    └── BACKPORTING_NEW_SOURCES.md
```

## Pipeline at a glance

```
queries.py + vocab.py
        │
        ▼
<slug>_search.py            (harvest → spec filter → DOI/title dedup → augment → write CSV
                             → auto-archives results/raw/ at end; pass --no-cleanup to keep)
        │
        ▼
results/<slug>_bibliography.csv
        │
        ▼
scripts/embed_filter.py     (Ollama BGE-M3 → score against ANCHORS → embed_score, embed_top_anchor)
        │
        ▼
scripts/embed_diagnostic.py  ← run before labeling; iterate ANCHORS if needed
        │
        ▼
scripts/label_with_stanford.py  (strict-JSON 13-field schema;
                                  --label-backend stanford|ollama, default stanford;
                                  ollama needs local Ollama + a chat model pulled)
        │
        ▼
results/<slug>_labeled_corpus.csv  ← the deliverable
```

See [docs/DEPLOYING_A_NEW_SEARCH.md](docs/DEPLOYING_A_NEW_SEARCH.md) for the full per-step checklist.

## V1 status (2026-04-29)

Carved out of the native-sand template after 4 sibling projects
established the pattern. V1 is a **byte-copy scaffold**: each project
gets its own copy of `api_clients.py`, `dedup.py`, and the scripts
under `scripts/`. Defensive fixes land in litsweep first, then
propagate to projects on next iteration.

## V2 roadmap

- Refactor the shared code into a `pip install -e litsweep` package
  so new projects do `from litsweep.api_clients import …` and updates
  reach all projects on the next reinstall. Existing projects migrate
  individually.
- Schema centralization: SYSTEM_PROMPT enums + `_error_label` +
  `LABEL_COLUMNS` derived from a single Pydantic / dataclass spec
  in litsweep.
- ~~Automated `scripts/disk_hygiene.py`~~ **Done** — `scripts/disk_hygiene.py`
  enforces [DISK_HYGIENE.md](docs/DISK_HYGIENE.md) policies; auto-invoked by
  `<slug>_search.py --cleanup` (on by default).
- ~~Pluggable label/embed backends~~ **Done** — `label_backends.py` registry
  supports Stanford and Ollama; `embed_backends.py` supports Ollama
  (Voyage/Jina slots reserved).
- Auto-discovery: a `scripts/list_projects.py` that walks
  `~/Documents/projects/*-lit/` and reports per-project disk usage,
  staleness, and the litsweep version each was scaffolded from.

## License

Private. MIT-style for personal use; not intended for redistribution.
