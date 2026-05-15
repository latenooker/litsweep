# Swappable backends, output-layout cleanup, and deploy skills

**Date:** 2026-05-14
**Status:** Draft — awaiting user review
**Scope:** litsweep V1.x (additive; preserves byte-copy scaffold model)

## Goal

Make litsweep shareable with colleagues outside the Stanford ecosystem
by (a) abstracting the embedding and labeling backends behind a small
registry, (b) tidying the per-project file layout that has drifted in
six sister repos, and (c) shipping installable skills for Claude Code
and Codex so a colleague can scaffold and run a first lit search by
asking their agent.

## Non-goals

- Refactoring litsweep into a pip-installable package (V2 roadmap;
  this design preserves byte-copy compatibility).
- Implementing OpenAI, Anthropic, Voyage, Jina, or Cohere backends
  today. The registry shape makes them one-file adds later.
- Renaming the canonical pipeline filenames (`<slug>_bibliography.csv`
  → `<slug>_bibliography_embedded.csv` → `<slug>_labeled_corpus.csv`).
  Cross-project scripts hardcode them; moving is more cost than value.
- Migrating sister projects' historical artifacts en masse. The
  migration script is opt-in per project.

## Audience

The author plus five-to-ten academic colleagues with a mix of access:
some have the Stanford gateway, some only direct API keys, some only
local compute. The design must serve all three.

## Architecture

Three independent slices, each landable in its own PR:

1. **Backend registry layer** (`label_backends.py`, `embed_backends.py`
   at the repo root; `scripts/label_with_stanford.py` and
   `scripts/embed_filter.py` shrink to drivers).
2. **Output layout cleanup** (additive subdirectories under
   `results/`; canonical filenames unchanged; one-shot
   `scripts/migrate_layout.py` for existing projects).
3. **Deploy skills** (`skills/claude/litsweep-deploy/SKILL.md`,
   `skills/codex/litsweep-deploy/SKILL.md`,
   `scripts/install_skills.sh`).

## 1. Backend registry layer

### Files

```
litsweep/
├── label_backends.py    # NEW — registry + StanfordBackend + OllamaLabelBackend
└── embed_backends.py    # NEW — registry + OllamaEmbedBackend
```

Both live at the repo root alongside `api_clients.py` and `dedup.py`,
the established home for slug-agnostic shared infra. They are added to
`SHARED_INFRA` in `scripts/scaffold_new_search.py` (byte-copied without
slug substitution).

### `label_backends.py` shape

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

class LabelBackend(Protocol):
    def call(self, prompt: str, system_prompt: str) -> dict[str, Any]: ...
    def check(self) -> None: ...
    def error_label(self, reason: str) -> dict[str, Any]: ...

@dataclass
class StanfordBackend:
    api_key: str
    api_url: str = "https://aiapi-prod.stanford.edu/v1"
    model: str = "gemini-2.0-flash-lite-001"
    temperature: float = 0.0
    max_tokens: int = 400
    min_interval_s: float = 0.0
    # methods carry the existing call_stanford / check_stanford /
    # _error_label / 429 retry logic verbatim.

@dataclass
class OllamaLabelBackend:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.0
    num_predict: int = 400
    timeout_s: int = 300
    max_retries: int = 3
    # call(): POST /api/chat with format="json" so Ollama enforces
    # JSON-shaped output. Same retry shape as Stanford, no 429
    # handling (local). Same error_label.

BACKENDS: dict[str, type] = {
    "stanford": StanfordBackend,
    "ollama":   OllamaLabelBackend,
}

def make_backend(name: str, **kwargs) -> LabelBackend:
    """Build a backend by name; unknown name → friendly SystemExit."""
    if name not in BACKENDS:
        raise SystemExit(
            f"Unknown label backend: {name!r}. "
            f"Valid: {sorted(BACKENDS)}"
        )
    cls = BACKENDS[name]
    return cls(**{k: v for k, v in kwargs.items() if hasattr(cls, k)})
```

The `make_backend` filter (`hasattr(cls, k)`) means the driver can pass
the *union* of all backends' kwargs (parsed from argparse) and each
backend takes only what it understands. New backends slot in by adding
a dataclass + one `BACKENDS` entry; no driver edits needed.

### `embed_backends.py` shape

Mirror image. Today's only entry is `OllamaEmbedBackend` wrapping the
existing `EmbedConfig` + `ping_ollama` + `_post_batch` + `embed_texts`
logic.

### Driver changes

`scripts/label_with_stanford.py` keeps its filename (sibling projects'
byte-copies still resolve it) and its CLI surface, plus new flags:

```
--label-backend {stanford,ollama}    [default: stanford]
--ollama-host URL                    [default: http://localhost:11434]
--ollama-label-model NAME            [default: llama3.1:8b-instruct-q4_K_M]
```

Internal flow: parse args → `backend = make_backend(args.label_backend,
**kwargs)` → `backend.check()` → loop over rows calling
`backend.call(prompt, SYSTEM_PROMPT)`. The 429 retry, checkpoint resume,
and chunk-save logic all live in the driver; backends do one call and
return parsed JSON or an error dict.

`scripts/embed_filter.py` gets `--embed-backend ollama` (single
choice today; flag exists so future backends slot in without breaking
any CLI invocation).

### Backend selection in practice

The scripts default to `stanford` for label (existing behavior) and
`ollama` for embed. The **skill** is responsible for picking the
right backend at deploy time based on environment:

- `STANFORD_API_KEY` set → recommend `--label-backend stanford`.
- Otherwise → check `OLLAMA_HOST` (or default localhost), confirm
  `ollama list` returns a chat-capable model, recommend
  `--label-backend ollama`.

This keeps the script's CLI explicit (no env-var magic in the
script) while making the agent-driven path frictionless.

### What does NOT change

- `SYSTEM_PROMPT` stays in `label_with_stanford.py` (per-project,
  scaffold-replaces it). The backend receives it as a parameter.
- `ANCHORS` stays in `embed_filter.py` (per-project, scaffold-
  replaces it).
- `_error_label` schema stays in the driver — it's tied to the
  project's `SYSTEM_PROMPT` schema, not the backend.

## 2. Output layout cleanup

### Observed problems (from sister-repo survey, 2026-05-14)

1. **Cruft co-mingled with deliverables.** `*.bak.csv`,
   `_authors_fixed.csv`, versioned checkpoints (`.checkpoints_v1`,
   `.checkpoints_v2`, `_supplementary_chkpt/`), multiple log
   generations (`label.log` + `label_v2.log` + `label_v3.log`).
2. **Gap-fills flat in `results/`.** `wos_gap_records_*`,
   `hawaiian_gap_*`, `inversion_gap_*` clutter the top level.
3. **Analysis outputs mixed in.** `gap_matrix_cells.csv`,
   `cross_project_bridges_*.csv`, `management_levers_*.{png,pdf}`
   alongside pipeline CSVs.
4. **`raw/` + `raw_archive.parquet` duplication.** reworming-lit
   still has 294 unarchived JSON files despite a parquet archive
   sitting next to them.
5. **Naming drift.** `_labeled_corpus.csv` (main) vs `_labeled.csv`
   (gap-fill) for the same shape of output.

### Schema

```
<project>/
├── data/                                  # inputs (unchanged)
├── results/
│   ├── <slug>_bibliography.csv            # canonical pipeline
│   ├── <slug>_bibliography.bib            #   (unchanged filenames
│   ├── <slug>_bibliography_embedded.csv   #    so cross-project tools
│   ├── <slug>_bibliography_embedded.embeddings.npy
│   ├── <slug>_bibliography_embedded.embeddings.ids.txt
│   ├── <slug>_labeled_corpus.csv          #    keep working)
│   ├── <slug>_labeled_corpus.checkpoints/ # auto-deleted on success
│   ├── gapfills/<name>/                   # NEW: side channels
│   │   ├── bibliography.csv
│   │   ├── bibliography_embedded.csv
│   │   └── labeled.csv
│   ├── pilots/                            # NEW: smoke runs
│   │   └── pilot50_labeled.csv
│   ├── analysis/                          # NEW: derived (gap_matrix,
│   │   ├── gap_matrix_cells.csv           #   bridges, figures)
│   │   ├── cross_project_bridges_<date>.csv
│   │   └── *.png, *.pdf
│   ├── archive/                           # NEW: parquet raw, *.bak.*
│   │   └── raw_archive.parquet
│   └── logs/                              # NEW: harvest/embed/label.log
└── scripts/, docs/, ...
```

**Three rules:**

- Top-level canonical filenames inside `results/` **do not move**.
  Cross-project tools (`cross_project_bridge.py`,
  `recover_source_databases_from_bib.py`, gap-matrix scripts) keep
  reading from the same paths.
- New subdirectories (`gapfills/`, `pilots/`, `analysis/`,
  `archive/`, `logs/`) are additive. The scaffold creates them with
  `.gitkeep` so new projects start clean.
- `results/raw/` is automatically archived to `results/archive/
  raw_archive.parquet` and deleted at end of pipeline when
  `--cleanup` is set (default on; opt-out with `--no-cleanup`).

### Scaffold changes

`scripts/scaffold_new_search.py` adds the five new directories to its
tree-creation step. No filename substitution needed (they're
slug-agnostic). One line added to README/CLAUDE templates noting where
gap-fills and analysis live.

### Orchestrator and script changes

- `litsweep_search.py`: default log path becomes
  `results/logs/harvest.log` (was `results/errors.log`). The
  `--output` flag still resolves the same way; only the log
  sub-path moves.
- `embed_filter.py` / `label_with_stanford.py`: gain `--log-dir`
  (default `results/logs/`).
- `wos_gap_fill.py`, `merge_gap_fill.py`: default
  `--gap-dir results/gapfills/<name>` instead of a sibling
  top-level dir.
- New `scripts/disk_hygiene.py`: parquet-archive `results/raw/`
  with zstd-9, md5-verify, delete the JSON directory. Called by
  the orchestrator as its last step under `--cleanup`. Standalone
  invocation supported.

### Migration script for existing projects

`scripts/migrate_layout.py` (slug-agnostic, idempotent):

```
Usage: python scripts/migrate_layout.py [project_root] [--apply]

Without --apply: prints a dry-run plan.
With --apply:    performs the moves.

Moves:
  results/*.log                  → results/logs/
  results/*.bak.*                → results/archive/
  results/raw_archive*.parquet   → results/archive/
  results/*_gap_*.csv            → results/gapfills/<inferred_name>/
  results/pilot*_labeled*.csv    → results/pilots/
  results/*.{png,pdf}            → results/analysis/
  results/gap_matrix_*.csv       → results/analysis/
  results/cross_project_*.csv    → results/analysis/
  results/coverage_matrix_*.csv  → results/analysis/

Untouched:
  results/<slug>_bibliography*.csv
  results/<slug>_bibliography*.bib
  results/<slug>_bibliography*.embeddings.{npy,ids.txt}
  results/<slug>_labeled_corpus.csv
  results/<slug>_labeled_corpus.checkpoints/
```

Sister projects adopt at their own pace. The byte-copy model is
preserved: existing copies of scripts still write to the old
top-level paths until they're re-scaffolded or hand-migrated.

## 3. Deploy skills

### File layout

```
litsweep/
├── skills/
│   ├── claude/litsweep-deploy/
│   │   └── SKILL.md
│   └── codex/litsweep-deploy/
│       └── SKILL.md
└── scripts/install_skills.sh
```

Two SKILL.md files share their body content; only the YAML
frontmatter differs (CC uses `allowed-tools:`, Codex uses its own
format). At two files the cost of dual-edit is lower than building a
templating system.

### Skill body outline

The skill is tightly scoped to **first-deploy through first labeled
corpus**, matching how the user actually uses litsweep:

1. **Trigger description** — fire when the user says any of: "set up
   a new lit search," "scaffold a litsweep project," "deploy
   litsweep," "start a new bibliographic search."
2. **Where to scaffold** — emphatically *outside* the litsweep repo
   (sibling directory). The scaffold script is invoked from
   `/path/to/litsweep/scripts/scaffold_new_search.py`.
3. **Prerequisite check (agent-side)** — verify Python 3.11+,
   Ollama daemon reachable, `bge-m3` pulled. Detect Stanford key
   vs. Ollama-only and pick the label backend accordingly. Skip
   the `gh repo create` step if `gh` isn't auth'd.
4. **Run the scaffold** with the right `--from-existing-corpus`
   flags if the user mentions sibling projects.
5. **Walk the four topic files** in sequence: `queries.py`,
   `vocab.py`, `ANCHORS` in `embed_filter.py`, `SYSTEM_PROMPT` in
   `label_with_stanford.py`. The skill includes concrete prompts
   the agent should produce for each — these are the
   high-difficulty bits where colleagues need the most help.
6. **Smoke test** — `python <slug>_search.py --dry-run`.
7. **First harvest** — `python <slug>_search.py
   --email <user-email>`.
8. **Embed** — `python scripts/embed_filter.py …`.
9. **Anchor diagnostic** — `python scripts/embed_diagnostic.py
   --markdown docs/anchor_coverage_$(date +%F).md`. Iterate anchors
   if coverage is thin before paying for labeling.
10. **50-record pilot** — `python scripts/label_with_stanford.py
    --limit 50 …`. Spot-check schema and `relevance_llm` distribution.
11. **Full label** — same command without `--limit`.
12. **Post-run hygiene** — parquet-archive `results/raw/` (or trust
    `--cleanup`), rclone push the labeled corpus + embeddings to
    user's remote, verify md5.
13. **Common failures** — 429 rate limits (`--min-interval-s 1.5`),
    missing abstracts (`scripts/backfill_abstracts.py`), embed cache
    shape mismatch.

The skill body should aim for ≤2000 tokens. Detail beyond that lives
in `docs/DEPLOYING_A_NEW_SEARCH.md`, which the skill links to.

### `scripts/install_skills.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for tool in claude codex; do
    skill_src="$REPO_ROOT/skills/$tool/litsweep-deploy"
    skill_dst="$HOME/.${tool}/skills/litsweep-deploy"
    if [[ ! -d "$skill_src" ]]; then
        echo "missing source: $skill_src" >&2
        continue
    fi
    mkdir -p "$(dirname "$skill_dst")"
    ln -sfn "$skill_src" "$skill_dst"
    echo "linked $skill_dst -> $skill_src"
done
echo "done. Restart your agent CLI to pick up new skills."
```

Idempotent (`-sfn` replaces existing symlinks atomically). Runs as
`bash scripts/install_skills.sh` after `git clone litsweep`.

## 4. Testing

Three smoke tests added to `tests/`:

1. **Scaffold smoke** (`tests/test_scaffold_smoke.py`). Scaffolds
   into `tmp_path`, runs `<slug>_search.py --dry-run`, asserts exit
   code 0 and presence of `results/{gapfills,pilots,analysis,
   archive,logs}/` and the two new shared-infra files.
2. **Backend dispatch** (`tests/test_backends.py`). For each entry
   in `label_backends.BACKENDS` and `embed_backends.BACKENDS`,
   instantiate with minimal kwargs and confirm `.check()` either
   succeeds (mocked HTTP) or raises `SystemExit` with a helpful
   message. No real network.
3. **Migration script** (`tests/test_migrate_layout.py`). Builds a
   fixture `results/` mirroring the messes catalogued in the
   sister-repo survey (versioned checkpoints, unfolded gap-fills,
   top-level logs, raw/ + raw_archive.parquet). Runs
   `migrate_layout.py --apply`, asserts canonical files unmoved and
   cruft sorted into the right subdirs. Re-runs the script to
   confirm idempotency.

CLAUDE.md's "smoke-test the scaffold after every infra edit"
discipline is now codified rather than convention.

## 5. Back-compat and rollout

- **Defaults preserve current behavior.** `--label-backend stanford`
  is the default. Every existing invocation across the four
  pre-litsweep projects continues to work after they re-scaffold or
  hand-port the changes.
- **No filename rewrites in `results/`.** Cross-project scripts that
  hardcode `results/<slug>_bibliography.csv` keep working.
- **New directories are additive.** A sister project that ignores
  `gapfills/`, `pilots/`, etc., keeps running; nothing in the core
  pipeline reads from them.
- **Migration is opt-in.** `scripts/migrate_layout.py` exists; sister
  projects run it at the maintainer's discretion. No bulk migration.
- **The labeler script keeps its filename** (`label_with_stanford.py`)
  even though it's no longer Stanford-specific. Renaming breaks
  every cross-project reference, every commit message, every doc
  link. Cost-benefit doesn't justify it. Future scaffold could
  alias to `scripts/label.py`; out of scope here.

## 6. Resolved design points

- **Checkpoint cleanup.** `label_with_stanford.py` deletes
  `<out>.checkpoints/` after the merged CSV writes successfully.
  Checkpoints are crash insurance, not deliverables; a successful
  write means they've served their purpose. A failed write leaves
  them in place so a resume picks up where it stopped.
- **Skill install mechanism.** Symlink (not copy). Upgrades reach
  every colleague's agent on `git pull`. The tradeoff —
  symlinks break if the litsweep checkout moves — is acceptable
  because re-running `bash scripts/install_skills.sh` is the fix.
  Documented in `install_skills.sh` itself.
- **Empty backend stubs (Voyage, OpenAI, Anthropic).** Not shipped.
  Dead code diverges from real implementations. The registry shape
  is documented in this spec and demonstrated by the two real
  backends; that's the extension point.

## 7. Roll-out order

Three PRs in this order:

1. **Backend registry** (`label_backends.py`, `embed_backends.py`,
   driver shrinks, `--label-backend` flag, `OllamaLabelBackend`).
   Lowest blast radius; all defaults unchanged.
2. **Output layout** (additive subdirs in scaffold, `disk_hygiene.py`,
   `migrate_layout.py`, log path defaults). Independent of (1).
3. **Skills** (`skills/`, `install_skills.sh`, skill body referencing
   both backends and the new layout). Depends on (1) and (2) being
   in place because the skill instructs colleagues on the new
   commands.

Each PR ships with the corresponding smoke test from §4.
