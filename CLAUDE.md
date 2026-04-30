# CLAUDE.md — litsweep

Project-specific guidance for Claude Code working in this repository.

## What this is

`litsweep` is the canonical scaffold for multilingual literature-
search projects. It carves out the shared infrastructure
(`api_clients.py`, `dedup.py`, the scripts under `scripts/`, the
orchestrator template `litsweep_search.py`) so individual search
projects (native-sand, reworming-lit, microtexture-lit-search,
worm-tea-lit, future ones) start from a consistent base.

## Key rules

1. **Source of truth.** Every defensive fix or improvement to the
   shared infra belongs **here first**. Sibling projects byte-copy
   at scaffold time and accumulate divergence; if you patch a bug
   directly in a sibling project, it never reaches new projects.
2. **Test the scaffold after every infra edit.** Run
   `python scripts/scaffold_new_search.py /tmp/test_scaffold --name foo_lit --no-git`,
   inspect the result, run `python /tmp/test_scaffold/foo_lit_search.py --dry-run`,
   delete `/tmp/test_scaffold`. Catches regressions in the
   substitution / docstring-replace / vocab-stub paths.
3. **Don't introduce topic-specific code into shared files.**
   `api_clients.py`, `dedup.py`, and the `scripts/*.py` should be
   topic-agnostic. The orchestrator template (`litsweep_search.py`)
   should also stay generic — the only project-specific bits in it
   are the `_priority_score` heuristics, which use VOCAB_AXES
   abstractly without naming any axis.
4. **Don't break backwards compatibility unannounced.** The four
   existing projects (native-sand and friends) consume the litsweep
   scripts via byte-copy on a date in the past; their copies don't
   update automatically. But cross-project tools — like the cross-
   project bridge in some siblings — sometimes import each other's
   code. Be explicit about deprecations in commit messages.
5. **Update docs alongside code.** `DEPLOYING_A_NEW_SEARCH.md`,
   `DIRECTORY_STRUCTURE.md`, and `DISK_HYGIENE.md` are load-bearing.
   When you change scaffold behavior or add a script, update the
   relevant doc in the same commit.

## File map

| File | Edit when |
|---|---|
| `litsweep_search.py` | Orchestrator logic changes (harvest, dedup, augment, write). Avoid topic-specific edits. |
| `api_clients.py` | New API source, defensive parser, rate-limit fix. |
| `dedup.py` | Rare; only if dedup heuristics need tuning. |
| `scripts/scaffold_new_search.py` | Scaffold steps, stub content, TEMPLATED_FILES. |
| `scripts/embed_filter.py` | BGE-M3 / Ollama protocol, score calculation. ANCHORS get *replaced* per project. |
| `scripts/label_with_stanford.py` | Stanford gateway protocol, prompt assembly, schema. SYSTEM_PROMPT and `_error_label` schema get *replaced* per project. |
| `scripts/embed_diagnostic.py` | Diagnostic table format, anchor labeling. |
| `scripts/merge_gap_fill.py` | Gap-fill merge logic. |
| `scripts/backfill_abstracts.py` | Abstract-rebuild from raw/ caches. |
| `docs/DEPLOYING_A_NEW_SEARCH.md` | The full per-project checklist. |
| `docs/DIRECTORY_STRUCTURE.md` | The canonical project layout. |
| `docs/DISK_HYGIENE.md` | Compression, rclone push norms. |

## Conventions

- Python 3.11+. `from __future__ import annotations` at the top of
  every module. Modern type hints (`str | None`, `list[str]`).
- Google-style docstrings on public functions and classes.
- Private helpers prefixed with `_`. Module-level constants
  `UPPER_CASE`.
- Prefer the standard library; deps pinned in `requirements.txt`.
- No async; APIs are rate-limited and serial calls keep things
  simple.

## When adding a new shared script

1. Write the script in `scripts/` so it's slug-agnostic — accept
   inputs via `--csv`, `--out`, `--gap-dir`, etc., and auto-detect
   single matches in `results/` rather than hardcoding project
   filenames.
2. Add it to `TEMPLATED_FILES` in `scripts/scaffold_new_search.py`
   so future scaffolds include it automatically.
3. Update `README.md` and `docs/DEPLOYING_A_NEW_SEARCH.md`'s
   pipeline diagram if the script becomes part of the canonical
   flow.
4. Smoke-test the scaffold: re-run it into `/tmp` and confirm
   the new script appears.

## When adding a vocab axis pattern (B2/B3)

Already done in V1. Future axes get added via `VOCAB_AXES` in a
project's own `vocab.py`; no edits to `litsweep_search.py` or
`CSV_COLUMNS` are needed because the orchestrator iterates the
registry and the CSV write step auto-extends columns.

## V2 roadmap reference

See README.md "V2 roadmap" section for the Phase 2 work:
pip-installable package, schema centralization, disk hygiene
automation, project auto-discovery. Track these as separate PRs/
commits, not bundled with the V1 stabilization.
