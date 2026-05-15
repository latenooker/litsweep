---
name: litsweep-deploy
description: Use when the user asks to "set up a new lit search", "scaffold a litsweep project", "deploy litsweep", "start a new bibliographic search", or describes building a topic-specific multilingual literature corpus. Drives scaffold → fill the 4 topic files → first labeled corpus.
allowed-tools: Bash, Read, Write, Edit, AskUserQuestion
---

# litsweep-deploy

Drive a colleague from a litsweep checkout to a first labeled corpus on
their own topic. litsweep harvests up to 13 bibliographic databases,
dedups, embeds locally with Ollama BGE-M3, and labels with an LLM
(Stanford gateway or local Ollama).

## When to use

- "Set up a new lit search for me on <topic>."
- "Scaffold a litsweep project / deploy litsweep."
- "Start a multilingual bibliography on <topic>."

## When NOT to use

- The project is already scaffolded and the user is editing
  queries/vocab/anchors — just help in-line.
- The user is re-running an existing pipeline — just run the commands.

## Prerequisite check (do this FIRST, before scaffolding)

Run and report what's missing:

```bash
python3 --version                       # need 3.11+
ollama list 2>/dev/null || echo "no ollama"
echo "STANFORD_API_KEY=${STANFORD_API_KEY:+set}"
which rclone gh 2>/dev/null
```

Pick the label backend from what's available:

- `STANFORD_API_KEY` set → default `--label-backend stanford`.
- No Stanford key but Ollama has a chat model → `--label-backend
  ollama --ollama-host http://localhost:11434 --model <chat_model>`
  (suggest `ollama pull llama3.1` if none).
- Neither → stop; tell the user they need one or the other.

Embedding always needs local Ollama with `bge-m3`
(`ollama pull bge-m3`) — this is required regardless of label backend.

## Scaffold (into a SIBLING dir, never inside litsweep)

```bash
python /path/to/litsweep/scripts/scaffold_new_search.py \
    /path/to/<topic>-lit --name <topic>_lit
```

If a sibling project's corpus overlaps, skip its DOIs:

```bash
python /path/to/litsweep/scripts/scaffold_new_search.py \
    /path/to/<topic>-lit --name <topic>_lit \
    --from-existing-corpus /path/to/sibling/results/<sibling>_labeled_corpus.csv
```

Flags: `--no-remote` (skip GitHub repo), `--public`, `--no-git`.

## Fill the four topic files (the work the scaffold can't do)

All four are TODO-marked; `grep -rn TODO` in the new project. Ask the
user about their topic and draft each:

1. `queries.py` — search strings per database. 10-25 English OpenAlex
   prose queries; mirror into French (HAL/TEL/theses.fr), Portuguese
   (BDTD/SciELO), Spanish (SciELO); WoS uses `TS=(...)` syntax.
2. `vocab.py` — multilingual regex axes via the `VOCAB_AXES` registry.
3. `scripts/embed_filter.py` :: `ANCHORS` — 6-10 prose facet
   descriptions (1-3 sentences each); include one multilingual
   umbrella anchor (BGE-M3 is multilingual).
4. `scripts/label_with_stanford.py` :: `SYSTEM_PROMPT` — strict-JSON
   schema: identity, IN SCOPE, OUT OF SCOPE, "Return ONLY a JSON
   object", enum schema, Rules. The schema is load-bearing; adding a
   field later means re-labeling the whole corpus.

## Run the pipeline

```bash
cd /path/to/<topic>-lit
python -m pip install -r requirements.txt

# 1. Validate queries (no API calls)
python <topic>_lit_search.py --dry-run

# 2. Harvest + dedup (auto-archives results/raw/ at the end;
#    pass --no-cleanup to keep the raw JSON cache)
python <topic>_lit_search.py --email <user_email>

# 3. Embed (local Ollama bge-m3; --embed-backend ollama is the default)
python scripts/embed_filter.py \
    --csv results/<topic>_lit_bibliography.csv \
    --out results/<topic>_lit_bibliography_embedded.csv

# 4. Anchor coverage diagnostic (auto-detects the embedded CSV;
#    no API cost — iterate ANCHORS here before spending on labels)
python scripts/embed_diagnostic.py \
    --markdown docs/anchor_coverage_$(date +%F).md

# 5. 50-record pilot — REQUIRED before the full label run
python scripts/label_with_stanford.py \
    --csv results/<topic>_lit_bibliography_embedded.csv \
    --out results/pilots/pilot50_labeled.csv \
    --limit 50 --min-score 0.45 \
    --label-backend <stanford|ollama>

# Spot-check the pilot, then full label
python scripts/label_with_stanford.py \
    --csv results/<topic>_lit_bibliography_embedded.csv \
    --out results/<topic>_lit_labeled_corpus.csv \
    --min-score 0.45 --label-backend <stanford|ollama>
```

For the ollama backend add `--ollama-host http://localhost:11434
--model llama3.1:8b-instruct-q4_K_M`; add `--min-interval-s 2` if the
gateway/daemon starts refusing requests.

Note: `embed_filter.py` uses `--host` (not `--ollama-host`) to
override the Ollama endpoint for embedding:

```bash
python scripts/embed_filter.py \
    --csv results/<topic>_lit_bibliography.csv \
    --out results/<topic>_lit_bibliography_embedded.csv \
    --host http://localhost:11434    # only needed if not default
```

## Post-run

`<topic>_lit_search.py` already parquet-archives and deletes
`results/raw/` (unless `--no-cleanup`). Push deliverables to the
user's rclone remote (ask for it: `rclone listremotes`):

```bash
rclone copy results/<topic>_lit_labeled_corpus.csv <remote>:<topic>_lit/$(date +%F)/
rclone copy results/<topic>_lit_bibliography_embedded.csv <remote>:<topic>_lit/$(date +%F)/
rclone copy results/<topic>_lit_bibliography_embedded.embeddings.npy <remote>:<topic>_lit/$(date +%F)/
rclone copy results/archive/raw_archive.parquet <remote>:<topic>_lit/$(date +%F)/
rclone hashsum md5 <remote>:<topic>_lit/$(date +%F)/<topic>_lit_labeled_corpus.csv
```

## Common failures

- 429s on the Stanford label: add `--min-interval-s 1.5`.
- Many empty abstracts: `python scripts/backfill_abstracts.py`.
- Embed cache shape mismatch: delete
  `results/*_bibliography_embedded.embeddings.npy` and
  `.embeddings.ids.txt`, re-run `embed_filter.py`.
- Ollama model not pulled: `ollama pull <model>`.
- Stanford key invalid: confirm `echo $STANFORD_API_KEY` in the
  active shell (a `~/.zshrc` export needs a new shell).

## Hand-off

When the labeled corpus is on disk and verified on the remote, point
the user at `docs/DEPLOYING_A_NEW_SEARCH.md` for iteration patterns
(gap-fills, cross-project bridges, anchor revisions).
