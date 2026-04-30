# Disk hygiene

Lit-search projects accumulate a lot of artifacts: per-query JSON
caches (often 1-2 GB), embedding matrices (~80 MB per project), labeled
corpora (~50-200 MB), checkpoint chunks. This doc is the canonical
"what do I delete, what do I keep, what do I push" guide.

Goal: every active project should be **under 500 MB on disk** and
every archived project should be **under 50 MB on disk** (with the
heavy artifacts living on `rclone:su-drive`).

---

## Per-stage hygiene

### After harvest (`<slug>_search.py`)

The harvest writes:

- `results/<slug>_bibliography.csv` (text, small, ~5-50 MB)
- `results/<slug>_bibliography.bib` (text, similar size)
- `results/raw/*.json` (one file per query × source; **gigabytes**)

Push the bibliography CSV+BIB to rclone immediately so we have the
canonical post-dedup state checkpointed, then **archive raw/ to a
parquet** before doing anything else. Parquet+zstd compresses the
JSON cache ~10×.

```bash
DATE=$(date +%F)
SLUG=$(basename results/*_bibliography.csv | cut -d_ -f1-2)  # extract slug

# 1. Push bibliography artifacts
rclone copy results/${SLUG}_bibliography.csv  su-drive:${SLUG}/${DATE}/
rclone copy results/${SLUG}_bibliography.bib  su-drive:${SLUG}/${DATE}/

# 2. Compress raw/ to a single parquet
python -c "
import pandas as pd, json, pathlib
rows = []
for fp in pathlib.Path('results/raw').glob('*.json'):
    rows.append({'source': fp.stem.split('__', 1)[0],
                 'query':  fp.stem.split('__', 1)[-1],
                 'payload_json': fp.read_text()})
pd.DataFrame(rows).to_parquet(
    'results/raw_archive.parquet',
    compression='zstd', compression_level=9)
print(f'archived {len(rows)} JSON files')
"

# 3. Push the parquet, then verify checksum, then delete raw/
rclone copy results/raw_archive.parquet su-drive:${SLUG}/${DATE}/
rclone hashsum md5 su-drive:${SLUG}/${DATE}/raw_archive.parquet
md5 results/raw_archive.parquet  # macOS; sha256sum on Linux
# if both match:
rm -rf results/raw
```

If a backfill run is needed later (`scripts/backfill_abstracts.py`
re-extracts abstracts from the cache), download the parquet, expand
back to JSON, run, re-archive.

### After embed (`scripts/embed_filter.py`)

Adds:

- `results/<slug>_bibliography_embedded.csv` (~1.2× bibliography size)
- `results/<slug>_bibliography_embedded.embeddings.npy` (~80 MB
  for 30k records × 1024 dims × float32)
- `results/<slug>_bibliography_embedded.embeddings.ids.txt`

Push all three to rclone — the `.npy` is expensive to recompute
(Ollama is local but takes ~15 min for 30k records).

```bash
rclone copy results/${SLUG}_bibliography_embedded.csv               su-drive:${SLUG}/${DATE}/
rclone copy results/${SLUG}_bibliography_embedded.embeddings.npy    su-drive:${SLUG}/${DATE}/
rclone copy results/${SLUG}_bibliography_embedded.embeddings.ids.txt su-drive:${SLUG}/${DATE}/
```

### After label (`scripts/label_with_stanford.py`)

Adds:

- `results/<slug>_labeled_corpus.csv` (~1.3× embedded size)
- `results/<slug>_labeled_corpus.checkpoints/chunk_*.csv` (per-50
  rows; ~600 chunks for a full 30k run)

Push the labeled corpus, then **delete the checkpoints** once the
final CSV is written.

```bash
rclone copy results/${SLUG}_labeled_corpus.csv su-drive:${SLUG}/${DATE}/
rm -rf results/${SLUG}_labeled_corpus.checkpoints
```

The checkpoints are only useful for resuming a partial run; once
the final CSV exists, they're dead weight. Keeping them around
costs 50-200 MB of CSV chunks for nothing.

---

## What to keep on local disk

For an **active project** (under iteration):

- All Python code (`*.py`, `docs/`, `requirements.txt`, etc.)
- Latest `*_bibliography.csv`, `*_embedded.csv`, `*_labeled_corpus.csv`
- Latest `*.embeddings.npy` (only if doing embed-dependent work this
  week; otherwise keep on rclone)
- Latest `raw_archive.parquet` (compact, ~50-200 MB)

Delete:

- `results/raw/` after parquet archive
- `*.checkpoints/` after final labeled CSV is written
- Old `*.log` files older than the last clean run
- Stale intermediate CSVs from prior iterations (keep only the
  current and one previous)

For an **archived project** (no edits in >30 days):

- Keep the git repo (code + docs only) on local disk
- Keep one snapshot of the labeled corpus on local disk
- Move everything else to rclone-only

---

## rclone layout convention

```
su-drive:
└── <slug>/
    └── YYYY-MM-DD/                 # one snapshot per major stage
        ├── <slug>_bibliography.csv
        ├── <slug>_bibliography.bib
        ├── raw_archive.parquet
        ├── <slug>_bibliography_embedded.csv
        ├── <slug>_bibliography_embedded.embeddings.npy
        ├── <slug>_bibliography_embedded.embeddings.ids.txt
        └── <slug>_labeled_corpus.csv
```

Snapshots are immutable. A new harvest = a new dated subfolder.
`rclone hashsum md5` after each push to verify integrity.

---

## Compression and disk budget

Rough sizes for a 30k-record project:

| Artifact | Uncompressed | Compressed | Lives where |
|---|--:|--:|---|
| `raw/` JSON cache | 1.5-2.5 GB | — | local during stage, then deleted |
| `raw_archive.parquet` (zstd) | — | 100-200 MB | rclone + local |
| `<slug>_bibliography.csv` | 50-150 MB | — | rclone + local |
| `<slug>_bibliography_embedded.csv` | 60-180 MB | — | rclone + local |
| `*.embeddings.npy` (float32) | 80-150 MB | — | rclone + local (during use) |
| `<slug>_labeled_corpus.csv` | 70-200 MB | — | rclone + local |
| Checkpoint chunks | 50-200 MB | — | local only, delete after |

**Active project on-disk budget: ~500 MB** (after parquet archive,
before checkpoint cleanup; ~350 MB after both).

**Archived project on-disk budget: ~50 MB** (just code + one labeled
CSV).

---

## Routine cleanup script

A `scripts/disk_hygiene.py` (TODO: write once the patterns stabilize)
should automate:

1. Detect stale `results/raw/` (parquet exists and is newer): delete.
2. Detect closed checkpoints (final labeled CSV exists and is newer
   than the checkpoint dir): delete.
3. Compress and rclone-push any new `*.csv` / `*.npy` / `*.parquet`
   that doesn't have a matching `su-drive:<slug>/<date>/<file>`.
4. Print a per-project disk report.

Until that script exists, run the per-stage commands above by hand
after every successful pipeline stage. The cost of forgetting is
~1 GB per project per harvest.

---

## Anti-patterns

- **Committing `results/` to git.** Even small CSVs balloon the repo
  history. `results/` is in `.gitignore` for a reason.
- **Keeping multiple `*.embeddings.npy` versions locally.** They
  don't compress well (already binary float32) and a single 80 MB
  file × 4 projects × 3 iterations is over a GB on disk. Keep
  exactly one local copy per project; rclone has the history.
- **Keeping `raw/` after parquet archive.** The parquet has
  everything; the loose JSONs are 10× larger and serve no purpose.
- **Skipping `rclone hashsum md5` verification.** Pushes occasionally
  fail silently (network, quota); always verify before deleting the
  local copy.
