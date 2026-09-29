# Dedup refinements (flagged, not implemented)

*Flagged 2026-09-29 from the tracemakers-lit `soil-image-analysis`
sweep (11,664 records after dedup). This lists what `dedup.py` misses
today and proposes fixes. Nothing here has been changed in code yet.*

## Size of the problem

The test grouped records whose titles match after normalization (NFKC,
case-folded, alphanumerics only, at least 20 characters) and whose
years differ by 1 or less.

- **Result:** 67 groups, with **70 surplus rows (0.6% of the corpus)**.
- **Effect:** small for recall statistics. The duplicates are visible,
  though. About 10 pairs landed in a top-500 reading list, and they
  also inflate embedding and labelling cost.

| Primary cause | Groups | Example (from that corpus) |
|---|---:|---|
| 1. Language-bucket mismatch | 39 | OpenAlex `en` vs WoS Expanded `English` vs Crossref/CORE `None`: *DINOv2 rocks geological image analysis…* (arXiv DOI vs JRMGE DOI) |
| 2. Short-title gate (`min_title_tokens=6`) | 24 | *Using deep learning for Digital Soil Mapping* (SOIL Discussions `10.5194/soil-2018-28` vs SOIL `10.5194/soil-5-79-2019`; 5 tokens) |
| 3. Same source ID repeated | 2 (+1 short) | `WOS:000181313300003`, `WOS:A1992HL65300002`, `core:21667539` each appear twice with identical title and abstract |
| 4. Hyphenation splits tokens | 2 | "multielectrode" vs "multi-electrode"; "ortho-mosaic" vs "Orthomosaic" push Jaccard below 0.85 |
| 5. Subtitle drift, preprint vs journal | 1 (in top 500) | *Deep learning image analysis for filamentous fungi…* bioRxiv vs *Biology Methods & Protocols*: J = 0.81, and language `None` vs `en` |

## Root causes in the code

1. **Language bucketing keys on the raw string** (`dedup.py`,
   `title_buckets[lang]`).
   - Clients return codes (`en`), full names (`English`, from WoS
     Expanded `_wos_exp_language`), or `None` (Crossref, Semantic
     Scholar, Europe PMC, arXiv sometimes, CORE sometimes).
   - Records in different buckets are never compared. Across the corpus
     the values were `en` 9,789, `English` 1,165, missing 613, then
     smaller counts of `Portuguese`, `Spanish` and so on alongside `pt`
     and `es`.
2. **`min_title_tokens` gates exact matches too.** The gate exists to
   stop fuzzy false positives on generic titles ("Reply on RC2").
   However, it also blocks titles that are *identical* after
   normalization and share a year, venue or author.
3. **No source-ID key.**
   - `dedup()` never compares `id`.
   - Only `search_openalex` keeps a `seen_ids` set across queries.
   - WoS Expanded, CORE and the other clients return the same record
     once per matching query, and a DOI-less, short-titled repeat
     survives every pass.
4. **Tokenization splits on hyphens.** `_TOKEN_RE` treats `-` and
   U+2010 as separators, so hyphenated and closed compounds give
   different token sets.
5. **Preprint and journal DOIs differ by design** (arXiv/DataCite,
   bioRxiv, OSF/EarthArXiv, Copernicus Discussions). These pairs rely
   entirely on the title pass, so causes 1, 2 and 4 hit them hardest.

## Proposed refinements (in priority order)

- **R1: normalize language before bucketing.**
  - Map full names and variants to ISO 639-1: `English`→`en`,
    `Portuguese`→`pt`, `Spanish`→`es`, `French`→`fr`, and so on.
  - Treat missing language as a wildcard that is compared against every
    bucket. It should not be its own bucket.
  - The spec's reason for bucketing (cross-language false positives)
    still holds, because Jaccard of 0.85 or more across real languages
    is rare anyway.
  - Expected to fix about 60% of misses.
- **R2: exact normalized-title pass ahead of the fuzzy pass.**
  - Normalize titles: NFKC, casefold, drop all non-alphanumerics, which
    also joins hyphenated compounds.
  - On an exact match, merge when year differs by 1 or less **and**
    first-author surname or venue agrees. Skip `min_title_tokens` for
    this pass.
  - Keep a small stoplist of generic titles ("Editorial", "Preface",
    "Reply to…", "Corrigendum", "Erratum", "Book review") that never
    merge on title alone.
  - Fixes cause 2 and part of cause 4.
- **R3: key on source ID.**
  - Add a pass 0 in `dedup()` that merges on an exact `id`.
  - Also add `seen_ids` to the clients that lack it, starting with WoS
    Expanded and CORE.
  - Trivial and zero-risk.
- **R4: hyphen-insensitive tokens for Jaccard.**
  - Tokenize a version of the title with hyphens removed, alongside the
    split version, and take the higher Jaccard.
  - Alternatively, compare character 3-gram sets as a tiebreak.
- **R5: author-assisted relaxed threshold for preprint and journal
  pairs.**
  - When the first-author surname matches and years differ by 1 or
    less, accept Jaccard of 0.75 or more.
  - Optionally use relation metadata where the client has it: Crossref
    `relation.is-preprint-of` / `has-preprint`, and the Copernicus
    `…-discussions` pattern.
- **R6: post-hoc report, not merge.** Add a
  `scripts/dedup_report.py` that lists near-duplicate groups (the
  normalized-title and year test above) for human review. It is cheap,
  catches whatever R1–R5 miss, and would have flagged all 67 groups.

## Tests to add (`tests/test_dedup.py`)

Build one fixture per row of the table above using the real titles,
DOIs and language values. Then assert:
- each pair collapses to one row;
- `source_databases` holds the union;
- the generic-title stoplist still prevents merges (for example two
  distinct "Editorial" records in the same year).

## Cautions

- **Merges are order-dependent.** The first record encountered keeps
  its fields. With R1 and R2, a WoS or CORE record may become the kept
  row where OpenAlex used to be. Consider preferring the row with a DOI
  and an abstract when choosing the survivor.
- **Existing corpora change size when re-deduped.** Embedding caches key
  on `id`, so re-deduping is safe for them. Labelled corpora will lose
  rows. Announce this in the commit message, per the CLAUDE.md
  backwards-compatibility rule.
- **Sibling projects hold byte-copies of `dedup.py`.** After a fix, list
  it in `BACKPORTING_NEW_SOURCES.md`, as was done for the idempotence
  fix.
