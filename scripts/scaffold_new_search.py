"""Scaffold a new lit-search project from the litsweep template.

Copies the shared infrastructure (api_clients, dedup, requirements,
all scripts) into a new project directory, stubs out the four
topic-specific files (queries.py, vocab.py, ANCHORS, SYSTEM_PROMPT),
runs ``git init`` + the initial commit, and (by default) creates a
private GitHub repo via the ``gh`` CLI and pushes. Source of truth
lives in this litsweep repo; running this script is the only
supported way to start a new search.

Pass ``--no-remote`` to skip the GitHub repo step (e.g. offline, or
to push later). Pass ``--public`` if the new search shouldn't be
private. Pass ``--no-git`` to skip the git workflow entirely.

What it does
------------

1. Creates the target directory tree
   (results/raw/, docs/, docs/session_logs/, scripts/).
2. Copies shared-infra files unchanged
   (api_clients.py, dedup.py, requirements.txt, .gitignore).
3. Copies project-shaped files with name substitution
   (orchestrator, scripts/embed_filter.py, scripts/label_with_stanford.py,
   scripts/citation_chase.py, scripts/wos_gap_fill.py,
   scripts/wos_expanded_ping.py, scripts/screen_wos_export.py,
   scripts/compare_wos_to_pipeline.py).
4. Replaces topic-specific content with TODO-marked placeholders:
   - ``queries.py`` — minimal example with one entry per source.
   - ``vocab.py`` — empty LITHOLOGIES / ZONES / MINERALS dicts.
   - ``ANCHORS`` in scripts/embed_filter.py.
   - ``SYSTEM_PROMPT`` in scripts/label_with_stanford.py.
5. Writes a fresh README.md and CLAUDE.md skeleton.
6. ``git init -b main`` (unless ``--no-git``).

What it does NOT do
-------------------

- Write your queries (the actual research-scoping work).
- Write your anchor descriptions for the embedder.
- Write your label schema and system prompt.
- Set up your API keys (those live in your shell or .env).

See ``docs/DEPLOYING_A_NEW_SEARCH.md`` for the full checklist.

Usage
-----

::

    python scripts/scaffold_new_search.py /path/to/new-project --name my_topic

    # Use a different source project as the template:
    python scripts/scaffold_new_search.py ../microbial-mat-search \\
        --name microbial_mat \\
        --from /Users/looker/Documents/projects/microtexture-lit-search

    # Skip git init (e.g. when scaffolding inside an existing monorepo):
    python scripts/scaffold_new_search.py ./new-search --name foo --no-git
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

THIS_PROJECT = Path(__file__).resolve().parent.parent

# Files copied byte-for-byte (no substitution).
SHARED_INFRA = [
    "api_clients.py",
    "dedup.py",
    "requirements.txt",
    ".gitignore",
]

# Files copied with name substitution. Optional ones are skipped if
# missing in the source.
TEMPLATED_FILES = [
    "scripts/embed_filter.py",
    "scripts/label_with_stanford.py",
    "scripts/citation_chase.py",
    "scripts/wos_gap_fill.py",
    "scripts/wos_expanded_ping.py",
    "scripts/screen_wos_export.py",
    "scripts/compare_wos_to_pipeline.py",
    "scripts/add_openalex_queries.py",
    # backfill_abstracts.py is slug-agnostic (auto-detects the
    # bibliography CSV from results/*_bibliography.csv) so it
    # passes through the substitution step unchanged. Including
    # it here so every scaffolded project ships with the abstract-
    # backfill path for sources that return only metadata.
    "scripts/backfill_abstracts.py",
    # embed_diagnostic.py is the post-embed coverage table; same
    # rationale as backfill — slug-agnostic, every project benefits.
    "scripts/embed_diagnostic.py",
    # merge_gap_fill.py merges a sibling-output dedup-merge into the
    # main bibliography (used for WoS-Expanded gap-fills, etc.).
    "scripts/merge_gap_fill.py",
]


def _slug_variants(slug: str) -> dict[str, str]:
    """Build the substitution map from a snake_case slug."""
    snake = slug
    dashed = slug.replace("_", "-")
    title = " ".join(p.capitalize() for p in slug.split("_"))
    return {
        "native_sand": snake,
        "native-sand": dashed,
        "Native-Sand": title,
        "native sand": dashed,  # in prose
        "microtexture": snake,
        "microtexture-lit-search": dashed,
        "Microtexture": title.split()[0] if title else snake,
    }


def _substitute(text: str, src_slug: str, dst_slug: str) -> str:
    """Apply slug substitution. ``src_slug`` is the source project's slug
    (``native_sand`` or ``microtexture``); ``dst_slug`` is the new one."""
    src_dashed = src_slug.replace("_", "-")
    dst_dashed = dst_slug.replace("_", "-")
    src_title = " ".join(p.capitalize() for p in src_slug.split("_"))
    dst_title = " ".join(p.capitalize() for p in dst_slug.split("_"))
    out = text
    out = out.replace(src_slug, dst_slug)
    out = out.replace(src_dashed, dst_dashed)
    out = out.replace(src_title, dst_title)
    return out


def _detect_src_slug(src_root: Path) -> str:
    """Infer the source project's slug from its orchestrator filename."""
    for p in src_root.glob("*_search.py"):
        return p.stem.removesuffix("_search")
    raise SystemExit(
        f"Cannot detect source slug: no *_search.py at {src_root}. "
        "Pass --src-slug explicitly."
    )


def _copy_byte(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_with_subst(src: Path, dst: Path, src_slug: str, dst_slug: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst.write_text(_substitute(text, src_slug, dst_slug), encoding="utf-8")


_PLACEHOLDER_DOCSTRING = (
    '"""Multilingual literature search for {dst_dashed} — TODO topic.\n'
    "\n"
    "Orchestrates harvest -> dedup -> embed-filter -> label across the\n"
    "configured bibliographic sources. Topic-specific search strings\n"
    "live in ``queries.py``; vocabulary axes in ``vocab.py``.\n"
    "\n"
    "Replace this docstring with the actual project scope before\n"
    "running. ``python {dst_slug}_search.py --help`` shows it as the\n"
    "CLI description.\n"
    '"""'
)


def _replace_module_docstring(text: str, dst_slug: str) -> str:
    """Replace the orchestrator's module docstring with a generic placeholder.

    Looks for the leading triple-quoted string (the conventional location
    of a module docstring) and substitutes a topic-agnostic TODO marker.
    If no leading docstring is found, return ``text`` unchanged.
    """
    pattern = re.compile(r'^\s*"""(?:(?!""").)*?"""', re.DOTALL)
    match = pattern.match(text)
    if not match:
        return text
    dst_dashed = dst_slug.replace("_", "-")
    placeholder = _PLACEHOLDER_DOCSTRING.format(
        dst_slug=dst_slug, dst_dashed=dst_dashed,
    )
    return placeholder + text[match.end():]


def _git_init_and_commit(target: Path) -> bool:
    """Run ``git init`` and the initial commit in *target*.

    Returns True on success. Logs and returns False if git is missing
    or any step fails — the caller will skip the remote step.
    """
    try:
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=target, check=True, capture_output=True,
        )
        print("  git init -b main")
        subprocess.run(
            ["git", "add", "."],
            cwd=target, check=True, capture_output=True,
        )
        msg = (
            "Initial scaffold from litsweep\n"
            "\n"
            "Topic-specific files (queries.py, vocab.py, ANCHORS, "
            "SYSTEM_PROMPT) are stubbed; fill them in per "
            "litsweep/docs/DEPLOYING_A_NEW_SEARCH.md."
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=target, check=True, capture_output=True,
        )
        print("  git commit (initial scaffold)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        stderr = getattr(exc, "stderr", b"")
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else str(exc)
        print(f"  git init/commit skipped: {stderr_text.strip()}",
              file=sys.stderr)
        return False


def _gh_create_and_push(
    *,
    target: Path,
    slug: str,
    owner: str | None,
    private: bool,
) -> str | None:
    """Create a GitHub repo via the ``gh`` CLI and push the initial commit.

    Repo name is derived from *slug* (snake_case → kebab-case). Visibility
    is private by default to match the rest of the lit-search family.
    Returns the new remote URL on success or None on failure (gh
    missing/unauth'd, repo already exists, network error). Never raises
    — a missing remote is recoverable; the user can create it manually.
    """
    repo_name = slug.replace("_", "-")
    full_name = f"{owner}/{repo_name}" if owner else repo_name
    visibility_flag = "--private" if private else "--public"
    description = (
        f"{repo_name} — multilingual lit-search corpus, scaffolded "
        f"from litsweep."
    )
    cmd = [
        "gh", "repo", "create", full_name,
        visibility_flag,
        "--source", str(target),
        "--description", description,
        "--push",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=target, check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        print("  gh CLI not found; skipping remote create.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(f"  gh repo create failed: {stderr}", file=sys.stderr)
        return None
    # gh prints the URL on stdout; first line is canonical.
    url = (result.stdout or "").strip().splitlines()
    remote_url = url[0] if url else f"https://github.com/{full_name}"
    print(f"  gh repo create {full_name} ({'private' if private else 'public'})")
    print(f"  pushed initial commit to {remote_url}")
    return remote_url


def _replace_anchors(embed_filter_path: Path) -> None:
    """Replace the ANCHORS list with a TODO-marked stub."""
    text = embed_filter_path.read_text(encoding="utf-8")
    new_block = (
        "ANCHORS: list[str] = [\n"
        "    # TODO: write 6-10 anchor descriptions for your topic.\n"
        "    # Each is a 1-3 sentence prose description of one facet.\n"
        "    # BGE-M3 is multilingual, so non-English vocab is OK.\n"
        '    "TODO anchor 0 — primary topic facet",\n'
        '    "TODO anchor 1 — secondary process facet",\n'
        '    "TODO anchor 2 — methodological facet",\n'
        "]\n"
    )
    pattern = re.compile(
        r"ANCHORS: list\[str\] = \[.*?\n\]\n", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(
            f"ANCHORS block not found in {embed_filter_path}; cannot stub."
        )
    embed_filter_path.write_text(pattern.sub(new_block, text), encoding="utf-8")


def _replace_system_prompt(label_path: Path) -> None:
    """Replace SYSTEM_PROMPT with a strict-JSON-shaped TODO stub."""
    text = label_path.read_text(encoding="utf-8")
    new_block = (
        'SYSTEM_PROMPT = """\\\n'
        "TODO: rewrite this prompt for your topic. Keep the structure:\n"
        "1) one paragraph stating who-you-are and what corpus this is.\n"
        "2) one paragraph defining IN SCOPE.\n"
        "3) one paragraph defining OUT OF SCOPE.\n"
        '4) "Return ONLY a JSON object - no prose, no code fences."\n'
        "5) the JSON schema, with enum values for every closed-set field.\n"
        "6) explicit Rules section covering edge cases (missing abstract,\n"
        '   "core" vs "adjacent" boundary, etc.)\n'
        "\n"
        "Return ONLY a JSON object - no prose, no code fences. Schema:\n"
        "{\n"
        '  "relevance": "core" | "adjacent" | "off_topic",\n'
        '  "rationale": "<one short sentence (<=25 words)>"\n'
        "}\n"
        '"""\n'
    )
    pattern = re.compile(
        r'SYSTEM_PROMPT = """\\.*?\n"""\n', re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(
            f"SYSTEM_PROMPT block not found in {label_path}; cannot stub."
        )
    label_path.write_text(pattern.sub(new_block, text), encoding="utf-8")


def _write_queries_stub(target: Path, slug: str) -> None:
    title = " ".join(p.capitalize() for p in slug.split("_"))
    content = f'''"""Search queries for `{slug}`: TODO one-line topic description.

Each constant is a list of strings passed to the corresponding API
client. OpenAlex/SemanticScholar/Google Scholar take prose queries;
WoS uses field-tagged TS=(...) syntax; HAL/TEL/theses.fr/BDTD use
each platform's own grammar.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OpenAlex (`search` parameter — full-text-ish search of titles + abstracts)
# ---------------------------------------------------------------------------

OPENALEX_GROUP_A_CORE_EN: list[str] = [
    # TODO: 10-25 prose queries covering the core of your topic in English.
    "TODO replace with topic query",
]

OPENALEX_GROUP_B_PROCESS: list[str] = [
    # TODO: 5-10 process-level queries (the dynamics / kinetics / processes).
]

OPENALEX_ALL: list[str] = (
    OPENALEX_GROUP_A_CORE_EN
    + OPENALEX_GROUP_B_PROCESS
)

# ---------------------------------------------------------------------------
# Semantic Scholar — subset of the OpenAlex queries (Eng-only matters there)
# ---------------------------------------------------------------------------

SEMANTIC_SCHOLAR: list[str] = OPENALEX_GROUP_A_CORE_EN[:8]

# ---------------------------------------------------------------------------
# WoS Starter — field-tagged TS=(...) syntax. Same strings reused for
# WoS Expanded (Expanded uses the same advanced syntax, different endpoint).
# ---------------------------------------------------------------------------

WOS_STARTER: list[str] = [
    # TODO: 6-12 TS=(...) strings. Use AND/OR/NOT (uppercase), * for
    # wildcards, "quoted phrases" for exact matches.
    'TS=(TODO_TERM AND (process_term1 OR process_term2))',
]

WOS_EXPANDED: list[str] = WOS_STARTER

# ---------------------------------------------------------------------------
# HAL / TEL (French theses) — Solr query strings.
# ---------------------------------------------------------------------------

HAL: list[str] = [
    # TODO: French translations of the core queries.
]

# ---------------------------------------------------------------------------
# theses.fr (French thesis metadata) — keyword strings.
# ---------------------------------------------------------------------------

THESES_FR: list[str] = HAL

# ---------------------------------------------------------------------------
# BASE (global thesis aggregator) — keyword strings.
# ---------------------------------------------------------------------------

BASE: list[str] = OPENALEX_GROUP_A_CORE_EN[:6]

# ---------------------------------------------------------------------------
# BDTD (Brazilian theses) — Portuguese keyword strings.
# ---------------------------------------------------------------------------

BDTD: list[str] = [
    # TODO: Portuguese translations.
]
'''
    (target / "queries.py").write_text(content, encoding="utf-8")


def _write_vocab_stub(target: Path) -> None:
    content = '''"""Topic vocabulary.

Each axis is a ``tag -> [surface forms]`` regex dictionary. Adding a
new axis is a four-step pattern:

1. Define the dict (e.g. ``MANAGEMENT``).
2. Compile patterns via ``_compile()``.
3. Define a ``find_<axis>(text)`` helper.
4. Register a ``VocabAxis`` in ``VOCAB_AXES`` below.

The orchestrator iterates ``VOCAB_AXES`` to derive one column per
axis (e.g. ``minerals_mentioned``) on every record, and the CSV
write step automatically picks up registered columns. No further
editing of the orchestrator or ``CSV_COLUMNS`` is needed.

Optional: ``TOPIC_PRESENCE`` lets you declare a single boolean
gate (e.g. "any earthworm taxon mentioned?") for use as a hard
``--require-column`` filter before LLM labeling. Set to None when
no such gate makes sense for your topic.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Vocabulary dictionaries — fill these in for your topic.
# ---------------------------------------------------------------------------

# Example axis. Rename, replicate, and repopulate for your domain.
EXAMPLE_AXIS: dict[str, list[str]] = {
    # "tag_name": ["surface form 1", "surface form 2", ...]
    # TODO: e.g. {"granite": ["granite", "granitic", "granitique", ...]}
}

# Title-substring blocklist for early off-topic filtering. Anything matching
# is dropped pre-dedup, so reserve for high-confidence excludes.
TITLE_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    # TODO
)


# ---------------------------------------------------------------------------
# Compiled patterns + lookup helpers (do not edit unless adding a new axis)
# ---------------------------------------------------------------------------


def _compile(vocab: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    return {
        tag: re.compile(r"\\b(" + "|".join(re.escape(s) for s in surfaces) + r")\\b",
                        re.IGNORECASE)
        for tag, surfaces in vocab.items()
        if surfaces
    }


_EXAMPLE_PATTERNS = _compile(EXAMPLE_AXIS)


def find_example(text: str | None) -> list[str]:
    """Tags from EXAMPLE_AXIS matched in *text* (case-insensitive)."""
    if not text:
        return []
    return [tag for tag, pat in _EXAMPLE_PATTERNS.items() if pat.search(text)]


# ---------------------------------------------------------------------------
# Vocab axis registry — declarative; consumed by the orchestrator.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocabAxis:
    """Spec for one regex-driven vocabulary axis.

    Attributes:
        name: Short id (used in logs).
        column: CSV output column (e.g. ``"minerals_mentioned"``).
        find: Function taking text and returning matched tags.
    """

    name: str
    column: str
    find: Callable[[str | None], list[str]]


VOCAB_AXES: list[VocabAxis] = [
    VocabAxis("example", "example_mentioned", find_example),
    # TODO: add one VocabAxis per regex axis you define above.
]


# Optional topic-presence boolean used as a hard gate before LLM
# labeling. Tuple of ``(column_name, predicate)`` or None.
# Example for a project gating on a taxon presence regex:
#     TOPIC_PRESENCE = ("organism_mentioned", mentions_organism)
TOPIC_PRESENCE: tuple[str, Callable[[str | None], bool]] | None = None
'''
    (target / "vocab.py").write_text(content, encoding="utf-8")


def _write_readme(target: Path, slug: str) -> None:
    title = " ".join(p.capitalize() for p in slug.split("_"))
    content = f"""# {slug.replace('_', '-')}

TODO one-paragraph description of the topic this corpus covers.

## Layout

```
{slug.replace('_', '-')}/
├── {slug}_search.py         # CLI entry
├── queries.py               # search queries grouped by source
├── vocab.py                 # topic-specific vocabularies
├── api_clients.py           # API wrappers (shared)
├── dedup.py                 # DOI + Jaccard title-similarity dedup (shared)
├── requirements.txt
├── docs/
└── scripts/
    ├── embed_filter.py      # local Ollama BGE-M3 + anchor cosine
    └── label_with_stanford.py  # Stanford gateway → strict-JSON labels
```

## Quick start

```bash
pip install -r requirements.txt
python {slug}_search.py --dry-run                # validate queries
python {slug}_search.py --email you@example.com  # full run

# Two-stage triage:
python scripts/embed_filter.py \\
    --csv results/{slug}_bibliography.csv \\
    --out results/{slug}_bibliography_embedded.csv
python scripts/label_with_stanford.py \\
    --csv results/{slug}_bibliography_embedded.csv \\
    --out results/{slug}_labeled_corpus.csv
```

See `docs/DEPLOYING_A_NEW_SEARCH.md` (in the source project that
scaffolded this one) for the deployment checklist.
"""
    (target / "README.md").write_text(content, encoding="utf-8")


def _write_claude_md(target: Path, slug: str) -> None:
    title = " ".join(p.capitalize() for p in slug.split("_"))
    content = f"""# CLAUDE.md — {slug.replace('_', '-')}

Project-specific guidance for Claude Code working in this repository.

## What this is

Scaffolded from the lit-search template. TODO: write one paragraph
describing the actual topic.

## Conventions

- Python only. `from __future__ import annotations` at the top of every module.
- Modern type hints (`str | None`, `list[str]`).
- Google-style docstrings on public functions and classes.
- Private helpers prefixed with `_`.
- Prefer the standard library; deps pinned in `requirements.txt`.
- No async; APIs are rate-limited and serial calls keep things simple.

## Topic-specific files (the four things to write)

| File | What goes there |
| --- | --- |
| `queries.py` | All search strings, grouped by database. |
| `vocab.py` | Multilingual term dictionaries for tag-derivation. |
| `scripts/embed_filter.py` :: `ANCHORS` | 6-10 prose anchor descriptions for cosine scoring. |
| `scripts/label_with_stanford.py` :: `SYSTEM_PROMPT` | LLM label schema (strict JSON). |

Everything else is shared infrastructure — copy from the template,
do not edit unless extending the framework.

## Testing

Smoke-test before committing changes to queries.py or
{slug}_search.py:

```bash
python {slug}_search.py --dry-run
```
"""
    (target / "CLAUDE.md").write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path,
                        help="Path to the new project directory.")
    parser.add_argument("--name", required=True,
                        help="Slug for the new project (snake_case).")
    parser.add_argument("--from", dest="from_root", type=Path,
                        default=THIS_PROJECT,
                        help="Source project to copy from. "
                             "Defaults to the project this script lives in.")
    parser.add_argument("--src-slug",
                        help="Override the auto-detected source slug.")
    parser.add_argument("--no-git", action="store_true",
                        help="Skip `git init`, the initial commit, and the "
                             "remote-create step in the target.")
    parser.add_argument("--no-remote", action="store_true",
                        help="Run `git init` and the initial commit but skip "
                             "the `gh repo create` step. Use when you want "
                             "to scaffold offline or push the remote later.")
    parser.add_argument("--remote-owner", default=None,
                        help="GitHub owner (user or org) for the new remote. "
                             "Default: the gh CLI's auth'd user.")
    parser.add_argument("--public", action="store_true",
                        help="Create the GitHub repo as public. Default is "
                             "private (matches the rest of the lit-search family).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the target if it already exists.")
    args = parser.parse_args(argv)

    target: Path = args.target.resolve()
    src_root: Path = args.from_root.resolve()
    dst_slug: str = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", dst_slug):
        raise SystemExit("--name must be snake_case (lowercase + underscores).")

    src_slug = args.src_slug or _detect_src_slug(src_root)
    print(f"source : {src_root}  (slug = {src_slug})")
    print(f"target : {target}    (slug = {dst_slug})")

    if target.exists():
        if not args.force:
            raise SystemExit(f"target {target} exists; pass --force to overwrite.")
        shutil.rmtree(target)

    # Create dir tree
    for sub in ("scripts", "docs/session_logs", "results/raw"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    (target / "results/raw/.gitkeep").touch()

    # Copy shared infra exactly
    for rel in SHARED_INFRA:
        src = src_root / rel
        if src.exists():
            _copy_byte(src, target / rel)
            print(f"  byte-copy: {rel}")

    # Copy the orchestrator (renamed) with substitution
    src_orch = src_root / f"{src_slug}_search.py"
    if src_orch.exists():
        # Replace `import parent_lithologies as minerals_mod` with `import vocab as vocab_mod`
        # (and update references). Topic-vocab module is renamed to vocab.py for clarity.
        text = src_orch.read_text(encoding="utf-8")
        text = _substitute(text, src_slug, dst_slug)
        text = re.sub(
            r"import parent_lithologies as minerals_mod",
            "import vocab as vocab_mod", text)
        text = re.sub(r"import minerals as minerals_mod",
                      "import vocab as vocab_mod", text)
        text = text.replace("minerals_mod.", "vocab_mod.")
        # Replace the (topic-specific) module docstring with a generic
        # placeholder so `python <slug>_search.py --help` doesn't
        # mislead users into thinking the new project's scope is
        # whatever the source template's was. The substitution above
        # only renames slugs; the prose (e.g. "particle-size /
        # comminution studies") survives.
        text = _replace_module_docstring(text, dst_slug)
        (target / f"{dst_slug}_search.py").write_text(text, encoding="utf-8")
        print(f"  templated: {dst_slug}_search.py")

    # Copy templated scripts with substitution
    for rel in TEMPLATED_FILES:
        src = src_root / rel
        if src.exists():
            _copy_with_subst(src, target / rel, src_slug, dst_slug)
            print(f"  templated: {rel}")
        else:
            print(f"  skip (not present in source): {rel}")

    # Stub the four topic-specific files
    _write_queries_stub(target, dst_slug)
    print(f"  stubbed: queries.py")
    _write_vocab_stub(target)
    print(f"  stubbed: vocab.py")
    _replace_anchors(target / "scripts/embed_filter.py")
    print(f"  stubbed: ANCHORS in scripts/embed_filter.py")
    _replace_system_prompt(target / "scripts/label_with_stanford.py")
    print(f"  stubbed: SYSTEM_PROMPT in scripts/label_with_stanford.py")

    # Fresh top-level docs
    _write_readme(target, dst_slug)
    _write_claude_md(target, dst_slug)
    print(f"  wrote: README.md, CLAUDE.md")

    # git init + first commit + (default) gh repo create + push
    remote_url: str | None = None
    if not args.no_git:
        if not _git_init_and_commit(target):
            args.no_remote = True  # fall through; can't push without a commit
        if not args.no_remote:
            remote_url = _gh_create_and_push(
                target=target,
                slug=dst_slug,
                owner=args.remote_owner,
                private=not args.public,
            )

    print()
    print(f"Scaffolded {target}.")
    if remote_url:
        print(f"Remote : {remote_url}")
    print("Next steps (in order):")
    print("  1. Edit queries.py with topic-specific search strings.")
    print("  2. Edit vocab.py with topic vocabularies.")
    print("  3. Edit scripts/embed_filter.py :: ANCHORS (6-10 prose anchors).")
    print("  4. Edit scripts/label_with_stanford.py :: SYSTEM_PROMPT.")
    print("  5. python -m pip install -r requirements.txt")
    print(f"  6. python {dst_slug}_search.py --dry-run   # smoke-test queries")
    print(f"  7. python {dst_slug}_search.py             # full run")
    if not remote_url and not args.no_git and not args.no_remote:
        print()
        print("Note: remote was not created (gh CLI unavailable, not auth'd, "
              "or repo already exists). Create manually with:")
        repo_name = dst_slug.replace("_", "-")
        print(f"  gh repo create <owner>/{repo_name} --private "
              f"--source={target} --push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
