"""Rebuild the faceted-filter HTML for every sister lit-search repo.

litsweep is the single source of truth for ``build_filter_html.py``; the
sibling projects no longer carry their own copies (which drifted). This
driver imports that builder and rebuilds each deployment's HTML from a
declarative manifest, so adding a new repo is a one-line ``BUILDS`` edit
rather than a script copy that falls out of sync.

Each entry is one HTML output. A build may combine several corpora that
share an embedding space (the "corpus" facet appears automatically), run
a repo-local ``prep`` step first (e.g. reworming's ``_x``/``_y`` merge
collapse), or restrict to a regex-matched subset (``match``). Paths are
resolved against the workspace root, so this runs correctly from any cwd.

Usage::

    python scripts/refresh_all_filters.py            # rebuild everything
    python scripts/refresh_all_filters.py --only reworming worm-tea
    python scripts/refresh_all_filters.py --list     # show the manifest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import build_filter_html

# Workspace root: litsweep/scripts/this.py -> .../projects/
_WORKSPACE = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Build:
    """One faceted-filter HTML deployment.

    Attributes:
        name: Short handle for ``--only`` selection and logging.
        out: Output HTML path, relative to the workspace root.
        title: Page title shown in the header.
        csv: Input labeled-corpus CSV(s), relative to the workspace root.
            More than one adds a cross-corpus "corpus" facet.
        relevance: ``relevance_llm`` values to keep, or "all".
        match: Regex groups; a record must match every group (``a|b``
            within one group is OR). Restricts to a focused subset.
        prep: Optional ``(repo_dir, script)`` run before the build, with
            ``repo_dir`` (relative to the workspace root) as cwd — for
            repo-specific corpus preprocessing.
    """

    name: str
    out: str
    title: str
    csv: tuple[str, ...]
    relevance: str = "core,adjacent"
    match: tuple[str, ...] = ()
    prep: tuple[str, str] | None = None


# The particle-paradox family shares one embedding space and is browsed
# as a single combined view; the worm/soil deployments are standalone.
_PPARADOX_CORPORA = (
    "pparadox-lit/results/pparadox_lit_labeled_corpus_dedup.csv",
    "native-sand/results/native_sand_labeled_corpus_dedup.csv",
    "microtexture-lit-search/results/microtexture_labeled_corpus_dedup.csv",
    "dust-loess-lit/results/dust_loess_lit_labeled_corpus_dedup.csv",
)

BUILDS: tuple[Build, ...] = (
    Build(
        name="reworming",
        out="reworming-lit/results/analysis/articles_filter.html",
        title="Reworming — global earthworm restoration literature",
        # Collapse the pandas merge `_x`/`_y` columns into a clean view first.
        prep=("reworming-lit", "scripts/collapse_labeled_view.py"),
        csv=("reworming-lit/results/reworming_lit_labeled_corpus_view.csv",),
    ),
    Build(
        name="worm-tea",
        out="worm-tea-lit/results/analysis/articles_filter.html",
        title="Worm-tea / vermicompost-extract literature",
        csv=("worm-tea-lit/results/worm_tea_lit_labeled_corpus.csv",),
    ),
    Build(
        name="soilmag",
        out="soilmag/results/analysis/articles_filter.html",
        title="Soil magnetism literature",
        csv=("soilmag/results/soilmag_lit_labeled_corpus.csv",),
    ),
    Build(
        name="pparadox",
        out="pparadox-lit/results/articles_filter.html",
        title="Particle-paradox literature (4 corpora)",
        csv=_PPARADOX_CORPORA,
    ),
    Build(
        name="pparadox-weathering-sem",
        out="pparadox-lit/results/articles_filter_weathering_sem.html",
        title="Particle paradox — weathering + SEM",
        csv=_PPARADOX_CORPORA,
        match=(r"\bsem\b|scanning electron|exoscop",
               r"weather|dissolution|\betch|saprolit"),
    ),
)


def _run_prep(prep: tuple[str, str]) -> None:
    """Run a repo-local preprocessing script before its build.

    Args:
        prep: ``(repo_dir, script)`` both relative to the workspace root;
            the script runs with ``repo_dir`` as its working directory.

    Raises:
        subprocess.CalledProcessError: if the prep script exits non-zero.
    """

    repo_dir, script = prep
    cwd = _WORKSPACE / repo_dir
    print(f"  prep: {script}  (cwd {repo_dir})")
    subprocess.run([sys.executable, script], cwd=cwd, check=True)


def _argv_for(build: Build) -> list[str]:
    """Construct the ``build_filter_html`` CLI args for one build."""

    argv = ["--out", str(_WORKSPACE / build.out),
            "--title", build.title,
            "--relevance", build.relevance,
            "--csv", *[str(_WORKSPACE / c) for c in build.csv]]
    for group in build.match:
        argv += ["--match", group]
    return argv


def run(build: Build) -> None:
    """Prep (if any) and render one deployment's HTML."""

    print(f"\n▶ {build.name}  ->  {build.out}")
    if build.prep is not None:
        _run_prep(build.prep)
    build_filter_html.main(_argv_for(build))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", nargs="+", metavar="NAME",
                   help="rebuild only these builds (by name)")
    p.add_argument("--list", action="store_true",
                   help="print the manifest and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Rebuild the selected (default: all) filter pages."""

    args = _parse_args(argv)
    if args.list:
        for b in BUILDS:
            print(f"{b.name:24s} {len(b.csv)} corpus(es) -> {b.out}")
        return

    builds = BUILDS
    if args.only:
        want = set(args.only)
        builds = tuple(b for b in BUILDS if b.name in want)
        missing = want - {b.name for b in builds}
        if missing:
            sys.exit(f"unknown build name(s): {', '.join(sorted(missing))}")

    for build in builds:
        run(build)
    print(f"\nDone — rebuilt {len(builds)} filter page(s).")


if __name__ == "__main__":
    main()
