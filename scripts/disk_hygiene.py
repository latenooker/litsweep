"""End-of-pipeline disk hygiene: parquet-archive raw/, verify, delete.

Reads every JSON file in ``results/raw/``, packs them into a single
zstd-9 Parquet at ``results/archive/raw_archive.parquet``, logs the
md5, and (only on success) deletes the JSON directory. Idempotent: if
``results/raw/`` is missing or empty this is a no-op.

Usage::

    python scripts/disk_hygiene.py                       # results/ in cwd
    python scripts/disk_hygiene.py --results /path/to/results
    python scripts/disk_hygiene.py --no-delete           # archive but keep raw/
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("disk_hygiene")


def _md5(path: Path) -> str:
    """Return the hex md5 digest of a file, streamed in 1 MiB chunks.

    Args:
        path: File to digest.

    Returns:
        The hexadecimal md5 digest string.
    """
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_raw(results: Path, delete: bool = True) -> Path | None:
    """Pack ``results/raw/*.json`` into ``results/archive/raw_archive.parquet``.

    Args:
        results: The project's ``results/`` directory.
        delete: When true (default), remove ``results/raw/`` after a
            successful archive write.

    Returns:
        The archive path on success, or ``None`` if ``raw/`` is missing
        or empty (a no-op).
    """
    raw = results / "raw"
    archive_dir = results / "archive"
    archive = archive_dir / "raw_archive.parquet"

    if not raw.is_dir():
        logger.info("no raw/ directory at %s — nothing to archive", raw)
        return None

    json_files = sorted(raw.glob("*.json"))
    if not json_files:
        logger.info("raw/ is empty — nothing to archive")
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for fp in json_files:
        source, _, query = fp.stem.partition("__")
        rows.append({
            "source": source,
            "query": query or "(no_query)",
            "payload_json": fp.read_text(encoding="utf-8"),
        })
    df = pd.DataFrame(rows)
    if archive.exists():
        logger.warning("overwriting existing %s", archive)
    df.to_parquet(archive, compression="zstd", compression_level=9)

    # Verify the archive round-trips before irreversibly deleting raw/.
    # A truncated or structurally-corrupt parquet that didn't raise on
    # write must NOT cost us the source JSON cache.
    try:
        written_rows = len(pd.read_parquet(archive))
    except Exception as exc:
        raise RuntimeError(
            f"Archive at {archive} failed to read back ({exc}); "
            f"raw/ left intact."
        )
    if written_rows != len(rows):
        raise RuntimeError(
            f"Archive row-count mismatch: wrote {len(rows)} records but "
            f"read back {written_rows}; raw/ left intact ({archive})."
        )

    md5 = _md5(archive)
    logger.info("wrote %s (%d JSON files, md5=%s)", archive, len(rows), md5)

    if delete:
        shutil.rmtree(raw)
        logger.info("deleted %s", raw)
    return archive


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"),
                   help="Path to results/ directory.")
    p.add_argument("--no-delete", action="store_true",
                   help="Archive but keep raw/ in place.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    archive_raw(args.results.resolve(), delete=not args.no_delete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
