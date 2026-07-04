"""Migrate the deprecated PDF corpus into the canonical knowledge repository."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cial_knowledge_os.config import KnowledgeOSConfig


@dataclass(frozen=True, slots=True)
class MigrationSummary:
    source_count: int
    copied_count: int
    skipped_existing_count: int
    destination: Path
    move: bool
    dry_run: bool


def migrate_pdf_corpus(
    config: KnowledgeOSConfig,
    *,
    move: bool = False,
    dry_run: bool = False,
) -> MigrationSummary:
    """Copy or move legacy PDFs into ``knowledge_root/legacy_pdf``."""

    source_files = (
        sorted(
            path
            for path in config.legacy_pdf_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        if config.legacy_pdf_root.exists()
        else []
    )
    destination = config.knowledge_root / "legacy_pdf"
    copied_count = 0
    skipped_existing_count = 0
    reserved_names: set[str] = set()

    for source in source_files:
        target = destination / source.name
        normalized_name = source.name.casefold()
        if target.exists() or normalized_name in reserved_names:
            skipped_existing_count += 1
            continue
        reserved_names.add(normalized_name)
        copied_count += 1
        if dry_run:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)

    return MigrationSummary(
        source_count=len(source_files),
        copied_count=copied_count,
        skipped_existing_count=skipped_existing_count,
        destination=destination,
        move=move,
        dry_run=dry_run,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy PDFs from the deprecated corpus into "
            "data/files/legacy_pdf by default."
        )
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move source PDFs instead of copying them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned work without changing files.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = KnowledgeOSConfig(project_root=args.project_root)
    summary = migrate_pdf_corpus(config, move=args.move, dry_run=args.dry_run)
    action = "move" if summary.move else "copy"
    suffix = " (dry run)" if summary.dry_run else ""
    print(f"Migration mode: {action}{suffix}")
    print(f"Source count: {summary.source_count}")
    print(f"Copied count: {summary.copied_count}")
    print(f"Skipped existing count: {summary.skipped_existing_count}")
    print(f"Destination: {summary.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
