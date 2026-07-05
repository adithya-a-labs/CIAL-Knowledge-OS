"""Migrate one embedded Qdrant collection to a local Qdrant server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be greater than zero")
    return parsed


def migrate_collection(
    source: QdrantClient,
    target: QdrantClient,
    *,
    collection_name: str,
    batch_size: int = 512,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Copy collection configuration and points while preserving IDs and payloads."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not source.collection_exists(collection_name):
        raise ValueError(f"Source collection '{collection_name}' does not exist.")

    source_info = source.get_collection(collection_name)
    source_count = int(
        source.count(collection_name=collection_name, exact=True).count
    )
    target_exists = target.collection_exists(collection_name)
    if target_exists and not force:
        raise ValueError(
            f"Target collection '{collection_name}' already exists. "
            "Use --force to overwrite it."
        )

    print(
        f"{'[dry-run] ' if dry_run else ''}Migrating {source_count:,} points "
        f"from collection '{collection_name}' in batches of {batch_size}.",
        flush=True,
    )
    if dry_run:
        print("Dry run complete; the target was not modified.", flush=True)
        return {
            "source_points": source_count,
            "migrated_points": 0,
            "batches": 0,
            "dry_run": True,
        }

    if target_exists:
        target.delete_collection(collection_name)

    params = source_info.config.params
    target.create_collection(
        collection_name=collection_name,
        vectors_config=params.vectors,
        sparse_vectors_config=getattr(params, "sparse_vectors", None),
    )

    migrated = 0
    batches = 0
    offset: Any | None = None
    while True:
        records, offset = source.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if records:
            points = [
                PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in records
            ]
            target.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            migrated += len(points)
            batches += 1
            print(
                f"Migrated {migrated:,}/{source_count:,} points "
                f"({batches} batches).",
                flush=True,
            )
        if offset is None:
            break

    target_count = int(
        target.count(collection_name=collection_name, exact=True).count
    )
    if target_count != source_count:
        raise RuntimeError(
            f"Migration verification failed: source has {source_count:,} points "
            f"but target has {target_count:,}."
        )
    print(
        f"Migration complete: {migrated:,} points in {batches} batches; "
        f"target count verified at {target_count:,}.",
        flush=True,
    )
    return {
        "source_points": source_count,
        "migrated_points": migrated,
        "batches": batches,
        "dry_run": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--api-key")
    parser.add_argument("--collection", default="cial_phase4")
    parser.add_argument("--batch-size", type=positive_integer, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = args.source.expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(
            f"Embedded Qdrant source path does not exist: {source_path}"
        )

    source = QdrantClient(path=str(source_path))
    target = QdrantClient(url=args.url, api_key=args.api_key)
    try:
        try:
            target.get_collections()
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant server at {args.url} is not reachable. "
                "Start it with: docker compose -f "
                "docker-compose.qdrant.yml up -d"
            ) from exc
        migrate_collection(
            source,
            target,
            collection_name=args.collection,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            force=args.force,
        )
    finally:
        source.close()
        target.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
