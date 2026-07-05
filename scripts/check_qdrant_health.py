"""Print a concise health report for a local Qdrant collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cial_knowledge_os.infra.qdrant_health import check_qdrant_health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a local Qdrant server and collection."
    )
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--collection", default="cial_phase4")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    return parser.parse_args()


def _display(value: Any) -> str:
    return "unknown" if value is None else str(value)


def main() -> int:
    args = parse_args()
    client = QdrantClient(url=args.url, api_key=args.api_key)
    try:
        report = check_qdrant_health(
            client,
            args.collection,
            embedding_dimension=args.embedding_dimension,
        )
    finally:
        client.close()

    print(f"reachable: {'yes' if report['reachable'] else 'no'}")
    print(
        "collection: "
        f"{'present' if report['collection_exists'] else 'missing'} "
        f"({args.collection})"
    )
    print(f"collection status: {_display(report['collection_status'])}")
    print(f"point count: {report['point_count']}")
    print(f"indexed vector count: {report['indexed_vector_count']}")
    print(f"optimizer status: {_display(report['optimizer_status'])}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 0 if report["reachable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
