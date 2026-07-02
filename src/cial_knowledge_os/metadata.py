"""Canonical metadata access for retrieval and context construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

RetrievalResult: TypeAlias = dict[str, Any]
ChunkIdentity: TypeAlias = tuple[str, Any, str]


def result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable copy of nested result metadata."""

    value = result.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def source_path(result: Mapping[str, Any]) -> str:
    """Return the most traceable source identifier available."""

    metadata = result_metadata(result)
    return str(
        metadata.get("source")
        or result.get("source")
        or metadata.get("file_name")
        or ""
    )


def source_label(result: Mapping[str, Any]) -> str:
    """Return a human-readable document label without losing path metadata."""

    metadata = result_metadata(result)
    value = metadata.get("file_name") or result.get("source") or source_path(result)
    return Path(str(value)).name if value else "Unknown document"


def page_number(result: Mapping[str, Any]) -> Any:
    """Read a page value from either the public result or nested metadata."""

    metadata = result_metadata(result)
    return result.get("page_number", metadata.get("page_number", metadata.get("page")))


def chunk_id(result: Mapping[str, Any]) -> str:
    """Read a stable chunk identifier."""

    metadata = result_metadata(result)
    value = result.get("chunk_id", metadata.get("chunk_id"))
    return "" if value is None else str(value)


def chunk_index(result: Mapping[str, Any]) -> int | None:
    """Read the source-relative integer chunk position when present."""

    metadata = result_metadata(result)
    value = result.get("chunk_index", metadata.get("chunk_index"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def chunk_identity(result: Mapping[str, Any]) -> ChunkIdentity:
    """Build the required ``(source, page, chunk_id)`` deduplication key."""

    return (source_path(result), page_number(result), chunk_id(result))


def normalize_result(result: Mapping[str, Any]) -> RetrievalResult:
    """Expose canonical citation fields while preserving all input metadata."""

    normalized = dict(result)
    metadata = result_metadata(result)
    normalized.update(
        {
            "metadata": metadata,
            "source": source_label(result),
            "source_path": source_path(result),
            "page_number": page_number(result),
            "chunk_id": chunk_id(result),
            "chunk_index": chunk_index(result),
            "text": str(result.get("text", "")),
        }
    )
    score = result.get("score")
    normalized["score"] = float(score) if score is not None else None
    return normalized
