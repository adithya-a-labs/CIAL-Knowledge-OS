"""Read-only health inspection for embedded and self-hosted Qdrant."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

RED_OPTIMIZER_WARNING = (
    "Qdrant collection is available but optimizer status is red. Retrieval may "
    "work, but storage should be repaired before production use."
)

_VALID_HEALTH_COLORS = {"green", "yellow", "red"}


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _status_text(
    value: Any,
    *,
    optimizer: bool = False,
    _depth: int = 0,
) -> str:
    if value is None:
        return "unknown"
    if _depth >= 3:
        return "unknown"
    if optimizer and _value(value, "error") is not None:
        return "red"
    nested = _value(value, "status")
    if nested is not None and nested is not value:
        return _status_text(nested, optimizer=optimizer, _depth=_depth + 1)
    raw = getattr(value, "value", value)
    normalized = str(raw).strip().casefold()
    if optimizer:
        if normalized.startswith("error="):
            return "red"
        if normalized in {"ok", "green"}:
            return "green"
        if normalized in {"error", "failed", "red"}:
            return "red"
        if normalized in {"yellow", "optimizing"}:
            return "yellow"
    return normalized if normalized in _VALID_HEALTH_COLORS else normalized


def _vector_dimension(collection_info: Any) -> int | None:
    config = _value(collection_info, "config")
    params = _value(config, "params")
    vectors = _value(params, "vectors")
    if vectors is None:
        return None
    if isinstance(vectors, dict):
        if len(vectors) != 1:
            return None
        vectors = next(iter(vectors.values()))
    size = _value(vectors, "size")
    return int(size) if size is not None else None


def parse_collection_health(
    collection_info: Any,
    *,
    embedding_dimension: int | None = None,
) -> dict[str, Any]:
    """Normalize Qdrant collection health across client response versions."""

    collection_status = _status_text(_value(collection_info, "status"))
    optimizer_status = _status_text(
        _value(collection_info, "optimizer_status"),
        optimizer=True,
    )
    point_count = int(_value(collection_info, "points_count", 0) or 0)
    indexed_vector_count = int(
        _value(collection_info, "indexed_vectors_count", 0) or 0
    )
    vector_dimension = _vector_dimension(collection_info)
    warnings: list[str] = []
    errors: list[str] = []

    if collection_status == "yellow":
        warnings.append(
            "Qdrant collection status is yellow; indexing or optimization may "
            "still be in progress."
        )
    elif collection_status == "red":
        warnings.append(
            "Qdrant collection status is red; retrieval availability and data "
            "integrity require investigation."
        )
    if optimizer_status == "red":
        warnings.append(RED_OPTIMIZER_WARNING)
    if (
        embedding_dimension is not None
        and vector_dimension is not None
        and vector_dimension != embedding_dimension
    ):
        errors.append(
            f"Qdrant vector dimension is {vector_dimension}, but the configured "
            f"embedding dimension is {embedding_dimension}."
        )

    return {
        "collection_status": collection_status,
        "point_count": point_count,
        "indexed_vector_count": indexed_vector_count,
        "optimizer_status": optimizer_status,
        "vector_dimension": vector_dimension,
        "dimension_matches": (
            None
            if embedding_dimension is None or vector_dimension is None
            else vector_dimension == embedding_dimension
        ),
        "warnings": warnings,
        "errors": errors,
    }


def check_qdrant_health(
    client: QdrantClient,
    collection_name: str,
    *,
    embedding_dimension: int | None = None,
) -> dict[str, Any]:
    """Check reachability, collection presence, counts, status, and dimensions."""

    report: dict[str, Any] = {
        "reachable": False,
        "collection_name": collection_name,
        "collection_exists": False,
        "collection_status": "missing",
        "point_count": 0,
        "indexed_vector_count": 0,
        "optimizer_status": "unknown",
        "vector_dimension": None,
        "dimension_matches": None,
        "warnings": [],
        "errors": [],
    }
    try:
        collections = client.get_collections()
        report["reachable"] = True
        exists_method = getattr(client, "collection_exists", None)
        if callable(exists_method):
            exists = bool(exists_method(collection_name))
        else:
            exists = collection_name in {
                str(_value(item, "name"))
                for item in (_value(collections, "collections", []) or [])
            }
        report["collection_exists"] = exists
        if not exists:
            return {**report, "passed": True}

        parsed = parse_collection_health(
            client.get_collection(collection_name),
            embedding_dimension=embedding_dimension,
        )
        report.update(parsed)
    except Exception as exc:
        report["errors"].append(f"Qdrant is not reachable: {exc}")

    report["passed"] = not report["errors"]
    return report
