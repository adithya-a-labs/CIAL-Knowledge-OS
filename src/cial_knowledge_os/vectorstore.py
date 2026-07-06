"""Local embedded and self-hosted Qdrant storage helpers."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import KnowledgeOSConfig

logger = logging.getLogger(__name__)

_LOCK_MESSAGE = (
    "Embedded Qdrant storage is locked. Only one process can access the same "
    "local Qdrant path at a time. Close other Qdrant clients or restart notebook "
    "kernels using this path, then retry. Use Qdrant server mode when multiple "
    "processes need concurrent access."
)

_SERVER_ERROR_TEMPLATE = """Qdrant server mode is enabled but {url} is not reachable. Start local Qdrant with:
docker compose -f docker-compose.qdrant.yml up -d
or restart an existing container with:
docker start cial-qdrant"""


def _raise_useful_lock_error(exc: Exception) -> None:
    message = str(exc).lower()
    if any(
        token in message
        for token in (
            "lock",
            "already accessed",
            "resource busy",
            "used by another process",
            "permission denied",
            "access is denied",
        )
    ):
        raise RuntimeError(_LOCK_MESSAGE) from exc
    raise exc


def create_qdrant_client(config: KnowledgeOSConfig) -> QdrantClient:
    """Open the configured local Qdrant backend and verify server connectivity."""

    if config.qdrant_mode == "embedded":
        config.qdrant_dir.mkdir(parents=True, exist_ok=True)
        try:
            return QdrantClient(path=str(config.qdrant_dir))
        except Exception as exc:
            _raise_useful_lock_error(exc)
            raise
    if config.qdrant_mode == "server":
        client: QdrantClient | None = None
        try:
            client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
            )
            client.get_collections()
        except Exception as exc:
            if client is not None:
                client.close()
            raise RuntimeError(
                _SERVER_ERROR_TEMPLATE.format(url=config.qdrant_url)
            ) from exc
        return client
    raise ValueError("Unsupported qdrant_mode")


def reset_qdrant_storage(config: KnowledgeOSConfig) -> None:
    """Delete the configured embedded store or server collection."""

    if config.qdrant_mode == "server":
        client = create_qdrant_client(config)
        try:
            if client.collection_exists(config.qdrant_collection_name):
                client.delete_collection(config.qdrant_collection_name)
        finally:
            client.close()
        return
    if config.qdrant_mode != "embedded":
        raise ValueError("Unsupported qdrant_mode")

    path = Path(config.qdrant_dir)
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except (OSError, PermissionError) as exc:
        _raise_useful_lock_error(exc)


def recreate_collection(
    client: QdrantClient, config: KnowledgeOSConfig, vector_size: int
) -> None:
    """Destructively recreate the collection when explicitly requested.

    Normal indexing should use :func:`ensure_collection` so reruns preserve the
    existing local collection and its points.
    """

    try:
        if client.collection_exists(config.qdrant_collection_name):
            client.delete_collection(config.qdrant_collection_name)
        client.create_collection(
            collection_name=config.qdrant_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    except Exception as exc:
        _raise_useful_lock_error(exc)


def _collection_vector_size(client: QdrantClient, collection_name: str) -> int:
    """Read the size of the collection's unnamed dense-vector configuration."""

    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    if isinstance(vectors, dict):
        raise ValueError(
            f"Qdrant collection '{collection_name}' uses named vectors, but this "
            "pipeline requires one unnamed dense vector."
        )
    return int(vectors.size)


def ensure_collection(
    client: QdrantClient, config: KnowledgeOSConfig, vector_size: int
) -> None:
    """Create the local collection once and validate it on later reruns.

    An existing collection is never recreated or deleted implicitly. A dimension
    mismatch is reported clearly because vectors from different embedding models
    cannot safely coexist in the same collection.
    """

    if vector_size <= 0:
        raise ValueError("Embedding vector size must be greater than zero.")
    collection_name = config.qdrant_collection_name
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return

        existing_size = _collection_vector_size(client, collection_name)
        if existing_size != vector_size:
            raise ValueError(
                f"Qdrant collection '{collection_name}' expects vectors of size "
                f"{existing_size}, but the configured embedding model produces "
                f"{vector_size}. Use a different collection name or explicitly "
                "reset the vector store before changing embedding models."
            )
    except Exception as exc:
        _raise_useful_lock_error(exc)


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata.items()
    }


def _stable_point_id(chunk: Document) -> str:
    """Return a deterministic Qdrant-compatible UUID for a chunk location."""

    metadata = chunk.metadata
    location_parts = (
        str(metadata.get("chunk_index", "")),
        str(metadata.get("start_index", "")),
        str(metadata.get("chunk_id", "")),
    )
    identity = "|".join(
        (
            str(metadata.get("source", "")),
            str(metadata.get("page_number", "")),
            *location_parts,
            # Public callers may supply unchunked Documents. Content prevents
            # those metadata-free points from all receiving the same UUID.
            "" if any(location_parts) else chunk.page_content,
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def index_chunks(
    client: QdrantClient,
    chunks: list[Document],
    embeddings: np.ndarray,
    config: KnowledgeOSConfig,
    *,
    execution_manager: Any | None = None,
) -> None:
    """Idempotently upsert chunk text, vectors, and metadata into local Qdrant."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks and "
            f"{len(embeddings)} embeddings."
        )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D array; received shape {embeddings.shape}."
        )
    expected_size = _collection_vector_size(
        client,
        config.qdrant_collection_name,
    )
    actual_size = int(embeddings.shape[1])
    if actual_size != expected_size:
        raise ValueError(
            f"Embedding vectors have size {actual_size}, but Qdrant collection "
            f"'{config.qdrant_collection_name}' expects size {expected_size}."
        )
    if not chunks:
        return

    batch_size = config.qdrant_batch_size
    total_points = len(chunks)
    total_batches = (total_points + batch_size - 1) // batch_size
    started_at = time.perf_counter()
    for batch_index, start in enumerate(
        range(0, total_points, batch_size),
        start=1,
    ):
        end = min(start + batch_size, total_points)
        points = [
            PointStruct(
                id=_stable_point_id(chunk),
                vector=np.asarray(vector, dtype=float).tolist(),
                payload={
                    "text": chunk.page_content,
                    "metadata": _json_safe_metadata(dict(chunk.metadata)),
                },
            )
            for chunk, vector in zip(
                chunks[start:end],
                embeddings[start:end],
                strict=True,
            )
        ]
        try:
            client.upsert(
                collection_name=config.qdrant_collection_name,
                points=points,
                wait=config.qdrant_upsert_wait,
            )
        except Exception as exc:
            _raise_useful_lock_error(exc)
        elapsed = time.perf_counter() - started_at
        remaining_batches = total_batches - batch_index
        estimated_remaining_seconds = (
            elapsed / batch_index * remaining_batches
            if batch_index and remaining_batches
            else 0.0
        )
        logger.info(
            "qdrant_upsert_batch_complete",
            extra={
                "event": "qdrant_upsert",
                "collection_name": config.qdrant_collection_name,
                "qdrant_mode": config.qdrant_mode,
                "batch_index": batch_index,
                "total_batches": total_batches,
                "batch_points": len(points),
                "indexed_points": end,
                "total_points": total_points,
                "elapsed_seconds": round(elapsed, 3),
                "remaining_batches": remaining_batches,
                "estimated_remaining_seconds": round(
                    estimated_remaining_seconds,
                    3,
                ),
            },
        )
        if execution_manager is not None:
            execution_manager.emit(
                "indexing_progress",
                stage="qdrant_upsert",
                status="running",
                elapsed_seconds=elapsed,
                metrics={
                    "qdrant_upsert_latency_seconds": elapsed,
                },
                payload={
                    "collection_name": config.qdrant_collection_name,
                    "qdrant_mode": config.qdrant_mode,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "batch_size": batch_size,
                    "points_upserted": end,
                    "total_points": total_points,
                    "eta_seconds": estimated_remaining_seconds,
                },
                source="vectorstore.index_chunks",
            )
        points = []


def _document_filter(
    *,
    document_id: str | None = None,
    relative_path: str | None = None,
) -> Filter:
    conditions = []
    if document_id:
        conditions.append(
            FieldCondition(
                key="metadata.document_id",
                match=MatchValue(value=document_id),
            )
        )
    if relative_path:
        conditions.append(
            FieldCondition(
                key="metadata.relative_path",
                match=MatchValue(value=relative_path),
            )
        )
    if not conditions:
        raise ValueError("document_id or relative_path is required.")
    return Filter(should=conditions)


def delete_document_chunks(
    client: QdrantClient,
    config: KnowledgeOSConfig,
    *,
    document_id: str | None = None,
    relative_path: str | None = None,
) -> int:
    """Delete and return the number of points belonging to one document."""

    if not client.collection_exists(config.qdrant_collection_name):
        return 0
    query_filter = _document_filter(
        document_id=document_id,
        relative_path=relative_path,
    )
    try:
        removed = int(
            client.count(
                collection_name=config.qdrant_collection_name,
                count_filter=query_filter,
                exact=True,
            ).count
        )
        client.delete(
            collection_name=config.qdrant_collection_name,
            points_selector=FilterSelector(filter=query_filter),
            wait=True,
        )
        return removed
    except Exception as exc:
        _raise_useful_lock_error(exc)
        raise


def load_indexed_chunks(
    client: QdrantClient,
    config: KnowledgeOSConfig,
) -> list[Document]:
    """Read the complete stored chunk corpus for BM25 and notebook inspection."""

    if not client.collection_exists(config.qdrant_collection_name):
        return []
    chunks: list[Document] = []
    offset: Any | None = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=config.qdrant_collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunks.append(
                    Document(
                        page_content=str(payload.get("text", "")),
                        metadata=dict(payload.get("metadata") or {}),
                    )
                )
            if offset is None:
                break
    except Exception as exc:
        _raise_useful_lock_error(exc)
        raise
    return chunks
