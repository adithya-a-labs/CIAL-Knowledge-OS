"""Embedded local Qdrant storage helpers."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import KnowledgeOSConfig

_LOCK_MESSAGE = (
    "Embedded Qdrant storage is locked. Only one process can access the same "
    "local Qdrant path at a time. Close other Qdrant clients or restart notebook "
    "kernels using this path, then retry. Use Qdrant server mode when multiple "
    "processes need concurrent access."
)


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
    """Open an embedded Qdrant client at the configured local path."""

    config.qdrant_dir.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(config.qdrant_dir))
    except Exception as exc:
        _raise_useful_lock_error(exc)
        raise


def reset_qdrant_storage(config: KnowledgeOSConfig) -> None:
    """Delete local runtime storage before opening a new embedded client."""

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
    points = [
        PointStruct(
            id=_stable_point_id(chunk),
            vector=np.asarray(vector, dtype=float).tolist(),
            payload={
                "text": chunk.page_content,
                "metadata": _json_safe_metadata(dict(chunk.metadata)),
            },
        )
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    if not points:
        return
    try:
        client.upsert(
            collection_name=config.qdrant_collection_name,
            points=points,
            wait=True,
        )
    except Exception as exc:
        _raise_useful_lock_error(exc)
