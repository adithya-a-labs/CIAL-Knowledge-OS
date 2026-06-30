"""Embedded local Qdrant storage helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import KnowledgeOSConfig

_LOCK_MESSAGE = (
    "Embedded Qdrant storage is locked. Only one process can access the same "
    "local Qdrant path at a time. Close other notebooks, Python processes, and "
    "Qdrant clients using this path, then retry."
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
    """Recreate the experiment collection with cosine distance."""

    try:
        if client.collection_exists(config.qdrant_collection_name):
            client.delete_collection(config.qdrant_collection_name)
        client.create_collection(
            collection_name=config.qdrant_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    except Exception as exc:
        _raise_useful_lock_error(exc)


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata.items()
    }


def index_chunks(
    client: QdrantClient,
    chunks: list[Document],
    embeddings: np.ndarray,
    config: KnowledgeOSConfig,
) -> None:
    """Index chunk text and complete metadata payloads in local Qdrant."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks and "
            f"{len(embeddings)} embeddings."
        )
    points = [
        PointStruct(
            id=index,
            vector=np.asarray(vector, dtype=float).tolist(),
            payload={
                "text": chunk.page_content,
                "metadata": _json_safe_metadata(dict(chunk.metadata)),
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    try:
        client.upsert(
            collection_name=config.qdrant_collection_name,
            points=points,
            wait=True,
        )
    except Exception as exc:
        _raise_useful_lock_error(exc)
