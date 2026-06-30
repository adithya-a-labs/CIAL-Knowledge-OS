"""Token-conscious semantic retrieval helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .config import KnowledgeOSConfig
from .embeddings import embed_texts


def search_similar_chunks(
    client: QdrantClient,
    query: str,
    embedding_model: SentenceTransformer,
    config: KnowledgeOSConfig,
) -> list[dict[str, Any]]:
    """Embed a query locally and return normalized, inspectable search results."""

    query_vector = embed_texts(embedding_model, [query])[0]
    response = client.query_points(
        collection_name=config.qdrant_collection_name,
        query=query_vector.tolist(),
        limit=config.top_k,
        with_payload=True,
    )
    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        metadata = dict(payload.get("metadata") or {})
        results.append(
            {
                "id": point.id,
                "score": float(point.score),
                "text": str(payload.get("text", "")),
                "metadata": metadata,
                "source": str(metadata.get("file_name") or metadata.get("source", "")),
                "page_number": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id"),
            }
        )
    return results


def format_retrieved_context(
    results: list[dict[str, Any]], max_chars: int
) -> str:
    """Format a bounded context with citation metadata."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    blocks: list[str] = []
    used = 0
    for rank, result in enumerate(results, start=1):
        source = Path(str(result.get("source") or "unknown")).name
        page = result.get("page_number")
        page_label = f"p. {page}" if page is not None else "page n/a"
        header = (
            f"[{rank}] {source} | {page_label} | chunk "
            f"{result.get('chunk_id', 'n/a')} | score {result.get('score', 0.0):.3f}\n"
        )
        remaining = max_chars - used - len(header)
        if remaining <= 0:
            break
        text = str(result.get("text", "")).strip()[:remaining]
        block = header + text
        blocks.append(block)
        used += len(block) + 2
        if used >= max_chars:
            break
    return "\n\n".join(blocks)


def print_retrieval_results(results: list[dict[str, Any]]) -> None:
    """Print scores, citation fields, and compact text previews."""

    if not results:
        print("No chunks retrieved.")
        return
    for rank, result in enumerate(results, start=1):
        preview = " ".join(str(result.get("text", "")).split())
        if len(preview) > 240:
            preview = preview[:237] + "..."
        print(
            f"{rank}. score={result.get('score', 0.0):.4f} | "
            f"source={result.get('source') or 'unknown'} | "
            f"page={result.get('page_number') or 'n/a'} | "
            f"chunk={result.get('chunk_id') or 'n/a'}"
        )
        print(f"   {preview}")
