"""Evidence merging, exact deduplication, and neighbor expansion."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from langchain_core.documents import Document

from .metadata import (
    RetrievalResult,
    chunk_identity,
    chunk_index,
    normalize_result,
    page_number,
    source_path,
)
from .query_transformations import QueryVariant

logger = logging.getLogger(__name__)
SearchFunction = Callable[[str], list[dict[str, Any]]]


def deduplicate_results(
    results: Iterable[Mapping[str, Any]],
) -> list[RetrievalResult]:
    """Deduplicate by ``(source, page, chunk_id)`` and retain best evidence."""

    input_count = 0
    unique: dict[tuple[str, Any, str], RetrievalResult] = {}
    for raw_result in results:
        input_count += 1
        result = normalize_result(raw_result)
        key = chunk_identity(result)
        if not key[0] or not key[2]:
            point_id = result.get("id")
            fallback = (
                f"point:{point_id}"
                if point_id is not None
                else f"occurrence:{input_count}"
            )
            key = (f"__incomplete__:{fallback}", key[1], key[2])
        existing = unique.get(key)
        if existing is None:
            unique[key] = result
            continue

        existing_queries = list(existing.get("matched_queries") or [])
        incoming_queries = list(result.get("matched_queries") or [])
        existing["matched_queries"] = list(
            dict.fromkeys([*existing_queries, *incoming_queries])
        )
        old_score = existing.get("score")
        new_score = result.get("score")
        if new_score is not None and (old_score is None or new_score > old_score):
            preserved_queries = existing["matched_queries"]
            unique[key] = result
            unique[key]["matched_queries"] = preserved_queries

    deduplicated = list(unique.values())
    deduplicated.sort(
        key=lambda item: (
            item.get("score") is not None,
            item.get("score") if item.get("score") is not None else float("-inf"),
        ),
        reverse=True,
    )
    logger.debug("Deduplicated %d results to %d", input_count, len(deduplicated))
    return deduplicated


def retrieve_multiple_queries(
    variants: Sequence[QueryVariant],
    search: SearchFunction,
    *,
    deduplicate: bool = True,
) -> tuple[list[RetrievalResult], dict[str, list[RetrievalResult]]]:
    """Retrieve per variant, merge evidence, and optionally deduplicate it."""

    merged: list[RetrievalResult] = []
    by_technique: dict[str, list[RetrievalResult]] = {}
    for variant in variants:
        query_results: list[RetrievalResult] = []
        for raw_result in search(variant.query):
            result = normalize_result(raw_result)
            result["matched_queries"] = [variant.technique]
            result["retrieval_query"] = variant.query
            query_results.append(result)
        by_technique[variant.technique] = query_results
        merged.extend(query_results)
    return (deduplicate_results(merged) if deduplicate else merged), by_technique


def _document_result(document: Document) -> RetrievalResult:
    metadata = dict(document.metadata)
    return normalize_result(
        {
            "text": document.page_content,
            "metadata": metadata,
            "source": metadata.get("file_name") or metadata.get("source"),
            "page_number": metadata.get("page_number"),
            "chunk_id": metadata.get("chunk_id"),
            "score": None,
        }
    )


def expand_neighbor_chunks(
    results: Sequence[Mapping[str, Any]],
    corpus_chunks: Sequence[Document | Mapping[str, Any]],
    *,
    window: int = 1,
) -> list[RetrievalResult]:
    """Add source-relative neighboring chunks around every retrieved chunk."""

    if window < 0:
        raise ValueError("window must be non-negative.")
    seeds = deduplicate_results(results)
    if window == 0 or not corpus_chunks:
        return seeds

    corpus_results = [
        _document_result(chunk) if isinstance(chunk, Document) else normalize_result(chunk)
        for chunk in corpus_chunks
    ]
    by_position = {
        (source_path(item), chunk_index(item)): item
        for item in corpus_results
        if chunk_index(item) is not None
    }
    expanded: list[RetrievalResult] = []
    for seed in seeds:
        seed_index = chunk_index(seed)
        if seed_index is None:
            expanded.append(seed)
            continue
        retained_seed = dict(seed)
        retained_seed["metadata"] = dict(seed["metadata"])
        retained_seed["is_neighbor"] = False
        retained_seed["seed_chunk_id"] = seed.get("chunk_id")
        retained_seed["neighbor_offset"] = 0
        expanded.append(retained_seed)
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor = by_position.get((source_path(seed), seed_index + offset))
            if neighbor is None:
                continue
            enriched = dict(neighbor)
            enriched["metadata"] = dict(neighbor["metadata"])
            enriched["matched_queries"] = list(seed.get("matched_queries") or [])
            enriched["score"] = seed.get("score")
            enriched["is_neighbor"] = offset != 0
            enriched["seed_chunk_id"] = seed.get("chunk_id")
            enriched["neighbor_offset"] = offset
            expanded.append(enriched)
    return deduplicate_results(expanded)
