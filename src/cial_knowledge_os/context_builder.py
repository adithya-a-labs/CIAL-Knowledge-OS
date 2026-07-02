"""Inspectable Phase 2 evidence-to-context construction."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from .config import Phase2Config
from .metadata import (
    RetrievalResult,
    chunk_index,
    normalize_result,
    page_number,
    source_label,
    source_path,
)
from .retrieval_postprocessing import deduplicate_results, expand_neighbor_chunks

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_RESPONSE = (
    "The retrieved documents do not contain sufficient evidence to answer this "
    "question. Based only on the indexed corpus, no reliable answer could be generated."
)


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """All inspectable stages produced by context construction."""

    retrieved: list[RetrievalResult]
    deduplicated: list[RetrievalResult]
    expanded: list[RetrievalResult]
    merged: list[RetrievalResult]
    compressed: list[RetrievalResult]
    context: str

    def stage_counts(self) -> dict[str, int]:
        return {
            "retrieved": len(self.retrieved),
            "deduplicated": len(self.deduplicated),
            "expanded": len(self.expanded),
            "merged": len(self.merged),
            "compressed": len(self.compressed),
        }


def _append_without_overlap(left: str, right: str, max_overlap: int = 250) -> str:
    left = left.rstrip()
    right = right.lstrip()
    limit = min(len(left), len(right), max_overlap)
    for size in range(limit, 0, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return f"{left}\n{right}"


def _merge_group(group: Sequence[RetrievalResult]) -> RetrievalResult:
    ordered = sorted(
        group,
        key=lambda item: chunk_index(item)
        if chunk_index(item) is not None
        else float("inf"),
    )
    merged = dict(ordered[0])
    merged["metadata"] = dict(ordered[0].get("metadata") or {})
    text = str(ordered[0].get("text", ""))
    for item in ordered[1:]:
        text = _append_without_overlap(text, str(item.get("text", "")))

    chunk_ids = [
        str(item.get("chunk_id", ""))
        for item in ordered
        if item.get("chunk_id")
    ]
    pages = list(
        dict.fromkeys(page_number(item) for item in ordered if page_number(item) is not None)
    )
    scores = [item["score"] for item in ordered if item.get("score") is not None]
    matched_queries = list(
        dict.fromkeys(
            query
            for item in ordered
            for query in (item.get("matched_queries") or [])
        )
    )
    if not chunk_ids:
        merged_chunk_id = ""
    elif len(chunk_ids) == 1:
        merged_chunk_id = chunk_ids[0]
    else:
        merged_chunk_id = f"{chunk_ids[0]} .. {chunk_ids[-1]}"
    merged.update(
        {
            "text": text,
            "chunk_ids": chunk_ids,
            "chunk_id": merged_chunk_id,
            "page_numbers": pages,
            "page_number": pages[0] if len(pages) == 1 else ", ".join(map(str, pages)),
            "score": max(scores) if scores else None,
            "matched_queries": matched_queries,
            "merged_chunk_count": len(ordered),
        }
    )
    return normalize_result(merged)


def merge_overlapping_chunks(
    results: Sequence[Mapping[str, Any]],
) -> list[RetrievalResult]:
    """Merge contiguous source chunks and remove splitter text overlap."""

    normalized = [normalize_result(result) for result in results]
    grouped: list[tuple[int, list[RetrievalResult]]] = []
    by_source: dict[str, list[tuple[int, RetrievalResult]]] = {}
    for position, item in enumerate(normalized):
        if chunk_index(item) is None:
            grouped.append((position, [item]))
        else:
            by_source.setdefault(source_path(item), []).append((position, item))

    for source_items in by_source.values():
        ordered = sorted(source_items, key=lambda pair: chunk_index(pair[1]) or 0)
        current_group: list[tuple[int, RetrievalResult]] = []
        previous_index: int | None = None
        for position, item in ordered:
            item_index = chunk_index(item)
            if (
                current_group
                and previous_index is not None
                and item_index != previous_index + 1
            ):
                grouped.append(
                    (
                        min(group_position for group_position, _ in current_group),
                        [group_item for _, group_item in current_group],
                    )
                )
                current_group = []
            current_group.append((position, item))
            previous_index = item_index
        if current_group:
            grouped.append(
                (
                    min(group_position for group_position, _ in current_group),
                    [group_item for _, group_item in current_group],
                )
            )
    return [_merge_group(group) for _, group in sorted(grouped, key=lambda value: value[0])]


def _header(result: Mapping[str, Any], reference_id: int) -> str:
    page = page_number(result)
    page_text = str(page) if page is not None and page != "" else "Not provided"
    score = result.get("score")
    score_text = f"{float(score):.3f}" if score is not None else "Not scored"
    return (
        f"[{reference_id}]\n"
        f"Document: {source_label(result)}\n"
        f"Page: {page_text}\n"
        f"Chunk ID: {result.get('chunk_id') or 'Not provided'}\n"
        f"Similarity Score: {score_text}\n"
    )


def compress_context(
    results: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
    enabled: bool = True,
) -> tuple[list[RetrievalResult], str]:
    """Select and truncate ranked blocks to a deterministic character budget."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    selected: list[RetrievalResult] = []
    blocks: list[str] = []
    used = 0
    for result in results:
        normalized = normalize_result(result)
        header = _header(normalized, len(selected) + 1)
        text = normalized["text"].strip()
        remaining = max_chars - used - len(header) if enabled else len(text)
        if enabled and remaining <= 0:
            break
        if enabled and len(text) > remaining:
            text = text[:remaining].rstrip()
            normalized["context_truncated"] = True
        normalized["text"] = text
        block = header + text
        selected.append(normalized)
        blocks.append(block)
        used += len(block) + 2
        if enabled and used >= max_chars:
            break
    return selected, "\n\n".join(blocks)


class ContextBuilder:
    """Compose Phase 2 post-retrieval stages without model dependencies."""

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

    def build(
        self,
        retrieved: Sequence[Mapping[str, Any]],
        *,
        corpus_chunks: Sequence[Document | Mapping[str, Any]] = (),
    ) -> ContextBuildResult:
        """Run deduplication, expansion, merging, and compression in order."""

        raw = [normalize_result(result) for result in retrieved]
        deduplicated = deduplicate_results(raw)
        expanded = (
            expand_neighbor_chunks(
                deduplicated,
                corpus_chunks,
                window=self.config.neighbor_window,
            )
            if self.config.enable_neighbor_expansion
            else deduplicated
        )
        merged = (
            merge_overlapping_chunks(expanded)
            if self.config.enable_overlap_merging
            else expanded
        )
        compressed, context = compress_context(
            merged,
            max_chars=self.config.max_context_chars,
            enabled=self.config.enable_context_compression,
        )
        logger.info(
            "Context stages: %s",
            {
                "retrieved": len(raw),
                "deduplicated": len(deduplicated),
                "expanded": len(expanded),
                "merged": len(merged),
                "compressed": len(compressed),
            },
        )
        return ContextBuildResult(
            retrieved=raw,
            deduplicated=deduplicated,
            expanded=expanded,
            merged=merged,
            compressed=compressed,
            context=context,
        )
