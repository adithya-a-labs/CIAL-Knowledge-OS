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
from .token_budget import TokenBudgetManager, TokenBudgetUsage

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
    token_usage: TokenBudgetUsage | None = None

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
    rrf_scores = [
        float(item["rrf_score"])
        for item in ordered
        if item.get("rrf_score") is not None
    ]
    matched_queries = list(
        dict.fromkeys(
            query
            for item in ordered
            for query in (item.get("matched_queries") or [])
        )
    )
    retrieval_sources = list(
        dict.fromkeys(
            source
            for item in ordered
            for source in (item.get("retrieval_sources") or [])
        )
    )
    retrieval_ranks: dict[str, int] = {}
    retrieval_scores: dict[str, float | None] = {}
    for item in ordered:
        for source, rank in (item.get("retrieval_ranks") or {}).items():
            try:
                numeric_rank = int(rank)
            except (TypeError, ValueError):
                continue
            retrieval_ranks[source] = min(
                numeric_rank,
                retrieval_ranks.get(source, numeric_rank),
            )
        for source, score in (item.get("retrieval_scores") or {}).items():
            if score is None:
                retrieval_scores.setdefault(source, None)
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            existing_score = retrieval_scores.get(source)
            if existing_score is None or numeric_score > existing_score:
                retrieval_scores[source] = numeric_score
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
            **({"rrf_score": max(rrf_scores)} if rrf_scores else {}),
            **(
                {
                    "retrieval_sources": retrieval_sources,
                    "retrieval_ranks": retrieval_ranks,
                    "retrieval_scores": retrieval_scores,
                }
                if retrieval_sources
                else {}
            ),
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
    score_label = (
        "RRF Score"
        if result.get("rrf_score") is not None
        else "Similarity Score"
    )
    return (
        f"[{reference_id}]\n"
        f"Document: {source_label(result)}\n"
        f"Page: {page_text}\n"
        f"Chunk ID: {result.get('chunk_id') or 'Not provided'}\n"
        f"{score_label}: {score_text}\n"
    )


def compress_context(
    results: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
    enabled: bool = True,
    token_budget_manager: TokenBudgetManager | None = None,
) -> tuple[list[RetrievalResult], str]:
    """Select and truncate ranked blocks to the configured context budget.

    Supplying ``token_budget_manager`` activates tokenizer-aware fitting.
    Omitting it preserves the exact Phase 2 character behavior.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    selected: list[RetrievalResult] = []
    blocks: list[str] = []
    used = 0
    truncated_sections = 0
    omitted_sections = 0
    for position, result in enumerate(results):
        normalized = normalize_result(result)
        header = _header(normalized, len(selected) + 1)
        text = normalized["text"].strip()
        separator = "\n\n" if blocks else ""
        if enabled and token_budget_manager is not None:
            prefix_tokens = token_budget_manager.count(separator + header)
            consumed_tokens = used + prefix_tokens
            if consumed_tokens >= token_budget_manager.max_tokens:
                omitted_sections = len(results) - position
                break
            remaining_tokens = token_budget_manager.remaining(
                used_tokens=consumed_tokens,
                max_tokens=token_budget_manager.max_tokens,
            )
            original_text = text
            text = token_budget_manager.truncate(text, remaining_tokens)
            if not text:
                omitted_sections = len(results) - position
                break
            candidate = "\n\n".join([*blocks, header + text])
            while (
                text
                and token_budget_manager.count(candidate)
                > token_budget_manager.max_tokens
            ):
                text = token_budget_manager.truncate(
                    text,
                    token_budget_manager.count(text) - 1,
                )
                candidate = "\n\n".join([*blocks, header + text])
            if not text:
                omitted_sections = len(results) - position
                break
            if text != original_text:
                normalized["context_truncated"] = True
                truncated_sections += 1
        else:
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
        if token_budget_manager is not None:
            used = token_budget_manager.count("\n\n".join(blocks))
        else:
            used += len(block) + 2
        if enabled and token_budget_manager is not None:
            if used >= token_budget_manager.max_tokens:
                break
        elif enabled and used >= max_chars:
            break
    context = "\n\n".join(blocks)
    if token_budget_manager is not None:
        token_budget_manager.record_usage(
            used=token_budget_manager.count(context),
            truncated_sections=truncated_sections,
            omitted_sections=omitted_sections,
        )
    return selected, context


class ContextBuilder:
    """Compose Phase 2 post-retrieval stages without model dependencies."""

    def __init__(
        self,
        config: Phase2Config,
        *,
        token_budget_manager: TokenBudgetManager | None = None,
    ) -> None:
        self.config = config
        self.token_budget_manager = token_budget_manager

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
            token_budget_manager=self.token_budget_manager,
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
            token_usage=(
                self.token_budget_manager.last_usage
                if self.token_budget_manager is not None
                else None
            ),
        )
