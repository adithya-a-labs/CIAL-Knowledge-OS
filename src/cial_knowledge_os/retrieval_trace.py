"""Question-level Phase 3 retrieval, context, and generation traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metadata import chunk_identity
from .retrievers import default_bm25_tokenize
from .token_budget import TokenManager


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _source(result: Mapping[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(
        metadata.get("file_name")
        or result.get("source")
        or metadata.get("source")
        or "Unknown"
    )


def _page(result: Mapping[str, Any]) -> Any:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return result.get("page_number", metadata.get("page_number"))


def _chunk_id(result: Mapping[str, Any]) -> Any:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return result.get("chunk_id", metadata.get("chunk_id"))


def _result_record(
    result: Mapping[str, Any],
    *,
    rank: int,
    token_manager: TokenManager,
    query: str = "",
    query_variant: str = "",
    link_resolver: Any | None = None,
) -> dict[str, Any]:
    text = str(result.get("text") or "")
    retrieval_sources = [
        str(value) for value in (result.get("retrieval_sources") or [])
    ]
    query_terms = set(default_bm25_tokenize(query))
    matched_terms = sorted(
        query_terms.intersection(default_bm25_tokenize(text))
    )
    link = None
    build_link = getattr(link_resolver, "build", None)
    if callable(build_link):
        link = build_link(result)
    return {
        "rank": rank,
        "source": _source(result),
        "page": _page(result),
        "chunk_id": _chunk_id(result),
        "score": result.get("score"),
        "rrf_score": result.get("rrf_score"),
        "dense_rank": (result.get("retrieval_ranks") or {}).get("dense"),
        "bm25_rank": (result.get("retrieval_ranks") or {}).get("bm25"),
        "retrieval_sources": retrieval_sources,
        "retrieval_source": (
            "both"
            if {"dense", "bm25"}.issubset(set(retrieval_sources))
            else (retrieval_sources[0] if retrieval_sources else "unknown")
        ),
        "matched_terms": matched_terms,
        "query_variant": query_variant,
        "query": query,
        "token_count": token_manager.count(text),
        "text_preview": " ".join(text.split())[:500],
        "citation_link": link,
    }


def _stage_records(
    response: Mapping[str, Any],
    stage: str,
    *,
    token_manager: TokenManager,
    link_resolver: Any | None,
) -> list[dict[str, Any]]:
    stages = response.get("context_stages")
    stages = stages if isinstance(stages, Mapping) else {}
    return [
        _result_record(
            item,
            rank=rank,
            token_manager=token_manager,
            link_resolver=link_resolver,
        )
        | {
            "is_neighbor": bool(item.get("is_neighbor")),
            "context_truncated": bool(item.get("context_truncated")),
        }
        for rank, item in enumerate(_items(stages.get(stage)), start=1)
    ]


def _deduplicated_identities(
    values: Sequence[Mapping[str, Any]],
) -> set[tuple[Any, ...]]:
    return {chunk_identity(value) for value in values}


def _diagnostics(
    *,
    dense_ids: set[tuple[Any, ...]],
    bm25_ids: set[tuple[Any, ...]],
    final_records: Sequence[Mapping[str, Any]],
    token_utilization: float,
    latency: Mapping[str, float],
    unique_sources: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    final_sources = {
        str(source)
        for record in final_records
        for source in (record.get("retrieval_sources") or [])
    }
    bm25_only_final = any(
        record.get("retrieval_source") == "bm25" for record in final_records
    )
    union = dense_ids | bm25_ids
    overlap_ratio = len(dense_ids & bm25_ids) / len(union) if union else 1.0
    if token_utilization > 95:
        messages.append(
            {
                "signal": "token_budget",
                "recommendation": (
                    "Context is hitting the token budget. Consider lowering "
                    "retrieval depth or increasing the token budget."
                ),
            }
        )
    else:
        messages.append(
            {
                "signal": "token_budget",
                "recommendation": "Context remains below the configured token limit.",
            }
        )
    if bm25_only_final:
        messages.append(
            {
                "signal": "bm25_value",
                "recommendation": (
                    "BM25 contributed unique final evidence. Hybrid retrieval "
                    "was useful for this question."
                ),
            }
        )
    elif "bm25" in final_sources:
        messages.append(
            {
                "signal": "bm25_value",
                "recommendation": (
                    "BM25 agreed with dense retrieval but did not uniquely "
                    "supply a final chunk."
                ),
            }
        )
    else:
        messages.append(
            {
                "signal": "bm25_value",
                "recommendation": "BM25 did not contribute to final context.",
            }
        )
    if overlap_ratio < 0.25:
        messages.append(
            {
                "signal": "retriever_agreement",
                "recommendation": (
                    "Dense and BM25 overlap is low. Inspect query transformations "
                    "or test a larger candidate depth."
                ),
            }
        )
    else:
        messages.append(
            {
                "signal": "retriever_agreement",
                "recommendation": (
                    f"Dense/BM25 candidate overlap ratio is {overlap_ratio:.1%}."
                ),
            }
        )
    retrieval_latency = float(latency.get("retrieval_seconds") or 0.0)
    generation_latency = float(latency.get("generation_seconds") or 0.0)
    if generation_latency > max(0.001, retrieval_latency * 2):
        messages.append(
            {
                "signal": "latency",
                "recommendation": (
                    "Generation is the latency bottleneck. Consider smaller "
                    "context, lower generation limits, or a shorter answer style."
                ),
            }
        )
    else:
        messages.append(
            {
                "signal": "latency",
                "recommendation": "Generation is not more than twice retrieval latency.",
            }
        )
    if unique_sources <= 1 and final_records:
        messages.append(
            {
                "signal": "source_diversity",
                "recommendation": (
                    "Evidence is concentrated in one document. Check whether "
                    "broader support is needed."
                ),
            }
        )
    else:
        messages.append(
            {
                "signal": "source_diversity",
                "recommendation": (
                    f"Final context represents {unique_sources} unique documents."
                ),
            }
        )
    return messages


def build_question_trace(
    *,
    question: str,
    response: Mapping[str, Any],
    modality_results_by_query: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    token_manager: TokenManager,
    config: Any,
    metrics: Mapping[str, Any],
    link_resolver: Any | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe, decision-focused trace for one answered question."""

    variants = [
        dict(value)
        for value in _items(response.get("query_variants"))
    ]
    dense_records: list[dict[str, Any]] = []
    bm25_records: list[dict[str, Any]] = []
    fused_records: list[dict[str, Any]] = []
    dense_raw: list[Mapping[str, Any]] = []
    bm25_raw: list[Mapping[str, Any]] = []
    for variant in variants:
        query = str(variant.get("query") or "")
        technique = str(variant.get("technique") or "")
        modality = modality_results_by_query.get(query, {})
        dense_values = _items(modality.get("dense"))
        bm25_values = _items(modality.get("bm25"))
        fused_values = _items(modality.get("fused"))
        dense_raw.extend(dense_values)
        bm25_raw.extend(bm25_values)
        dense_records.extend(
            _result_record(
                item,
                rank=rank,
                token_manager=token_manager,
                query=query,
                query_variant=technique,
                link_resolver=link_resolver,
            )
            for rank, item in enumerate(dense_values, start=1)
        )
        bm25_records.extend(
            _result_record(
                item,
                rank=rank,
                token_manager=token_manager,
                query=query,
                query_variant=technique,
                link_resolver=link_resolver,
            )
            for rank, item in enumerate(bm25_values, start=1)
        )
        fused_records.extend(
            _result_record(
                item,
                rank=rank,
                token_manager=token_manager,
                query=query,
                query_variant=technique,
                link_resolver=link_resolver,
            )
            for rank, item in enumerate(fused_values, start=1)
        )

    stage_names = ("retrieved", "deduplicated", "expanded", "merged", "compressed")
    stage_records = {
        stage: _stage_records(
            response,
            stage,
            token_manager=token_manager,
            link_resolver=link_resolver,
        )
        for stage in stage_names
    }
    dense_ids = _deduplicated_identities(dense_raw)
    bm25_ids = _deduplicated_identities(bm25_raw)
    overlap = dense_ids & bm25_ids
    final_records = stage_records["compressed"]
    token_usage_value = response.get("token_usage")
    token_usage = (
        dict(token_usage_value)
        if isinstance(token_usage_value, Mapping)
        else {}
    )
    used = int(token_usage.get("context_tokens") or token_usage.get("used") or 0)
    budget = token_usage.get("budget")
    remaining = token_usage.get("remaining")
    utilization = (
        round(100 * used / int(budget), 2)
        if budget not in {None, "", 0}
        else 0.0
    )
    sources = {record["source"] for record in final_records}
    pages = {
        (record["source"], record["page"])
        for record in final_records
        if record.get("page") not in {None, ""}
    }
    latency = {
        "retrieval_seconds": float(metrics.get("retrieval_latency") or 0.0),
        "context_construction_seconds": float(
            metrics.get("context_construction_latency") or 0.0
        ),
        "generation_seconds": float(metrics.get("generation_latency") or 0.0),
        "total_pipeline_seconds": float(
            metrics.get("total_pipeline_latency") or 0.0
        ),
        "artifact_export_seconds": None,
    }
    citations_value = response.get("citations")
    citations = []
    for index, citation in enumerate(_items(citations_value)):
        final = final_records[index] if index < len(final_records) else {}
        citations.append(
            dict(citation)
            | {
                "retrieval_sources": final.get("retrieval_sources") or [],
                "retrieval_source": final.get("retrieval_source") or "unknown",
            }
        )
    context = str(response.get("context") or "")
    answer = str(response.get("raw_answer") or response.get("answer") or "")
    prompt = str(response.get("prompt") or "")
    stage_counts = {
        "dense_raw": len(dense_records),
        "bm25_raw": len(bm25_records),
        "combined": len(dense_records) + len(bm25_records),
        "fused": len(fused_records),
        **{stage: len(records) for stage, records in stage_records.items()},
    }
    stage_tokens = {
        **{
            stage: sum(int(record["token_count"]) for record in records)
            for stage, records in stage_records.items()
        },
        "final_context": token_manager.count(context),
    }
    trace = {
        "question": question,
        "retrieval_mode": response.get("retrieval_mode"),
        "query_variants": variants,
        "dense_results": dense_records,
        "bm25_results": bm25_records,
        "fused_results": fused_records,
        "overlap": {
            "dense_only_count": len(dense_ids - bm25_ids),
            "bm25_only_count": len(bm25_ids - dense_ids),
            "both_count": len(overlap),
            "union_count": len(dense_ids | bm25_ids),
        },
        "deduplication": {
            "before": len(stage_records["retrieved"]),
            "after": len(stage_records["deduplicated"]),
            "duplicates_removed": max(
                0,
                len(stage_records["retrieved"])
                - len(stage_records["deduplicated"]),
            ),
            "key": "source + page + chunk_id",
        },
        "neighbor_expansion": {
            "original_chunks": len(stage_records["deduplicated"]),
            "neighbors_added": sum(
                bool(record.get("is_neighbor"))
                for record in stage_records["expanded"]
            ),
            "total_after_expansion": len(stage_records["expanded"]),
            "expanded_chunks": stage_records["expanded"],
        },
        "context_funnel": {
            "counts": stage_counts,
            "token_counts": stage_tokens,
        },
        "token_usage": token_usage
        | {
            "max_context_tokens": budget,
            "context_tokens_used": used,
            "remaining_tokens": remaining,
            "chunks_included": len(final_records),
            "chunks_skipped": int(token_usage.get("omitted_sections") or 0),
            "utilization_percent": utilization,
        },
        "final_context": context,
        "final_context_chunks": final_records,
        "generation": {
            "model_name": getattr(config, "ollama_model_name", ""),
            "prompt_tokens": token_manager.count(prompt),
            "context_tokens": token_manager.count(context),
            "answer_tokens": token_manager.count(answer),
            "latency_seconds": latency["generation_seconds"],
            "status": response.get("answer_status"),
        },
        "answer": answer,
        "citations": citations,
        "latency": latency,
        "source_diversity": {
            "unique_source_count": len(sources),
            "unique_page_count": len(pages),
            "sources": sorted(sources),
        },
        "artifacts": {},
    }
    trace["decision_summary"] = _diagnostics(
        dense_ids=dense_ids,
        bm25_ids=bm25_ids,
        final_records=final_records,
        token_utilization=utilization,
        latency=latency,
        unique_sources=len(sources),
    )
    return trace
