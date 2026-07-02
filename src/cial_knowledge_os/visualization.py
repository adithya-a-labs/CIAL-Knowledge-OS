"""Lightweight, reusable tables and plots for inspectable RAG diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.documents import Document

from .benchmarking import benchmark_pipeline_steps
from .metadata import (
    chunk_identity,
    chunk_index,
    normalize_result,
    page_number,
    source_label,
    source_path,
)


def plot_chunk_size_distribution(chunks: list[Document]):
    """Plot chunk character counts."""

    fig, ax = plt.subplots()
    ax.hist([len(chunk.page_content) for chunk in chunks], bins="auto")
    ax.set(title="Chunk size distribution", xlabel="Characters", ylabel="Chunks")
    fig.tight_layout()
    return ax


def plot_retrieval_scores(results: list[dict[str, Any]]):
    """Plot ranked retrieval similarity scores."""

    fig, ax = plt.subplots()
    ranks = list(range(1, len(results) + 1))
    ax.bar(ranks, [float(result.get("score", 0.0)) for result in results])
    ax.set(
        title="Retrieval scores",
        xlabel="Result rank",
        ylabel="Cosine similarity",
        xticks=ranks,
    )
    fig.tight_layout()
    return ax


def plot_timing_breakdown(metrics: dict[str, Any]):
    """Plot available standard pipeline timings."""

    benchmark = benchmark_pipeline_steps(metrics)
    fig, ax = plt.subplots()
    ax.bar(
        [name.replace("_", " ") for name in benchmark],
        list(benchmark.values()),
    )
    ax.set(title="Pipeline timing breakdown", ylabel="Seconds")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return ax


def query_variants_table(
    variants: Iterable[Any],
) -> pd.DataFrame:
    """Build an inspectable table of generated query variants."""

    rows: list[dict[str, Any]] = []
    original_query = ""
    for position, variant in enumerate(variants, start=1):
        if isinstance(variant, Mapping):
            technique = str(variant.get("technique") or "")
            query = str(variant.get("query") or "")
        else:
            technique = str(getattr(variant, "technique", ""))
            query = str(getattr(variant, "query", ""))
        if not original_query:
            original_query = query
        rows.append(
            {
                "variant_order": position,
                "technique": technique,
                "query": query,
                "changed_from_original": query.casefold()
                != original_query.casefold(),
                "characters": len(query),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "variant_order",
            "technique",
            "query",
            "changed_from_original",
            "characters",
        ],
    )


def retrieval_chunks_table(
    results: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    preview_characters: int = 140,
) -> pd.DataFrame:
    """Normalize retrieval evidence into a metadata-rich debugging table."""

    if preview_characters <= 0:
        raise ValueError("preview_characters must be greater than zero.")
    rows: list[dict[str, Any]] = []
    for rank, raw_result in enumerate(results, start=1):
        result = normalize_result(raw_result)
        text = " ".join(result["text"].split())
        if len(text) > preview_characters:
            text = text[: preview_characters - 3].rstrip() + "..."
        rows.append(
            {
                "stage": stage,
                "rank": rank,
                "document": source_label(result),
                "page": page_number(result),
                "chunk_id": result.get("chunk_id") or None,
                "chunk_index": chunk_index(result),
                "similarity_score": result.get("score"),
                "evidence_role": (
                    "added_neighbor"
                    if result.get("is_neighbor")
                    else "retrieved"
                ),
                "matched_queries": ", ".join(
                    str(value)
                    for value in (result.get("matched_queries") or [])
                ),
                "text_preview": text,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "stage",
            "rank",
            "document",
            "page",
            "chunk_id",
            "chunk_index",
            "similarity_score",
            "evidence_role",
            "matched_queries",
            "text_preview",
        ],
    )


def retrieval_comparison_table(
    single_query_results: Sequence[Mapping[str, Any]],
    multi_query_results: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Compare retrieval volume, diversity, and scores across two modes."""

    rows: list[dict[str, Any]] = []
    for label, results in (
        ("single_query", single_query_results),
        ("multi_query", multi_query_results),
    ):
        normalized = [normalize_result(result) for result in results]
        scores = [
            float(result["score"])
            for result in normalized
            if result.get("score") is not None
        ]
        rows.append(
            {
                "retrieval_mode": label,
                "returned_chunks": len(normalized),
                "unique_chunks": len(
                    {chunk_identity(result) for result in normalized}
                ),
                "unique_documents": len(
                    {source_path(result) for result in normalized}
                ),
                "mean_similarity": (
                    round(sum(scores) / len(scores), 4) if scores else None
                ),
                "maximum_similarity": round(max(scores), 4) if scores else None,
            }
        )
    return pd.DataFrame(rows)


def plot_retrieval_comparison(
    single_query_results: Sequence[Mapping[str, Any]],
    multi_query_results: Sequence[Mapping[str, Any]],
):
    """Plot returned and unique chunk counts for single vs multi-query retrieval."""

    table = retrieval_comparison_table(
        single_query_results,
        multi_query_results,
    )
    fig, ax = plt.subplots()
    positions = list(range(len(table)))
    width = 0.36
    ax.bar(
        [position - width / 2 for position in positions],
        table["returned_chunks"],
        width,
        label="Returned chunks",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        table["unique_chunks"],
        width,
        label="Unique chunks",
    )
    ax.set(
        title="Single-query vs multi-query retrieval",
        ylabel="Chunk count",
        xticks=positions,
        xticklabels=table["retrieval_mode"],
    )
    ax.legend()
    fig.tight_layout()
    return ax


def duplicate_chunk_frequency_table(
    results: Iterable[Mapping[str, Any]],
    *,
    duplicates_only: bool = False,
) -> pd.DataFrame:
    """Count evidence frequency by the canonical chunk identity."""

    grouped: dict[tuple[str, Any, str], dict[str, Any]] = {}
    for raw_result in results:
        result = normalize_result(raw_result)
        identity = chunk_identity(result)
        row = grouped.setdefault(
            identity,
            {
                "source": identity[0],
                "document": source_label(result),
                "page": identity[1],
                "chunk_id": identity[2],
                "frequency": 0,
                "maximum_similarity": None,
                "matched_queries": [],
            },
        )
        row["frequency"] += 1
        score = result.get("score")
        if score is not None and (
            row["maximum_similarity"] is None
            or float(score) > row["maximum_similarity"]
        ):
            row["maximum_similarity"] = float(score)
        row["matched_queries"] = list(
            dict.fromkeys(
                [
                    *row["matched_queries"],
                    *[
                        str(value)
                        for value in (result.get("matched_queries") or [])
                    ],
                ]
            )
        )

    rows = [
        {
            **row,
            "is_duplicate": row["frequency"] > 1,
            "matched_queries": ", ".join(row["matched_queries"]),
        }
        for row in grouped.values()
        if not duplicates_only or row["frequency"] > 1
    ]
    rows.sort(
        key=lambda row: (
            -int(row["frequency"]),
            str(row["document"]),
            str(row["chunk_id"]),
        )
    )
    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "document",
            "page",
            "chunk_id",
            "frequency",
            "is_duplicate",
            "maximum_similarity",
            "matched_queries",
        ],
    )


def plot_duplicate_chunk_frequency(
    results: Iterable[Mapping[str, Any]],
    *,
    max_chunks: int = 15,
):
    """Plot the most frequently repeated canonical chunks."""

    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than zero.")
    table = duplicate_chunk_frequency_table(results).head(max_chunks)
    fig, ax = plt.subplots()
    labels = [
        f"{row.document} | p.{row.page} | {row.chunk_id}"
        for row in table.itertuples()
    ]
    ax.barh(labels[::-1], table["frequency"].tolist()[::-1])
    ax.set(
        title="Duplicate chunk frequency before deduplication",
        xlabel="Retrieval occurrences",
        ylabel="Canonical chunk identity",
    )
    fig.tight_layout()
    return ax


def neighbor_expansion_table(
    retrieved: Sequence[Mapping[str, Any]],
    expanded: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Show seed chunks and adjacent chunks introduced by expansion."""

    retrieved_identities = {
        chunk_identity(normalize_result(result)) for result in retrieved
    }
    rows: list[dict[str, Any]] = []
    for raw_result in expanded:
        result = normalize_result(raw_result)
        identity = chunk_identity(result)
        rows.append(
            {
                "document": source_label(result),
                "page": page_number(result),
                "chunk_id": result.get("chunk_id") or None,
                "chunk_index": chunk_index(result),
                "expansion_role": (
                    "retrieved_seed"
                    if identity in retrieved_identities
                    else "added_adjacent_chunk"
                ),
                "neighbor_offset": result.get("neighbor_offset"),
                "seed_chunk_id": result.get("seed_chunk_id"),
                "similarity_score": result.get("score"),
            }
        )
    return pd.DataFrame(rows)


def _trace_stage_values(trace: Any, stage: str) -> list[Any]:
    if isinstance(trace, Mapping):
        stages = trace.get("context_stages")
        if isinstance(stages, Mapping):
            value = stages.get(stage)
        else:
            value = trace.get(stage)
    else:
        value = getattr(trace, stage, None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _trace_context(trace: Any) -> str:
    if isinstance(trace, Mapping):
        return str(trace.get("context") or "")
    return str(getattr(trace, "context", "") or "")


def context_stage_counts_table(trace: Any) -> pd.DataFrame:
    """Summarize every context construction stage and final context length."""

    stages = [
        ("initial_retrieved_chunks", "retrieved"),
        ("after_deduplication", "deduplicated"),
        ("after_neighbor_expansion", "expanded"),
        ("after_overlap_merging", "merged"),
        ("final_context_sections", "compressed"),
    ]
    context_characters = len(_trace_context(trace))
    return pd.DataFrame(
        [
            {
                "stage": label,
                "section_count": len(_trace_stage_values(trace, key)),
                "final_context_characters": (
                    context_characters if key == "compressed" else None
                ),
            }
            for label, key in stages
        ]
    )


def plot_context_stage_counts(trace: Any):
    """Plot evidence reduction and expansion through context construction."""

    table = context_stage_counts_table(trace)
    context_characters = int(
        table["final_context_characters"].dropna().iloc[0]
        if table["final_context_characters"].notna().any()
        else 0
    )
    fig, ax = plt.subplots()
    bars = ax.bar(table["stage"], table["section_count"])
    ax.bar_label(bars)
    ax.set(
        title=f"Context construction stages | final context: {context_characters:,} chars",
        ylabel="Chunk or section count",
    )
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return ax


def citation_quality_table(
    citations: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build the final citation metadata quality table."""

    rows = [
        {
            "reference_id": citation.get("reference_id"),
            "document": citation.get("source_file") or citation.get("source"),
            "page": citation.get("page_number"),
            "chunk_id": citation.get("chunk_id"),
            "similarity_score": citation.get("score"),
            "source_path": citation.get("source_path"),
        }
        for citation in citations
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "reference_id",
            "document",
            "page",
            "chunk_id",
            "similarity_score",
            "source_path",
        ],
    )


def batch_retrieval_trace_table(
    rows_or_csv: Iterable[Mapping[str, Any]] | str | Path,
) -> pd.DataFrame:
    """Load or normalize Phase 2 batch retrieval traces for notebook display."""

    if isinstance(rows_or_csv, (str, Path)):
        frame = pd.read_csv(rows_or_csv, encoding="utf-8-sig")
    else:
        frame = pd.DataFrame(list(rows_or_csv))
    columns = [
        "question",
        "answer_status",
        "chunks_before_deduplication",
        "chunks_after_deduplication",
        "chunks_after_neighbor_expansion",
        "final_context_sections",
        "final_context_characters",
        "retrieval_trace",
    ]
    return frame.reindex(columns=columns)
