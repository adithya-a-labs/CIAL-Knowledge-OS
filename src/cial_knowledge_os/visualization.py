"""Lightweight, reusable tables and plots for inspectable RAG diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.documents import Document

from .benchmarking import benchmark_pipeline_steps
from .context_builder import compress_context
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

    rows: list[dict[str, Any]] = []
    for citation in citations:
        document = citation.get("source_file") or citation.get("source")
        page = citation.get("page_number")
        chunk = citation.get("chunk_id")
        score = citation.get("score")
        source = citation.get("source_path")
        rows.append(
            {
                "reference_id": citation.get("reference_id"),
                "document": document,
                "page": page,
                "chunk_id": chunk,
                "similarity_score": score,
                "source_path": source,
                "metadata_complete": all(
                    value is not None and value != ""
                    for value in (document, page, chunk, score, source)
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "reference_id",
            "document",
            "page",
            "chunk_id",
            "similarity_score",
            "source_path",
            "metadata_complete",
        ],
    )


def batch_retrieval_trace_table(
    rows_or_csv: Iterable[Mapping[str, Any]] | str | Path,
) -> pd.DataFrame:
    """Load or normalize Phase 2 batch retrieval traces for notebook display."""

    frame = _batch_frame(rows_or_csv)
    columns = [
        "question",
        "answer_status",
        "chunks_before_deduplication",
        "chunks_after_deduplication",
        "chunks_after_neighbor_expansion",
        "final_context_sections",
        "final_context_characters",
        "retrieval_latency_seconds",
        "answer_latency_seconds",
        "total_latency_seconds",
        "retrieval_trace",
    ]
    return frame.reindex(columns=columns)


def _batch_frame(
    rows_or_csv: Iterable[Mapping[str, Any]] | str | Path | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(rows_or_csv, pd.DataFrame):
        return rows_or_csv.copy()
    if isinstance(rows_or_csv, (str, Path)):
        return pd.read_csv(rows_or_csv, encoding="utf-8-sig")
    return pd.DataFrame(list(rows_or_csv))


def _variant_contribution_frame(trace: Any) -> pd.DataFrame:
    retrieved = [
        normalize_result(result)
        for result in _trace_stage_values(trace, "retrieved")
        if isinstance(result, Mapping)
    ]
    final_sections = [
        normalize_result(result)
        for result in _trace_stage_values(trace, "compressed")
        if isinstance(result, Mapping)
    ]
    variants = list(
        dict.fromkeys(
            str(variant)
            for result in retrieved
            for variant in (result.get("matched_queries") or [])
        )
    )
    rows: list[dict[str, Any]] = []
    for variant in variants:
        matching = [
            result
            for result in retrieved
            if variant in (result.get("matched_queries") or [])
        ]
        scores = [
            float(result["score"])
            for result in matching
            if result.get("score") is not None
        ]
        rows.append(
            {
                "query_variant": variant,
                "retrieval_occurrences": len(matching),
                "unique_chunks": len(
                    {chunk_identity(result) for result in matching}
                ),
                "unique_documents": len(
                    {source_path(result) for result in matching}
                ),
                "mean_similarity": (
                    round(sum(scores) / len(scores), 4) if scores else None
                ),
                "final_context_sections": sum(
                    variant in (section.get("matched_queries") or [])
                    for section in final_sections
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "query_variant",
            "retrieval_occurrences",
            "unique_chunks",
            "unique_documents",
            "mean_similarity",
            "final_context_sections",
        ],
    )


def display_query_variant_contribution_table(trace: Any) -> pd.DataFrame:
    """Return query-variant retrieval and final-context contribution metrics."""

    return _variant_contribution_frame(trace)


def plot_query_variant_contribution(trace: Any):
    """Plot unique retrieved chunks and final sections by query variant."""

    table = _variant_contribution_frame(trace)
    fig, ax = plt.subplots()
    positions = list(range(len(table)))
    width = 0.38
    ax.bar(
        [position - width / 2 for position in positions],
        table["unique_chunks"],
        width,
        label="Unique retrieved chunks",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        table["final_context_sections"],
        width,
        label="Final context sections",
    )
    ax.set(
        title="Query variant contribution",
        ylabel="Evidence count",
        xticks=positions,
        xticklabels=table["query_variant"],
    )
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    return ax


def display_top_sources_table(
    results: Iterable[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> pd.DataFrame:
    """Return the most frequent retrieved documents with score diagnostics."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    rows = retrieval_chunks_table(results, stage="source_distribution")
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "document",
                "chunk_count",
                "page_count",
                "mean_similarity",
                "maximum_similarity",
            ]
        )
    return (
        rows.groupby("document", dropna=False)
        .agg(
            chunk_count=("chunk_id", "size"),
            page_count=("page", "nunique"),
            mean_similarity=("similarity_score", "mean"),
            maximum_similarity=("similarity_score", "max"),
        )
        .sort_values(
            ["chunk_count", "maximum_similarity"],
            ascending=[False, False],
        )
        .head(limit)
        .reset_index()
    )


def plot_source_distribution(
    results: Iterable[Mapping[str, Any]],
    *,
    max_sources: int = 15,
):
    """Plot retrieval concentration across source documents."""

    table = display_top_sources_table(results, limit=max_sources)
    fig, ax = plt.subplots()
    ax.barh(
        table["document"].tolist()[::-1],
        table["chunk_count"].tolist()[::-1],
    )
    ax.set(
        title="Retrieved chunk distribution by source",
        xlabel="Chunk count",
        ylabel="Document",
    )
    fig.tight_layout()
    return ax


def plot_page_distribution(
    results: Iterable[Mapping[str, Any]],
    *,
    max_pages: int = 20,
):
    """Plot where retrieved chunks are localized by document and page."""

    if max_pages <= 0:
        raise ValueError("max_pages must be greater than zero.")
    table = retrieval_chunks_table(results, stage="page_distribution")
    if not table.empty:
        table["page_label"] = table["page"].fillna("Not provided").astype(str)
        grouped = (
            table.groupby(["document", "page_label"], dropna=False)
            .size()
            .reset_index(name="chunk_count")
            .sort_values("chunk_count", ascending=False)
            .head(max_pages)
        )
    else:
        grouped = pd.DataFrame(
            columns=["document", "page_label", "chunk_count"]
        )
    labels = [
        f"{row.document} | page {row.page_label}"
        for row in grouped.itertuples()
    ]
    fig, ax = plt.subplots()
    ax.barh(labels[::-1], grouped["chunk_count"].tolist()[::-1])
    ax.set(
        title="Retrieved chunks by document page",
        xlabel="Chunk count",
        ylabel="Document and page",
    )
    fig.tight_layout()
    return ax


def plot_score_distribution(
    results: Iterable[Mapping[str, Any]],
    *,
    bins: int | str = "auto",
):
    """Plot the similarity-score distribution for retrieved evidence."""

    scores = [
        float(result["score"])
        for result in (normalize_result(value) for value in results)
        if result.get("score") is not None
    ]
    fig, ax = plt.subplots()
    ax.hist(scores, bins=bins)
    ax.set(
        title="Retrieval similarity-score distribution",
        xlabel="Similarity score",
        ylabel="Chunk frequency",
    )
    if scores:
        ax.axvline(
            sum(scores) / len(scores),
            linestyle="--",
            label=f"Mean: {sum(scores) / len(scores):.3f}",
        )
        ax.legend()
    fig.tight_layout()
    return ax


def plot_score_by_query_variant(trace: Any):
    """Compare retrieved similarity scores across query transformations."""

    grouped: dict[str, list[float]] = {}
    for raw_result in _trace_stage_values(trace, "retrieved"):
        if not isinstance(raw_result, Mapping):
            continue
        result = normalize_result(raw_result)
        score = result.get("score")
        if score is None:
            continue
        for variant in result.get("matched_queries") or []:
            grouped.setdefault(str(variant), []).append(float(score))
    fig, ax = plt.subplots()
    labels = list(grouped)
    values = [grouped[label] for label in labels]
    if values:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set(
        title="Similarity scores by query variant",
        xlabel="Query variant",
        ylabel="Similarity score",
    )
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return ax


def _context_compression_metrics(trace: Any) -> dict[str, float]:
    merged = [
        result
        for result in _trace_stage_values(trace, "merged")
        if isinstance(result, Mapping)
    ]
    _, uncompressed_context = compress_context(
        merged,
        max_chars=1,
        enabled=False,
    )
    merged_characters = len(uncompressed_context)
    final_characters = len(_trace_context(trace))
    ratio = (
        final_characters / merged_characters
        if merged_characters
        else 0.0
    )
    return {
        "merged_evidence_characters": float(merged_characters),
        "final_context_characters": float(final_characters),
        "retained_ratio": ratio,
        "compression_percent": (1.0 - ratio) * 100.0,
    }


def plot_context_compression_ratio(trace: Any):
    """Plot character-based context size before and after compression."""

    metrics = _context_compression_metrics(trace)
    fig, ax = plt.subplots()
    bars = ax.bar(
        ["Uncompressed merged context", "Final prompt context"],
        [
            metrics["merged_evidence_characters"],
            metrics["final_context_characters"],
        ],
    )
    ax.bar_label(bars, fmt="%.0f")
    ax.set(
        title=(
            "Context compression (characters) | retained "
            f"{metrics['retained_ratio']:.1%}"
        ),
        ylabel="Characters",
    )
    fig.tight_layout()
    return ax


def display_context_sections_table(trace: Any) -> pd.DataFrame:
    """Return final context sections with size and citation metadata."""

    sections = [
        result
        for result in _trace_stage_values(trace, "compressed")
        if isinstance(result, Mapping)
    ]
    table = retrieval_chunks_table(sections, stage="final_context")
    if table.empty:
        table["section_characters"] = pd.Series(dtype="int64")
        table["context_truncated"] = pd.Series(dtype="bool")
        return table
    table["section_characters"] = [
        len(str(result.get("text") or "")) for result in sections
    ]
    table["context_truncated"] = [
        bool(result.get("context_truncated")) for result in sections
    ]
    return table


def plot_context_section_lengths(trace: Any):
    """Plot whether final context is balanced across merged sections."""

    table = display_context_sections_table(trace)
    fig, ax = plt.subplots()
    labels = [
        f"[{rank}] {document}"
        for rank, document in zip(
            table["rank"],
            table["document"],
            strict=True,
        )
    ]
    ax.bar(labels, table["section_characters"])
    ax.set(
        title="Final context section lengths",
        xlabel="Context section",
        ylabel="Characters",
    )
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return ax


def plot_retrieval_funnel(trace: Any):
    """Plot chunk counts from retrieval through final compressed context."""

    table = context_stage_counts_table(trace)
    fig, ax = plt.subplots()
    bars = ax.barh(
        table["stage"].tolist()[::-1],
        table["section_count"].tolist()[::-1],
    )
    ax.bar_label(bars)
    ax.set(
        title="Phase 2 retrieval and context funnel",
        xlabel="Chunk or section count",
        ylabel="Pipeline stage",
    )
    fig.tight_layout()
    return ax


def display_retrieval_trace_table(
    rows_or_csv: Iterable[Mapping[str, Any]] | str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Return concise per-question Phase 2 retrieval audit trails."""

    return batch_retrieval_trace_table(rows_or_csv)


def display_low_score_chunks_table(
    results: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    limit: int = 20,
) -> pd.DataFrame:
    """Return the weakest scored chunks for retrieval debugging."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    table = retrieval_chunks_table(results, stage="low_score_review")
    scores = pd.to_numeric(table["similarity_score"], errors="coerce")
    return (
        table.loc[scores < threshold]
        .sort_values("similarity_score", ascending=True)
        .head(limit)
        .reset_index(drop=True)
    )


def display_high_duplicate_chunks_table(
    results: Iterable[Mapping[str, Any]],
    *,
    minimum_frequency: int = 2,
    limit: int = 20,
) -> pd.DataFrame:
    """Return canonical chunks repeated most often across query variants."""

    if minimum_frequency < 2:
        raise ValueError("minimum_frequency must be at least 2.")
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    table = duplicate_chunk_frequency_table(results)
    return (
        table.loc[table["frequency"] >= minimum_frequency]
        .head(limit)
        .reset_index(drop=True)
    )


def plot_answer_status_distribution(
    rows_or_csv: Iterable[Mapping[str, Any]] | str | Path | pd.DataFrame,
):
    """Plot answered versus insufficient-evidence batch outcomes."""

    frame = _batch_frame(rows_or_csv)
    statuses = (
        frame.get("answer_status", pd.Series(dtype="object"))
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )
    fig, ax = plt.subplots()
    bars = ax.bar(statuses.index.tolist(), statuses.values.tolist())
    ax.bar_label(bars)
    ax.set(
        title="Batch answer status distribution",
        xlabel="Answer status",
        ylabel="Question count",
    )
    fig.tight_layout()
    return ax


def plot_latency_by_question(
    rows_or_csv: Iterable[Mapping[str, Any]] | str | Path | pd.DataFrame,
    *,
    max_questions: int = 20,
):
    """Plot the slowest batch questions by total execution time."""

    if max_questions <= 0:
        raise ValueError("max_questions must be greater than zero.")
    frame = _batch_frame(rows_or_csv)
    working = frame.reindex(
        columns=["question", "total_latency_seconds"]
    ).copy()
    working["total_latency_seconds"] = pd.to_numeric(
        working["total_latency_seconds"],
        errors="coerce",
    )
    working = (
        working.dropna(subset=["total_latency_seconds"])
        .sort_values("total_latency_seconds", ascending=False)
        .head(max_questions)
    )
    labels = [
        question if len(question) <= 70 else question[:67] + "..."
        for question in working["question"].fillna("").astype(str)
    ]
    fig, ax = plt.subplots()
    ax.barh(
        labels[::-1],
        working["total_latency_seconds"].tolist()[::-1],
    )
    ax.set(
        title="Slowest batch questions",
        xlabel="Total execution time (seconds)",
        ylabel="Question",
    )
    fig.tight_layout()
    return ax
