"""Notebook-oriented tables and decision plots for Phase 3 question traces."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .phase3_reporting import render_safe_markdown


def load_question_traces(path: str | Path) -> list[dict[str, Any]]:
    """Load the per-question trace list written to ``retrieval.json``."""

    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("retrieval.json must contain a list of question traces.")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _records(trace: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = trace.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def query_variants_frame(trace: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(_records(trace, "query_variants")).reindex(
        columns=["technique", "query"]
    )


def retrieval_results_frame(
    trace: Mapping[str, Any],
    key: str,
) -> pd.DataFrame:
    columns = [
        "rank",
        "query_variant",
        "source",
        "page",
        "chunk_id",
        "score",
        "matched_terms",
        "retrieval_source",
        "text_preview",
        "citation_link",
    ]
    frame = pd.DataFrame(_records(trace, key))
    return frame.reindex(columns=columns)


def rrf_contribution_frame(trace: Mapping[str, Any]) -> pd.DataFrame:
    columns = [
        "rank",
        "query_variant",
        "source",
        "page",
        "chunk_id",
        "dense_rank",
        "bm25_rank",
        "rrf_score",
        "retrieval_source",
    ]
    return pd.DataFrame(_records(trace, "fused_results")).reindex(columns=columns)


def funnel_frame(trace: Mapping[str, Any]) -> pd.DataFrame:
    funnel = trace.get("context_funnel")
    funnel = funnel if isinstance(funnel, Mapping) else {}
    counts = funnel.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    tokens = funnel.get("token_counts")
    tokens = tokens if isinstance(tokens, Mapping) else {}
    order = [
        "dense_raw",
        "bm25_raw",
        "combined",
        "fused",
        "retrieved",
        "deduplicated",
        "expanded",
        "merged",
        "compressed",
    ]
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "chunk_count": counts.get(stage, 0),
                "token_count": tokens.get(stage),
            }
            for stage in order
            if stage in counts
        ]
    )


def decision_frame(trace: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(_records(trace, "decision_summary")).reindex(
        columns=["signal", "recommendation"]
    )


def _final_context_html(trace: Mapping[str, Any]) -> str:
    chunks = _records(trace, "final_context_chunks")
    blocks = []
    for chunk in chunks:
        summary = (
            f"{chunk.get('source', 'Unknown')} — page {chunk.get('page', '')} "
            f"— chunk {chunk.get('chunk_id', '')} — "
            f"{chunk.get('retrieval_source', 'unknown')} — "
            f"{chunk.get('token_count', 0)} tokens"
        )
        blocks.append(
            "<details><summary>"
            + html.escape(summary)
            + "</summary><pre style=\"white-space:pre-wrap\">"
            + html.escape(str(chunk.get("text_preview") or ""))
            + "</pre></details>"
        )
    return "<h4>Final context preview</h4>" + (
        "".join(blocks) if blocks else "<p>No final context.</p>"
    )


def _citations_html(trace: Mapping[str, Any]) -> str:
    cards = []
    for citation in _records(trace, "citations"):
        source = citation.get("source_file") or citation.get("source") or "Unknown"
        details = (
            f"[{citation.get('reference_id', '?')}] {source} — "
            f"Page {citation.get('page_number', '')} — "
            f"Chunk {citation.get('chunk_id', '')} — "
            f"Score {citation.get('score', '')} — "
            f"{citation.get('retrieval_source', 'unknown')}"
        )
        link = citation.get("pdf_link")
        cards.append(
            '<li style="margin:.5em 0">'
            + html.escape(details)
            + (
                f' — <a href="{html.escape(str(link), quote=True)}" target="_blank">Open PDF</a>'
                if link
                else " — PDF link unavailable"
            )
            + "</li>"
        )
    return "<h4>Citations</h4><ol>" + (
        "".join(cards) if cards else "<li>No citations.</li>"
    ) + "</ol>"


def display_question_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Display one complete question assembly line and return its figures."""

    from IPython.display import HTML, Markdown, display

    question = str(trace.get("question") or "")
    display(Markdown(f"### Input Question\n\n{question}"))
    display(Markdown("#### Query Transformations"))
    display(query_variants_frame(trace))
    display(Markdown("#### Dense Retrieval Results"))
    display(retrieval_results_frame(trace, "dense_results"))
    display(Markdown("#### BM25 Retrieval Results"))
    display(retrieval_results_frame(trace, "bm25_results"))

    overlap = trace.get("overlap")
    overlap = overlap if isinstance(overlap, Mapping) else {}
    overlap_frame = pd.DataFrame(
        [
            ("dense only", overlap.get("dense_only_count", 0)),
            ("BM25 only", overlap.get("bm25_only_count", 0)),
            ("both", overlap.get("both_count", 0)),
        ],
        columns=["membership", "chunks"],
    )
    display(Markdown("#### Dense vs BM25 Overlap"))
    display(overlap_frame)
    display(Markdown("#### RRF Fusion Contribution"))
    display(rrf_contribution_frame(trace))

    dedup = trace.get("deduplication") or {}
    neighbors = trace.get("neighbor_expansion") or {}
    display(
        pd.DataFrame(
            [
                {
                    "before_deduplication": dedup.get("before", 0),
                    "after_deduplication": dedup.get("after", 0),
                    "duplicates_removed": dedup.get("duplicates_removed", 0),
                    "duplicate_key": dedup.get("key", ""),
                    "neighbors_added": neighbors.get("neighbors_added", 0),
                    "after_neighbor_expansion": neighbors.get(
                        "total_after_expansion",
                        0,
                    ),
                }
            ]
        )
    )
    expanded_chunks = [
        value
        for value in (neighbors.get("expanded_chunks") or [])
        if isinstance(value, Mapping)
    ]
    display(Markdown("#### Neighbor Expansion"))
    display(
        pd.DataFrame(expanded_chunks).reindex(
            columns=[
                "source",
                "page",
                "chunk_id",
                "is_neighbor",
                "retrieval_source",
                "token_count",
            ]
        )
    )
    funnel = funnel_frame(trace)
    display(Markdown("#### Context Construction Funnel"))
    display(funnel)

    token_usage = trace.get("token_usage")
    token_usage = token_usage if isinstance(token_usage, Mapping) else {}
    display(Markdown("#### Token Budget Usage"))
    display(pd.DataFrame([token_usage]))
    display(HTML(_final_context_html(trace)))

    generation = trace.get("generation")
    generation = generation if isinstance(generation, Mapping) else {}
    display(Markdown("#### Generation"))
    display(pd.DataFrame([generation]))
    display(Markdown("#### Final Answer"))
    display(HTML(f'<div class="answer-content">{render_safe_markdown(str(trace.get("answer") or ""))}</div>'))
    display(HTML(_citations_html(trace)))

    artifacts = trace.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    display(Markdown("#### Artifact Export"))
    display(
        pd.DataFrame(
            [{"artifact": key, "path": value} for key, value in artifacts.items()]
        )
    )

    figures: dict[str, Any] = {}
    if not funnel.empty:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.bar(funnel["stage"], funnel["chunk_count"], color="#2f6f8f")
        ax.set(title="Retrieval stage funnel", ylabel="Chunks")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        display(fig)
        plt.close(fig)
        figures["funnel"] = fig

    final_chunks = _records(trace, "final_context_chunks")
    contributions = pd.Series(
        [chunk.get("retrieval_source", "unknown") for chunk in final_chunks]
    ).value_counts()
    if not contributions.empty:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(contributions.index, contributions.values, color="#3f8f6f")
        ax.set(title="Final-context retriever contribution", ylabel="Chunks")
        fig.tight_layout()
        display(fig)
        plt.close(fig)
        figures["contribution"] = fig

    used = float(token_usage.get("context_tokens_used") or 0)
    remaining = float(token_usage.get("remaining_tokens") or 0)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(["used", "remaining"], [used, remaining], color=["#2f6f8f", "#c9d9de"])
    ax.set(title="Token budget utilization", ylabel="Tokens")
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    figures["tokens"] = fig

    latency = trace.get("latency")
    latency = latency if isinstance(latency, Mapping) else {}
    latency_values = {
        key: float(value)
        for key, value in latency.items()
        if value is not None
    }
    if latency_values:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(
            [key.replace("_seconds", "") for key in latency_values],
            list(latency_values.values()),
            color="#d28b26",
        )
        ax.set(title="Latency breakdown", ylabel="Seconds")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        display(fig)
        plt.close(fig)
        figures["latency"] = fig

    diversity = trace.get("source_diversity")
    diversity = diversity if isinstance(diversity, Mapping) else {}
    display(Markdown("#### Source Diversity"))
    display(pd.DataFrame([diversity]))
    display(Markdown("#### Decision Summary"))
    display(decision_frame(trace))
    return figures
