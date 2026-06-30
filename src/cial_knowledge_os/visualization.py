"""Matplotlib-only pipeline diagnostics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from langchain_core.documents import Document

from .benchmarking import benchmark_pipeline_steps


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
