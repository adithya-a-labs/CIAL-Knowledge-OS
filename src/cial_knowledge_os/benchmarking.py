"""Small timing utilities for inspectable notebook benchmarks."""

from __future__ import annotations

import time
from contextlib import ContextDecorator
from typing import Any


class Timer(ContextDecorator):
    """Measure a block and store elapsed seconds in a metrics dictionary."""

    def __init__(self, metrics: dict[str, float], metric_name: str) -> None:
        self.metrics = metrics
        self.metric_name = metric_name
        self.started_at = 0.0

    def __enter__(self) -> "Timer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.metrics[self.metric_name] = time.perf_counter() - self.started_at


_METRIC_LABELS = {
    "document_loading_time": "Document loading",
    "pdf_loading_time": "PDF loading",
    "chunking_time": "Chunking",
    "embedding_time": "Embedding",
    "indexing_time": "Indexing",
    "retrieval_latency": "Retrieval",
    "generation_latency": "Generation",
    "total_pipeline_latency": "Total pipeline",
}


def benchmark_pipeline_steps(metrics: dict[str, Any]) -> dict[str, float]:
    """Return finite timing metrics in the standard pipeline order."""

    return {
        name: float(metrics[name])
        for name in _METRIC_LABELS
        if metrics.get(name) is not None
    }


def print_benchmark_table(metrics: dict[str, Any]) -> None:
    """Print a dependency-free benchmark table."""

    benchmark = benchmark_pipeline_steps(metrics)
    if not benchmark:
        print("No benchmark metrics recorded.")
        return
    print(f"{'Pipeline step':<24} {'Seconds':>12}")
    print("-" * 37)
    for name, seconds in benchmark.items():
        print(f"{_METRIC_LABELS[name]:<24} {seconds:>12.4f}")
