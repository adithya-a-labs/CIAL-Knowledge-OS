"""Latency and outcome aggregation for execution events."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any

from .events import ExecutionEvent


class MetricsCollector:
    _LATENCY_EVENTS = {
        "retrieval_completed": "retrieval",
        "reranking_completed": "reranking",
        "evidence_selection_completed": "evidence_selection",
        "generation_completed": "generation",
        "question_completed": "total_question",
        "question_failed": "total_question",
        "agent_completed": "agent",
        "indexing_completed": "indexing",
    }

    def __init__(self) -> None:
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.warning_count = 0
        self.error_count = 0
        self.answer_statuses: Counter[str] = Counter()
        self.consensus_decisions: Counter[str] = Counter()

    def __call__(self, event: ExecutionEvent) -> None:
        category = self._LATENCY_EVENTS.get(event.event_type)
        if category and event.elapsed_seconds is not None:
            self.latencies[category].append(float(event.elapsed_seconds))
        for key, value in event.metrics.items():
            if key.endswith("_latency_seconds") or key.endswith("_seconds"):
                try:
                    self.latencies[key.removesuffix("_seconds")].append(
                        float(value)
                    )
                except (TypeError, ValueError):
                    pass
        if event.event_type == "warning":
            self.warning_count += 1
        if event.event_type in {"error", "question_failed", "agent_failed"}:
            self.error_count += 1
        if event.event_type in {"question_completed", "question_failed"}:
            status = str(
                event.payload.get("answer_status") or event.status or "unknown"
            )
            self.answer_statuses[status] += 1
        if event.event_type == "consensus_decided":
            decision = str(event.payload.get("decision") or "unknown")
            self.consensus_decisions[decision] += 1

    def summary(self) -> dict[str, Any]:
        timing = {}
        for name, values in self.latencies.items():
            timing[name] = {
                "count": len(values),
                "average": round(sum(values) / len(values), 6),
                "median": round(median(values), 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }
        return {
            "timings": timing,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "answer_status_distribution": dict(self.answer_statuses),
            "consensus_decision_distribution": dict(
                self.consensus_decisions
            ),
        }
