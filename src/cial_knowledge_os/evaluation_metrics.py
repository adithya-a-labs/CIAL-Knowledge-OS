"""Deterministic answer and aggregate metrics for offline RAG evaluation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import Any

from .benchmark_loader import BenchmarkQuestion


SAFE_FAILURE_MARKERS = (
    "insufficient evidence",
    "do not contain sufficient",
    "could not find enough evidence",
    "no reliable answer",
    "cannot answer from",
    "not available in the",
)


def _normalized(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_answer(
    question: BenchmarkQuestion,
    generated_answer: str,
    *,
    answer_status: str = "",
    citations: Iterable[Any] = (),
    keyword_pass_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one answer without model calls or external services."""

    answer = _normalized(generated_answer)
    keywords = question.expected_keywords
    matches = sum(_normalized(keyword) in answer for keyword in keywords)
    keyword_score = matches / len(keywords) if keywords else (1.0 if answer else 0.0)
    forbidden_matches = sum(
        _normalized(keyword) in answer for keyword in question.forbidden_keywords
    )
    status = _normalized(answer_status).replace(" ", "_")
    safe_failure = status in {"insufficient_evidence", "safe_failure"} or any(
        marker in answer for marker in SAFE_FAILURE_MARKERS
    )
    citation_values = list(citations)
    citation_count = len(citation_values)
    if not citation_count:
        citation_count = len(set(re.findall(r"\[(\d+)\]", generated_answer)))

    if question.should_answer:
        passed = bool(answer) and not safe_failure and keyword_score >= keyword_pass_threshold
        hallucinated = forbidden_matches > 0
    else:
        passed = safe_failure and forbidden_matches == 0
        hallucinated = bool(answer) and not safe_failure

    citation_quality = 0.0
    if citation_count:
        citation_quality = 1.0
        mappings = [item for item in citation_values if isinstance(item, Mapping)]
        if mappings:
            complete = sum(
                bool(
                    item.get("source")
                    or item.get("source_file")
                    or item.get("document")
                )
                for item in mappings
            )
            citation_quality = complete / len(mappings)
    return {
        "passed_answer_test": passed,
        "keyword_score": round(keyword_score, 6),
        "matched_keywords": matches,
        "forbidden_keyword_matches": forbidden_matches,
        "safe_failure": safe_failure,
        "hallucinated": hallucinated,
        "citation_count": citation_count,
        "citation_quality": round(citation_quality, 6),
    }


def aggregate_experiment(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one experiment while accepting extensible metric columns."""

    records = list(rows)
    if not records:
        return {}

    def average(name: str) -> float:
        return round(fmean(_number(row.get(name)) for row in records), 6)

    passed = sum(bool(row.get("passed_answer_test")) for row in records)
    hallucinations = sum(bool(row.get("hallucinated")) for row in records)
    should_answer = [row for row in records if bool(row.get("should_answer", True))]
    unsupported = [row for row in records if not bool(row.get("should_answer", True))]
    category: dict[str, list[bool]] = defaultdict(list)
    difficulty: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        category[str(row.get("category") or "uncategorized")].append(
            bool(row.get("passed_answer_test"))
        )
        difficulty[str(row.get("difficulty") or "unknown")].append(
            bool(row.get("passed_answer_test"))
        )

    first = records[0]
    config_keys = [
        key for key in first
        if key.startswith("config_")
        or key in {
            "retrieval_top_k", "max_context_chars", "neighbor_window",
            "multi_query_enabled", "neighbor_expansion_enabled",
            "max_context_tokens", "token_encoding",
        }
    ]
    return {
        "experiment_id": first.get("experiment_id", ""),
        **{key: first.get(key, "") for key in config_keys},
        "question_count": len(records),
        "answer_accuracy": round(passed / len(records), 6),
        "supported_answer_accuracy": round(
            sum(bool(row.get("passed_answer_test")) for row in should_answer)
            / len(should_answer),
            6,
        ) if should_answer else 0.0,
        "unsupported_question_accuracy": round(
            sum(bool(row.get("passed_answer_test")) for row in unsupported)
            / len(unsupported),
            6,
        ) if unsupported else 0.0,
        "hallucination_rate": round(hallucinations / len(records), 6),
        "keyword_coverage": average("keyword_score"),
        "citation_quality": average("citation_quality"),
        "average_citation_count": average("citation_count"),
        "average_latency": average("total_latency"),
        "average_retrieval_latency": average("retrieval_latency"),
        "average_context_construction_latency": average(
            "context_construction_latency"
        ),
        "average_generation_latency": average("generation_latency"),
        "average_context_size": average("final_context_characters"),
        "average_context_tokens": average("context_tokens"),
        "average_context_sections": average("final_context_sections"),
        "average_retrieved_chunks": average("chunks_before_deduplication"),
        "average_deduplicated_chunks": average("chunks_after_deduplication"),
        "average_expanded_chunks": average("chunks_after_neighbor_expansion"),
        "average_merged_sections": average("merged_context_sections"),
        "pass_rate_by_category": {
            key: round(sum(values) / len(values), 6)
            for key, values in sorted(category.items())
        },
        "pass_rate_by_difficulty": {
            key: round(sum(values) / len(values), 6)
            for key, values in sorted(difficulty.items())
        },
    }


def rank_experiments(
    summaries: Iterable[Mapping[str, Any]],
    *,
    quality_weight: float = 0.55,
    citation_weight: float = 0.15,
    safety_weight: float = 0.2,
    latency_weight: float = 0.1,
) -> list[dict[str, Any]]:
    """Return Pareto-aware overall rankings with normalized latency."""

    records = [dict(summary) for summary in summaries]
    if not records:
        return []
    latencies = [_number(row.get("average_latency")) for row in records]
    low, high = min(latencies), max(latencies)
    span = high - low
    for row, latency in zip(records, latencies, strict=True):
        latency_score = 1.0 if span == 0 else 1.0 - ((latency - low) / span)
        row["overall_score"] = round(
            quality_weight * _number(row.get("answer_accuracy"))
            + citation_weight * _number(row.get("citation_quality"))
            + safety_weight * (1.0 - _number(row.get("hallucination_rate")))
            + latency_weight * latency_score,
            6,
        )
    ranked = sorted(
        records,
        key=lambda row: (
            -_number(row.get("overall_score")),
            _number(row.get("average_latency")),
            str(row.get("experiment_id")),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked
