"""Engineering recommendations derived from deterministic experiment summaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .evaluation_metrics import rank_experiments


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _best(
    rows: list[Mapping[str, Any]],
    metric: str,
    *,
    minimize: bool = False,
) -> Mapping[str, Any]:
    return min(rows, key=lambda row: _number(row.get(metric))) if minimize else max(
        rows, key=lambda row: _number(row.get(metric))
    )


def build_recommendations(
    summaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build machine-readable recommendations without phase-specific assumptions."""

    ranked = rank_experiments(summaries)
    if not ranked:
        return {
            "recommended_default_configuration": None,
            "recommendations": [],
            "tradeoffs": [],
            "bottlenecks": [],
            "phase_3_improvements": [],
        }
    fastest = _best(ranked, "average_latency", minimize=True)
    quality = _best(ranked, "answer_accuracy")
    citations = _best(ranked, "citation_quality")
    safest = _best(ranked, "hallucination_rate", minimize=True)
    best = ranked[0]
    canonical = {
        "retrieval_top_k", "max_context_chars", "max_context_tokens",
        "token_encoding", "neighbor_window",
        "multi_query_enabled", "neighbor_expansion_enabled",
    }
    config_names = [key for key in best if key in canonical]
    config_names.extend(
        key
        for key in best
        if key.startswith("config_")
        and key.removeprefix("config_") not in canonical
    )
    recommendations = [
        {
            "parameter": key.removeprefix("config_"),
            "value": best.get(key),
            "basis": "highest balanced overall score",
        }
        for key in config_names
    ]
    latency_delta = (
        (_number(best.get("average_latency")) / _number(fastest.get("average_latency")) - 1)
        if _number(fastest.get("average_latency"))
        else 0.0
    )
    accuracy_delta = _number(best.get("answer_accuracy")) - _number(
        fastest.get("answer_accuracy")
    )
    latency_parts = {
        "retrieval": _number(best.get("average_retrieval_latency")),
        "context construction": _number(
            best.get("average_context_construction_latency")
        ),
        "generation": _number(best.get("average_generation_latency")),
    }
    bottleneck = max(latency_parts, key=latency_parts.get)
    return {
        "recommended_default_configuration": best.get("experiment_id"),
        "best_configuration": best.get("experiment_id"),
        "fastest_configuration": fastest.get("experiment_id"),
        "highest_answer_accuracy": quality.get("experiment_id"),
        "lowest_hallucination_rate": safest.get("experiment_id"),
        "highest_citation_quality": citations.get("experiment_id"),
        "recommendations": recommendations,
        "tradeoffs": [
            (
                f"The recommended configuration changes answer accuracy by "
                f"{accuracy_delta:+.1%} and latency by {latency_delta:+.1%} "
                f"relative to the fastest configuration."
            )
        ],
        "bottlenecks": [
            f"The largest measured latency component is {bottleneck} "
            f"({latency_parts[bottleneck]:.3f}s average)."
        ],
        "phase_3_improvements": [
            "Add hybrid lexical/vector retrieval and compare recall at the same Top-K.",
            "Add reranker relevance and citation-entailment metrics as optional columns.",
            "Profile retrieval, context construction, and generation independently.",
        ],
    }


def render_recommendation_markdown(recommendations: Mapping[str, Any]) -> str:
    """Render the recommendation payload as an inspectable Markdown report."""

    lines = [
        "# Automated Experiment Recommendations",
        "",
        f"- Recommended default: `{recommendations.get('recommended_default_configuration') or 'n/a'}`",
        f"- Fastest configuration: `{recommendations.get('fastest_configuration') or 'n/a'}`",
        f"- Highest answer accuracy: `{recommendations.get('highest_answer_accuracy') or 'n/a'}`",
        f"- Lowest hallucination rate: `{recommendations.get('lowest_hallucination_rate') or 'n/a'}`",
        f"- Highest citation quality: `{recommendations.get('highest_citation_quality') or 'n/a'}`",
        "",
        "## Parameter Recommendations",
        "",
    ]
    items = recommendations.get("recommendations") or []
    lines.extend(
        f"- `{item['parameter']}`: `{item['value']}` ({item['basis']})"
        for item in items
    )
    for heading, key in (
        ("Trade-offs", "tradeoffs"),
        ("Observed Bottlenecks", "bottlenecks"),
        ("Suggested Improvements for Phase 3", "phase_3_improvements"),
    ):
        lines.extend(["", f"## {heading}", ""])
        lines.extend(f"- {item}" for item in recommendations.get(key) or ["None."])
    lines.extend(
        [
            "",
            "## Machine-readable Payload",
            "",
            "```json",
            json.dumps(recommendations, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_recommendation_report(
    summaries: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    recommendations = build_recommendations(summaries)
    destination.write_text(
        render_recommendation_markdown(recommendations),
        encoding="utf-8",
    )
    return destination.resolve(), recommendations
