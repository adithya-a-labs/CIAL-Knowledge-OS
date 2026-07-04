"""Serializable Phase 4 execution traces and decision diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metadata import chunk_identity


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _trace_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected: bool | None,
    final_identities: set[tuple[Any, ...]],
    compact: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in candidates:
        record = dict(value)
        if selected is not None:
            record["selected"] = selected
        record["final_context_inclusion"] = chunk_identity(record) in final_identities
        text = str(record.get("text") or "")
        record["text_preview"] = text[:400]
        if compact:
            record.pop("text", None)
            metadata = record.get("metadata")
            if isinstance(metadata, Mapping):
                record["metadata"] = {
                    key: metadata.get(key)
                    for key in (
                        "source",
                        "file_name",
                        "page_number",
                        "chunk_id",
                        "chunk_index",
                    )
                    if key in metadata
                }
        records.append(record)
    return records


def phase4_diagnostics(
    *,
    token_reduction_percent: float,
    average_reranker_score: float,
    medium_score_threshold: float,
    discarded: Sequence[Mapping[str, Any]],
    latency: Mapping[str, Any],
    unique_source_count: int,
    selected_chunk_count: int,
    candidate_chunk_count: int = 0,
    selected_evidence_tokens: int = 0,
    answer_status: str = "",
) -> list[dict[str, str]]:
    """Build deterministic operational recommendations from Phase 4 metrics.

    Inputs are measured token, score, discard, latency, and diversity values.
    Outputs are short signal/recommendation dictionaries for notebook and HTML
    control-room displays. These rules are diagnostics, not benchmark claims,
    and do not alter Phase 3 or answer behavior.
    """

    diagnostics: list[dict[str, str]] = []
    normalized_status = answer_status.casefold().replace(" ", "_")
    if candidate_chunk_count > 0 and selected_chunk_count == 0:
        diagnostics.append(
            {
                "signal": "evidence_starvation",
                "recommendation": (
                    "Candidates existed but selection retained zero chunks. "
                    "Inspect invalid text, token limits, and fallback settings."
                ),
            }
        )
    if token_reduction_percent > 90:
        diagnostics.append(
            {
                "signal": "excessive_token_reduction",
                "recommendation": (
                    "Token reduction exceeds 90%. Verify that evidence "
                    "selection is not harming answerability."
                ),
            }
        )
    elif token_reduction_percent >= 40:
        diagnostics.append(
            {
                "signal": "token_reduction",
                "recommendation": "Phase 4 reduced context size significantly.",
            }
        )
    else:
        diagnostics.append(
            {
                "signal": "token_reduction",
                "recommendation": (
                    "Context reduction is limited; inspect selection thresholds "
                    "and candidate redundancy."
                ),
            }
        )
    if (
        normalized_status == "answered"
        and 0 < selected_evidence_tokens < 500
    ):
        diagnostics.append(
            {
                "signal": "low_selected_evidence_tokens",
                "recommendation": (
                    "An answered question used fewer than 500 selected evidence "
                    "tokens. Inspect whether context is too narrow."
                ),
            }
        )
    if candidate_chunk_count > 0 and average_reranker_score == 0:
        diagnostics.append(
            {
                "signal": "zero_average_reranker_score",
                "recommendation": (
                    "Candidates existed but the selected-evidence average "
                    "reranker score is zero. Check score propagation and "
                    "fallback selection."
                ),
            }
        )
    if selected_chunk_count and average_reranker_score < medium_score_threshold:
        diagnostics.append(
            {
                "signal": "evidence_strength",
                "recommendation": (
                    "Evidence may be weak. Inspect retrieved candidates or "
                    "broaden retrieval."
                ),
            }
        )
    reasons = Counter(
        str(item.get("discard_reason") or "unspecified") for item in discarded
    )
    if reasons.get("redundancy", 0) >= 2:
        diagnostics.append(
            {
                "signal": "redundancy",
                "recommendation": (
                    "Phase 3 retrieved overlapping evidence. Evidence selection "
                    "improved efficiency."
                ),
            }
        )
    generation = float(latency.get("generation_seconds") or 0.0)
    reranking = float(latency.get("reranking_seconds") or 0.0)
    retrieval = float(latency.get("retrieval_seconds") or 0.0)
    if generation > max(0.001, retrieval + reranking):
        diagnostics.append(
            {
                "signal": "generation_latency",
                "recommendation": (
                    "Generation remains the bottleneck. Consider lowering "
                    "answer length or context budget."
                ),
            }
        )
    if selected_chunk_count and unique_source_count <= 1:
        diagnostics.append(
            {
                "signal": "source_diversity",
                "recommendation": (
                    "Evidence is concentrated. Consider source diversity settings."
                ),
            }
        )
    if reranking > max(0.001, retrieval):
        diagnostics.append(
            {
                "signal": "reranker_latency",
                "recommendation": (
                    "Reranking latency dominates retrieval. The reranker model "
                    "may be too large or its batch size too small."
                ),
            }
        )
    return diagnostics


@dataclass(frozen=True, slots=True)
class Phase4Trace:
    """Wrap one JSON-safe Phase 4 question trace.

    The payload records retrieval through answer and artifact paths. ``to_dict``
    and ``to_json`` produce portable outputs; ``from_dict`` and ``from_json``
    restore them for tests, notebooks, and future benchmark tooling. The wrapper
    accepts additive fields, so later Phase 4 diagnostics do not invalidate
    previously persisted Phase 3-compatible data.
    """

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe trace dictionary."""

        return dict(_json_safe(self.payload))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the trace as UTF-8-safe JSON text."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Phase4Trace":
        """Create a trace from a mapping, validating the required question."""

        payload = dict(_json_safe(value))
        if not str(payload.get("question") or "").strip():
            raise ValueError("Phase4Trace requires a non-blank question.")
        return cls(payload)

    @classmethod
    def from_json(cls, value: str) -> "Phase4Trace":
        """Deserialize one trace previously produced by :meth:`to_json`."""

        parsed = json.loads(value)
        if not isinstance(parsed, Mapping):
            raise ValueError("Phase4Trace JSON must contain an object.")
        return cls.from_dict(parsed)


def build_phase4_trace(
    *,
    question: str,
    phase3_trace: Mapping[str, Any],
    candidate_pool: Sequence[Mapping[str, Any]],
    reranked_candidates: Sequence[Mapping[str, Any]],
    selected_chunks: Sequence[Mapping[str, Any]],
    discarded_chunks: Sequence[Mapping[str, Any]],
    final_context_chunks: Sequence[Mapping[str, Any]],
    evidence_quality: Mapping[str, Any],
    token_usage: Mapping[str, Any],
    latency: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    answer: str,
    answer_status: str,
    trace_mode: str,
    medium_score_threshold: float,
) -> Phase4Trace:
    """Build the complete Retrieved -> Reranked -> Selected -> Answer trace.

    Inputs combine the preserved Phase 3 trace with Phase 4 candidate,
    selection, quality, token, latency, citation, and answer data. The output is
    a serializable :class:`Phase4Trace`. ``compact`` mode removes full candidate
    text while retaining previews and decisions; ``full`` mode keeps it for
    engineering inspection. Phase 3 trace data remains nested and key retrieval
    lists are also promoted for reporting compatibility.
    """

    if trace_mode not in {"compact", "full"}:
        raise ValueError("trace_mode must be 'compact' or 'full'.")
    final_identities = {chunk_identity(item) for item in final_context_chunks}
    quality_summary = evidence_quality.get("summary")
    quality_summary = (
        quality_summary if isinstance(quality_summary, Mapping) else {}
    )
    reduction = float(token_usage.get("token_reduction_percent") or 0.0)
    diagnostics = phase4_diagnostics(
        token_reduction_percent=reduction,
        average_reranker_score=float(
            quality_summary.get("average_reranker_score") or 0.0
        ),
        medium_score_threshold=medium_score_threshold,
        discarded=discarded_chunks,
        latency=latency,
        unique_source_count=int(
            quality_summary.get("unique_source_count") or 0
        ),
        selected_chunk_count=len(selected_chunks),
        candidate_chunk_count=len(candidate_pool),
        selected_evidence_tokens=int(
            token_usage.get("selected_evidence_tokens") or 0
        ),
        answer_status=answer_status,
    )
    payload = {
        "schema_version": "phase4-trace-v1",
        "question": question,
        "pipeline_flow": [
            "hybrid_retrieval",
            "candidate_pool",
            "reranker",
            "evidence_selector",
            "token_aware_context",
            "grounded_answer",
        ],
        "trace_mode": trace_mode,
        "query_variants": phase3_trace.get("query_variants") or [],
        "dense_results": phase3_trace.get("dense_results") or [],
        "bm25_results": phase3_trace.get("bm25_results") or [],
        "rrf_fused_candidates": phase3_trace.get("fused_results") or [],
        "candidate_pool": _trace_candidates(
            candidate_pool,
            selected=None,
            final_identities=final_identities,
            compact=trace_mode == "compact",
        ),
        "reranked_candidates": _trace_candidates(
            reranked_candidates,
            selected=None,
            final_identities=final_identities,
            compact=trace_mode == "compact",
        ),
        "selected_chunks": _trace_candidates(
            selected_chunks,
            selected=True,
            final_identities=final_identities,
            compact=trace_mode == "compact",
        ),
        "discarded_chunks": _trace_candidates(
            discarded_chunks,
            selected=False,
            final_identities=final_identities,
            compact=trace_mode == "compact",
        ),
        "final_context_chunks": _trace_candidates(
            final_context_chunks,
            selected=True,
            final_identities=final_identities,
            compact=trace_mode == "compact",
        ),
        "evidence_quality": evidence_quality,
        "token_usage": token_usage,
        "latency": latency,
        "citations": list(citations),
        "answer": answer,
        "answer_status": answer_status,
        "decision_summary": diagnostics,
        "phase3_trace": phase3_trace,
        "artifacts": {},
    }
    return Phase4Trace.from_dict(payload)
