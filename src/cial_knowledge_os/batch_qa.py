"""Versioned CSV export for local, inspectable batch question answering."""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from .llm import GenerationFailedError
from .execution import ExecutionManager
from .token_budget import (
    DEFAULT_TIKTOKEN_ENCODING,
    TokenManager,
    create_token_manager,
)

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "question",
    "answer",
    "sources",
    "source_files",
    "page_numbers",
    "chunk_ids",
    "retrieval_scores",
    "top_k",
    "retrieved_chunks",
    "answer_latency_seconds",
    "retrieval_latency_seconds",
    "total_latency_seconds",
    "model_name",
    "embedding_model",
    "timestamp",
    "status",
    "error",
]

PHASE2_CSV_COLUMNS = [
    "query_variants",
    "chunks_before_deduplication",
    "chunks_after_deduplication",
    "chunks_after_neighbor_expansion",
    "merged_context_sections",
    "final_context_sections",
    "final_context_characters",
    "final_context_tokens_estimate",
    "answer_status",
    "retrieval_trace",
]

PHASE3_CSV_COLUMNS = [
    "retrieval_mode",
    "dense_top_k",
    "bm25_top_k",
    "rrf_k",
    "final_context_tokens",
    "context_budget",
    "context_budget_type",
    "token_encoding",
    "pdf_links",
    "retrieval_sources",
    "dense_result_count",
    "bm25_result_count",
    "fused_result_count",
    "final_context_chunk_count",
    "context_tokens_used",
    "token_utilization",
    "generation_latency_seconds",
    "citation_count",
    "unique_source_count",
]

PHASE4_CSV_COLUMNS = [
    "candidate_chunk_count",
    "reranked_candidate_count",
    "selected_chunk_count",
    "discarded_chunk_count",
    "candidate_tokens",
    "selected_evidence_tokens",
    "token_reduction_percent",
    "average_reranker_score",
    "strong_evidence_count",
    "medium_evidence_count",
    "weak_evidence_count",
    "reranker_latency_seconds",
    "evidence_selection_latency_seconds",
    "usable_candidate_count",
    "threshold_pass_count",
    "fallback_used",
    "evidence_confidence",
    "extractive_fallback_used",
    "fallback_blocked",
    "unsupported_query_detected",
]

PHASE5_CSV_COLUMNS = [
    "phase5_enabled",
    "query_intent",
    "response_format",
    "critic_passed",
    "compliance_passed",
    "risk_passed",
    "verification_rate",
    "consensus_decision",
    "revision_used",
    "final_status",
    "agent_latency_total_ms",
    "model_map",
]

_OUTPUT_SUBDIRECTORIES = (
    "batch_answers",
    "evaluations",
    "benchmarks",
    "logs",
    "exports",
)
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class BatchQAPipeline(Protocol):
    """Minimum pipeline surface required by :func:`export_batch_answers`."""

    config: Any
    metrics: Mapping[str, Any]

    def answer(self, question: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class BatchAnswerCollection:
    """Rows and full responses from one failure-tolerant local batch."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    responses: tuple[Mapping[str, Any] | None, ...]

QuestionCompleteCallback = Callable[
    [int, dict[str, Any], Mapping[str, Any] | None],
    None,
]


def _require_pipeline_ready(pipeline: BatchQAPipeline) -> None:
    """Reject a known uninitialized pipeline before starting a batch."""

    readiness = getattr(pipeline, "is_ready_for_answering", None)
    if callable(readiness):
        readiness = readiness()
    if readiness is not None and not bool(readiness):
        raise RuntimeError(
            "The pipeline is not indexed and cannot answer questions. "
            "Call pipeline.load(), pipeline.chunk(), pipeline.embed(), and "
            "pipeline.index() before export_batch_answers()."
        )


def _json_cell(values: Iterable[Any]) -> str:
    """Serialize list-like CSV values as compact, machine-readable JSON."""

    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _sanitize_run_name(value: str) -> str:
    """Return a cross-platform-safe folder and file stem."""

    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip(" ._")
    if not sanitized:
        return "batch_qa"
    if sanitized.upper() in _WINDOWS_RESERVED_NAMES:
        return f"_{sanitized}"
    return sanitized[:120].rstrip(" ._") or "batch_qa"


def _infer_run_name(pipeline: BatchQAPipeline) -> str:
    """Infer a readable experiment name from the pipeline class."""

    class_name = type(pipeline).__name__
    if class_name.endswith("Pipeline"):
        class_name = class_name[: -len("Pipeline")]
    readable = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", class_name)
    return _sanitize_run_name(readable or "batch_qa")


def _load_questions(path: Path) -> list[str]:
    """Load one-question-per-line text or a CSV ``question`` column."""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "question" not in reader.fieldnames:
                raise ValueError(
                    f"Question CSV '{path}' must contain a 'question' column."
                )
            return [
                str(row.get("question") or "").strip()
                for row in reader
                if str(row.get("question") or "").strip()
            ]
    raise ValueError("questions_path must point to a .txt or .csv file.")


def _resolve_questions(
    *,
    questions: Iterable[str] | None,
    questions_path: str | Path | None,
    project_root: Path,
) -> list[str]:
    if questions is not None and questions_path is not None:
        raise ValueError("Provide questions or questions_path, not both.")
    if questions_path is not None:
        path = Path(questions_path).expanduser()
        if not path.is_absolute():
            path = project_root / path
        resolved = _load_questions(path.resolve())
    elif questions is not None:
        resolved = [str(question).strip() for question in questions]
    else:
        raise ValueError("Provide questions or questions_path.")
    if not resolved:
        raise ValueError("At least one question is required.")
    return resolved


def _create_output_structure(project_root: Path) -> Path:
    """Create the standard local output tree and return batch output root."""

    outputs_root = project_root / "outputs"
    for directory in _OUTPUT_SUBDIRECTORIES:
        (outputs_root / directory).mkdir(parents=True, exist_ok=True)
    return outputs_root / "batch_answers"


def _next_version(output_dir: Path, run_name: str) -> int:
    pattern = re.compile(rf"^{re.escape(run_name)}-v(\d+)\.csv$")
    versions = [
        int(match.group(1))
        for path in output_dir.iterdir()
        if path.is_file() and (match := pattern.fullmatch(path.name))
    ]
    return max(versions, default=0) + 1


def _write_versioned_csv(
    rows: list[dict[str, Any]],
    batch_root: Path,
    run_name: str,
    *,
    columns: list[str] | None = None,
) -> Path:
    """Write a new CSV using exclusive creation so no prior export is replaced."""

    fieldnames = columns or CSV_COLUMNS
    output_dir = batch_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(output_dir, run_name)
    while True:
        output_path = output_dir / f"{run_name}-v{version}.csv"
        try:
            with output_path.open(
                "x",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return output_path.resolve()
        except FileExistsError:
            # Another local process claimed this version after it was calculated.
            version += 1


def _metadata_lists(
    retrieved: Iterable[Mapping[str, Any]],
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any]]:
    sources: list[Any] = []
    source_files: list[Any] = []
    page_numbers: list[Any] = []
    chunk_ids: list[Any] = []
    retrieval_scores: list[Any] = []
    for result in retrieved:
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source = metadata.get("source") or result.get("source")
        source_file = metadata.get("file_name")
        if not source_file and source:
            source_file = Path(str(source)).name
        sources.append(source)
        source_files.append(source_file)
        page_numbers.append(
            result.get("page_number", metadata.get("page_number"))
        )
        chunk_ids.append(result.get("chunk_id", metadata.get("chunk_id")))
        retrieval_scores.append(result.get("score"))
    return sources, source_files, page_numbers, chunk_ids, retrieval_scores


def _blank_row(
    *,
    question: str,
    top_k: int,
    model_name: str,
    embedding_model: str,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    fieldnames = columns or CSV_COLUMNS
    return {
        column: ""
        for column in fieldnames
    } | {
        "question": question,
        "top_k": top_k,
        "model_name": model_name,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "failed",
    }


def _stage_items(
    response: Mapping[str, Any],
    stage_name: str,
) -> list[Mapping[str, Any]]:
    stages = response.get("context_stages")
    if not isinstance(stages, Mapping):
        return []
    value = stages.get(stage_name)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _stage_count(response: Mapping[str, Any], stage_name: str) -> int:
    counts = response.get("stage_counts")
    if isinstance(counts, Mapping):
        value = counts.get(stage_name)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            pass
    return len(_stage_items(response, stage_name))


def _query_variant_values(
    response: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    variants_value = response.get("query_variants")
    if not isinstance(variants_value, Iterable) or isinstance(
        variants_value,
        (str, bytes, Mapping),
    ):
        return [], []

    variants: list[dict[str, str]] = []
    trace_steps: list[str] = []
    labels = {
        "original": "Original Query",
        "rewritten": "Rewritten Query",
        "keyword_expanded": "Keyword Expansion",
        "domain_reformulation": "Domain Reformulation",
    }
    for value in variants_value:
        if not isinstance(value, Mapping):
            continue
        technique = str(value.get("technique") or "")
        query = str(value.get("query") or "")
        variants.append({"technique": technique, "query": query})
        trace_steps.append(f"{labels.get(technique, technique or 'Query')}: {query}")
    return variants, trace_steps


def _phase2_row_values(
    response: Mapping[str, Any],
    token_manager: TokenManager,
) -> dict[str, Any]:
    """Extract the inspectable Phase 2 retrieval and context audit trail."""

    variants, trace_steps = _query_variant_values(response)
    retrieved_count = _stage_count(response, "retrieved")
    deduplicated_count = _stage_count(response, "deduplicated")
    expanded_count = _stage_count(response, "expanded")
    merged_count = _stage_count(response, "merged")
    final_sections = _stage_count(response, "compressed")
    context = str(response.get("context") or "")
    answer_status_value = str(response.get("answer_status") or "")
    if not answer_status_value:
        answer_text = str(
            response.get("raw_answer") or response.get("answer") or ""
        )
        answer_status_value = (
            "insufficient_evidence"
            if "no reliable answer could be generated" in answer_text.casefold()
            else "answered"
        )
    normalized_status = answer_status_value.casefold().replace(" ", "_")
    answer_status = {
        "answered": "Answered",
        "insufficient_evidence": "Insufficient Evidence",
        "unsupported_query": "Unsupported Query",
        "current_data_required": "Unsupported Query",
        "generation_failed": "Generation Failed",
    }.get(
        normalized_status,
        answer_status_value,
    )

    trace_steps.extend(
        [
            f"Retrieved {retrieved_count} chunks",
            f"Deduplicated to {deduplicated_count}",
            f"Neighbor Expanded to {expanded_count}",
            f"Final Context: {final_sections} merged sections",
        ]
    )
    return {
        "query_variants": json.dumps(
            variants,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "chunks_before_deduplication": retrieved_count,
        "chunks_after_deduplication": deduplicated_count,
        "chunks_after_neighbor_expansion": expanded_count,
        "merged_context_sections": merged_count,
        "final_context_sections": final_sections,
        "final_context_characters": len(context),
        # Preserve the legacy column name while writing an exact token count.
        "final_context_tokens_estimate": token_manager.count(context),
        "answer_status": answer_status,
        "retrieval_trace": " → ".join(trace_steps),
    }


def _phase3_row_values(
    response: Mapping[str, Any],
    config: Any,
    token_manager: TokenManager,
) -> dict[str, Any]:
    token_usage = response.get("token_usage")
    token_usage = token_usage if isinstance(token_usage, Mapping) else {}
    citations = response.get("citations")
    citations = (
        citations
        if isinstance(citations, Iterable)
        and not isinstance(citations, (str, bytes, Mapping))
        else []
    )
    final_results = _stage_items(response, "compressed")
    question_trace = response.get("question_trace")
    question_trace = (
        question_trace if isinstance(question_trace, Mapping) else {}
    )
    funnel = question_trace.get("context_funnel")
    funnel = funnel if isinstance(funnel, Mapping) else {}
    trace_counts = funnel.get("counts")
    trace_counts = trace_counts if isinstance(trace_counts, Mapping) else {}
    trace_token_usage = question_trace.get("token_usage")
    trace_token_usage = (
        trace_token_usage
        if isinstance(trace_token_usage, Mapping)
        else token_usage
    )
    latency = question_trace.get("latency")
    latency = latency if isinstance(latency, Mapping) else {}
    source_diversity = question_trace.get("source_diversity")
    source_diversity = (
        source_diversity if isinstance(source_diversity, Mapping) else {}
    )
    retrieval_sources = list(
        dict.fromkeys(
            source
            for result in final_results
            for source in (
                result.get("retrieval_sources")
                if isinstance(result.get("retrieval_sources"), list)
                else []
            )
        )
    )
    return {
        "retrieval_mode": str(response.get("retrieval_mode") or ""),
        "dense_top_k": getattr(config, "dense_top_k", ""),
        "bm25_top_k": getattr(config, "bm25_top_k", ""),
        "rrf_k": getattr(config, "rrf_k", ""),
        "final_context_tokens": token_manager.count(
            str(response.get("context") or "")
        ),
        "context_budget": (
            token_usage.get("budget")
            if token_usage.get("budget_type") == "tokens"
            else token_usage.get("character_budget", "")
        ),
        "context_budget_type": token_usage.get("budget_type", ""),
        "token_encoding": token_manager.encoding_name,
        "pdf_links": _json_cell(
            citation.get("pdf_link")
            for citation in citations
            if isinstance(citation, Mapping) and citation.get("pdf_link")
        ),
        "retrieval_sources": _json_cell(retrieval_sources),
        "dense_result_count": int(trace_counts.get("dense_raw") or 0),
        "bm25_result_count": int(trace_counts.get("bm25_raw") or 0),
        "fused_result_count": int(trace_counts.get("fused") or 0),
        "final_context_chunk_count": int(
            trace_counts.get("compressed") or len(final_results)
        ),
        "context_tokens_used": int(
            trace_token_usage.get("context_tokens_used")
            or token_manager.count(str(response.get("context") or ""))
        ),
        "token_utilization": float(
            trace_token_usage.get("utilization_percent") or 0.0
        ),
        "generation_latency_seconds": float(
            latency.get("generation_seconds") or 0.0
        ),
        "citation_count": len(citations),
        "unique_source_count": int(
            source_diversity.get("unique_source_count") or 0
        ),
    }


def _phase4_row_values(response: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten additive Phase 4 diagnostics into machine-readable CSV values.

    The input is a Phase 4 response and the output contains only scalar summary
    columns; full reranking and selection records remain in ``retrieval.json``.
    Existing Phase 1--3 columns are untouched and retain their order.
    """

    trace = response.get("question_trace")
    trace = trace if isinstance(trace, Mapping) else {}
    token_usage = trace.get("token_usage")
    token_usage = token_usage if isinstance(token_usage, Mapping) else {}
    quality = trace.get("evidence_quality")
    quality = quality if isinstance(quality, Mapping) else {}
    quality_summary = quality.get("summary")
    quality_summary = (
        quality_summary if isinstance(quality_summary, Mapping) else {}
    )
    strengths = quality_summary.get("strength_distribution")
    strengths = strengths if isinstance(strengths, Mapping) else {}
    latency = trace.get("latency")
    latency = latency if isinstance(latency, Mapping) else {}
    return {
        "candidate_chunk_count": int(
            token_usage.get("candidate_chunk_count") or 0
        ),
        "reranked_candidate_count": len(
            trace.get("reranked_candidates") or []
        ),
        "selected_chunk_count": int(
            token_usage.get("selected_chunk_count") or 0
        ),
        "discarded_chunk_count": int(
            token_usage.get("discarded_chunk_count") or 0
        ),
        "candidate_tokens": int(token_usage.get("candidate_tokens") or 0),
        "selected_evidence_tokens": int(
            token_usage.get("selected_evidence_tokens") or 0
        ),
        "token_reduction_percent": float(
            token_usage.get("token_reduction_percent") or 0.0
        ),
        "average_reranker_score": float(
            quality_summary.get("average_reranker_score") or 0.0
        ),
        "strong_evidence_count": int(strengths.get("strong") or 0),
        "medium_evidence_count": int(strengths.get("medium") or 0),
        "weak_evidence_count": int(strengths.get("weak") or 0),
        "reranker_latency_seconds": float(
            latency.get("reranking_seconds") or 0.0
        ),
        "evidence_selection_latency_seconds": float(
            latency.get("evidence_selection_seconds") or 0.0
        ),
        "usable_candidate_count": int(
            token_usage.get("usable_candidate_count") or 0
        ),
        "threshold_pass_count": int(
            token_usage.get("threshold_pass_count") or 0
        ),
        "fallback_used": bool(token_usage.get("fallback_used")),
        "evidence_confidence": str(
            token_usage.get("evidence_confidence") or ""
        ),
        "extractive_fallback_used": bool(
            token_usage.get("extractive_fallback_used")
        ),
        "fallback_blocked": bool(token_usage.get("fallback_blocked")),
        "unsupported_query_detected": bool(
            token_usage.get("unsupported_query_detected")
        ),
    }


def _phase5_row_values(response: Mapping[str, Any]) -> dict[str, Any]:
    intent = response.get("query_intent")
    intent = intent if isinstance(intent, Mapping) else {}
    plan = response.get("response_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    critic = response.get("critic_review")
    critic = critic if isinstance(critic, Mapping) else {}
    compliance = response.get("compliance_review")
    compliance = compliance if isinstance(compliance, Mapping) else {}
    risk = response.get("risk_review")
    risk = risk if isinstance(risk, Mapping) else {}
    verification = response.get("evidence_verification")
    verification = verification if isinstance(verification, Mapping) else {}
    consensus = response.get("consensus_decision")
    consensus = consensus if isinstance(consensus, Mapping) else {}
    return {
        "phase5_enabled": bool(response.get("phase5_enabled")),
        "query_intent": str(intent.get("intent") or ""),
        "response_format": str(plan.get("format") or ""),
        "critic_passed": bool(critic.get("passed")),
        "compliance_passed": bool(compliance.get("passed")),
        "risk_passed": bool(risk.get("passed")),
        "verification_rate": float(
            verification.get("verification_rate") or 0
        ),
        "consensus_decision": str(consensus.get("decision") or ""),
        "revision_used": bool(response.get("revision_used")),
        "final_status": str(response.get("final_status") or ""),
        "agent_latency_total_ms": float(
            response.get("agent_latency_total_ms") or 0
        ),
        "model_map": json.dumps(
            response.get("model_map") or {},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def collect_batch_answers(
    *,
    pipeline: BatchQAPipeline,
    questions: Iterable[str] | None = None,
    questions_path: str | Path | None = None,
    top_k: int | None = None,
    on_question_complete: QuestionCompleteCallback | None = None,
    execution_manager: ExecutionManager | None = None,
) -> BatchAnswerCollection:
    """Collect rows and optionally checkpoint each completed question.

    Existing inputs and outputs remain unchanged. ``on_question_complete`` is
    an additive callback receiving the one-based batch position, mutable row,
    and response immediately after each attempt. Phase 4 uses it for durable
    checkpoints; earlier phases omit it and retain their prior behavior.
    """

    _require_pipeline_ready(pipeline)
    config = pipeline.config
    project_root = Path(config.project_root).expanduser().resolve()
    resolved_questions = _resolve_questions(
        questions=questions,
        questions_path=questions_path,
        project_root=project_root,
    )
    logger.info(
        "batch_question_count_resolved",
        extra={
            "event": "batch_qa",
            "question_count": len(resolved_questions),
        },
    )
    retrieval_depth_attribute = (
        "retrieval_top_k" if hasattr(config, "retrieval_top_k") else "top_k"
    )
    configured_top_k = getattr(config, retrieval_depth_attribute)
    requested_top_k = int(top_k if top_k is not None else configured_top_k)
    if requested_top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    model_name = str(getattr(config, "ollama_model_name", "") or "")
    embedding_model = str(getattr(config, "embedding_model_name", "") or "")
    phase2_export = retrieval_depth_attribute == "retrieval_top_k"
    phase3_export = hasattr(config, "retrieval_mode")
    phase4_export = hasattr(config, "reranker_model_name")
    phase5_export = bool(getattr(pipeline, "enabled", False))
    token_manager_value = getattr(pipeline, "token_manager", None)
    token_manager = (
        token_manager_value
        if isinstance(token_manager_value, TokenManager)
        else create_token_manager(
            encoding_name=str(
                getattr(
                    config,
                    "tokenizer_encoding_name",
                    DEFAULT_TIKTOKEN_ENCODING,
                )
            )
        )
    )
    columns = [
        *CSV_COLUMNS,
        *(PHASE2_CSV_COLUMNS if phase2_export else []),
        *(PHASE3_CSV_COLUMNS if phase3_export else []),
        *(PHASE4_CSV_COLUMNS if phase4_export else []),
        *(PHASE5_CSV_COLUMNS if phase5_export else []),
    ]
    rows: list[dict[str, Any]] = []
    responses: list[Mapping[str, Any] | None] = []
    manager = (
        execution_manager
        or getattr(pipeline, "execution_manager", None)
        or ExecutionManager.disabled()
    )

    try:
        setattr(config, retrieval_depth_attribute, requested_top_k)
        for position, question in enumerate(resolved_questions, start=1):
            manager.start_question(position, len(resolved_questions), question)
            row = _blank_row(
                question=question,
                top_k=requested_top_k,
                model_name=model_name,
                embedding_model=embedding_model,
                columns=columns,
            )
            started_at = time.perf_counter()
            response: Mapping[str, Any] | None = None
            try:
                if not question:
                    raise ValueError("Question must not be blank.")
                response = pipeline.answer(question)
                final_context_results = (
                    _stage_items(response, "compressed")
                    if phase2_export
                    else []
                )
                retrieved_value = (
                    final_context_results
                    or response.get("retrieved")
                    or []
                )
                retrieved = [
                    result
                    for result in retrieved_value
                    if isinstance(result, Mapping)
                ]
                response_retrieved = response.get("retrieved") or []
                retrieved_chunks = len(
                    [
                        result
                        for result in response_retrieved
                        if isinstance(result, Mapping)
                    ]
                )
                sources, files, pages, chunks, scores = _metadata_lists(retrieved)
                metrics = pipeline.metrics
                row.update(
                    {
                        "answer": str(response.get("answer") or ""),
                        "sources": _json_cell(sources),
                        "source_files": _json_cell(files),
                        "page_numbers": _json_cell(pages),
                        "chunk_ids": _json_cell(chunks),
                        "retrieval_scores": _json_cell(scores),
                        "retrieved_chunks": retrieved_chunks,
                        "answer_latency_seconds": round(
                            float(metrics.get("generation_latency", 0.0)),
                            6,
                        ),
                        "retrieval_latency_seconds": round(
                            float(metrics.get("retrieval_latency", 0.0)),
                            6,
                        ),
                        "status": "success",
                    }
                )
                if phase2_export:
                    row.update(_phase2_row_values(response, token_manager))
                if phase3_export:
                    row.update(
                        _phase3_row_values(
                            response,
                            config,
                            token_manager,
                        )
                    )
                if phase4_export:
                    row.update(_phase4_row_values(response))
                if phase5_export:
                    row.update(_phase5_row_values(response))
            except Exception as exc:
                if isinstance(exc, GenerationFailedError):
                    row["answer_status"] = "generation_failed"
                    row["status"] = "failed"
                    row["error"] = (
                        f"{exc.original_error_type}: "
                        f"{exc.original_error_message}; "
                        f"attempts={exc.attempts}"
                    )
                else:
                    row["error"] = str(exc)
                logger.exception(
                    "batch_question_failed",
                    extra={"event": "batch_qa", "question": question},
                )
            finally:
                row["total_latency_seconds"] = round(
                    time.perf_counter() - started_at,
                    6,
                )
                rows.append(row)
                responses.append(response)
                answer_status = str(
                    row.get("answer_status")
                    or ("answered" if row.get("status") == "success" else "failed")
                ).casefold().replace(" ", "_")
                if row.get("status") == "success":
                    manager.complete_question(
                        answer_status=answer_status,
                        total_latency_seconds=row["total_latency_seconds"],
                    )
                else:
                    manager.fail_question(
                        str(row.get("error") or "Question failed."),
                        answer_status=answer_status,
                        total_latency_seconds=row["total_latency_seconds"],
                    )
                if on_question_complete is not None:
                    on_question_complete(position, row, response)
    finally:
        setattr(config, retrieval_depth_attribute, configured_top_k)
    return BatchAnswerCollection(
        columns=tuple(columns),
        rows=tuple(rows),
        responses=tuple(responses),
    )


def export_batch_answers(
    *,
    pipeline: BatchQAPipeline,
    questions: Iterable[str] | None = None,
    questions_path: str | Path | None = None,
    run_name: str | None = None,
    top_k: int | None = None,
) -> Path:
    """Answer questions locally and export a failure-tolerant, versioned CSV.

    The pipeline must already be ready for answering (for ``BasicRAGPipeline``,
    complete ``load()``, ``chunk()``, ``embed()``, and ``index()`` first).
    A known uninitialized pipeline is rejected before the batch starts.
    Per-question failures are recorded and do not stop the remainder of the
    batch.
    """

    config = pipeline.config
    project_root = Path(config.project_root).expanduser().resolve()
    collection = collect_batch_answers(
        pipeline=pipeline,
        questions=questions,
        questions_path=questions_path,
        top_k=top_k,
    )

    safe_run_name = (
        _sanitize_run_name(run_name) if run_name else _infer_run_name(pipeline)
    )

    batch_root = _create_output_structure(project_root)
    return _write_versioned_csv(
        list(collection.rows),
        batch_root,
        safe_run_name,
        columns=list(collection.columns),
    )
