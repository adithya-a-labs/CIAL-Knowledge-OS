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
from typing import Any, Protocol

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
    "pdf_links",
    "retrieval_sources",
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


def _estimate_context_tokens(context: str) -> int:
    """Return a deterministic tokenizer-independent approximation."""

    return len(re.findall(r"\w+|[^\w\s]", context, flags=re.UNICODE))


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


def _phase2_row_values(response: Mapping[str, Any]) -> dict[str, Any]:
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
    answer_status = (
        "Insufficient Evidence"
        if answer_status_value.casefold().replace(" ", "_")
        == "insufficient_evidence"
        else "Answered"
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
        "final_context_tokens_estimate": _estimate_context_tokens(context),
        "answer_status": answer_status,
        "retrieval_trace": " → ".join(trace_steps),
    }


def _phase3_row_values(
    response: Mapping[str, Any],
    config: Any,
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
        "final_context_tokens": token_usage.get("used", ""),
        "context_budget": token_usage.get("budget", ""),
        "context_budget_type": token_usage.get("budget_type", ""),
        "pdf_links": _json_cell(
            citation.get("pdf_link")
            for citation in citations
            if isinstance(citation, Mapping) and citation.get("pdf_link")
        ),
        "retrieval_sources": _json_cell(retrieval_sources),
    }


def collect_batch_answers(
    *,
    pipeline: BatchQAPipeline,
    questions: Iterable[str] | None = None,
    questions_path: str | Path | None = None,
    top_k: int | None = None,
) -> BatchAnswerCollection:
    """Collect backward-compatible rows while retaining full Phase 3 traces."""

    _require_pipeline_ready(pipeline)
    config = pipeline.config
    project_root = Path(config.project_root).expanduser().resolve()
    resolved_questions = _resolve_questions(
        questions=questions,
        questions_path=questions_path,
        project_root=project_root,
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
    columns = [
        *CSV_COLUMNS,
        *(PHASE2_CSV_COLUMNS if phase2_export else []),
        *(PHASE3_CSV_COLUMNS if phase3_export else []),
    ]
    rows: list[dict[str, Any]] = []
    responses: list[Mapping[str, Any] | None] = []

    try:
        setattr(config, retrieval_depth_attribute, requested_top_k)
        for question in resolved_questions:
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
                    row.update(_phase2_row_values(response))
                if phase3_export:
                    row.update(_phase3_row_values(response, config))
            except Exception as exc:
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
