"""Versioned CSV export for local, inspectable batch question answering."""

from __future__ import annotations

import csv
import json
import re
import time
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


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
) -> Path:
    """Write a new CSV using exclusive creation so no prior export is replaced."""

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
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
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
) -> dict[str, Any]:
    return {
        column: ""
        for column in CSV_COLUMNS
    } | {
        "question": question,
        "top_k": top_k,
        "model_name": model_name,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "failed",
    }


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
    call ``index()`` first). Per-question failures are recorded and do not stop
    the remainder of the batch.
    """

    config = pipeline.config
    project_root = Path(config.project_root).expanduser().resolve()
    resolved_questions = _resolve_questions(
        questions=questions,
        questions_path=questions_path,
        project_root=project_root,
    )
    requested_top_k = int(top_k if top_k is not None else config.top_k)
    if requested_top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    safe_run_name = (
        _sanitize_run_name(run_name) if run_name else _infer_run_name(pipeline)
    )
    model_name = str(getattr(config, "ollama_model_name", "") or "")
    embedding_model = str(getattr(config, "embedding_model_name", "") or "")
    original_top_k = config.top_k
    rows: list[dict[str, Any]] = []

    try:
        config.top_k = requested_top_k
        for question in resolved_questions:
            row = _blank_row(
                question=question,
                top_k=requested_top_k,
                model_name=model_name,
                embedding_model=embedding_model,
            )
            started_at = time.perf_counter()
            try:
                if not question:
                    raise ValueError("Question must not be blank.")
                response = pipeline.answer(question)
                retrieved_value = response.get("retrieved") or []
                retrieved = [
                    result
                    for result in retrieved_value
                    if isinstance(result, Mapping)
                ]
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
                        "retrieved_chunks": len(retrieved),
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
            except Exception as exc:
                row["error"] = str(exc)
            finally:
                row["total_latency_seconds"] = round(
                    time.perf_counter() - started_at,
                    6,
                )
                rows.append(row)
    finally:
        config.top_k = original_top_k

    batch_root = _create_output_structure(project_root)
    return _write_versioned_csv(rows, batch_root, safe_run_name)
