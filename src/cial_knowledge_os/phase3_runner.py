"""End-to-end Phase 3 execution and reproducible run-bundle generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from .batch_qa import (
    BatchAnswerCollection,
    BatchQAPipeline,
    QuestionCompleteCallback,
    collect_batch_answers,
)
from .benchmark_loader import Benchmark
from .config import Phase3Config
from .evaluation_metrics import evaluate_answer
from .logging_config import close_logging, configure_logging
from .phase3_reporting import (
    write_results_csv,
    write_results_xlsx,
    write_latency_svg,
    write_standalone_html,
)
from .run_manager import RunManager, RunPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Phase3RunResult:
    """Paths and summaries produced by one complete Phase 3 run."""

    paths: RunPaths
    summary: dict[str, Any]
    metrics: dict[str, Any]


class Phase3Runner:
    """Coordinate answering and artifacts while keeping notebooks declarative."""

    def __init__(
        self,
        *,
        pipeline: BatchQAPipeline,
        config: Phase3Config | None = None,
        run_manager: RunManager | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or pipeline.config
        if not isinstance(self.config, Phase3Config):
            raise TypeError("Phase3Runner requires a Phase3Config.")
        if pipeline.config is not self.config:
            raise ValueError(
                "Phase3Runner config must be the same instance used by the "
                "pipeline so exported configuration matches executed behavior."
            )
        self.run_manager = run_manager or RunManager.from_config(self.config)

    @staticmethod
    def _context_markdown(
        *,
        question: str,
        response: dict[str, Any] | Any,
        error: str,
    ) -> str:
        response = response if isinstance(response, dict) else {}
        stages = response.get("context_stages")
        stages = stages if isinstance(stages, dict) else {}
        retrieved = stages.get("retrieved") or response.get("retrieved") or []
        retrieved_json = json.dumps(
            retrieved,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        return (
            "# Question\n\n"
            f"{question}\n\n"
            "# Retrieved Chunks\n\n"
            f"```json\n{retrieved_json}\n```\n\n"
            "# Merged Context\n\n"
            f"```\n{response.get('context') or ''}\n```\n\n"
            "# Prompt\n\n"
            f"```\n{response.get('prompt') or ''}\n```\n\n"
            "# Generated Answer\n\n"
            f"{response.get('answer') or ''}\n\n"
            "# Error\n\n"
            f"{error or 'None'}\n"
        )

    @staticmethod
    def _summaries(
        rows: list[dict[str, Any]],
        responses: list[dict[str, Any] | None],
        *,
        retrieval_mode: str,
        benchmark: Benchmark | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        successes = [row for row in rows if row.get("status") == "success"]
        latencies = [
            float(row.get("total_latency_seconds") or 0.0) for row in rows
        ]
        retrieval_latencies = [
            float(row.get("retrieval_latency_seconds") or 0.0) for row in rows
        ]
        answered = sum(row.get("answer_status") == "Answered" for row in rows)
        safe_failures = sum(
            row.get("answer_status") == "Insufficient Evidence" for row in rows
        )
        unsupported = sum(
            row.get("answer_status") == "Unsupported Query" for row in rows
        )
        generation_failed = sum(
            str(row.get("answer_status") or "")
            .casefold()
            .replace(" ", "_")
            == "generation_failed"
            for row in rows
        )
        summary: dict[str, Any] = {
            "question_count": len(rows),
            "successful_questions": len(successes),
            "failed_questions": len(rows) - len(successes),
            "answered_questions": answered,
            "insufficient_evidence_questions": safe_failures,
            "unsupported_query_questions": unsupported,
            "generation_failed_questions": generation_failed,
            "average_latency_seconds": round(fmean(latencies), 6)
            if latencies
            else 0.0,
            "retrieval_mode": retrieval_mode,
        }
        metrics: dict[str, Any] = {
            **summary,
            "average_retrieval_latency_seconds": round(
                fmean(retrieval_latencies),
                6,
            )
            if retrieval_latencies
            else 0.0,
            "average_retrieved_chunks": round(
                fmean(float(row.get("retrieved_chunks") or 0) for row in rows),
                6,
            )
            if rows
            else 0.0,
            "average_context_tokens": round(
                fmean(float(row.get("final_context_tokens") or 0) for row in rows),
                6,
            )
            if rows
            else 0.0,
        }
        if benchmark is not None:
            by_question = {item.question: item for item in benchmark.questions}
            evaluations = []
            for row, response in zip(rows, responses, strict=True):
                question = by_question.get(str(row.get("question") or ""))
                if question is None:
                    continue
                response = response or {}
                evaluations.append(
                    evaluate_answer(
                        question,
                        str(row.get("answer") or ""),
                        answer_status=str(response.get("answer_status") or ""),
                        citations=response.get("citations") or (),
                    )
                )
            if evaluations:
                metrics["answer_accuracy"] = round(
                    sum(bool(value["passed_answer_test"]) for value in evaluations)
                    / len(evaluations),
                    6,
                )
                metrics["hallucination_rate"] = round(
                    sum(bool(value["hallucinated"]) for value in evaluations)
                    / len(evaluations),
                    6,
                )
                metrics["citation_quality"] = round(
                    fmean(float(value["citation_quality"]) for value in evaluations),
                    6,
                )
                summary.update(
                    {
                        "answer_accuracy": metrics["answer_accuracy"],
                        "hallucination_rate": metrics["hallucination_rate"],
                        "citation_quality": metrics["citation_quality"],
                    }
                )
        return summary, metrics

    def run(
        self,
        *,
        questions: list[str] | tuple[str, ...] | None = None,
        questions_path: str | Path | None = None,
        benchmark: Benchmark | None = None,
        top_k: int | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        initial_rows: Sequence[Mapping[str, Any]] = (),
        initial_responses: Sequence[Mapping[str, Any] | None] = (),
        on_question_complete: QuestionCompleteCallback | None = None,
    ) -> Phase3RunResult:
        """Execute questions and generate the complete configured artifact set.

        ``initial_rows``/``initial_responses`` and ``on_question_complete`` are
        additive resume hooks. Normal Phase 1--3 callers omit them and preserve
        the previous lifecycle. Resume callers provide aligned checkpoint
        records; new rows are merged and ordered by their private checkpoint
        index before the standard exports are regenerated.
        """

        if benchmark is not None and questions is None and questions_path is None:
            questions = [item.question for item in benchmark.questions]
        if len(initial_rows) != len(initial_responses):
            raise ValueError(
                "initial_rows and initial_responses must have equal lengths."
            )
        metadata = {
            str(key).strip(): value
            for key, value in (run_metadata or {}).items()
            if str(key).strip()
        }
        paths = self.run_manager.create()
        log_handlers = configure_logging(
            level=self.config.log_level,
            structured=self.config.structured_logging,
            log_path=paths.logs,
        )
        benchmark_details = (
            {
                **benchmark.metadata,
                "source_path": benchmark.source_path,
                "question_count": len(benchmark.questions),
            }
            if benchmark is not None
            else {}
        )
        self.run_manager.write_effective_config(
            self.config,
            benchmark=benchmark_details,
            run_overrides={
                "retrieval_top_k": (
                    top_k if top_k is not None else self.config.retrieval_top_k
                ),
                **metadata,
            },
        )
        logger.info(
            "phase3_run_started",
            extra={
                "event": "run",
                "retrieval_mode": self.config.retrieval_mode,
                "run_path": str(paths.root),
                "input_question_count": (
                    len(questions) if questions is not None else None
                ),
                **metadata,
            },
        )
        if (
            questions is not None
            and len(questions) == 0
            and questions_path is None
            and initial_rows
        ):
            collection = BatchAnswerCollection(
                columns=tuple(
                    key
                    for key in initial_rows[0]
                    if not str(key).startswith("__checkpoint_")
                ),
                rows=(),
                responses=(),
            )
        else:
            collection = collect_batch_answers(
                pipeline=self.pipeline,
                questions=questions,
                questions_path=questions_path,
                top_k=top_k,
                on_question_complete=on_question_complete,
            )
        row_response_pairs = [
            *[
                (dict(row), dict(response) if response is not None else None)
                for row, response in zip(
                    initial_rows,
                    initial_responses,
                    strict=True,
                )
            ],
            *[
                (dict(row), dict(response) if response is not None else None)
                for row, response in zip(
                    collection.rows,
                    collection.responses,
                    strict=True,
                )
            ],
        ]
        if any("__checkpoint_index" in row for row, _ in row_response_pairs):
            row_response_pairs.sort(
                key=lambda pair: int(
                    pair[0].get("__checkpoint_index") or 2**31
                )
            )
        rows = [
            {
                key: value
                for key, value in row.items()
                if not str(key).startswith("__checkpoint_")
            }
            for row, _ in row_response_pairs
        ]
        for row in rows:
            row.update(metadata)
        result_columns = (
            *collection.columns,
            *(
                key
                for key in metadata
                if key not in collection.columns
            ),
        )
        responses = [response for _, response in row_response_pairs]
        artifact_started_at = perf_counter()
        execution_manager = getattr(
            self.pipeline, "execution_manager", None
        )
        if execution_manager is not None:
            execution_manager.emit(
                "export_started",
                stage="export",
                status="running",
                payload={"artifact_root": str(paths.root)},
                source="phase3_runner",
            )
        write_results_csv(paths.results_csv, rows, result_columns)
        write_results_xlsx(paths.results_xlsx, rows, result_columns)
        write_latency_svg(
            paths.figures / self.config.artifact_names.latency_figure,
            rows,
        )

        retrieval_records: list[dict[str, Any]] = []
        for index, (row, response) in enumerate(
            zip(rows, responses, strict=True),
            start=1,
        ):
            response_value = response or {}
            context_path = self.run_manager.context_path(
                index,
                str(row.get("question") or ""),
            )
            context_path.write_text(
                self._context_markdown(
                    question=str(row.get("question") or ""),
                    response=response_value,
                    error=str(row.get("error") or ""),
                ),
                encoding="utf-8",
            )
            trace_value = response_value.get("question_trace")
            trace = (
                dict(trace_value)
                if isinstance(trace_value, Mapping)
                else {
                    "question": row.get("question"),
                    "retrieval_mode": response_value.get("retrieval_mode"),
                    "query_variants": response_value.get("query_variants") or [],
                    "retrieved_by_query": response_value.get("retrieved_by_query")
                    or {},
                    "stage_counts": response_value.get("stage_counts") or {},
                    "token_usage": response_value.get("token_usage") or {},
                    "citations": response_value.get("citations") or [],
                }
            )
            trace["question"] = row.get("question")
            trace["context_artifact"] = context_path.name
            trace["run_metadata"] = metadata
            trace["artifacts"] = {
                "run_directory": paths.root,
                "results_csv": paths.results_csv,
                "results_xlsx": paths.results_xlsx,
                "report_html": paths.report_html,
                "config_json": paths.config_json,
                "summary_json": paths.summary_json,
                "metrics_json": paths.metrics_json,
                "retrieval_json": paths.retrieval_json,
                "logs": paths.logs,
                "context": context_path,
                "figures": paths.figures,
            }
            latency_value = trace.get("latency")
            latency = (
                dict(latency_value)
                if isinstance(latency_value, Mapping)
                else {}
            )
            latency["artifact_export_seconds"] = round(
                perf_counter() - artifact_started_at,
                6,
            )
            trace["latency"] = latency
            response_value["question_trace"] = trace
            retrieval_records.append(trace)

        summary, metrics = self._summaries(
            rows,
            responses,
            retrieval_mode=self.config.retrieval_mode,
            benchmark=benchmark,
        )
        summary.update(metadata)
        metrics.update(metadata)
        self.run_manager.write_json(paths.summary_json, summary)
        self.run_manager.write_json(paths.metrics_json, metrics)
        self.run_manager.write_json(paths.retrieval_json, retrieval_records)
        write_standalone_html(
            paths.report_html,
            rows=rows,
            responses=responses,
            summary=summary,
            metrics=metrics,
        )
        if execution_manager is not None:
            execution_manager.emit(
                "export_completed",
                stage="export",
                status="completed",
                elapsed_seconds=perf_counter() - artifact_started_at,
                payload={"artifact_root": str(paths.root)},
                source="phase3_runner",
            )
        logger.info(
            "phase3_run_complete",
            extra={
                "event": "run",
                "question_count": len(rows),
                "failed_questions": summary["failed_questions"],
            },
        )
        result = Phase3RunResult(paths=paths, summary=summary, metrics=metrics)
        close_logging(log_handlers)
        return result
