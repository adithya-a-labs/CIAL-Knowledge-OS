"""Generic offline experiment execution, export, aggregation, and reporting."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .benchmark_loader import Benchmark, BenchmarkQuestion
from .evaluation_metrics import aggregate_experiment, evaluate_answer, rank_experiments
from .evaluation_report import write_recommendation_report
from .experiment_config import (
    ExperimentConfig,
    ExperimentGrid,
    ensure_experiment_configs,
)
from .token_budget import (
    DEFAULT_TIKTOKEN_ENCODING,
    TokenManager,
    create_token_manager,
)

logger = logging.getLogger(__name__)


class EvaluationPipeline(Protocol):
    config: Any
    metrics: Mapping[str, Any]

    def answer(self, question: str) -> Mapping[str, Any]: ...


PipelineFactory = Callable[[ExperimentConfig], EvaluationPipeline]
MetricHook = Callable[
    [BenchmarkQuestion, Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


class ReconfiguringPipelineFactory:
    """Reuse one indexed pipeline while applying each sweep configuration.

    This avoids reloading documents, embeddings, and the vector index for every
    configuration. Original values are restored by :class:`ExperimentRunner`.
    """

    _ALIASES = {
        "multi_query_enabled": "enable_multi_query",
        "neighbor_expansion_enabled": "enable_neighbor_expansion",
        "top_k": "retrieval_top_k",
    }

    def __init__(self, pipeline: EvaluationPipeline) -> None:
        self.pipeline = pipeline
        self._original: dict[str, Any] = {}

    def __call__(self, experiment: ExperimentConfig) -> EvaluationPipeline:
        config = self.pipeline.config
        for name, value in experiment.parameters.items():
            target = self._ALIASES.get(name, name)
            if not hasattr(config, target):
                raise AttributeError(
                    f"Pipeline config does not support experiment parameter '{name}'."
                )
            if target not in self._original:
                self._original[target] = getattr(config, target)
            setattr(config, target, value)
        hook = getattr(self.pipeline, "on_config_changed", None)
        if callable(hook):
            hook()
        return self.pipeline

    def close(self) -> None:
        for name, value in self._original.items():
            setattr(self.pipeline.config, name, value)
        self._original.clear()
        hook = getattr(self.pipeline, "on_config_changed", None)
        if callable(hook):
            hook()


CORE_EXPERIMENT_COLUMNS = [
    "experiment_id", "question_id", "question", "category", "difficulty",
    "expected_answer", "expected_keywords", "expected_behavior", "should_answer",
    "generated_answer", "answer_status", "passed_answer_test", "keyword_score",
    "safe_failure", "hallucinated", "citation_count", "citation_quality",
    "citations", "retrieved_documents", "retrieved_pages",
    "retrieved_chunk_ids", "similarity_scores", "query_variants",
    "chunks_before_deduplication", "chunks_after_deduplication",
    "chunks_after_neighbor_expansion", "merged_context_sections",
    "final_context_sections", "final_context_characters", "estimated_tokens",
    "retrieval_trace", "retrieval_top_k", "max_context_chars",
    "neighbor_window", "multi_query_enabled", "neighbor_expansion_enabled",
    "retrieval_mode", "dense_top_k", "bm25_top_k", "rrf_k",
    "max_context_tokens", "context_tokens", "token_encoding",
    "total_latency", "retrieval_latency", "context_construction_latency",
    "generation_latency", "status", "error",
]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _stage(response: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    stages = response.get("context_stages")
    return _items(stages.get(name)) if isinstance(stages, Mapping) else []


def _count(response: Mapping[str, Any], name: str) -> int:
    counts = response.get("stage_counts")
    if isinstance(counts, Mapping) and counts.get(name) is not None:
        try:
            return int(counts[name])
        except (TypeError, ValueError):
            pass
    return len(_stage(response, name))


def _metadata_trace(
    results: list[Mapping[str, Any]],
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    documents, pages, chunk_ids, scores = [], [], [], []
    for result in results:
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        documents.append(
            metadata.get("file_name")
            or metadata.get("source")
            or result.get("source")
        )
        pages.append(result.get("page_number", metadata.get("page_number")))
        chunk_ids.append(result.get("chunk_id", metadata.get("chunk_id")))
        scores.append(result.get("score"))
    return documents, pages, chunk_ids, scores


def _response_row(
    question: BenchmarkQuestion,
    config: ExperimentConfig,
    response: Mapping[str, Any],
    metrics: Mapping[str, Any],
    elapsed: float,
    metric_hooks: Iterable[MetricHook],
    token_manager: TokenManager,
) -> dict[str, Any]:
    answer = str(
        response.get("generated_answer")
        or response.get("answer")
        or response.get("raw_answer")
        or ""
    )
    citations = response.get("citations") or []
    evaluation = evaluate_answer(
        question,
        answer,
        answer_status=str(response.get("answer_status") or ""),
        citations=citations if isinstance(citations, Sequence) else (),
    )
    final_results = _stage(response, "compressed")
    retrieved = _items(response.get("retrieved"))
    traced = final_results or retrieved
    documents, pages, chunks, scores = _metadata_trace(traced)
    context = str(response.get("context") or "")
    parameters = dict(config.parameters)
    aliases = {
        "retrieval_top_k": parameters.get(
            "retrieval_top_k", parameters.get("top_k", "")
        ),
        "max_context_chars": parameters.get("max_context_chars", ""),
        "neighbor_window": parameters.get("neighbor_window", ""),
        "multi_query_enabled": parameters.get(
            "multi_query_enabled", parameters.get("enable_multi_query", "")
        ),
        "neighbor_expansion_enabled": parameters.get(
            "neighbor_expansion_enabled",
            parameters.get("enable_neighbor_expansion", ""),
        ),
        "retrieval_mode": parameters.get("retrieval_mode", ""),
        "dense_top_k": parameters.get("dense_top_k", ""),
        "bm25_top_k": parameters.get("bm25_top_k", ""),
        "rrf_k": parameters.get("rrf_k", ""),
        "max_context_tokens": parameters.get("max_context_tokens", ""),
    }
    token_usage = response.get("token_usage")
    context_tokens = token_manager.count(context)
    token_encoding = (
        token_usage.get("encoding_name", token_manager.encoding_name)
        if isinstance(token_usage, Mapping)
        else token_manager.encoding_name
    )
    row: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "question_id": question.question_id,
        "question": question.question,
        "category": question.category,
        "difficulty": question.difficulty,
        "expected_answer": question.expected_answer,
        "expected_keywords": _json(question.expected_keywords),
        "expected_behavior": question.expected_behavior,
        "should_answer": question.should_answer,
        "generated_answer": answer,
        "answer_status": response.get("answer_status") or "",
        **evaluation,
        "citations": _json(citations),
        "retrieved_documents": _json(documents),
        "retrieved_pages": _json(pages),
        "retrieved_chunk_ids": _json(chunks),
        "similarity_scores": _json(scores),
        "query_variants": _json(response.get("query_variants") or []),
        "chunks_before_deduplication": _count(response, "retrieved")
        or len(retrieved),
        "chunks_after_deduplication": _count(response, "deduplicated"),
        "chunks_after_neighbor_expansion": _count(response, "expanded"),
        "merged_context_sections": _count(response, "merged"),
        "final_context_sections": _count(response, "compressed"),
        "final_context_characters": len(context),
        # Preserve the legacy name while reporting an exact tokenizer count.
        "estimated_tokens": context_tokens,
        "context_tokens": context_tokens,
        "token_encoding": token_encoding,
        "retrieval_trace": _json(
            response.get("retrieval_trace")
            or {
                "query_variants": response.get("query_variants") or [],
                "retrieved_by_query": response.get("retrieved_by_query") or {},
                "stage_counts": response.get("stage_counts") or {},
            }
        ),
        **aliases,
        "total_latency": float(
            metrics.get("total_pipeline_latency")
            or metrics.get("total_latency")
            or elapsed
        ),
        "retrieval_latency": float(metrics.get("retrieval_latency") or 0.0),
        "context_construction_latency": float(
            metrics.get("context_construction_latency") or 0.0
        ),
        "generation_latency": float(metrics.get("generation_latency") or 0.0),
        "status": "success",
        "error": "",
        **{f"config_{key}": value for key, value in parameters.items()},
    }
    for hook in metric_hooks:
        row.update(hook(question, response, row))
    return row


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted(
        {key for row in rows for key in row if key not in CORE_EXPERIMENT_COLUMNS}
    )
    columns = [*CORE_EXPERIMENT_COLUMNS, *extras]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True, slots=True)
class ExperimentSweepResult:
    output_root: Path
    experiment_files: tuple[Path, ...]
    summary_file: Path
    recommendation_file: Path
    dashboard_file: Path
    summaries: tuple[dict[str, Any], ...]


class ExperimentRunner:
    """Run configurations and always regenerate downstream offline reports."""

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory,
        benchmark: Benchmark,
        output_root: str | Path,
        metric_hooks: Iterable[MetricHook] = (),
        continue_on_error: bool = True,
    ) -> None:
        self.pipeline_factory = pipeline_factory
        self.benchmark = benchmark
        self.output_root = Path(output_root).expanduser().resolve()
        self.metric_hooks = tuple(metric_hooks)
        self.continue_on_error = continue_on_error

    def run(
        self,
        configurations: ExperimentGrid
        | Iterable[ExperimentConfig | Mapping[str, Any]],
    ) -> ExperimentSweepResult:
        configs = ensure_experiment_configs(configurations)
        logger.info(
            "evaluation_sweep_started",
            extra={
                "event": "evaluation",
                "configuration_count": len(configs),
                "question_count": len(self.benchmark.questions),
            },
        )
        experiment_files: list[Path] = []
        summaries: list[dict[str, Any]] = []
        try:
            for config in configs:
                logger.info(
                    "evaluation_configuration_started",
                    extra={
                        "event": "evaluation",
                        "experiment_id": config.experiment_id,
                    },
                )
                pipeline = self.pipeline_factory(config)
                token_manager_value = getattr(pipeline, "token_manager", None)
                token_manager = (
                    token_manager_value
                    if isinstance(token_manager_value, TokenManager)
                    else create_token_manager(
                        encoding_name=str(
                            getattr(
                                pipeline.config,
                                "tokenizer_encoding_name",
                                DEFAULT_TIKTOKEN_ENCODING,
                            )
                        )
                    )
                )
                rows: list[dict[str, Any]] = []
                for question in self.benchmark.questions:
                    started = time.perf_counter()
                    try:
                        response = pipeline.answer(question.question)
                        elapsed = time.perf_counter() - started
                        rows.append(
                            _response_row(
                                question,
                                config,
                                response,
                                getattr(pipeline, "metrics", {}),
                                elapsed,
                                self.metric_hooks,
                                token_manager,
                            )
                        )
                    except Exception as exc:
                        if not self.continue_on_error:
                            raise
                        logger.exception(
                            "evaluation_question_failed",
                            extra={
                                "event": "evaluation",
                                "experiment_id": config.experiment_id,
                                "question_id": question.question_id,
                            },
                        )
                        rows.append(
                            {
                                "experiment_id": config.experiment_id,
                                "question_id": question.question_id,
                                "question": question.question,
                                "category": question.category,
                                "difficulty": question.difficulty,
                                "expected_answer": question.expected_answer,
                                "expected_keywords": _json(question.expected_keywords),
                                "expected_behavior": question.expected_behavior,
                                "should_answer": question.should_answer,
                                "status": "failed",
                                "error": str(exc),
                                "total_latency": time.perf_counter() - started,
                                **{
                                    f"config_{key}": value
                                    for key, value in config.parameters.items()
                                },
                            }
                        )
                experiment_path = (
                    self.output_root / "experiments" / f"{config.experiment_id}.csv"
                )
                _write_csv(experiment_path, rows)
                experiment_files.append(experiment_path.resolve())
                summary = aggregate_experiment(rows)
                summary["failed_questions"] = sum(
                    row.get("status") == "failed" for row in rows
                )
                summaries.append(summary)
                logger.info(
                    "evaluation_configuration_complete",
                    extra={
                        "event": "evaluation",
                        "experiment_id": config.experiment_id,
                        "failed_questions": summary["failed_questions"],
                    },
                )
        finally:
            close = getattr(self.pipeline_factory, "close", None)
            if callable(close):
                close()

        ranked = rank_experiments(summaries)
        summary_path = self.output_root / "summary" / "experiment_summary.csv"
        _write_summary(summary_path, ranked)
        recommendation_path, recommendations = write_recommendation_report(
            ranked,
            self.output_root / "reports" / "recommendation.md",
        )
        from .visualization_dashboard import generate_dashboard

        dashboard_path = generate_dashboard(
            self.output_root,
            summaries=ranked,
            experiment_rows=None,
            recommendations=recommendations,
        )
        logger.info(
            "evaluation_sweep_complete",
            extra={
                "event": "evaluation",
                "configuration_count": len(configs),
            },
        )
        return ExperimentSweepResult(
            output_root=self.output_root,
            experiment_files=tuple(experiment_files),
            summary_file=summary_path.resolve(),
            recommendation_file=recommendation_path,
            dashboard_file=dashboard_path,
            summaries=tuple(ranked),
        )


def _write_summary(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: _json(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
        )
    columns = sorted({key for row in flattened for key in row})
    preferred = ["rank", "experiment_id", "overall_score"]
    fieldnames = [key for key in preferred if key in columns] + [
        key for key in columns if key not in preferred
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
