"""Phase 4 run orchestration built on the Phase 3 artifact lifecycle."""

from __future__ import annotations

import csv
import json
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from .benchmark_loader import Benchmark
from .config import Phase4Config
from .phase3_runner import Phase3Runner
from .phase4_reporting import write_phase4_figures, write_phase4_html
from .run_manager import RunManager, RunPaths


@dataclass(frozen=True, slots=True)
class Phase4RunResult:
    """Expose one completed Phase 4 bundle and its aggregate diagnostics.

    ``paths`` is the unchanged Phase 3-compatible run path contract; ``summary``
    and ``metrics`` include additive reranking, evidence, token, and mode fields.
    ``run_mode`` records whether the caller executed smoke, manual QA,
    benchmark, or export-only behavior.
    """

    paths: RunPaths
    summary: dict[str, Any]
    metrics: dict[str, Any]
    run_mode: str


class Phase4Runner(Phase3Runner):
    """Generate Phase 4 artifacts while reusing the Phase 3 RunManager.

    Inputs are a Phase 4-compatible pipeline/config and optional RunManager.
    :meth:`run` produces the established CSV/XLSX/HTML/JSON/log/context bundle,
    then enriches summaries, figures, and HTML with Phase 4 diagnostics.

    The class subclasses :class:`Phase3Runner` intentionally: artifact paths,
    collision handling, batch failure isolation, citations, and legacy columns
    remain one implementation. Phase 4 only appends fields and overwrites the
    report with a richer standalone view.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        config: Phase4Config | None = None,
        run_manager: RunManager | None = None,
    ) -> None:
        phase4_config = config or pipeline.config
        if not isinstance(phase4_config, Phase4Config):
            raise TypeError("Phase4Runner requires a Phase4Config.")
        super().__init__(
            pipeline=pipeline,
            config=phase4_config,
            run_manager=run_manager,
        )
        self.config = phase4_config

    def _apply_mode_limits(
        self,
        questions: Sequence[str] | None,
        *,
        run_mode: str,
    ) -> Sequence[str] | None:
        if questions is None:
            return None
        values = list(questions)
        if run_mode == "smoke":
            return values[: min(3, len(values))]
        if (
            run_mode in {"manual_qa", "export_only"}
            and len(values) > self.config.max_inline_manual_questions
            and not self.config.allow_large_run
        ):
            warnings.warn(
                "Phase 4 manual input exceeds "
                f"{self.config.max_inline_manual_questions} questions; only the "
                "first configured limit will run. Set allow_large_run=True and "
                "use export_only mode for an intentional large batch.",
                stacklevel=3,
            )
            return values[: self.config.max_inline_manual_questions]
        return values

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _phase4_metrics(
        rows: Sequence[Mapping[str, Any]],
        traces: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        def numbers(key: str) -> list[float]:
            values = []
            for row in rows:
                try:
                    values.append(float(row.get(key) or 0.0))
                except (TypeError, ValueError):
                    values.append(0.0)
            return values

        strengths: Counter[str] = Counter()
        discard_reasons: Counter[str] = Counter()
        diagnostic_flags: Counter[str] = Counter()
        fallback_questions = 0
        weak_evidence_questions = 0
        for trace in traces:
            quality = trace.get("evidence_quality")
            quality = quality if isinstance(quality, Mapping) else {}
            summary = quality.get("summary")
            summary = summary if isinstance(summary, Mapping) else {}
            strengths.update(summary.get("strength_distribution") or {})
            token_usage = trace.get("token_usage")
            token_usage = (
                token_usage if isinstance(token_usage, Mapping) else {}
            )
            discard_reasons.update(
                token_usage.get("discard_reason_distribution") or {}
            )
            fallback_questions += bool(token_usage.get("fallback_used"))
            weak_evidence_questions += bool(token_usage.get("weak_evidence"))
            diagnostic_flags.update(
                str(item.get("signal") or "unspecified")
                for item in (trace.get("decision_summary") or [])
                if isinstance(item, Mapping)
            )
        reductions = numbers("token_reduction_percent")
        reranker_scores = numbers("average_reranker_score")
        return {
            "average_token_reduction_percent": (
                round(fmean(reductions), 6) if reductions else 0.0
            ),
            "average_reranker_score": (
                round(fmean(reranker_scores), 6) if reranker_scores else 0.0
            ),
            "selected_chunk_count": int(sum(numbers("selected_chunk_count"))),
            "discarded_chunk_count": int(sum(numbers("discarded_chunk_count"))),
            "average_selected_chunk_count": (
                round(fmean(numbers("selected_chunk_count")), 6) if rows else 0.0
            ),
            "average_discarded_chunk_count": (
                round(fmean(numbers("discarded_chunk_count")), 6) if rows else 0.0
            ),
            "average_reranker_latency_seconds": (
                round(fmean(numbers("reranker_latency_seconds")), 6)
                if rows
                else 0.0
            ),
            "average_selected_evidence_tokens": (
                round(fmean(numbers("selected_evidence_tokens")), 6)
                if rows
                else 0.0
            ),
            "average_citation_count": (
                round(fmean(numbers("citation_count")), 6) if rows else 0.0
            ),
            "fallback_question_count": fallback_questions,
            "weak_evidence_question_count": weak_evidence_questions,
            "discard_reason_distribution": dict(
                sorted(discard_reasons.items())
            ),
            "diagnostic_flag_counts": dict(
                sorted(diagnostic_flags.items())
            ),
            "evidence_strength_distribution": {
                name: strengths.get(name, 0)
                for name in ("strong", "medium", "weak")
            },
        }

    def run(
        self,
        *,
        questions: list[str] | tuple[str, ...] | None = None,
        questions_path: str | Path | None = None,
        benchmark: Benchmark | None = None,
        top_k: int | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        run_mode: str | None = None,
    ) -> Phase4RunResult:
        """Execute Phase 4 and write the complete compatible artifact bundle.

        Inputs match :class:`Phase3Runner` plus ``run_mode``. Smoke mode caps an
        in-memory list at three questions. Manual/export-only lists above the
        configured notebook-safe limit are warned and truncated unless
        ``allow_large_run`` is explicit. Benchmark mode is never silently
        truncated. Outputs include Phase 4 CSV fields, serialized traces,
        manager-friendly XLSX, seven SVG diagnostics, and standalone HTML.
        """

        effective_mode = run_mode or self.config.phase4_run_mode
        allowed = {"smoke", "manual_qa", "benchmark", "export_only"}
        if effective_mode not in allowed:
            raise ValueError(
                "run_mode must be smoke, manual_qa, benchmark, or export_only."
            )
        mode_questions: Sequence[str] | None = questions
        if (
            mode_questions is None
            and benchmark is not None
            and effective_mode == "smoke"
        ):
            mode_questions = [item.question for item in benchmark.questions]
        limited_questions = self._apply_mode_limits(
            mode_questions,
            run_mode=effective_mode,
        )
        metadata = dict(run_metadata or {})
        metadata.setdefault("run_mode", effective_mode)
        phase3_result = super().run(
            questions=list(limited_questions)
            if limited_questions is not None
            else None,
            questions_path=questions_path,
            benchmark=benchmark,
            top_k=top_k,
            run_metadata=metadata,
        )
        started = perf_counter()
        rows = self._read_rows(phase3_result.paths.results_csv)
        traces_value = json.loads(
            phase3_result.paths.retrieval_json.read_text(encoding="utf-8")
        )
        traces = [
            dict(trace)
            for trace in traces_value
            if isinstance(trace, Mapping)
        ]
        additions = self._phase4_metrics(rows, traces)
        summary = dict(phase3_result.summary) | {
            "phase": "Phase 4",
            "qualification_status": "implemented_not_benchmark_qualified",
            "run_mode": effective_mode,
            "average_token_reduction_percent": additions[
                "average_token_reduction_percent"
            ],
        }
        metrics = dict(phase3_result.metrics) | additions | {
            "phase": "Phase 4",
            "run_mode": effective_mode,
        }
        write_phase4_figures(phase3_result.paths.figures, traces)
        elapsed = perf_counter() - started
        for trace in traces:
            latency = trace.get("latency")
            latency = dict(latency) if isinstance(latency, Mapping) else {}
            latency["artifact_export_seconds"] = round(
                float(latency.get("artifact_export_seconds") or 0.0) + elapsed,
                6,
            )
            trace["latency"] = latency
        self.run_manager.write_json(phase3_result.paths.summary_json, summary)
        self.run_manager.write_json(phase3_result.paths.metrics_json, metrics)
        self.run_manager.write_json(phase3_result.paths.retrieval_json, traces)
        write_phase4_html(
            phase3_result.paths.report_html,
            rows=rows,
            traces=traces,
            summary=summary,
            metrics=metrics,
        )
        return Phase4RunResult(
            paths=phase3_result.paths,
            summary=summary,
            metrics=metrics,
            run_mode=effective_mode,
        )
