"""Run-scoped coordinator for passive execution observers."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any, Mapping
from uuid import uuid4

from .event_bus import EventBus
from .events import ExecutionEvent
from .json_trace import JSONTraceWriter, ProgressSnapshotWriter
from .metrics import MetricsCollector
from .progress import ProgressTracker
from .renderers import create_console_renderer
from .schemas import ExecutionOptions
from .telemetry import TelemetryCollector


class ExecutionManager:
    """Emit facts about work while remaining unable to schedule or decide it."""

    def __init__(
        self,
        *,
        phase: str = "",
        run_mode: str = "",
        run_id: str | None = None,
        options: ExecutionOptions | None = None,
        event_bus: EventBus | None = None,
        project_path: str | Path | None = None,
    ) -> None:
        self.options = options or ExecutionOptions()
        self.run_id = run_id or uuid4().hex
        self.phase = phase
        self.run_mode = run_mode
        self.event_bus = event_bus or EventBus(enabled=self.options.enabled)
        self.progress = ProgressTracker()
        self.metrics = MetricsCollector()
        self._started_at = monotonic()
        self._stage_started: dict[str, float] = {}
        self._question_started: float | None = None
        self._current_question_index: int | None = None
        self._current_question_total: int | None = None
        self._current_question_id = ""
        self._current_question_preview = ""
        self._last_telemetry = 0.0
        self._telemetry = TelemetryCollector(project_path=project_path)
        self.run_dir = self.options.output_dir / self.run_id
        self._snapshot_writer: ProgressSnapshotWriter | None = None
        self._progress_log_path: Path | None = None
        if self.options.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.event_bus.subscribe(self.progress)
            self.event_bus.subscribe(self.metrics)
            if self.options.trace_jsonl:
                self.event_bus.subscribe(
                    JSONTraceWriter(self.run_dir / "execution_trace.jsonl")
                )
            self._snapshot_writer = ProgressSnapshotWriter(
                self.run_dir / "progress.json"
            )
            self.event_bus.subscribe(self._write_progress)
            if self.options.progress_log:
                self._progress_log_path = self.run_dir / "progress.log"
            if self.options.console:
                self.event_bus.subscribe(
                    create_console_renderer(
                        self.progress,
                        rich=self.options.rich,
                        refresh_seconds=self.options.console_refresh_seconds,
                    )
                )

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        phase: str = "",
        run_mode: str = "",
        run_id: str | None = None,
    ) -> "ExecutionManager":
        return cls(
            phase=phase,
            run_mode=run_mode,
            run_id=run_id,
            options=ExecutionOptions.from_config(config),
            project_path=getattr(config, "project_root", None),
        )

    @classmethod
    def disabled(cls) -> "ExecutionManager":
        return cls(options=ExecutionOptions(enabled=False, console=False))

    @property
    def enabled(self) -> bool:
        return self.options.enabled

    def _write_progress(self, event: ExecutionEvent) -> None:
        snapshot = self.progress.snapshot()
        snapshot["metrics"] = self.metrics.summary()
        if self._snapshot_writer is not None:
            self._snapshot_writer.write(snapshot)
        if self._progress_log_path is not None:
            with self._progress_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{event.timestamp} {event.event_type} "
                    f"status={event.status or '-'} stage={event.stage or '-'} "
                    f"progress={snapshot['completed']}/{snapshot['total']} "
                    f"message={event.message or event.error or event.warning}\n"
                )

    def emit(
        self,
        event_type: str,
        *,
        stage: str = "",
        step: str = "",
        status: str = "",
        message: str = "",
        metrics: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        warning: str = "",
        error: str = "",
        elapsed_seconds: float | None = None,
        source: str = "",
        question_index: int | None = None,
        question_total: int | None = None,
        question_id: str = "",
        question_preview: str = "",
    ) -> ExecutionEvent:
        event = ExecutionEvent(
            event_type=event_type,
            run_id=self.run_id,
            phase=self.phase,
            run_mode=self.run_mode,
            stage=stage,
            step=step,
            status=status,
            message=message,
            metrics=dict(metrics or {}),
            payload=dict(payload or {}),
            warning=warning,
            error=error,
            elapsed_seconds=elapsed_seconds,
            source=source,
            question_index=(
                question_index
                if question_index is not None
                else self._current_question_index
            ),
            question_total=(
                question_total
                if question_total is not None
                else self._current_question_total
            ),
            question_id=question_id or self._current_question_id,
            question_preview=(
                question_preview or self._current_question_preview
            )[:240],
        )
        self.event_bus.emit(event)
        self._maybe_emit_telemetry(event_type)
        return event

    def _maybe_emit_telemetry(self, triggering_event: str) -> None:
        if (
            not self.options.enabled
            or not self.options.telemetry
            or triggering_event == "telemetry_update"
        ):
            return
        now = monotonic()
        if now - self._last_telemetry < self.options.telemetry_interval_seconds:
            return
        self._last_telemetry = now
        self.emit(
            "telemetry_update",
            stage=self.progress.current_stage,
            status="observed",
            payload=self._telemetry.collect(),
            source="execution.telemetry",
        )

    def start_run(self, *, total_questions: int = 0, resumed: bool = False) -> None:
        self._started_at = monotonic()
        self.emit(
            "run_resumed" if resumed else "run_started",
            status="running",
            question_total=total_questions,
            payload={"output_dir": str(self.run_dir)},
            source="execution.manager",
        )

    def complete_run(self, **payload: Any) -> None:
        self.emit(
            "run_completed",
            status="completed",
            elapsed_seconds=monotonic() - self._started_at,
            payload=payload,
            source="execution.manager",
        )

    def fail_run(self, error: BaseException | str) -> None:
        self.emit(
            "run_failed",
            status="failed",
            error=str(error),
            elapsed_seconds=monotonic() - self._started_at,
            source="execution.manager",
        )

    def start_stage(self, stage: str, *, event_type: str = "stage_started",
                    **payload: Any) -> None:
        self._stage_started[stage] = monotonic()
        self.emit(event_type, stage=stage, status="running", payload=payload)

    def complete_stage(
        self, stage: str, *, event_type: str = "stage_completed",
        metrics: Mapping[str, Any] | None = None, **payload: Any
    ) -> None:
        started = self._stage_started.pop(stage, None)
        self.emit(
            event_type,
            stage=stage,
            status="completed",
            elapsed_seconds=monotonic() - started if started else None,
            metrics=metrics,
            payload=payload,
        )

    def start_question(
        self, index: int, total: int, question: str, *, question_id: str = ""
    ) -> None:
        self._question_started = monotonic()
        self._current_question_index = index
        self._current_question_total = total
        self._current_question_id = question_id
        self._current_question_preview = question
        self.emit(
            "question_started",
            stage="question",
            status="running",
            question_index=index,
            question_total=total,
            question_id=question_id,
            question_preview=question,
        )

    def complete_question(
        self, *, answer_status: str = "answered", **payload: Any
    ) -> None:
        elapsed = (
            monotonic() - self._question_started
            if self._question_started is not None else None
        )
        self.emit(
            "question_completed",
            stage="question",
            status="completed",
            elapsed_seconds=elapsed,
            payload={"answer_status": answer_status, **payload},
        )
        self._question_started = None
        self._clear_question_context()

    def fail_question(self, error: BaseException | str, **payload: Any) -> None:
        elapsed = (
            monotonic() - self._question_started
            if self._question_started is not None else None
        )
        self.emit(
            "question_failed",
            stage="question",
            status="failed",
            error=str(error),
            elapsed_seconds=elapsed,
            payload=payload,
        )
        self._question_started = None
        self._clear_question_context()

    def _clear_question_context(self) -> None:
        self._current_question_index = None
        self._current_question_total = None
        self._current_question_id = ""
        self._current_question_preview = ""

    def emit_warning(self, message: str, **payload: Any) -> None:
        self.emit("warning", status="warning", warning=message, payload=payload)

    def emit_error(self, message: str, **payload: Any) -> None:
        self.emit("error", status="error", error=message, payload=payload)

    def write_checkpoint_event(self, path: str | Path, **payload: Any) -> None:
        self.emit(
            "checkpoint_written",
            status="completed",
            payload={"path": str(path), **payload},
        )

class NoOpExecutionManager(ExecutionManager):
    """Explicit safe no-op manager for tests and unobserved pipelines."""

    def __init__(self) -> None:
        super().__init__(
            options=ExecutionOptions(enabled=False, console=False)
        )
