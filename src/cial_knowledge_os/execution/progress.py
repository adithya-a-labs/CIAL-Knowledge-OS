"""Derived progress state built only from emitted events."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any

from .events import ExecutionEvent


class ProgressTracker:
    """Consume lifecycle events and derive counters, rates, and ETA."""

    def __init__(self) -> None:
        self.started_at = monotonic()
        self.run_id = ""
        self.phase = ""
        self.run_mode = ""
        self.total = 0
        self.completed = 0
        self.failed = 0
        self.current_question_index = 0
        self.current_question_preview = ""
        self.current_stage = ""
        self.current_model = ""
        self.current_agent = ""
        self.revision_count = 0
        self.retry_count = 0
        self.checkpoint_path = ""
        self.warning_count = 0
        self.error_count = 0
        self.status_counts: Counter[str] = Counter()
        self.telemetry: dict[str, Any] = {}
        self._question_latencies: list[float] = []

    def __call__(self, event: ExecutionEvent) -> None:
        self.run_id = event.run_id
        self.phase = event.phase or self.phase
        self.run_mode = event.run_mode or self.run_mode
        if event.question_total is not None:
            self.total = event.question_total
        if event.question_index is not None:
            self.current_question_index = event.question_index
        if event.question_preview:
            self.current_question_preview = event.question_preview
        if event.stage:
            self.current_stage = event.stage
        model = event.payload.get("model") or event.payload.get("model_used")
        agent = event.payload.get("agent")
        if model:
            self.current_model = str(model)
        if agent:
            self.current_agent = str(agent)
        if event.event_type == "question_completed":
            self.completed += 1
            status = str(
                event.payload.get("answer_status")
                or event.status
                or "answered"
            ).casefold().replace(" ", "_")
            self.status_counts[status] += 1
            if event.elapsed_seconds is not None:
                self._question_latencies.append(event.elapsed_seconds)
        elif event.event_type == "question_failed":
            self.completed += 1
            self.failed += 1
            self.status_counts["generation_failed" if event.payload.get(
                "answer_status"
            ) == "generation_failed" else "failed"] += 1
            if event.elapsed_seconds is not None:
                self._question_latencies.append(event.elapsed_seconds)
        elif event.event_type == "warning":
            self.warning_count += 1
        elif event.event_type == "error":
            self.error_count += 1
        elif event.event_type == "checkpoint_written":
            self.checkpoint_path = str(event.payload.get("path") or "")
        elif event.event_type == "telemetry_update":
            self.telemetry = dict(event.payload)
        if "revision" in event.payload:
            self.revision_count = max(
                self.revision_count, int(event.payload.get("revision") or 0)
            )
        if "retry_count" in event.metrics:
            self.retry_count += int(event.metrics["retry_count"] or 0)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, monotonic() - self.started_at)

    @property
    def average_seconds(self) -> float:
        if self._question_latencies:
            return sum(self._question_latencies) / len(self._question_latencies)
        return self.elapsed_seconds / self.completed if self.completed else 0.0

    @property
    def eta_seconds(self) -> float | None:
        if not self.total or not self.completed:
            return None
        return max(0, self.total - self.completed) * self.average_seconds

    def snapshot(self) -> dict[str, Any]:
        eta = self.eta_seconds
        finish = (
            datetime.now(timezone.utc) + timedelta(seconds=eta)
            if eta is not None
            else None
        )
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "run_mode": self.run_mode,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "percent": round(
                self.completed / self.total * 100, 2
            ) if self.total else 0.0,
            "elapsed": round(self.elapsed_seconds, 3),
            "eta": round(eta, 3) if eta is not None else None,
            "estimated_finish_time": finish.isoformat() if finish else None,
            "average_seconds_per_question": round(self.average_seconds, 3),
            "throughput_questions_per_hour": round(
                self.completed / self.elapsed_seconds * 3600, 3
            ) if self.completed and self.elapsed_seconds else 0.0,
            "status_counts": dict(self.status_counts),
            "current_question_index": self.current_question_index,
            "current_question": self.current_question_preview,
            "current_stage": self.current_stage,
            "current_model": self.current_model,
            "current_agent": self.current_agent,
            "revision_count": self.revision_count,
            "retry_count": self.retry_count,
            "checkpoint_path": self.checkpoint_path,
            "warnings": self.warning_count,
            "errors": self.error_count,
            "telemetry": dict(self.telemetry),
        }
