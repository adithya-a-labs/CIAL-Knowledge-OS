"""Typed lifecycle events for local execution observability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


EVENT_TYPES = frozenset(
    {
        "run_started",
        "run_completed",
        "run_failed",
        "run_resumed",
        "preflight_started",
        "preflight_completed",
        "stage_started",
        "stage_completed",
        "indexing_started",
        "indexing_progress",
        "indexing_completed",
        "indexing_failed",
        "qdrant_health_checked",
        "bm25_health_checked",
        "question_started",
        "retrieval_started",
        "retrieval_completed",
        "reranking_started",
        "reranking_completed",
        "evidence_selection_started",
        "evidence_selection_completed",
        "generation_started",
        "generation_completed",
        "generation_failed",
        "question_completed",
        "question_failed",
        "checkpoint_written",
        "export_started",
        "export_completed",
        "batch_completed",
        "agent_started",
        "agent_completed",
        "agent_failed",
        "consensus_decided",
        "telemetry_update",
        "warning",
        "error",
    }
)


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One JSON-safe fact emitted by a pipeline or runner."""

    event_type: str
    run_id: str
    event_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = ""
    run_mode: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stage: str = ""
    step: str = ""
    question_index: int | None = None
    question_total: int | None = None
    question_id: str = ""
    question_preview: str = ""
    status: str = ""
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    warning: str = ""
    error: str = ""
    elapsed_seconds: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event_type must not be blank.")
        if not self.run_id:
            raise ValueError("run_id must not be blank.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionEvent":
        fields = cls.__dataclass_fields__
        kwargs = {key: value[key] for key in fields if key in value}
        return cls(**kwargs)
