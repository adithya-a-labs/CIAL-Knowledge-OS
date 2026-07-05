"""Structured event contracts for the live command center."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

EVENT_TYPES = frozenset(
    {
        "run_started",
        "run_completed",
        "run_failed",
        "stage_started",
        "stage_completed",
        "agent_started",
        "agent_completed",
        "agent_failed",
        "phase4_started",
        "phase4_completed",
        "evidence_selected",
        "draft_generated",
        "critic_completed",
        "compliance_completed",
        "risk_completed",
        "verification_completed",
        "consensus_decided",
        "revision_started",
        "revision_completed",
        "telemetry_update",
    }
)


@dataclass(frozen=True, slots=True)
class LiveEvent:
    """One JSON-safe pipeline or telemetry event."""

    event_type: str
    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent: str = ""
    stage: str = ""
    status: str = ""
    progress: float | None = None
    model: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported live event type: {self.event_type}")
        if self.progress is not None and not 0 <= self.progress <= 100:
            raise ValueError("progress must be between 0 and 100.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LiveEvent":
        known = {
            "event_type", "run_id", "timestamp", "agent", "stage", "status",
            "progress", "model", "data",
        }
        data_value = value.get("data")
        data = dict(data_value) if isinstance(data_value, Mapping) else {}
        data.update({str(key): item for key, item in value.items() if key not in known})
        return cls(
            event_type=str(value.get("event_type") or ""),
            run_id=str(value.get("run_id") or ""),
            timestamp=str(
                value.get("timestamp")
                or datetime.now(timezone.utc).isoformat()
            ),
            agent=str(value.get("agent") or ""),
            stage=str(value.get("stage") or ""),
            status=str(value.get("status") or ""),
            progress=(
                float(value["progress"])
                if value.get("progress") is not None
                else None
            ),
            model=str(value.get("model") or ""),
            data=data,
        )
