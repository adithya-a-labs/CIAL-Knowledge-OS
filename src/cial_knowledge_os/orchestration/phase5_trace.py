"""Serializable multi-agent execution trace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..agents.base import AgentResult


@dataclass(slots=True)
class Phase5Trace:
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def capture(self, result: AgentResult, *, revision: int = 0) -> None:
        self.events.append(
            {
                "agent_name": result.agent_name,
                "input_summary": {"revision": revision},
                "output_summary": result.output,
                "latency_ms": result.latency_ms,
                "model_used": result.diagnostics.get("model_used", ""),
                "model_profile": result.diagnostics.get("model_profile", ""),
                "fallback_used": bool(
                    result.diagnostics.get("fallback_used", False)
                ),
                "success": result.success,
                "warnings": list(result.warnings),
                "errors": list(result.errors),
                "token_estimate": result.diagnostics.get("token_estimate"),
                "revision": revision,
            }
        )

    @property
    def latency_total_ms(self) -> float:
        return round(sum(float(item.get("latency_ms") or 0) for item in self.events), 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "phase5-trace-v1",
            "run_id": self.run_id,
            "events": list(self.events),
            "latency_total_ms": self.latency_total_ms,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Phase5Trace":
        trace = cls(run_id=str(value.get("run_id") or ""))
        trace.events = [
            dict(item) for item in value.get("events") or []
            if isinstance(item, Mapping)
        ]
        return trace
