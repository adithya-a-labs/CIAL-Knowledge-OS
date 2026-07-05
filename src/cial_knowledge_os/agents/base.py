"""Common agent interface and structured-output recovery."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

from .state import AgentState


@dataclass(slots=True)
class AgentResult:
    agent_name: str
    success: bool
    output: dict[str, Any]
    updated_state: AgentState
    diagnostics: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "output": self.output,
            "diagnostics": self.diagnostics,
            "latency_ms": self.latency_ms,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class Agent(ABC):
    name = "agent"

    @abstractmethod
    def run(self, state: AgentState) -> AgentResult:
        """Run the agent without hidden global state."""


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S
        )
        candidate = fenced.group(1) if fenced else ""
        if not candidate:
            start, end = text.find("{"), text.rfind("}")
            candidate = text[start : end + 1] if start >= 0 < end else ""
        parsed = json.loads(candidate)
    if not isinstance(parsed, Mapping):
        raise ValueError("Structured agent output must be a JSON object.")
    return dict(parsed)


class StructuredAgent(Agent):
    """Base for agents backed by an injected ModelRouter."""

    state_field = ""
    required_capabilities = frozenset({"text", "structured_json"})

    def __init__(self, router: Any, *, agent_name: str | None = None) -> None:
        self.router = router
        if agent_name:
            self.name = agent_name

    @abstractmethod
    def build_prompt(self, state: AgentState) -> str: ...

    def normalize(self, output: dict[str, Any], state: AgentState) -> dict[str, Any]:
        return output

    def run(self, state: AgentState) -> AgentResult:
        started = perf_counter()
        warnings: list[str] = []
        errors: list[str] = []
        diagnostics: dict[str, Any] = {}
        output: dict[str, Any] = {}
        updated = state
        try:
            response = self.router.generate(
                self.name,
                self.build_prompt(state),
                required_capabilities=self.required_capabilities,
                json_mode=True,
            )
            raw = getattr(response, "content", response)
            output = self.normalize(parse_json_object(raw), state)
            diagnostics = {
                "model_used": getattr(response, "model", ""),
                "model_profile": getattr(response, "profile", ""),
                "fallback_used": bool(getattr(response, "fallback_used", False)),
                "token_estimate": getattr(response, "token_estimate", None),
            }
            if self.state_field:
                updated = state.evolve(**{self.state_field: output})
            success = True
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            success = False
        return AgentResult(
            agent_name=self.name,
            success=success,
            output=output,
            updated_state=updated,
            diagnostics=diagnostics,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            errors=errors,
            warnings=warnings,
        )
