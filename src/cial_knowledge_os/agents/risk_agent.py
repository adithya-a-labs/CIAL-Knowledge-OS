"""Enterprise and aviation risk review."""

from __future__ import annotations

from typing import Any

from .base import StructuredAgent
from .state import AgentState


class RiskAgent(StructuredAgent):
    name = "risk_agent"
    state_field = "risk_review"

    def build_prompt(self, state: AgentState) -> str:
        return f"""Review risks. Return JSON only with passed, risks,
missing_caveats, risk_level (low|medium|high), revision_required.
For each applicable operational, cybersecurity, aviation safety, governance,
and compliance risk include category, severity, likelihood, mitigation_status,
and description. Flag unsafe recommendations and overconfident language.
QUESTION: {state.question}
ANSWER: {state.draft_answer}"""

    def normalize(self, output: dict[str, Any], state: AgentState) -> dict[str, Any]:
        output.setdefault("passed", False)
        output.setdefault("risks", [])
        output.setdefault("missing_caveats", [])
        output.setdefault("risk_level", "medium")
        output.setdefault("revision_required", not bool(output["passed"]))
        return output
