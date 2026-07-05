"""Grounding and citation compliance review."""

from __future__ import annotations

from typing import Any

from .base import StructuredAgent
from .state import AgentState


class ComplianceAgent(StructuredAgent):
    name = "compliance_agent"
    state_field = "compliance_review"

    def build_prompt(self, state: AgentState) -> str:
        return f"""Review answer compliance against selected evidence. Return JSON
only with passed, unsupported_claims, citation_issues, grounding_score (0..1),
revision_required. Check citation discipline, weak evidence disclosure, and
enterprise policy caution.
EVIDENCE: {[item.to_dict() for item in state.selected_evidence]}
ANSWER: {state.draft_answer}"""

    def normalize(self, output: dict[str, Any], state: AgentState) -> dict[str, Any]:
        output.setdefault("passed", False)
        output.setdefault("unsupported_claims", [])
        output.setdefault("citation_issues", [])
        output["grounding_score"] = min(
            1.0, max(0.0, float(output.get("grounding_score") or 0))
        )
        output.setdefault("revision_required", not bool(output["passed"]))
        return output
