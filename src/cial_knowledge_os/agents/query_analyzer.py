"""Query intent classification agent."""

from __future__ import annotations

from typing import Any

from .base import StructuredAgent
from .state import AgentState


class QueryAnalyzer(StructuredAgent):
    name = "query_analyzer"
    state_field = "query_intent"

    def build_prompt(self, state: AgentState) -> str:
        return f"""Classify this enterprise knowledge question. Return JSON only.
Allowed intents: definition, comparison, procedure, checklist, risk_analysis,
prioritization, architecture, troubleshooting, decision_support, compliance,
policy_interpretation, current_data_query, unsupported_query, mixed.
Required keys: intent, domain, requires_current_data, requires_risk_review,
requires_compliance_review, recommended_answer_depth, reasoning.
QUESTION: {state.question}"""

    def normalize(self, output: dict[str, Any], state: AgentState) -> dict[str, Any]:
        output.setdefault("intent", "mixed")
        output.setdefault("domain", "enterprise")
        output.setdefault("requires_current_data", False)
        output.setdefault("requires_risk_review", False)
        output.setdefault("requires_compliance_review", False)
        output.setdefault("recommended_answer_depth", "standard")
        output.setdefault("reasoning", "")
        return output
