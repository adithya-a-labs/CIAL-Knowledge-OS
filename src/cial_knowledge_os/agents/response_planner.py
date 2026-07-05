"""Evidence-aware response format planning agent."""

from __future__ import annotations

import json
from typing import Any

from .base import StructuredAgent
from .state import AgentState


class ResponsePlanner(StructuredAgent):
    name = "response_planner"
    state_field = "response_plan"

    def build_prompt(self, state: AgentState) -> str:
        modalities = sorted({item.modality for item in state.selected_evidence})
        return f"""Plan an enterprise answer. Return JSON only.
Allowed formats: executive_brief, comparison_table, checklist, risk_matrix,
priority_matrix, step_by_step_procedure, architecture_explanation,
decision_report, narrative_synthesis, troubleshooting_guide, compliance_mapping.
Required keys: format, sections, citation_strategy, tone, must_include, avoid,
reasoning. Preserve available non-text evidence in the plan.
QUESTION: {state.question}
INTENT: {json.dumps(state.query_intent)}
PHASE4_STATUS: {state.phase4_answer_status}
EVIDENCE_COUNT: {len(state.selected_evidence)}
MODALITIES: {modalities}"""

    def normalize(self, output: dict[str, Any], state: AgentState) -> dict[str, Any]:
        output.setdefault("format", "narrative_synthesis")
        output.setdefault("sections", ["Direct answer", "Evidence", "Caveats"])
        output.setdefault("citation_strategy", "cite every major claim")
        output.setdefault("tone", "enterprise")
        output.setdefault("must_include", [])
        output.setdefault("avoid", ["unsupported claims"])
        output.setdefault("reasoning", "")
        return output
