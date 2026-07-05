"""Answer organization and completeness critic."""

from __future__ import annotations

import json
from typing import Any

from .base import StructuredAgent
from .state import AgentState


class AnswerCritic(StructuredAgent):
    name = "critic_agent"
    state_field = "critic_review"

    def build_prompt(self, state: AgentState) -> str:
        return f"""Critique only; do not rewrite. Return JSON only with:
passed, issues, severity (low|medium|high), revision_instructions.
Check completeness, missing sections, organization, repetition, unanswered
parts, weak reasoning, prioritization, and missing caveats.
QUESTION: {state.question}
PLAN: {json.dumps(state.response_plan)}
ANSWER: {state.draft_answer}"""

    def normalize(self, output: dict[str, Any], state: AgentState) -> dict[str, Any]:
        output.setdefault("passed", False)
        output.setdefault("issues", [])
        output.setdefault("severity", "medium")
        output.setdefault("revision_instructions", [])
        return output
