"""Deterministic prompt composition over Phase 4 selected evidence."""

from __future__ import annotations

import json
from time import perf_counter

from .base import Agent, AgentResult
from .state import AgentState


class PromptComposer(Agent):
    name = "prompt_composer"

    def run(self, state: AgentState) -> AgentResult:
        started = perf_counter()
        records = []
        for index, evidence in enumerate(state.selected_evidence, start=1):
            records.append(
                {
                    "reference_id": index,
                    **evidence.to_dict(),
                    "content": evidence.textual_content,
                }
            )
        prompt = f"""You are a local, evidence-grounded enterprise assistant.
Use only SELECTED EVIDENCE. Cite major claims with [n]. Do not invent facts.
Preserve and identify table, figure, image, screenshot, diagram, and OCR
evidence when present. If the Phase 4 status is unsupported_query or
insufficient_evidence, preserve that limitation and do not force an answer.

PHASE4 STATUS
{state.phase4_answer_status}

QUESTION
{state.question}

RESPONSE PLAN
{json.dumps(state.response_plan, ensure_ascii=False)}

EVIDENCE QUALITY
{json.dumps(state.evidence_quality, ensure_ascii=False, default=str)}

SELECTED EVIDENCE
{json.dumps(records, ensure_ascii=False, default=str)}

ANSWER"""
        updated = state.evolve(composed_prompt=prompt)
        return AgentResult(
            agent_name=self.name,
            success=True,
            output={"prompt": prompt, "evidence_count": len(records)},
            updated_state=updated,
            latency_ms=round((perf_counter() - started) * 1000, 3),
        )
