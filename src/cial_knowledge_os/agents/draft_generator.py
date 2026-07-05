"""Single-pass grounded draft generation."""

from __future__ import annotations

from time import perf_counter

from .base import AgentResult, StructuredAgent
from .state import AgentState


class DraftGenerator(StructuredAgent):
    name = "draft_generator"
    state_field = ""
    required_capabilities = frozenset({"text"})

    def build_prompt(self, state: AgentState) -> str:
        return state.composed_prompt

    def run(self, state: AgentState) -> AgentResult:
        if state.phase4_answer_status in {
            "unsupported_query", "insufficient_evidence", "generation_failed"
        }:
            started = perf_counter()
            answer = state.draft_answer or {
                "unsupported_query": (
                    "The indexed documents do not support this query; it may "
                    "require current or external data."
                ),
                "insufficient_evidence": (
                    "The selected evidence is insufficient for a reliable answer."
                ),
                "generation_failed": "Answer generation failed.",
            }[state.phase4_answer_status]
            return AgentResult(
                agent_name=self.name,
                success=state.phase4_answer_status != "generation_failed",
                output={"answer": answer},
                updated_state=state.evolve(draft_answer=answer),
                latency_ms=round((perf_counter() - started) * 1000, 3),
            )
        started = perf_counter()
        try:
            response = self.router.generate(
                self.name,
                self.build_prompt(state),
                required_capabilities=self.required_capabilities,
            )
            answer = str(getattr(response, "content", response)).strip()
            return AgentResult(
                agent_name=self.name,
                success=bool(answer),
                output={"answer": answer},
                updated_state=state.evolve(draft_answer=answer),
                diagnostics={
                    "model_used": getattr(response, "model", ""),
                    "model_profile": getattr(response, "profile", ""),
                    "fallback_used": bool(
                        getattr(response, "fallback_used", False)
                    ),
                },
                latency_ms=round((perf_counter() - started) * 1000, 3),
            )
        except Exception as exc:
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={},
                updated_state=state,
                latency_ms=round((perf_counter() - started) * 1000, 3),
                errors=[f"{type(exc).__name__}: {exc}"],
            )
