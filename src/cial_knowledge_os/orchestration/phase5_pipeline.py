"""Adaptive response planning layered over the stable Phase 4 evidence engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from ..agents import (
    Agent,
    AgentState,
    AnswerCritic,
    ComplianceAgent,
    DraftGenerator,
    EvidenceVerifier,
    PromptComposer,
    QueryAnalyzer,
    ResponsePlanner,
    RiskAgent,
)
from .consensus_engine import ConsensusEngine
from .phase5_trace import Phase5Trace


class Phase5Pipeline:
    """Run local agents after Phase 4 retrieval and evidence selection."""

    def __init__(
        self,
        *,
        phase4_pipeline: Any,
        config: Mapping[str, Any] | None = None,
        model_router: Any | None = None,
        agents: Mapping[str, Agent] | None = None,
    ) -> None:
        self.phase4_pipeline = phase4_pipeline
        raw_config = dict(config or {})
        # Expose the Phase 4 config contract so existing batch exporters can
        # consume this wrapper without losing any legacy columns.
        self.config = getattr(phase4_pipeline, "config", raw_config)
        phase5 = raw_config.get("phase5", {})
        self.phase5_config = dict(phase5) if isinstance(phase5, Mapping) else {}
        self.enabled = bool(self.phase5_config.get("enabled", False))
        self.model_router = model_router
        defaults: dict[str, Agent] = {}
        if model_router is not None:
            defaults.update(
                {
                    "query_analyzer": QueryAnalyzer(model_router),
                    "response_planner": ResponsePlanner(model_router),
                    "draft_generator": DraftGenerator(model_router),
                    "critic_agent": AnswerCritic(model_router),
                    "compliance_agent": ComplianceAgent(model_router),
                    "risk_agent": RiskAgent(model_router),
                }
            )
        defaults.update(
            {
                "prompt_composer": PromptComposer(),
                "evidence_verifier": EvidenceVerifier(model_router),
                "consensus_engine": ConsensusEngine(),
            }
        )
        defaults.update(dict(agents or {}))
        self.agents = defaults
        self.metrics: dict[str, Any] = {}

    def _run_agent(
        self, name: str, state: AgentState, trace: Phase5Trace, *, revision: int = 0
    ) -> AgentState:
        agent = self.agents.get(name)
        if agent is None:
            raise RuntimeError(f"Phase 5 agent '{name}' is not configured.")
        result = agent.run(state)
        trace.capture(result, revision=revision)
        if not result.success:
            raise RuntimeError(
                f"{name} failed: " + "; ".join(result.errors or ["unknown error"])
            )
        return result.updated_state

    def answer(self, question: str, *, run_id: str | None = None) -> dict[str, Any]:
        phase4_response = dict(self.phase4_pipeline.answer(question))
        if not self.enabled:
            return phase4_response

        state = AgentState.from_phase4(
            question,
            phase4_response,
            config=self.phase5_config,
            run_id=run_id or uuid4().hex,
        )
        trace = Phase5Trace(state.run_id)
        for name in (
            "query_analyzer",
            "response_planner",
            "prompt_composer",
            "draft_generator",
            "critic_agent",
            "compliance_agent",
            "risk_agent",
            "evidence_verifier",
            "consensus_engine",
        ):
            state = self._run_agent(name, state, trace)

        revision_used = False
        decision = str(state.consensus_decision.get("decision") or "")
        max_revisions = min(1, max(0, int(self.phase5_config.get("max_revision_loops", 1))))
        if decision == "revise_once" and max_revisions:
            revision_used = True
            revision_notes = {
                "critic": state.critic_review.get("revision_instructions") or [],
                "compliance": state.compliance_review,
                "risk": state.risk_review,
                "verification": state.evidence_verification,
            }
            state = state.evolve(
                composed_prompt=(
                    state.composed_prompt
                    + "\n\nONE-TIME REVISION REQUIREMENTS\n"
                    + str(revision_notes)
                )
            )
            for name in (
                "draft_generator",
                "critic_agent",
                "compliance_agent",
                "risk_agent",
                "evidence_verifier",
                "consensus_engine",
            ):
                state = self._run_agent(name, state, trace, revision=1)

        decision = str(state.consensus_decision.get("decision") or "")
        final_status = str(
            state.consensus_decision.get("final_status")
            or state.phase4_answer_status
        )
        final_answer = (
            state.draft_answer
            if decision != "reject" or final_status in {
                "unsupported_query", "insufficient_evidence", "generation_failed"
            }
            else "The answer was rejected by Phase 5 validation."
        )
        state = state.evolve(
            final_answer=final_answer,
            metadata={
                **state.metadata,
                "revision_used": revision_used,
                "model_map": {
                    item["agent_name"]: item.get("model_profile", "")
                    for item in trace.events
                    if item.get("model_profile")
                },
            },
            trace_events=list(trace.events),
        )
        self.metrics = {
            **dict(getattr(self.phase4_pipeline, "metrics", {}) or {}),
            "agent_latency_total_ms": trace.latency_total_ms,
        }
        return {
            **phase4_response,
            "answer": final_answer,
            "answer_status": final_status,
            "phase5_enabled": True,
            "query_intent": state.query_intent,
            "response_plan": state.response_plan,
            "selected_evidence": [
                item.to_dict() for item in state.selected_evidence
            ],
            "draft_answer": state.draft_answer,
            "critic_review": state.critic_review,
            "compliance_review": state.compliance_review,
            "risk_review": state.risk_review,
            "evidence_verification": state.evidence_verification,
            "consensus_decision": state.consensus_decision,
            "revision_used": revision_used,
            "final_status": final_status,
            "agent_latency_total_ms": trace.latency_total_ms,
            "model_map": state.metadata["model_map"],
            "phase5_trace": trace.to_dict(),
            "phase5_state": state.to_dict(),
        }
