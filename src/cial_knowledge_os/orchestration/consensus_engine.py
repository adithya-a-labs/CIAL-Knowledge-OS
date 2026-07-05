"""Deterministic consensus rules for Phase 5 validation."""

from __future__ import annotations

from time import perf_counter

from ..agents.base import Agent, AgentResult
from ..agents.state import AgentState


class ConsensusEngine(Agent):
    name = "consensus_engine"

    def run(self, state: AgentState) -> AgentResult:
        started = perf_counter()
        status = state.phase4_answer_status
        critic = state.critic_review
        compliance = state.compliance_review
        risk = state.risk_review
        verification = state.evidence_verification

        if status in {"unsupported_query", "insufficient_evidence", "generation_failed"}:
            decision = "reject"
            reason = f"Phase 4 status '{status}' is preserved."
            final_status = status
            revision_allowed = False
        else:
            high_risk = str(risk.get("risk_level") or "").casefold() == "high"
            unsupported = verification.get("unsupported_claims") or []
            high_unsupported = any(
                str(item.get("severity") or "").casefold() == "high"
                for item in unsupported
                if isinstance(item, dict)
            )
            failures = [
                not bool(critic.get("passed")),
                not bool(compliance.get("passed")),
                not bool(risk.get("passed")),
                not bool(verification.get("passed")),
            ]
            if (high_risk and high_unsupported) or (
                high_risk and not bool(compliance.get("passed"))
            ):
                decision, final_status, revision_allowed = "reject", "rejected", False
                reason = "High-risk unsupported or non-compliant claims were found."
            elif any(failures):
                decision, final_status, revision_allowed = (
                    "revise_once", "answered", True
                )
                reason = "One or more validation checks require a single revision."
            else:
                decision, final_status, revision_allowed = "accept", "answered", False
                reason = "All validation checks passed."
        output = {
            "decision": decision,
            "reason": reason,
            "revision_allowed": revision_allowed,
            "final_status": final_status,
        }
        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            updated_state=state.evolve(consensus_decision=output),
            latency_ms=round((perf_counter() - started) * 1000, 3),
        )
