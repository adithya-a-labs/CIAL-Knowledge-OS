"""Adaptive response planning layered over the stable Phase 4 evidence engine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from statistics import fmean
from time import perf_counter
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
    """Run local agents after Phase 4 retrieval and evidence selection.

    ``event_bus`` is an optional observer. When omitted, no live module is
    imported and execution remains identical to the non-live workflow.
    """

    def __init__(
        self,
        *,
        phase4_pipeline: Any,
        config: Mapping[str, Any] | None = None,
        model_router: Any | None = None,
        agents: Mapping[str, Agent] | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.phase4_pipeline = phase4_pipeline
        raw_config = dict(config or {})
        self.config = getattr(phase4_pipeline, "config", raw_config)
        phase5 = raw_config.get("phase5", {})
        self.phase5_config = dict(phase5) if isinstance(phase5, Mapping) else {}
        self.enabled = bool(self.phase5_config.get("enabled", False))
        self.model_router = model_router
        self.event_bus = event_bus
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

    def _emit(
        self,
        event_type: str,
        *,
        run_id: str,
        agent: str = "",
        stage: str = "",
        status: str = "",
        progress: float | None = None,
        model: str = "",
        data: Mapping[str, Any] | None = None,
    ) -> None:
        if self.event_bus is None:
            return
        from ..live.schemas import LiveEvent

        self.event_bus.publish(
            LiveEvent(
                event_type=event_type,
                run_id=run_id,
                agent=agent,
                stage=stage,
                status=status,
                progress=progress,
                model=model,
                data=dict(data or {}),
            )
        )

    def _agent_model(self, name: str) -> str:
        if self.model_router is None:
            return ""
        try:
            profile = self.model_router.profile_for(name)
            return str(getattr(profile, "model", "") or "")
        except Exception:
            return ""

    def _emit_result_alias(
        self,
        name: str,
        state: AgentState,
        output: Mapping[str, Any],
        *,
        progress: float | None,
        model: str,
    ) -> None:
        supported = {
            "draft_generator",
            "critic_agent",
            "compliance_agent",
            "risk_agent",
            "evidence_verifier",
            "consensus_engine",
        }
        if name not in supported:
            return
        output = output if isinstance(output, Mapping) else {}
        aliases: dict[str, tuple[str, dict[str, Any]]] = {
            "draft_generator": (
                "draft_generated",
                {"answer": output.get("answer", "")},
            ),
            "critic_agent": (
                "critic_completed",
                {
                    "passed": output.get("passed"),
                    "issue_count": len(output.get("issues") or []),
                    "severity": output.get("severity"),
                },
            ),
            "compliance_agent": (
                "compliance_completed",
                {
                    "passed": output.get("passed"),
                    "grounding_score": output.get("grounding_score"),
                },
            ),
            "risk_agent": (
                "risk_completed",
                {
                    "passed": output.get("passed"),
                    "risk_level": output.get("risk_level"),
                },
            ),
            "evidence_verifier": (
                "verification_completed",
                {
                    "passed": output.get("passed"),
                    "verification_rate": output.get("verification_rate"),
                    "unsupported_claim_count": len(
                        output.get("unsupported_claims") or []
                    ),
                    "citation_mismatch_count": len(
                        output.get("citation_mismatches") or []
                    ),
                },
            ),
            "consensus_engine": (
                "consensus_decided",
                {
                    "decision": output.get("decision"),
                    "final_status": output.get("final_status"),
                    "reason": output.get("reason"),
                },
            ),
        }
        event_type, data = aliases[name]
        self._emit(
            event_type,
            run_id=state.run_id,
            agent=name,
            stage=name,
            status="completed",
            progress=progress,
            model=model,
            data=data,
        )

    def _run_agent(
        self,
        name: str,
        state: AgentState,
        trace: Phase5Trace,
        *,
        revision: int = 0,
        progress: float | None = None,
    ) -> AgentState:
        agent = self.agents.get(name)
        if agent is None:
            raise RuntimeError(f"Phase 5 agent '{name}' is not configured.")
        configured_model = self._agent_model(name)
        self._emit(
            "stage_started",
            run_id=state.run_id,
            agent=name,
            stage=name,
            status="running",
            progress=progress,
            model=configured_model,
            data={"revision": revision},
        )
        self._emit(
            "agent_started",
            run_id=state.run_id,
            agent=name,
            stage=name,
            status="running",
            progress=progress,
            model=configured_model,
            data={"revision": revision},
        )
        try:
            result = agent.run(state)
        except Exception as exc:
            self._emit(
                "agent_failed",
                run_id=state.run_id,
                agent=name,
                stage=name,
                status="failed",
                progress=progress,
                model=configured_model,
                data={"errors": [f"{type(exc).__name__}: {exc}"]},
            )
            raise
        trace.capture(result, revision=revision)
        model = str(result.diagnostics.get("model_used") or configured_model)
        event_data = {
            "latency_ms": result.latency_ms,
            "model_used": model,
            "fallback_used": bool(
                result.diagnostics.get("fallback_used", False)
            ),
            "tokens_generated": result.diagnostics.get("token_estimate"),
            "warnings": list(result.warnings),
            "errors": list(result.errors),
            "summary": result.output,
            "revision": revision,
        }
        if not result.success:
            self._emit(
                "agent_failed",
                run_id=state.run_id,
                agent=name,
                stage=name,
                status="failed",
                progress=progress,
                model=model,
                data=event_data,
            )
            raise RuntimeError(
                f"{name} failed: "
                + "; ".join(result.errors or ["unknown error"])
            )
        self._emit(
            "agent_completed",
            run_id=state.run_id,
            agent=name,
            stage=name,
            status="completed",
            progress=progress,
            model=model,
            data=event_data,
        )
        self._emit(
            "stage_completed",
            run_id=state.run_id,
            agent=name,
            stage=name,
            status="completed",
            progress=progress,
            model=model,
            data={"latency_ms": result.latency_ms, "revision": revision},
        )
        self._emit_result_alias(
            name,
            state,
            result.output,
            progress=progress,
            model=model,
        )
        return result.updated_state

    @staticmethod
    def _evidence_metrics(
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected = [
            item
            for item in response.get("selected_evidence") or []
            if isinstance(item, Mapping)
        ]
        scores = []
        sources = set()
        modalities: Counter[str] = Counter()
        for item in selected:
            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            score = (
                item.get("score")
                if item.get("score") is not None
                else item.get("reranker_score")
            )
            try:
                if score is not None:
                    scores.append(float(score))
            except (TypeError, ValueError):
                pass
            source = (
                item.get("relative_path")
                or item.get("source")
                or metadata.get("source")
            )
            if source:
                sources.add(str(source))
            modalities[
                str(item.get("modality") or metadata.get("modality") or "text")
            ] += 1
        sufficiency = min(
            1.0,
            max(0.0, fmean(scores) if scores else len(selected) / 3),
        )
        return {
            "selected_evidence_count": len(selected),
            "evidence_sufficiency_score": round(sufficiency, 4),
            "source_diversity": len(sources),
            "modality_mix": dict(modalities),
        }

    def answer(self, question: str, *, run_id: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            return dict(self.phase4_pipeline.answer(question))

        execution_run_id = run_id or uuid4().hex
        self._emit(
            "run_started",
            run_id=execution_run_id,
            stage="Starting",
            status="running",
            progress=0,
            data={"question": question},
        )
        try:
            phase4_started = perf_counter()
            self._emit(
                "phase4_started",
                run_id=execution_run_id,
                agent="phase4_retrieval",
                stage="Phase 4 retrieval",
                status="running",
                progress=3,
            )
            self._emit(
                "stage_started",
                run_id=execution_run_id,
                agent="phase4_retrieval",
                stage="Phase 4 retrieval",
                status="running",
                progress=3,
            )
            phase4_response = dict(self.phase4_pipeline.answer(question))
            self._emit(
                "phase4_completed",
                run_id=execution_run_id,
                agent="phase4_retrieval",
                stage="Phase 4 retrieval",
                status="completed",
                progress=12,
                data={
                    "latency_ms": round(
                        (perf_counter() - phase4_started) * 1000, 3
                    ),
                    "answer_status": phase4_response.get("answer_status"),
                },
            )
            self._emit(
                "stage_completed",
                run_id=execution_run_id,
                agent="phase4_retrieval",
                stage="Phase 4 retrieval",
                status="completed",
                progress=12,
            )
            self._emit(
                "stage_started",
                run_id=execution_run_id,
                agent="evidence_selection",
                stage="Evidence selection",
                status="running",
                progress=14,
            )
            self._emit(
                "evidence_selected",
                run_id=execution_run_id,
                agent="evidence_selection",
                stage="Evidence selection",
                status="completed",
                progress=18,
                data=self._evidence_metrics(phase4_response),
            )
            self._emit(
                "stage_completed",
                run_id=execution_run_id,
                agent="evidence_selection",
                stage="Evidence selection",
                status="completed",
                progress=18,
            )

            state = AgentState.from_phase4(
                question,
                phase4_response,
                config=self.phase5_config,
                run_id=execution_run_id,
            )
            trace = Phase5Trace(state.run_id)
            progress_map = {
                "query_analyzer": 25,
                "response_planner": 32,
                "prompt_composer": 42,
                "draft_generator": 52,
                "critic_agent": 60,
                "compliance_agent": 67,
                "risk_agent": 74,
                "evidence_verifier": 82,
                "consensus_engine": 89,
            }
            for name, progress in progress_map.items():
                state = self._run_agent(
                    name, state, trace, progress=progress
                )

            revision_used = False
            decision = str(state.consensus_decision.get("decision") or "")
            max_revisions = min(
                1,
                max(
                    0,
                    int(self.phase5_config.get("max_revision_loops", 1)),
                ),
            )
            if decision == "revise_once" and max_revisions:
                revision_used = True
                self._emit(
                    "revision_started",
                    run_id=execution_run_id,
                    stage="One-time revision",
                    status="running",
                    progress=90,
                    data={"revision": 1},
                )
                revision_notes = {
                    "critic": (
                        state.critic_review.get("revision_instructions") or []
                    ),
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
                    state = self._run_agent(
                        name,
                        state,
                        trace,
                        revision=1,
                        progress=93,
                    )
                self._emit(
                    "revision_completed",
                    run_id=execution_run_id,
                    stage="One-time revision",
                    status="completed",
                    progress=95,
                    data={"revision": 1},
                )

            self._emit(
                "agent_started",
                run_id=execution_run_id,
                agent="finalizer",
                stage="Finalizer",
                status="running",
                progress=97,
            )
            self._emit(
                "stage_started",
                run_id=execution_run_id,
                agent="finalizer",
                stage="Finalizer",
                status="running",
                progress=97,
            )
            decision = str(state.consensus_decision.get("decision") or "")
            final_status = str(
                state.consensus_decision.get("final_status")
                or state.phase4_answer_status
            )
            final_answer = (
                state.draft_answer
                if decision != "reject"
                or final_status
                in {
                    "unsupported_query",
                    "insufficient_evidence",
                    "generation_failed",
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
            self._emit(
                "agent_completed",
                run_id=execution_run_id,
                agent="finalizer",
                stage="Finalizer",
                status="completed",
                progress=99,
                data={
                    "latency_ms": 0.0,
                    "summary": {"final_status": final_status},
                },
            )
            self._emit(
                "stage_completed",
                run_id=execution_run_id,
                agent="finalizer",
                stage="Finalizer",
                status="completed",
                progress=99,
            )
            self.metrics = {
                **dict(getattr(self.phase4_pipeline, "metrics", {}) or {}),
                "agent_latency_total_ms": trace.latency_total_ms,
            }
            response = {
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
            self._emit(
                "run_completed",
                run_id=execution_run_id,
                stage="Completed",
                status="completed",
                progress=100,
                data={
                    "final_status": final_status,
                    "answer": final_answer,
                    "citations": response.get("citations") or [],
                    "revision_used": revision_used,
                },
            )
            return response
        except Exception as exc:
            self._emit(
                "run_failed",
                run_id=execution_run_id,
                stage="Failed",
                status="failed",
                data={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise
