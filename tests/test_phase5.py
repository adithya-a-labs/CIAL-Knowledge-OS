from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cial_knowledge_os.agents import (
    AgentResult,
    AgentState,
    Evidence,
    ModelRouter,
    QueryAnalyzer,
)
from cial_knowledge_os.agents.evidence_verifier import EvidenceVerifier
from cial_knowledge_os.batch_qa import CSV_COLUMNS, PHASE5_CSV_COLUMNS
from cial_knowledge_os.agents.base import Agent
from cial_knowledge_os.orchestration import (
    ConsensusEngine,
    Phase5Pipeline,
    Phase5Runner,
)
from cial_knowledge_os.reporting.phase5_html import render_phase5_html
from cial_knowledge_os.reporting.phase5_visuals import decision_recommendation


class StaticAgent(Agent):
    def __init__(self, name: str, field: str, output: dict):
        self.name, self.field, self.output = name, field, output

    def run(self, state: AgentState) -> AgentResult:
        updated = state.evolve(**{self.field: self.output}) if self.field else state
        if self.name == "draft_generator":
            updated = state.evolve(draft_answer=self.output["answer"])
        return AgentResult(self.name, True, self.output, updated, latency_ms=1)


class Phase4:
    metrics = {}

    def __init__(self, status: str = "answered"):
        self.status = status
        self.config = SimpleNamespace(
            project_root=Path.cwd(),
            top_k=3,
            tokenizer_encoding_name="cl100k_base",
            ollama_model_name="mock",
            embedding_model_name="mock",
        )

    def answer(self, question: str):
        return {
            "question": question,
            "answer": "Phase 4 answer",
            "raw_answer": "Phase 4 answer",
            "answer_status": self.status,
            "selected_evidence": [
                {
                    "text": "Grounded evidence.",
                    "source": "manual.pdf",
                    "page_number": 12,
                    "chunk_id": "c1",
                    "reranker_score": 0.82,
                }
            ],
            "citations": [],
            "evidence_quality": {},
        }


def test_evidence_accepts_text_and_multimodal_fields():
    text = Evidence.from_mapping({"text": "hello", "chunk_id": "1"})
    image = Evidence(
        evidence_id="img-1",
        modality="figure",
        image_path="figures/a.png",
        caption="Architecture",
        bbox=(0, 0, 100, 80),
    )
    assert text.modality == "text"
    assert image.is_visual
    assert image.textual_content == "Architecture"


def test_agent_state_and_result_are_typed():
    state = AgentState("What is the procedure?", selected_evidence=[
        {"text": "Step one", "source": "ops.pdf"}
    ])
    result = AgentResult("mock", True, {"ok": True}, state)
    assert isinstance(state.selected_evidence[0], Evidence)
    assert result.to_dict()["success"] is True


def test_model_router_capabilities_fallback_and_json_recovery():
    class Failing:
        def generate(self, **kwargs):
            raise TimeoutError("offline model busy")

    class Working:
        def generate(self, **kwargs):
            return {"response": '```json\n{"intent":"procedure"}\n```'}

    config = {
        "phase5": {
            "model_profiles": {
                "fast": {
                    "provider": "ollama",
                    "model": "fast",
                    "capabilities": ["text", "structured_json"],
                    "fallback_profiles": ["backup"],
                },
                "backup": {
                    "provider": "ollama",
                    "model": "backup",
                    "capabilities": ["text", "structured_json"],
                },
            },
            "agents": {"query_analyzer": {"model_profile": "fast"}},
        }
    }
    router = ModelRouter(config, clients={"fast": Failing(), "backup": Working()})
    result = QueryAnalyzer(router).run(AgentState("How do I do this?"))
    assert result.success
    assert result.output["intent"] == "procedure"
    assert result.diagnostics["fallback_used"] is True


@pytest.mark.parametrize(
    ("changes", "decision"),
    [
        ({}, "accept"),
        ({"critic_review": {"passed": False, "severity": "medium"}}, "revise_once"),
        (
            {
                "compliance_review": {"passed": False},
                "risk_review": {"passed": False, "risk_level": "high"},
                "evidence_verification": {
                    "passed": False,
                    "unsupported_claims": [{"severity": "high"}],
                },
            },
            "reject",
        ),
    ],
)
def test_consensus_rules(changes, decision):
    defaults = {
        "critic_review": {"passed": True, "severity": "low"},
        "compliance_review": {"passed": True},
        "risk_review": {"passed": True, "risk_level": "low"},
        "evidence_verification": {"passed": True, "unsupported_claims": []},
    }
    state = AgentState("Question", phase4_answer_status="answered", **(defaults | changes))
    assert ConsensusEngine().run(state).output["decision"] == decision


def _agents(critic_passed=True):
    return {
        "query_analyzer": StaticAgent("query_analyzer", "query_intent", {"intent": "procedure"}),
        "response_planner": StaticAgent("response_planner", "response_plan", {"format": "checklist"}),
        "prompt_composer": StaticAgent("prompt_composer", "composed_prompt", "prompt"),
        "draft_generator": StaticAgent("draft_generator", "", {"answer": "Claim [1]."}),
        "critic_agent": StaticAgent("critic_agent", "critic_review", {"passed": critic_passed, "severity": "low" if critic_passed else "medium", "issues": []}),
        "compliance_agent": StaticAgent("compliance_agent", "compliance_review", {"passed": True}),
        "risk_agent": StaticAgent("risk_agent", "risk_review", {"passed": True, "risk_level": "low", "risks": []}),
        "evidence_verifier": StaticAgent("evidence_verifier", "evidence_verification", {"passed": True, "verification_rate": 1, "unsupported_claims": [], "citation_mismatches": [], "verified_claims": ["Claim"]}),
        "consensus_engine": ConsensusEngine(),
    }


def test_disabled_pipeline_returns_phase4_unchanged():
    phase4 = Phase4()
    expected = phase4.answer("Question")
    actual = Phase5Pipeline(
        phase4_pipeline=phase4, config={"phase5": {"enabled": False}}
    ).answer("Question")
    assert actual == expected


def test_pipeline_preserves_status_and_limits_revision():
    pipeline = Phase5Pipeline(
        phase4_pipeline=Phase4(),
        config={"phase5": {"enabled": True, "max_revision_loops": 99}},
        agents=_agents(critic_passed=False),
    )
    result = pipeline.answer("Question")
    revisions = [x["revision"] for x in result["phase5_trace"]["events"]]
    assert result["revision_used"] is True
    assert max(revisions) == 1

    unsupported = Phase5Pipeline(
        phase4_pipeline=Phase4("unsupported_query"),
        config={"phase5": {"enabled": True}},
        agents=_agents(),
    ).answer("Live weather?")
    assert unsupported["final_status"] == "unsupported_query"


def test_html_is_self_contained_and_has_decision_visuals():
    record = {
        "question": "Can we proceed?",
        "answer": "# Decision\nProceed [1].",
        "selected_evidence": [{"source": "a.pdf", "modality": "text", "score": 0.9}],
        "critic_review": {"passed": True, "severity": "low", "issues": []},
        "compliance_review": {"passed": True},
        "risk_review": {"passed": True, "risk_level": "low", "risks": []},
        "evidence_verification": {
            "passed": True,
            "verification_rate": 1,
            "verified_claims": ["Proceed"],
            "unsupported_claims": [],
            "citation_mismatches": [],
        },
        "consensus_decision": {"decision": "accept", "final_status": "answered"},
    }
    page = render_phase5_html([record])
    assert "Ready for decision" in page
    assert "Decision Intelligence Dashboard" in page
    assert "localStorage" in page
    assert "cdn" not in page.casefold()
    assert decision_recommendation(record)[0] == "Ready for decision"


def test_verifier_skips_vision_for_text_only_evidence():
    state = AgentState(
        "Question",
        draft_answer="A grounded claim [1].",
        selected_evidence=[{"text": "A grounded claim.", "chunk_id": "1"}],
    )
    output = EvidenceVerifier().run(state).output
    assert output["vision_verification"]["status"] == "not_applicable"
    assert output["verification_rate"] == 1


def test_runner_keeps_legacy_columns_and_appends_phase5(tmp_path):
    pipeline = Phase5Pipeline(
        phase4_pipeline=Phase4(),
        config={"phase5": {"enabled": True}},
        agents=_agents(),
    )
    paths = Phase5Runner(pipeline, tmp_path).run(["Question"])
    header = paths["csv"].read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert header[: len(CSV_COLUMNS)] == CSV_COLUMNS
    assert all(column in header for column in PHASE5_CSV_COLUMNS)
    assert paths["html"].read_text(encoding="utf-8").startswith("<!doctype html>")
