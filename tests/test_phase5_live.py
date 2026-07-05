from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

from cial_knowledge_os.agents import AgentResult, AgentState
from cial_knowledge_os.agents.base import Agent
from cial_knowledge_os.live.command_center import CommandCenterState, create_app
from cial_knowledge_os.live.event_bus import EventBus
from cial_knowledge_os.live.schemas import LiveEvent
from cial_knowledge_os.live.telemetry import TelemetryCollector
from cial_knowledge_os.orchestration import Phase5Pipeline, Phase5Runner


def make_test_client(app):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
        )
        from starlette.testclient import TestClient

    return TestClient(app)


class StaticAgent(Agent):
    def __init__(self, name: str, field: str, output):
        self.name, self.field, self.output = name, field, output

    def run(self, state: AgentState) -> AgentResult:
        updated = state.evolve(**{self.field: self.output}) if self.field else state
        if self.name == "draft_generator":
            updated = state.evolve(draft_answer=self.output["answer"])
        return AgentResult(self.name, True, self.output, updated, latency_ms=2)


class Phase4Fixture:
    metrics = {}
    config = SimpleNamespace(
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
            "answer_status": "answered",
            "selected_evidence": [
                {
                    "text": "Grounded evidence.",
                    "source": "manual.pdf",
                    "chunk_id": "c1",
                    "reranker_score": 0.8,
                }
            ],
            "citations": [],
            "evidence_quality": {},
        }


def agents():
    return {
        "query_analyzer": StaticAgent(
            "query_analyzer", "query_intent", {"intent": "procedure"}
        ),
        "response_planner": StaticAgent(
            "response_planner", "response_plan", {"format": "checklist"}
        ),
        "prompt_composer": StaticAgent(
            "prompt_composer", "composed_prompt", "prompt"
        ),
        "draft_generator": StaticAgent(
            "draft_generator", "", {"answer": "Claim [1]."}
        ),
        "critic_agent": StaticAgent(
            "critic_agent",
            "critic_review",
            {"passed": True, "severity": "low", "issues": []},
        ),
        "compliance_agent": StaticAgent(
            "compliance_agent",
            "compliance_review",
            {"passed": True, "grounding_score": 1},
        ),
        "risk_agent": StaticAgent(
            "risk_agent",
            "risk_review",
            {"passed": True, "risk_level": "low", "risks": []},
        ),
        "evidence_verifier": StaticAgent(
            "evidence_verifier",
            "evidence_verification",
            {
                "passed": True,
                "verification_rate": 1,
                "unsupported_claims": [],
                "citation_mismatches": [],
            },
        ),
        "consensus_engine": StaticAgent(
            "consensus_engine",
            "consensus_decision",
            {
                "decision": "accept",
                "final_status": "answered",
                "reason": "passed",
            },
        ),
    }


def test_event_bus_publish_subscribe_and_async_stream():
    bus = EventBus()
    received = []
    token = bus.subscribe(received.append)
    event = bus.publish(
        LiveEvent(event_type="run_started", run_id="run-1")
    )
    bus.unsubscribe(token)
    assert received[0]["sequence"] == event["sequence"]
    assert bus.history()[0]["run_id"] == "run-1"

    async def read_one():
        stream = bus.stream()
        value = await anext(stream)
        await stream.aclose()
        return value

    assert asyncio.run(read_one())["event_type"] == "run_started"


def test_telemetry_gracefully_handles_missing_gpu():
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    stats = TelemetryCollector(command_runner=missing).collect()
    assert stats["available"] is True
    assert stats["gpu"] == {"available": False, "devices": []}
    assert stats["process_memory_bytes"] > 0


def test_telemetry_parses_mocked_nvidia_smi():
    def mocked(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout="42, 2048, 8192, NVIDIA Test GPU\n",
            stderr="",
        )

    gpu = TelemetryCollector(command_runner=mocked).collect()["gpu"]
    assert gpu["available"] is True
    assert gpu["usage_percent"] == 42
    assert gpu["memory_used_mb"] == 2048
    assert gpu["devices"][0]["name"] == "NVIDIA Test GPU"


def test_dashboard_server_routes_and_sse_emit_events():
    bus = EventBus()
    bus.publish(
        LiveEvent(
            event_type="run_started",
            run_id="run-live",
            data={"question": "Question"},
        )
    )
    app = create_app(
        event_bus=bus,
        telemetry=TelemetryCollector(
            command_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
                FileNotFoundError()
            )
        ),
        telemetry_interval=60,
    )
    with make_test_client(app) as client:
        assert client.get("/api/health").json() == {
            "status": "ok",
            "offline": True,
        }
        assert client.get("/").status_code == 200
        response = client.get("/events?once=true")
        assert response.status_code == 200
        assert "event: snapshot" in response.text
        assert "event: run_started" in response.text
        assert "run-live" in response.text


def test_pipeline_emits_expected_live_events():
    bus = EventBus()
    pipeline = Phase5Pipeline(
        phase4_pipeline=Phase4Fixture(),
        config={"phase5": {"enabled": True}},
        agents=agents(),
        event_bus=bus,
    )
    result = pipeline.answer("Question", run_id="run-events")
    types = [item["event_type"] for item in bus.history()]
    assert result["final_status"] == "answered"
    for expected in (
        "run_started",
        "phase4_started",
        "phase4_completed",
        "evidence_selected",
        "stage_started",
        "stage_completed",
        "agent_started",
        "agent_completed",
        "draft_generated",
        "critic_completed",
        "compliance_completed",
        "risk_completed",
        "verification_completed",
        "consensus_decided",
        "run_completed",
    ):
        assert expected in types


def test_failed_run_marks_unreached_agents_skipped():
    state = CommandCenterState()
    state.apply(
        LiveEvent(
            event_type="run_started",
            run_id="failed-run",
            data={"question": "Question"},
        ).to_dict()
    )
    state.apply(
        LiveEvent(
            event_type="agent_failed",
            run_id="failed-run",
            agent="query_analyzer",
            data={"errors": ["failure"]},
        ).to_dict()
    )
    state.apply(
        LiveEvent(
            event_type="run_failed",
            run_id="failed-run",
            data={"error": "failure"},
        ).to_dict()
    )
    snapshot = state.snapshot()
    assert snapshot["agents"]["query_analyzer"]["status"] == "failed"
    assert snapshot["agents"]["response_planner"]["status"] == "skipped"


def test_normal_runner_does_not_start_live_server(tmp_path, monkeypatch):
    import cial_knowledge_os.live.command_center as command_center

    monkeypatch.setattr(
        command_center,
        "start_in_thread",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("live server should not start")
        ),
    )
    pipeline = Phase5Pipeline(
        phase4_pipeline=Phase4Fixture(),
        config={"phase5": {"enabled": True}},
        agents=agents(),
    )
    paths = Phase5Runner(pipeline, tmp_path).run(["Question"])
    assert paths["json"].exists()
    assert pipeline.event_bus is None


def test_dashboard_accepts_missing_telemetry_event():
    bus = EventBus()
    missing = SimpleNamespace(
        collect=lambda: {"available": False, "error": "unavailable"}
    )
    app = create_app(
        event_bus=bus, telemetry=missing, telemetry_interval=60
    )
    with make_test_client(app) as client:
        state = client.get("/api/state").json()
    assert state["telemetry"]["available"] is False
    assert state["telemetry"]["error"] == "unavailable"
