"""FastAPI application for the fully local Phase 5 command center."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Any

from .event_bus import EventBus
from .schemas import LiveEvent
from .telemetry import TelemetryCollector

AGENTS = (
    "query_analyzer",
    "response_planner",
    "phase4_retrieval",
    "evidence_selection",
    "prompt_composer",
    "draft_generator",
    "critic_agent",
    "compliance_agent",
    "risk_agent",
    "evidence_verifier",
    "consensus_engine",
    "finalizer",
)
STATIC_DIR = Path(__file__).with_name("static")


class CommandCenterState:
    """Reduce the event stream into the latest dashboard snapshot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started_monotonic: float | None = None
        self.value: dict[str, Any] = self._empty()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "run_id": "",
            "question": "",
            "run_status": "idle",
            "current_stage": "Waiting for a Phase 5 run",
            "answer_status": "",
            "elapsed_seconds": 0.0,
            "progress": 0.0,
            "current_agent": "",
            "revision": {"used": False, "loop": 0, "status": "not_started"},
            "agents": {
                name: {
                    "status": "pending",
                    "latency_ms": None,
                    "model": "",
                    "fallback_used": False,
                    "warnings": [],
                    "errors": [],
                    "summary": {},
                }
                for name in AGENTS
            },
            "metrics": {
                "selected_evidence_count": 0,
                "evidence_sufficiency_score": 0.0,
                "verification_rate": 0.0,
                "unsupported_claim_count": 0,
                "citation_mismatch_count": 0,
                "critic_issue_count": 0,
                "risk_level": "unknown",
                "consensus_decision": "",
                "revision_used": False,
                "final_status": "",
                "source_diversity": 0,
                "modality_mix": {},
            },
            "telemetry": {"available": False, "gpu": {"available": False}},
            "draft_answer": "",
            "final_answer": "",
            "citations": [],
            "events": [],
        }

    def apply(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("event_type") or "")
        data = event.get("data")
        data = dict(data) if isinstance(data, Mapping) else {}
        with self._lock:
            state = self.value
            if event_type == "run_started":
                self.value = self._empty()
                state = self.value
                self._started_monotonic = monotonic()
                state.update(
                    {
                        "run_id": event.get("run_id", ""),
                        "question": data.get("question", ""),
                        "run_status": "running",
                        "current_stage": "Starting",
                    }
                )
            if event.get("progress") is not None:
                state["progress"] = float(event["progress"])
            if event.get("stage"):
                state["current_stage"] = str(event["stage"])
            if event_type == "agent_started":
                agent = str(event.get("agent") or "")
                state["current_agent"] = agent
                if agent in state["agents"]:
                    state["agents"][agent].update(
                        {"status": "running", "model": event.get("model", "")}
                    )
            elif event_type in {"agent_completed", "agent_failed"}:
                agent = str(event.get("agent") or "")
                if agent in state["agents"]:
                    state["agents"][agent].update(
                        {
                            "status": (
                                "completed"
                                if event_type == "agent_completed"
                                else "failed"
                            ),
                            "latency_ms": data.get("latency_ms"),
                            "model": event.get("model")
                            or data.get("model_used", ""),
                            "fallback_used": bool(data.get("fallback_used")),
                            "warnings": data.get("warnings") or [],
                            "errors": data.get("errors") or [],
                            "summary": data.get("summary") or {},
                        }
                    )
                state["current_agent"] = ""
                if event_type == "agent_completed" and event.get("model"):
                    latency = data.get("latency_ms")
                    tokens = data.get("tokens_generated")
                    throughput = None
                    if tokens is not None and latency and float(latency) > 0:
                        throughput = round(
                            float(tokens) / (float(latency) / 1000), 3
                        )
                    state["telemetry"].update(
                        {
                            "current_model": event.get("model"),
                            "model_latency_ms": latency,
                            "tokens_generated": tokens,
                            "tokens_per_second": throughput,
                        }
                    )
            elif event_type == "phase4_started":
                state["agents"]["phase4_retrieval"]["status"] = "running"
            elif event_type == "phase4_completed":
                state["agents"]["phase4_retrieval"].update(
                    {
                        "status": "completed",
                        "latency_ms": data.get("latency_ms"),
                        "summary": {"answer_status": data.get("answer_status")},
                    }
                )
                state["answer_status"] = data.get("answer_status", "")
            elif event_type == "evidence_selected":
                state["agents"]["evidence_selection"].update(
                    {"status": "completed", "summary": data}
                )
                state["metrics"].update(data)
            elif event_type == "draft_generated":
                state["draft_answer"] = data.get("answer", "")
            elif event_type == "critic_completed":
                state["metrics"]["critic_issue_count"] = int(
                    data.get("issue_count") or 0
                )
            elif event_type == "compliance_completed":
                state["metrics"]["compliance_passed"] = bool(data.get("passed"))
            elif event_type == "risk_completed":
                state["metrics"]["risk_level"] = data.get("risk_level", "unknown")
            elif event_type == "verification_completed":
                state["metrics"].update(
                    {
                        "verification_rate": float(
                            data.get("verification_rate") or 0
                        ),
                        "unsupported_claim_count": int(
                            data.get("unsupported_claim_count") or 0
                        ),
                        "citation_mismatch_count": int(
                            data.get("citation_mismatch_count") or 0
                        ),
                    }
                )
            elif event_type == "consensus_decided":
                state["metrics"]["consensus_decision"] = data.get("decision", "")
            elif event_type == "revision_started":
                state["revision"] = {"used": True, "loop": 1, "status": "running"}
                state["metrics"]["revision_used"] = True
            elif event_type == "revision_completed":
                state["revision"]["status"] = "completed"
            elif event_type == "telemetry_update":
                old_telemetry = dict(state.get("telemetry") or {})
                previous = {**old_telemetry, **data}
                for key in (
                    "current_model",
                    "model_latency_ms",
                    "tokens_generated",
                    "tokens_per_second",
                ):
                    if data.get(key) in (None, "") and old_telemetry.get(key):
                        previous[key] = old_telemetry[key]
                state["telemetry"] = previous
            elif event_type == "run_completed":
                state.update(
                    {
                        "run_status": "completed",
                        "current_stage": "Completed",
                        "answer_status": data.get("final_status", ""),
                        "final_answer": data.get("answer", ""),
                        "citations": data.get("citations") or [],
                    }
                )
                state["metrics"]["final_status"] = data.get("final_status", "")
            elif event_type == "run_failed":
                state.update(
                    {
                        "run_status": "failed",
                        "current_stage": "Failed",
                        "answer_status": "failed",
                    }
                )
                for agent in state["agents"].values():
                    if agent["status"] == "pending":
                        agent["status"] = "skipped"
            if self._started_monotonic is not None:
                state["elapsed_seconds"] = round(
                    monotonic() - self._started_monotonic, 2
                )
            state["events"].append(dict(event))
            state["events"] = state["events"][-250:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = deepcopy(self.value)
            if (
                self._started_monotonic is not None
                and result["run_status"] == "running"
            ):
                result["elapsed_seconds"] = round(
                    monotonic() - self._started_monotonic, 2
                )
            return result


def _require_fastapi() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "The live command center requires FastAPI and Uvicorn. "
            "Install the project requirements before using --live."
        ) from exc
    globals()["Request"] = Request
    return FastAPI, Request, FileResponse, JSONResponse, StreamingResponse


def create_app(
    *,
    event_bus: EventBus | None = None,
    telemetry: TelemetryCollector | None = None,
    telemetry_interval: float = 1.0,
) -> Any:
    """Create the local server without affecting normal Phase 5 imports."""

    if telemetry_interval <= 0:
        raise ValueError("telemetry_interval must be greater than zero.")
    FastAPI, Request, FileResponse, JSONResponse, StreamingResponse = (
        _require_fastapi()
    )
    bus = event_bus or EventBus()
    collector = telemetry or TelemetryCollector()
    state = CommandCenterState()
    for existing in bus.history():
        state.apply(existing)
    state_token = bus.subscribe(state.apply)

    def update_model_stats(event: Mapping[str, Any]) -> None:
        if (
            event.get("event_type") != "agent_completed"
            or not event.get("model")
            or not hasattr(collector, "update_model")
        ):
            return
        data = event.get("data")
        data = data if isinstance(data, Mapping) else {}
        collector.update_model(
            name=str(event.get("model") or ""),
            latency_ms=data.get("latency_ms"),
            tokens_generated=data.get("tokens_generated"),
        )

    model_token = bus.subscribe(update_model_stats)

    @asynccontextmanager
    async def lifespan(_app: Any):
        async def publish_telemetry() -> None:
            while True:
                bus.publish(
                    LiveEvent(
                        event_type="telemetry_update",
                        run_id=state.snapshot().get("run_id", ""),
                        data=collector.collect(),
                    )
                )
                await asyncio.sleep(telemetry_interval)

        task = asyncio.create_task(publish_telemetry())
        try:
            yield
        finally:
            task.cancel()
            bus.unsubscribe(state_token)
            bus.unsubscribe(model_token)
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="CIAL Knowledge OS Phase 5 Command Center",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.event_bus = bus
    app.state.command_center = state
    app.state.telemetry = collector

    @app.get("/")
    async def index() -> Any:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/app.js")
    async def app_js() -> Any:
        return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")

    @app.get("/styles.css")
    async def styles() -> Any:
        return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")

    @app.get("/api/state")
    async def api_state() -> Any:
        return JSONResponse(state.snapshot())

    @app.get("/api/health")
    async def health() -> Any:
        return {"status": "ok", "offline": True}

    @app.get("/events")
    async def events(request: Request, after: int = 0, once: bool = False) -> Any:
        async def generate():
            snapshot = json.dumps(state.snapshot(), ensure_ascii=False, default=str)
            yield f"event: snapshot\ndata: {snapshot}\n\n"
            if once:
                for event in bus.history(after_sequence=after):
                    payload = json.dumps(
                        event, ensure_ascii=False, default=str
                    )
                    yield (
                        f"event: {event['event_type']}\n"
                        f"data: {payload}\n\n"
                    )
                return
            async for event in bus.stream(after_sequence=after):
                if await request.is_disconnected():
                    break
                payload = json.dumps(event, ensure_ascii=False, default=str)
                yield f"event: {event['event_type']}\ndata: {payload}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app


def start_in_thread(
    *,
    event_bus: EventBus,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[Any, threading.Thread]:
    """Start a daemonized local Uvicorn server for programmatic batch runs."""

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Live mode requires Uvicorn. Install the project requirements."
        ) from exc
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(event_bus=event_bus),
            host=host,
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(
        target=server.run,
        name="cial-phase5-command-center",
        daemon=True,
    )
    thread.start()
    return server, thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local Phase 5 Agentic Command Center."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "FastAPI/Uvicorn are unavailable. Install project requirements."
        ) from exc
    url = f"http://{args.host}:{args.port}"
    print(f"Phase 5 Agentic Command Center: {url}")
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
