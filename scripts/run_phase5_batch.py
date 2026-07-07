"""Run Phase 5 agentic planning over the existing Phase 4 batch pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import webbrowser
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic, sleep
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


# =====================================================
# USER CONFIGURATION
# Edit these values for day-to-day Phase 5 runs.
# All model providers remain local.
# =====================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "manual_qa"
    / "CIAL_Enterprise_Stress_Test_500_Questions.txt"
)
RUN_MODE = "manual_qa"
MAX_ANSWER_WORDS = 1200
ADAPTIVE_ANSWER_SECTIONS = True
GENERATION_RETRIES = 2
RETRY_COOLDOWN_SECONDS = 20
RERANKER_DEVICE = "auto"
RERANKER_BATCH_SIZE = 16
LOCAL_FILES_ONLY = False
FORCE_REBUILD_INDEX = False
RESUME_RUN_FOLDER = None
QDRANT_MODE = "server"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None
QDRANT_BATCH_SIZE = 32
QDRANT_UPSERT_WAIT = True

PHASE5_ENABLED = True
PHASE5_MAX_REVISION_LOOPS = 1
PHASE5_OUTPUT_NAME = "05_Agentic_Response_Planning"
PHASE5_MODEL_PROFILE = "local_primary"
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8765

PHASE5_AGENT_NAMES = (
    "query_analyzer",
    "response_planner",
    "draft_generator",
    "critic_agent",
    "compliance_agent",
    "risk_agent",
    "evidence_verifier",
)


def _print(message: str = "") -> None:
    """Print progress immediately in terminals and VS Code."""

    print(message, flush=True)


def _enable_immediate_output() -> None:
    """Use line-buffered, write-through output where Python supports it."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except (OSError, ValueError):
                pass


def positive_integer(value: str) -> int:
    """Parse a strictly positive CLI integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return parsed


def non_negative_integer(value: str) -> int:
    """Parse a non-negative CLI integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def non_negative_number(value: str) -> float:
    """Parse a non-negative CLI number."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def resolve_path(
    path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve paths consistently regardless of the launch directory."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def load_questions(
    path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    """Load a TXT question list or a CSV containing a question column."""

    source = resolve_path(path, project_root=project_root)
    guidance = (
        f"Expected path: {source}\n"
        "Use a TXT file with one question per line, or a CSV file with a "
        "'question' column. Edit QUESTIONS_FILE or pass --questions-file."
    )
    if not source.is_file():
        raise FileNotFoundError(f"Questions file is missing.\n{guidance}")

    if source.suffix.casefold() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "question" not in reader.fieldnames:
                raise ValueError(
                    "Questions CSV must contain a 'question' column.\n"
                    f"{guidance}"
                )
            questions = [
                str(row.get("question") or "").strip()
                for row in reader
                if str(row.get("question") or "").strip()
            ]
    elif source.suffix.casefold() == ".txt":
        questions = [
            line.strip()
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError(
            "Questions file must use a .txt or .csv extension.\n"
            f"{guidance}"
        )

    if not questions:
        raise ValueError(f"Questions file is empty.\n{guidance}")
    return questions


def build_parser() -> argparse.ArgumentParser:
    """Provide optional one-off overrides for user configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 5 over the stable Phase 4 pipeline using local models. "
            "Every argument is optional."
        )
    )
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument(
        "--mode",
        choices=("smoke", "manual_qa", "benchmark"),
    )
    parser.add_argument(
        "--limit",
        "--max-questions",
        dest="limit",
        type=positive_integer,
        help="Run only the first N questions (useful for smoke tests).",
    )
    parser.add_argument("--max-answer-words", type=positive_integer)
    parser.add_argument(
        "--generation-retries",
        type=non_negative_integer,
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=non_negative_number,
    )
    parser.add_argument(
        "--reranker-device",
        choices=("cpu", "cuda", "auto"),
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=positive_integer,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--phase5",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the optional Phase 5 agentic layer.",
    )
    parser.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the existing local Phase 5 command center.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=positive_integer,
        default=DASHBOARD_PORT,
    )
    parser.add_argument(
        "--browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the local dashboard in the default browser.",
    )
    parser.add_argument(
        "--resume-run-folder",
        "--resume",
        dest="resume_run_folder",
        type=Path,
    )
    parser.add_argument(
        "--force-rebuild-index",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore the document manifest and rebuild the vector index.",
    )
    return parser


def _value(override: Any, configured: Any) -> Any:
    return configured if override is None else override


def build_phase4_config(args: argparse.Namespace) -> Any:
    """Build the unchanged Phase 4 engine configuration."""

    from cial_knowledge_os.config import Phase4Config

    return Phase4Config(
        project_root=PROJECT_ROOT,
        phase_output_name=PHASE5_OUTPUT_NAME,
        phase4_run_mode=_value(args.mode, RUN_MODE),
        max_answer_words=_value(args.max_answer_words, MAX_ANSWER_WORDS),
        adaptive_answer_sections=ADAPTIVE_ANSWER_SECTIONS,
        generation_retries=_value(
            args.generation_retries,
            GENERATION_RETRIES,
        ),
        retry_cooldown_seconds=_value(
            args.retry_cooldown_seconds,
            RETRY_COOLDOWN_SECONDS,
        ),
        reranker_device=_value(args.reranker_device, RERANKER_DEVICE),
        reranker_batch_size=_value(
            args.reranker_batch_size,
            RERANKER_BATCH_SIZE,
        ),
        reranker_local_files_only=_value(
            args.local_files_only,
            LOCAL_FILES_ONLY,
        ),
        force_rebuild_index=_value(
            args.force_rebuild_index,
            FORCE_REBUILD_INDEX,
        ),
        qdrant_mode=QDRANT_MODE,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_batch_size=QDRANT_BATCH_SIZE,
        qdrant_upsert_wait=QDRANT_UPSERT_WAIT,
        allow_large_run=True,
    )


def build_phase5_config(
    args: argparse.Namespace,
    phase4_config: Any,
) -> dict[str, Any]:
    """Build local-only Phase 5 routing and agent configuration."""

    enabled = bool(_value(args.phase5, PHASE5_ENABLED))
    model_name = str(phase4_config.ollama_model_name)
    return {
        "phase5": {
            "enabled": enabled,
            "max_revision_loops": PHASE5_MAX_REVISION_LOOPS,
            "model_profiles": {
                PHASE5_MODEL_PROFILE: {
                    "provider": "ollama",
                    "model": model_name,
                    "capabilities": ["text", "structured_json"],
                    "timeout_seconds": 120,
                    "retries": 0,
                    "temperature": 0,
                }
            },
            "agents": {
                name: {"model_profile": PHASE5_MODEL_PROFILE}
                for name in PHASE5_AGENT_NAMES
            },
        }
    }


def select_inputs(
    args: argparse.Namespace,
    config: Any,
) -> tuple[list[str], Any | None, str]:
    """Load configured questions and preserve benchmark CSV behavior."""

    source = resolve_path(
        _value(args.questions_file, QUESTIONS_FILE),
        project_root=config.project_root,
    )
    benchmark = None
    questions = load_questions(source, project_root=config.project_root)

    if config.phase4_run_mode == "benchmark" and source.suffix.casefold() == ".csv":
        from cial_knowledge_os.benchmark_loader import load_benchmark

        benchmark = load_benchmark(source)
        questions = [item.question for item in benchmark.questions]

    if args.limit is not None:
        questions = questions[: args.limit]
        if benchmark is not None:
            from cial_knowledge_os.benchmark_loader import Benchmark

            benchmark = Benchmark(
                questions=benchmark.questions[: args.limit],
                metadata=dict(benchmark.metadata),
                source_path=benchmark.source_path,
            )
    if not questions:
        raise ValueError(
            f"No questions remain after applying --limit.\nExpected path: {source}"
        )
    return questions, benchmark, str(source)


def _resume_path(args: argparse.Namespace) -> Path | None:
    value = _value(args.resume_run_folder, RESUME_RUN_FOLDER)
    return resolve_path(value) if value is not None else None


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    value = max(0, int(seconds))
    return f"{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}"


class Phase5ProgressReporter:
    """Print durable batch progress from the shared execution event stream."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.started_at = monotonic()

    def __call__(self, event: Any) -> None:
        if event.event_type == "question_started":
            _print(
                f"Question {event.question_index}/{event.question_total}: "
                f"{event.question_preview}"
            )
            return
        if event.event_type not in {"question_completed", "question_failed"}:
            return
        snapshot = self.manager.progress.snapshot()
        elapsed = monotonic() - self.started_at
        statuses = ", ".join(
            f"{name}={count}"
            for name, count in sorted(snapshot["status_counts"].items())
        )
        _print(
            f"Progress {snapshot['completed']}/{snapshot['total']} "
            f"({snapshot['percent']:.1f}%) | "
            f"elapsed {_duration(elapsed)} | "
            f"ETA {_duration(snapshot['eta'])} | "
            f"status counts: {statuses or 'none'}"
        )


class LiveExecutionAdapter:
    """Bridge batch/index events into the existing Phase 5 live event bus."""

    def __init__(
        self,
        *,
        live_bus: Any,
        execution_manager: Any,
        routing_summary: str,
        phase5_enabled: bool,
    ) -> None:
        self.live_bus = live_bus
        self.execution_manager = execution_manager
        self.routing_summary = routing_summary
        self.phase5_enabled = phase5_enabled
        self.batch: dict[str, Any] = {}
        self.qdrant: dict[str, Any] = {}

    def _publish_context(self, run_id: str = "") -> None:
        from cial_knowledge_os.live.schemas import LiveEvent

        self.live_bus.publish(
            LiveEvent(
                event_type="telemetry_update",
                run_id=run_id or self.execution_manager.run_id,
                data={
                    "batch": dict(self.batch),
                    "model_routing_summary": self.routing_summary,
                    "qdrant": dict(self.qdrant),
                },
            )
        )

    def handle_execution(self, event: Any) -> None:
        from cial_knowledge_os.live.schemas import LiveEvent

        if event.event_type == "qdrant_health_checked":
            self.qdrant = dict(event.payload)
            self._publish_context(event.run_id)
            return
        if event.event_type not in {
            "question_started",
            "question_completed",
            "question_failed",
            "run_started",
            "run_resumed",
            "batch_completed",
        }:
            return
        snapshot = self.execution_manager.progress.snapshot()
        self.batch = {
            "question_index": event.question_index
            or snapshot["current_question_index"],
            "question_total": event.question_total or snapshot["total"],
            "current_question": event.question_preview
            or snapshot["current_question"],
            "completed": snapshot["completed"],
            "percent": snapshot["percent"],
            "elapsed_seconds": snapshot["elapsed"],
            "eta_seconds": snapshot["eta"],
            "answer_status_counts": snapshot["status_counts"],
        }
        if not self.phase5_enabled and event.event_type == "question_started":
            self.live_bus.publish(
                LiveEvent(
                    event_type="run_started",
                    run_id=event.run_id,
                    data={"question": self.batch["current_question"]},
                )
            )
        self._publish_context(event.run_id)
        if (
            not self.phase5_enabled
            and event.event_type in {"question_completed", "question_failed"}
        ):
            self.live_bus.publish(
                LiveEvent(
                    event_type=(
                        "run_completed"
                        if event.event_type == "question_completed"
                        else "run_failed"
                    ),
                    run_id=event.run_id,
                    data={
                        "final_status": event.payload.get("answer_status"),
                    },
                )
            )

    def handle_live(self, event: Mapping[str, Any]) -> None:
        if event.get("event_type") == "run_started":
            self._publish_context(str(event.get("run_id") or ""))


def _available_dashboard_port(preferred: int) -> int:
    """Use the requested local port, falling back to an ephemeral free port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind((DASHBOARD_HOST, preferred))
        except OSError:
            candidate.bind((DASHBOARD_HOST, 0))
        return int(candidate.getsockname()[1])


def start_dashboard(
    args: argparse.Namespace,
    *,
    live_bus: Any,
) -> tuple[Any | None, Any | None, str | None]:
    """Start the reusable command center without making batch success depend on it."""

    if not args.dashboard:
        _print("Dashboard disabled")
        return None, None, None
    server: Any | None = None
    thread: Any | None = None
    try:
        from cial_knowledge_os.live.command_center import start_in_thread

        requested_port = int(args.dashboard_port)
        port = _available_dashboard_port(requested_port)
        if port != requested_port:
            _print(
                f"Dashboard port {requested_port} is occupied; using {port}."
            )
        server, thread = start_in_thread(
            event_bus=live_bus,
            host=DASHBOARD_HOST,
            port=port,
        )
        deadline = monotonic() + 10
        while thread.is_alive() and not bool(getattr(server, "started", False)):
            if monotonic() >= deadline:
                raise RuntimeError("dashboard server did not become ready")
            sleep(0.05)
        if not thread.is_alive():
            raise RuntimeError("dashboard server stopped during startup")
        url = f"http://{DASHBOARD_HOST}:{port}"
        _print(f"Dashboard URL: {url}")
        if args.browser:
            try:
                opened = bool(webbrowser.open(url, new=2))
            except Exception as exc:
                _print(f"Browser open failed ({exc}). Open manually: {url}")
            else:
                if not opened:
                    _print(f"Browser did not open. Open manually: {url}")
        else:
            _print("Browser auto-open disabled")
        return server, thread, url
    except Exception as exc:
        stop_dashboard(server, thread)
        _print(f"WARNING: dashboard unavailable ({exc}); continuing batch run.")
        return None, None, None


def stop_dashboard(server: Any | None, thread: Any | None) -> None:
    """Request a clean dashboard shutdown without blocking pipeline cleanup."""

    if server is None:
        return
    server.should_exit = True
    if thread is not None and thread.is_alive():
        thread.join(timeout=5)


def model_routing_summary(config: Mapping[str, Any]) -> str:
    """Return a compact, non-secret local model routing summary."""

    phase5 = config.get("phase5")
    phase5 = phase5 if isinstance(phase5, Mapping) else {}
    profiles = phase5.get("model_profiles")
    profiles = profiles if isinstance(profiles, Mapping) else {}
    agents = phase5.get("agents")
    agents = agents if isinstance(agents, Mapping) else {}
    models = sorted(
        {
            str(value.get("model") or "")
            for value in profiles.values()
            if isinstance(value, Mapping) and value.get("model")
        }
    )
    return (
        f"local models={', '.join(models) or 'none'}; "
        f"agent assignments={len(agents)}"
    )


def _latest_checkpoint_responses(run_root: Path) -> list[dict[str, Any]]:
    """Load the latest full response for every checkpoint identity."""

    path = run_root / "partial_retrieval.jsonl"
    if not path.is_file():
        return []
    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                break
            raise
        response = value.get("response")
        if not isinstance(response, Mapping):
            continue
        latest[str(value.get("key") or line_number)] = (
            int(value.get("index") or line_number),
            dict(response),
        )
    return [
        response
        for _, response in sorted(latest.values(), key=lambda item: item[0])
    ]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def mark_run_interrupted(run_root: Path) -> None:
    """Mark the durable checkpoint interrupted while retaining resumability."""

    checkpoint_path = run_root / "checkpoint.json"
    if not checkpoint_path.is_file():
        return
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["status"] = "interrupted"
    _write_json(checkpoint_path, checkpoint)


def finalize_phase5_artifacts(
    run_root: Path,
    *,
    phase5_config: Mapping[str, Any],
) -> None:
    """Add Phase 5 records to the established Phase 4 artifact bundle."""

    responses = _latest_checkpoint_responses(run_root)
    _write_json(run_root / "results.json", responses)

    enabled = bool(
        (phase5_config.get("phase5") or {}).get("enabled", False)
    )
    if enabled and responses:
        from cial_knowledge_os.reporting.phase5_html import write_phase5_html

        write_phase5_html(run_root / "report.html", responses)

    retrieval_path = run_root / "retrieval.json"
    if retrieval_path.is_file():
        traces = json.loads(retrieval_path.read_text(encoding="utf-8"))
        for trace, response in zip(traces, responses):
            if not isinstance(trace, dict):
                continue
            trace["phase5_enabled"] = bool(response.get("phase5_enabled"))
            for key in (
                "query_intent",
                "response_plan",
                "critic_review",
                "compliance_review",
                "risk_review",
                "evidence_verification",
                "consensus_decision",
                "revision_used",
                "final_status",
                "agent_latency_total_ms",
                "model_map",
                "phase5_trace",
            ):
                if key in response:
                    trace[key] = response[key]
        _write_json(retrieval_path, traces)

    status_counts = Counter(
        str(response.get("answer_status") or "unknown")
        .casefold()
        .replace(" ", "_")
        for response in responses
    )
    for name in ("summary.json", "metrics.json"):
        path = run_root / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "phase": "Phase 5" if enabled else "Phase 4",
                "phase5_enabled": enabled,
                "phase5_answer_status_counts": dict(status_counts),
                "phase5_model_routing": model_routing_summary(phase5_config),
            }
        )
        _write_json(path, payload)

    config_path = run_root / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["phase5"] = dict(phase5_config.get("phase5") or {})
        _write_json(config_path, payload)


def _prepare_run_manager(config: Any, resume_run: Path | None) -> Any:
    from cial_knowledge_os.run_manager import RunManager

    manager = (
        RunManager.from_existing(config, resume_run)
        if resume_run is not None
        else RunManager.from_config(config)
    )
    manager.create()
    return manager


def execute(
    args: argparse.Namespace,
    *,
    phase4_config: Any,
    phase5_config: dict[str, Any],
    questions: list[str],
    benchmark: Any | None,
    source_label: str,
) -> tuple[Any | None, Path, bool]:
    """Initialize Phase 4, wrap it with Phase 5, and run the shared batch."""

    from cial_knowledge_os.agents.model_router import ModelRouter
    from cial_knowledge_os.execution import ExecutionManager
    from cial_knowledge_os.live.event_bus import EventBus as LiveEventBus
    from cial_knowledge_os.orchestration.phase5_pipeline import Phase5Pipeline
    from cial_knowledge_os.phase4_pipeline import Phase4RAGPipeline
    from cial_knowledge_os.phase4_runner import Phase4Runner

    resume_run = _resume_path(args)
    run_manager = _prepare_run_manager(phase4_config, resume_run)
    run_root = run_manager.require_paths().root
    execution_manager = ExecutionManager.from_config(
        phase4_config,
        phase="Phase 5",
        run_mode=phase4_config.phase4_run_mode,
    )
    execution_manager.event_bus.subscribe(
        Phase5ProgressReporter(execution_manager)
    )
    live_bus = LiveEventBus()
    live_adapter = LiveExecutionAdapter(
        live_bus=live_bus,
        execution_manager=execution_manager,
        routing_summary=model_routing_summary(phase5_config),
        phase5_enabled=bool(phase5_config["phase5"]["enabled"]),
    )
    execution_manager.event_bus.subscribe(live_adapter.handle_execution)
    dashboard_server, dashboard_thread, _dashboard_url = start_dashboard(
        args,
        live_bus=live_bus,
    )
    live_bus.subscribe(live_adapter.handle_live)

    _print("CIAL Knowledge OS — Phase 5 Batch Runner")
    _print(f"Qdrant mode: {phase4_config.qdrant_mode}")
    _print(f"Questions file: {source_label}")
    _print(f"Output folder: {run_root}")
    _print(f"Model routing: {model_routing_summary(phase5_config)}")

    phase5_enabled = bool(phase5_config["phase5"]["enabled"])
    if not phase5_enabled:
        _print(
            "WARNING: phase5.enabled=false; output will be Phase 4-equivalent."
        )

    phase4_pipeline = Phase4RAGPipeline(phase4_config)
    phase4_pipeline.execution_manager = execution_manager
    phase5_pipeline: Any | None = None
    interrupted = False
    result: Any | None = None
    try:
        _print("Initializing Phase 4 retrieval pipeline")
        phase4_pipeline.load()
        phase4_pipeline.chunk()
        phase4_pipeline.embed()
        phase4_pipeline.index()
        _print("Indexing complete")

        config = phase5_config
        phase5_pipeline = Phase5Pipeline(
            phase4_pipeline=phase4_pipeline,
            config=config,
            model_router=ModelRouter(config),
            event_bus=live_bus,
            execution_manager=execution_manager,
        )
        phase5_pipeline.metrics = phase4_pipeline.metrics
        phase5_pipeline.indexing_summary = phase4_pipeline.indexing_summary

        _print("Starting QA")
        result = Phase4Runner(
            pipeline=phase5_pipeline,
            config=phase4_config,
            run_manager=run_manager,
            execution_manager=execution_manager,
        ).run(
            questions=questions,
            benchmark=benchmark,
            run_mode=phase4_config.phase4_run_mode,
            run_metadata={
                "run_label": "terminal_phase5_batch",
                "question_source": source_label,
                "large_run": (
                    phase4_config.phase4_run_mode == "manual_qa"
                    and len(questions)
                    > phase4_config.max_inline_manual_questions
                ),
            },
            resume_run=resume_run,
        )
    except KeyboardInterrupt:
        interrupted = True
        mark_run_interrupted(run_root)
        _print(
            "Interrupted. Checkpoint and partial outputs were preserved in "
            f"{run_root}"
        )
    finally:
        try:
            finalize_phase5_artifacts(
                run_root,
                phase5_config=phase5_config,
            )
        finally:
            try:
                if phase5_pipeline is not None:
                    close_phase5 = getattr(phase5_pipeline, "close", None)
                    if callable(close_phase5):
                        close_phase5()
                phase4_pipeline.close()
            finally:
                stop_dashboard(dashboard_server, dashboard_thread)
    return result, run_root, interrupted


def print_artifact_paths(run_root: Path) -> None:
    """Print the complete Phase 5 artifact bundle."""

    artifacts = (
        "results.csv",
        "results.xlsx",
        "results.json",
        "report.html",
        "config.json",
        "summary.json",
        "metrics.json",
        "retrieval.json",
        "logs.txt",
        "checkpoint.json",
        "partial_results.csv",
        "partial_results.jsonl",
        "partial_retrieval.jsonl",
        "context",
        "figures",
    )
    for name in artifacts:
        path = run_root / name
        if path.exists():
            _print(f"  {name}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run Phase 5 directly; CLI arguments are optional overrides."""

    _enable_immediate_output()
    args = build_parser().parse_args(argv)
    phase4_config = build_phase4_config(args)
    phase5_config = build_phase5_config(args, phase4_config)
    questions, benchmark, source_label = select_inputs(args, phase4_config)

    result, run_root, interrupted = execute(
        args,
        phase4_config=phase4_config,
        phase5_config=phase5_config,
        questions=questions,
        benchmark=benchmark,
        source_label=source_label,
    )
    if interrupted:
        return 130
    if result is None:
        return 1
    _print(f"Exported run: {run_root}")
    print_artifact_paths(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
