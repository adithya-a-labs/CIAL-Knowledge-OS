from __future__ import annotations

import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cial_knowledge_os.batch_qa import collect_batch_answers
from cial_knowledge_os.execution import (
    EventBus,
    ExecutionEvent,
    ExecutionManager,
    ExecutionOptions,
    MetricsCollector,
    ProgressTracker,
    TelemetryCollector,
)
from cial_knowledge_os.execution.json_trace import (
    JSONTraceWriter,
    ProgressSnapshotWriter,
)
from cial_knowledge_os.execution.renderers import PlainConsoleRenderer


class ExecutionEventTests(unittest.TestCase):
    def test_event_creation_is_typed_and_serializable(self) -> None:
        event = ExecutionEvent(
            event_type="question_started",
            run_id="run-1",
            phase="Phase 4",
            question_index=1,
            question_total=2,
        )
        self.assertTrue(event.event_id)
        self.assertEqual(event.to_dict()["phase"], "Phase 4")


class EventBusTests(unittest.TestCase):
    def test_subscribe_emit_and_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[str] = []

        def handler(event: ExecutionEvent) -> None:
            received.append(event.event_type)

        bus.subscribe(handler)
        bus.emit(ExecutionEvent(event_type="run_started", run_id="run-1"))
        bus.unsubscribe(handler)
        bus.emit(ExecutionEvent(event_type="run_completed", run_id="run-1"))
        self.assertEqual(received, ["run_started"])

    def test_handler_failure_is_isolated_and_captured(self) -> None:
        bus = EventBus()
        received: list[str] = []
        bus.subscribe(lambda _: (_ for _ in ()).throw(RuntimeError("observer")))
        bus.subscribe(lambda event: received.append(event.event_type))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bus.emit(ExecutionEvent(event_type="warning", run_id="run-1"))
        self.assertEqual(received, ["warning"])
        self.assertIn("observer", bus.handler_warnings[0])


class ProgressAndMetricsTests(unittest.TestCase):
    def test_progress_counters_and_eta(self) -> None:
        tracker = ProgressTracker()
        tracker(
            ExecutionEvent(
                event_type="run_started",
                run_id="run-1",
                question_total=4,
            )
        )
        tracker(
            ExecutionEvent(
                event_type="question_completed",
                run_id="run-1",
                elapsed_seconds=2,
                payload={"answer_status": "answered"},
            )
        )
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["completed"], 1)
        self.assertEqual(snapshot["status_counts"]["answered"], 1)
        self.assertEqual(snapshot["eta"], 6)

    def test_metrics_aggregate_latency(self) -> None:
        collector = MetricsCollector()
        for latency in (1.0, 3.0):
            collector(
                ExecutionEvent(
                    event_type="retrieval_completed",
                    run_id="run-1",
                    elapsed_seconds=latency,
                )
            )
        summary = collector.summary()["timings"]["retrieval"]
        self.assertEqual(summary["average"], 2.0)
        self.assertEqual(summary["median"], 2.0)


class WriterAndConsoleTests(unittest.TestCase):
    def test_jsonl_and_progress_snapshot_writers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = JSONTraceWriter(root / "execution_trace.jsonl")
            trace(ExecutionEvent(event_type="run_started", run_id="run-1"))
            value = json.loads(
                (root / "execution_trace.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(value["event_type"], "run_started")

            snapshot = ProgressSnapshotWriter(root / "progress.json")
            snapshot.write({"completed": 3})
            self.assertEqual(
                json.loads(
                    (root / "progress.json").read_text(encoding="utf-8")
                )["completed"],
                3,
            )

    def test_plain_console_fallback_writes_progress(self) -> None:
        tracker = ProgressTracker()
        tracker(
            ExecutionEvent(
                event_type="run_started",
                run_id="run-1",
                question_total=2,
            )
        )
        stream = io.StringIO()
        renderer = PlainConsoleRenderer(
            tracker, stream=stream, refresh_seconds=0
        )
        renderer(
            ExecutionEvent(event_type="run_started", run_id="run-1")
        )
        self.assertIn("[EOF] 0/2", stream.getvalue())


class TelemetryTests(unittest.TestCase):
    def test_gracefully_handles_missing_psutil_and_nvidia_smi(self) -> None:
        with mock.patch(
            "cial_knowledge_os.execution.telemetry.importlib.import_module",
            side_effect=ImportError,
        ):
            collector = TelemetryCollector(
                command_runner=mock.Mock(side_effect=FileNotFoundError)
            )
            value = collector.collect()
        self.assertFalse(value["psutil_available"])
        self.assertFalse(value["gpu_available"])

    def test_collects_mocked_psutil_and_gpu(self) -> None:
        psutil = SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(
                used=2, total=8, percent=25
            ),
            disk_usage=lambda _: SimpleNamespace(
                used=10, free=30, percent=25
            ),
            cpu_percent=lambda interval=None: 50,
            Process=lambda: SimpleNamespace(
                memory_info=lambda: SimpleNamespace(rss=123)
            ),
        )
        result = SimpleNamespace(stdout="71, 6800, 12000, 64\n")
        collector = TelemetryCollector(
            psutil_module=psutil,
            command_runner=mock.Mock(return_value=result),
        )
        value = collector.collect()
        self.assertEqual(value["cpu_percent"], 50)
        self.assertEqual(value["gpu"]["vram_used_mb"], 6800)


class ExecutionManagerTests(unittest.TestCase):
    def test_noop_manager_does_not_write_or_crash(self) -> None:
        manager = ExecutionManager.disabled()
        manager.start_run(total_questions=1)
        manager.start_question(1, 1, "Question?")
        manager.complete_question(answer_status="answered")
        manager.complete_run()
        self.assertFalse(manager.enabled)

    def test_manager_writes_trace_snapshot_and_progress_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = ExecutionOptions(
                console=False,
                telemetry=False,
                output_dir=Path(directory),
            )
            manager = ExecutionManager(
                run_id="run-1",
                phase="Phase 4",
                run_mode="manual_qa",
                options=options,
            )
            manager.start_run(total_questions=1)
            manager.start_question(1, 1, "What is EOF?")
            manager.complete_question(answer_status="answered")
            manager.complete_run()
            run_dir = Path(directory) / "run-1"
            self.assertTrue((run_dir / "execution_trace.jsonl").is_file())
            self.assertTrue((run_dir / "progress.json").is_file())
            self.assertTrue((run_dir / "progress.log").is_file())
            progress = json.loads(
                (run_dir / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["completed"], 1)

    def test_batch_runner_emits_question_events_without_changing_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                project_root=Path(directory),
                top_k=3,
                ollama_model_name="local-model",
                embedding_model_name="local-embedding",
                tokenizer_encoding_name="cl100k_base",
            )

            class Pipeline:
                is_ready_for_answering = True
                metrics = {
                    "generation_latency": 0.1,
                    "retrieval_latency": 0.2,
                }

                def __init__(self) -> None:
                    self.config = config

                def answer(self, question: str) -> dict[str, object]:
                    return {
                        "question": question,
                        "answer": "Observed answer",
                        "retrieved": [],
                    }

            manager = ExecutionManager(
                run_id="batch-run",
                options=ExecutionOptions(
                    console=False,
                    telemetry=False,
                    output_dir=Path(directory) / "runs",
                ),
            )
            events: list[str] = []
            manager.event_bus.subscribe(
                lambda event: events.append(event.event_type)
            )
            collection = collect_batch_answers(
                pipeline=Pipeline(),
                questions=["What happened?"],
                execution_manager=manager,
            )
            self.assertEqual(collection.rows[0]["answer"], "Observed answer")
            self.assertIn("question_started", events)
            self.assertIn("question_completed", events)

    def test_disabled_observability_preserves_batch_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                project_root=Path(directory),
                top_k=1,
                ollama_model_name="",
                embedding_model_name="",
                tokenizer_encoding_name="cl100k_base",
            )
            calls: list[str] = []

            class Pipeline:
                is_ready_for_answering = True
                metrics: dict[str, float] = {}

                def __init__(self) -> None:
                    self.config = config

                def answer(self, question: str) -> dict[str, object]:
                    calls.append(question)
                    return {"answer": "same", "retrieved": []}

            result = collect_batch_answers(
                pipeline=Pipeline(),
                questions=["Question?"],
                execution_manager=ExecutionManager.disabled(),
            )
            self.assertEqual(calls, ["Question?"])
            self.assertEqual(result.rows[0]["answer"], "same")


if __name__ == "__main__":
    unittest.main()
