from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cial_knowledge_os.live.command_center import CommandCenterState
from cial_knowledge_os.live.event_bus import EventBus
from cial_knowledge_os.live.schemas import LiveEvent
from cial_knowledge_os.orchestration.phase5_pipeline import Phase5Pipeline


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_phase5_batch.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("run_phase5_batch", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
phase5_cli = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(phase5_cli)


class _Phase4AnswerPipeline:
    def __init__(self) -> None:
        self.config = SimpleNamespace(project_root=Path.cwd())

    def answer(self, question: str) -> dict[str, str]:
        return {
            "question": question,
            "answer": "Phase 4 answer",
            "answer_status": "answered",
        }


class Phase5BatchScriptTests(unittest.TestCase):
    def test_default_input_and_limit_match_phase5_batch_contract(self) -> None:
        args = phase5_cli.build_parser().parse_args(["--limit", "5"])
        config = phase5_cli.build_phase4_config(args)
        questions, benchmark, source = phase5_cli.select_inputs(args, config)

        self.assertEqual(len(questions), 5)
        self.assertIsNone(benchmark)
        self.assertEqual(
            Path(source).name,
            phase5_cli.QUESTIONS_FILE.name,
        )

    def test_dashboard_cli_defaults_and_resume_alias(self) -> None:
        defaults = phase5_cli.build_parser().parse_args([])
        self.assertTrue(defaults.dashboard)
        self.assertTrue(defaults.browser)
        self.assertEqual(defaults.dashboard_port, 8765)

        overridden = phase5_cli.build_parser().parse_args(
            [
                "--no-dashboard",
                "--no-browser",
                "--dashboard-port",
                "9876",
                "--resume-run-folder",
                "run-folder",
            ]
        )
        self.assertFalse(overridden.dashboard)
        self.assertFalse(overridden.browser)
        self.assertEqual(overridden.dashboard_port, 9876)
        self.assertEqual(overridden.resume_run_folder, Path("run-folder"))

    def test_phase5_is_enabled_by_default_and_can_be_disabled(self) -> None:
        enabled_args = phase5_cli.build_parser().parse_args([])
        config = phase5_cli.build_phase4_config(enabled_args)
        enabled = phase5_cli.build_phase5_config(enabled_args, config)
        self.assertTrue(enabled["phase5"]["enabled"])
        self.assertIn("local models=", phase5_cli.model_routing_summary(enabled))

        disabled_args = phase5_cli.build_parser().parse_args(["--no-phase5"])
        disabled = phase5_cli.build_phase5_config(disabled_args, config)
        self.assertFalse(disabled["phase5"]["enabled"])

    def test_disabled_phase5_returns_phase4_compatible_response(self) -> None:
        phase4 = _Phase4AnswerPipeline()
        pipeline = Phase5Pipeline(
            phase4_pipeline=phase4,
            config={"phase5": {"enabled": False}},
        )

        self.assertEqual(
            pipeline.answer("Question?"),
            phase4.answer("Question?"),
        )

    def test_phase5_artifacts_preserve_full_agent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = {
                "question": "Question?",
                "answer": "Answer [1].",
                "answer_status": "answered",
                "phase5_enabled": True,
                "query_intent": {"intent": "procedure"},
                "response_plan": {"format": "checklist"},
                "phase5_trace": {"events": []},
            }
            (root / "partial_retrieval.jsonl").write_text(
                json.dumps(
                    {
                        "key": "1:key",
                        "index": 1,
                        "response": response,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "retrieval.json").write_text(
                '[{"question": "Question?"}]\n',
                encoding="utf-8",
            )
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            (root / "metrics.json").write_text("{}\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")

            phase5_cli.finalize_phase5_artifacts(
                root,
                phase5_config={
                    "phase5": {
                        "enabled": True,
                        "model_profiles": {},
                        "agents": {},
                    }
                },
            )

            exported = json.loads(
                (root / "results.json").read_text(encoding="utf-8")
            )
            trace = json.loads(
                (root / "retrieval.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exported[0]["query_intent"]["intent"], "procedure")
            self.assertEqual(trace[0]["response_plan"]["format"], "checklist")
            self.assertTrue((root / "report.html").is_file())
            self.assertTrue(
                json.loads(
                    (root / "summary.json").read_text(encoding="utf-8")
                )["phase5_enabled"]
            )

    def test_live_adapter_adds_batch_routing_and_qdrant_context(self) -> None:
        live_bus = EventBus()
        state = CommandCenterState()
        live_bus.subscribe(state.apply)
        progress = SimpleNamespace(
            snapshot=lambda: {
                "current_question_index": 2,
                "total": 5,
                "current_question": "Question two?",
                "completed": 1,
                "percent": 20.0,
                "elapsed": 12.0,
                "eta": 48.0,
                "status_counts": {"answered": 1},
            }
        )
        manager = SimpleNamespace(run_id="batch-run", progress=progress)
        adapter = phase5_cli.LiveExecutionAdapter(
            live_bus=live_bus,
            execution_manager=manager,
            routing_summary="local models=test; agent assignments=7",
            phase5_enabled=True,
        )
        live_bus.subscribe(adapter.handle_live)

        adapter.handle_execution(
            SimpleNamespace(
                event_type="qdrant_health_checked",
                payload={"collection_status": "green", "point_count": 42},
                run_id="batch-run",
                question_index=None,
                question_total=None,
                question_preview="",
            )
        )
        adapter.handle_execution(
            SimpleNamespace(
                event_type="question_started",
                payload={},
                run_id="batch-run",
                question_index=2,
                question_total=5,
                question_preview="Question two?",
            )
        )
        live_bus.publish(
            LiveEvent(
                event_type="run_started",
                run_id="question-run",
                data={"question": "Question two?"},
            )
        )

        telemetry = state.snapshot()["telemetry"]
        self.assertEqual(telemetry["batch"]["question_index"], 2)
        self.assertEqual(telemetry["batch"]["answer_status_counts"], {"answered": 1})
        self.assertEqual(telemetry["qdrant"]["collection_status"], "green")
        self.assertIn("local models=test", telemetry["model_routing_summary"])

    def test_dashboard_failure_does_not_abort_batch_startup(self) -> None:
        args = phase5_cli.build_parser().parse_args(["--no-browser"])
        with patch(
            "cial_knowledge_os.live.command_center.start_in_thread",
            side_effect=RuntimeError("unavailable"),
        ):
            server, thread, url = phase5_cli.start_dashboard(
                args,
                live_bus=EventBus(),
            )
        self.assertIsNone(server)
        self.assertIsNone(thread)
        self.assertIsNone(url)

    def test_interrupt_marker_preserves_checkpoint_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = {
                "status": "in_progress",
                "completed_questions": [{"index": 1}],
            }
            (root / "checkpoint.json").write_text(
                json.dumps(checkpoint),
                encoding="utf-8",
            )

            phase5_cli.mark_run_interrupted(root)

            updated = json.loads(
                (root / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["status"], "interrupted")
            self.assertEqual(updated["completed_questions"], [{"index": 1}])


if __name__ == "__main__":
    unittest.main()
