from __future__ import annotations

import csv
import importlib.util
import io
import json
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from cial_knowledge_os.config import Phase4Config
from cial_knowledge_os.phase4_runner import Phase4Runner


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_phase4_batch.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("run_phase4_batch", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
phase4_cli = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(phase4_cli)


class _FastPhase4Pipeline:
    """Return deterministic Phase 4-shaped answers without model execution."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.metrics: dict[str, float] = {}
        self.is_ready_for_answering = True
        self.calls = 0

    def answer(self, question: str) -> dict[str, Any]:
        self.calls += 1
        token_usage = {
            "budget": self.config.max_context_tokens,
            "used": 0,
            "remaining": self.config.max_context_tokens,
            "context_tokens": 0,
            "context_tokens_used": 0,
            "encoding_name": self.config.tokenizer_encoding_name,
            "truncated_sections": 0,
            "omitted_sections": 0,
            "budget_type": "tokens",
            "candidate_tokens": 0,
            "selected_evidence_tokens": 0,
            "final_context_tokens": 0,
            "token_reduction_percent": 0.0,
            "candidate_chunk_count": 0,
            "selected_chunk_count": 0,
            "discarded_chunk_count": 0,
            "discard_reason_distribution": {},
            "usable_candidate_count": 0,
            "threshold_pass_count": 0,
            "fallback_used": False,
            "weak_evidence": False,
            "evidence_confidence": "none",
        }
        trace = {
            "schema_version": "phase4-trace-v1",
            "question": question,
            "candidate_pool": [],
            "reranked_candidates": [],
            "selected_chunks": [],
            "discarded_chunks": [],
            "final_context_chunks": [],
            "evidence_quality": {
                "chunks": [],
                "summary": {
                    "average_reranker_score": 0.0,
                    "unique_source_count": 0,
                    "strength_distribution": {
                        "strong": 0,
                        "medium": 0,
                        "weak": 0,
                    },
                },
            },
            "token_usage": token_usage,
            "latency": {
                "retrieval_seconds": 0.0,
                "reranking_seconds": 0.0,
                "evidence_selection_seconds": 0.0,
                "context_construction_seconds": 0.0,
                "generation_seconds": 0.0,
                "total_pipeline_seconds": 0.0,
                "artifact_export_seconds": None,
            },
            "citations": [],
            "answer": "Grounded test answer.",
            "answer_status": "answered",
            "decision_summary": [],
            "phase3_trace": {},
            "artifacts": {},
        }
        return {
            "question": question,
            "answer": "Grounded test answer.",
            "raw_answer": "Grounded test answer.",
            "answer_status": "answered",
            "retrieved": [],
            "query_variants": [],
            "retrieved_by_query": {},
            "context_stages": {
                "retrieved": [],
                "deduplicated": [],
                "expanded": [],
                "merged": [],
                "compressed": [],
            },
            "stage_counts": {},
            "context": "",
            "prompt": "",
            "citations": [],
            "token_usage": token_usage,
            "retrieval_mode": "hybrid",
            "question_trace": trace,
        }


class _InterruptingPhase4Pipeline(_FastPhase4Pipeline):
    def __init__(self, config: Any, *, interrupt_at: int) -> None:
        super().__init__(config)
        self.interrupt_at = interrupt_at
        self.attempts = 0

    def answer(self, question: str) -> dict[str, Any]:
        self.attempts += 1
        if self.attempts == self.interrupt_at:
            raise KeyboardInterrupt("simulated process interruption")
        return super().answer(question)


class Phase4TerminalQuestionCountTests(unittest.TestCase):
    def test_cli_config_is_unbounded_without_large_run_flag(self) -> None:
        args = phase4_cli.build_parser().parse_args([])
        config = phase4_cli.build_config(args)

        self.assertFalse(args.large_run)
        self.assertTrue(config.allow_large_run)

    def test_reliability_cli_flags_update_config(self) -> None:
        args = phase4_cli.build_parser().parse_args(
            [
                "--generation-retries",
                "1",
                "--retry-cooldown-seconds",
                "0",
                "--max-answer-words",
                "450",
            ]
        )
        config = phase4_cli.build_config(args)

        self.assertEqual(config.generation_retries, 1)
        self.assertEqual(config.retry_cooldown_seconds, 0.0)
        self.assertEqual(config.max_answer_words, 450)

    def test_max_questions_is_the_only_manual_cli_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions_path = Path(directory) / "questions.txt"
            questions_path.write_text(
                "\n".join(f"Question {index}?" for index in range(440)) + "\n",
                encoding="utf-8",
            )
            args = phase4_cli.build_parser().parse_args(
                [
                    "--questions-file",
                    str(questions_path),
                    "--max-questions",
                    "25",
                ]
            )
            config = phase4_cli.build_config(args)
            questions, benchmark, _ = phase4_cli.select_inputs(args, config)

        self.assertEqual(len(questions), 25)
        self.assertIsNone(benchmark)

    def test_notebook_and_smoke_limits_remain_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Phase4Config(project_root=Path(directory))
            runner = Phase4Runner(
                pipeline=_FastPhase4Pipeline(config),
                config=config,
            )
            questions = [f"Question {index}?" for index in range(440)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                notebook_manual = runner._apply_mode_limits(
                    questions,
                    run_mode="manual_qa",
                )
            smoke = runner._apply_mode_limits(questions, run_mode="smoke")
            benchmark = runner._apply_mode_limits(
                questions,
                run_mode="benchmark",
            )

        self.assertEqual(len(notebook_manual), 25)
        self.assertEqual(len(smoke), 3)
        self.assertEqual(len(benchmark), 440)

    def test_manual_cli_counts_reach_every_export(self) -> None:
        for question_count in (5, 25, 440):
            with self.subTest(question_count=question_count):
                with tempfile.TemporaryDirectory() as directory:
                    args = phase4_cli.build_parser().parse_args([])
                    config = phase4_cli.build_config(args)
                    config.output_root = (
                        Path(directory) / "outputs" / "batch_answers"
                    ).resolve()
                    config.phase4_trace_mode = "compact"
                    pipeline = _FastPhase4Pipeline(config)
                    questions = [
                        f"Question {index}?" for index in range(question_count)
                    ]

                    result = Phase4Runner(
                        pipeline=pipeline,
                        config=config,
                    ).run(
                        questions=questions,
                        run_mode="manual_qa",
                    )

                    with result.paths.results_csv.open(
                        encoding="utf-8-sig",
                        newline="",
                    ) as handle:
                        csv_count = sum(1 for _ in csv.DictReader(handle))
                    summary = json.loads(
                        result.paths.summary_json.read_text(encoding="utf-8")
                    )
                    metrics = json.loads(
                        result.paths.metrics_json.read_text(encoding="utf-8")
                    )

                self.assertEqual(csv_count, question_count)
                self.assertEqual(summary["question_count"], question_count)
                self.assertEqual(metrics["question_count"], question_count)

    def test_checkpoint_and_resume_skip_completed_duplicate_occurrences(
        self,
    ) -> None:
        questions = [
            "Duplicate question?",
            "Duplicate question?",
            "Third question?",
            "Fourth question?",
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = phase4_cli.build_parser().parse_args([])
            config = phase4_cli.build_config(args)
            config.output_root = (
                Path(directory) / "outputs" / "batch_answers"
            ).resolve()
            interrupted_pipeline = _InterruptingPhase4Pipeline(
                config,
                interrupt_at=2,
            )
            interrupted_runner = Phase4Runner(
                pipeline=interrupted_pipeline,
                config=config,
            )
            with self.assertRaises(KeyboardInterrupt):
                interrupted_runner.run(
                    questions=questions,
                    run_mode="manual_qa",
                )
            run_path = interrupted_runner.run_manager.require_paths().root
            checkpoint_path = run_path / "checkpoint.json"
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )

            self.assertEqual(len(checkpoint["completed_questions"]), 1)
            self.assertEqual(len(checkpoint["failed_questions"]), 1)
            self.assertEqual(
                checkpoint["question_manifest"][0]["question_hash"],
                checkpoint["question_manifest"][1]["question_hash"],
            )
            self.assertNotEqual(
                checkpoint["question_manifest"][0]["key"],
                checkpoint["question_manifest"][1]["key"],
            )
            with (run_path / "partial_results.jsonl").open(
                encoding="utf-8"
            ) as handle:
                self.assertEqual(sum(1 for _ in handle), 2)
            with (run_path / "partial_retrieval.jsonl").open(
                encoding="utf-8"
            ) as handle:
                self.assertEqual(sum(1 for _ in handle), 2)
            with (run_path / "partial_results.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 2)

            resumed_pipeline = _FastPhase4Pipeline(config)
            resumed_result = Phase4Runner(
                pipeline=resumed_pipeline,
                config=config,
            ).run(
                questions=questions,
                run_mode="manual_qa",
                resume_run=run_path,
            )
            final_checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            with resumed_result.paths.results_csv.open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        self.assertEqual(resumed_pipeline.calls, 3)
        self.assertEqual(
            [row["question"] for row in rows[:2]],
            ["Duplicate question?", "Duplicate question?"],
        )
        self.assertEqual(len(final_checkpoint["completed_questions"]), 4)
        self.assertEqual(final_checkpoint["failed_questions"], [])
        self.assertEqual(final_checkpoint["status"], "completed")


class Phase4StartupExperienceTests(unittest.TestCase):
    def _question_file(self, directory: str, count: int = 3) -> Path:
        path = Path(directory) / "user_supplied_input.txt"
        path.write_text(
            "\n".join(f"Question {index}?" for index in range(count)) + "\n",
            encoding="utf-8",
        )
        return path

    def test_verbose_dry_run_reports_resolved_settings_without_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions_path = self._question_file(directory)
            output = io.StringIO()
            with mock.patch.object(phase4_cli, "execute") as execute:
                with redirect_stdout(output):
                    status = phase4_cli.main(
                        [
                            "--questions-file",
                            str(questions_path),
                            "--output-dir",
                            str(Path(directory) / "results"),
                            "--verbose",
                            "--dry-run",
                        ]
                    )

        self.assertEqual(status, 0)
        execute.assert_not_called()
        rendered = output.getvalue()
        self.assertIn("Resolved configuration:", rendered)
        self.assertIn("Execution settings:", rendered)
        self.assertIn("Question preview:", rendered)
        self.assertIn('"checkpoint"', rendered)
        self.assertIn('"reranker"', rendered)
        self.assertIn('"llm"', rendered)

    def test_dynamic_question_source_reporting_uses_actual_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions_path = self._question_file(directory, count=4)
            args = phase4_cli.build_parser().parse_args(
                ["--questions-file", str(questions_path)]
            )
            config = phase4_cli.build_config(args)
            questions, _, source = phase4_cli.select_inputs(args, config)
            output = io.StringIO()
            with redirect_stdout(output):
                phase4_cli.report_question_source(questions, source)

        self.assertEqual(
            output.getvalue(),
            f"Loaded 4 questions from:\n{questions_path.resolve()}\n",
        )

    def test_manual_mode_has_no_implicit_question_filename(self) -> None:
        args = phase4_cli.build_parser().parse_args([])
        config = phase4_cli.build_config(args)

        with self.assertRaisesRegex(ValueError, "requires --questions-file"):
            phase4_cli.select_inputs(args, config)
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("DEFAULT_QUESTIONS_FILE", script)

    def test_startup_and_execution_steps_are_timestamped(self) -> None:
        class FakeReranker:
            load_source = "cache"

            def load(self) -> object:
                return object()

        class FakePipeline:
            def __init__(self, config: Any) -> None:
                self.config = config
                self.reranker = FakeReranker()
                self.llm = None

            def load(self) -> list[object]:
                return [object()]

            def chunk(self) -> list[object]:
                return [object()]

            def embed(self) -> list[object]:
                return [object()]

            def index(self) -> object:
                return object()

            def close(self) -> None:
                pass

        class FakeRunner:
            def __init__(self, **_: Any) -> None:
                pass

            def run(self, **_: Any) -> Any:
                return type("Result", (), {"summary": {"question_count": 1}})()

        args = phase4_cli.build_parser().parse_args(
            ["--questions-file", "input.txt"]
        )
        config = phase4_cli.build_config(args)
        output = io.StringIO()
        with (
            mock.patch(
                "cial_knowledge_os.phase4_pipeline.Phase4RAGPipeline",
                FakePipeline,
            ),
            mock.patch(
                "cial_knowledge_os.phase4_runner.Phase4Runner",
                FakeRunner,
            ),
            mock.patch(
                "cial_knowledge_os.llm.create_local_llm",
                return_value=object(),
            ),
            redirect_stdout(output),
        ):
            phase4_cli.execute(
                args,
                config=config,
                questions=["Question?"],
                source_label=str(Path("input.txt").resolve()),
                reporter=phase4_cli.StartupReporter(),
            )

        rendered = output.getvalue()
        for message in (
            "Initializing pipeline",
            "Loading documents",
            "Building/loading indexes",
            "Loading reranker",
            "Checking LLM availability",
            "Starting execution",
        ):
            self.assertRegex(rendered, rf"\[\s*\d+\.\d+s\] {message}")
        self.assertIn(f"Configured LLM:\n{config.ollama_model_name}", rendered)
        self.assertIn(
            f"Configured reranker:\n{config.reranker_model_name}",
            rendered,
        )

    def test_reporter_and_cli_prints_request_flush(self) -> None:
        reporter = phase4_cli.StartupReporter(verbose=True)
        with mock.patch("builtins.print") as print_mock:
            reporter.step("step")
            reporter.detail("detail")
            phase4_cli._print("plain")

        self.assertEqual(print_mock.call_count, 3)
        for call in print_mock.call_args_list:
            self.assertTrue(call.kwargs["flush"])

    def test_health_check_reports_pass_and_does_not_run_qa(self) -> None:
        class AvailableReranker:
            load_source = "local path"

            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def load(self) -> object:
                return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "data" / "sample").mkdir(parents=True)
            (root / "data" / "sample" / "document.txt").write_text(
                "content",
                encoding="utf-8",
            )
            (root / "data" / "qdrant").mkdir(parents=True)
            questions_path = self._question_file(directory)
            args = phase4_cli.build_parser().parse_args(
                [
                    "--project-root",
                    str(root),
                    "--questions-file",
                    str(questions_path),
                    "--health-check",
                ]
            )
            config = phase4_cli.build_config(args)
            questions, _, source = phase4_cli.select_inputs(args, config)
            output = io.StringIO()
            with (
                mock.patch(
                    "cial_knowledge_os.reranker.CrossEncoderReranker",
                    AvailableReranker,
                ),
                mock.patch(
                    "cial_knowledge_os.llm.create_local_llm",
                    return_value=object(),
                ),
                redirect_stdout(output),
            ):
                healthy = phase4_cli.run_health_check(
                    args,
                    config,
                    questions=questions,
                    source=source,
                    input_error=None,
                )

        self.assertTrue(healthy)
        rendered = output.getvalue()
        for check in (
            "Project structure",
            "Question source",
            "Document directories",
            "Vector database",
            "Reranker",
            "LLM",
            "Output directory",
        ):
            self.assertIn(f"PASS {check}", rendered)
        self.assertIn("QA execution was not started", rendered)


if __name__ == "__main__":
    unittest.main()
