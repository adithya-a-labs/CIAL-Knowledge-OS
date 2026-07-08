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
    def test_indexing_summary_is_added_to_phase4_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Phase4Config(
                project_root=Path(directory),
                phase4_trace_mode="compact",
            )
            pipeline = _FastPhase4Pipeline(config)
            pipeline.indexing_summary = {
                "new_files": 1,
                "changed_files": 0,
                "unchanged_files": 2,
                "deleted_files": 0,
            }
            result = Phase4Runner(
                pipeline=pipeline,
                config=config,
            ).run(questions=["Question?"], run_mode="smoke")

            self.assertEqual(
                result.summary["indexing_summary"],
                pipeline.indexing_summary,
            )
            self.assertEqual(
                result.metrics["indexing_summary"],
                pipeline.indexing_summary,
            )
            report = result.paths.report_html.read_text(encoding="utf-8")
            self.assertIn("Index new files", report)
            self.assertIn("Index unchanged files", report)

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

    def test_force_rebuild_index_cli_and_user_default(self) -> None:
        default_args = phase4_cli.build_parser().parse_args([])
        self.assertEqual(
            phase4_cli.build_config(default_args).force_rebuild_index,
            phase4_cli.FORCE_REBUILD_INDEX,
        )
        forced_args = phase4_cli.build_parser().parse_args(
            ["--force-rebuild-index"]
        )
        self.assertTrue(phase4_cli.build_config(forced_args).force_rebuild_index)

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

    def test_manual_default_is_unbounded_and_explicit_limits_remain_scoped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Phase4Config(project_root=Path(directory))
            runner = Phase4Runner(
                pipeline=_FastPhase4Pipeline(config),
                config=config,
            )
            questions = [f"Question {index}?" for index in range(440)]
            manual = runner._apply_mode_limits(
                questions,
                run_mode="manual_qa",
            )
            smoke = runner._apply_mode_limits(questions, run_mode="smoke")
            benchmark = runner._apply_mode_limits(
                questions,
                run_mode="benchmark",
            )
            capped_config = Phase4Config(
                project_root=Path(directory),
                max_inline_manual_questions=25,
            )
            capped_runner = Phase4Runner(
                pipeline=_FastPhase4Pipeline(capped_config),
                config=capped_config,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                capped_manual = capped_runner._apply_mode_limits(
                    questions,
                    run_mode="manual_qa",
                )

        self.assertEqual(len(manual), 440)
        self.assertEqual(len(smoke), 3)
        self.assertEqual(len(benchmark), 440)
        self.assertEqual(len(capped_manual), 25)

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
            f"Loaded 4 questions from {questions_path.resolve()}\n",
        )

    def test_no_argument_run_uses_user_configuration_question_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions_path = self._question_file(directory, count=2)
            with mock.patch.object(
                phase4_cli,
                "QUESTIONS_FILE",
                questions_path,
            ):
                args = phase4_cli.build_parser().parse_args([])
                config = phase4_cli.build_config(args)
                questions, benchmark, source = phase4_cli.select_inputs(
                    args,
                    config,
                )

        self.assertEqual(questions, ["Question 0?", "Question 1?"])
        self.assertIsNone(benchmark)
        self.assertEqual(source, str(questions_path.resolve()))
        self.assertEqual(config.phase4_run_mode, phase4_cli.RUN_MODE)
        self.assertEqual(
            config.generation_retries,
            phase4_cli.GENERATION_RETRIES,
        )
        self.assertEqual(
            config.retry_cooldown_seconds,
            phase4_cli.RETRY_COOLDOWN_SECONDS,
        )
        self.assertEqual(
            config.max_answer_words,
            phase4_cli.MAX_ANSWER_WORDS,
        )
        self.assertEqual(
            config.adaptive_answer_sections,
            phase4_cli.ADAPTIVE_ANSWER_SECTIONS,
        )
        self.assertEqual(config.reranker_device, phase4_cli.RERANKER_DEVICE)
        self.assertEqual(
            config.reranker_batch_size,
            phase4_cli.RERANKER_BATCH_SIZE,
        )
        self.assertEqual(
            config.reranker_local_files_only,
            phase4_cli.LOCAL_FILES_ONLY,
        )

    def test_cli_arguments_override_user_configuration_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions_path = self._question_file(directory)
            args = phase4_cli.build_parser().parse_args(
                [
                    "--questions-file",
                    str(questions_path),
                    "--mode",
                    "smoke",
                    "--generation-retries",
                    "5",
                    "--retry-cooldown-seconds",
                    "1.5",
                    "--max-answer-words",
                    "300",
                    "--reranker-device",
                    "cpu",
                    "--reranker-batch-size",
                    "8",
                    "--local-files-only",
                ]
            )
            config = phase4_cli.build_config(args)
            _, _, source = phase4_cli.select_inputs(args, config)

        self.assertEqual(source, str(questions_path.resolve()))
        self.assertEqual(config.phase4_run_mode, "smoke")
        self.assertEqual(config.generation_retries, 5)
        self.assertEqual(config.retry_cooldown_seconds, 1.5)
        self.assertEqual(config.max_answer_words, 300)
        self.assertEqual(config.reranker_device, "cpu")
        self.assertEqual(config.reranker_batch_size, 8)
        self.assertTrue(config.reranker_local_files_only)

    def test_custom_user_configuration_runs_the_simple_main_flow(self) -> None:
        class FakePipeline:
            def __init__(self, config: Any) -> None:
                self.config = config
                self.closed = False

            def load(self) -> list[object]:
                return [object()]

            def chunk(self) -> list[object]:
                return [object()]

            def embed(self) -> list[object]:
                return [object()]

            def index(self) -> object:
                return object()

            def close(self) -> None:
                self.closed = True

        class FakeRunner:
            last_run: dict[str, Any] = {}

            def __init__(self, **_: Any) -> None:
                pass

            def run(self, **kwargs: Any) -> Any:
                FakeRunner.last_run = kwargs
                root = Path("fake-run")
                return type(
                    "Result",
                    (),
                    {"paths": type("Paths", (), {"root": root})()},
                )()

        with tempfile.TemporaryDirectory() as directory:
            questions_path = self._question_file(directory, count=2)
            resume_path = Path(directory) / "existing-run"
            output = io.StringIO()
            with (
                mock.patch.object(phase4_cli, "QUESTIONS_FILE", questions_path),
                mock.patch.object(
                    phase4_cli,
                    "RESUME_RUN_FOLDER",
                    resume_path,
                ),
                mock.patch(
                    "cial_knowledge_os.phase4_pipeline.Phase4RAGPipeline",
                    FakePipeline,
                ),
                mock.patch(
                    "cial_knowledge_os.phase4_runner.Phase4Runner",
                    FakeRunner,
                ),
                mock.patch.object(phase4_cli, "print_artifact_paths"),
                redirect_stdout(output),
            ):
                status = phase4_cli.main([])

        self.assertEqual(status, 0)
        self.assertEqual(
            FakeRunner.last_run["questions"],
            ["Question 0?", "Question 1?"],
        )
        self.assertEqual(
            FakeRunner.last_run["resume_run"],
            resume_path.resolve(),
        )
        rendered = output.getvalue()
        for message in (
            "Starting Phase 4 batch run",
            f"Loaded 2 questions from {questions_path.resolve()}",
            "Initializing pipeline",
            "Indexing complete",
            "Starting QA",
            f"Exported run: {Path('fake-run')}",
        ):
            self.assertIn(message, rendered)

    def test_missing_or_empty_configured_file_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.txt"
            with self.assertRaisesRegex(
                FileNotFoundError,
                r"Expected path: .*missing\.txt",
            ) as missing_error:
                phase4_cli.load_questions(missing)
            self.assertIn(
                "one question per line",
                str(missing_error.exception),
            )
            self.assertIn(
                "'question' column",
                str(missing_error.exception),
            )

            empty = Path(directory) / "empty.txt"
            empty.write_text(
                "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Questions file is empty"):
                phase4_cli.load_questions(empty)

    def test_script_prints_always_flush(self) -> None:
        with mock.patch("builtins.print") as print_mock:
            phase4_cli._print("progress")

        print_mock.assert_called_once_with("progress", flush=True)


if __name__ == "__main__":
    unittest.main()
