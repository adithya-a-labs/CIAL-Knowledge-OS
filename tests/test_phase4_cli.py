from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

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

    def answer(self, question: str) -> dict[str, Any]:
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


class Phase4TerminalQuestionCountTests(unittest.TestCase):
    def test_cli_config_is_unbounded_without_large_run_flag(self) -> None:
        args = phase4_cli.build_parser().parse_args([])
        config = phase4_cli.build_config(args)

        self.assertFalse(args.large_run)
        self.assertTrue(config.allow_large_run)

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


if __name__ == "__main__":
    unittest.main()
